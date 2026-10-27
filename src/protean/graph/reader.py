"""The reader seat: one fresh, read-only model call, made by the gate, never by the director, for one of two checks.

Branch admission: the reader is given the parents (accepted retained predicates, each with the last commit it passed
on), the children proposed to inherit them and the director's rationale, in a fresh clone of the workspace at the
accepted tip, and answers one question: do the children together preserve what the parents promised? Its answer is
exactly one JSON object, `{"verdict": "yes"|"no", "note": "<one line>"}`. Only `yes` admits; a call that cannot be
made, fails, times out, reaches its cap, leaves no receipt or answers anything else never does. The reader never passes a unit: the grader checks the children on the candidate that introduces them.

The weakening check (`gate.weakening`): the brief says so (`"check": "weakening"`), the clone sits at the candidate
instead, and the answer also names the kind of change, one of `KINDS`:
`{"verdict": "yes"|"no", "kind": "<kind>", "note": "<one line>"}`, `yes` meaning weakened.

Containment: `--tools Read Grep Glob` restricts the seat as it does the director, and the process runs under
`sandbox-exec` with the check command's file rules (every write denied but the temp dirs and `/dev`, the clone and
the workspace denied even under a temp dir, the task's `console.json` unreadable) and the network left open, since
the model call needs its transport. Fresh per call: no `--resume`, nothing carried into any other call. The call is
recorded before launch like every seat's (`Call`, seat `reader`) and its charge is the gate's to book.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import tempfile
from collections.abc import Collection
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from protean.graph import budget, grade, worker
from protean.graph.budget import Caps
from protean.runner import NO_MCP, Charge, Runner, RunResult, charged_cap, floor_reason, no_charge, usd_cap
from protean.state import Call, Root, Task, Unit, save

TOOLS = ("Read", "Grep", "Glob")
VERDICTS = ("yes", "no")
NOTE_LIMIT = 300
CLONE = "_reader"  # no unit id starts with "_", so this never meets a unit's clone
FENCED = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)
ANSWER = '{"verdict": "yes"|"no", "note": "<one line>"}'
# The weakening check's kinds of change; its answer names one.
KINDS = ("deleted tests", "weakened assertions", "skipped or expected-fail", "legitimate refactor", "strengthened",
         "unrelated")
KIND_ANSWER = '{"verdict": "yes"|"no", "kind": "<kind>", "note": "<one line>"}'


@dataclass
class Check:
    """One reader check. `verdict` is the reader's `yes` or `no`, None when no verdict came back; `reason` is the
    verdict, or why there was none (floor, sandbox, clone, timed_out, no_receipt, error, cap_hit, malformed); `note` is
    the reader's one line, or what went wrong. `kind` is the kind of change a weakening check's answer names, None
    otherwise. `call` is the record, None when no call was made."""

    verdict: str | None
    note: str
    reason: str
    call: Call | None = None
    kind: str | None = None

    @property
    def admitted(self) -> bool:
        return self.verdict == "yes"


def clone_dir(root: Root, task: Task) -> Path:
    """Where the reader's fresh clone lives while its call runs, beside (never inside) the units' clones."""
    return root.clones_dir(task.id).resolve() / CLONE


def sandbox_profile(clone: Path, workspace: Path, console_file: Path | None = None) -> str:
    """The reader's profile: allow default, network open for the model call, every write denied except the temp dirs
    and `/dev` (the CLI's own scratch), the clone and the workspace write-denied last so the deny holds even when they
    sit under a temp dir, and reading `console_file` denied."""
    scratch = [os.path.realpath(tempfile.gettempdir()), *grade.SANDBOX_WRITABLE]
    allowed = " ".join(f"(subpath {grade.quoted(path)})" for path in dict.fromkeys(scratch))
    denied = " ".join(f"(subpath {grade.quoted(os.path.realpath(path))})" for path in (clone, workspace))
    return (f"(version 1)(allow default)(deny file-write*)(allow file-write* {allowed})(deny file-write* {denied})"
            + grade.read_deny(console_file))


def argv(task: Task, caps: Caps, root: Root, clone: Path, sandbox: str, spent: float = 0.0) -> list[str]:
    """The reader call, wrapped in its sandbox: no user settings, json output, read-only tools restricted with
    `--tools` (as the director's), the reader cap clipped to what the task has left after `spent`, the prompt."""
    return [
        sandbox, "-p", sandbox_profile(clone, Path(task.workspace), root.console_file(task.id)),
        "claude", "-p", "--setting-sources", "", *NO_MCP, "--output-format", "json",
        "--max-budget-usd", usd_cap(caps.reader_call_usd, task, caps, spent, reader=True),
        "--model", caps.reader_model, "--permission-mode", "dontAsk", "--tools", *TOOLS, "--allowedTools", *TOOLS,
        "--system-prompt", root.reader_prompt.read_text(encoding="utf-8"),
    ]


def message(brief: dict[str, Any]) -> str:
    """The reader's user message: the gate's brief (a branch's parents, children, rationale and accepted tip, or the
    weakening check's intent, touched paths and hunks) as one JSON object."""
    return json.dumps(brief, indent=2) + "\n"


def parse(text: str) -> tuple[str | None, str]:
    """The verdict and its one-line note from a branch check's answer, or (None, what is wrong with it): `parse_answer`
    with no kind allowed."""
    verdict, _, note = parse_answer(text)
    return verdict, note


def parse_answer(text: str, kinds: Collection[str] = ()) -> tuple[str | None, str | None, str]:
    """(verdict, kind, note) from the reader's answer, or (None, None, what is wrong with it). The whole answer must be
    one JSON object with exactly `verdict` (`yes` or `no`) and a non-empty `note`, plus, when `kinds` is given (the
    weakening check), a `kind` among them; one surrounding code fence is tolerated, nothing else."""
    keys, shape = ({"verdict", "kind", "note"}, KIND_ANSWER) if kinds else ({"verdict", "note"}, ANSWER)
    body = text.strip()
    fenced = FENCED.fullmatch(body)
    if fenced:
        body = fenced.group(1)
    try:
        obj = json.loads(body)
    except (json.JSONDecodeError, RecursionError):
        return None, None, f"the answer is not one JSON object {shape}"
    if not isinstance(obj, dict) or set(obj) != keys:
        return None, None, f"the answer must be exactly {shape}"
    if obj["verdict"] not in VERDICTS:
        return None, None, f"the verdict must be yes or no, not {obj['verdict']!r}"[:NOTE_LIMIT]
    if kinds and obj["kind"] not in kinds:
        return None, None, f"the kind must be one of {', '.join(kinds)}, not {obj['kind']!r}"[:NOTE_LIMIT]
    if not isinstance(obj["note"], str) or not obj["note"].strip():
        return None, None, "the note must be a non-empty string"
    return obj["verdict"], obj.get("kind"), " ".join(obj["note"].split())[:NOTE_LIMIT]


def _answer(res: RunResult, timeout: int, cap: float,
            kinds: Collection[str] = ()) -> tuple[dict | None, str | None, str | None, str, str]:
    """(receipt, verdict, kind, reason, note) from one run. Only a usable receipt whose result parses gives a
    verdict."""
    if res.timed_out:
        return None, None, None, "timed_out", f"the reader call timed out after {timeout}s"
    try:
        receipt = json.loads(res.stdout)
    except (json.JSONDecodeError, RecursionError):
        receipt = None
    if not isinstance(receipt, dict):
        return None, None, None, "no_receipt", f"the reader call left no receipt (exit {res.returncode})"
    if budget.cap_hit(receipt):
        return receipt, None, None, "cap_hit", f"the reader call reached its ${cap:.2f} cap"
    if receipt.get("is_error"):
        return receipt, None, None, "error", f"the reader call ended in error: {receipt.get('subtype', '')}".strip()
    if not isinstance(receipt.get("result"), str):
        return receipt, None, None, "malformed", "the reader's receipt carries no result text"
    verdict, kind, note = parse_answer(receipt["result"], kinds)
    return receipt, verdict, kind, verdict or "malformed", note


def check(task: Task, unit: Unit, brief: dict[str, Any], caps: Caps, root: Root, runner: Runner, *,
          spent: float = 0.0, charge: Charge = no_charge, base: str | None = None,
          kinds: Collection[str] = (), attempt: int = 1) -> Check:
    """One fresh reader call on `brief` for `unit`, in a fresh clone at `base` (the accepted tip when None: a branch
    check; the candidate for the weakening check), removed afterwards; with `kinds` the answer must name one. `attempt`
    is the record's (the unit's attempt whose candidate a weakening check reads; 1 for a branch).
    No call is made under the call floor (after `spent`, what the caller has not booked yet), without the sandbox
    binary, or when the clone cannot be prepared. The record is saved with the task before launch and resolved from
    the receipt; a call that leaves none is charged its cap. A call interrupted before it returns is resolved
    `interrupted` and its cap goes through `charge` before the interruption propagates. Booking is the caller's."""
    floor = floor_reason(task, caps, spent, reader=True)
    if floor:
        return Check(None, f"no reader call: {floor}", "floor")
    try:
        sandbox = worker.sandbox_binary()
    except worker.WorkerError as exc:
        return Check(None, str(exc), "sandbox")
    clone = clone_dir(root, task)
    try:
        try:
            worker.fresh_clone(task, clone, base or worker.accepted_tip(task))
        except worker.WorkerError as exc:
            return Check(None, f"the reader's clone could not be prepared: {exc}", "clone")
        call, stdin = argv(task, caps, root, clone, sandbox, spent), message(brief)
        record = budget.open_call(task, "reader", attempt, charged_cap(call), stdin, unit.id)
        try:
            save(root, task)
            res = runner(call, stdin, clone, caps.reader_wall_seconds)
        except BaseException:
            budget.resolve(record, charged_cap(call), interrupted=True)
            charge(charged_cap(call))
            raise
    finally:
        shutil.rmtree(clone, ignore_errors=True)
    receipt, verdict, kind, reason, note = _answer(res, caps.reader_wall_seconds, charged_cap(call), kinds)
    budget.resolve(record, charged_cap(call) if receipt is None else budget.cost_of(receipt), receipt,
                  timed_out=res.timed_out)
    save(root, task)
    return Check(verdict, note, reason, record, kind)
