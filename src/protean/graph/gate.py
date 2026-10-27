"""The gate's branch admission.

A unit whose predicates carry `parents` proposes a branch: those accepted retained predicates (X) hand their promise
to its children. Before the unit enters the task, inside the same tick, the runtime sends X, the children and the
director's rationale straight to a fresh `reader` (never through the director) and acts on its verdict. `yes` admits
the branch (`branch_admitted`); anything else refuses it (`branch_refused`), and the whole plan goes back to the
director as one correctable line naming the unit and the reader's note. The director sees that line only: never the
reader's input, transcript or receipt. The operator sees both events on the stream and is not asked.

Admission retires nothing. The parents stay in the re-check set until the branching unit is integrated: every other
unit's candidate is re-checked against them as before (`broke <parent>` rejects it), and a failed or descoped branching
unit changes nothing. On the branching unit's own candidate, once every child passes, the parents of those children are
excused from its re-check (`excused`): the unit may rename or remove what they pinned, and no parent is a `broke` there;
any child failing rejects the candidate on its own grade, and the parents hold untouched. At integration (`accept`),
each parent gains its children, records the last accepted commit it passed on, and leaves the re-check set
(`grade.rechecked`) for good (`branch_accepted`).

The weakening check (the gate owns acceptance-test authorization; reader-first) runs pre-integration on a candidate that already passed every free check: its own grade, the
re-checks and the clean clone. Each retained `command` names the `protected` paths that control its result; the
protected set in force is the union over every integrated unit's command still in the re-check set, less the parents
excused on this candidate (the candidate's own commands never count: it writes those tests for the first time). The
gate diffs the candidate against the accepted tip it branched from; no protected path touched, nothing happens and
nothing is spent. Touched, exactly one fresh reader call reads the hunks at the candidate: `no` (not weakened) lets
the candidate integrate; `yes` (weakened) rejects it like a failed predicate, its kind, note and touched paths
reaching the next attempt's findings; a call that gives no verdict never integrates either (rejected, "weakening check
could not run"). Protected means a second look, never frozen.

A reader call that reaches its dollar cap decides nothing on its own, in either check (`ask`): the same brief is asked
once more under twice `reader_call_usd` (`budget.wider`, clipped by what the allowance has left), and that verdict is
applied as the first one's would have been. A second cap hit, or no room under the call floor for the wider call,
holds the attempt or the plan on a runtime `reader_cap` question (`questions.reader_cap`), opened in the save that books
the call: `yes` asks once more under the wider cap, `no` takes the cap hit's old path (the branch refused with reason
`cap_hit`, the attempt rejected unchecked; `declined`).

Deterministic: no model call here; the reader seat makes the one call.
"""

from __future__ import annotations

import os
import subprocess
from collections.abc import Collection, Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any

from protean import state
from protean.graph import budget, grade, questions, reader, worker
from protean.graph.budget import Caps
from protean.runner import Charge, Runner, no_charge
from protean.state import Call, Predicate, Question, Root, Task, Unit

QUESTION = ("Do the children together preserve what the parents promised, with no significant loss of what was "
            "verified?")
WEAKENING_QUESTION = "Does this change weaken the acceptance checks the protected paths control?"
HUNK_BOUND = 16_000  # characters of the protected hunks the reader is sent; the rest is cut at an explicit marker
UNTOUCHED, NOT_WEAKENED, WEAKENED, UNCHECKED = "untouched", "not_weakened", "weakened", "unchecked"
HELD = "held"  # a weakening check held on `reader_cap`: the attempt is not decided, and no verdict records it


@dataclass
class Admission:
    """What the gate did with one plan: `refused` holds the line the director is returned (empty when every branch
    was admitted, or there was none); `lines` says, one per reader call, what happened, for the tick's echo. `waits`:
    admission stopped at the call floor or on a reader call that left no receipt, which refuses nothing (the plan
    stays held)."""

    refused: list[str] = field(default_factory=list)
    lines: list[str] = field(default_factory=list)
    waits: bool = False


def children_of(unit: Unit) -> list[str]:
    """`"<unit> <pid>"` of each of the unit's predicates that carries parents, in order."""
    return [f"{unit.id} {pid}" for pid, p in zip(grade.predicate_ids(unit), unit.predicates) if p.parents]


def parents_of(unit: Unit) -> list[str]:
    """Every parent the unit's predicates name, each once, in the order first named."""
    return list(dict.fromkeys(parent for p in unit.predicates for parent in p.parents))


def excused(unit: Unit, passed_ids: Collection[str]) -> dict[str, list[str]]:
    """The parents excused from the re-check on `unit`'s own candidate, each with the children that carry it: every
    parent the unit names, once EVERY child is among `passed_ids` (its own grade). Empty when any child did not pass,
    or the unit proposes no branch."""
    children = [(pid, p) for pid, p in zip(grade.predicate_ids(unit), unit.predicates) if p.parents]
    if not children or any(pid not in passed_ids for pid, _ in children):
        return {}
    carried: dict[str, list[str]] = {}
    for pid, predicate in children:
        for parent in dict.fromkeys(predicate.parents):
            carried.setdefault(parent, []).append(f"{unit.id} {pid}")
    return carried


def branching(units: Iterable[Unit]) -> list[Unit]:
    """The units that propose a branch: any predicate carrying parents."""
    return [unit for unit in units if any(p.parents for p in unit.predicates)]


def find(task: Task, key: str) -> tuple[Unit, Predicate] | None:
    """The task's unit and predicate `"<unit> <pid>"` names, or None."""
    unit_id, _, pid = key.partition(" ")
    unit = next((u for u in task.units if u.id == unit_id), None)
    if unit is None:
        return None
    found = dict(zip(grade.predicate_ids(unit), unit.predicates)).get(pid)
    return None if found is None else (unit, found)


def last_passed(root: Root, task_id: str, key: str) -> str | None:
    """The last accepted commit `"<unit> <pid>"` passed on, from the verdict records in the log: its own unit's passing
    grade, or a passing candidate's re-check that lists it `passed`. None when no record shows one."""
    unit_id, _, pid = key.partition(" ")
    commit = None
    for event in state.read_log(root, task_id):
        if event.get("event") != "verdict" or not event.get("passed"):
            continue
        if event.get("unit") == unit_id and pid in (event.get("passed_ids") or []):
            commit = event.get("commit")
        recheck = event.get("recheck") or {}
        if {"unit": unit_id, "id": pid, "result": "passed"} in (recheck.get("predicates") or []):
            commit = recheck.get("candidate")
    return commit


def brief(task: Task, unit: Unit, root: Root) -> dict[str, Any]:
    """What the reader is given: the parents (kind, args, the last commit each passed on), the children (kind, args,
    which parents each inherits), the director's rationale, the accepted tip its clone holds, and the one question."""
    parents = []
    for key in parents_of(unit):
        found = find(task, key)
        if found is not None:
            parents.append({"id": key, "kind": found[1].kind, "args": found[1].args,
                            "last_passed_commit": last_passed(root, task.id, key)})
    children = [{"id": f"{unit.id} {pid}", "kind": p.kind, "args": p.args, "parents": list(p.parents)}
                for pid, p in zip(grade.predicate_ids(unit), unit.predicates) if p.parents]
    return {"unit": unit.id, "parents": parents, "children": children, "rationale": unit.rationale,
            "accepted_tip": task.accepted, "question": QUESTION}


def refusal(unit: Unit, check: reader.Check) -> str:
    """The one line the director is returned for a refused branch: the reader's note on a `no`, else why no verdict
    came back. Nothing else of the reader's exchange."""
    if check.verdict == "no":
        return f"unit {unit.id!r}: branch refused by the reader: {check.note}"
    return f"unit {unit.id!r}: branch refused, the reader gave no verdict ({check.reason}): {check.note}"


def admit(task: Task, units: list[Unit], caps: Caps, root: Root, runner: Runner, *, spent: float = 0.0,
          charge: Charge = no_charge, to_director: bool = False, answered: Question | None = None) -> Admission:
    """Admit or refuse each branch the plan proposes, one reader check per branching unit in plan order, stopping at
    the first refusal (the whole plan goes back). Each call's charge is booked on the task at once; `spent` is what the
    caller has spent and not yet booked (the director's own call in this plan), so the reader's cap is clipped after
    it and an interruption books it with the reader's cap through `charge`. With `to_director` (a held plan, released
    after its director call) the refusal event carries the line as a note for the director's next message, and a
    reader call the call floor stops is no refusal: admission stops there, nothing is logged, and `waits` says so (the
    plan stays held until an extension). A reader call that leaves no receipt is no refusal either: its cap is booked,
    paid work holds on a runtime question (`questions.unknown_spend`), and `waits` says so; a plan in `auto` is then
    held unadmitted, so its release asks the reader again once the operator says `yes`. Nor is a reader call that
    reached its cap twice (`ask`): the plan holds on `reader_cap` and `waits` says so. `answered` is that question,
    answered and not yet acted on: `yes` asks its unit's reader under the wider cap first, `no` refuses the branch
    (reason `cap_hit`) with no call."""
    admission = Admission()
    info = answered.context["reader_cap"] if answered is not None else None
    if answered is not None:
        answered.context["settled"] = True
        unit = next((u for u in branching(units) if u.id == info["unit"]), None)
        if answered.disposition != "reasked" and unit is not None:
            call = next((c for c in task.calls if c.id == info["calls"][-1]), None)
            _refuse(admission, task, unit, reader.Check(None, info["note"], "cap_hit", call), root, to_director)
            return admission
    for unit in branching(units):
        wide_after = info["calls"][-1] if info is not None and unit.id == info["unit"] else None
        asked = ask(task, unit, brief(task, unit, root), caps, root, runner, wide_after=wide_after, spent=spent,
                    charge=lambda usd: charge(spent + usd))
        check = asked.check
        if asked.held:  # no verdict and no refusal: the plan holds on the operator
            admission.waits = True
            said = _hold(task, root, asked, caps, {"check": "admission", "unit": unit.id},
                         info["calls"] if wide_after else [])
            admission.lines.append(f"{unit.id}'s branch is held: {said}")
            break
        if to_director and check.reason == "floor":
            admission.waits = True
            admission.lines.append(f"{unit.id}'s branch waits for an extension: {check.note}")
            break
        cost = ""
        if check.call is not None:
            cost = f" (${check.call.charge_usd or 0.0:.2f})"
            if check.call.actual_usd == budget.UNKNOWN:  # no receipt: no verdict and no refusal; paid work holds
                admission.waits = True
                admission.lines.append(f"{unit.id}'s branch waits: the reader call left no receipt ({check.reason}), "
                                       f"paid work held{cost}")
                questions.unknown_spend(root, task, [check.call])
                break
        if check.admitted:
            event = _branch_event(unit, check)
            state.append_log(root, task.id, {"event": "branch_admitted", **event})
            admission.lines.append(f"reader admitted {unit.id}'s branch "
                                   f"({', '.join(event['parents'])} -> {', '.join(event['children'])}){cost}")
            continue
        _refuse(admission, task, unit, check, root, to_director, cost)
        break
    return admission


def _branch_event(unit: Unit, check: reader.Check) -> dict[str, Any]:
    """The fields `branch_admitted` and `branch_refused` share."""
    return {"unit": unit.id, "parents": parents_of(unit), "children": children_of(unit), "rationale": unit.rationale,
            "note": check.note, "call": check.call.id if check.call else None}


def _refuse(admission: Admission, task: Task, unit: Unit, check: reader.Check, root: Root, to_director: bool,
            cost: str = "") -> None:
    """Log one refused branch; its line goes back to the director (`refusal`)."""
    line = refusal(unit, check)
    refused = {"event": "branch_refused", **_branch_event(unit, check), "reason": check.reason}
    if to_director:
        refused["to_director"] = line
    state.append_log(root, task.id, refused)
    admission.refused.append(line)
    admission.lines.append(f"reader refused {unit.id}'s branch ({check.reason}): {check.note}{cost}")


@dataclass
class Asked:
    """One reader check a cap hit does not end (`ask`). `check` is the one whose verdict applies: the re-ask's after a
    cap hit, the cap hit's own when no wider call could start. `calls` are the calls made, in order, each booked but,
    when `held`, the last one that reached its cap (`_hold` books it with the question). `held`: no verdict to apply,
    the wider cap reached too, or the wider call under the call floor (`floor`, its note). `retry_of`: the call the
    last one asked again."""

    check: reader.Check
    calls: list[Call] = field(default_factory=list)
    held: bool = False
    floor: str | None = None
    retry_of: str | None = None


def _capped(check: reader.Check) -> bool:
    """The call reached its dollar cap, on a receipt that says what it spent (one that cannot is unknown spend)."""
    return check.reason == "cap_hit" and check.call is not None and check.call.actual_usd != budget.UNKNOWN


def _call_event(call: Call, retry_of: str | None) -> dict[str, Any]:
    """The `call` event of a reader call; a re-ask names the call it asked again (`retry_of`)."""
    return {"event": "call", **asdict(call), **({"retry_of": retry_of} if retry_of else {})}


def _book(task: Task, root: Root, call: Call, retry_of: str | None = None) -> None:
    """Book one reader call on the task at once: saved, then its `call` event."""
    budget.book(task, call)
    state.save(root, task)
    state.append_log(root, task.id, _call_event(call, retry_of))


def ask(task: Task, unit: Unit, brief: dict[str, Any], caps: Caps, root: Root, runner: Runner, *,
        wide_after: str | None = None, **kwargs: Any) -> Asked:
    """One reader check (`reader.check`, `kwargs` passed through) that a cap hit does not end. A call that reached its
    cap is booked and the same brief is asked once more under twice the cap (`budget.wider`; `caps` are a call's, so
    the allowance's left still clips it); that verdict is the check's. Reached again, or with no room under the call
    floor for the wider call, the check is `held`. `wide_after` (the last call of a `reader_cap` question answered
    `yes`) asks under the wider cap from the first call, which is then the only one: its cap hit holds again."""
    check = reader.check(task, unit, brief, budget.wider(caps) if wide_after else caps, root, runner, **kwargs)
    asked = Asked(check, [check.call] if check.call else [], retry_of=wide_after)
    if _capped(check) and wide_after:
        asked.held = True
        return asked
    if check.call is not None:
        _book(task, root, check.call, wide_after)
    if not _capped(check):
        return asked
    retry = reader.check(task, unit, brief, budget.wider(caps), root, runner, **kwargs)
    if retry.reason == "floor":
        asked.held, asked.floor = True, retry.note
        return asked
    asked.check, asked.retry_of = retry, check.call.id
    if retry.call is not None:
        asked.calls.append(retry.call)
        if _capped(retry):
            asked.held = True
            return asked
        _book(task, root, retry.call, check.call.id)
    return asked


def _hold(task: Task, root: Root, asked: Asked, caps: Caps, held: dict[str, Any], prior: list[str]) -> str:
    """Hold a check on `reader_cap` (`questions.reader_cap`), with `held` (what the runtime needs to act on the answer)
    and every reader call on this check so far: `prior`, the calls of the question answered `yes` before, then this
    ask's. Its last call, when it reached the cap unbooked, is booked in the save that opens the question."""
    logged = []
    last = asked.calls[-1] if asked.calls else None
    if last is not None and not last.booked:
        budget.book(task, last)
        logged.append(_call_event(last, asked.retry_of))
    calls = [call for call in task.calls if call.id in prior] + asked.calls
    return questions.reader_cap(root, task, {**held, "note": asked.check.note}, calls,
                                (caps.reader_call_usd, budget.wider(caps).reader_call_usd), asked.floor, logged)


def accept(task: Task, unit: Unit, root: Root) -> dict[str, Any] | None:
    """At the branching unit's integration: each parent gains its children, records the last accepted commit it passed
    on, and so leaves the re-check set; its record is otherwise kept. The caller saves with the integration and logs
    the returned `branch_accepted` event. None when the unit proposed no branch."""
    children = children_of(unit)
    if not children:
        return None
    for pid, predicate in zip(grade.predicate_ids(unit), unit.predicates):
        for key in predicate.parents:
            found = find(task, key)
            if found is not None and f"{unit.id} {pid}" not in found[1].children:
                found[1].children.append(f"{unit.id} {pid}")
    parents = parents_of(unit)
    for key in parents:
        found = find(task, key)
        if found is not None:
            found[1].last_passed_commit = last_passed(root, task.id, key)
    return {"event": "branch_accepted", "unit": unit.id, "parents": parents, "children": children,
            "candidate": unit.candidate,
            "last_passed": {key: found[1].last_passed_commit for key in parents if (found := find(task, key))}}


@dataclass
class Weakening:
    """The weakening check on one candidate. `protected` is the set in force; `touched`, the changed paths it covers.
    `result`: `untouched` (nothing touched, no call), `not_weakened` (the reader said no), `weakened` (yes),
    `unchecked` (no verdict came back, or the diff could not be read), or `held` (the reader reached its cap twice:
    the attempt waits on `reader_cap`, never recorded in a verdict). `checks` names the accepted commands whose
    protected paths were touched; `reason` is the reader's verdict or why there was none; `kind` the kind of change its
    answer named; `note` its line, or why the check could not run; `call` the reader's record, None without a call."""

    protected: list[str]
    touched: list[str] = field(default_factory=list)
    result: str = UNTOUCHED
    checks: list[str] = field(default_factory=list)
    reason: str | None = None
    kind: str | None = None
    note: str | None = None
    call: Call | None = None

    @property
    def rejects(self) -> bool:
        """The candidate may not integrate: weakened, or no verdict."""
        return self.result in (WEAKENED, UNCHECKED)

    def record(self) -> dict[str, Any]:
        """The verdict event's `weakening`: the set, what was touched and the result; with what the check found once
        anything was touched (or the diff could not be read)."""
        record: dict[str, Any] = {"protected": self.protected, "touched": self.touched, "result": self.result}
        if self.result != UNTOUCHED:
            record.update(checks=self.checks, reason=self.reason, kind=self.kind, note=self.note,
                          call=self.call.id if self.call else None)
        return record


def protected_set(task: Task, excused: Collection[str]) -> dict[str, list[str]]:
    """The protected paths in force on a candidate, each with the accepted commands (`"<unit> <pid>"`) that name it:
    every integrated unit's retained `command` still in the re-check set (`grade.rechecked`), less the parents in
    `excused` (`gate.excused`: carried by the candidate's own passing children). The candidate's unit is not
    integrated, so its own commands never count."""
    paths: dict[str, list[str]] = {}
    for other in task.units:
        if not other.integrated:
            continue
        for pid, predicate in zip(grade.predicate_ids(other), other.predicates):
            key = f"{other.id} {pid}"
            if predicate.kind == "command" and grade.rechecked(predicate) and key not in excused:
                for path in predicate.protected:
                    paths.setdefault(path, []).append(key)
    return paths


def covers(protected: str, path: str) -> bool:
    """Whether a protected path (a file or a directory) covers the changed `path`: the same path, or a leading run of
    its components, so `tests` covers `tests/test_x.py` and never `tests_old.py`, and `.` covers the whole tree.
    Compared case-folded: on a case-insensitive disk `Tests/` is `tests/`, so a false touch costs one reader call
    where a miss would cost the check."""
    parts = PurePosixPath(os.path.normpath(protected).casefold()).parts
    return PurePosixPath(path.casefold()).parts[:len(parts)] == parts


def _diff(clone: Path, *args: str) -> str:
    """One hardened `git diff` in the grading clone (`grade.GIT_HARDENING`, `grade.git_env()`: check commands wrote
    in it), decoded leniently. Never a rename pairing (a protected file moved out shows as its deletion, not only as the
    new name), an external diff, a textconv filter or a replace object."""
    proc = subprocess.run(["git", *grade.GIT_HARDENING, "--literal-pathspecs", "diff", "--no-ext-diff", "--no-textconv",
                           "--no-renames", *args], cwd=clone, capture_output=True, env=grade.git_env())
    if proc.returncode != 0:
        raise worker.WorkerError(f"git diff: {proc.stderr.decode('utf-8', 'replace').strip()[-200:]}")
    return proc.stdout.decode("utf-8", "replace")


def touched(clone: Path, tip: str, candidate: str, protected: Collection[str]) -> list[str]:
    """The paths `tip..candidate` changes that a protected path covers, in git's order."""
    names = _diff(clone, "--name-only", "-z", f"{tip}..{candidate}").split("\0")
    return [name for name in names if name and any(covers(path, name) for path in protected)]


def hunks(clone: Path, tip: str, candidate: str, paths: list[str]) -> str:
    """`git diff tip..candidate -- paths`, cut at `HUNK_BOUND` characters with a marker saying so."""
    text = _diff(clone, f"{tip}..{candidate}", "--", *paths)
    if len(text) <= HUNK_BOUND:
        return text
    return (text[:HUNK_BOUND] + f"\n[... the diff is cut here at {HUNK_BOUND} of {len(text)} characters; the files "
            "themselves are in your working directory ...]\n")


def weakening_brief(task: Task, unit: Unit, tip: str, candidate: str, check: Weakening, diff: str) -> dict[str, Any]:
    """What the reader is given for the weakening check: which check this is, the unit's intent, the accepted commands
    whose protected paths were touched (kind, args, protected), the touched paths, the bounded hunks, and the
    question. Its working directory holds the candidate."""
    checks = []
    for key in check.checks:
        found = find(task, key)
        if found is not None:
            checks.append({"id": key, "kind": found[1].kind, "args": found[1].args,
                           "protected": list(found[1].protected)})
    return {"check": "weakening", "unit": unit.id, "intent": unit.intent, "accepted_tip": tip, "candidate": candidate,
            "checks": checks, "touched": check.touched, "hunks": diff, "question": WEAKENING_QUESTION}


def weakening(task: Task, unit: Unit, clone: Path, tip: str, candidate: str, excused: Collection[str], caps: Caps,
              root: Root, runner: Runner, *, charge: Charge = no_charge, answered: Question | None = None,
              attempt: dict[str, Any] | None = None) -> Weakening:
    """The weakening check on `unit`'s `candidate`, which branched from the accepted `tip` and already passed its own
    grade, the re-checks and the clean-clone check (the caller's order). No protected path touched: no call, nothing
    logged; the caller records the result in the verdict. Touched: one fresh reader call in a clone at the
    candidate, its record booked on the task at once (saved, `call` event) and one `weakening` event logged; a call
    that reached its cap is asked once more under the wider cap (`ask`). Reached again, the result is `held` and
    nothing is logged but the `reader_cap` question, whose context carries `attempt` (what the caller needs to finish
    the attempt); `answered` is that question answered `yes`, and the reader is asked under the wider cap at once. The
    worker's spend is already booked, so `charge` takes the reader's cap alone if the call is interrupted."""
    in_force = protected_set(task, excused)
    check = Weakening(sorted(in_force))
    if not in_force:
        return check
    try:
        check.touched = touched(clone, tip, candidate, in_force)
        diff = hunks(clone, tip, candidate, check.touched) if check.touched else ""
    except worker.WorkerError as exc:
        check.result, check.reason = UNCHECKED, "diff"
        check.note = f"weakening check could not run: the diff could not be read: {exc}"
        state.append_log(root, task.id, {"event": "weakening", **_event(unit, tip, candidate, check)})
        return check
    if not check.touched:
        return check
    check.checks = list(dict.fromkeys(key for path, keys in in_force.items()
                                      if any(covers(path, name) for name in check.touched) for key in keys))
    prior = answered.context["reader_cap"]["calls"] if answered is not None else []
    if answered is not None:
        answered.context["settled"] = True
    asked = ask(task, unit, weakening_brief(task, unit, tip, candidate, check, diff), caps, root, runner,
                wide_after=prior[-1] if prior else None, charge=charge, base=candidate, kinds=reader.KINDS,
                attempt=unit.attempts)
    found = asked.check
    check.reason, check.kind, check.call = found.reason, found.kind, found.call
    if asked.held:  # undecided: no `weakening` event until the operator's answer decides it
        check.result, check.note = HELD, f"weakening check could not run: {found.note}"
        _hold(task, root, asked, caps, {"check": "weakening", "unit": unit.id, "attempt": unit.attempts,
                                        "accepted_tip": tip, "candidate": candidate, "protected": check.protected,
                                        "touched": check.touched, "checks": check.checks, **(attempt or {})}, prior)
        return check
    if found.verdict is None:
        check.result, check.note = UNCHECKED, f"weakening check could not run: {found.note}"
    else:
        check.result, check.note = WEAKENED if found.verdict == "yes" else NOT_WEAKENED, found.note
    state.append_log(root, task.id, {"event": "weakening", **_event(unit, tip, candidate, check)})
    return check


def declined(task: Task, unit: Unit, question: Question, root: Root) -> Weakening:
    """`no` on a held weakening check: the attempt is rejected exactly as a cap hit rejected it before, `unchecked` with
    reason `cap_hit`, the note of the last call that reached its cap and that call; its `weakening` event logged."""
    info = question.context["reader_cap"]
    question.context["settled"] = True
    call = next((c for c in task.calls if c.id == info["calls"][-1]), None)
    check = Weakening(info["protected"], info["touched"], UNCHECKED, info["checks"], "cap_hit", None,
                      f"weakening check could not run: {info['note']}", call)
    state.append_log(root, task.id, {"event": "weakening",
                                     **_event(unit, info["accepted_tip"], info["candidate"], check)})
    return check


def _event(unit: Unit, tip: str, candidate: str, check: Weakening) -> dict[str, Any]:
    """The `weakening` event's fields."""
    return {"unit": unit.id, "attempt": unit.attempts, "accepted_tip": tip, "candidate": candidate,
            "touched": check.touched, "checks": check.checks, "result": check.result, "reason": check.reason,
            "kind": check.kind, "note": check.note, "call": check.call.id if check.call else None}
