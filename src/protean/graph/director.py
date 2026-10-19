"""The director seat in its thinker role: one call turns the thought record into one move.

`thinker_message` is what it is shown, `decode_move` reads its one move plus `state`, `no_progress` says whether a
reply moved the standing, and `next_move` makes the call. The seat's key stays `director` (config, `Call.seat`, prices,
ceilings, a question's `asker`, the `director` log event); "thinker" is the role's name in prose.

The reply is decoded here and never trusted for status: a unit a `task` move adds starts `pending` whatever the reply
says, and only the grader can pass it.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from protean.graph import budget, grade
from protean.graph.budget import Caps
from protean.operator import Operator, load_operator
from protean.runner import (Charge, Runner, RunResult, cap_bound, charged_cap, floor_reason, no_charge, usd_cap,
                            usd_left)
from protean.state import (DIRECTOR, LOCK, MOVE_KINDS, Idea, Move, OpenItem, Predicate, Root, Task, Unit,
                           predicate_view, save)

TOOLS = ("Read", "Grep", "Glob")
UNIT_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}")
PARENT_ID = re.compile(r"(\S+) (p[1-9][0-9]*)")
# The gate's admission of a decoded plan's branches (`gate.admit`, bound by the loop): the plan's units and what this
# plan has spent but not booked; the refusal line(s) to retry with, empty when admitted. The director never sees more.
Admit = Callable[[list[Unit], float], list[str]]


def read_log(root: Root, task_id: str) -> list[dict[str, Any]]:
    """Every event in the task's log, oldest first; an absent log is empty."""
    try:
        lines = root.log_file(task_id).read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        return []
    return [json.loads(line) for line in lines if line.strip()]


def pending_notes(root: Root, task_id: str) -> list[str]:
    """The `to_director` texts logged since the last director call: the operator's answer to the director's own question,
    runtime refusals. A runtime question's answer never carries one."""
    notes: list[str] = []
    for event in read_log(root, task_id):
        if event.get("event") in ("director", "manager"):
            notes = []
        elif note := event.get("to_director") or event.get("to_manager"):
            notes.append(note)
    return notes


def _units_view(task: Task) -> list[dict[str, Any]]:
    """The units as the seat sees them: status, attempts, needs, the worker's last summary with its grade, and the
    predicates."""
    return [{"id": u.id, "intent": u.intent, "status": u.status, "attempts": u.attempts, "needs": u.needs,
             "summary": u.summary, "predicates": [predicate_view(p) for p in u.predicates]} for u in task.units]


def _move_view(move: Move) -> dict[str, Any]:
    """A move as the thinker sees it: id, kind, line, cost and outcome, then only the pointers it carries; an
    explore's report is named unverified."""
    shown: dict[str, Any] = {"id": move.id, "kind": move.kind, "line": move.line, "cost_usd": move.cost_usd,
                             "outcome": move.outcome}
    if move.branch:
        shown["branch"] = move.branch
    if move.units:
        shown["units"] = list(move.units)
    if move.question:
        shown["question"] = move.question
    if move.ideas:
        shown["ideas"] = [asdict(idea) for idea in move.ideas]
    if move.report is not None:
        shown["unverified_report"] = move.report
    return shown


def last_answer(task: Task) -> dict[str, str] | None:
    """The operator's answer to the thinker's last `ask`: the seat's latest question of its own, once answered; None while it is
    open or when it never asked. A runtime question is never one."""
    asked = [q for q in task.questions if q.asker == DIRECTOR and q.reason == "question"]
    if not asked or asked[-1].status != "answered" or asked[-1].answer is None:
        return None
    return {"id": asked[-1].id, "question": asked[-1].text, "answer": asked[-1].answer}


def thinker_message(task: Task, caps: Caps, notes: list[str], operator: dict[str, str] | None = None) -> dict[str, Any]:
    """What the thinker is shown each call: the thought record (thought, standing, open, decided),
    every thought in order, the units with their statuses and the worker's summaries, the moves, the budget left for
    the session (its allowance less its cost), the notes the loop passes, and the operator's answer
    to its last `ask`. Built from the record and `notes` alone, so no runtime question or answer, transcript line
    or reader exchange is ever in it."""
    message: dict[str, Any] = {
        "thought": task.thought,
        "standing": task.standing,
        "open": [asdict(item) for item in task.open],
        "decided": [asdict(decision) for decision in task.decided],
        "thoughts": [t.text for t in task.thoughts],
        "operator": operator if operator is not None else Operator().as_message(),
        "workspace": task.workspace,
        "tick": task.tick,
        "budget_left_usd": round(budget.allowance(task, caps) - task.cost_usd, 2),
        "ticks_left": caps.max_ticks + task.extra_ticks - task.tick,
        "max_attempts_per_unit": caps.max_attempts_per_unit,
        "units": _units_view(task),
        "moves": [_move_view(move) for move in task.moves],
        "notes": notes,
    }
    answer = last_answer(task)
    if answer is not None:
        message["answer"] = answer
    return message


def argv(task: Task, caps: Caps, root: Root, session: str | None, spent: float = 0.0) -> list[str]:
    """The thinker's call: no user settings, read-only tools on the workspace, the prompt on every call, resumed when
    it can be."""
    args = [
        "claude", "-p", "--setting-sources", "", "--output-format", "json",
        "--max-budget-usd", usd_cap(caps.director_call_usd, task, caps, spent),
        "--model", caps.director_model, "--permission-mode", "dontAsk", "--tools", *TOOLS, "--allowedTools", *TOOLS,
        "--add-dir", task.workspace,
    ]
    if session:
        args += ["--resume", session]
    return args + ["--system-prompt", root.thinker_prompt.read_text(encoding="utf-8")]


def _closers(text: str, start: int) -> str:
    """The closing brackets the JSON value at `start` still owes when the text ends; "" inside a string."""
    stack: list[str] = []
    in_string = escaped = False
    for char in text[start:]:
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
        elif char == '"':
            in_string = True
        elif char in "{[":
            stack.append("}" if char == "{" else "]")
        elif char in "}]" and stack and stack[-1] == char:
            stack.pop()
    return "" if in_string else "".join(reversed(stack))


def first_object(text: str) -> tuple[dict | None, str]:
    """The first `{...}` in `text` that decodes as a JSON object, and why the first `{` did not when it did not.

    An object the text cuts off before its closing brackets is completed and read; the director has
    ended a whole plan one `}` short.
    """
    decoder = json.JSONDecoder()
    problem = ""
    for index, char in enumerate(text):
        if char != "{":
            continue
        try:
            obj, _ = decoder.raw_decode(text, index)
        except json.JSONDecodeError as exc:
            problem = problem or f"the object at the first {{ did not parse: {exc.msg} at char {exc.pos}"
            closers = _closers(text, index)
            if not closers:
                continue
            try:
                obj, _ = decoder.raw_decode(text + closers, index)
            except json.JSONDecodeError:
                continue
        if isinstance(obj, dict):
            return obj, problem
    return None, problem


def _decode_units(items: object, task: Task) -> tuple[list[Unit], list[str]]:
    if not isinstance(items, list) or not items:
        return [], ["units must be a non-empty list"]
    seen = {u.id for u in task.units}
    units, errors = [], []
    for n, item in enumerate(items, start=1):
        if not isinstance(item, dict):
            errors.append(f"unit {n}: must be an object")
            continue
        uid, intent, preds, needs = item.get("id"), item.get("intent"), item.get("predicates"), item.get("needs", [])
        label = f"unit {uid!r}" if isinstance(uid, str) else f"unit {n}"
        if not isinstance(needs, list) or not all(isinstance(need, str) for need in needs):
            errors.append(f"{label}: needs must be a list of unit ids")
            needs = []
        if not isinstance(uid, str) or not UNIT_ID.fullmatch(uid):
            errors.append(f"{label}: id must match {UNIT_ID.pattern}")
        elif uid in seen:
            errors.append(f"{label}: id is already used")
        if not isinstance(intent, str) or not intent.strip():
            errors.append(f"{label}: intent must be a non-empty string")
        if not isinstance(preds, list) or not all(isinstance(p, dict) and {"kind", "args"} <= p.keys() for p in preds):
            errors.append(f"{label}: predicates must be a list of {{\"kind\", \"args\"}} objects")
            continue
        predicates = []
        for index, p in enumerate(preds, start=1):
            retain, parents, protected = p.get("retain", True), p.get("parents", []), p.get("protected", [])
            if not isinstance(retain, bool):
                errors.append(f"{label}: p{index} {p['kind']}: retain must be true or false, not {retain!r}")
            if not isinstance(parents, list) or not all(isinstance(key, str) for key in parents):
                errors.append(f"{label}: p{index} {p['kind']}: parents must be a list of \"<unit> <pid>\" ids")
                parents = []
            if not isinstance(protected, list) or not all(isinstance(path, str) and path.strip() for path in protected):
                errors.append(f"{label}: p{index} {p['kind']}: protected must be a list of paths relative to the tree")
                protected = None
            predicate = Predicate(p["kind"], p["args"], retain is not False, list(dict.fromkeys(parents)),
                                  protected=list(dict.fromkeys(protected or [])))
            predicates.append(predicate)
            if protected is not None:
                errors += [f"{label}: p{index} {p['kind']}: {problem}" for problem in _protected_problems(predicate)]
        errors += [f"{label}: {problem}" for problem in grade.validate(predicates)]
        errors += [f"{label}: {problem}" for problem in _path_errors(predicates, task)]
        rationale, branch = item.get("rationale", ""), any(p.parents for p in predicates)
        rationale = rationale.strip() if isinstance(rationale, str) else ""
        if branch and not rationale:
            errors.append(f"{label}: a unit whose predicates carry parents needs a non-empty rationale (what the "
                          "parents promised and how the children together keep it)")
        # The operator's plain sentence is optional: a plan without one decodes as every plan before it did.
        summary = item.get("summary")
        summary = summary.strip() if isinstance(summary, str) else ""
        if isinstance(uid, str):
            seen.add(uid)
            units.append(Unit(id=uid, intent=str(intent), predicates=predicates, needs=list(dict.fromkeys(needs)),
                              rationale=rationale if branch else "", plain_summary=summary))
    return units, errors + _need_errors(units, task) + _parent_errors(units, task)


def _parent_errors(units: list[Unit], task: Task) -> list[str]:
    """Each parent a new predicate names that is not an accepted, retained predicate still in the re-check set and
    unclaimed: no such predicate in the task, its unit not integrated, delivery-only or never re-checked, already
    branched, or already claimed by another unit still running (in the task or this plan). A delivery-only predicate
    carries no parents. One parent may pass to several children of one unit, and one child may take several parents."""
    claims = {key: u.id for u in task.units if u.status == "pending" or (u.status == "passed" and not u.integrated)
              for p in u.predicates for key in p.parents}
    errors = []
    for unit in units:
        for pid, predicate in zip(grade.predicate_ids(unit), unit.predicates):
            label = f"unit {unit.id!r}: {pid} {predicate.kind}"
            if predicate.parents and predicate.retain is False:
                errors.append(f"{label}: a delivery-only predicate (retain: false) cannot carry parents")
                continue
            for key in predicate.parents:
                problems = _parent_problem(key, task)
                if not problems and claims.setdefault(key, unit.id) != unit.id:
                    problems = [f"is already claimed by unit {claims[key]!r}, whose branch is not settled"]
                errors += [f"{label}: parent {key!r} {problem}" for problem in problems]
    return errors


def _parent_problem(key: str, task: Task) -> list[str]:
    """Why `key` cannot be a parent: not `<unit> <pid>` of a task unit, or not accepted, retained and unbranched."""
    match = PARENT_ID.fullmatch(key)
    owner = next((u for u in task.units if match and u.id == match.group(1)), None)
    found = dict(zip(grade.predicate_ids(owner), owner.predicates)).get(match.group(2)) if owner else None
    if found is None:
        return ["is no predicate of a unit in the task (parents name accepted predicates as \"<unit> <pid>\")"]
    if not owner.integrated:
        return [f"belongs to {owner.id}, which is not accepted ({owner.status}, not integrated)"]
    if found.retain is False:
        return ["is delivery-only (retain: false), not a lasting promise"]
    if found.kind not in grade.RECHECKABLE:
        return [f"is never re-checked (kind {found.kind})"]
    if found.children:
        return [f"already has children ({', '.join(found.children)})"]
    return []


def _protected_problems(predicate: Predicate) -> list[str]:
    """`protected` belongs on a `command` only, and a retained command must name at least one path that controls its
    result. A delivery-only
    command may omit it; what it names is kept and never consulted."""
    if predicate.kind != "command":
        return ["protected is only for a command predicate"] if predicate.protected else []
    if predicate.retain and not predicate.protected:
        return ['a retained command needs "protected": the paths that control its result (its tests, their config), '
                'or "retain": false when it only proves this unit\'s delivery']
    return []


def _path_errors(predicates: list[Predicate], task: Task) -> list[str]:
    """Each well-formed file predicate whose path, and each path a `command` names as protected, the grader's
    confinement rule (`grade.confined`) refuses in the workspace: absolute, inside `.git`, climbing out with `..`, or
    leaving the tree through a symlink. A correctable error for the director, never a veto; a command's own `cmd` is
    not path-checked."""
    errors = []
    for index, predicate in enumerate(predicates, start=1):
        if predicate.kind in grade.PATH_KINDS and not grade.validate([predicate]):
            problem = grade.confined(Path(task.workspace), predicate.args["path"])
            if problem:
                errors.append(f"p{index} {predicate.kind}: {problem}")
        if predicate.kind == "command":
            errors += [f"p{index} {predicate.kind}: protected {problem}" for path in predicate.protected
                       if (problem := grade.confined(Path(task.workspace), path))]
    return errors


def _need_errors(units: list[Unit], task: Task) -> list[str]:
    """A need naming no unit in the task or this plan, a unit needing itself, and needs that form a cycle."""
    graph = {u.id: u.needs for u in task.units + units}
    legacy = {u.id for u in task.units if u.status == "passed" and not u.candidate}
    errors = []
    for unit in units:
        for need in unit.needs:
            if need == unit.id:
                errors.append(f"unit {unit.id!r}: needs itself")
            elif need not in graph:
                errors.append(f"unit {unit.id!r}: needs {need!r}, which is no unit in the task or this plan")
            elif need in legacy:
                errors.append(f"unit {unit.id!r}: needs {need!r}, which passed before the accepted branch existed "
                              "and cannot be built on")
        path = _cycle(unit.id, graph)
        if path:
            errors.append(f"unit {unit.id!r}: needs form a cycle: {' -> '.join(path)}")
    return errors


def _cycle(start: str, graph: dict[str, list[str]]) -> list[str] | None:
    """A path of needs from `start` back to itself, or None; a unit needing itself is reported apart."""
    stack, seen = [[start]], {start}
    while stack:
        path = stack.pop()
        for need in graph.get(path[-1], []):
            if need == start and len(path) > 1:
                return path + [start]
            if need in graph and need not in seen:
                seen.add(need)
                stack.append(path + [need])
    return None


IDEAS_MIN, IDEAS_MAX = 3, 7
STANDING_SENTENCES = 5
STATE_KEYS = ("standing", "open")
BRAIN_KEYS = ("decided", "moves")  # the thought record's parts only the brain writes
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


@dataclass
class MoveReply:
    """One decoded thinker answer: exactly one move (`kind`, one of `MOVE_KINDS`) and the `state` it leaves, its
    `standing` stripped and its `open` items. The move's content sits in the field its kind uses: `ideas`, `brief`
    (explore), `units` (task, every one `pending`) or `question` (ask); a rest carries none."""

    kind: str
    standing: str
    open: list[OpenItem] = field(default_factory=list)
    ideas: list[Idea] = field(default_factory=list)
    brief: str = ""
    units: list[Unit] = field(default_factory=list)
    question: str = ""


def _one_line(value: object) -> str | None:
    """`value` stripped when it is one non-empty line of text, else None."""
    if not isinstance(value, str) or not value.strip() or "\n" in value.strip():
        return None
    return value.strip()


def _text(value: object) -> str | None:
    return value.strip() if isinstance(value, str) and value.strip() else None


def sentences(text: str) -> int:
    """How many sentences `text` holds: pieces ended by `.`, `!` or `?` and a space, plus any unended tail."""
    return len([piece for piece in _SENTENCE_END.split(text.strip()) if piece.strip()])


def _decode_state(present: bool, value: object) -> tuple[str, list[OpenItem] | None, list[str]]:
    """`state`: a non-empty `standing` of at most five sentences and an `open` list of id + one line; never `decided`
    or `moves`, nothing else. The open items are None when they cannot be read, so nothing is checked against them."""
    if not present:
        return "", None, ['the reply must carry state: {"standing": "...", "open": [{"id": "...", "line": "..."}]}']
    if not isinstance(value, dict):
        return "", None, ["state must be an object with standing and open"]
    errors = [f"state cannot carry {key}: only the brain writes it" for key in BRAIN_KEYS if key in value]
    unknown = [key for key in value if key not in STATE_KEYS + BRAIN_KEYS]
    if unknown:
        errors.append(f"state holds only standing and open; it carried {unknown}")
    standing = value.get("standing")
    if not isinstance(standing, str) or not standing.strip():
        errors.append("state.standing must be a non-empty string")
        standing = ""
    elif (count := sentences(standing)) > STANDING_SENTENCES:
        errors.append(f"state.standing must be at most {STANDING_SENTENCES} sentences; it has {count}")
    items = value.get("open")
    if not isinstance(items, list):
        errors.append("state.open must be a list of {\"id\", \"line\"} items, empty when nothing is open")
        return standing.strip(), None, errors
    open_items, seen = [], set()
    for n, item in enumerate(items, start=1):
        if not isinstance(item, dict):
            errors.append(f"open {n}: must be an object")
            continue
        oid, line = item.get("id"), _one_line(item.get("line"))
        label = f"open {oid!r}" if isinstance(oid, str) else f"open {n}"
        if not isinstance(oid, str) or not UNIT_ID.fullmatch(oid):
            errors.append(f"{label}: id must match {UNIT_ID.pattern}")
        elif oid in seen:
            errors.append(f"{label}: id is already used")
        if line is None:
            errors.append(f"{label}: line must be one non-empty line")
        if isinstance(oid, str):
            seen.add(oid)
            open_items.append(OpenItem(oid, line or ""))
    return standing.strip(), open_items, errors


def _decode_ideas(value: object, open_items: list[OpenItem] | None) -> tuple[list[Idea], list[str]]:
    """3 to 7 ideas, each a unique id, a one-line case, a one-line cost to explore it, and the id of an item in this
    reply's `open` that it serves."""
    if not isinstance(value, list) or not IDEAS_MIN <= len(value) <= IDEAS_MAX:
        had = f"; it has {len(value)}" if isinstance(value, list) else ""
        return [], [f"ideas must be a list of {IDEAS_MIN} to {IDEAS_MAX} items{had}"]
    open_ids = None if open_items is None else [item.id for item in open_items]
    ideas, errors, seen = [], [], set()
    for n, item in enumerate(value, start=1):
        if not isinstance(item, dict):
            errors.append(f"idea {n}: must be an object")
            continue
        iid, serves = item.get("id"), item.get("serves")
        label = f"idea {iid!r}" if isinstance(iid, str) else f"idea {n}"
        if not isinstance(iid, str) or not UNIT_ID.fullmatch(iid):
            errors.append(f"{label}: id must match {UNIT_ID.pattern}")
        elif iid in seen:
            errors.append(f"{label}: id is already used")
        lines = {key: _one_line(item.get(key)) for key in ("case", "cost")}
        errors += [f"{label}: {key} must be one non-empty line" for key, line in lines.items() if line is None]
        if not isinstance(serves, str) or not serves:
            errors.append(f"{label}: serves must name the open id it serves")
        elif open_ids is not None and serves not in open_ids:
            errors.append(f"{label}: serves {serves!r}, which is no open id in this state "
                          f"({', '.join(open_ids) or 'none'})")
        if isinstance(iid, str):
            seen.add(iid)
            ideas.append(Idea(iid, lines["case"] or "", lines["cost"] or "", serves if isinstance(serves, str) else ""))
    return ideas, errors


def _decode_one(kind: str, value: object, task: Task, open_items: list[OpenItem] | None) -> tuple[MoveReply, list[str]]:
    """The move's own content; a `task` move's units through the plan's own validation (`_decode_units`)."""
    reply = MoveReply(kind=kind, standing="")
    if kind == "ideas":
        reply.ideas, errors = _decode_ideas(value, open_items)
        return reply, errors
    if kind == "task":
        reply.units, errors = _decode_units(value, task)
        return reply, errors
    if kind == "rest":
        return reply, [] if value is True else ["rest must be true"]
    text, what = _text(value), {"explore": "brief", "ask": "question"}[kind]
    if text is None:
        return reply, [f"{kind} must be one non-empty {what}"]
    if kind == "explore":
        reply.brief = text
    else:
        reply.question = text
    return reply, []


def decode_move(text: str, task: Task) -> tuple[MoveReply | None, list[str]]:
    """Decode a thinker reply against the task: exactly one of the five moves plus `state`, nothing else. Any other
    shape returns every problem found, for the retry with its errors (the same path as a malformed plan); `decided`
    and `moves` are the brain's, and the reply never sets a unit's status."""
    obj, problem = first_object(text)
    if obj is None:
        return None, [problem or "the reply holds no JSON object"]
    errors = []
    moves = [key for key in MOVE_KINDS if key in obj]
    if len(moves) != 1:
        carried = f"it carried {moves or 'none'}" + (f" ({problem})" if problem else "")
        errors.append(f"the reply must carry exactly one move of {', '.join(MOVE_KINDS)}; {carried}")
    extra = [key for key in obj if key not in MOVE_KINDS and key != "state"]
    if extra:
        errors.append(f"the reply carries one move and state only; drop {', '.join(map(repr, extra))}")
    standing, open_items, state_errors = _decode_state("state" in obj, obj.get("state"))
    errors += state_errors
    reply = None
    if len(moves) == 1:
        reply, move_errors = _decode_one(moves[0], obj[moves[0]], task, open_items)
        errors += move_errors
    if errors or reply is None:
        return None, errors
    reply.standing, reply.open = standing, open_items or []
    return reply, []


def no_progress(last_standing: str, standing: str) -> bool:
    """a reply makes no progress when its `standing` is byte-identical to the last accepted
    move's (the record's `standing`). Pure; what follows (returned once, then `no_progress` asked) is the loop's."""
    return standing == last_standing


def _receipt(res: RunResult, timeout: int) -> tuple[dict | None, list[str]]:
    """The call's stdout receipt (None when it left none), and what makes it unusable as a reply."""
    if res.timed_out:
        return None, [f"the call timed out after {timeout}s"]
    try:
        receipt = json.loads(res.stdout)
    except json.JSONDecodeError:
        receipt = None
    if not isinstance(receipt, dict):
        return None, [f"stdout was not one JSON object (exit {res.returncode}): {res.stderr.strip()[-300:]}"]
    if receipt.get("is_error"):
        return receipt, [f"the call ended in error: {receipt.get('subtype', '')} {receipt.get('errors', '')}".strip()]
    if not isinstance(receipt.get("result"), str):
        return receipt, ["the receipt carries no result text"]
    return receipt, []


@dataclass
class Reply:
    """One thinker call's outcome: the decoded `move`, exactly one of the five plus its `state`, or why there is none.

    `unchanged` is set beside the move when its `standing` is byte-identical to the record's (`no_progress`): the loop
    does not act on it, and no reader is paid to admit it. `question` is set instead when the reply was invalid twice:
    the runtime's question, never the thinker's (its answer never reaches the seat). `stop` is set instead when a retry
    was due but the session had less left than the call floor; `errors` then carries what the rejected answer got
    wrong. `ceiling` is set instead when a call's receipt confirms it reached its dollar cap: the call, its effective
    cap, the constraint that bound it (`per_call_limit` or `task_budget`) and its charge. That call is never retried.
    `unreceipted` is set instead when a call left no receipt (timed out, or no JSON object on stdout): what went wrong,
    in one line. That call is never retried either; its spend is UNKNOWN and the loop holds paid work on it (ruled
    2026-10-11). `thoughts` is set beside a move: how many of the operator's thoughts the call's message carried, counted as it
    was built, so the loop can tell a think that landed during the call.
    """

    move: MoveReply | None = None
    unchanged: bool = False
    question: str | None = None
    cost_usd: float = 0.0
    session_id: str | None = None
    raw: str = ""
    unreceipted: str | None = None
    stop: str = ""
    errors: list[str] = field(default_factory=list)
    ceiling: dict[str, Any] | None = None
    thoughts: int | None = None

    @property
    def kind(self) -> str:
        """What the call came to: a move's kind, `no_progress`, `invalid`, `ceiling`, `stopped` or `unreceipted`."""
        if self.stop:
            return "stopped"
        if self.ceiling is not None:
            return "ceiling"
        if self.unreceipted is not None:
            return "unreceipted"
        if self.question is not None or self.move is None:
            return "invalid"
        return "no_progress" if self.unchanged else self.move.kind


def next_move(task: Task, caps: Caps, root: Root, runner: Runner, charge: Charge = no_charge,
              admit: Admit | None = None) -> Reply:
    """Call the thinker once, retry once with the errors, and park on the runtime's question after two invalid
    replies. A call that leaves no receipt is not retried (`Reply.unreceipted`).

    A reply whose `standing` is the record's own (`no_progress`) is returned at once, marked `unchanged`: what follows
    is the loop's (decision 4). With `admit` (the loop binds the gate in `auto`), a `task` move's branches pass the
    gate before the move is returned: a refusal line is a correctable error on the same retry path as a bad path, and
    nothing else of the reader's exchange reaches this seat.

    A call whose receipt confirms it reached its dollar cap (`budget.cap_hit`) is not retried: the reply carries
    `ceiling` at once. An incomplete reply without that signal takes the retry. The retry is skipped, and the reply
    carries `stop`, when the first call left less than the call floor. A call that leaves no receipt is charged the
    `--max-budget-usd` it carried. A call interrupted before it returns is charged the same way through `charge`, before
    the interruption propagates. Each call's record is saved with the task before it launches and again once its
    receipt is read.
    """
    seat = root.seat_dir(task.id).resolve()
    seat.mkdir(parents=True, exist_ok=True)
    notes = pending_notes(root, task.id)
    with LOCK:  # counted under the lock a console think lands under: the thoughts the message carries, no more
        seen = len(task.thoughts)
        message = json.dumps(thinker_message(task, caps, notes, load_operator(root).as_message()), indent=2)
    session, cost, errors, raw = task.director_session, 0.0, [], ""
    for attempt in range(2):
        stdin = message
        clip = budget.call_caps(task, caps)  # what the allowance has left bounds this call
        if attempt:
            stop = floor_reason(task, clip, cost)
            if stop:
                return Reply(stop=stop, errors=errors, cost_usd=cost, session_id=session, raw=raw)
            stdin += "\n\nYour previous answer was rejected:\n- " + "\n- ".join(errors)
            stdin += "\nAnswer again with exactly one JSON object."
        call = record = None
        bound, left = cap_bound(caps.director_call_usd, task, clip, cost), usd_left(task, clip, cost)
        try:
            call = argv(task, clip, root, session, cost)
            record = budget.open_call(task, "director", attempt + 1, charged_cap(call), stdin)
            save(root, task)
            res = runner(call, stdin, seat, caps.director_wall_seconds)
        except BaseException:
            if record is not None:
                budget.resolve(record, charged_cap(call), interrupted=True)
            charge(cost + (charged_cap(call) if call else float(usd_cap(caps.director_call_usd, task, clip, cost))))
            raise
        receipt, errors = _receipt(res, caps.director_wall_seconds)
        charged = charged_cap(call) if receipt is None else budget.cost_of(receipt)
        cost += charged
        budget.resolve(record, charged, receipt, timed_out=res.timed_out)
        save(root, task)
        if receipt is None:  # no receipt: never retried, the loop holds paid work on the record's UNKNOWN spend
            return Reply(unreceipted=errors[0], errors=errors, cost_usd=cost, session_id=session, raw=res.stdout[-2000:])
        if isinstance(receipt.get("session_id"), str):
            session = receipt["session_id"]
        raw = receipt.get("result") if isinstance(receipt.get("result"), str) else res.stdout[-2000:]
        if budget.cap_hit(receipt):
            ceiling = {"seat": "director", "call": record.id, "cap_usd": record.cap_usd, "bound_by": bound,
                       "per_call_limit_usd": caps.director_call_usd, "task_left_usd": round(left, 2),
                       "charged_usd": charged, "subtype": budget.CAP_HIT}
            return Reply(ceiling=ceiling, errors=errors, cost_usd=cost, session_id=session, raw=raw)
        if not errors:
            move, errors = decode_move(raw, task)
            if move is not None and no_progress(task.standing, move.standing):
                return Reply(move=move, unchanged=True, cost_usd=cost, session_id=session, raw=raw, thoughts=seen)
            if move is not None and move.units and admit is not None:
                errors = admit(move.units, cost)
                move = None if errors else move
            if move is not None:
                return Reply(move=move, cost_usd=cost, session_id=session, raw=raw, thoughts=seen)
    return Reply(question=f"The thinker returned an invalid reply twice: {'; '.join(errors)}", cost_usd=cost,
                 session_id=session, raw=raw, errors=errors)
