"""Task state: dataclasses, one JSON file per task, and an append-only event log.

The brain's console thread writes the same files as the loop, so every save and every log append takes `LOCK`.
"""

from __future__ import annotations

import copy
import json
import os
import re
import threading
from dataclasses import KW_ONLY, InitVar, asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from protean.graph import graph


@dataclass
class Predicate:
    """One checkable condition: a kind from `grade.PREDICATE_KINDS` and its arguments.

    `retain` is the director's flag, true unless it says otherwise: a retained predicate is a lasting promise,
    re-checked on every later candidate once its unit is accepted; a delivery-only one (false) must pass its own unit's
    grade and is never checked again. Nothing flips it after decode; a record from before the flag loads retained.

    Branching: `parents` names the accepted retained predicates (`"<unit> <pid>"`) this new one inherits, admitted once
    by the reader at the gate. When its unit is integrated, each parent gains it in `children`, records
    `last_passed_commit` (the last accepted commit it passed on, from the verdict records) and leaves the re-check set;
    its record is kept, never deleted or edited otherwise.

    `protected` names, on a `command`, the paths relative to the tree that control its result (its tests, their config;
    a directory covers everything under it). A retained command must name at least one; once its unit is accepted, a
    later candidate that touches one gets the gate's weakening check (one reader call) before it integrates. A record
    from before the field loads with none."""

    kind: str
    args: dict[str, Any]
    retain: bool = True
    parents: list[str] = field(default_factory=list)
    children: list[str] = field(default_factory=list)
    last_passed_commit: str | None = None
    protected: list[str] = field(default_factory=list)


def predicate_view(predicate: Predicate) -> dict[str, Any]:
    """A predicate as a seat or a client is shown it: `kind` and `args`, plus `"retain": false` on a delivery-only one
    (absent means retained, as in a plan), `parents` on a child, `children` on a parent the branch replaced and
    `protected` on a command, each only when non-empty."""
    view: dict[str, Any] = {"kind": predicate.kind, "args": copy.deepcopy(predicate.args)}
    if predicate.retain is False:
        view["retain"] = False
    if predicate.parents:
        view["parents"] = list(predicate.parents)
    if predicate.children:
        view["children"] = list(predicate.children)
    if predicate.protected:
        view["protected"] = list(predicate.protected)
    return view


@dataclass
class Unit:
    """A small piece of work. `status` is one of pending, passed, failed, descoped, blocked.

    `needs` names units that must pass and be integrated first. `candidate` is the commit a pass
    accepted; `integrated` turns true once `task/<task>` has moved to it. `rationale` is the director's account of a
    branch (what the parents promised, how the children keep it), set only on a unit whose predicates carry parents.
    `summary` is the runtime's: the worker's last report with its grade, or why the unit was blocked or descoped.
    `plain_summary` is the director's one plain sentence for the operator, beside `intent`, the editor's brief (without it,
    plans read as "hyper concentrated gibberish slop"); the wire calls it `summary`, and a record without it loads
    with none."""

    id: str
    intent: str
    predicates: list[Predicate]
    status: str = "pending"
    attempts: int = 0
    branch: str | None = None
    summary: str = ""
    needs: list[str] = field(default_factory=list)
    candidate: str | None = None
    integrated: bool = False
    rationale: str = ""
    plain_summary: str = ""


def _no_tokens() -> dict[str, int | None]:
    return {"input": None, "output": None, "cache_read": None, "cache_creation": None}


@dataclass
class Call:
    """One model call in the receipt (`task.json` and `log.jsonl`, never a director message).

    Saved with `outcome` "pending" and its reserved `cap_usd` (the clipped `--max-budget-usd`) before the
    process starts; resolved to ok, error, timed_out, no_receipt or interrupted when it ends, or to lost
    when a resume finds it still pending. `attempt` is the unit's attempt for the worker and the try within
    one plan (1 or 2) for the director. `charge_usd` is what the task is charged; `receipt_cost_usd` is
    what the receipt reported, None without one. `actual_usd` is the receipt's figure, or "UNKNOWN" when no
    receipt can say; the worker and the reader are fresh sessions per call; the thinker resumes its session within a warm
    stretch, whose receipt reports the whole session, so its figure is taken less the session's earlier receipt
    (`budget.session_before`); `tokens` the same way. `booked` turns true once `task.cost_usd` includes the charge. A measurement the
    receipt does not carry is None, never 0. `walk` is the id of the walk (`Fired.walk`) the call was opened in; None on
    a record from before the field and on a call opened outside a walk."""

    id: str
    seat: str
    attempt: int
    cap_usd: float
    unit: str | None = None
    tick: int = 0
    message_bytes: int | None = None
    outcome: str = "pending"
    subtype: str | None = None
    charge_usd: float | None = None
    receipt_cost_usd: float | None = None
    actual_usd: float | str | None = None
    tokens: dict[str, int | None] = field(default_factory=_no_tokens)
    session_id: str | None = None
    booked: bool = False
    started: str | None = None  # when the pending record was opened (UTC, ISO seconds); None on an older record
    walk: str | None = None


DIRECTOR, RUNTIME = "director", "runtime"


@dataclass
class Question:
    """One question for the operator, kept in `task.json`. `asker` is director or runtime; `reason` names why it was asked
    (question, invalid_plan, no_progress, stale_tip, cap_hit, unknown_spend; descope and done_refused on
    goal-mode records). `status` is open, answered
    or superseded; an answer binds to this id only, once, with its declared `actor`. `disposition` is what the
    runtime did once it acted on the item; `context` holds what it needs to act (the descope's units, the calls)."""

    id: str
    asker: str
    text: str
    reason: str
    created: str = ""
    status: str = "open"
    answer: str | None = None
    actor: str | None = None
    disposition: str | None = None
    context: dict[str, Any] = field(default_factory=dict)


@dataclass
class Thought:
    """One of the operator's contributions to the session (a console `think`; a `steer` on a goal-mode record),
    carried whole and in order in every later message to the seat: direction, an answer, a possibility or a decision,
    as the seat reads it. `request_id` is the client's id for the `think` that landed it (`Request`); None when the
    client sent none. The seat is sent the text only."""

    text: str
    created: str = ""
    actor: str = "console"
    request_id: str | None = None


Direction = Thought  # goal mode's name, kept for the tests that still build them by it


@dataclass
class OpenItem:
    """One possibility or question still live in the thought record: an id and one line, as the thinker's `state`
    last named it."""

    id: str
    line: str


@dataclass
class Decision:
    """One of the operator's decisions: their text verbatim, when, and the index in `thoughts` (from 1)
    of the `think` that said it. Only a console `think` marked `decide` adds one; the thinker never does."""

    text: str
    created: str = ""
    think: int = 0


MOVE_KINDS = ("ideas", "explore", "task", "ask", "rest")


@dataclass
class Idea:
    """One option of an `ideas` move: its id, a one-line case, a one-line cost to explore it, and the `open` id it
    serves."""

    id: str
    case: str
    cost: str
    serves: str


@dataclass
class Move:
    """One move in the session's log, written by the brain only. `kind` is one of `MOVE_KINDS`; `line` says it in one
    line; `cost_usd` is what it spent; `outcome` what became of it. Its pointers are set where the kind carries one:
    `branch` (an explore's `explore/<session>/<n>`, a task's `task/<session>`), `units` (a task's unit ids), `question`
    (an ask's question id), `ideas` (an ideas move's items), `brief` (an explore's brief, whole, as the thinker gave it)
    and `report` (an explore's worker report, unverified by construction: nothing grades an explore), and `paths` (a
    landed explore's own changed paths, read in the workspace when it lands). `fire` names the registry nodes the move
    fires (`graph.FIRES`: a task or an explore the worker, an ask or ideas the console, a rest the rest), as decoded. A
    record from before `brief` or `paths` loads with none, and one from before `fire` with the firing its kind
    derives."""

    id: str
    kind: str
    line: str = ""
    cost_usd: float = 0.0
    outcome: str = ""
    branch: str | None = None
    units: list[str] = field(default_factory=list)
    question: str | None = None
    ideas: list[Idea] = field(default_factory=list)
    report: str | None = None
    brief: str = ""
    paths: list[str] = field(default_factory=list)
    fire: list[str] = field(default_factory=list)


@dataclass
class Firing:
    """One firing a tick ran: a thinker-fired node (`worker`, `console` or `rest`) with the move it came from and, for a
    task unit, the unit. An explore is a worker firing with no unit."""

    node: str
    move: str | None = None
    unit: str | None = None


@dataclass
class Fired:
    """One tick's firing set: `tick` as the tick's `Call` records carry it (the ticks counted before it), the firings
    it drew (a worker's from the queue; a console or rest firing from its own thinker call) and `walked`, the registry
    nodes it ran, in order, up to where it ended (`block` twice when it reaches the end). The loop rewrites it at each
    tick's start and every save in the tick persists it, so `task.json` keeps only the latest; the log keeps one
    `fired` event per tick."""

    tick: int = 0
    fire: list[Firing] = field(default_factory=list)
    walked: list[str] = field(default_factory=list)
    # The walk's own id (`loop.run_tick`), unique within the session; `tick` stays the ticks counted before it, so a walk
    # cut before its save and the walk that runs that tick again share `tick` and differ here. None on an older record.
    walk: str | None = None


@dataclass
class Request:
    """One accepted operation a client named with a `request_id` (`think` or `answer`), kept in `task.json` by the same
    save that landed it: the id, the operation, its payload as received (think `text` and `decide`; answer `id` and
    `text`) and the event it was answered with. The same id with the same operation and payload is answered with
    `event` again and changes nothing; with any other, it is refused. A refused or failed operation is never kept."""

    request_id: str
    op: str
    payload: dict[str, Any]
    event: dict[str, Any]


AUTO, HOLD = "auto", "hold"
MODES = (AUTO, HOLD)
PLAN, RETRY, EXPLORE = "plan", "retry", "explore"


@dataclass
class Held:
    """A progression waiting for `approve` in `hold` mode: a `task` move's `plan` (its `units`, which enter the session
    only when released), a unit's `retry` (`unit`, and the `attempt` it would be) or an `explore` (its `brief`). `move`
    names the move a plan or an explore came from. `directions` is how many thoughts the session had when it was held:
    one after that returns it to the thinker. `approved` is set by `approve`; the loop releases it at the tick
    boundary. `admitted` marks a plan whose branches already passed the gate inside its thinker call (the call started
    in `auto`), so its release makes no second reader call. A record from before `move` and `brief` loads with
    none."""

    kind: str
    units: list[Unit] = field(default_factory=list)
    unit: str | None = None
    attempt: int = 0
    directions: int = 0
    approved: bool = False
    admitted: bool = False
    move: str | None = None
    brief: str = ""


@dataclass
class Task:
    """A session (a task, in goal mode): the operator's thought and its units. `terminal` is None while running, else done,
    stopped or interrupted. `accepted` is the tip of `task/<id>` in the workspace, None until the first unit runs.
    `calls` is every model call's record, oldest first. `questions`, `thoughts`, `paused`, `mode` (auto or hold) and
    `held` (the progression waiting for `approve`, or None) are the console's state.

    The thought record lives here, so one save writes a move, its booking and the record
    together: `thought` is the operator's seed, verbatim, never edited; `thoughts` every contribution since, in order;
    `standing` (at most five plain sentences) and `open` are what the thinker's last accepted move said; `decided`
    is the operator's only and `moves` the brain's only. `awaiting_thought` is true after an `ideas` or a `rest` until a thought
    lands. `active_seconds` is the wall time the loop has spent driving ticks, each tick's added at its save; it accrues
    the dollar allowance. `fired` is the latest tick's firing set (`Fired`), None before the first tick and on a record
    from before the field. `requests` is the ledger of operations a client named with a `request_id` (`Request`), oldest
    first; a record from before it loads with none. `thought.md` beside `task.json` renders the thought record on every
    save (`render_thought`).

    `goal` and `directions` are goal mode's names for `thought` and `thoughts`: the constructor takes them, they read
    and write the same values, an older record loads them, and nothing writes them (as `manager_session` became
    `director_session`)."""

    id: str
    thought: str = ""
    workspace: str = ""
    project: str = "_root"
    tick: int = 0
    units: list[Unit] = field(default_factory=list)
    cost_usd: float = 0.0
    terminal: str | None = None
    stop_reason: str = ""
    director_session: str | None = None
    director_called_at: str | None = None  # when the last thinker call returned (UTC, seconds): the stretch clock
    created: str = ""
    extra_usd: float = 0.0
    extra_ticks: int = 0
    accepted: str | None = None
    calls: list[Call] = field(default_factory=list)
    questions: list[Question] = field(default_factory=list)
    paused: bool = False
    mode: str = AUTO
    held: Held | None = None
    thoughts: list[Thought] = field(default_factory=list)
    standing: str = ""
    open: list[OpenItem] = field(default_factory=list)
    decided: list[Decision] = field(default_factory=list)
    moves: list[Move] = field(default_factory=list)
    awaiting_thought: bool = False
    active_seconds: float = 0.0
    fired: Fired | None = None
    requests: list[Request] = field(default_factory=list)
    _: KW_ONLY
    goal: InitVar[str | None] = None
    directions: InitVar[list[Thought] | None] = None

    def __post_init__(self, goal: str | None, directions: list[Thought] | None) -> None:
        if goal is not None:
            self.thought = goal
        if directions is not None:
            self.thoughts = directions


def _alias(name: str) -> property:
    return property(lambda task: getattr(task, name), lambda task, value: setattr(task, name, value),
                    doc=f"goal mode's name for `{name}`; reads and writes it")


# After the class, so the generated constructor keeps its None defaults for the two names.
Task.goal = _alias("thought")  # type: ignore[assignment]
Task.directions = _alias("thoughts")  # type: ignore[assignment]


def is_closed(task: Task) -> bool:
    """The record admits no operation and no resume: the operator closed the session (`close`, 2026-10-20), abandoned it
    (`resume --abandon`), or it is a goal-mode record that ended `done` (runs 1–6)."""
    return task.terminal == "done" or task.stop_reason in ("abandoned", "closed")


def open_questions(task: Task) -> list[Question]:
    """Every question still waiting for the operator, oldest first."""
    return [q for q in task.questions if q.status == "open"]


def find_question(task: Task, question_id: str) -> Question | None:
    return next((q for q in task.questions if q.id == question_id), None)


def add_question(task: Task, asker: str, reason: str, text: str, context: dict[str, Any] | None = None) -> Question:
    """A new open question with the next id; an existing item is never replaced or reworded."""
    question = Question(id=f"q{len(task.questions) + 1}", asker=asker, text=text, reason=reason,
                        created=_utc_now().isoformat(timespec="seconds"), context=dict(context or {}))
    task.questions.append(question)
    return question


def to_json(task: Task) -> str:
    """The task as indented JSON text."""
    return json.dumps(asdict(task), indent=2) + "\n"


def from_json(text: str) -> Task:
    """Rebuild a task, its units, their predicates, call records, questions, thoughts and the thought record from
    `to_json` output. A record saved before a field existed loads with that field's default (a move's `fire`: the
    firing its kind derives); the file is never versioned. Goal mode's `goal` and `directions` load as `thought` and `thoughts`. The pre-console `question` text
    is dropped (questions live in `questions` now), and so are the two money fields a record of 2026-10-17 carries
    (the allowance replaced them)."""
    data = json.loads(text)
    if "manager_session" in data:  # Saved tasks from before the role rename.
        data.setdefault("director_session", data.pop("manager_session"))
    if "goal" in data:  # Saved tasks from before the thinking sessions.
        data.setdefault("thought", data.pop("goal"))
    if "directions" in data:
        data.setdefault("thoughts", data.pop("directions"))
    data.pop("question", None)
    data.pop("sitting_start_usd", None)
    data.pop("sitting_extra_usd", None)
    units = [_unit(u) for u in data.pop("units", [])]
    calls = [Call(**c) for c in data.pop("calls", [])]
    questions = [Question(**q) for q in data.pop("questions", [])]
    thoughts = [Thought(**t) for t in data.pop("thoughts", [])]
    open_items = [OpenItem(**o) for o in data.pop("open", [])]
    decided = [Decision(**d) for d in data.pop("decided", [])]
    moves = [Move(**{**m, "ideas": [Idea(**i) for i in m.get("ideas", [])],
                     "fire": m["fire"] if "fire" in m else graph.fires(m["kind"])}) for m in data.pop("moves", [])]
    held = data.pop("held", None)
    held = None if held is None else Held(**{**held, "units": [_unit(u) for u in held.get("units", [])]})
    fired = data.pop("fired", None)
    fired = None if fired is None else Fired(**{**fired, "fire": [Firing(**f) for f in fired.get("fire", [])]})
    requests = [Request(**r) for r in data.pop("requests", [])]
    return Task(**data, units=units, calls=calls, questions=questions, thoughts=thoughts, open=open_items,
                decided=decided, moves=moves, held=held, fired=fired, requests=requests)


def _unit(data: dict[str, Any]) -> Unit:
    return Unit(**{**data, "predicates": [Predicate(**p) for p in data["predicates"]]})


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def new_task_id(now: datetime | None = None) -> str:
    """`task-YYYYMMDDTHHMMSS`, from `now` or the current UTC time."""
    return "task-" + (now or _utc_now()).strftime("%Y%m%dT%H%M%S")


@dataclass
class Root:
    """The brain root: tracked prompts and config, plus generated task state."""

    path: Path

    def __post_init__(self) -> None:
        self.path = Path(self.path)

    def state_dir(self, task_id: str) -> Path:
        return self.path / "state" / task_id

    def task_file(self, task_id: str) -> Path:
        return self.state_dir(task_id) / "task.json"

    def log_file(self, task_id: str) -> Path:
        return self.state_dir(task_id) / "log.jsonl"

    def seat_dir(self, task_id: str) -> Path:
        return self.state_dir(task_id) / "seat"

    def clones_dir(self, task_id: str) -> Path:
        return self.state_dir(task_id) / "clones"

    def transcript_file(self, task_id: str) -> Path:
        """The editor's live transcript: one `transcript` event per stdout line of each editor call. A console field
        only, kept out of `task.json` and the log, so nothing in it reaches the director or a verdict."""
        return self.state_dir(task_id) / "transcript.jsonl"

    def thought_file(self, task_id: str) -> Path:
        """`thought.md`: the thought record as the operator reads it, rendered from `task.json` on every save; a view, never read
        back (it can be ahead of `task.json` after a save that failed at its commit: `save`)."""
        return self.state_dir(task_id) / "thought.md"

    def console_file(self, task_id: str) -> Path:
        """Port, token and pid of the live brain's console; mode 0600, gone when no brain serves the task."""
        return self.state_dir(task_id) / "console.json"

    def memory_file(self, project: str) -> Path:
        """The brain's long-term memory of one project (`graph/memory.py`), one JSON line per remembered event. The
        name keeps `[A-Za-z0-9._-]` and turns every other character into `_`, and a name of dots alone is `_`, so a
        project never names a path outside `memory/`; the directory is created on write."""
        safe = re.sub(r"[^A-Za-z0-9._-]", "_", project)
        if not safe.strip("."):
            safe = "_"
        return self.path / "memory" / f"{safe}.jsonl"

    @property
    def current_file(self) -> Path:
        return self.path / "state" / "current"

    @property
    def lobby_file(self) -> Path:
        """Host, port, token and pid of the lobby (`graph/lobby.py`); mode 0600, gone when no lobby serves."""
        return self.path / "state" / "lobby.json"

    @property
    def lobby_log_dir(self) -> Path:
        """Where the lobby writes each spawned brain's stdout and stderr, one `launch-<UTC>.log` per spawn."""
        return self.path / "state" / "lobby"

    @property
    def config_file(self) -> Path:
        return self.path / "config.json"

    @property
    def thinker_prompt(self) -> Path:
        """The director seat's prompt in its thinker role."""
        return self.path / "thinker.md"

    @property
    def editor_prompt(self) -> Path:
        return self.path / "editor.md"

    @property
    def reader_prompt(self) -> Path:
        return self.path / "reader.md"


def _write_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


LOCK = threading.RLock()  # every save and log append; the console thread takes it around each operation


NONE = "(none)"


def _one_line(text: str) -> str:
    return " ".join(text.split())


def _move_line(move: Move) -> str:
    """One move on one line: id, kind, its line, cost and outcome, then each pointer it carries."""
    parts = [move.id, move.kind, _one_line(move.line), f"${move.cost_usd:.2f}", _one_line(move.outcome)]
    if move.branch:
        parts.append(f"branch {move.branch}")
    if move.units:
        parts.append("units " + ", ".join(move.units))
    if move.question:
        parts.append(f"question {move.question}")
    if move.ideas:
        parts.append("ideas " + ", ".join(idea.id for idea in move.ideas))
    if move.report is not None:
        first = next((line for line in move.report.splitlines() if line.strip()), "")
        parts.append(f"report (unverified): {_one_line(first)}".rstrip())
    return "- " + " · ".join(part for part in parts if part)


def render_thought(task: Task) -> str:
    """`thought.md` for `task`, pure: the thought verbatim, then standing, open, decided (the operator's text verbatim, its
    later lines indented under the item) and the moves, one line each. Every section is there, `(none)` when empty."""
    lines = ["# Thought", "", task.thought, "", "## Standing", "", task.standing or NONE, "", "## Open", ""]
    lines += [f"- {item.id}: {_one_line(item.line)}" for item in task.open] or [NONE]
    lines += ["", "## Decided", ""]
    lines += [f"- think {d.think}, {d.created}: " + d.text.replace("\n", "\n  ") for d in task.decided] or [NONE]
    lines += ["", "## Moves", ""]
    lines += [_move_line(move) for move in task.moves] or [NONE]
    return "\n".join(lines) + "\n"


def save(root: Root, task: Task) -> None:
    """Write the view `thought.md`, then `task.json`, replacing the previous copy in one rename: that rename is the
    commit point, so a save that raises has committed nothing (a failed view write never leaves the record saved).
    When the view was written and `task.json` was not, `thought.md` is ahead of the record until the next save; it is
    never read back, so nothing loads it as state."""
    with LOCK:
        _write_atomic(root.thought_file(task.id), render_thought(task))
        _write_atomic(root.task_file(task.id), to_json(task))


def load(root: Root, task_id: str) -> Task:
    """Read one task. A missing task file raises `FileNotFoundError`."""
    return from_json(root.task_file(task_id).read_text(encoding="utf-8"))


def current(root: Root) -> Task | None:
    """The task `state/current` names, or None when no task has been started."""
    try:
        task_id = root.current_file.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        return None
    return load(root, task_id) if task_id else None


def set_current(root: Root, task_id: str) -> None:
    """Point `state/current` at a task id."""
    _write_atomic(root.current_file, task_id + "\n")


def _append(path: Path, event: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps({"ts": _utc_now().isoformat(timespec="seconds"), **event})
    with LOCK, path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def append_log(root: Root, task_id: str, event: dict[str, Any]) -> None:
    """Append one event as a JSON line, stamped with a UTC `ts`."""
    _append(root.log_file(task_id), event)


def append_transcript(root: Root, task_id: str, event: dict[str, Any]) -> None:
    """Append one transcript event to `transcript.jsonl`, stamped like a log line; never to the log."""
    _append(root.transcript_file(task_id), event)


def read_log(root: Root, task_id: str) -> list[dict[str, Any]]:
    """Every event in the task's log, oldest first; an absent log is empty."""
    try:
        lines = root.log_file(task_id).read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        return []
    return [json.loads(line) for line in lines if line.strip()]


def pending_unit(task: Task) -> Unit | None:
    """The first pending unit whose every need has passed and been integrated, or None."""
    ready = {unit.id for unit in task.units if unit.status == "passed" and unit.integrated}
    return next((unit for unit in task.units if unit.status == "pending" and set(unit.needs) <= ready), None)


def block_dependents(task: Task) -> list[tuple[Unit, Unit]]:
    """Mark `blocked` every pending unit that needs a failed, descoped or blocked unit, transitively.
    Returns each newly blocked unit with the need that blocked it."""
    by_id = {unit.id: unit for unit in task.units}
    blocked: list[tuple[Unit, Unit]] = []
    changed = True
    while changed:
        changed = False
        for unit in task.units:
            dead = [by_id[n] for n in unit.needs if n in by_id and by_id[n].status in ("failed", "descoped", "blocked")]
            if unit.status == "pending" and dead:
                unit.status, unit.summary = "blocked", f"blocked: needs {dead[0].id} ({dead[0].status})"
                blocked.append((unit, dead[0]))
                changed = True
    return blocked
