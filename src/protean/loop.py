"""The loop: the ring, a walker over the node registry (`graph/graph.py`). Each tick asks the thinker for one move,
runs one runnable unit, or runs one explore; only the grader passes a unit.

A tick is a walk: the entries in every tick (repair, block, budget, release, pick, hold, block again, save), the tick's
firing and the nodes it brings, in the one order the registry's edges allow (`_walk`); each node's act (`ACTS`) may
end the tick, and then nothing after it runs, save included. `pick` draws the tick's firing from the queue, one worker
firing per tick (the runnable unit, else the queued explore), or with neither the runtime fires the thinker
(`director`). The thinker's move fires one node (`Move.fire`): a `task` or an `explore` fires the worker, which waits
in the queue for a later tick; an `ask` or `ideas` fires the console and a `rest` fires rest, both in the thinker's
own tick, after it, with what they bring (memory's line after a rest). A worker unit brings the grader, the gate and the
ref move, entries of the one tick that share its pass (`Tick`). The tick's firing set and the order it walked are the
record's `fired` and one `fired` log event per tick.

The thinker is the director seat in its thinking-session role (`graph/director.py`): one call, one move plus `state`.
`ideas` and `rest` leave the session awaiting a thought (a hold at the tick boundary, like a question, until a console
`think` lands); `ask` is the seat's question; `task` adds units exactly as a plan did; `explore` is one worker call in its own
clone on `explore/<session>/<n>`, never graded, retried or integrated. Every accepted move rewrites the record's
`standing` and `open`; a reply whose standing is unchanged is not acted on (returned once, then the runtime asks).

A delivered attempt is graded only when the worker's own closing line is `exit_code: 0`; a nonzero or
missing line makes the attempt ungradeable before any predicate is read. Zero permits grading, never
substitutes for it.

A pass is two writes: the unit is saved `passed` with its candidate commit, then `task/<task>` moves to
that commit and the unit is saved integrated. A tick or resume first finishes any pass cut between them.

While a task is driven, its console serves (`graph/console.py`). Between ticks, at the boundary, the loop acts on
what the console landed in state and holds while a question is open, the task is paused, a ceiling stopped it, or,
in `hold` mode, a `task` move's plan, a retry or an explore waits for `approve` (a `think` meanwhile returns it to the
thinker; the accepted-ref move and the thinker's own call are never held), or the session awaits a thought. A hold is
a wait, never a tick. Unattended, a hold other than a pause parks after the timeout and `protean resume` continues.
Runtime questions are the floating node's (`graph/questions.py`).

The session's dollar ceiling is its allowance (`budget.allowance`):
`max_usd`, plus every `extend`, plus `accrual_usd` for every `accrual_seconds` of active work. Active work is the wall
time a tick runs, read on `clock` and added to `active_seconds` at the tick's save (a tick cut before it forfeits its
time); waits at the tick boundary and parks happen outside a tick and add nothing. Spend that reaches the allowance
stops the session on its ceiling in the save that books the call; every call is clipped to what the allowance has left
(`budget.call_caps`).

A plan that proposes a predicate branch passes the gate (`graph/gate.py`) inside the tick that dispatches it: one
fresh reader call per branching unit, decided once per plan by the mode its thinker call started in: from `auto` within the call's retry loop, from `hold` when the held plan is released. A held plan is
released after the ceilings (the budget node's edge), and a `think` that lands during its admission call still
returns it. On the
branching unit's own candidate, once every child passes, the parents of those children are excused from the re-check
(recorded `excused`, never a `broke`); at its integration they leave the re-check set.

A candidate passes the gate again before it integrates, free checks first (ruled 2026-10-11): its own grade, the
re-checks, the clean clone; only then the weakening check (`gate.weakening`), one reader call when the candidate
touched a protected path of an accepted command, none otherwise. Weakened, or no verdict, rejects the attempt. All of
it runs in the grading clone (`worker.grading_clone`), a fresh clone of the workspace at the candidate, never in the
editor's own clone, whose `.git` the editor wrote.
"""

from __future__ import annotations

import fcntl
import os
import re
import shutil
import time
import uuid
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from protean import state
from protean.graph import budget, console, director, gate, grade, graph, memory, questions, worker
from protean.graph.budget import Caps
from protean.graph.console import Console, Notify
from protean.runner import Charge, Runner, floor_reason
from protean.state import (AUTO, DIRECTOR, EXPLORE, HOLD, MODES, MOVE_KINDS, PLAN, RETRY, RUNTIME, Call, Fired, Firing,
                           Held, Move, Question, Root, Task, Unit)

Echo = Callable[[str], None]
NOTE_LIMIT = 200
LINE_LIMIT = 160  # a move's one line
NO_PROGRESS_NOTE = "no progress: standing unchanged"  # the first time
SESSION_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]*")  # a session id `protean resume <session>` may name
# The scratch workspace's one empty commit: the brain's own, so no user hook or signing applies to it.
SCRATCH_GIT = ["-c", "user.name=protean", "-c", "user.email=protean@protean.invalid", "-c", "commit.gpgsign=false"]
clock: Callable[[], float] = time.monotonic  # the wall clock a tick's active time is read on (the battery scripts it)


def walk_id() -> str:
    """A new walk's id (`Fired.walk`): opaque, random, unique within the session."""
    return uuid.uuid4().hex


def _silent(_: str) -> None:
    pass


class TaskRunning(RuntimeError):
    """Another process holds the task's lock: a brain is driving it."""


def _deliver(task: Task, question: Question) -> dict[str, Any]:
    """Act once on an answer the console landed, routed to its asker; returns the log event. The thinker's own
    question is answered in its next message's `answer` (`director.last_answer`), once, never again as a note, and its
    `ask` move records it; a runtime answer goes to the runtime-questions node (`questions.consume`), which acts on it,
    and never reaches the thinker."""
    if question.asker == RUNTIME:
        return questions.consume(question)
    question.disposition = "delivered"
    for move in task.moves:
        if move.kind == "ask" and move.question == question.id:
            move.outcome = "answered"
    return {"event": "delivered", "id": question.id, "disposition": question.disposition}


def _log_calls(root: Root, task_id: str, calls: list[Call]) -> None:
    """One `call` event per record the task has just booked: the receipt's per-call history."""
    for call in calls:
        state.append_log(root, task_id, {"event": "call", **asdict(call)})


def _book(task: Task, root: Root, booked: list[Call], caps: Caps) -> None:
    """Save and log the calls just booked. Spend that reaches the allowance stops a running session on its ceiling in
    this same save, the `stopped` event logged after the calls; spend that leaves it under
    the call floor stops it at the next call's ceiling step instead. A call that left no receipt (timed out, no receipt,
    interrupted or lost: its actual spend is UNKNOWN) holds paid work on a runtime question that only `yes` releases
    (ruled 2026-10-11: every receipt-less call, not only a kill); the question is saved with the booking, so no resume
    re-asks it."""
    with state.LOCK:
        reached = budget.allowance_reason(task, caps) if task.terminal is None else None
        if reached:
            task.terminal, task.stop_reason = "stopped", reached
        state.save(root, task)
        _log_calls(root, task.id, booked)
        if reached:
            state.append_log(root, task.id, {"event": "stopped", "reason": reached})
    unreceipted = [call for call in booked if call.actual_usd == budget.UNKNOWN]
    if unreceipted:
        questions.unknown_spend(root, task, unreceipted)


def _charger(task: Task, root: Root, seat: str, caps: Caps) -> Charge:
    """Book an interrupted call's spend on the task and save it first, so no dollar is lost to a Ctrl-C; the call left
    no receipt, so the same save holds paid work (`_book`)."""
    def charge(usd: float) -> None:
        task.cost_usd += usd
        _book(task, root, budget.mark_booked(task), caps)
        state.append_log(root, task.id, {"event": "interrupted", "seat": seat, "charged_usd": usd})
    return charge


def _thinker_step(tick: Tick) -> str:
    """One thinker call. Branch admission is decided once, by the mode the call starts in: from `auto` a decoded
    `task` move's branches pass the gate inside the call's retry loop (this tick), the registry's gate node handed to
    the call as its admit hook (its `within` edge, `graph.nested`); from `hold` they pass it when the held plan is
    released. A mode switch during the call neither skips the reader nor pays it twice. A move whose firing runs now (a
    `console` or `rest` firing: not `queued`) extends this tick with that firing, whose node accepts the move
    (`_accept_fired`); a worker firing is accepted here and waits in the queue for a later tick. The tick's line, or ""
    when the firing's node finishes it."""
    task, caps, root, runner = tick.task, tick.caps, tick.root, tick.runner
    said: list[str] = []
    admitted = task.mode == AUTO and graph.nested("gate", "director")
    waited: list[bool] = []

    def admit(units: list[Unit], spent: float) -> list[str]:
        admission = gate.admit(task, units, budget.call_caps(task, caps), root, runner, spent=spent,
                               charge=_charger(task, root, "reader", caps))
        said.extend(admission.lines)
        waited.append(admission.waits)
        return admission.refused

    reply = director.next_move(task, caps, root, runner, _charger(task, root, "director", caps),
                               admit if admitted else None)
    if any(waited):  # a reader call left no receipt: the plan is held unadmitted, its release asks the reader again
        admitted = False
    task.cost_usd += reply.cost_usd
    booked = budget.mark_booked(task)
    task.director_session = reply.session_id or task.director_session
    _book(task, root, booked, caps)
    state.append_log(root, task.id, {"event": "director", "reply": reply.kind, "cost_usd": reply.cost_usd,
                                     "session_id": reply.session_id, "raw": reply.raw[-2000:]})
    tail = f" (${reply.cost_usd:.2f})" + "".join(f"; {s}" for s in said)
    if reply.kind in MOVE_KINDS and not graph.node(reply.move.fire[0]).queued:
        tick.accepting = (reply, admitted, tail)
        task.fired.fire.append(Firing(reply.move.fire[0], move=_move_id(task)))
        return ""
    return f"thinker {_apply(reply, task, caps, root, admitted=admitted)}" + tail


def _move_id(task: Task) -> str:
    """The id the session's next move takes."""
    return f"m{len(task.moves) + 1}"


def _accept_fired(tick: Tick) -> bool:
    """The node of this tick's console or rest firing accepts the thinker's move (`_accept`): an ask opens its
    question, an ideas move proposes its options, a rest leaves the session waiting for a thought. The move's line
    finishes the tick's."""
    reply, admitted, tail = tick.accepting
    said = _accept(reply.move, reply.cost_usd, tick.task, tick.root, admitted=admitted, seen=reply.thoughts)
    tick.line = f"thinker {said}{tail}"
    return False


def _apply(reply: director.Reply, task: Task, caps: Caps, root: Root, *, admitted: bool) -> str:
    """Act on one thinker call's outcome. A call that reached its dollar cap logs a `ceiling` event and holds the
    session on a runtime question; so does a reply invalid twice. A reply that left the standing unchanged is not acted
    on (`_no_progress`). Anything else is one move, accepted (`_accept`)."""
    if reply.ceiling is not None:
        return questions.cap_hit(root, task, reply.ceiling)
    if reply.unreceipted is not None:
        return f"call left no receipt ({reply.unreceipted}); paid work held"
    if reply.stop:
        task.terminal, task.stop_reason = "stopped", reply.stop
        note = "Your previous answer was rejected: " + "; ".join(reply.errors)
        state.append_log(root, task.id, {"event": "stopped", "reason": reply.stop, "to_director": note})
        return f"answer rejected, no retry: {reply.stop}"
    if reply.question is not None or reply.move is None:
        return questions.invalid_plan(root, task, reply.question or "The thinker returned no move.")
    if reply.unchanged:
        return _no_progress(task, root, reply.move)
    return _accept(reply.move, reply.cost_usd, task, root, admitted=admitted, seen=reply.thoughts)


def _no_progress(task: Task, root: Root, reply: director.MoveReply) -> str:
    """a reply whose standing is the record's is not acted on. The first time it goes back to
    the thinker as a note in its next message (one more call, next tick); the second in a row opens the runtime's
    `no_progress` question, and nothing paid runs until the operator answers it."""
    event = {"event": "no_progress", "move": reply.kind}
    if questions.stalled(root, task.id):
        state.append_log(root, task.id, event)
        return f"no progress again ({reply.kind}); " + questions.no_progress(root, task)
    state.append_log(root, task.id, {**event, "to_director": NO_PROGRESS_NOTE})
    return f"no progress ({reply.kind}): returned to the thinker"


def _move_line(reply: director.MoveReply) -> str:
    """The move in one line, as the record and the thinker's next message show it."""
    if reply.kind == "ideas":
        text = "ideas " + ", ".join(idea.id for idea in reply.ideas)
    elif reply.kind == "task":
        text = "task " + ", ".join(unit.id for unit in reply.units)
    else:
        text = {"explore": reply.brief, "ask": reply.question, "rest": "rest"}[reply.kind]
    text = " ".join(text.split())
    return text if len(text) <= LINE_LIMIT else text[:LINE_LIMIT - 1] + "…"


def _accept(reply: director.MoveReply, cost: float, task: Task, root: Root, *, admitted: bool,
            seen: int | None = None) -> str:
    """Record one move, with its firing set (`fire`), and act on it; its `state` rewrites the record's `standing` and
    `open` (the brain alone writes `moves`). `ideas` and `rest` leave the session awaiting a thought, unless a think
    landed during the call (more thoughts than the `seen` its message carried): the next call carries it. `ask` opens
    the thinker's question; a `task` move's units start pending, or its plan waits for `approve` in `hold` (and for its
    admission when it branches and its call did not admit it: the call started in `hold`); an `explore` is queued for
    the next tick, held for `approve` in `hold`. Saved, then logged: the kind's own event, then `thought`, the record's
    change. The thinker node calls it for a worker firing, which is queued; the console and rest nodes call it for
    theirs, in the same tick (`_accept_fired`); a rest's memory line is the memory node's, after it (`_remember`)."""
    move = Move(id=_move_id(task), kind=reply.kind, line=_move_line(reply), cost_usd=cost, fire=list(reply.fire))
    task.moves.append(move)
    task.standing, task.open = reply.standing, list(reply.open)
    events: list[dict[str, Any]] = []
    if reply.kind in ("ideas", "rest"):
        if reply.kind == "ideas":
            move.ideas, move.outcome = list(reply.ideas), "proposed"
            events.append({"event": "ideas", "move": move.id, "ideas": [asdict(idea) for idea in reply.ideas]})
            said = f"ideas ({', '.join(idea.id for idea in reply.ideas)})"
        else:
            move.outcome = "rested"
            events.append({"event": "rest", "move": move.id})
            said = "rest"
    elif reply.kind == "ask":
        said = questions.ask(root, task, DIRECTOR, "question", reply.question)
        move.question, move.outcome = task.questions[-1].id, "asked"
    elif reply.kind == "explore":
        move.brief, move.outcome = reply.brief, "pending"
        if task.mode == HOLD:
            said = _hold_for_approve(task, Held(kind=EXPLORE, move=move.id, brief=reply.brief), root)
        else:
            said = f"explore {move.id} queued"
    else:
        move.units, move.branch = [unit.id for unit in reply.units], f"task/{task.id}"
        if task.mode == HOLD or (not admitted and gate.branching(reply.units)):
            move.outcome = "held"
            said = _hold_for_approve(task, Held(kind=PLAN, units=reply.units, admitted=admitted, move=move.id), root)
        else:
            task.units.extend(reply.units)
            move.outcome = "added"
            said = f"added {', '.join(u.id for u in reply.units)}"
    with state.LOCK:
        if reply.kind in ("ideas", "rest"):  # under the lock a console think lands under, so none slips between
            woke = seen is not None and len(task.thoughts) > seen
            task.awaiting_thought = not woke
            said += "; a thought landed during the call: the thinker moves again" if woke else "; awaiting a thought"
        state.save(root, task)
        for event in events + [{"event": "thought", "move": move.id, "kind": move.kind, "standing": task.standing,
                                "open": [asdict(item) for item in task.open]}]:
            state.append_log(root, task.id, event)
    return f"{move.id} {said}"


def _remember(task: Task, root: Root) -> None:
    """The memory node after a rest: the brain's long-term memory of the project gains one line for the record as the
    rest left it, and its `memory` event follows the rest's (none for a scratch session)."""
    with state.LOCK:
        remembered = memory.logged(root, task, "rest")
        if remembered is not None:
            state.append_log(root, task.id, remembered)


def _hold_for_approve(task: Task, held: Held, root: Root) -> str:
    """`hold` mode: a `task` move's plan (its units kept outside the session), a unit's retry or an explore waits for
    `approve`. Saved before it is logged, so `approve` finds it on disk. It records how many thoughts the session had
    (`Held.directions`, on the wire `thoughts`): one landing after it returns the progression to the thinker."""
    with state.LOCK:
        held.directions = len(task.thoughts)
        task.held = held
        state.save(root, task)
        view = console.held_view(held)
        state.append_log(root, task.id, {"event": "held", **{k: v for k, v in view.items() if k != "approved"}})
    return f"held {console.held_label(held)} for approve"


def _release_plan(task: Task, caps: Caps, root: Root, runner: Runner, echo: Echo) -> None:
    """A held plan released by `approve` (or by `auto`) passes the gate first, in this tick, unless its thinker call
    already admitted it (`admitted`): admitted, its units enter the task; refused, it is dropped, none of its units
    enter, and the refusal line reaches the thinker's next message; stopped at the call floor between two admission
    calls, it stays held (approved) and the next tick's ceiling stops the task; held on `reader_cap`, it stays held
    until the answer, which this release then acts on (`questions.owed`: `yes` re-asks, `no` refuses). The plan stays held (approved) while
    the reader runs, so a kill meanwhile releases it again on resume, and a `think` meanwhile returns it
    (`_release`)."""
    if not task.held.admitted:
        answered = questions.owed(task, "admission")
        admission = gate.admit(task, task.held.units, budget.call_caps(task, caps), root, runner,
                               charge=_charger(task, root, "reader", caps),
                               to_director=True, answered=answered)
        for line in admission.lines:
            echo(line)
        if admission.waits:
            return
        if admission.refused:
            with state.LOCK:
                _mark(task, task.held.move, "refused")
                task.held = None
                state.save(root, task)
            return
    _release(task, root, echo)


def _mark(task: Task, move_id: str | None, outcome: str) -> None:
    """Set what became of the move a held progression came from. A plan returned or refused never entered the
    session, so its move keeps its line and points at no unit and no branch (a later unit may reuse the ids)."""
    for move in task.moves:
        if move.id == move_id:
            move.outcome = outcome
            if outcome in ("returned", "refused"):
                move.units, move.branch = [], None


def _awaits_approve(task: Task) -> bool:
    """Something is held in `hold` mode and `approve` has not released it."""
    return task.held is not None and not task.held.approved and task.mode == HOLD


def _release(task: Task, root: Root, echo: Echo) -> bool:
    """Let the held progression go: approved, or the session is back in `auto`. A plan's units enter the session; a
    retry or an explore runs in this tick (True). A `think` that landed since it was held wins over the `approve`
   : checked under the lock the units would enter under, it returns the progression to the thinker
    instead (False)."""
    held = task.held
    with state.LOCK:
        if len(task.thoughts) > held.directions:
            event = _returned(task, held)
        else:
            if held.kind == PLAN:
                task.units.extend(held.units)
                _mark(task, held.move, "added")
            task.held = None
            event = {"event": "approved", **console.held_event(held), "by": "approve" if held.approved else "mode"}
        state.save(root, task)
        state.append_log(root, task.id, event)
    if event["event"] == "returned":
        echo(f"returned: the held {event['kind']} went back to the thinker with {event['thoughts']} thought(s)")
        return False
    echo(f"approved: {console.held_label(held)}" + ("" if held.approved else " (mode auto)"))
    return True


def _retry_admitted(task: Task, unit: Unit, root: Root, echo: Echo) -> bool:
    """In `hold`, a unit's retry (attempt 2 or later) waits for `approve`: the first time it is picked it is held,
    and the tick ends without a call. True when it may run now: approved, or the task is in `auto`."""
    held, attempt = task.held, unit.attempts + 1
    if held is not None and held.kind == RETRY and held.unit == unit.id and held.attempt == attempt:
        if held.approved or task.mode == AUTO:
            return _release(task, root, echo)
        return False
    if task.mode == AUTO:
        return True
    _hold_for_approve(task, Held(kind=RETRY, unit=unit.id, attempt=attempt), root)
    return False


def _explore_admitted(task: Task, move: Move, root: Root, echo: Echo) -> bool:
    """In `hold`, an explore waits for `approve` before its worker call: held when it is
    accepted, or the first time it is picked if the session went to `hold` since, and the tick ends without a call.
    True when it may run now: approved, or the session is in `auto`."""
    held = task.held
    if held is not None and held.kind == EXPLORE and held.move == move.id:
        if held.approved or task.mode == AUTO:
            return _release(task, root, echo)
        return False
    if task.mode == AUTO:
        return True
    _hold_for_approve(task, Held(kind=EXPLORE, move=move.id, brief=move.brief), root)
    return False


def _returned(task: Task, held: Held) -> dict[str, Any]:
    """A `think` landed while `held` waited for approve: it goes back to the thinker with every thought, and the
    thinker moves again. A held plan is dropped, its units never entering the session; a held explore never runs; a
    held retry does not run and its unit is failed as it stands. Either way the thinker's next message carries a
    note."""
    task.held = None
    if held.kind == PLAN:
        _mark(task, held.move, "returned")
        _drop_reader_caps(task, held)
        note = (f"The operator added a thought while your plan ({', '.join(u.id for u in held.units)}) waited for their "
                "approval: it was returned with their thoughts and none of its units entered the session. Move again "
                "under the thoughts.")
    elif held.kind == EXPLORE:
        _mark(task, held.move, "returned")
        note = (f"The operator added a thought while your explore ({held.move}) waited for their approval: it was returned with "
                "their thoughts and did not run. Move again under the thoughts.")
    else:
        for unit in task.units:
            if unit.id == held.unit and unit.status == "pending":
                unit.status = "failed"
        note = (f"The operator added a thought while {held.unit}'s retry (attempt {held.attempt}) waited for their approval: the "
                f"retry was returned with their thoughts and did not run, so {held.unit} is failed as it stands. Give "
                "its work again in a new task, smaller, or explore what stopped it.")
    return {"event": "returned", **console.held_event(held), "thoughts": len(task.thoughts), "to_director": note}


def _drop_reader_caps(task: Task, held: Held) -> None:
    """A returned plan takes its `reader_cap` admission questions with it: an open one is superseded (`returned`), an
    answered one is never acted on."""
    ids = {unit.id for unit in held.units}
    for question in task.questions:
        info = question.context.get("reader_cap") if question.reason == "reader_cap" else None
        if info and info["check"] == "admission" and info["unit"] in ids and not question.context.get("settled"):
            question.context["settled"] = True
            if question.status == "open":
                question.status, question.disposition = "superseded", "returned"


def _rechecks(task: Task, clone: Path, root: Root,
              excused: dict[str, list[str]]) -> tuple[dict[str, str], list[dict[str, Any]]]:
    """Every integrated unit's re-check set (`grade.rechecked`: its retained predicates that read the tree) graded
    again in this clone: `<unit> <pid>` -> note for each one that no longer holds, and every result in order (an empty
    list when nothing is accepted). A delivery-only predicate is never graded here, so it can never break. A parent in
    `excused` (`gate.excused`: the candidate's own children all passed) is not graded and is recorded `excused` with
    the children that carry it, so it can never break either."""
    broke, checked = {}, []
    for other in task.units:
        if other.integrated:
            pids = grade.predicate_ids(other)
            skip = {pid for pid, p in zip(pids, other.predicates)
                    if grade.rechecked(p) and f"{other.id} {pid}" in excused}
            verdict = grade.grade(other, clone, None, recheck=True, console_file=root.console_file(task.id), skip=skip)
            results = {pid: result for result, ids in (("passed", verdict.passed_ids), ("failed", verdict.failed_ids),
                                                       ("ungradeable", verdict.ungradeable_ids)) for pid in ids}
            checked += [{"unit": other.id, "id": pid, "result": "excused", "children": excused[f"{other.id} {pid}"]}
                        if pid in skip else {"unit": other.id, "id": pid, "result": results[pid]}
                        for pid in pids if pid in results or pid in skip]
            for pid in verdict.failed_ids + verdict.ungradeable_ids:
                broke[f"{other.id} {pid}"] = verdict.notes[pid]
    return broke, checked


def _integrate(task: Task, unit: Unit, root: Root) -> bool:
    """The second write of a pass: move `task/<task>` from the accepted tip to the unit's candidate, then
    record it. `update-ref` checks the old value; a ref already at the candidate is only recorded. A ref
    moved anywhere else outside the loop holds the task on a runtime question (once) and returns False."""
    workspace, ref = Path(task.workspace), worker.accepted_ref(task)
    try:
        found = worker.git(["rev-parse", "--verify", "--quiet", ref], workspace).strip()
    except worker.WorkerError:
        found = "missing"
    if found not in (unit.candidate, task.accepted):
        questions.stale_tip(root, task, unit, found, ref)
        return False
    if found != unit.candidate:
        worker.git(["update-ref", ref, str(unit.candidate), str(task.accepted)], workspace)
    task.accepted, unit.integrated = unit.candidate, True
    branch = gate.accept(task, unit, root)  # its parents leave the re-check set in the same save
    state.save(root, task)
    state.append_log(root, task.id, {"event": "integrated", "unit": unit.id, "accepted": unit.candidate})
    if branch is not None:
        state.append_log(root, task.id, branch)
    return True


def _repair(task: Task, root: Root) -> bool:
    """Finish every pass a kill cut between its two writes; the unit's worker never runs again. Write once the memory
    line of a rest whose memory node a kill cut off, and log its `memory` event (`memory.owed`; the memory node stays
    the line's writer, this is the net under it). False when a tip moved outside the loop leaves a pass unfinished."""
    with state.LOCK:
        remembered = memory.owed(root, task)
        if remembered is not None:
            state.append_log(root, task.id, remembered)
    for unit in task.units:
        if unit.status == "passed" and unit.candidate and not unit.integrated and not _integrate(task, unit, root):
            return False
    return True


def _recover(task: Task, root: Root, hold: bool, caps: Caps) -> bool:
    """Book, once, every call a kill left unbooked (`budget.recover`). A call lost before any receipt holds
    paid work: with `hold`, a runtime question opens in the same save that books the call (`_book`). True when it did."""
    booked, lost = budget.recover(task)
    if booked and hold:
        _book(task, root, booked, caps)
    elif booked:
        state.save(root, task)
        _log_calls(root, task.id, booked)
    return bool(lost) and hold


_BLOCKED_BY = {"failed": "failed", "descoped": "was descoped", "blocked": "is blocked"}


def _block(task: Task, root: Root) -> None:
    """Block every pending unit whose need failed, was descoped (a goal-mode record) or is blocked, and tell the
    thinker."""
    for unit, need in state.block_dependents(task):
        note = (f"{unit.id} is blocked: it needs {need.id}, which {_BLOCKED_BY[need.status]}, so it never runs. "
                "Give a new unit in a later task if its work is still wanted.")
        state.append_log(root, task.id, {"event": "blocked", "unit": unit.id, "need": need.id, "to_director": note})


def _earlier_verdict(root: Root, task: Task, unit: Unit) -> dict[str, Any] | None:
    """The recorded `verdict` of this unit's latest earlier attempt (`log.jsonl`): the findings a retry's editor is
    shown. None on a first attempt, or when no earlier attempt left a verdict."""
    if unit.attempts < 2:
        return None
    earlier = [event for event in state.read_log(root, task.id) if event.get("event") == "verdict"
               and event.get("unit") == unit.id and event.get("attempt", 0) < unit.attempts]
    return earlier[-1] if earlier else None


def _work(tick: Tick) -> None:
    """The worker on the picked unit: one attempt in its clone, its spend booked on the session and on the task move
    the unit came from. What the pass's later nodes need (the result, the accepted tip it branched from) stays in the
    tick."""
    task, unit, caps, root = tick.task, tick.unit, tick.caps, tick.root
    unit.attempts += 1
    findings = worker.findings(task, unit, _earlier_verdict(root, task, unit))
    res = worker.run_unit(task, unit, budget.call_caps(task, caps), root, tick.runner,
                          _charger(task, root, "worker", caps), findings=findings)
    tick.tip = task.accepted  # the tip the unit branched from: nothing integrates between its clone and its grade
    task.cost_usd += res.cost_usd
    for move in task.moves:  # the task move the unit came from carries its workers' spend
        if move.kind == "task" and unit.id in move.units:
            move.cost_usd += res.cost_usd
    _book(task, root, budget.mark_booked(task), caps)
    tick.res = res


def _conclude(task: Task, unit: Unit, res: worker.WorkerResult, caps: Caps, root: Root, verdict: grade.Verdict,
              broke: dict[str, str], rechecked: list[dict[str, Any]] | None, weakening: gate.Weakening | None,
              graded: bool) -> str:
    """Decide one graded (or ungradeable) attempt: passed, or rejected (failed after the last), its `verdict` logged,
    saved, and a pass integrated; the tick's line."""
    passed = verdict.passed and not broke and not (weakening is not None and weakening.rejects)
    if passed:
        unit.status, unit.candidate = "passed", res.candidate
    elif unit.attempts >= caps.max_attempts_per_unit:
        unit.status = "failed"
    misses = [f"{pid} {verdict.notes[pid]}"[:NOTE_LIMIT] for pid in verdict.failed_ids + verdict.ungradeable_ids]
    misses += [f"broke {key}: {note}"[:NOTE_LIMIT] for key, note in broke.items()]
    if weakening is not None and weakening.result == gate.WEAKENED:
        misses.append(f"weakened {', '.join(weakening.touched)}: {weakening.kind}: {weakening.note}"[:NOTE_LIMIT])
    elif weakening is not None and weakening.result == gate.UNCHECKED:
        misses.append((weakening.note or "")[:NOTE_LIMIT])
    grade_line = "passed" if passed else "; ".join(misses)
    unit.branch, unit.summary = res.branch, f"{res.summary} [grade: {grade_line}]"
    shutil.rmtree(res.clone, ignore_errors=True)
    event = {
        "event": "verdict", "unit": unit.id, "attempt": unit.attempts, "status": unit.status,
        "passed": passed, "passed_ids": verdict.passed_ids, "failed_ids": verdict.failed_ids,
        "ungradeable_ids": verdict.ungradeable_ids, "notes": verdict.notes, "exit_code": res.exit_code,
        "cost_usd": res.cost_usd, "branch": res.branch, "changed": res.changed, "delivered": res.delivered,
        "commit": res.candidate if graded else None,
    }
    if rechecked is not None:
        event["recheck"] = {"candidate": res.candidate, "predicates": rechecked}
    if weakening is not None:
        event["weakening"] = weakening.record()
    if broke:
        event["broke"] = broke
        event["to_director"] = (f"{unit.id} broke accepted work and was not accepted (attempt {unit.attempts}): "
                                + "; ".join(misses))
    state.append_log(root, task.id, event)
    state.save(root, task)
    if passed:
        _integrate(task, unit, root)
    parts = [f"{label} {' '.join(ids)}" for label, ids in
             (("failed", verdict.failed_ids), ("ungradeable", verdict.ungradeable_ids), ("broke", list(broke))) if ids]
    if weakening is not None and weakening.rejects and weakening.touched:
        parts.append(f"{weakening.result} {' '.join(weakening.touched)}")
    detail = f" ({'; '.join(parts)})" if parts else ""
    return f"{unit.id} attempt {unit.attempts}: {unit.status}{detail} (${res.cost_usd:.2f})" + _reader_said(weakening)


def _grade(task: Task, unit: Unit, res: worker.WorkerResult,
           root: Root) -> tuple[grade.Verdict, dict[str, str], list[dict[str, Any]] | None, dict[str, list[str]],
                                Path | None]:
    """A delivered candidate's free checks, all in the grading clone (`worker.grading_clone`: a fresh clone of the
    workspace at the candidate), never in the editor's own clone: its own grade, the re-checks and the clean clone.
    (verdict, broke, re-checked, excused, the clone): the clone is kept for the gate's weakening check, which removes
    it, and is None, already removed, when it could not be prepared, when a check left it changed, or on any exception
    here."""
    console_file = root.console_file(task.id)
    try:
        clone = worker.grading_clone(task, root, res.candidate)
    except worker.WorkerError as exc:
        return grade.ungradeable(unit, f"the grading clone could not be prepared: {exc}"), {}, None, {}, None
    try:
        verdict = grade.grade(unit, clone, res.exit_code, console_file=console_file)
        excused = gate.excused(unit, verdict.passed_ids)
        broke, rechecked = _rechecks(task, clone, root, excused) if verdict.passed else ({}, None)
        tampered = worker.clone_changes(clone, res.candidate, console_file)
    except BaseException:
        shutil.rmtree(clone, ignore_errors=True)
        raise
    if tampered:
        shutil.rmtree(clone, ignore_errors=True)
        return grade.ungradeable(unit, tampered), {}, None, {}, None
    return verdict, broke, rechecked, excused, clone


def _attempt_record(res: worker.WorkerResult, verdict: grade.Verdict, rechecked: list[dict[str, Any]] | None,
                    excused: dict[str, list[str]]) -> dict[str, Any]:
    """What finishing an attempt held on `reader_cap` needs, kept in the question's context: the worker's result (its
    clone is gone), its own grade, the re-checks and the excused parents, all of which passed."""
    return {"worker": {key: value for key, value in asdict(res).items() if key != "clone"},
            "grade": asdict(verdict), "recheck": rechecked, "excused": excused}


def _answered_weakening(task: Task, question: Question, caps: Caps, root: Root, runner: Runner) -> str:
    """Act once on the answer to a weakening check held on `reader_cap`. `yes`: the reader is asked once more under
    the wider cap, in a fresh grading clone at the candidate (`gate.weakening`), and its verdict decides the attempt,
    or a cap hit there holds it again on a new question; `no`: the attempt is rejected as a cap hit rejected it
    (`gate.declined`). The free checks are the held attempt's own, never run again."""
    info = question.context["reader_cap"]
    unit = next(u for u in task.units if u.id == info["unit"])
    res = worker.WorkerResult(clone=root.clones_dir(task.id).resolve() / unit.id, **info["worker"])
    verdict, rechecked = grade.Verdict(**info["grade"]), info["recheck"]
    if question.disposition != "reasked":
        weakening = gate.declined(task, unit, question, root)
        return _conclude(task, unit, res, caps, root, verdict, {}, rechecked, weakening, True)
    try:
        clone = worker.grading_clone(task, root, res.candidate)
    except worker.WorkerError as exc:
        question.context["settled"] = True
        verdict = grade.ungradeable(unit, f"the grading clone could not be prepared: {exc}")
        return _conclude(task, unit, res, caps, root, verdict, {}, None, None, True)
    try:
        weakening = gate.weakening(task, unit, clone, info["accepted_tip"], res.candidate, info["excused"],
                                   budget.call_caps(task, caps), root, runner,
                                   charge=_charger(task, root, "reader", caps), answered=question,
                                   attempt=_attempt_record(res, verdict, rechecked, info["excused"]))
    finally:
        shutil.rmtree(clone, ignore_errors=True)
    if weakening.result == gate.HELD:
        return f"{unit.id} attempt {unit.attempts}: held again, the reader reached its cap" + _reader_said(weakening)
    return _conclude(task, unit, res, caps, root, verdict, {}, rechecked, weakening, True)


def _reader_said(weakening: gate.Weakening | None) -> str:
    """The weakening check's part of a worker tick's line: what the reader said and what it cost; "" when the check
    did not run or touched nothing."""
    if weakening is None or weakening.result == gate.UNTOUCHED:
        return ""
    said = {gate.WEAKENED: f"weakened ({weakening.kind})", gate.NOT_WEAKENED: f"not weakened ({weakening.kind})"}
    cost = f" (${weakening.call.charge_usd or 0.0:.2f})" if weakening.call is not None else ""
    return f"; reader: {said.get(weakening.result, f'no verdict ({weakening.reason})')}{cost}"


def _explore_step(task: Task, move: Move, caps: Caps, root: Root, runner: Runner) -> str:
    """One explore: its branch `explore/<session>/<n>` (n counts the session's explores
    that ran), one worker call (`worker.run_explore`), the cost booked on the session and the move, the report stored
    unverified, the clone discarded. No grade, no retry, no integration: `task/<session>` never moves for it. Logged
    at launch with the brief and on landing with the branch, the exit line, the cost and the report."""
    n = 1 + sum(m.kind == EXPLORE and m.branch is not None for m in task.moves if m is not move)
    move.branch, move.outcome = move.branch or worker.explore_branch(task, n), "running"
    with state.LOCK:
        state.save(root, task)
        state.append_log(root, task.id, {"event": "explore", "stage": "launch", "move": move.id, "brief": move.brief,
                                         "branch": move.branch})
    opened = len(task.calls)
    try:
        res = worker.run_explore(task, move, budget.call_caps(task, caps), root, runner,
                                 _charger(task, root, "worker", caps))
    except BaseException as exc:  # an interrupted call is booked by its charge (`_charger`); the move carries it too
        move.cost_usd += sum(call.charge_usd or 0.0 for call in task.calls[opened:] if call.booked)
        move.outcome = "interrupted" if isinstance(exc, KeyboardInterrupt) else f"failed: {exc}"[:NOTE_LIMIT]
        state.save(root, task)
        raise
    task.cost_usd += res.cost_usd
    move.cost_usd += res.cost_usd
    move.report = res.summary
    move.paths = res.changed  # the explore's own changes (from its base), read in the workspace: the console's artifacts
    move.outcome = ("landed" if res.delivered else "not fetched") + (f": {res.delivery_note}"[:NOTE_LIMIT]
                                                                   if res.delivery_note else "")
    shutil.rmtree(res.clone, ignore_errors=True)
    _book(task, root, budget.mark_booked(task), caps)
    state.append_log(root, task.id, {
        "event": "explore", "stage": "landed", "move": move.id, "branch": move.branch if res.delivered else None,
        "commit": res.candidate or None, "changed": res.changed, "exit_code": res.exit_code, "cost_usd": res.cost_usd,
        "report": move.report, "note": res.delivery_note, "unverified": True})
    return f"{move.id} explore: {move.outcome} on {move.branch} (unverified) (${res.cost_usd:.2f})"


def ceiling(task: Task, caps: Caps) -> str | None:
    """The ceiling the task has reached, or the call floor what its allowance has left is under; None while a call may
    be made."""
    return budget.stop_reason(task, caps) or floor_reason(task, budget.call_caps(task, caps))


@dataclass
class Tick:
    """What one tick's nodes share: the tick's arguments and when it started (`clock`); the unit `pick` drew or, with
    none, the explore it drew (both None: the thinker's turn); the thinker move whose console or rest firing runs in
    this tick (`accepting`: the reply, whether its call admitted its branches, the line's tail); the pass that worker
    → grader → gate → ref share (the worker's result, the accepted tip it branched from, the verdict, the broken
    retained predicates, the re-checks, the excused parents, the grading clone until the gate removes it, the weakening
    check); the line the tick echoes; and `adds`, the nodes the runtime fired into the tick beyond its firings (the
    thinker)."""

    task: Task
    caps: Caps
    root: Root
    runner: Runner
    echo: Echo
    unit: Unit | None = None
    explore: Move | None = None
    line: str = ""
    started: float = 0.0
    adds: list[str] = field(default_factory=list)
    accepting: tuple[director.Reply, bool, str] | None = None
    res: worker.WorkerResult | None = None
    tip: str | None = None
    graded: bool = False
    verdict: grade.Verdict | None = None
    broke: dict[str, str] = field(default_factory=dict)
    rechecked: list[dict[str, Any]] | None = None
    excused: dict[str, list[str]] = field(default_factory=dict)
    clone: Path | None = None
    weakening: gate.Weakening | None = None


def _repair_step(tick: Tick) -> bool:
    """Finish every cut pass and a rest's cut memory line (`_repair`). The tick ends here when a pass stays cut, the
    session is terminal, a question is open, something waits for `approve` or the session awaits a thought."""
    task = tick.task
    return (not _repair(task, tick.root) or task.terminal is not None or bool(state.open_questions(task))
            or _awaits_approve(task) or task.awaiting_thought)


def _release_step(tick: Tick) -> bool:
    """A weakening check the operator answered on `reader_cap` is acted on first, and the tick ends there with its
    line, saved and counted inside this node (`_answered_weakening`, then save's act, which the tick's `walked`
    records as `save`, as the walker records every node it runs). A held plan that `approve` (or
    `auto`) released passes the gate; admitted, its units enter the task. Its reader calls (booked in the gate's own
    save) may have reached the allowance or left it under the call floor, or left no receipt, or held it on
    `reader_cap`: then the tick ends before any further call, on the ceiling stop (budget's act, again) or on the
    question."""
    answered = questions.owed(tick.task, "weakening")
    if answered is not None:
        tick.line = _answered_weakening(tick.task, answered, tick.caps, tick.root, tick.runner)
        tick.task.fired.walked.append("save")  # before its act, so its own save records it
        _save_step(tick)
        return True
    if tick.task.held is not None and tick.task.held.kind == PLAN:
        _release_plan(tick.task, tick.caps, tick.root, tick.runner, tick.echo)
        return _budget_step(tick) or bool(state.open_questions(tick.task))
    return False


def _block_step(tick: Tick) -> bool:
    """Block the pending units whose needs died."""
    _block(tick.task, tick.root)
    return False


def _budget_step(tick: Tick) -> bool:
    """At a ceiling (the allowance, or the ticks) or under the call floor the session stops, and the tick ends without
    a call."""
    reason = ceiling(tick.task, tick.caps)
    if reason:
        tick.task.terminal, tick.task.stop_reason = "stopped", reason
        state.append_log(tick.root, tick.task.id, {"event": "stopped", "reason": reason})
        state.save(tick.root, tick.task)
    return bool(reason)


def _pick_step(tick: Tick) -> bool:
    """Draw the tick's firing from the queue, one worker firing per tick: the runnable unit (units already in the
    session run first), else the queued explore. With neither, the runtime fires the thinker."""
    task = tick.task
    tick.unit = state.pending_unit(task)
    tick.explore = None if tick.unit is not None else next(
        (m for m in task.moves if m.kind == EXPLORE and m.outcome == "pending"), None)
    if tick.unit is not None:
        move = next((m.id for m in task.moves if m.kind == "task" and tick.unit.id in m.units), None)
        task.fired.fire.append(Firing("worker", move=move, unit=tick.unit.id))
    elif tick.explore is not None:
        task.fired.fire.append(Firing("worker", move=tick.explore.id))
    else:
        tick.adds.append("director")
    return False


def _hold_step(tick: Tick) -> bool:
    """A picked unit's retry, or the picked explore, runs once admitted; in `hold` it is held for `approve`, and the
    tick ends without a call."""
    if tick.explore is not None:
        return not _explore_admitted(tick.task, tick.explore, tick.root, tick.echo)
    unit = tick.unit
    return unit is not None and bool(unit.attempts) and not _retry_admitted(tick.task, unit, tick.root, tick.echo)


def _director_step(tick: Tick) -> bool:
    """Ask the thinker for its next move (`_thinker_step`)."""
    tick.line = _thinker_step(tick)
    return False


def _worker_step(tick: Tick) -> bool:
    """Run the picked explore, or one attempt of the picked unit (its grader, gate and ref move follow it)."""
    if tick.explore is not None:
        tick.line = _explore_step(tick.task, tick.explore, tick.caps, tick.root, tick.runner)
    else:
        _work(tick)
    return False


def _grader_step(tick: Tick) -> bool:
    """The attempt's free checks. Graded only when the worker's own closing line is `exit_code: 0` and its branch
    carries a change; else ungradeable before any predicate is read (`_grade` for the rest)."""
    res, unit = tick.res, tick.unit
    tick.graded = res.delivered and res.exit_code == 0
    if tick.graded:
        tick.verdict, tick.broke, tick.rechecked, tick.excused, tick.clone = _grade(tick.task, unit, res, tick.root)
    elif res.delivered:
        reported = f"exit_code {res.exit_code}" if res.exit_code is not None else "no exit_code line"
        tick.verdict = grade.ungradeable(unit, f"the worker reported {reported}; nothing was graded")
    else:
        tick.verdict = grade.ungradeable(unit, res.delivery_note)
    return False


def _gate_step(tick: Tick) -> bool:
    """The gate before integration: once every free check passed, the weakening check (`gate.weakening`: one reader
    call when the candidate touched a protected path, none otherwise) in the grading clone, which it then removes."""
    clone, tick.clone = tick.clone, None
    if clone is None:
        return False
    task, unit, res = tick.task, tick.unit, tick.res
    try:
        if tick.verdict.passed and not tick.broke:  # every free check passed: only now the paid one
            tick.weakening = gate.weakening(
                task, unit, clone, tick.tip, res.candidate, tick.excused, budget.call_caps(task, tick.caps), tick.root,
                tick.runner, charge=_charger(task, tick.root, "reader", tick.caps),
                attempt=_attempt_record(res, tick.verdict, tick.rechecked, tick.excused))
    finally:
        shutil.rmtree(clone, ignore_errors=True)
    return False


def _ref_step(tick: Tick) -> bool:
    """The ref move: decide the attempt, and on a pass make its two durable writes (`_conclude`). An attempt whose
    weakening check is held on `reader_cap` stays undecided: its clone goes, nothing is written."""
    unit, res, weakening = tick.unit, tick.res, tick.weakening
    if weakening is not None and weakening.result == gate.HELD:
        shutil.rmtree(res.clone, ignore_errors=True)
        tick.line = (f"{unit.id} attempt {unit.attempts}: held, the reader reached its cap twice (${res.cost_usd:.2f})"
                     + _reader_said(weakening))
        return False
    tick.line = _conclude(tick.task, unit, res, tick.caps, tick.root, tick.verdict, tick.broke, tick.rechecked,
                          weakening, tick.graded)
    return False


def _console_step(tick: Tick) -> bool:
    """The thinker's console firing (an ask or an ideas move), accepted in its own tick (`_accept_fired`)."""
    return _accept_fired(tick)


def _rest_step(tick: Tick) -> bool:
    """The thinker's rest firing, accepted in its own tick (`_accept_fired`); memory's line follows it."""
    return _accept_fired(tick)


def _memory_step(tick: Tick) -> bool:
    """The memory line after a rest (`_remember`)."""
    _remember(tick.task, tick.root)
    return False


def _task_outcomes(task: Task) -> None:
    """Each `task` move whose units entered the session says what became of them."""
    status = {unit.id: unit.status for unit in task.units}
    for move in task.moves:
        if move.kind == "task" and move.outcome not in ("held", "returned", "refused") and move.units:
            move.outcome = ", ".join(f"{uid} {status.get(uid, 'missing')}" for uid in move.units)


def _save_step(tick: Tick) -> bool:
    """Count the tick and its active time (what `clock` read since the tick started), save the
    session, echo the tick's line."""
    tick.task.tick += 1
    tick.task.active_seconds += clock() - tick.started
    _task_outcomes(tick.task)
    state.save(tick.root, tick.task)
    tick.echo(f"tick {tick.task.tick}: {tick.line}")
    return False


# What each registry node does when the walker reaches it (`graph.NODES` declares the nodes and their edges; the reader
# runs only inside the gate and the release, so it has no act of its own). An act returns True when the tick ends there.
ACTS: dict[str, Callable[[Tick], bool]] = {
    "repair": _repair_step, "block": _block_step, "budget": _budget_step, "release": _release_step,
    "pick": _pick_step, "hold": _hold_step, "director": _director_step, "worker": _worker_step,
    "grader": _grader_step, "gate": _gate_step, "ref": _ref_step, "console": _console_step, "rest": _rest_step,
    "memory": _memory_step, "save": _save_step,
}


def _announced(runner: Runner, task: Task, root: Root) -> Runner:
    """`runner`, logging one `launch` event per model call as its process is about to start. Every seat saves the
    call's `pending` record and then hands it to the runner (director, worker, explore and reader alike), so the event
    lands after that save: a client that refreshes on it sees the call in the snapshot's `doing`. A runner that streams
    lines keeps doing so."""
    def announce() -> None:
        with state.LOCK:
            call = task.calls[-1] if task.calls and task.calls[-1].outcome == "pending" else None
            if call is not None:
                state.append_log(root, task.id, {"event": "launch", "call": call.id, "seat": call.seat,
                                                 "unit": call.unit, "attempt": call.attempt, "tick": call.tick,
                                                 **({"walk": call.walk} if call.walk is not None else {})})

    def run(argv: list[str], stdin: str | None, cwd: Path, timeout: int, **kwargs: Any) -> Any:
        announce()
        return runner(argv, stdin, cwd, timeout, **kwargs)

    run.streams_lines = getattr(runner, "streams_lines", False)  # type: ignore[attr-defined]
    return run


def _walk(tick: Tick) -> Iterator[str]:
    """The registry's walk of one tick: the entries in every tick, the nodes the runtime fired into it (`Tick.adds`),
    its firings and the nodes each brings, in the one order their edges allow (`graph.order`). Taken again after every
    node, since a node can add to the tick (pick draws a firing, the thinker's move adds its console or rest firing);
    an addition that would have to run before a node already walked is refused (`graph.Unordered`). Each entry walked
    is appended to the tick's `walked` before it runs, so a save inside it records it."""
    walked: list[str] = []
    while True:
        firings = tick.task.fired.fire
        ids = set(graph.always()) | set(tick.adds) | {f.node for f in firings}
        ids |= {name for f in firings for name in graph.brought(f.node, f.unit)}
        ordered = graph.order(ids)
        if ordered[:len(walked)] != walked:
            raise graph.Unordered(f"{', '.join(k for k in ordered if k not in walked)} entered the tick behind "
                                  f"{walked[-1]}, which must follow it")
        if len(ordered) == len(walked):
            return
        key = ordered[len(walked)]
        walked.append(key)
        name = graph.entries()[key].name
        tick.task.fired.walked.append(name)
        yield name


def run_tick(task: Task, caps: Caps, root: Root, runner: Runner, echo: Echo = _silent) -> Task:
    """One tick: the registry's walk (`_walk`), each node's act (`ACTS`) in turn, until one ends the tick. The tick's
    firing set and the order walked are the record's `fired`, rewritten here, and one `fired` event once the walk
    ends (none when an exception cuts it). Its active time is read from its start. Every model call the tick makes
    logs its `launch` (`_announced`). Each walk gets its own `walk` id (`walk_id`), which its calls carry; `tick` stays
    the ticks counted, so a walk cut before its save and the walk that runs that tick again differ only in `walk`."""
    tick = Tick(task, caps, root, _announced(runner, task, root), echo, started=clock())
    task.fired = Fired(tick=task.tick, walk=walk_id())
    try:
        for name in _walk(tick):
            if ACTS[name](tick):
                break
    finally:
        if tick.clone is not None:  # a pass cut between the grader and the gate: the grading clone goes
            shutil.rmtree(tick.clone, ignore_errors=True)
    state.append_log(root, task.id, {"event": "fired", **asdict(task.fired)})
    return task


def terminal_line(root: Root, task: Task) -> str:
    """One line naming where the task ended."""
    if task.terminal == "interrupted":
        asked = state.open_questions(task)
        if asked:
            return (f"terminal: interrupted — open question {asked[0].id} ({asked[0].asker}): "
                    f"{questions.how(asked[0])}, then `protean resume`")
        if _awaits_approve(task):
            return (f"terminal: interrupted — held {console.held_label(task.held)}: `protean ctl approve` releases "
                    "it (`protean ctl think <text>` returns it to the thinker), then `protean resume`")
        if task.awaiting_thought:
            return (f"terminal: interrupted — awaiting a thought after {awaited(task)}: `protean ctl think <text>` "
                    "adds one, then `protean resume`")
        return "terminal: interrupted — no open question; `protean resume` continues the session"
    if task.terminal == "stopped":
        return f"terminal: stopped — {task.stop_reason}"
    passed = sum(u.status == "passed" for u in task.units)
    return f"terminal: {task.terminal} — {passed} unit(s) verified, ${task.cost_usd:.2f}, {task.tick} ticks"


def _settle(task: Task, caps: Caps, root: Root, echo: Echo) -> None:
    """At the tick boundary: act on every answer the console landed, once, lift a ceiling stop that an extension
    has cleared, and return to the thinker whatever a `think` reached while it was held. Runs under the state lock,
    so an operation landing meanwhile waits its turn."""
    with state.LOCK:
        events = [_deliver(task, q) for q in task.questions if q.status == "answered" and q.disposition is None]
        if task.held is not None and len(task.thoughts) > task.held.directions:
            events.append(_returned(task, task.held))
        if task.terminal == "stopped" and task.stop_reason not in ("abandoned", "closed") and ceiling(task, caps) is None:
            events.append({"event": "released", "hold": "ceiling", "reason": task.stop_reason})
            task.terminal, task.stop_reason = None, ""
        if events:
            state.save(root, task)
            for event in events:
                state.append_log(root, task.id, event)
    for event in events:
        if event["event"] == "released":
            echo(f"released: the ceiling cleared ({event['reason']})")
        elif event["event"] == "returned":
            echo(f"returned: the held {event['kind']} went back to the thinker with {event['thoughts']} "
                 "thought(s)")
        else:
            echo(f"answer {event['id']}: {event['disposition']}")


def awaited(task: Task) -> str:
    """The move after which the session awaits a thought: its latest `ideas` or `rest`."""
    return next((m.id for m in reversed(task.moves) if m.kind in ("ideas", "rest")), "the last move")


def _hold(task: Task) -> tuple[str, ...] | None:
    """What the session waits on at the tick boundary: an open question, a pause, a ceiling stop, something held
    for `approve`, or a thought (after `ideas` or `rest`)."""
    asked = state.open_questions(task)
    if asked:
        return ("question", asked[0].id)
    if task.paused and task.terminal is None:
        return ("paused",)
    if task.terminal == "stopped" and task.stop_reason not in ("abandoned", "closed"):
        return ("ceiling", task.stop_reason)
    if _awaits_approve(task):
        return ("held", task.held.kind)
    if task.awaiting_thought and task.terminal is None:
        return ("thought", awaited(task))
    return None


def _waiting_line(task: Task, hold: tuple[str, ...], unattended: float) -> str:
    if hold[0] == "paused":
        return "waiting: paused — `protean ctl resume` releases it"
    if hold[0] == "ceiling":
        line = f"waiting: stopped ({hold[1]}) — `protean ctl extend $X` (or `N` ticks) widens it"
    elif hold[0] == "held":
        line = (f"waiting: held {console.held_label(task.held)} — `protean ctl approve` releases it; "
                "`protean ctl think <text>` returns it to the thinker")
    elif hold[0] == "thought":
        line = f"waiting: a thought after {hold[1]} — `protean ctl think <text>` adds one"
    else:
        question = state.find_question(task, hold[1])
        line = f"waiting: question {question.id} ({question.asker}) — {questions.how(question)}"
    after = f"{unattended / 60:g}m" if unattended >= 60 else f"{unattended:g}s"
    return line + (f"; parks in {after} unless a client attaches" if unattended else "; parks now, no client attached")


def _park(task: Task, hold: tuple[str, ...], caps: Caps, root: Root) -> None:
    """Nobody attached in time: an open question, a progression held for `approve` or a wait for a thought parks the
    session `interrupted` (the held one stays in `task.json`); a ceiling stop stays `stopped`, naming the ceiling that
    binds now (an extension may have cleared the first)."""
    if hold[0] in ("question", "held", "thought"):
        task.terminal = "interrupted"
    else:
        task.stop_reason = ceiling(task, caps) or task.stop_reason
    with state.LOCK:
        state.save(root, task)
        state.append_log(root, task.id, {"event": "parked", "hold": hold[0], "on": hold[1]})


def _mark_stopped(task: Task, root: Root, desk: Console) -> None:
    """The console's `stop`, Ctrl-C, SIGTERM and SIGHUP all end the drive with KeyboardInterrupt (`cli._interrupt`);
    the record then says `interrupted`, as an unattended park does, with a `parked` event (`hold` `stop`) — so a reader
    without the process tells a stopped session from a lost brain, and `protean resume` clears it as it clears a park
    (audit 2026-10-23, D2). The mark goes on the record AS THE TICK LAST SAVED IT, never on the loop's in-memory task:
    that one carries what the tick has not saved yet — a unit's `attempts` counted before its clone, a verdict's
    status before its event — and persisting it burned a unit's attempt before it ever ran (audit F1). The console then
    serves THAT record, so the stream's last snapshot reads as `task.json` does, never the dirty copy (audit H3) — and
    takes it the moment it is read, before the mark, so a second signal mid-mark (a double Ctrl-C, SIGTERM then
    SIGHUP) leaves it serving the record once the read is done: after the save only the `parked` event is missing;
    inside the save the stream says `interrupted` before disk does; during the read the loop's dirty task is still
    what the console holds — as when the record cannot be read back at all, which marks nothing (audit I3, J1). The
    cut call's spend is already booked (`_charger`, which saves). A record already terminal (a ceiling, `close`,
    `abandon`) keeps its own word."""
    with state.LOCK:
        try:
            disk = state.load(root, task.id)
        except (OSError, ValueError):  # the record cannot be read back: nothing is marked; the interruption propagates
            return
        desk.task = disk
        if disk.terminal is None:
            disk.terminal = "interrupted"
            state.save(root, disk)
            state.append_log(root, task.id, {"event": "parked", "hold": "stop", "on": "the operator"})


def _drive(task: Task, caps: Caps, root: Root, runner: Runner, echo: Echo, desk: Console) -> Task:
    """Tick until the task ends or parks, honoring the console at every tick boundary."""
    while True:
        seen = desk.mark()
        _settle(task, caps, root, echo)
        hold = _hold(task)
        if hold is None:
            desk.settled()
            if task.terminal is not None:
                break
            task = run_tick(task, caps, root, runner, echo)
            continue
        line = _waiting_line(task, hold, desk.unattended)
        if not desk.wait(hold, seen, line, unattended=hold[0] != "paused"):
            _park(task, hold, caps, root)
            break
    echo(terminal_line(root, task))
    return task


@contextmanager
def _task_lock(root: Root, task_id: str) -> Iterator[None]:
    """Hold `state/<task>/lock` while this process drives the task; a second process is refused."""
    path = root.state_dir(task_id) / "lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+", encoding="utf-8") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            handle.seek(0)
            holder = handle.read().strip() or "unknown"
            raise TaskRunning(f"task {task_id} is already running (pid {holder})") from None
        handle.truncate(0)
        handle.write(f"{os.getpid()}\n")
        handle.flush()
        yield


def live_holder(root: Root, task_id: str) -> str | None:
    """The pid of the brain driving the session now (its lock is held: `_task_lock`), or None when none does. The lock
    is the liveness test: a brain holds it for the whole of its drive, the console file included, and the kernel drops
    it with the process, so a console file a kill left behind never counts."""
    try:
        handle = (root.state_dir(task_id) / "lock").open("r+", encoding="utf-8")
    except FileNotFoundError:
        return None
    with handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return handle.read().strip() or "unknown"
        fcntl.flock(handle, fcntl.LOCK_UN)
    return None


def _refuse_a_second(root: Root, other_than: str | None = None) -> None:
    """A live brain refuses a second: while the current session is driven, no new session
    starts and no other one is resumed. A parked session (awaiting a thought, interrupted on a question, stopped) never
    refuses: `current` moves to the new one and the old one stays as it was."""
    try:
        current_id = root.current_file.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        return
    holder = live_holder(root, current_id) if current_id and current_id != other_than else None
    if holder is not None:
        raise TaskRunning(f"session {current_id} is live (pid {holder}); a second brain is refused")


def scratch_workspace(root: Root, task_id: str) -> Path:
    """The tree of a session started with no workspace: a fresh git repository under the
    brain, `workspaces/<session>/` (throwaway, gitignored), with one empty commit on `main`, so an explore and a task
    still have a tree to branch from."""
    path = root.path / "workspaces" / task_id
    path.mkdir(parents=True)
    worker.git(["init", "--quiet", "-b", "main"], path)
    worker.git([*SCRATCH_GIT, "commit", "--quiet", "--allow-empty", "--no-verify", "-m",
                f"protean: scratch workspace for {task_id}"], path)
    return path


def run(thought: str, workspace: str | Path | None, root: Root, caps: Caps, runner: Runner,
        project: str = "_root", echo: Echo = _silent, *, unattended: float = 0.0,
        notify: Notify | None = console.notify, mode: str = AUTO, paused: bool = False) -> Task:
    """Start a session on the operator's `thought` (`protean think`) in `mode` (logged with its start; `hold` holds even the
    first `task` move's plan) and run it, its console served, until it parks or stops. With no `workspace` the session
    gets a scratch repository (`scratch_workspace`). Refused only while another brain is live; a parked session stays
    as it was and `current` moves to the new one. `unattended` is how long a hold waits with no client attached
    before it parks; `notify` fires once per unattended hold. `paused` starts the session held at its first tick
    boundary, as if a `pause` had landed before it (the fixture brain: nothing runs until a client presses Resume)."""
    if mode not in MODES:
        raise ValueError(f"mode {mode!r}: expected one of {', '.join(MODES)}")
    if workspace is not None:
        if not Path(workspace).is_dir():
            raise ValueError(f"workspace {workspace} is not a directory")
        worker.git(["rev-parse", "--git-dir"], Path(workspace))
    _refuse_a_second(root)
    now = datetime.now(timezone.utc)
    task_id = state.new_task_id(now)
    if root.task_file(task_id).exists():
        raise ValueError(f"session {task_id} already exists; start again in a second")
    scratch = workspace is None
    if scratch:
        workspace = scratch_workspace(root, task_id)
    task = Task(id=task_id, thought=thought, workspace=str(Path(workspace).resolve()),
                project=project, created=now.isoformat(timespec="seconds"), mode=mode)
    with _task_lock(root, task.id):
        state.save(root, task)
        state.set_current(root, task.id)
        state.append_log(root, task.id, {"event": "start", "thought": thought, "workspace": task.workspace,
                                         "mode": task.mode})
        echo(f"session {task.id}: {thought}" + (" (mode hold: a task move's plan, each retry and each explore wait "
                                                "for approve)" if task.mode == HOLD else ""))
        if scratch:
            echo(f"workspace: {task.workspace} (a fresh scratch repository: one empty commit)")
        if paused:  # landed before the console serves, so written as an offline operation is
            task.paused = True
            state.save(root, task)
            state.append_log(root, task.id, {"event": "pause", "actor": "console", "live": False})
        with console.serve(root, task, caps, echo, unattended, notify) as desk:
            try:
                return _drive(task, caps, root, runner, echo, desk)
            except KeyboardInterrupt:
                _mark_stopped(task, root, desk)
                raise


def resume(root: Root, caps: Caps, runner: Runner, extend_usd: float = 0.0, extend_ticks: int = 0,
           abandon: bool = False, echo: Echo = _silent, *, unattended: float = 0.0,
           notify: Notify | None = console.notify, session: str | None = None) -> Task:
    """Continue a session, its console served, or abandon it: the named `session` (`protean resume <session>`, which
    then becomes the current one, unless another brain is live) or, with none named, the current one. Spend a kill left
    unbooked is booked first; a call lost before its receipt opens a runtime question. `extend_usd`/`extend_ticks`
    (`resume --extend`) are one console `extend`, landed before the loop starts."""
    if session is not None:
        if not SESSION_NAME.fullmatch(session) or not root.task_file(session).is_file():
            raise ValueError(f"no session {session} in {root.path}")
        task_id = session
        _refuse_a_second(root, other_than=task_id)
    else:
        task = state.current(root)
        if task is None:
            raise ValueError(f"no session in {root.path}")
        task_id = task.id
    with _task_lock(root, task_id):
        if session is not None:
            state.set_current(root, task_id)
        task = _prepare(state.load(root, task_id), root, caps, extend_usd, extend_ticks, abandon, echo)
        if state.is_closed(task):
            echo(terminal_line(root, task))
            return task
        with console.serve(root, task, caps, echo, unattended, notify) as desk:
            try:
                return _drive(task, caps, root, runner, echo, desk)
            except KeyboardInterrupt:
                _mark_stopped(task, root, desk)
                raise


def _prepare(task: Task, root: Root, caps: Caps, extend_usd: float, extend_ticks: int, abandon: bool,
             echo: Echo) -> Task:
    """Everything a resume settles before the console serves: lost calls, an abandon, an extension, a cut pass."""
    if state.is_closed(task):
        return task
    _recover(task, root, not abandon, caps)
    for move in task.moves:  # an explore a kill cut mid-call never runs again: no retry
        if move.kind == EXPLORE and move.outcome == "running":
            move.outcome = "lost: the brain stopped during its call"
    if abandon:
        closed = state.open_questions(task)
        for question in closed:
            question.status, question.disposition = "superseded", "abandoned"
        shutil.rmtree(root.clones_dir(task.id), ignore_errors=True)
        task.terminal, task.stop_reason = "stopped", "abandoned"
        state.save(root, task)
        state.append_log(root, task.id, {"event": "abandoned", "superseded": [q.id for q in closed]})
        return task
    if extend_usd or extend_ticks:
        event = console.apply(task, caps, "extend", {"usd": extend_usd, "ticks": extend_ticks}, False)
        state.save(root, task)
        state.append_log(root, task.id, event)
        echo(console.describe(event))
    if _repair(task, root):
        questions.repaired(root, task)
    if task.terminal == "interrupted":
        task.terminal = None
    state.save(root, task)
    return task


def offline_op(root: Root, caps: Caps, op: str, body: dict[str, Any]) -> dict[str, Any] | None:
    """Land a console operation straight in the current session's `task.json` when no brain drives it (the file is
    the truth): an answer, a think or an extend for a parked session, then `protean resume`. None when a brain holds
    the session; refused (`console.Refused`) when the operation needs one. A think or an answer whose `request_id` the
    record already holds returns the original event and writes nothing (`console.replayed`), as the live console
    does, except the event itself when an earlier log append failed (`console.relog`); with another payload it is
    refused (409). A failed save raises and lands nothing; a log append that fails after the save raises
    `console.Unlogged`, whose message says the operation landed."""
    task = state.current(root)
    if task is None:
        raise ValueError(f"no session in {root.path}")
    try:
        with _task_lock(root, task.id):
            task = state.load(root, task.id)
            original = console.replayed(task, op, body)
            if original is not None:
                try:
                    console.relog(root, task, [console.request_id(op, body)])
                except OSError as exc:
                    raise console.unlogged(op, exc) from exc
                return original
            event = console.apply(task, caps, op, body, False)
            state.save(root, task)
            try:
                state.append_log(root, task.id, event)
            except OSError as exc:
                raise console.unlogged(op, exc) from exc
            if op == "close":
                remembered = memory.logged(root, task, "close")
                if remembered is not None:
                    state.append_log(root, task.id, remembered)
                shutil.rmtree(root.clones_dir(task.id), ignore_errors=True)
            return event
    except TaskRunning:
        return None
