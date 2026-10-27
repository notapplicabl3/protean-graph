"""The runtime-questions node: the runtime's own exchanges with the operator, floating beside the ring.

Deterministic and model-free. It is attached to the console on one side (the question is emitted there, the answer
lands there) and to the runtime on the other (the loop calls these functions and acts on what they return); it has no
edge to the director. A runtime question or its answer never becomes a `to_director` note, and this module never
depends on `graph/director.py`. The reasons: `unknown_spend`, `cap_hit`, `invalid_plan`, `no_progress`, `stale_tip`,
`reader_cap` (`done_refused` only on goal-mode records, which load and are never resumed as sessions). Spend that reaches the
session's allowance is no question: it is the ceiling stop (`loop._book`, `loop.ceiling`), which only `extend` clears.

`ask` is the one place any question is opened, the director's own included: that is bookkeeping in `task.json`; the
director's questions and their answers stay the director's (`loop._deliver`).
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from protean import state
from protean.graph.console import is_yes
from protean.runner import PER_CALL_LIMIT
from protean.state import RUNTIME, Call, Question, Root, Task, Unit


def ask(root: Root, task: Task, asker: str, reason: str, text: str, *, logged: Iterable[dict[str, Any]] = (),
        **context: Any) -> str:
    """Open a question for the operator in `task.json`; `context` is what the runtime needs to act on the answer. Saved
    before it is logged, so no client sees an id the file lacks; `logged` are events of what that same save holds (a
    booking), logged before the question. The task holds at the next tick boundary."""
    with state.LOCK:
        question = state.add_question(task, asker, reason, text, context)
        state.save(root, task)
        for event in logged:
            state.append_log(root, task.id, event)
        state.append_log(root, task.id, {"event": "question", "id": question.id, "asker": asker, "reason": reason,
                                         "text": text, **context})
    return f"asked the operator ({question.id}): {' '.join(text.split())[:160]}"


def consume(question: Question) -> dict[str, Any]:
    """Act once on a runtime answer the console landed; returns the log event. It stays with the runtime: the
    disposition says what the runtime did, and nothing reaches the director."""
    if question.reason == "reader_cap":  # `yes` asks the reader again; `no` takes the cap hit's own path
        refused = "refused" if question.context["reader_cap"]["check"] == "admission" else "rejected"
        question.disposition = "reasked" if is_yes(question.answer or "") else refused
    else:
        question.disposition = {"cap_hit": "extended", "stale_tip": "retried"}.get(question.reason, "continued")
    return {"event": "runtime_answer", "id": question.id, "reason": question.reason,
            "disposition": question.disposition}


def how(question: Question) -> str:
    """What releases a question, as the terminal says it."""
    if question.reason == "cap_hit":
        return "`protean ctl extend $X` (or `N` ticks) releases it"
    if question.reason == "unknown_spend":
        return f"`protean ctl answer {question.id} yes` continues under the unknown spend"
    if question.reason == "no_progress":
        return f"`protean ctl answer {question.id} yes` calls the thinker again"
    if question.reason == "reader_cap":
        undone = ("refuses the branch" if question.context.get("reader_cap", {}).get("check") == "admission"
                  else "rejects the attempt")
        return (f"`protean ctl answer {question.id} yes` asks the reader once more under the wider cap; "
                f"`protean ctl answer {question.id} no` {undone} as a cap hit")
    return f"answer it with `protean ctl answer {question.id} <text>`"


def _ceiling_question(hit: dict[str, Any]) -> str:
    """The runtime question for a director call that reached its dollar cap: the cap, what bound it, and what
    `extend` will and will not widen (per-call limits never widen live)."""
    if hit["bound_by"] == PER_CALL_LIMIT:
        bound = f"the per-call limit (director_call_usd ${hit['per_call_limit_usd']:.2f})"
        then = "the per-call limit never widens, so the director is called again under the same cap"
    else:
        bound = f"the task's remaining budget (${hit['task_left_usd']:.2f} left)"
        then = "the director is called again with its cap widened by the extension, up to the per-call limit"
    return (f"The director's call {hit['call']} reached its ${hit['cap_usd']:.2f} cap, set by {bound}; "
            f"it was charged ${hit['charged_usd']:.2f} and is not retried. Paid work is held: "
            f"`extend` on the console (`protean ctl extend $X`, or `N` ticks) releases it ({then}); "
            "`stop` ends the run, and `protean resume --abandon` closes the task.")


def cap_hit(root: Root, task: Task, hit: dict[str, Any]) -> str:
    """A confirmed director cap hit: the `ceiling` event, then the task held on a runtime question that only
    `extend` releases."""
    state.append_log(root, task.id, {"event": "ceiling", **hit})
    return ask(root, task, RUNTIME, "cap_hit", _ceiling_question(hit), cap_hit=hit["call"])


def invalid_plan(root: Root, task: Task, text: str) -> str:
    """The thinker returned an invalid reply twice: the runtime asks, and the answer never reaches the seat."""
    return ask(root, task, RUNTIME, "invalid_plan", text)


def stalled(root: Root, task_id: str) -> bool:
    """The thinker's previous reply made no progress either: a `no_progress` event is logged
    and, since it, no move was accepted (a `thought` event) and no `no_progress` question was answered."""
    stalled = False
    for event in state.read_log(root, task_id):
        if event.get("event") == "no_progress":
            stalled = True
        elif event.get("event") == "thought" or (event.get("event") == "runtime_answer"
                                                 and event.get("reason") == "no_progress"):
            stalled = False
    return stalled


def no_progress(root: Root, task: Task) -> str:
    """A second reply in a row left the standing unchanged: the runtime asks the operator, and no further paid call is made
    until they answer. `yes` lets the thinker be called again; the thoughts that land meanwhile ride in that call, the
    answer never does."""
    question = ("The thinker's last two replies left the standing unchanged, so neither was acted on and paid work is "
                "held. Answer `yes` to call it again (a thought you add first rides in that call); "
                "`protean resume --abandon` closes the session.")
    return ask(root, task, RUNTIME, "no_progress", question)


def stale_tip(root: Root, task: Task, unit: Unit, found: str, ref: str) -> None:
    """`task/<id>` was moved outside the loop: hold the task on a runtime question, once."""
    if any(q.reason == "stale_tip" for q in state.open_questions(task)):
        return
    question = (f"task/{task.id} was moved outside the loop: it should be at {task.accepted} "
                f"(the accepted tip before {unit.id}) but is {found[:12]}. Reset it with "
                f"`git update-ref {ref} {task.accepted}` in {task.workspace}, then any answer retries "
                "(or `protean resume` after a park); `protean resume --abandon` closes the task.")
    ask(root, task, RUNTIME, "stale_tip", question, stale_tip=unit.id)


def repaired(root: Root, task: Task) -> None:
    """A resume found the accepted ref restored: each open stale-tip question is superseded, `repaired`."""
    for question in [q for q in state.open_questions(task) if q.reason == "stale_tip"]:
        question.status, question.disposition = "superseded", "repaired"
        state.append_log(root, task.id, {"event": "repaired", "id": question.id})


def unknown_spend(root: Root, task: Task, unreceipted: list[Call]) -> None:
    """Calls that ended without a receipt (timed out, no receipt, interrupted, or lost to a kill): their reserved caps
    are booked, their actual spend is UNKNOWN, and paid work holds on a runtime question that only `yes` releases."""
    held = "; ".join(f"{c.id} ({c.seat}{' ' + c.unit if c.unit else ''}, attempt {c.attempt}, {c.outcome}, "
                     f"${c.cap_usd:.2f} reserved)" for c in unreceipted)
    question = (f"{len(unreceipted)} model call(s) ended without a receipt: {held}. "
                "Each reserved cap is booked on the task, but the actual spend is UNKNOWN and can exceed it. "
                "Paid work is held. Answer `yes` to continue under that uncertainty; "
                "`protean resume --abandon` closes the task.")
    ask(root, task, RUNTIME, "unknown_spend", question, unknown_spend=[c.id for c in unreceipted])


def reader_cap(root: Root, task: Task, held: dict[str, Any], calls: list[Call], caps_usd: tuple[float, float],
               floor: str | None = None, logged: Iterable[dict[str, Any]] = ()) -> str:
    """A reader call reached its dollar cap under the wider cap too (or the wider call had no room under the call
    floor): the attempt or the plan `held` names (`check` `weakening` or `admission`, `unit`) is held, neither refused
    nor rejected, on a runtime question. `calls` are the reader's calls on this check so far, each booked, the last in
    this same save (`logged`); `caps_usd` the per-call cap and the wider one. `yes` asks once more under the wider cap
    (`owed`); `no` takes the path a cap hit took before (the attempt rejected unchecked, the branch refused); `extend`
    alone never releases it."""
    cost = sum(call.charge_usd or 0.0 for call in calls)
    held = {**held, "calls": [call.id for call in calls], "cost_usd": cost}
    narrow, wide = caps_usd
    tried = ", then ".join(f"{call.id} at ${call.cap_usd:.2f}" for call in calls)
    if floor:
        tried += f"; the wider call could not start ({floor})"
    if held["check"] == "weakening":
        what = f"weakening check on {held['unit']} (attempt {held['attempt']})"
        hold, yes, no = ("The attempt is held, neither graded nor rejected.", "grades the attempt on its verdict",
                         "rejects the attempt as unchecked (weakening check could not run)")
    else:
        what = f"branch admission check on {held['unit']}"
        hold, yes, no = ("The plan is held unadmitted, not refused.", "admits or refuses on its verdict",
                         "refuses the branch (cap_hit) and the plan goes back to the thinker")
    text = (f"The reader's {what} reached its dollar cap: {tried}. The per-call cap is reader_call_usd "
            f"${narrow:.2f}, the re-ask's ${wide:.2f} (twice it, clipped by what the allowance has left); the reader "
            f"has cost ${cost:.2f} on this check so far. {hold} Answer `yes` to ask the reader once more under the "
            f"${wide:.2f} cap (it {yes}); `no` {no}. `extend` alone does not release it; "
            "`protean resume --abandon` closes the session.")
    return ask(root, task, RUNTIME, "reader_cap", text, logged=logged, reader_cap=held)


def owed(task: Task, check: str) -> Question | None:
    """The answered `reader_cap` question on `check` (`weakening` or `admission`) the runtime has not acted on yet
    (`settled`), or None."""
    return next((q for q in task.questions if q.reason == "reader_cap" and q.status == "answered"
                 and q.context["reader_cap"]["check"] == check and not q.context.get("settled")), None)
