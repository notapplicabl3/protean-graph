"""Cap-hit stop: a director call the receipt says reached its dollar cap holds the task at once on a runtime
question and a `ceiling` event naming the effective cap and what bound it; no second call on that input;
only `extend` releases it (an answer is refused). An incomplete reply without the cap signal keeps the
malformed-plan retry, whose second failure is a runtime question too."""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import pytest

from protean import loop
from protean.cli import parse_extend
from protean.graph import console, director
from protean.graph.budget import Caps
from protean.dry import REPO, ScriptedRunner, make_workspace
from protean.state import MOVE_KINDS, Root, Task, load, open_questions, save, set_current

UNIT = {"id": "u-1", "intent": "write a.txt", "predicates": [{"kind": "file_exists", "args": {"path": "a.txt"}}]}
PASS = {"write": {"a.txt": "A\n"}, "commit": True, "exit_code": 0, "result": "wrote a.txt", "total_cost_usd": 0.5}
AT_CAP = json.loads((REPO / "evidence" / "captures" / "p4-at-cap.stdout.json").read_text())


STANDINGS = itertools.count(1)


def _with_state(obj: object) -> object:
    """A thinker move without `state` gets one whose standing no earlier reply had."""
    if isinstance(obj, dict) and "state" not in obj and any(kind in obj for kind in MOVE_KINDS):
        return {**obj, "state": {"standing": f"Standing {next(STANDINGS)}.", "open": []}}
    return obj


def _reply(obj: dict | str, cost: float = 0.1) -> dict:
    obj = _with_state(obj)
    text = obj if isinstance(obj, str) else json.dumps(obj)
    return {"type": "result", "is_error": False, "subtype": "success", "session_id": "s-1", "total_cost_usd": cost,
            "result": text}


def _at_cap(cost: float | None = None) -> dict:
    """The real at-cap receipt, its reported spend replaced when `cost` is given."""
    return AT_CAP if cost is None else {**AT_CAP, "total_cost_usd": cost}


@pytest.fixture
def env(tmp_path: Path, root: Root) -> tuple[Root, Path]:
    root.path.mkdir(parents=True)
    root.thinker_prompt.write_text("m\n")
    root.editor_prompt.write_text("e\n")
    return root, make_workspace(tmp_path / "ws")


def _task(root: Root, workspace: Path, **fields: object) -> Task:
    task = Task(id="task-20261010T130000", goal="g", workspace=str(workspace), **fields)
    save(root, task)
    set_current(root, task.id)
    return task


def _events(root: Root, task_id: str, kind: str) -> list[dict]:
    return [e for e in director.read_log(root, task_id) if e["event"] == kind]


def _cap_arg(argv: list[str]) -> str:
    return argv[argv.index("--max-budget-usd") + 1]


def _assert_held_on_the_ceiling(root: Root, task: Task, runner: ScriptedRunner, cap: float, bound_by: str,
                                named: str, terminal: str | None = None) -> dict:
    """One director call, the task held on a runtime question naming the cap and the constraint, one
    `ceiling` event with both facts, the call's record capped and booked once; nothing for the director. A charge
    that reached the session's allowance has stopped it as well, in the save that booked it (`terminal`)."""
    assert len(runner.calls) == 1 and _cap_arg(runner.calls[0][0]) == f"{cap:.2f}"
    (held,) = open_questions(task)
    assert (task.terminal, held.asker, held.reason) == (terminal, "runtime", "cap_hit")
    assert f"${cap:.2f} cap" in held.text and named in held.text
    (event,) = _events(root, task.id, "ceiling")
    assert (event["seat"], event["cap_usd"], event["bound_by"]) == ("director", cap, bound_by)
    (call,) = task.calls
    assert event["call"] == call.id and event["charged_usd"] == call.charge_usd
    assert (call.seat, call.attempt, call.cap_usd, call.outcome, call.subtype, call.booked) == (
        "director", 1, cap, "error", "error_max_budget_usd", True)
    assert call.charge_usd == call.receipt_cost_usd == call.actual_usd
    assert [e["id"] for e in _events(root, task.id, "call")] == [call.id]
    (asked,) = _events(root, task.id, "question")
    assert asked["cap_hit"] == call.id == held.context["cap_hit"] and asked["id"] == held.id
    assert "to_director" not in asked and "to_director" not in event
    assert director.pending_notes(root, task.id) == []
    assert load(root, task.id) == task
    return event


def test_a_confirmed_cap_hit_bound_by_the_per_call_limit_parks_at_once_without_a_second_call(env: tuple[Root, Path]):
    root, workspace = env
    task = _task(root, workspace)
    runner = ScriptedRunner([_at_cap(), _reply({"task": [UNIT]})], [])
    loop.run_tick(task, Caps(), root, runner)
    event = _assert_held_on_the_ceiling(root, task, runner, 2.0, "per_call_limit",
                                          "the per-call limit (director_call_usd $2.00)")
    assert event["per_call_limit_usd"] == 2.0 and event["task_left_usd"] == 25.0
    assert task.cost_usd == AT_CAP["total_cost_usd"] and task.tick == 1 and task.units == []
    assert len(runner.director) == 1  # the scripted retry reply was never asked for


def test_a_cap_hit_clipped_by_the_remaining_task_budget_names_the_task_budget(env: tuple[Root, Path]):
    root, workspace = env
    task = _task(root, workspace, cost_usd=24.0)
    runner = ScriptedRunner([_at_cap(1.02), _reply({"task": [UNIT]})], [])
    loop.run_tick(task, Caps(), root, runner)
    event = _assert_held_on_the_ceiling(root, task, runner, 1.0, "task_budget",
                                        "the task's remaining budget ($1.00 left)", terminal="stopped")
    assert event["per_call_limit_usd"] == 2.0 and event["task_left_usd"] == 1.0
    assert "per-call limit" not in open_questions(task)[0].text.split(";")[0]
    assert round(task.cost_usd, 2) == 25.02 and task.stop_reason == "max_usd: 25.02 >= 25.0"


def test_resume_extend_after_a_cap_hit_calls_the_thinker_with_the_widened_cap(env: tuple[Root, Path]):
    root, workspace = env
    task = _task(root, workspace, cost_usd=24.0)
    runner = ScriptedRunner([_at_cap(1.02), _reply({"task": [UNIT]}), _reply({"rest": True})], [PASS])
    loop.run_tick(task, Caps(), root, runner)
    (asked,) = open_questions(task)
    lines: list[str] = []
    held = loop.resume(root, Caps(), runner, echo=lines.append)  # no extension, no answer: no paid call
    assert (held.terminal, [q.id for q in open_questions(held)], len(runner.calls)) == ("interrupted", [asked.id], 1)
    assert "protean ctl extend" in lines[-1]
    with pytest.raises(console.Refused, match="cap hit"):  # an answer alone does not release a cap hit
        loop.offline_op(root, Caps(), "answer", {"id": asked.id, "text": "yes"})
    assert loop.resume(root, Caps(), runner).terminal == "interrupted" and len(runner.calls) == 1
    usd, ticks = parse_extend("$1.5")
    task = loop.resume(root, Caps(), runner, usd, ticks)
    assert (task.terminal, task.awaiting_thought, task.extra_usd) == ("interrupted", True, 1.5)
    second_argv, second_stdin = runner.calls[1][0], runner.calls[1][1]
    assert _cap_arg(second_argv) == "1.48"  # $25 + $1.50 - $25.02 left; $0.98 less without the extension
    assert json.loads(second_stdin)["notes"] == [] and "runtime:" not in second_stdin and "reached its" not in second_stdin
    capped = task.calls[0]
    assert (capped.outcome, capped.charge_usd) == ("error", 1.02)
    assert [e["id"] for e in _events(root, task.id, "call")].count(capped.id) == 1
    assert [c.charge_usd for c in task.calls] == [1.02, 0.1, 0.5, 0.1]
    assert sum(c.charge_usd for c in task.calls) + 24.0 == pytest.approx(task.cost_usd)
    assert len(_events(root, task.id, "ceiling")) == 1
    (released,) = task.questions
    assert (released.status, released.actor, released.disposition) == ("answered", "console", "extended")
    (extend,) = _events(root, task.id, "extend")
    assert (extend["usd"], extend["max_usd"], extend["released"]) == (1.5, 26.5, [asked.id])


def test_a_cap_hit_on_the_retry_parks_on_the_ceiling_not_the_invalid_plan_question(env: tuple[Root, Path]):
    root, workspace = env
    task = _task(root, workspace)
    runner = ScriptedRunner([_reply("no plan here", cost=0.3), _at_cap(1.75), _reply({"task": [UNIT]})], [])
    loop.run_tick(task, Caps(), root, runner)
    assert len(runner.calls) == 2 and _cap_arg(runner.calls[1][0]) == "2.00"
    (event,) = _events(root, task.id, "ceiling")
    assert (event["call"], event["cap_usd"], event["bound_by"], event["task_left_usd"]) == (
        "c2", 2.0, "per_call_limit", 24.7)
    (held,) = open_questions(task)
    assert (held.asker, held.reason) == ("runtime", "cap_hit") and "invalid plan" not in held.text
    assert round(task.cost_usd, 2) == 2.05


@pytest.mark.parametrize("incomplete", [
    _reply('{"task": [{"id": "u-1", "intent": "write a'),  # cut off inside a string: unreadable
    {"type": "result", "is_error": True, "subtype": "error_during_execution", "session_id": "s-1",
     "total_cost_usd": 0.2, "errors": ["boom"]},
], ids=["cut-off-text", "error-without-cap-subtype"])
def test_an_incomplete_reply_without_the_cap_signal_still_takes_the_malformed_plan_retry(
    env: tuple[Root, Path], incomplete: dict
):
    root, workspace = env
    task = _task(root, workspace)
    runner = ScriptedRunner([incomplete, incomplete], [])
    loop.run_tick(task, Caps(), root, runner)
    assert len(runner.calls) == 2 and "Your previous answer was rejected:" in runner.calls[1][1]
    (held,) = open_questions(task)
    assert (held.asker, held.reason) == ("runtime", "invalid_plan")
    assert held.text.startswith("The thinker returned an invalid reply twice")
    assert _events(root, task.id, "ceiling") == [] and "cap_hit" not in _events(root, task.id, "question")[0]
    assert [c.attempt for c in task.calls] == [1, 2]
