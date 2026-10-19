"""A plan's branch admission, its release and the mode toggle.

F2: admission is decided once per plan, by the mode its director call started in; a mode switch during the call
neither skips the reader (hold -> auto) nor pays it twice (auto -> hold). F3: the tick runs its ceilings before it
releases a held plan, so an approved plan's reader call is never paid at a ceiling and the call floor never drops an
approved plan; it stays held until an extension. F4: a think that lands while a released plan's admission call runs
wins over the `approve`: the plan goes back to the director and none of its units enter the task.
"""

from __future__ import annotations

import itertools
import json
import subprocess
from pathlib import Path

import pytest

from protean import loop, state
from protean.dry import ScriptedRunner, make_workspace
from protean.graph import console, director
from protean.graph.budget import Caps
from protean.state import AUTO, HOLD, MOVE_KINDS, PLAN, Held, Predicate, Root, Task, Unit, save, set_current

TASK_ID = "task-20261011T160000"
MEAN = "def mean(xs):\n    return sum(xs) / len(xs)\n"
RATIONALE = "u-1 p1 pinned the stats module; it is renamed to metrics.py, which u-2 p1 pins holding mean()."
BRANCH = {"id": "u-2", "intent": "rename stats.py to metrics.py", "rationale": RATIONALE, "predicates": [
    {"kind": "file_contains", "args": {"path": "metrics.py", "text": "def mean"}, "parents": ["u-1 p1"]}]}
W_RENAME = {"write": {"metrics.py": MEAN}, "delete": ["stats.py"], "commit": True, "exit_code": 0,
            "result": "renamed stats.py to metrics.py", "total_cost_usd": 0.5}
YES = {"type": "result", "is_error": False, "session_id": "dry-reader", "total_cost_usd": 0.05,
       "result": json.dumps({"verdict": "yes", "note": "metrics.py carries mean()."})}


STANDINGS = itertools.count(1)


def _with_state(obj: object) -> object:
    """A thinker move without `state` gets one whose standing no earlier reply had."""
    if isinstance(obj, dict) and "state" not in obj and any(kind in obj for kind in MOVE_KINDS):
        return {**obj, "state": {"standing": f"Standing {next(STANDINGS)}.", "open": []}}
    return obj


def _reply(obj: object, cost: float = 0.2) -> dict:
    obj = _with_state(obj)
    return {"type": "result", "is_error": False, "session_id": "dry-director", "total_cost_usd": cost,
            "result": json.dumps(obj)}


def _git(cwd: Path, *args: str) -> str:
    return subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t.invalid", *args], cwd=cwd,
                          capture_output=True, text=True, check=True).stdout.strip()


def _branch_unit(uid: str = "u-2") -> Unit:
    return Unit(uid, "rename stats.py to metrics.py",
                [Predicate("file_contains", {"path": "metrics.py", "text": "def mean"}, parents=["u-1 p1"])],
                rationale=RATIONALE)


@pytest.fixture
def accepted(tmp_path: Path, root: Root) -> tuple[Root, Task]:
    """A saved task whose u-1 (stats.py) is accepted at the workspace tip, nothing else planned."""
    root.path.mkdir(parents=True)
    for prompt in (root.thinker_prompt, root.editor_prompt, root.reader_prompt):
        prompt.write_text("p\n")
    workspace = make_workspace(tmp_path / "ws")
    (workspace / "stats.py").write_text(MEAN)
    _git(workspace, "add", "stats.py")
    _git(workspace, "commit", "-qm", "stats")
    tip = _git(workspace, "rev-parse", "HEAD")
    u1 = Unit("u-1", "write stats.py", [Predicate("file_exists", {"path": "stats.py"})], status="passed",
              candidate=tip, integrated=True)
    task = Task(id=TASK_ID, goal="g", workspace=str(workspace), units=[u1], accepted=tip)
    _git(workspace, "update-ref", f"refs/heads/task/{TASK_ID}", tip)
    save(root, task)
    set_current(root, task.id)
    return root, task


class Effect:
    """The scripted runner, doing `effect` when a call of `seat` starts (a console operation landing mid-call)."""

    def __init__(self, scripted: ScriptedRunner, seat: str, effect) -> None:
        self.scripted, self.seat, self.effect = scripted, seat, effect

    def __call__(self, argv, stdin, cwd, timeout):
        if ScriptedRunner.seat(argv) == self.seat:
            self.effect()
        return self.scripted(argv, stdin, cwd, timeout)

    def seats(self) -> list[str]:
        return [ScriptedRunner.seat(argv) for argv, _, _, _ in self.scripted.calls]


def _events(root: Root, name: str) -> list[dict]:
    return [e for e in state.read_log(root, TASK_ID) if e["event"] == name]


def _set_mode(task: Task, mode: str):
    def flip() -> None:
        with state.LOCK:  # what the console's `mode` operation does to the loop's copy
            task.mode = mode
    return flip


# --- F2: admission decided once per plan ---


def test_hold_to_auto_during_the_director_call_admits_the_branch_with_one_reader_call_before_it_enters(
    accepted: tuple[Root, Task]
):
    root, task = accepted
    task.mode = HOLD
    runner = Effect(ScriptedRunner([_reply({"task": [BRANCH]})], [W_RENAME], [YES]), "director",
                    _set_mode(task, AUTO))
    loop.run_tick(task, Caps(), root, runner)
    entered_after_tick_1 = [u.id for u in task.units]
    loop.run_tick(task, Caps(), root, runner)
    assert runner.seats() == ["director", "reader", "worker"]  # exactly one reader call, before the unit's worker
    assert entered_after_tick_1 == ["u-1"]  # the plan did not enter with the reply: it was held for admission
    assert len(_events(root, "branch_admitted")) == 1 and task.units[1].status == "passed"


def test_auto_to_hold_during_the_director_call_pays_the_reader_once(accepted: tuple[Root, Task]):
    root, task = accepted
    runner = Effect(ScriptedRunner([_reply({"task": [BRANCH]})], [W_RENAME], [YES, YES]), "director",
                    _set_mode(task, HOLD))
    loop.run_tick(task, Caps(), root, runner)  # admitted inside the call; the operator switched to hold meanwhile: held
    assert task.held is not None and task.held.kind == PLAN and [u.id for u in task.units] == ["u-1"]
    console.apply(task, Caps(), "approve", {}, False)
    loop.run_tick(task, Caps(), root, runner)
    assert runner.seats() == ["director", "reader", "worker"]  # one reader call in total
    assert len(_events(root, "branch_admitted")) == 1 and [u.id for u in task.units] == ["u-1", "u-2"]


# --- F3: ceilings before release ---


@pytest.mark.parametrize("at, extend, reason", [
    ({"tick": 100}, {"extend_ticks": 5}, "max_ticks: 100 >= 100"),
    ({"cost_usd": 24.6}, {"extend_usd": 5.0}, "max_usd: remaining $0.40 below the $0.50 call floor"),
])
def test_an_approved_held_plan_at_a_ceiling_makes_no_call_stays_held_and_is_released_after_extend(
    accepted: tuple[Root, Task], at: dict, extend: dict, reason: str
):
    root, task = accepted
    for key, value in at.items():
        setattr(task, key, value)
    task.mode, task.held = HOLD, Held(kind=PLAN, units=[_branch_unit()], approved=True)
    save(root, task)
    runner = ScriptedRunner([_reply({"rest": True})], [W_RENAME], [YES])
    loop.run_tick(task, Caps(), root, runner)
    assert (runner.calls, task.terminal, task.stop_reason) == ([], "stopped", reason)
    assert task.held is not None and task.held.approved and [u.id for u in task.units] == ["u-1"]
    assert _events(root, "branch_refused") == [] and _events(root, "approved") == []
    task = loop.resume(root, Caps(), runner, unattended=0.0, notify=None, **extend)
    assert [ScriptedRunner.seat(argv) for argv, _, _, _ in runner.calls] == ["reader", "worker", "director"]
    assert (task.terminal, task.awaiting_thought) == ("interrupted", True)
    assert [(e["kind"], e["by"]) for e in _events(root, "approved")] == [("plan", "approve")]


def test_the_call_floor_reached_between_two_admission_calls_keeps_the_plan_held(accepted: tuple[Root, Task]):
    root, task = accepted
    task.cost_usd, task.mode = 23.9, HOLD  # $1.10 left: the first reader call ($0.65) leaves less than the floor
    task.held = Held(kind=PLAN, units=[_branch_unit("u-2"), _branch_unit("u-3")], approved=True)
    save(root, task)
    costly = {**YES, "total_cost_usd": 0.65}
    runner = ScriptedRunner([], [], [costly])
    loop._release_plan(task, Caps(), root, runner, lambda line: None)
    assert len(runner.calls) == 1 and task.held is not None and task.held.approved
    assert [u.id for u in task.held.units] == ["u-2", "u-3"] and [u.id for u in task.units] == ["u-1"]
    assert all("to_director" not in e for e in _events(root, "branch_refused"))
    assert loop.ceiling(task, Caps()) == "max_usd: remaining $0.45 below the $0.50 call floor"


# --- a released plan's reader that leaves a question open ends the tick ---


def test_a_released_plan_s_reader_leaving_no_receipt_ends_the_tick_before_any_other_paid_call(
    accepted: tuple[Root, Task]
):
    """The reader times out: its cap is booked and paid work holds on `unknown_spend`, opened in the gate's own save.
    The tick ends there, so a unit already pending in the session does not run its worker under the unknown spend."""
    root, task = accepted
    task.units.append(Unit("u-3", "write c.txt", [Predicate("file_exists", {"path": "c.txt"})]))
    task.mode, task.held = HOLD, Held(kind=PLAN, units=[_branch_unit()], approved=True)
    save(root, task)
    write_c = {"write": {"c.txt": "c\n"}, "commit": True, "exit_code": 0, "result": "wrote c.txt",
               "total_cost_usd": 0.1}
    runner = ScriptedRunner([], [write_c], [{"timed_out": True}])
    loop.run_tick(task, Caps(), root, runner)
    assert [ScriptedRunner.seat(argv) for argv, _, _, _ in runner.calls] == ["reader"]  # no worker after it
    (asked,) = state.open_questions(task)
    assert asked.reason == "unknown_spend" and task.held is not None and task.held.approved
    assert [(u.id, u.attempts) for u in task.units] == [("u-1", 0), ("u-3", 0)] and task.tick == 0


# --- F4: a think during the admission call wins over the approve ---


def test_a_think_during_a_released_plans_admission_call_returns_it_to_the_thinker(accepted: tuple[Root, Task]):
    root, task = accepted
    task.mode, task.held = HOLD, Held(kind=PLAN, units=[_branch_unit()], approved=True)
    save(root, task)

    def think() -> None:
        with state.LOCK:
            console.apply(task, Caps(), "think", {"text": "keep stats.py"}, True)

    runner = Effect(ScriptedRunner([_reply({"ask": "rename or keep?"})], [W_RENAME], [YES]), "reader", think)
    loop.run_tick(task, Caps(), root, runner)
    assert [u.id for u in task.units] == ["u-1"] and task.held is None
    (returned,) = _events(root, "returned")
    assert (returned["kind"], returned["units"], returned["thoughts"]) == ("plan", ["u-2"], 1)
    assert _events(root, "approved") == [] and runner.seats() == ["reader", "director"]
    message = json.loads(runner.scripted.calls[-1][1])
    assert message["thoughts"] == ["keep stats.py"] and returned["to_director"] in message["notes"]
    assert director.pending_notes(root, TASK_ID) == []  # delivered with that director call
