"""Loop: the dry scenario's ticks, attempts then failure, delivery, a rest that awaits a thought, ceilings, resume."""

from __future__ import annotations

import fcntl
import itertools
import json
import os
import subprocess
from dataclasses import replace
from pathlib import Path

import pytest

from protean import loop, state
from protean.graph import grade, graph, director, worker
from protean.graph.budget import Caps
from protean.dry import FIXTURES, ScriptedRunner, make_workspace, read_jsonl
from protean.runner import RunResult
from protean.state import (Fired, Firing, Predicate, Question, Root, Task, Unit, load, open_questions, save,
                           set_current)

UNIT = {"id": "u-1", "intent": "write a.txt", "predicates": [
    {"kind": "file_exists", "args": {"path": "a.txt"}}, {"kind": "exit_code", "args": {"code": 0}}]}
PASS = {"write": {"a.txt": "A\n"}, "commit": True, "exit_code": 0, "result": "wrote a.txt", "total_cost_usd": 0.5}
FAIL = {"write": {"b.txt": "wrong file\n"}, "commit": True, "exit_code": 1, "result": "could not do it", "total_cost_usd": 0.5}


STANDINGS = itertools.count(1)


def _reply(obj: dict, cost: float = 0.1) -> dict:
    """A thinker reply; a move without `state` gets one whose standing no earlier reply had."""
    if "state" not in obj and any(kind in obj for kind in state.MOVE_KINDS):
        obj = {**obj, "state": {"standing": f"Standing {next(STANDINGS)}.", "open": []}}
    return {"type": "result", "is_error": False, "session_id": "s-1", "total_cost_usd": cost, "result": json.dumps(obj)}


def _awaits_a_thought(task: Task) -> bool:
    """Where a session that rested ends a run with no client attached: parked, awaiting a thought."""
    return (task.terminal, task.awaiting_thought, task.moves[-1].kind) == ("interrupted", True, "rest")


@pytest.fixture
def env(tmp_path: Path, root: Root) -> tuple[Root, Path]:
    root.path.mkdir(parents=True)
    root.thinker_prompt.write_text("m\n")
    root.editor_prompt.write_text("e\n")
    return root, make_workspace(tmp_path / "ws")


def _task(root: Root, workspace: Path, **fields: object) -> Task:
    task = Task(id="task-20260927T120000", goal="g", workspace=str(workspace), **fields)
    save(root, task)
    set_current(root, task.id)
    return task


def test_the_dry_scenario_rests_after_four_ticks_and_awaits_a_thought(env: tuple[Root, Path]):
    root, workspace = env
    runner = ScriptedRunner(read_jsonl(FIXTURES / "dry/director.jsonl"), read_jsonl(FIXTURES / "dry/worker.jsonl"))
    lines: list[str] = []
    task = loop.run("dry", workspace, root, Caps(), runner, echo=lines.append)
    assert _awaits_a_thought(task) and (task.tick, round(task.cost_usd, 2)) == (4, 1.3)
    assert [m.kind for m in task.moves] == ["task", "rest"]
    assert [(u.id, u.status, u.branch) for u in task.units] == [
        ("u-1", "passed", f"unit/{task.id}/u-1"), ("u-2", "passed", f"unit/{task.id}/u-2")]
    assert [line.split(":")[0] for line in lines[1:5]] == ["tick 1", "tick 2", "tick 3", "tick 4"]
    assert lines[-1].startswith("terminal: interrupted — awaiting a thought after m2") and load(root, task.id) == task
    assert list(root.clones_dir(task.id).iterdir()) == []
    branches = subprocess.run(["git", "branch", "--list", "unit/*"], cwd=workspace, capture_output=True, text=True).stdout
    assert branches.split() == [f"unit/{task.id}/u-1", f"unit/{task.id}/u-2"]
    verdicts = [e for e in director.read_log(root, task.id) if e["event"] == "verdict"]
    assert [v["passed_ids"] for v in verdicts] == [["p1", "p2"], ["p1", "p2"]]


def test_a_failing_unit_stays_pending_then_fails_after_two_attempts(env: tuple[Root, Path]):
    root, workspace = env
    task = _task(root, workspace)
    runner = ScriptedRunner([_reply({"task": [UNIT]})], [FAIL, FAIL])
    seen = []
    for _ in range(3):
        loop.run_tick(task, Caps(), root, runner)
        seen.append((task.units[0].status, task.units[0].attempts))
    assert seen == [("pending", 0), ("pending", 1), ("failed", 2)]
    assert ("[grade: p1 file_exists: the worker reported exit_code 1; nothing was graded; "
            "p2 exit_code: the worker reported exit_code 1; nothing was graded]") in task.units[0].summary


def test_an_uncommitted_write_is_not_delivered_and_cannot_pass(env: tuple[Root, Path], monkeypatch: pytest.MonkeyPatch):
    root, workspace = env
    graded = []
    real_grade = grade.grade
    monkeypatch.setattr(grade, "grade", lambda *a: graded.append(a) or real_grade(*a))
    task = _task(root, workspace)
    uncommitted = {**PASS, "commit": False}
    runner = ScriptedRunner([_reply({"task": [UNIT]}), _reply({"rest": True})], [uncommitted, uncommitted])
    loop.run_tick(task, Caps(), root, runner)
    loop.run_tick(task, Caps(), root, runner)
    assert (task.units[0].status, task.units[0].attempts) == ("pending", 1)
    (verdict,) = [e for e in director.read_log(root, task.id) if e["event"] == "verdict"]
    assert (verdict["passed_ids"], verdict["ungradeable_ids"], verdict["passed"]) == ([], ["p1", "p2"], False)
    assert all("uncommitted changes in the clone: a.txt" in note for note in verdict["notes"].values())
    assert "uncommitted changes in the clone: a.txt" in task.units[0].summary
    loop.run_tick(task, Caps(), root, runner)
    loop.run_tick(task, Caps(), root, runner)
    assert (task.units[0].status, task.terminal, graded) == ("failed", None, [])


def test_a_status_claim_in_the_plan_does_not_pass_the_unit(env: tuple[Root, Path]):
    root, workspace = env
    task = _task(root, workspace)
    runner = ScriptedRunner([_reply({"task": [{**UNIT, "status": "passed"}]})], [FAIL])
    loop.run_tick(task, Caps(), root, runner)
    assert task.units[0].status == "pending"
    loop.run_tick(task, Caps(), root, runner)
    assert (task.units[0].status, task.units[0].attempts, task.terminal) == ("pending", 1, None)


def test_the_task_stops_at_max_usd_before_any_call(env: tuple[Root, Path]):
    root, workspace = env
    task = _task(root, workspace, cost_usd=25.0)
    runner = ScriptedRunner([], [])
    loop.run_tick(task, Caps(), root, runner)
    assert (task.terminal, task.stop_reason, runner.calls) == ("stopped", "max_usd: 25.00 >= 25.0", [])
    assert load(root, task.id).terminal == "stopped"


def test_the_task_stops_at_the_call_floor_without_a_call(env: tuple[Root, Path]):
    root, workspace = env
    task = _task(root, workspace, cost_usd=24.8)
    runner = ScriptedRunner([], [])
    loop.run_tick(task, Caps(), root, runner)
    assert (task.terminal, task.stop_reason, runner.calls) == (
        "stopped", "max_usd: remaining $0.20 below the $0.50 call floor", [])


def _answer(root: Root, text: str) -> None:
    """the operator answers the open question of the parked current task with `protean ctl answer`, no brain live."""
    (asked, *_) = open_questions(state.current(root))
    assert loop.offline_op(root, Caps(), "answer", {"id": asked.id, "text": text})["id"] == asked.id


def test_resume_waits_for_an_answer_then_hands_it_to_the_thinker_once_as_its_answer(env: tuple[Root, Path]):
    root, workspace = env
    runner = ScriptedRunner([_reply({"ask": "which file?"})], [])
    task = loop.run("g", workspace, root, Caps(), runner)
    assert task.terminal == "interrupted" and task.moves[0].kind == "ask" and task.moves[0].question == "q1"
    assert loop.resume(root, Caps(), runner).terminal == "interrupted" and len(runner.calls) == 1
    _answer(root, "use a.txt\n")
    runner.director += [_reply({"task": [UNIT]}), _reply({"rest": True})]
    runner.worker.append(PASS)
    task = loop.resume(root, Caps(), runner)
    assert _awaits_a_thought(task) and open_questions(task) == []
    after = json.loads(runner.calls[1][1])
    assert after["answer"] == {"id": "q1", "question": "which file?", "answer": "use a.txt"}
    assert after["notes"] == [] and "the operator answered" not in runner.calls[1][1]  # the answer key only, never a note too
    assert json.loads(runner.calls[3][1])["notes"] == []
    (asked,) = task.questions
    assert (asked.status, asked.answer, asked.actor, asked.disposition) == ("answered", "use a.txt", "console", "delivered")
    assert task.moves[0].outcome == "answered"
    assert sorted(p.name for p in root.path.iterdir()) == ["editor.md", "state", "thinker.md"]  # no file flow


def test_an_extend_press_is_saved_with_its_resulting_ceiling_while_another_ceiling_still_binds(
    env: tuple[Root, Path]
):
    root, workspace = env
    task = _task(root, workspace, cost_usd=25.0, tick=100, terminal="stopped", stop_reason="max_usd: 25.00 >= 25.0")
    runner = ScriptedRunner([], [])
    task = loop.resume(root, Caps(), runner, extend_usd=5.0)
    assert (task.terminal, task.stop_reason, task.extra_usd, runner.calls) == ("stopped", "max_ticks: 100 >= 100", 5.0,
                                                                                [])
    assert load(root, task.id).extra_usd == 5.0
    (event,) = [e for e in director.read_log(root, task.id) if e["event"] == "extend"]
    assert (event["actor"], event["usd"], event["ticks"], event["max_usd"], event["max_ticks"]) == (
        "console", 5.0, 0, 30.0, 100)


def test_resume_refuses_while_another_process_holds_the_task_lock(env: tuple[Root, Path]):
    root, workspace = env
    task = _task(root, workspace, terminal="interrupted", questions=[Question("q1", "director", "q?", "question")])
    with (root.state_dir(task.id) / "lock").open("w") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        handle.write("4242\n")
        handle.flush()
        with pytest.raises(RuntimeError, match=rf"^task {task.id} is already running \(pid 4242\)$"):
            loop.resume(root, Caps(), ScriptedRunner([], []))
    assert loop.resume(root, Caps(), ScriptedRunner([], [])).terminal == "interrupted"


def test_run_holds_the_task_lock_while_it_drives(env: tuple[Root, Path]):
    root, workspace = env
    scripted, seen = ScriptedRunner([_reply({"ask": "q?"})], []), []
    def runner(argv, stdin, cwd, timeout):
        with (root.state_dir(state.current(root).id) / "lock").open("a") as handle:
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                seen.append("free")
            except BlockingIOError:
                seen.append("held")
        return scripted(argv, stdin, cwd, timeout)
    task = loop.run("g", workspace, root, Caps(), runner)
    assert seen == ["held"] and (root.state_dir(task.id) / "lock").read_text().strip() == str(os.getpid())
    with (root.state_dir(task.id) / "lock").open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)


def test_extend_widens_a_stopped_task_and_abandon_is_final(env: tuple[Root, Path]):
    root, workspace = env
    caps = Caps(max_usd=0.5)
    runner = ScriptedRunner([_reply({"task": [UNIT]}, cost=0.6), _reply({"rest": True})], [PASS])
    task = loop.run("g", workspace, root, caps, runner)
    assert (task.terminal, task.stop_reason) == ("stopped", "max_usd: 0.60 >= 0.5")
    assert loop.resume(root, caps, runner).terminal == "stopped" and len(runner.calls) == 1
    task = loop.resume(root, caps, runner, extend_usd=1.0)
    assert (task.terminal, task.stop_reason) == ("stopped", "max_usd: remaining $0.40 below the $0.50 call floor")
    assert loop.resume(root, caps, runner).terminal == "stopped" and len(runner.calls) == 2
    task = loop.resume(root, caps, runner, extend_usd=1.0)
    assert _awaits_a_thought(task) and (task.extra_usd, round(task.cost_usd, 2)) == (2.0, 1.2)
    _task(root, workspace, terminal="interrupted", questions=[Question("q1", "director", "q?", "question")])
    task = loop.resume(root, caps, runner, abandon=True)
    assert (task.terminal, task.stop_reason) == ("stopped", "abandoned")
    assert [(q.status, q.disposition) for q in load(root, task.id).questions] == [("superseded", "abandoned")]
    assert loop.resume(root, caps, runner, extend_ticks=5).stop_reason == "abandoned"
    assert load(root, task.id).extra_ticks == 0


def test_a_unit_that_changes_nothing_cannot_pass_on_pre_existing_files(env: tuple[Root, Path]):
    root, workspace = env
    task = _task(root, workspace)
    unit = {"id": "u-1", "intent": "write README.md", "predicates": [
        {"kind": "file_exists", "args": {"path": "README.md"}}, {"kind": "exit_code", "args": {"code": 0}}]}
    idle = {"exit_code": 0, "result": "it was already there", "total_cost_usd": 0.5}
    runner = ScriptedRunner([_reply({"task": [unit]})], [idle, idle])
    for _ in range(2):
        loop.run_tick(task, Caps(), root, runner)
    assert (task.units[0].status, task.units[0].attempts) == ("pending", 1)
    (verdict,) = [e for e in director.read_log(root, task.id) if e["event"] == "verdict"]
    assert (verdict["passed_ids"], verdict["ungradeable_ids"], verdict["passed"]) == ([], ["p1", "p2"], False)
    assert all("nothing was delivered" in note for note in verdict["notes"].values())
    loop.run_tick(task, Caps(), root, runner)
    assert task.units[0].status == "failed"


def _interrupt(argv, stdin, cwd, timeout):
    raise KeyboardInterrupt


def test_an_interrupted_director_call_is_charged_and_saved_before_it_propagates(env: tuple[Root, Path]):
    root, workspace = env
    task = _task(root, workspace)
    with pytest.raises(KeyboardInterrupt):
        loop.run_tick(task, Caps(), root, _interrupt)
    saved = load(root, task.id)
    assert (saved.cost_usd, saved.terminal, saved.tick) == (2.0, None, 0)
    (event,) = [e for e in director.read_log(root, task.id) if e["event"] == "interrupted"]
    assert (event["seat"], event["charged_usd"]) == ("director", 2.0)


def test_an_interrupted_retry_charges_the_first_call_too(env: tuple[Root, Path]):
    root, workspace = env
    task = _task(root, workspace)
    replies = [RunResult(0, json.dumps(_reply({"nonsense": 1}, cost=0.3)), "")]
    def runner(argv, stdin, cwd, timeout):
        if replies:
            return replies.pop()
        raise KeyboardInterrupt
    with pytest.raises(KeyboardInterrupt):
        loop.run_tick(task, Caps(), root, runner)
    assert round(load(root, task.id).cost_usd, 2) == 2.3


def test_an_interrupted_worker_call_is_charged_with_its_attempt_and_resume_carries_it(env: tuple[Root, Path]):
    root, workspace = env
    task = _task(root, workspace, units=[Unit("u-1", "write a.txt", [Predicate("file_exists", {"path": "a.txt"})])])
    with pytest.raises(KeyboardInterrupt):
        loop.run_tick(task, Caps(), root, _interrupt)
    saved = load(root, task.id)
    assert (saved.cost_usd, saved.units[0].attempts, saved.units[0].status, saved.terminal) == (4.0, 1, "pending", None)
    runner = ScriptedRunner([_reply({"rest": True})], [PASS])
    assert loop.resume(root, Caps(), runner).terminal == "interrupted" and runner.calls == []  # paid work held
    _answer(root, "yes")
    task = loop.resume(root, Caps(), runner)
    assert _awaits_a_thought(task) and (round(task.cost_usd, 2), task.units[0].attempts) == (4.6, 2)


def test_an_interrupted_worker_call_with_little_left_is_charged_the_clipped_cap(env: tuple[Root, Path]):
    root, workspace = env
    task = _task(root, workspace, cost_usd=23.5,
                 units=[Unit("u-1", "write a.txt", [Predicate("file_exists", {"path": "a.txt"})])])
    with pytest.raises(KeyboardInterrupt):
        loop.run_tick(task, Caps(), root, _interrupt)
    assert load(root, task.id).cost_usd == 25.0


NO_EXIT_UNIT = {"id": "u-1", "intent": "write a.txt", "predicates": [{"kind": "file_exists", "args": {"path": "a.txt"}}]}


def _one_verdict(root: Root, task: Task) -> dict:
    (verdict,) = [e for e in director.read_log(root, task.id) if e["event"] == "verdict"]
    return verdict


def test_a_worker_reporting_exit_code_1_cannot_pass_even_when_every_predicate_holds(env: tuple[Root, Path]):
    root, workspace = env
    task = _task(root, workspace)
    unmet = {**PASS, "exit_code": 1, "result": "wrote a.txt but the intent is unmet"}
    runner = ScriptedRunner([_reply({"task": [NO_EXIT_UNIT]})], [unmet])
    loop.run_tick(task, Caps(), root, runner)
    loop.run_tick(task, Caps(), root, runner)
    assert (task.units[0].status, task.units[0].attempts) == ("pending", 1)
    verdict = _one_verdict(root, task)
    assert (verdict["passed_ids"], verdict["ungradeable_ids"], verdict["passed"]) == ([], ["p1"], False)
    assert verdict["notes"]["p1"] == "file_exists: the worker reported exit_code 1; nothing was graded"


def test_a_worker_with_no_exit_code_line_cannot_pass(env: tuple[Root, Path]):
    root, workspace = env
    task = _task(root, workspace)
    silent = {key: value for key, value in PASS.items() if key != "exit_code"}
    runner = ScriptedRunner([_reply({"task": [NO_EXIT_UNIT]})], [silent])
    loop.run_tick(task, Caps(), root, runner)
    loop.run_tick(task, Caps(), root, runner)
    assert (task.units[0].status, task.units[0].attempts) == ("pending", 1)
    verdict = _one_verdict(root, task)
    assert (verdict["ungradeable_ids"], verdict["passed"]) == (["p1"], False)
    assert verdict["notes"]["p1"] == "file_exists: the worker reported no exit_code line; nothing was graded"


def test_a_deletion_is_delivered_and_graded_in_the_clone(env: tuple[Root, Path]):
    root, workspace = env
    task = _task(root, workspace)
    unit = {"id": "u-1", "intent": "remove README.md", "predicates": [
        {"kind": "command", "args": {"cmd": "test ! -e README.md", "expect_exit": 0}, "protected": ["README.md"]}]}
    removal = {"delete": ["README.md"], "commit": True, "exit_code": 0, "result": "removed it", "total_cost_usd": 0.5}
    runner = ScriptedRunner([_reply({"task": [unit]})], [removal])
    loop.run_tick(task, Caps(), root, runner)
    loop.run_tick(task, Caps(), root, runner)
    assert (task.units[0].status, task.units[0].attempts) == ("passed", 1)
    verdict = _one_verdict(root, task)
    assert (verdict["passed_ids"], verdict["changed"]) == (["p1"], ["README.md"])


def test_exit_code_0_permits_grading_and_never_substitutes_for_it(env: tuple[Root, Path]):
    root, workspace = env
    task = _task(root, workspace)
    wrong = {**PASS, "write": {"b.txt": "not the file asked for\n"}}
    runner = ScriptedRunner([_reply({"task": [NO_EXIT_UNIT]})], [wrong, wrong])
    for _ in range(3):
        loop.run_tick(task, Caps(), root, runner)
    assert (task.units[0].status, task.units[0].attempts) == ("failed", 2)
    first, _ = [e for e in director.read_log(root, task.id) if e["event"] == "verdict"]
    assert (first["failed_ids"], first["ungradeable_ids"], first["passed"]) == (["p1"], [], False)
    assert first["notes"]["p1"] == "file_exists: missing: a.txt"


# The tick order is the registry's (`graph.NODES`), walked by `loop._walk`. These three re-pin, against the registry's
# edges and its node acts, the tests that pinned the retired `loop.TICK` sequence (shape SPEC decision 9; its ceiling
# step is the budget node, its retry step the hold node, its seat the thinker or the worker).
THINKER_TICK = ["repair", "block", "budget", "release", "pick", "hold", "director", "block", "save"]


def _walk(*adds: str) -> list[str]:
    """The node names of the walk the registry's edges give the entries in every tick plus `adds`."""
    return [graph.entries()[key].name for key in graph.order(graph.always() + list(adds))]


def _traced(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Every node act the walker calls, in order, as it calls it."""
    visited: list[str] = []
    acts = {name: (lambda tick, name=name, act=act: visited.append(name) or act(tick)) for name, act in loop.ACTS.items()}
    monkeypatch.setattr(loop, "ACTS", acts)
    return visited


def test_the_registry_s_edges_give_the_record_s_tick_order_and_every_walked_node_has_one_act():
    assert _walk("director") == THINKER_TICK
    assert _walk("worker", "grader", "gate", "ref") == THINKER_TICK[:6] + ["worker", "grader", "gate", "ref", "block",
                                                                         "save"]
    assert set(loop.ACTS) == {node.name for node in graph.NODES} - {"reader"}  # the reader runs inside other nodes
    assert not hasattr(loop, "TICK")


def test_run_tick_walks_the_registry_s_order_until_a_node_ends_the_tick(
    env: tuple[Root, Path], monkeypatch: pytest.MonkeyPatch
):
    root, workspace = env
    visited = _traced(monkeypatch)
    task, runner = _task(root, workspace), ScriptedRunner([_reply({"task": [UNIT]})], [])
    loop.run_tick(task, Caps(), root, runner)
    assert (visited, task.tick, len(runner.calls)) == (THINKER_TICK, 1, 1)
    fired = [e for e in director.read_log(root, task.id) if e["event"] == "fired"]
    assert [(e["tick"], e["fire"], e["walked"]) for e in fired] == [(0, [], THINKER_TICK)]  # the task move is queued
    visited.clear()
    loop.run_tick(task, Caps(max_ticks=1), root, runner)  # the ceiling ends it
    assert (visited, task.terminal, len(runner.calls)) == (THINKER_TICK[:3], "stopped", 1)
    assert task.fired == Fired(tick=1, walked=THINKER_TICK[:3], walk=task.fired.walk) == load(root, task.id).fired
    visited.clear()
    loop.run_tick(task, Caps(), root, runner)  # a terminal task: the first node ends it
    assert visited == THINKER_TICK[:1]
    visited.clear()
    retry = Unit("u-1", "write a.txt", [Predicate("file_exists", {"path": "a.txt"})], "pending", 1)
    task = _task(root, workspace, mode="hold", units=[retry])
    loop.run_tick(task, Caps(), root, runner)  # in hold, the retry is held for approve
    assert (visited, task.held.kind, task.tick, len(runner.calls)) == (THINKER_TICK[:6], "retry", 0, 1)
    assert task.fired.fire == [Firing("worker", unit="u-1")]  # drawn, then held


def test_changing_an_entry_or_one_declared_edge_of_the_registry_changes_what_runs(
    env: tuple[Root, Path], monkeypatch: pytest.MonkeyPatch
):
    root, workspace = env
    declared, acts, visits = graph.NODES, dict(loop.ACTS), []

    def at_the_tick_ceiling() -> tuple:
        task, runner = _task(root, workspace, tick=100), ScriptedRunner([_reply({"ask": "which file?"})], [])
        loop.run_tick(task, Caps(), root, runner)
        return task.terminal, task.stop_reason, task.tick, len(runner.calls)

    stopped = ("stopped", "max_ticks: 100 >= 100", 100, 0)
    assert at_the_tick_ceiling() == stopped
    monkeypatch.setattr(loop, "ACTS", {**acts, "budget": lambda tick: visits.append(tick.task.tick) or False})
    assert at_the_tick_ceiling() == (None, "", 101, 1) and visits == [100]  # the thinker ran past the ceiling
    monkeypatch.setattr(loop, "ACTS", {**acts, "director": lambda tick: setattr(tick, "line", "thinker stubbed") or False})
    task, runner, lines = _task(root, workspace), ScriptedRunner([], []), []
    loop.run_tick(task, Caps(), root, runner, echo=lines.append)
    assert (lines, runner.calls, task.tick) == (["tick 1: thinker stubbed"], [], 1)
    monkeypatch.setattr(loop, "ACTS", acts)
    assert at_the_tick_ceiling() == stopped

    def a_rest_on_a_project() -> tuple[list[str], int]:
        task = _task(root, workspace, project="demo")
        loop.run_tick(task, Caps(), root, ScriptedRunner([_reply({"rest": True})], []))
        remembered = root.memory_file("demo")
        return task.fired.walked, len(remembered.read_text().splitlines()) if remembered.exists() else 0

    rest = THINKER_TICK[:7] + ["rest", "memory", "block", "save"]
    assert a_rest_on_a_project() == (rest, 1)
    unbrought = tuple(replace(node, brings=()) if node.name == "rest" else node for node in declared)
    monkeypatch.setattr(graph, "NODES", unbrought)  # one edge: a rest no longer brings memory's line
    assert a_rest_on_a_project() == ([n for n in rest if n != "memory"], 1)  # no line written
    monkeypatch.setattr(graph, "NODES", declared)
    assert a_rest_on_a_project() == (rest, 2)


def test_a_stop_during_the_drive_parks_the_record_interrupted(env: tuple[Root, Path]):
    """Audit 2026-10-23, D2: the console's `stop`, Ctrl-C, SIGTERM and SIGHUP all leave the drive with KeyboardInterrupt;
    the record used to keep `terminal: null`, which a client reads as a brain that vanished. Now it is `interrupted`, as
    an unattended park is, with a `parked` event, and `protean resume` clears it as it clears a park."""
    root, workspace = env
    task = _task(root, workspace, units=[Unit("u-1", "write a.txt", [Predicate("file_exists", {"path": "a.txt"})])])
    with pytest.raises(KeyboardInterrupt):
        loop.resume(root, Caps(), _interrupt)
    saved = load(root, task.id)
    assert (saved.terminal, saved.stop_reason or None, state.is_closed(saved)) == ("interrupted", None, False)
    assert saved.cost_usd == 4.0  # the cut worker call (the unit is pending) was booked before the mark (`_charger`)
    (parked,) = [e for e in director.read_log(root, task.id) if e["event"] == "parked"]
    assert (parked["hold"], parked["on"]) == ("stop", "the operator")
    interrupted = [e for e in director.read_log(root, task.id) if e["event"] == "interrupted"]
    assert [e["seat"] for e in interrupted] == ["worker"]


def test_a_stop_marks_the_record_on_disk_and_persists_nothing_the_tick_had_not_saved(env: tuple[Root, Path],
                                                                                   monkeypatch: pytest.MonkeyPatch):
    """Audit 2026-10-23, F1: the stop mark saved the loop's in-memory task, so a Stop during a unit's first clone
    persisted `attempts: 1` with no call — on resume a unit that never ran was held as "retry of u-1 (attempt 2)" in
    hold mode and burned as attempt 2, the maximum, in auto. The mark now writes `interrupted` on the record as the
    tick last saved it: the unit still has no attempt and no call."""
    root, workspace = env
    task = _task(root, workspace, units=[Unit("u-1", "write a.txt", [Predicate("file_exists", {"path": "a.txt"})])])

    def stop_during_the_clone(*args: object, **kwargs: object) -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr(worker, "fresh_clone", stop_during_the_clone)
    runner = ScriptedRunner([], [])
    with pytest.raises(KeyboardInterrupt):
        loop.resume(root, Caps(), runner)
    saved = load(root, task.id)
    assert (saved.terminal, saved.units[0].attempts, saved.calls, saved.cost_usd, runner.calls) == (
        "interrupted", 0, [], 0.0, [])
    (parked,) = [e for e in director.read_log(root, task.id) if e["event"] == "parked"]
    assert (parked["hold"], parked["on"]) == ("stop", "the operator")
