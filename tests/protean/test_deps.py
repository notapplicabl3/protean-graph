"""Unit needs, the accepted task branch, re-checking accepted work, and the two-step pass."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from protean import loop
from protean.graph import director, worker
from protean.graph.budget import Caps
from protean.dry import FIXTURES, ScriptedRunner, make_workspace, read_jsonl
from protean.runner import RunResult
from protean.state import Predicate, Question, Root, Task, Unit, load, open_questions, save, set_current

TASK_ID = "task-20260927T120000"


def _git(cwd: Path, *args: str) -> str:
    return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, check=True).stdout.strip()


def _commit(cwd: Path, *args: str) -> None:
    _git(cwd, "-c", "user.name=t", "-c", "user.email=t@t.invalid", *args)


@pytest.fixture
def env(tmp_path: Path, root: Root) -> tuple[Root, Path]:
    root.path.mkdir(parents=True)
    root.thinker_prompt.write_text("m\n")
    root.editor_prompt.write_text("e\n")
    return root, make_workspace(tmp_path / "ws")


def _scenario(name: str) -> ScriptedRunner:
    return ScriptedRunner(read_jsonl(FIXTURES / name / "director.jsonl"), read_jsonl(FIXTURES / name / "worker.jsonl"))


def _worker_units(runner: ScriptedRunner) -> list[str]:
    return [stdin.split(":")[0].removeprefix("Unit ") for argv, stdin, _, _ in runner.calls if "Write" in argv]


def test_deps_runs_the_needed_units_first_and_the_dependent_sees_their_work(env: tuple[Root, Path]):
    root, workspace = env
    main = _git(workspace, "rev-parse", "main")
    runner, lines = _scenario("deps"), []
    task = loop.run("deps", workspace, root, Caps(), runner, echo=lines.append)
    assert (task.terminal, task.awaiting_thought) == ("interrupted", True) and [m.kind for m in task.moves] == [
        "task", "rest"]
    assert [u.id for u in task.units] == ["u-3", "u-1", "u-2"] and task.moves[0].units == ["u-3", "u-1", "u-2"]
    assert _worker_units(runner) == ["u-1", "u-2", "u-3"]
    assert [line.split(": ")[1].split(" ")[0] for line in lines[2:5]] == ["u-1", "u-2", "u-3"]
    assert all((u.status, u.integrated) == ("passed", True) for u in task.units)
    u3 = task.units[0]
    assert task.accepted == u3.candidate == _git(workspace, "rev-parse", f"task/{task.id}")
    assert _git(workspace, "show", f"task/{task.id}:report.txt") == "alpha beta"
    for earlier in task.units[1:]:
        _git(workspace, "merge-base", "--is-ancestor", earlier.candidate, task.accepted)
    assert _git(workspace, "rev-parse", "main") == main and _git(workspace, "branch", "--show-current") == "main"
    integrated = [e["unit"] for e in director.read_log(root, task.id) if e["event"] == "integrated"]
    assert integrated == ["u-1", "u-2", "u-3"]


def test_deps_broken_fails_the_breaking_unit_and_leaves_the_accepted_tip(env: tuple[Root, Path]):
    root, workspace = env
    main = _git(workspace, "rev-parse", "main")
    runner = _scenario("deps-broken")
    task = Task(id=TASK_ID, goal="g", workspace=str(workspace))
    save(root, task)
    set_current(root, task.id)
    loop.run_tick(task, Caps(), root, runner)
    loop.run_tick(task, Caps(), root, runner)
    u1, u2 = task.units
    assert (u1.status, u1.integrated) == ("passed", True)
    before = task.accepted
    assert before == u1.candidate == _git(workspace, "rev-parse", f"task/{task.id}")
    while task.terminal is None and not open_questions(task):
        loop.run_tick(task, Caps(), root, runner)
    assert (u2.status, u2.attempts, u2.integrated, u2.candidate) == ("failed", 2, False, None)
    assert "broke u-1 p1" in u2.summary
    assert task.accepted == before == _git(workspace, "rev-parse", f"task/{task.id}")
    assert _git(workspace, "rev-parse", "main") == main
    verdicts = [e for e in director.read_log(root, task.id) if e["event"] == "verdict" and e["unit"] == "u-2"]
    assert [v["passed"] for v in verdicts] == [False, False]
    assert all(v["passed_ids"] == ["p1", "p2"] and "broke u-1 p1" in v["to_director"] for v in verdicts)
    notes = json.loads(runner.calls[-1][1])["notes"]
    assert len(notes) == 2 and all(note.startswith("u-2 broke") and "broke u-1 p1" in note for note in notes)
    assert task.terminal is None and [(q.asker, q.reason) for q in open_questions(task)] == [("director", "question")]
    assert [m.kind for m in task.moves] == ["task", "ask"] and task.moves[1].question == open_questions(task)[0].id


def test_deps_blocked_blocks_the_dependent_and_tells_the_director(env: tuple[Root, Path]):
    root, workspace = env
    runner = _scenario("deps-blocked")
    task = loop.run("deps-blocked", workspace, root, Caps(), runner)
    assert [(u.id, u.status) for u in task.units] == [("u-1", "failed"), ("u-2", "blocked")]
    assert _worker_units(runner) == ["u-1", "u-1"]
    (event,) = [e for e in director.read_log(root, task.id) if e["event"] == "blocked"]
    assert (event["unit"], event["need"]) == ("u-2", "u-1") and "u-1" in event["to_director"]
    view = json.loads(runner.calls[-1][1])
    assert any("u-2" in note and "u-1" in note for note in view["notes"])
    blocked = view["units"][1]
    assert (blocked["status"], blocked["needs"]) == ("blocked", ["u-1"]) and "u-1" in blocked["summary"]
    assert task.terminal == "interrupted"


def test_blocking_is_transitive_and_follows_a_goal_mode_descope(env: tuple[Root, Path]):
    root, workspace = env
    units = [Unit("u-1", "a", [Predicate("file_exists", {"path": "a"})], "failed", 2),
             Unit("u-2", "b", [Predicate("file_exists", {"path": "b"})], needs=["u-1"]),
             Unit("u-3", "c", [Predicate("file_exists", {"path": "c"})], needs=["u-2"]),
             Unit("u-4", "d", [Predicate("file_exists", {"path": "d"})], "descoped"),
             Unit("u-5", "e", [Predicate("file_exists", {"path": "e"})], needs=["u-4"])]
    task = Task(id=TASK_ID, goal="g", workspace=str(workspace), units=units)
    save(root, task)
    set_current(root, task.id)
    runner = ScriptedRunner([_reply({"rest": True, "state": {"standing": "Three units are blocked.", "open": []}})],
                            [])
    loop.run_tick(task, Caps(), root, runner)
    assert [u.status for u in task.units] == ["failed", "blocked", "blocked", "descoped", "blocked"]
    assert _worker_units(runner) == [] and task.terminal is None
    later = "so it never runs. Give a new unit in a later task if its work is still wanted."
    assert json.loads(runner.calls[0][1])["notes"][:3] == [
        f"u-2 is blocked: it needs u-1, which failed, {later}",
        f"u-3 is blocked: it needs u-2, which is blocked, {later}",
        f"u-5 is blocked: it needs u-4, which was descoped, {later}",
    ]
    assert load(root, task.id).units[4].status == "blocked"


# The kill between the two writes of a pass: the unit is saved `passed` with its candidate, the ref not yet moved.

def _passed_not_integrated(root: Root, workspace: Path, move_ref: bool, **fields: object) -> tuple[Task, str, str]:
    base = _git(workspace, "rev-parse", "HEAD")
    _git(workspace, "update-ref", f"refs/heads/task/{TASK_ID}", base)
    _git(workspace, "checkout", "-q", "-b", f"unit/{TASK_ID}/u-1")
    (workspace / "a.txt").write_text("A\n")
    _git(workspace, "add", "a.txt")
    _commit(workspace, "commit", "-qm", "a")
    candidate = _git(workspace, "rev-parse", "HEAD")
    _git(workspace, "checkout", "-q", "main")
    if move_ref:
        _git(workspace, "update-ref", f"refs/heads/task/{TASK_ID}", candidate, base)
    unit = Unit("u-1", "write a.txt", [Predicate("file_exists", {"path": "a.txt"})], "passed", 1,
                f"unit/{TASK_ID}/u-1", "wrote a.txt [grade: passed]", candidate=candidate)
    task = Task(id=TASK_ID, goal="g", workspace=str(workspace), units=[unit], accepted=base, **fields)
    save(root, task)
    set_current(root, task.id)
    return task, base, candidate


@pytest.mark.parametrize("move_ref", [False, True])
def test_a_tick_first_integrates_a_passed_unit_without_calling_anyone(env: tuple[Root, Path], move_ref: bool):
    root, workspace = env
    task, base, candidate = _passed_not_integrated(root, workspace, move_ref, tick=100)
    runner = ScriptedRunner([], [])
    loop.run_tick(task, Caps(), root, runner)
    assert runner.calls == [] and task.terminal == "stopped"
    saved = load(root, task.id)
    assert (saved.accepted, saved.units[0].integrated, saved.units[0].status) == (candidate, True, "passed")
    assert _git(workspace, "rev-parse", f"task/{TASK_ID}") == candidate != base
    (event,) = [e for e in director.read_log(root, task.id) if e["event"] == "integrated"]
    assert (event["unit"], event["accepted"]) == ("u-1", candidate)


@pytest.mark.parametrize("move_ref", [False, True])
def test_resume_first_integrates_a_passed_unit_without_calling_anyone(env: tuple[Root, Path], move_ref: bool):
    root, workspace = env
    task, _, candidate = _passed_not_integrated(root, workspace, move_ref, terminal="interrupted",
                                                questions=[Question("q1", "director", "q?", "question")])
    runner = ScriptedRunner([], [])
    task = loop.resume(root, Caps(), runner)
    assert runner.calls == [] and task.terminal == "interrupted"
    saved = load(root, task.id)
    assert (saved.accepted, saved.units[0].integrated) == (candidate, True)
    assert _git(workspace, "rev-parse", f"task/{TASK_ID}") == candidate


def test_a_repaired_unit_is_never_worked_again_and_the_session_can_rest(env: tuple[Root, Path]):
    root, workspace = env
    task, _, candidate = _passed_not_integrated(root, workspace, False)
    runner = ScriptedRunner([_reply({"rest": True, "state": {"standing": "a.txt is accepted.", "open": []}})], [])
    task = loop.resume(root, Caps(), runner)
    assert (task.terminal, task.awaiting_thought) == ("interrupted", True)
    assert _worker_units(runner) == [] and task.accepted == candidate


def _moved_elsewhere(root: Root, workspace: Path) -> tuple[Task, str, str, str]:
    task, base, candidate = _passed_not_integrated(root, workspace, False)
    _commit(workspace, "commit", "-q", "--allow-empty", "-m", "elsewhere")
    elsewhere = _git(workspace, "rev-parse", "HEAD")
    _git(workspace, "update-ref", f"refs/heads/task/{TASK_ID}", elsewhere, base)
    return task, base, candidate, elsewhere


# Review fix F2: a tip moved outside the loop parks the task on a question instead of raising.
def test_a_stale_accepted_tip_refuses_the_ref_move(env: tuple[Root, Path]):
    root, workspace = env
    task, base, _, elsewhere = _moved_elsewhere(root, workspace)
    runner = ScriptedRunner([], [])
    loop.run_tick(task, Caps(), root, runner)
    assert runner.calls == [] and task.terminal is None
    assert _git(workspace, "rev-parse", f"task/{TASK_ID}") == elsewhere
    saved = load(root, task.id)
    assert saved.units[0].integrated is False and saved.accepted == base
    (held,) = open_questions(saved)
    assert (held.asker, held.reason, held.context) == ("runtime", "stale_tip", {"stale_tip": "u-1"})
    question = held.text
    assert f"task/{TASK_ID} was moved outside the loop" in question and elsewhere[:12] in question and base in question
    assert "protean resume --abandon" in question
    assert loop.resume(root, Caps(), runner).terminal == "interrupted" and runner.calls == []
    assert load(root, task.id).units[0].integrated is False and len(load(root, task.id).questions) == 1


def test_a_stale_tip_task_can_be_abandoned(env: tuple[Root, Path]):
    root, workspace = env
    task, _, _, elsewhere = _moved_elsewhere(root, workspace)
    loop.run_tick(task, Caps(), root, ScriptedRunner([], []))
    task = loop.resume(root, Caps(), ScriptedRunner([], []), abandon=True)
    assert (task.terminal, task.stop_reason) == ("stopped", "abandoned")
    assert [(q.status, q.disposition) for q in task.questions] == [("superseded", "abandoned")]
    assert _git(workspace, "rev-parse", f"task/{TASK_ID}") == elsewhere


def test_a_restored_tip_lets_resume_integrate_and_continue(env: tuple[Root, Path]):
    root, workspace = env
    task, base, candidate, elsewhere = _moved_elsewhere(root, workspace)
    loop.run_tick(task, Caps(), root, ScriptedRunner([], []))
    _git(workspace, "update-ref", f"refs/heads/task/{TASK_ID}", base, elsewhere)
    runner = ScriptedRunner([_reply({"rest": True, "state": {"standing": "a.txt is accepted.", "open": []}})], [])
    task = loop.resume(root, Caps(), runner)
    assert (task.terminal, task.awaiting_thought) == ("interrupted", True)
    assert _worker_units(runner) == [] and len(runner.calls) == 1
    assert (task.accepted, task.units[0].integrated) == (candidate, True)
    assert _git(workspace, "rev-parse", f"task/{TASK_ID}") == candidate
    assert [(q.status, q.disposition) for q in task.questions] == [("superseded", "repaired")]


# The worker: the accepted branch, branching from its tip, and a candidate that must extend it.

def _acting(action) -> object:
    def runner(argv, stdin, cwd, timeout):
        action(Path(cwd))
        return RunResult(0, json.dumps({"is_error": False, "result": "done\nexit_code: 0", "total_cost_usd": 0.5}), "")
    return runner


def _write_b(clone: Path) -> None:
    (clone / "b.txt").write_text("B\n")
    _git(clone, "add", "b.txt")
    _commit(clone, "commit", "-qm", "b")


def _accepted_side_commit(workspace: Path) -> str:
    """A commit on task/<id> that main does not carry, as if a unit had been accepted."""
    _git(workspace, "checkout", "-q", "-b", f"task/{TASK_ID}")
    (workspace / "a.txt").write_text("A\n")
    _git(workspace, "add", "a.txt")
    _commit(workspace, "commit", "-qm", "accepted a")
    _git(workspace, "checkout", "-q", "main")
    return _git(workspace, "rev-parse", f"task/{TASK_ID}")


def test_the_first_unit_starts_the_accepted_branch_at_the_workspace_head(env: tuple[Root, Path]):
    root, workspace = env
    head = _git(workspace, "rev-parse", "HEAD")
    task = Task(id=TASK_ID, goal="g", workspace=str(workspace),
                units=[Unit("u-1", "write b.txt", [Predicate("file_exists", {"path": "b.txt"})])])
    res = worker.run_unit(task, task.units[0], Caps(), root, _acting(_write_b))
    assert task.accepted == head == _git(workspace, "rev-parse", f"task/{TASK_ID}")
    assert res.delivered and res.candidate == _git(workspace, "rev-parse", res.branch) != head


def test_the_clone_branches_from_the_accepted_tip_not_main(env: tuple[Root, Path]):
    root, workspace = env
    tip = _accepted_side_commit(workspace)
    task = Task(id=TASK_ID, goal="g", workspace=str(workspace), accepted=tip,
                units=[Unit("u-2", "write b.txt", [Predicate("file_exists", {"path": "b.txt"})])])
    seen = []
    def action(clone: Path) -> None:
        seen.append(((clone / "a.txt").exists(), _git(clone, "rev-parse", "HEAD"), _git(clone, "remote")))
        _write_b(clone)
    res = worker.run_unit(task, task.units[0], Caps(), root, _acting(action))
    assert seen == [(True, tip, "")] and res.delivered and res.changed == ["b.txt"]
    _git(workspace, "merge-base", "--is-ancestor", tip, res.candidate)


def test_a_candidate_that_drops_the_accepted_tip_is_not_delivered(env: tuple[Root, Path]):
    root, workspace = env
    tip = _accepted_side_commit(workspace)
    task = Task(id=TASK_ID, goal="g", workspace=str(workspace), accepted=tip,
                units=[Unit("u-2", "write b.txt", [Predicate("file_exists", {"path": "b.txt"})])])
    def rewrite(clone: Path) -> None:
        _git(clone, "reset", "-q", "--hard", "main")
        _write_b(clone)
    res = worker.run_unit(task, task.units[0], Caps(), root, _acting(rewrite))
    assert res.delivered is False and f"does not extend the accepted tip {tip[:12]}" in res.delivery_note


# Review fix F1: a check command that moves HEAD or leaves files behind cannot get a unit accepted.

def _reply(obj: dict) -> dict:
    return {"type": "result", "is_error": False, "session_id": "s", "total_cost_usd": 0.1, "result": json.dumps(obj)}


RESTORE_AND_COMMIT = ("git show HEAD~1:a.txt > a.txt && git add a.txt && "
                      "git -c user.name=t -c user.email=t@t.invalid commit -qm restore\n")
RESTORE_UNCOMMITTED = "git show HEAD~1:a.txt > a.txt\n"


@pytest.mark.parametrize("script, u1_predicates, note", [
    (RESTORE_AND_COMMIT, [{"kind": "file_exists", "args": {"path": "a.txt"}}], "check commands changed the clone: HEAD moved"),
    (RESTORE_UNCOMMITTED, [{"kind": "command", "args": {"cmd": "test -f a.txt"}, "protected": ["a.txt"]}],
     "check commands left the clone dirty"),
])
def test_a_check_command_that_changes_the_clone_does_not_pass_the_unit(env: tuple[Root, Path], script: str,
                                                                      u1_predicates: list, note: str):
    root, workspace = env
    units = [{"id": "u-1", "intent": "write a.txt", "predicates": u1_predicates},
             {"id": "u-2", "intent": "write check.sh", "needs": ["u-1"],
              "predicates": [{"kind": "command", "args": {"cmd": "sh check.sh"}, "protected": ["check.sh"]}]}]
    runner = ScriptedRunner([_reply({"task": units, "state": {"standing": "a.txt, then a check.", "open": []}})], [
        {"write": {"a.txt": "A\n"}, "commit": True, "exit_code": 0, "result": "wrote a.txt", "total_cost_usd": 0.5},
        {"write": {"check.sh": script}, "delete": ["a.txt"], "commit": True, "exit_code": 0, "result": "ok", "total_cost_usd": 0.5},
    ])
    task = Task(id=TASK_ID, goal="g", workspace=str(workspace))
    save(root, task)
    set_current(root, task.id)
    for _ in range(2):
        loop.run_tick(task, Caps(), root, runner)
    before = task.accepted
    assert before == task.units[0].candidate and task.units[0].integrated
    loop.run_tick(task, Caps(), root, runner)
    u2 = task.units[1]
    assert (u2.status, u2.integrated, u2.candidate) == ("pending", False, None)
    assert note in u2.summary
    assert task.accepted == before == _git(workspace, "rev-parse", f"task/{TASK_ID}")
    (verdict,) = [e for e in director.read_log(root, task.id) if e["event"] == "verdict" and e["unit"] == "u-2"]
    assert verdict["passed"] is False and all(note in n for n in verdict["notes"].values())
