"""Worker: a clone on unit/<task>/<unit>, the branch fetched back and never merged, delivery, the exit-code rule."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from pathlib import Path

import pytest

from protean.graph import worker
from protean.graph.budget import Caps
from protean.dry import ScriptedRunner, make_workspace
from protean.runner import RunResult, real_runner
from protean.state import Predicate, Root, Task, Unit

WRITE_A = {"write": {"a.txt": "first\n"}, "commit": True, "exit_code": 0, "result": "wrote a.txt", "total_cost_usd": 0.75}


def _git(cwd: Path, *args: str) -> str:
    return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, check=True).stdout.strip()


@pytest.fixture
def ready(tmp_path: Path, root: Root, task: Task) -> tuple[Root, Task]:
    root.path.mkdir(parents=True)
    root.editor_prompt.write_text("# editor prompt\n")
    task.workspace = str(make_workspace(tmp_path / "ws"))
    return root, task


def test_the_editor_may_run_its_own_units_check_commands(ready: tuple[Root, Task]):
    """Run 6 (2026-10-16): the grant allowed only `uv` and a few `git` verbs, so `bash tests/x.sh` was denied in
    dontAsk mode and an honest editor could not verify; the unit's own command predicates are granted as prefixes."""
    root, task = ready
    unit = Unit(id="u2", intent="make the acks tiered", predicates=[
        Predicate(kind="command", args={"cmd": "bash -n bin/turn.sh", "expect_exit": 0}, retain=False),
        Predicate(kind="command", args={"cmd": "bash tests/test_turn_acks.sh", "expect_exit": 0}, protected=["tests"]),
        Predicate(kind="file_contains", args={"path": "bin/turn.sh", "text": "EA_ACK_S:-"}),
        Predicate(kind="command", args={"cmd": "bash -n bin/turn.sh", "expect_exit": 0}, retain=False),
    ])
    task.units.append(unit)
    runner = ScriptedRunner([], [WRITE_A])
    worker.run_unit(task, unit, Caps(), root, runner)
    argv = runner.calls[0][0]
    grant = argv[argv.index("--allowedTools") + 1:argv.index("--system-prompt")]
    assert grant == [*worker.TOOLS, "Bash(bash -n bin/turn.sh:*)", "Bash(bash tests/test_turn_acks.sh:*)"]


def test_the_clone_works_on_unit_branch_and_the_branch_lands_unmerged(ready: tuple[Root, Task]):
    root, task = ready
    workspace = Path(task.workspace)
    head = _git(workspace, "rev-parse", "HEAD")
    runner = ScriptedRunner([], [WRITE_A])
    res = worker.run_unit(task, task.units[0], Caps(), root, runner)
    argv, stdin, cwd, timeout = runner.calls[0]
    assert argv[:2] == [worker.sandbox_binary(), "-p"]
    assert argv[2] == worker.sandbox_profile(res.clone, root.console_file(task.id))
    assert argv[3:] == [
        "claude", "-p", "--setting-sources", "", "--output-format", "stream-json", "--verbose", "--max-budget-usd", "4.00",
        "--model", "claude-opus-5",
        "--permission-mode", "dontAsk", "--allowedTools", "Read", "Edit", "Write", "Grep", "Glob", "Bash(uv:*)",
        "Bash(git status:*)", "Bash(git diff:*)", "Bash(git add:*)", "Bash(git commit:*)", "Bash(git checkout:*)",
        "Bash(git branch:*)", "Bash(git rm:*)", "--system-prompt", "# editor prompt\n",
    ]
    assert cwd == res.clone == root.clones_dir(task.id) / "u1" and timeout == 900
    assert stdin.startswith("Unit u1: write a.txt") and 'p1 file_exists {"path": "a.txt"}' in stdin
    branch = f"unit/{task.id}/u1"
    assert _git(res.clone, "branch", "--show-current") == branch
    assert (res.exit_code, res.cost_usd, res.branch, res.changed) == (0, 0.75, branch, ["a.txt"])
    assert (res.delivered, res.delivery_note) == (True, "")
    assert _git(workspace, "show", f"{branch}:a.txt") == "first"
    assert _git(workspace, "rev-parse", "HEAD") == head and _git(workspace, "branch", "--show-current") == "main"
    assert not (workspace / "a.txt").exists()


def test_a_second_attempt_starts_fresh_and_replaces_the_fetched_branch(ready: tuple[Root, Task]):
    root, task = ready
    second = {**WRITE_A, "write": {"b.txt": "second\n"}}
    runner = ScriptedRunner([], [WRITE_A, second])
    worker.run_unit(task, task.units[0], Caps(), root, runner)
    res = worker.run_unit(task, task.units[0], Caps(), root, runner)
    assert res.changed == ["b.txt"] and not (res.clone / "a.txt").exists()
    assert _git(Path(task.workspace), "show", f"unit/{task.id}/u1:b.txt") == "second"


def test_the_clone_has_no_remote_and_the_branch_still_lands(ready: tuple[Root, Task]):
    root, task = ready
    res = worker.run_unit(task, task.units[0], Caps(), root, ScriptedRunner([], [WRITE_A]))
    assert _git(res.clone, "remote") == ""
    assert _git(Path(task.workspace), "show", f"{res.branch}:a.txt") == "first"


def test_two_tasks_with_the_same_unit_id_keep_separate_branches(ready: tuple[Root, Task]):
    root, task = ready
    later = Task(id="task-20260927T130000", goal="g", workspace=task.workspace,
                 units=[Unit("u1", "write a.txt", [Predicate("file_exists", {"path": "a.txt"})])])
    worker.run_unit(task, task.units[0], Caps(), root, ScriptedRunner([], [WRITE_A]))
    rewrite = {**WRITE_A, "write": {"a.txt": "later\n"}}
    worker.run_unit(later, later.units[0], Caps(), root, ScriptedRunner([], [rewrite]))
    workspace = Path(task.workspace)
    assert _git(workspace, "branch", "--list", "unit/*").split() == [f"unit/{task.id}/u1", f"unit/{later.id}/u1"]
    assert _git(workspace, "show", f"unit/{task.id}/u1:a.txt") == "first"
    assert _git(workspace, "show", f"unit/{later.id}/u1:a.txt") == "later"


def _acting(action) -> object:
    """A worker that does `action` in its clone and reports success."""
    def runner(argv, stdin, cwd, timeout):
        action(Path(cwd))
        return RunResult(0, json.dumps({"type": "result", "is_error": False, "result": "done\nexit_code: 0",
                                        "total_cost_usd": 0.5}), "")
    return runner


def _write_only(clone: Path) -> None:
    (clone / "a.txt").write_text("A\n")


def _commit_then_leave(clone: Path) -> None:
    _write_only(clone)
    _git(clone, "add", "-A")
    _git(clone, "-c", "user.name=t", "-c", "user.email=t@t.invalid", "commit", "-qm", "a")
    _git(clone, "checkout", "-q", "main")


def _empty_commit(clone: Path) -> None:
    _git(clone, "-c", "user.name=t", "-c", "user.email=t@t.invalid", "commit", "-q", "--allow-empty", "-m", "nothing")


@pytest.mark.parametrize("action, block, note", [
    (_write_only, False, "uncommitted changes in the clone: a.txt"),
    (_commit_then_leave, False, "clone is on branch main, not unit/task-20260927T120000/u1"),
    (lambda clone: None, True, "fetch into the workspace failed: "),
    (lambda clone: None, False, "no change on the branch: nothing was delivered"),
    (lambda clone: (clone / "docs").mkdir(), False, "no change on the branch: nothing was delivered"),
    (_empty_commit, False, "no change on the branch: nothing was delivered"),
])
def test_delivery_is_checked_after_the_worker_run(ready: tuple[Root, Task], action, block: bool, note: str):
    root, task = ready
    if block:
        _git(Path(task.workspace), "branch", f"unit/{task.id}/u1/blocker")
    res = worker.run_unit(task, task.units[0], Caps(), root, _acting(action))
    assert res.delivered is False and note in res.delivery_note and res.cost_usd == 0.5


def test_the_call_cap_is_clipped_to_what_the_task_has_left(ready: tuple[Root, Task]):
    root, task = ready
    task.cost_usd = 23.5
    runner = ScriptedRunner([], [WRITE_A])
    worker.run_unit(task, task.units[0], Caps(), root, runner)
    argv = runner.calls[0][0]
    assert argv[argv.index("--max-budget-usd") + 1] == "1.50"


def test_a_timeout_reports_no_exit_code_and_is_charged_its_cap(ready: tuple[Root, Task]):
    root, task = ready
    res = worker.run_unit(task, task.units[0], Caps(), root, lambda *_: RunResult(-9, "", "", True))
    assert (res.exit_code, res.cost_usd, res.changed) == (None, 4.0, []) and "timed out after 900s" in res.summary
    task.cost_usd = 23.5
    res = worker.run_unit(task, task.units[0], Caps(), root, lambda *_: RunResult(-9, "", "", True))
    assert res.cost_usd == 1.5


def test_an_interrupt_after_the_call_returned_books_the_real_cost(ready: tuple[Root, Task], monkeypatch: pytest.MonkeyPatch):
    root, task = ready
    real_git, charged = worker.git, []
    def git_then_interrupt(args, cwd):
        if args[0] == "fetch":
            raise KeyboardInterrupt
        return real_git(args, cwd)
    monkeypatch.setattr(worker, "git", git_then_interrupt)
    with pytest.raises(KeyboardInterrupt):
        worker.run_unit(task, task.units[0], Caps(), root, ScriptedRunner([], [WRITE_A]), charged.append)
    assert charged == [0.75]


def _receipt(result: str, **extra: object) -> str:
    return json.dumps({"type": "result", "is_error": False, "result": result, "total_cost_usd": 1.5, **extra})


@pytest.mark.parametrize("res, exit_code, cost", [
    (RunResult(0, _receipt("fixed it\nexit_code: 1"), ""), 1, 1.5),
    (RunResult(0, _receipt("all green\n`exit_code: 0`\n"), ""), 0, 1.5),
    (RunResult(0, _receipt("exit_code: 0 was the plan, but I stopped"), ""), None, 1.5),
    (RunResult(0, _receipt("done, all checks green"), ""), None, 1.5),
    (RunResult(3, _receipt("no line at all"), ""), None, 1.5),
    (RunResult(1, _receipt("", is_error=True, subtype="error_max_budget_usd"), ""), None, 1.5),
    (RunResult(0, "not json", "boom"), None, 4.0),
    (RunResult(-9, "", "", True), None, 4.0),
])
def test_only_the_workers_own_line_is_exit_code_evidence(res: RunResult, exit_code: int | None, cost: float):
    code, dollars, _ = worker._outcome(res, 900, 4.0)
    assert (code, dollars) == (exit_code, cost)


def test_real_runner_passes_stdin_and_kills_the_whole_group_on_timeout(tmp_path: Path, reaped):
    ok = real_runner(["sh", "-c", "cat; echo err >&2; exit 3"], "hello", tmp_path, 10)
    assert (ok.returncode, ok.stdout, ok.stderr, ok.timed_out) == (3, "hello", "err\n", False)
    started = time.monotonic()
    res = real_runner(["sh", "-c", "(sleep 30; touch late) & echo $! > kid; sleep 30"], None, tmp_path, 0.5)
    assert res.timed_out and time.monotonic() - started < 5
    assert reaped(tmp_path / "kid") and not (tmp_path / "late").exists()  # the leader's child died with its group


def _build_output(clone: Path) -> None:
    """Commit one source file, then leave build output lying around the clone."""
    (clone / "src.py").write_text("x = 1\n")
    _git(clone, "add", "src.py")
    _git(clone, "-c", "user.name=t", "-c", "user.email=t@t.invalid", "commit", "-qm", "src")
    for rel in (".venv/x", "__pycache__/y", "pkg/__pycache__/m.pyc", "node_modules/z/i.js", ".DS_Store"):
        (clone / rel).parent.mkdir(parents=True, exist_ok=True)
        (clone / rel).write_text("built\n")


def _forgot_a_source_file(clone: Path) -> None:
    _build_output(clone)
    (clone / "new.py").write_text("x = 1\n")


def test_build_output_is_excluded_but_a_forgotten_source_file_is_not(ready: tuple[Root, Task]):
    root, task = ready
    res = worker.run_unit(task, task.units[0], Caps(), root, _acting(_build_output))
    assert (res.delivered, res.delivery_note) == (True, "")
    res = worker.run_unit(task, task.units[0], Caps(), root, _acting(_forgot_a_source_file))
    assert (res.delivered, res.delivery_note) == (False, "uncommitted changes in the clone: new.py")


def test_real_runner_kills_the_group_when_interrupted(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    started = []
    def interrupted(self, *args, **kwargs):
        started.append(self)
        raise KeyboardInterrupt
    monkeypatch.setattr(subprocess.Popen, "communicate", interrupted)
    with pytest.raises(KeyboardInterrupt):
        real_runner(["sh", "-c", "(sleep 1; touch late) & sleep 30"], None, tmp_path, 10)
    (proc,) = started
    assert proc.poll() is not None
    time.sleep(1.5)
    assert not (tmp_path / "late").exists()


# --- write confinement: the editor process runs under sandbox-exec, write-denied outside named trees ---


def _confined(clone: Path, script: str) -> subprocess.CompletedProcess:
    argv = ["sandbox-exec", "-p", worker.sandbox_profile(clone), "/bin/sh", "-c", script]
    return subprocess.run(argv, cwd=clone, capture_output=True, text=True)


@pytest.fixture
def clone(tmp_path: Path) -> Path:
    return make_workspace(tmp_path / "clone")


@pytest.fixture
def outside_home() -> Path:
    """A path under the home tree, outside every allowed tree; removed afterwards if a write ever lands."""
    probe = Path.home() / f".protean-confinement-probe-{os.getpid()}"
    yield probe
    if probe.is_symlink() or probe.is_file():
        probe.unlink()
    elif probe.is_dir():
        shutil.rmtree(probe)


def test_the_profile_denies_a_write_outside_the_clone(clone: Path, outside_home: Path):
    proc = _confined(clone, f"echo escaped > {outside_home}")
    assert proc.returncode != 0 and not outside_home.exists()
    assert "not permitted" in (proc.stderr + proc.stdout).lower()


def test_the_profile_denies_a_write_through_a_symlink_that_leaves_the_clone(clone: Path, outside_home: Path):
    outside_home.mkdir()
    (clone / "door").symlink_to(outside_home)
    proc = _confined(clone, "echo escaped > door/escape.txt")
    assert proc.returncode != 0 and not (outside_home / "escape.txt").exists()


def test_the_profile_allows_a_write_in_the_temp_root_the_tools_need(clone: Path, tmp_path: Path):
    proc = _confined(clone, f"echo scratch > {tmp_path / 'scratch.txt'}")
    assert proc.returncode == 0 and (tmp_path / "scratch.txt").exists()


def test_the_profile_allows_edits_deletion_a_test_run_and_a_commit_inside_the_clone(clone: Path):
    script = ("echo new > a.txt && git rm -q README.md && git add -A && "
              "git -c user.name=t -c user.email=t@t.invalid commit -qm confined && "
              "uv run --no-project python -c 'open(\"b.txt\", \"w\").write(\"ran\")'")
    proc = _confined(clone, script)
    assert proc.returncode == 0, proc.stderr
    assert (clone / "a.txt").exists() and not (clone / "README.md").exists() and (clone / "b.txt").read_text() == "ran"
    assert _git(clone, "log", "--format=%s", "-1") == "confined"


def test_the_profile_denies_writes_outside_the_clone_but_allows_the_named_caches(clone: Path):
    profile = worker.sandbox_profile(clone)
    assert profile.startswith("(version 1)(allow default)(deny file-write*)(allow file-write* ")
    assert "(deny network*)" not in profile
    assert f'(subpath "{clone.resolve()}")' in profile and '(subpath "/private/tmp")' in profile


def test_the_editor_call_is_wrapped_in_the_sandbox_and_refused_without_it(ready: tuple[Root, Task], monkeypatch: pytest.MonkeyPatch):
    root, task = ready
    runner = ScriptedRunner([], [WRITE_A])
    res = worker.run_unit(task, task.units[0], Caps(), root, runner)
    (argv, _, _, _) = runner.calls[0]
    assert argv[0].endswith("sandbox-exec") and argv[1] == "-p"
    assert argv[2] == worker.sandbox_profile(res.clone, root.console_file(task.id))
    assert argv[3:5] == ["claude", "-p"]
    monkeypatch.setattr(worker.shutil, "which", lambda name: None)
    second = ScriptedRunner([], [WRITE_A])
    with pytest.raises(worker.WorkerError, match="sandbox-exec"):
        worker.run_unit(task, task.units[0], Caps(), root, second)
    assert second.calls == []
