"""The brain never trusts the editor's `.git`.

A candidate is graded, re-checked and diffed in a fresh clone of the WORKSPACE at the candidate SHA, never in the
editor's own clone, and every brain-side git command that touches a repository the editor (or a check command) wrote
runs hardened: `core.fsmonitor` off, replace objects ignored, no system config; `git status` there runs inside the
check-command sandbox. Each test is an editor that games its clone's `.git` and must not get its way.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path

import pytest

from protean import loop, state
from protean.dry import make_workspace
from protean.graph.budget import Caps
from protean.runner import RunResult
from protean.state import Predicate, Root, Task, Unit, save, set_current

TASK_ID = "task-20261011T150000"
GOOD = {"kind": "file_contains", "args": {"path": "a.txt", "text": "good"}}


def _git(cwd: Path, *args: str, stdin: str | None = None) -> str:
    return subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t.invalid", *args], cwd=cwd, input=stdin,
                          capture_output=True, text=True, check=True).stdout.strip()


def _commit(clone: Path, files: dict[str, str]) -> str:
    for rel, text in files.items():
        (clone / rel).write_text(text)
    _git(clone, "add", "-A")
    _git(clone, "commit", "-qm", "work")
    return _git(clone, "rev-parse", "HEAD")


def _acting(action):
    """An editor that does `action` in its clone and reports success with a receipt."""
    def runner(argv, stdin, cwd, timeout):
        action(Path(cwd))
        return RunResult(0, json.dumps({"type": "result", "is_error": False, "result": "done\nexit_code: 0",
                                        "total_cost_usd": 0.5}), "")
    return runner


@pytest.fixture
def env(tmp_path: Path, root: Root) -> tuple[Root, Path]:
    root.path.mkdir(parents=True)
    root.thinker_prompt.write_text("m\n")
    root.editor_prompt.write_text("e\n")
    return root, make_workspace(tmp_path / "ws")


def _tick(env: tuple[Root, Path], predicates: list[dict], action, accepted: str | None = None) -> Task:
    root, workspace = env
    unit = Unit("u-1", "write a.txt", [Predicate(p["kind"], p["args"], retain=False) for p in predicates])
    task = Task(id=TASK_ID, goal="g", workspace=str(workspace), units=[unit], accepted=accepted)
    save(root, task)
    set_current(root, task.id)
    loop.run_tick(task, Caps(), root, _acting(action))
    return task


def _verdict(root: Root, task: Task) -> dict:
    (verdict,) = [e for e in state.read_log(root, task.id) if e["event"] == "verdict"]
    return verdict


def test_a_replaced_object_in_the_editors_clone_never_passes_the_real_commit(env: tuple[Root, Path]):
    """`git replace` in the clone: the real commit's a.txt says "bad", a substitute blob says "good"."""
    root, workspace = env
    real: list[str] = []

    def substitute(clone: Path) -> None:
        real.append(_commit(clone, {"a.txt": "bad\n"}))
        bad = _git(clone, "rev-parse", "HEAD:a.txt")
        good = _git(clone, "hash-object", "-w", "--stdin", stdin="good\n")
        _git(clone, "replace", bad, good)
        assert _git(clone, "cat-file", "-p", "HEAD:a.txt") == "good"  # what an unhardened reader of the clone sees

    task = _tick(env, [GOOD], substitute)
    (unit,), verdict = task.units, _verdict(root, task)
    assert (unit.status, unit.integrated, unit.candidate) == ("pending", False, None)
    assert verdict["passed"] is False and verdict["commit"] == real[0]
    assert verdict["notes"]["p1"] == "file_contains: a.txt does not contain 'good'"
    assert task.accepted != real[0] and _git(workspace, "rev-parse", f"task/{TASK_ID}") != real[0]


def test_an_fsmonitor_hook_in_the_editors_clone_never_runs(env: tuple[Root, Path], tmp_path: Path):
    marker = tmp_path / "fsmonitor-ran"
    hook = tmp_path / "fsmonitor-hook"
    hook.write_text(f"#!/bin/sh\ntouch '{marker}'\nprintf '/\\0'\n")
    hook.chmod(0o755)

    def plant(clone: Path) -> None:
        _commit(clone, {"a.txt": "good\n"})
        _git(clone, "config", "core.fsmonitor", str(hook))

    task = _tick(env, [GOOD], plant)
    assert not marker.exists()
    assert task.units[0].status == "passed" and task.units[0].integrated  # the editor's real work still lands


def test_an_ignored_file_in_the_editors_clone_is_absent_where_a_check_command_runs(env: tuple[Root, Path]):
    """An ignored `.venv` file (a `.pth` or `sitecustomize` would run inside every `uv run`) never reaches a check."""
    root, _ = env
    check = {"kind": "command", "args": {"cmd": "pwd; test -f .venv/cheat"}}

    def plant(clone: Path) -> None:
        _commit(clone, {"a.txt": "good\n"})
        (clone / ".venv").mkdir()
        (clone / ".venv" / "cheat").write_text("only in the editor's clone\n")

    task = _tick(env, [GOOD, check], plant)
    verdict = _verdict(root, task)
    editor_clone = root.clones_dir(TASK_ID).resolve() / "u-1"
    assert verdict["passed"] is False and verdict["failed_ids"] == ["p2"] and task.units[0].integrated is False
    assert verdict["notes"]["p2"].startswith("command: exit 1, expected 0: ")
    assert str(editor_clone) not in verdict["notes"]["p2"]  # the command ran elsewhere: the grading clone


def test_a_clean_filter_in_the_editors_clone_runs_confined_and_cannot_read_the_console_token(
    env: tuple[Root, Path], tmp_path: Path
):
    root, _ = env
    secret, leaked = root.console_file(TASK_ID), tmp_path / "leaked"
    secret.parent.mkdir(parents=True, exist_ok=True)
    secret.write_text('{"token": "t0ken"}')

    def plant(clone: Path) -> None:
        _commit(clone, {"a.txt": "good\n"})
        _git(clone, "config", "filter.leak.clean", f"sh -c 'cat \"{secret}\" > \"{leaked}\"; cat'")
        Path(clone, ".git", "info", "attributes").write_text("* filter=leak\n")
        later = time.time() + 5
        os.utime(clone / "a.txt", (later, later))  # stat-dirty: `git status` re-reads it through the filter

    task = _tick(env, [GOOD], plant)
    assert "t0ken" not in (leaked.read_text() if leaked.exists() else "")
    assert task.units[0].status == "passed"


def test_a_tag_the_editor_made_is_not_fetched_into_the_workspace(env: tuple[Root, Path]):
    _, workspace = env

    def tag(clone: Path) -> None:
        _commit(clone, {"a.txt": "good\n"})
        _git(clone, "tag", "v-editor")
        _git(clone, "tag", "-a", "-m", "annotated", "v-editor-annotated")

    task = _tick(env, [GOOD], tag)
    assert task.units[0].integrated and _git(workspace, "tag", "--list") == ""


def test_a_graft_in_the_editors_clone_cannot_fake_extending_the_accepted_tip(env: tuple[Root, Path]):
    """`info/grafts` survives GIT_NO_REPLACE_OBJECTS: ancestry is read in the workspace, never in the clone."""
    root, workspace = env
    _git(workspace, "checkout", "-q", "-b", f"task/{TASK_ID}")
    tip = _commit(workspace, {"keep.txt": "accepted\n"})
    _git(workspace, "checkout", "-q", "main")
    orphan: list[str] = []

    def rewrite(clone: Path) -> None:
        (clone / "a.txt").write_text("good\n")
        _git(clone, "add", "-A")
        tree = _git(clone, "write-tree")
        orphan.append(_git(clone, "commit-tree", tree, "-m", "no parent"))
        _git(clone, "reset", "-q", "--hard", orphan[0])
        Path(clone, ".git", "info", "grafts").write_text(f"{orphan[0]} {tip}\n")

    task = _tick(env, [GOOD], rewrite, accepted=tip)
    verdict = _verdict(root, task)
    assert verdict["delivered"] is False and task.units[0].integrated is False
    assert f"does not extend the accepted tip {tip[:12]}" in verdict["notes"]["p1"]
    assert task.accepted == tip == _git(workspace, "rev-parse", f"task/{TASK_ID}")
