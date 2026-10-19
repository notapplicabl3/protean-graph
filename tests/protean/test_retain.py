"""Retention: every predicate is retained unless the director marks it `"retain": false`.

Every predicate, retained or not, must pass its own unit's grade. Once the unit is accepted, only its retained
predicates join the re-check set every later candidate is graded against, so a delivery-only predicate is never
re-checked and can never produce `broke <unit> <pid>`. Seats and clients are shown `"retain": false` only on a
delivery-only predicate; absent means retained, as in a plan.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from protean import loop
from protean.dry import FIXTURES, ScriptedRunner, make_workspace, read_jsonl
from protean.graph import console, director, worker
from protean.graph.budget import Caps
from protean.state import Predicate, Root, Task, Unit, load

KEEP = Predicate("file_contains", {"path": "keep.txt", "text": "keep"})
SCAFFOLD = Predicate("file_exists", {"path": "scaffold.txt"}, retain=False)


@pytest.fixture
def env(tmp_path: Path, root: Root) -> tuple[Root, Path]:
    root.path.mkdir(parents=True)
    root.thinker_prompt.write_text("m\n")
    root.editor_prompt.write_text("e\n")
    return root, make_workspace(tmp_path / "ws")


def _git(cwd: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True)


def test_a_delivery_only_predicate_is_not_rechecked_after_acceptance_and_a_retained_one_is(env: tuple[Root, Path]):
    root, workspace = env
    folder = FIXTURES / "deps-delivery-only"
    runner = ScriptedRunner(read_jsonl(folder / "director.jsonl"), read_jsonl(folder / "worker.jsonl"))
    task = loop.run("deps-delivery-only", workspace, root, Caps(), runner)
    assert (task.terminal, task.awaiting_thought) == ("interrupted", True)
    assert [(u.id, u.status, u.integrated) for u in task.units] == [
        ("u-1", "passed", True), ("u-2", "passed", True), ("u-3", "passed", True)]
    assert load(root, task.id).units[0].predicates == [KEEP, SCAFFOLD]  # the flag is persisted as decoded
    verdicts = [e for e in director.read_log(root, task.id) if e["event"] == "verdict"]
    seen = [(v["unit"], v["attempt"], v["passed"], v["failed_ids"], v.get("broke"),
             (v.get("recheck") or {}).get("predicates")) for v in verdicts]
    assert seen == [
        # u-1's delivery-only p2 is missing: its own grade still rejects it
        ("u-1", 1, False, ["p2"], None, None),
        ("u-1", 2, True, [], None, []),
        # u-2 removes what u-1's p2 checked: accepted, and only u-1's retained p1 was re-checked
        ("u-2", 1, True, [], None, [{"unit": "u-1", "id": "p1", "result": "passed"}]),
        # u-3 rewrites what u-1's retained p1 checked: broke u-1 p1, as before retention
        ("u-3", 1, False, [], {"u-1 p1": "file_contains: keep.txt does not contain 'keep'"},
         [{"unit": "u-1", "id": "p1", "result": "failed"}, {"unit": "u-2", "id": "p1", "result": "passed"}]),
        ("u-3", 2, True, [], None,
         [{"unit": "u-1", "id": "p1", "result": "passed"}, {"unit": "u-2", "id": "p1", "result": "passed"}]),
    ]
    assert _git(workspace, "cat-file", "-e", f"task/{task.id}:scaffold.txt").returncode != 0
    assert _git(workspace, "cat-file", "-e", f"task/{task.id}:keep.txt").returncode == 0
    editor_inputs = [stdin for argv, stdin, _, _ in runner.calls if "Write" in argv]
    assert '- p2 file_exists {"path": "scaffold.txt"} (delivery-only)\n' in editor_inputs[0]
    assert '- p2 file_exists {"path": "scaffold.txt"} (delivery-only): failed — ' in editor_inputs[1]


def test_the_editor_message_marks_a_delivery_only_predicate_and_leaves_a_retained_one_unmarked():
    unit = Unit("u-1", "write keep.txt and a scaffold", [KEEP, SCAFFOLD])
    assert worker.message(unit) == (
        "Unit u-1: write keep.txt and a scaffold\n"
        "\n"
        "When you finish, a grader checks these in this directory:\n"
        '- p1 file_contains {"path": "keep.txt", "text": "keep"}\n'
        '- p2 file_exists {"path": "scaffold.txt"} (delivery-only)\n')


def test_the_snapshot_shows_each_units_predicates_with_retain_only_when_false():
    task = Task("t", "g", "/w", units=[Unit("u-1", "write keep.txt and a scaffold", [KEEP, SCAFFOLD])])
    (unit,) = console.snapshot(task, Caps(), live=False)["units"]
    assert unit["predicates"] == [{"kind": "file_contains", "args": {"path": "keep.txt", "text": "keep"}},
                                  {"kind": "file_exists", "args": {"path": "scaffold.txt"}, "retain": False}]
