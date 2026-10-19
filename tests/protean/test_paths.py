"""Pre-dispatch predicate-path validation: one confinement rule, `grade.confined`, applied by the
director's decode to the workspace and by the grader to the clone it grades.

A `file_exists` or `file_contains` path that is absolute, names `.git`, climbs out with `..`, or leaves the tree once
its symlinks resolve makes the plan invalid: the error names the unit, the predicate and the path, and goes back to
the director on the malformed-plan retry, before any worker call. Not a veto, not a question. `command` predicates
are not path-validated. The grader's own checks stay: it applies the same rule, then reads the branch's tree.
"""

from __future__ import annotations

import itertools
import json
import subprocess
from pathlib import Path

import pytest

from protean import loop
from protean.dry import ScriptedRunner, make_workspace
from protean.graph import director, grade
from protean.graph.budget import Caps
from protean.state import MOVE_KINDS, Predicate, Root, Task, Unit


def _git(cwd: Path, *args: str) -> None:
    subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t.invalid", *args], cwd=cwd, check=True,
                   capture_output=True)


STANDINGS = itertools.count(1)


def _with_state(obj: object) -> object:
    """A thinker move without `state` gets one whose standing no earlier reply had."""
    if isinstance(obj, dict) and "state" not in obj and any(kind in obj for kind in MOVE_KINDS):
        return {**obj, "state": {"standing": f"Standing {next(STANDINGS)}.", "open": []}}
    return obj


def _reply(obj: dict) -> dict:
    obj = _with_state(obj)
    return {"type": "result", "is_error": False, "session_id": "s", "total_cost_usd": 0.1, "result": json.dumps(obj)}


@pytest.fixture
def tree(tmp_path: Path) -> tuple[Path, Path]:
    """A repository holding notes.md, docs/x.txt and three committed symlinks: `door` to a directory outside it,
    `link.txt` to a file outside it, `inner` to its own docs/; and the outside directory."""
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "x.txt").write_text("SECRET\n")
    repo = make_workspace(tmp_path / "ws")
    (repo / "notes.md").write_text("pins\n")
    (repo / "docs").mkdir()
    (repo / "docs" / "x.txt").write_text("x\n")
    (repo / "door").symlink_to(outside)
    (repo / "link.txt").symlink_to(outside / "x.txt")
    (repo / "inner").symlink_to("docs")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "tree")
    return repo, outside


def _plan(*paths: str, kind: str = "file_exists") -> str:
    args = (lambda path: {"path": path, "text": "x"}) if kind == "file_contains" else (lambda path: {"path": path})
    return json.dumps({"task": [{"id": "u-2", "intent": "write it",
                                 "predicates": [{"kind": kind, "args": args(path)} for path in paths]}],
                       "state": {"standing": "s.", "open": []}})


# --- at decode, a path outside the tree is an error naming the unit, the predicate and the path ---


@pytest.mark.parametrize("kind", ["file_exists", "file_contains"])
def test_decode_refuses_a_path_outside_the_tree_naming_unit_predicate_and_path(tree: tuple[Path, Path], kind: str):
    repo, outside = tree
    task = Task(id="t", goal="g", workspace=str(repo))
    absolute = str(outside / "x.txt")
    plan = _plan("../x", absolute, ".git/config", "door/x.txt", "notes.md", kind=kind)
    reply, errors = director.decode_move(plan, task)
    assert reply is None and errors == [
        f"unit 'u-2': p1 {kind}: path climbs out of the tree: ../x",
        f"unit 'u-2': p2 {kind}: absolute path, not relative to the tree: {absolute}",
        f"unit 'u-2': p3 {kind}: path is inside .git, not on the branch: .git/config",
        f"unit 'u-2': p4 {kind}: path leaves the tree through a symlink: door/x.txt",
    ]


def test_decode_takes_paths_inside_the_tree_and_never_path_checks_a_command(tree: tuple[Path, Path]):
    repo, _ = tree
    task = Task(id="t", goal="g", workspace=str(repo))
    reply, errors = director.decode_move(_plan("notes.md", "new/later.txt", "./docs/x.txt", "inner/x.txt"), task)
    assert errors == [] and [p.args["path"] for p in reply.units[0].predicates][1] == "new/later.txt"
    command = json.dumps({"state": {"standing": "s.", "open": []},
                          "task": [{"id": "u-2", "intent": "i", "predicates": [
        {"kind": "command", "args": {"cmd": "cat ../x /etc/hosts .git/config door/x.txt"}, "protected": ["notes.md"]}]}]})
    assert director.decode_move(command, task)[1] == []


def test_a_plan_with_a_path_outside_the_tree_goes_back_to_the_director_before_any_worker_call(
    tree: tuple[Path, Path], tmp_path: Path
):
    repo, outside = tree
    root = Root(tmp_path / "brain")
    root.path.mkdir()
    root.thinker_prompt.write_text("m\n")
    root.editor_prompt.write_text("e\n")
    absolute = str(outside / "x.txt")
    bad = {"id": "u-1", "intent": "write a.txt", "predicates": [
        {"kind": "file_exists", "args": {"path": path}} for path in ("../x", absolute, ".git/x", "door/x.txt")]}
    good = {"id": "u-1", "intent": "write a.txt", "predicates": [{"kind": "file_exists", "args": {"path": "a.txt"}}]}
    runner = ScriptedRunner([_reply({"task": [bad]}), _reply({"task": [good]}), _reply({"rest": True})], [
        {"write": {"a.txt": "A\n"}, "commit": True, "exit_code": 0, "result": "wrote a.txt", "total_cost_usd": 0.5}])
    task = loop.run("g", repo, root, Caps(), runner, unattended=0.0, notify=None)
    seats = ["worker" if "Write" in argv else "director" for argv, _, _, _ in runner.calls]
    assert seats == ["director", "director", "worker", "director"]  # no worker call until the corrected plan
    retry = runner.calls[1][1]
    assert "Your previous answer was rejected:\n" in retry
    for line in ("- unit 'u-1': p1 file_exists: path climbs out of the tree: ../x\n",
                 f"- unit 'u-1': p2 file_exists: absolute path, not relative to the tree: {absolute}\n",
                 "- unit 'u-1': p3 file_exists: path is inside .git, not on the branch: .git/x\n",
                 "- unit 'u-1': p4 file_exists: path leaves the tree through a symlink: door/x.txt\n"):
        assert line in retry, line
    assert retry.startswith(runner.calls[0][1])  # the same message, the errors appended: the existing retry path
    assert (task.terminal, task.awaiting_thought, task.questions) == ("interrupted", True, [])
    assert [u.status for u in task.units] == ["passed"]
    assert [u.predicates[0].args["path"] for u in task.units] == ["a.txt"]


# --- the pre-dispatch rule and the grader agree, through the one rule in grade.py ---


def test_the_pre_dispatch_rule_and_the_grader_agree_path_by_path(tree: tuple[Path, Path]):
    """Each path decoded against the repository as the workspace, and graded as `file_exists` on the same repository
    as the clone: a path the decode refuses is ungradeable at grading, and one it takes is graded on the branch."""
    repo, outside = tree
    task = Task(id="t", goal="g", workspace=str(repo))
    table = {
        "notes.md": True, "./notes.md": True, "docs/x.txt": True, "docs/../notes.md": True, "inner/x.txt": True,
        "missing.txt": True,
        "../outside/x.txt": False, "docs/../../outside/x.txt": False, str(outside / "x.txt"): False,
        str(repo / "notes.md"): False, str(repo.resolve() / "notes.md"): False, ".git/HEAD": False,
        ".GIT/config": False, "./.Git/HEAD": False, "door/x.txt": False, "link.txt": False, "a\0b": False,
    }
    verdicts = {}
    for path in table:
        _, errors = director.decode_move(_plan(path), task)
        graded = grade.grade(Unit("u-2", "i", [Predicate("file_exists", {"path": path})]), repo, 0)
        verdicts[path] = (not errors, not graded.ungradeable_ids)
    assert verdicts == {path: (confined, confined) for path, confined in table.items()}
    passing = [p for p in table if grade.grade(Unit("u", "i", [Predicate("file_exists", {"path": p})]), repo, 0).passed]
    assert passing == ["notes.md", "./notes.md", "docs/x.txt", "docs/../notes.md"]


def test_the_grader_keeps_its_own_checks_past_the_rule(tree: tuple[Path, Path]):
    """The rule takes a leaf symlink that stays inside the tree; the grader, reading the branch, still refuses it."""
    repo, _ = tree
    (repo / "alias.md").symlink_to("notes.md")
    _git(repo, "add", "alias.md")
    _git(repo, "commit", "-qm", "alias")
    assert grade.confined(repo, "alias.md") == ""
    verdict = grade.grade(Unit("u", "i", [Predicate("file_exists", {"path": "alias.md"})]), repo, 0)
    assert verdict.ungradeable_ids == ["p1"] and "symlink on the branch" in verdict.notes["p1"]


def test_the_decode_and_the_grader_call_the_one_rule(tree: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch):
    repo, _ = tree
    monkeypatch.setattr(grade, "confined", lambda tree, rel: f"refused by the one rule: {rel}")
    _, errors = director.decode_move(_plan("notes.md"), Task(id="t", goal="g", workspace=str(repo)))
    assert errors == ["unit 'u-2': p1 file_exists: refused by the one rule: notes.md"]
    verdict = grade.grade(Unit("u", "i", [Predicate("file_contains", {"path": "notes.md", "text": "pins"})]), repo, 0)
    assert verdict.notes == {"p1": "file_contains: refused by the one rule: notes.md"}
