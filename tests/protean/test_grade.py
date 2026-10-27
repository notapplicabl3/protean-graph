"""The grader: false-success shapes as negative tests, confinement, bounds, and a positive control."""

from __future__ import annotations

import os
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest

from protean.graph import grade as grade_module
from protean.graph.grade import FILE_BOUND, PREDICATE_KINDS, grade, validate
from protean.state import Predicate as P
from protean.state import Unit


def _unit(*predicates: P) -> Unit:
    return Unit(id="u1", intent="do it", predicates=list(predicates))


def _git(clone: Path, *args: str) -> None:
    subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t.invalid", *args], cwd=clone, check=True,
                   capture_output=True)


def _commit(clone: Path, *rels: str) -> None:
    """Commit the named paths (forced past any ignore rule) so they are on the branch."""
    _git(clone, "add", "-f", "--", *rels)
    _git(clone, "commit", "-qm", f"add {' '.join(rels)}")


@pytest.fixture
def clone(tmp_path: Path) -> Path:
    """A one-commit checkout holding notes.md; file predicates grade what is on its branch."""
    path = tmp_path / "clone"
    path.mkdir()
    (path / "notes.md").write_text("nothing about pins here\n")
    _git(path, "init", "-q", "-b", "main")
    _commit(path, "notes.md")
    return path


def test_misspelled_exit_code_arg_is_rejected_at_validation(clone: Path):
    bad = P("exit_code", {"expected": 0})
    assert validate([bad]) == ["p1 exit_code: missing required arg 'code'", "p1 exit_code: unknown arg 'expected'"]
    verdict = grade(_unit(bad), clone, 1)
    assert verdict.passed is False and verdict.ungradeable_ids == ["p1"]


def test_misspelled_file_contains_arg_is_rejected_at_validation(clone: Path):
    bad = P("file_contains", {"path": "notes.md", "value": "## Pin inventory"})
    assert validate([bad]) == ["p1 file_contains: missing required arg 'text'", "p1 file_contains: unknown arg 'value'"]
    assert grade(_unit(bad), clone, 0).passed is False


def test_run2_unit_shape_does_not_pass(clone: Path):
    unit = _unit(
        P("file_exists", {"path": "notes.md"}),
        P("file_contains", {"path": "notes.md", "text": "## Pin inventory"}),
        P("exit_code", {"code": 0}),
    )
    assert validate(unit.predicates) == []
    verdict = grade(unit, clone, 1)
    assert verdict.passed is False
    assert (verdict.passed_ids, verdict.failed_ids, verdict.ungradeable_ids) == (["p1"], ["p2", "p3"], [])


def test_file_contains_with_unrelated_content_fails(clone: Path):
    verdict = grade(_unit(P("file_contains", {"path": "notes.md", "text": "REQUIRED CONTENT"})), clone, 0)
    assert verdict.passed is False and verdict.failed_ids == ["p1"]


def test_vocabulary_and_empty_predicate_list(clone: Path):
    assert {kind: (sorted(req), sorted(opt)) for kind, (req, opt) in PREDICATE_KINDS.items()} == {
        "file_exists": (["path"], []), "file_contains": (["path", "text"], []),
        "exit_code": (["code"], []), "command": (["cmd"], ["expect_exit"]),
    }
    assert validate([]) == ["a unit needs at least one predicate"]
    assert grade(_unit(), clone, 0).passed is False


@pytest.mark.parametrize(
    ("predicate", "message"),
    [
        (P("file_there", {"path": "a"}), "unknown kind 'file_there'"),
        (P(["file_exists"], {"path": "a"}), "unknown kind ['file_exists']"),
        (P("exit_code", {"code": "0"}), "arg 'code' must be an integer"),
        (P("exit_code", {"code": True}), "arg 'code' must be an integer"),
        (P("command", {"cmd": "true", "expect_exit": 1.0}), "arg 'expect_exit' must be an integer"),
        (P("file_exists", {"path": 3}), "arg 'path' must be a non-empty string"),
        (P("file_contains", {"path": "a", "text": ""}), "arg 'text' must be a non-empty string"),
        (P("command", {}), "missing required arg 'cmd'"),
        (P("file_exists", ["a"]), "args must be an object"),
    ],
)
def test_malformed_predicates_are_rejected(predicate: P, message: str):
    problems = validate([predicate])
    assert len(problems) == 1 and message in problems[0]


def test_a_fully_satisfied_unit_passes(clone: Path):
    unit = _unit(
        P("file_exists", {"path": "notes.md"}),
        P("file_contains", {"path": "notes.md", "text": "about pins"}),
        P("exit_code", {"code": 0}),
        P("command", {"cmd": "test -f notes.md"}),
        P("command", {"cmd": "exit 3", "expect_exit": 3}),
    )
    assert validate(unit.predicates) == []
    verdict = grade(unit, clone, 0)
    assert verdict.passed is True and verdict.passed_ids == ["p1", "p2", "p3", "p4", "p5"]
    assert set(verdict.notes) == set(verdict.passed_ids)


def test_missing_file_fails_file_exists_and_is_ungradeable_for_file_contains(clone: Path):
    verdict = grade(_unit(P("file_exists", {"path": "gone"}), P("file_contains", {"path": "gone", "text": "x"})), clone, 0)
    assert (verdict.failed_ids, verdict.ungradeable_ids, verdict.passed) == (["p1"], ["p2"], False)


def test_a_path_escaping_the_clone_is_ungradeable(clone: Path, tmp_path: Path):
    outside = tmp_path / "outside.txt"
    outside.write_text("SECRET")
    (clone / "link.txt").symlink_to(outside)
    (clone / "alias.md").symlink_to(clone / "notes.md")
    _commit(clone, "link.txt", "alias.md")
    for path in ["../outside.txt", str(outside), "link.txt", "alias.md"]:
        unit = _unit(P("file_contains", {"path": path, "text": "SECRET"}), P("file_exists", {"path": path}))
        verdict = grade(unit, clone, 0)
        assert verdict.ungradeable_ids == ["p1", "p2"] and verdict.passed is False, path
    assert all("symlink" in note for note in verdict.notes.values())
    assert grade(_unit(P("file_contains", {"path": "./notes.md", "text": "pins"})), clone, 0).passed is True


def test_over_bound_and_undecodable_files_are_ungradeable(clone: Path):
    (clone / "big.txt").write_text("x" * FILE_BOUND + "y")
    (clone / "edge.txt").write_text("x" * (FILE_BOUND - 1) + "y")
    (clone / "bin").write_bytes(b"\xff\xfe\x00text")
    _commit(clone, "big.txt", "edge.txt", "bin")
    over = grade(_unit(P("file_contains", {"path": "big.txt", "text": "x"})), clone, 0)
    assert over.ungradeable_ids == ["p1"] and over.passed is False
    assert grade(_unit(P("file_contains", {"path": "edge.txt", "text": "y"})), clone, 0).passed is True
    assert grade(_unit(P("file_contains", {"path": "bin", "text": "text"})), clone, 0).ungradeable_ids == ["p1"]


def test_exit_code_none_is_ungradeable_and_blocks_a_pass(clone: Path):
    verdict = grade(_unit(P("file_exists", {"path": "notes.md"}), P("exit_code", {"code": 0})), clone, None)
    assert (verdict.passed_ids, verdict.ungradeable_ids, verdict.failed_ids) == (["p1"], ["p2"], [])
    assert verdict.passed is False


def test_a_failing_command_fails_and_a_missing_clone_is_ungradeable(clone: Path, tmp_path: Path):
    verdict = grade(_unit(P("command", {"cmd": "echo boom >&2; exit 2"})), clone, 0)
    assert verdict.failed_ids == ["p1"] and "boom" in verdict.notes["p1"]
    assert grade(_unit(P("command", {"cmd": "true"})), tmp_path / "nope", 0).ungradeable_ids == ["p1"]


def test_a_command_timeout_is_ungradeable_and_kills_the_group(clone: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(grade_module, "COMMAND_TIMEOUT", 0.3)
    started = time.monotonic()
    verdict = grade(_unit(P("command", {"cmd": 'sh -c "sleep 1; touch late.txt"; true'})), clone, 0)
    assert verdict.ungradeable_ids == ["p1"] and verdict.passed is False
    assert time.monotonic() - started < 1
    time.sleep(1.5)
    assert not (clone / "late.txt").exists()


def test_a_command_predicate_meets_the_package_shim_not_the_model(clone: Path, shim_on_path: Path):
    assert shutil.which("claude") == str(shim_on_path / "claude")
    unit = _unit(P("command", {"cmd": "claude --version"}),
                 P("command", {"cmd": "claude --version", "expect_exit": 89}),
                 P("command", {"cmd": "command -v claude"}))
    verdict = grade(unit, clone, 0)
    assert (verdict.ungradeable_ids, verdict.passed_ids, verdict.passed) == (["p1", "p2"], ["p3"], False)
    for pid in ("p1", "p2"):
        assert "protean: model calls are not allowed from a command predicate" in verdict.notes[pid]
        assert "SHIM: the real binary was reached" not in verdict.notes[pid]
    assert str(grade_module.SHIM_DIR / "claude") in verdict.notes["p3"]


def test_a_command_predicate_runs_without_model_credentials(clone: Path, monkeypatch: pytest.MonkeyPatch):
    for name in ("ANTHROPIC_API_KEY", "CLAUDE_CODE_OAUTH_TOKEN", "ANTHROPIC_BASE_URL", "CLAUDE_CONFIG_DIR"):
        monkeypatch.setenv(name, "secret")
    monkeypatch.setenv("MVP_KEPT", "yes")
    cmd = ('test -z "$ANTHROPIC_API_KEY$CLAUDE_CODE_OAUTH_TOKEN$ANTHROPIC_BASE_URL$CLAUDE_CONFIG_DIR"'
           ' && test "$MVP_KEPT" = yes')
    assert grade(_unit(P("command", {"cmd": cmd})), clone, 0).passed is True


def _repo(clone: Path) -> Path:
    def git(*args: str) -> None:
        subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t.invalid", *args], cwd=clone, check=True,
                       capture_output=True)
    git("init", "-q", "-b", "main")
    (clone / ".gitignore").write_text("build/\n")
    (clone / "build").mkdir()
    (clone / "build" / "out.txt").write_text("REQUIRED\n")
    (clone / "build" / "kept.txt").write_text("REQUIRED\n")
    (clone / "via.txt").symlink_to("build/out.txt")
    git("add", ".gitignore", "notes.md", "via.txt")
    git("add", "-f", "build/kept.txt")
    git("commit", "-qm", "seed")
    return clone


def test_a_gitignored_path_is_ungradeable_for_both_file_kinds(clone: Path):
    _repo(clone)
    for path, reason in (("build/out.txt", "path is gitignored, not on the branch"), ("via.txt", "symlink")):
        unit = _unit(P("file_exists", {"path": path}), P("file_contains", {"path": path, "text": "REQUIRED"}))
        verdict = grade(unit, clone, 0)
        assert (verdict.ungradeable_ids, verdict.passed) == (["p1", "p2"], False), path
        assert all(reason in note for note in verdict.notes.values()), path
    tracked = _unit(P("file_exists", {"path": "build/kept.txt"}), P("file_contains", {"path": "notes.md", "text": "pins"}))
    assert grade(tracked, clone, 0).passed is True


def test_only_the_branch_is_graded_never_the_working_tree(clone: Path):
    (clone / "written.txt").write_text("REQUIRED\n")
    (clone / "notes.md").write_text("REQUIRED edit that was never committed\n")
    unit = _unit(P("file_exists", {"path": "written.txt"}), P("file_contains", {"path": "notes.md", "text": "REQUIRED"}),
                 P("command", {"cmd": "echo REQUIRED > made.txt"}), P("file_exists", {"path": "made.txt"}))
    verdict = grade(unit, clone, 0)
    assert (verdict.passed_ids, verdict.failed_ids, verdict.passed) == (["p3"], ["p1", "p2", "p4"], False)
    assert "missing: written.txt" in verdict.notes["p1"] and "does not contain" in verdict.notes["p2"]


def test_a_committed_nested_repository_is_not_on_the_branch(clone: Path):
    _git(clone, "init", "-q", "-b", "main", "sub")
    (clone / "sub" / "f.py").write_text("x = 1\n")
    _git(clone / "sub", "add", "f.py")
    _git(clone / "sub", "commit", "-qm", "inner")
    _git(clone, "add", "sub")
    _git(clone, "commit", "-qm", "gitlink")
    assert subprocess.run(["git", "status", "--porcelain"], cwd=clone, capture_output=True, text=True).stdout == ""
    unit = _unit(P("file_exists", {"path": "sub/f.py"}), P("file_exists", {"path": "sub"}),
                 P("file_contains", {"path": "sub/f.py", "text": "x = 1"}))
    verdict = grade(unit, clone, 0)
    assert (verdict.failed_ids, verdict.ungradeable_ids, verdict.passed) == (["p1", "p2"], ["p3"], False)
    assert "not a regular file: sub" in verdict.notes["p2"]


def test_an_interrupted_command_leaves_no_live_group(clone: Path, monkeypatch: pytest.MonkeyPatch):
    started = []
    def interrupted(self, *args, **kwargs):
        started.append(self)
        raise KeyboardInterrupt
    monkeypatch.setattr(subprocess.Popen, "communicate", interrupted)
    with pytest.raises(KeyboardInterrupt):
        grade(_unit(P("command", {"cmd": "(sleep 0.5; touch late.txt) & sleep 30"})), clone, 0)
    (proc,) = started
    assert proc.poll() is not None
    time.sleep(1.0)
    assert not (clone / "late.txt").exists()


def test_a_command_whose_escaped_child_holds_the_pipe_still_returns(clone: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(grade_module, "COMMAND_TIMEOUT", 0.3)
    monkeypatch.setattr(grade_module, "KILL_DRAIN_SECONDS", 0.5, raising=False)
    escaped = f'{sys.executable} -c "import os, time; os.setsid(); time.sleep(5)" & sleep 10'
    started = time.monotonic()
    verdict = grade(_unit(P("command", {"cmd": escaped})), clone, 0)
    assert verdict.ungradeable_ids == ["p1"] and time.monotonic() - started < 3


def test_a_path_inside_the_git_dir_is_ungradeable(clone: Path):
    (clone / ".git" / "info").mkdir(parents=True, exist_ok=True)
    (clone / ".git" / "info" / "exclude").write_text(".venv/\n")
    unit = _unit(P("file_exists", {"path": ".git/HEAD"}),
                 P("file_contains", {"path": ".git/info/exclude", "text": ".venv/"}),
                 P("file_exists", {"path": ".git"}),
                 P("file_exists", {"path": ".GIT/HEAD"}),
                 P("file_contains", {"path": "./.Git/info/exclude", "text": ".venv/"}))
    verdict = grade(unit, clone, 0)
    assert verdict.ungradeable_ids == ["p1", "p2", "p3", "p4", "p5"] and verdict.passed is False
    assert all("inside .git" in note for note in verdict.notes.values())


def test_a_directory_does_not_satisfy_file_exists(clone: Path):
    (clone / "docs").mkdir()
    (clone / "docs" / "x.txt").write_text("x\n")
    _commit(clone, "docs/x.txt")
    verdict = grade(_unit(P("file_exists", {"path": "docs"})), clone, 0)
    assert verdict.failed_ids == ["p1"] and "not a regular file: docs" in verdict.notes["p1"]


def test_a_command_predicate_cannot_open_a_network_connection(clone: Path):
    with socket.socket() as server:
        server.bind(("127.0.0.1", 0))
        server.listen(1)
        port = server.getsockname()[1]
        connect = f"import socket; socket.create_connection(('127.0.0.1', {port}), timeout=2)"
        verdict = grade(_unit(P("command", {"cmd": f'{sys.executable} -c "{connect}"'})), clone, 0)
    assert verdict.failed_ids == ["p1"] and verdict.passed is False
    assert "Operation not permitted" in verdict.notes["p1"]


def test_a_command_predicate_cannot_write_outside_the_clone(clone: Path):
    probe = Path.home() / f".protean-sandbox-probe-{os.getpid()}"
    try:
        cmd = f"echo out > {probe}; echo in > inside.txt; true"
        verdict = grade(_unit(P("command", {"cmd": cmd})), clone, 0)
        assert verdict.passed is True
        assert (clone / "inside.txt").read_text() == "in\n"
        assert not probe.exists()
    finally:
        probe.unlink(missing_ok=True)


def test_a_command_predicate_without_a_sandbox_is_ungradeable(clone: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(grade_module.shutil, "which", lambda name, *args, **kwargs: None)
    verdict = grade(_unit(P("command", {"cmd": "true"})), clone, 0)
    assert verdict.ungradeable_ids == ["p1"] and "no sandbox" in verdict.notes["p1"]
