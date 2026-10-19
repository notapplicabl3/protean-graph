"""The check-command profile cannot read the console token.

Code the editor wrote runs during grading as a `command` predicate. Before this fix the check-command profile denied
the network and writes only, so a predicate could `cat` the task's `console.json` and print the token into its own
verdict note. The profile now carries the editor profile's file-read deny for that one file. Proven against the real
`sandbox-exec`, never skipped: on a host without it every command predicate is ungradeable and this test fails.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from protean.graph import console, grade, worker
from protean.graph.budget import Caps
from protean.dry import make_workspace
from protean.state import Predicate, Root, Task, Unit, save, set_current


def _git(clone: Path, *args: str) -> None:
    subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t.invalid", *args], cwd=clone, check=True,
                   capture_output=True)


def test_a_check_command_under_the_real_profile_cannot_read_the_console_file_but_reads_the_clone(
    tmp_path: Path, root: Root
):
    root.path.mkdir(parents=True)
    workspace = make_workspace(tmp_path / "ws")
    task = Task(id="task-20261010T170000", goal="g", workspace=str(workspace),
                units=[Unit("u1", "write a.txt", [Predicate("file_exists", {"path": "a.txt"})])])
    save(root, task)
    set_current(root, task.id)
    clone = make_workspace(root.clones_dir(task.id) / "u1")
    (clone / "a.txt").write_text("in the clone\n")
    _git(clone, "add", "a.txt")
    _git(clone, "commit", "-qm", "a")
    with console.serve(root, task, Caps()) as desk:
        secret = root.console_file(task.id).resolve()
        assert json.loads(secret.read_text())["token"] == desk.token  # the brain itself can read it

        profile = grade._sandbox_profile(clone, root.console_file(task.id))
        assert f'(deny file-read* (literal "{secret}"))' in profile and "(deny network*)" in profile
        sandbox = worker.sandbox_binary()  # the real binary; raises (fails the test) when absent

        def confined(*command: str) -> subprocess.CompletedProcess:
            return subprocess.run([sandbox, "-p", profile, *command], cwd=clone, capture_output=True, text=True)

        read = confined("/bin/cat", str(secret))
        assert read.returncode != 0 and read.stdout == "" and "Operation not permitted" in read.stderr
        (clone / "door").symlink_to(secret)
        through_link = confined("/bin/cat", "door")
        assert through_link.returncode != 0 and through_link.stdout == ""
        hard_link = confined("/bin/ln", str(secret), "copy")
        assert hard_link.returncode != 0 and not (clone / "copy").exists()
        (clone / "door").unlink()
        clone_file = confined("/bin/cat", "a.txt")
        assert clone_file.returncode == 0 and clone_file.stdout == "in the clone\n"

        # The same through the grader: a `command` predicate that cats the token fails, its note carries no token,
        # and one that reads a clone file passes.
        unit = Unit("u1", "probe", [Predicate("command", {"cmd": f"cat {secret}"}),
                                    Predicate("command", {"cmd": "cat a.txt"})])
        verdict = grade.grade(unit, clone, 0, console_file=root.console_file(task.id))
        assert (verdict.failed_ids, verdict.passed_ids) == (["p1"], ["p2"])
        assert desk.token not in json.dumps(verdict.notes) and "Operation not permitted" in verdict.notes["p1"]

        # Without the path the profile is what it was: the deny clause is what closes the hole.
        assert "file-read" not in grade._sandbox_profile(clone)
