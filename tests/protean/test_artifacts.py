"""The documents a session's branches committed, as a client sees them: `artifacts` in the snapshot (the workspace's
HEAD as `base`, then `task/<session>`'s changed paths and each landed explore's own, each with the sha its branch points
at) and `GET /file`, which serves one listed path's committed bytes from the workspace's object store and nothing else.
No workspace, or any git failure, leaves the artifacts empty and the snapshot whole."""

from __future__ import annotations

import http.client
import json
import os
import subprocess
from collections.abc import Iterator
from pathlib import Path

import pytest

from protean.dry import GIT_ID, make_workspace
from protean.graph import console, grade
from protean.graph.budget import Caps
from protean.state import EXPLORE, Move, Root, Task

TASK_ID = "task-20261022T120000"
OWN = object()  # the console's own token
BYTES = b"# x\x00not text: \xff\xfe\x80\n"


def _git(workspace: Path, *args: str, data: bytes | None = None, index: Path | None = None) -> str:
    env = {**os.environ, "GIT_INDEX_FILE": str(index)} if index is not None else None
    proc = subprocess.run(["git", *GIT_ID, *args], cwd=workspace, input=data, capture_output=True, check=True, env=env)
    return proc.stdout.decode().strip()


def _commit(workspace: Path, parent: str, files: dict[str, bytes], modes: dict[str, str] | None = None,
            removed: tuple[str, ...] = ()) -> str:
    """A commit on top of `parent` adding `files` (mode 100644 unless `modes` says otherwise, e.g. 120000 for a symlink
    whose blob is its target text) and dropping `removed`, built in a private index: the working tree and HEAD never
    move."""
    index = workspace.parent / "index"
    _git(workspace, "read-tree", parent, index=index)
    for rel, data in files.items():
        blob = _git(workspace, "hash-object", "-w", "--stdin", data=data)
        mode = (modes or {}).get(rel, "100644")
        _git(workspace, "update-index", "--add", "--cacheinfo", f"{mode},{blob},{rel}", index=index)
    for rel in removed:
        _git(workspace, "update-index", "--force-remove", rel, index=index)
    tree = _git(workspace, "write-tree", index=index)
    index.unlink()
    return _git(workspace, "commit-tree", tree, "-p", parent, "-m", "documents")


def _session(tmp_path: Path, files: dict[str, bytes], modes: dict[str, str] | None = None,
             removed: tuple[str, ...] = ()) -> tuple[Task, str, str]:
    """A session on a real workspace whose `task/<session>` starts at HEAD and carries one commit adding `files` (and
    dropping `removed`); returns the session, the base (HEAD) and the task branch's tip."""
    workspace = make_workspace(tmp_path / "ws")
    base = _git(workspace, "rev-parse", "HEAD")
    tip = _commit(workspace, base, files, modes, removed)
    _git(workspace, "update-ref", f"refs/heads/task/{TASK_ID}", tip)
    return Task(id=TASK_ID, goal="g", workspace=str(workspace)), base, tip


def _explore(task: Task, n: int, parent: str, files: dict[str, bytes], outcome: str = "landed") -> str:
    """An explore branch `explore/<session>/<n>` committed on top of `parent`, and its move as the loop records it."""
    workspace, branch = Path(task.workspace), f"explore/{task.id}/{n}"
    tip = _commit(workspace, parent, files)
    _git(workspace, "update-ref", f"refs/heads/{branch}", tip)
    task.moves.append(Move(id=f"m{len(task.moves) + 1}", kind=EXPLORE, outcome=outcome, branch=branch,
                           paths=sorted(files)))
    return tip


@pytest.fixture
def desk(root: Root, tmp_path: Path) -> Iterator[tuple[console.Console, Task, str]]:
    """A served console on a session whose task branch adds `notes/x.md` (NUL and non-UTF-8 bytes in it), whose
    landed explore adds `notes/draft.md` on top of it, and whose explore that was not fetched adds `notes/lost.md`."""
    task, _, tip = _session(tmp_path, {"notes/x.md": BYTES})
    _explore(task, 1, tip, {"notes/draft.md": b"draft\n"})
    _explore(task, 2, tip, {"notes/lost.md": b"lost\n"}, outcome="not fetched: fetch into the workspace failed: x")
    served = console.Console(root, task, Caps())
    served.start()
    try:
        yield served, task, tip
    finally:
        served.close()


def _get(served: console.Console, path: str, method: str = "GET", token: object = OWN) -> tuple[int, str, bytes]:
    host, port = served.address
    conn = http.client.HTTPConnection(host, port, timeout=10)
    headers = {} if token is None else {"Authorization": f"Bearer {served.token if token is OWN else token}"}
    try:
        conn.request(method, path, headers=headers)
        resp = conn.getresponse()
        return resp.status, resp.getheader("Content-Type") or "", resp.read()
    finally:
        conn.close()


def test_the_snapshot_lists_the_task_branchs_changed_paths_with_its_tip_and_the_base(root: Root, tmp_path: Path):
    task, base, tip = _session(tmp_path, {"notes/x.md": b"# x\n"})
    assert console.Console(root, task, Caps()).snapshot()["artifacts"] == {
        "base": base, "branches": [{"branch": f"task/{task.id}", "tip": tip, "paths": ["notes/x.md"]}]}
    _git(Path(task.workspace), "update-ref", f"refs/heads/task/{task.id}", base)  # a branch with no change is left out
    assert console.Console(root, task, Caps()).snapshot()["artifacts"] == {"base": base, "branches": []}


def test_a_landed_explore_lists_only_its_own_paths_and_one_not_fetched_is_absent(root: Root, tmp_path: Path):
    task, base, tip = _session(tmp_path, {"notes/x.md": b"# x\n"})
    explored = _explore(task, 1, tip, {"notes/draft.md": b"draft\n"})
    _explore(task, 2, tip, {"notes/lost.md": b"lost\n"}, outcome="not fetched: no change on the branch")
    assert console.Console(root, task, Caps()).snapshot()["artifacts"] == {"base": base, "branches": [
        {"branch": f"task/{task.id}", "tip": tip, "paths": ["notes/x.md"]},
        {"branch": f"explore/{task.id}/1", "tip": explored, "paths": ["notes/draft.md"]}]}


def test_no_workspace_leaves_the_artifacts_empty_and_the_snapshot_whole(root: Root, tmp_path: Path):
    plain = tmp_path / "plain"
    plain.mkdir()
    for workspace in ("w", str(plain)):  # no such path; a directory that is not a repository
        task = Task(id=TASK_ID, goal="g", workspace=workspace)
        for snap in (console.snapshot(task, Caps(), live=False), console.Console(root, task, Caps()).snapshot()):
            assert snap["artifacts"] == {"base": None, "branches": []}
            assert (snap["task"], snap["workspace"], snap["moves"], snap["units"]) == (TASK_ID, workspace, [], [])
            assert {"thought", "ceilings", "prices", "questions", "thoughts"} <= set(snap)


def test_file_serves_a_listed_path_byte_identical(desk: tuple[console.Console, Task, str]):
    served, task, _ = desk
    status, kind, body = _get(served, f"/file?branch=task/{task.id}&path=notes/x.md")
    assert (status, kind, body) == (200, "application/octet-stream", BYTES)
    assert _get(served, f"/file?branch=explore/{task.id}/1&path=notes/draft.md")[::2] == (200, b"draft\n")


def test_file_refuses_everything_outside_the_artifact_list(desk: tuple[console.Console, Task, str],
                                                          monkeypatch: pytest.MonkeyPatch):
    served, task, tip = desk
    branch = f"task/{task.id}"
    (Path(task.workspace) / "notes").mkdir()
    (Path(task.workspace) / "notes" / "loose.md").write_text("in the working tree, never committed\n")
    for token in (None, "wrong"):
        status, kind, body = _get(served, f"/file?branch={branch}&path=notes/x.md", token=token)
        assert (status, kind, json.loads(body)) == (401, "application/json", {"ok": False, "error": "unauthorized"})
    for query in (f"branch={branch}&path=notes/none.md",  # not on the branch
                  f"branch={branch}&path=README.md",  # in the base, not changed on the branch
                  f"branch={branch}&path=notes/loose.md",  # in the working tree only
                  f"branch={branch}&path=../ws/notes/x.md", f"branch={branch}&path=notes/../notes/x.md",
                  f"branch={branch}&path=.git/config", "branch=main&path=README.md",  # a branch not in the list
                  f"branch=explore/{task.id}/2&path=notes/lost.md",  # an explore that was not fetched
                  f"branch=explore/{task.id}/1&path=notes/x.md"):  # the task's path, not the explore's own
        status, kind, body = _get(served, f"/file?{query}")
        assert (status, kind, json.loads(body)["ok"]) == (404, "application/json", False), query
    for query in ("", "?branch=" + branch, "?path=notes/x.md", "?branch=&path=notes/x.md", f"?branch={branch}&path="):
        status, _, body = _get(served, f"/file{query}")
        assert (status, json.loads(body)["ok"]) == (400, False), query
    status, _, body = _get(served, f"/file?branch={branch}&path=notes/x.md", method="POST")
    assert (status, json.loads(body)) == (405, {"ok": False, "error": "/file takes GET"})
    # The bound is `console.FILE_BOUND` (24 MiB, never the grader's), shrunk here so no test writes 24 MiB.
    assert (console.FILE_BOUND, grade.FILE_BOUND) == (25_165_824, 262_144)
    monkeypatch.setattr(console, "FILE_BOUND", 4096)
    at_bound = _commit(Path(task.workspace), tip, {"notes/big.md": b"b" * (console.FILE_BOUND + 1),
                                                   "notes/edge.md": b"e" * console.FILE_BOUND})
    _git(Path(task.workspace), "update-ref", f"refs/heads/{branch}", at_bound)  # as an integration moves it
    status, _, body = _get(served, f"/file?branch={branch}&path=notes/big.md")
    assert (status, json.loads(body)) == (413, {"ok": False, "error": "notes/big.md is over the 4096-byte bound"})
    status, _, body = _get(served, f"/file?branch={branch}&path=notes/edge.md")
    assert (status, len(body)) == (200, console.FILE_BOUND)


def test_a_non_ascii_or_quoted_path_is_listed_verbatim_and_served(root: Root, tmp_path: Path):
    """`git diff --name-only` quotes such a path (`"notes/caf\\303\\251.md"`) unless read with `-z`; a quoted name is
    listed but can never be served, so the list is read NUL-separated and the path reaches the client as it is."""
    from urllib.parse import quote
    files = {"notes/café.md": b"caf\xc3\xa9\n", 'notes/a"b.md': b"quoted\n"}
    task, base, tip = _session(tmp_path, files)
    assert console.artifacts(task) == {"base": base, "branches": [{"branch": f"task/{TASK_ID}", "tip": tip,
                                                                     "paths": sorted(files)}]}
    served = console.Console(root, task, Caps())
    served.start()
    try:
        for rel, data in files.items():
            assert _get(served, f"/file?branch=task/{task.id}&path={quote(rel, safe='')}")[::2] == (200, data), rel
    finally:
        served.close()


@pytest.fixture
def serve(root: Root) -> Iterator:
    """A served console on a given session, closed after the test."""
    opened: list[console.Console] = []

    def start(task: Task) -> console.Console:
        served = console.Console(root, task, Caps())
        served.start()
        opened.append(served)
        return served

    yield start
    for served in opened:
        served.close()


def test_a_symlink_on_the_branch_is_listed_but_never_served(serve, tmp_path: Path):
    """A committed symlink's blob is its target text (`../etc/passwd` here); the grader refuses symlinks and so does
    `/file`: the entry's mode is checked before any blob is read, and only a regular file is served."""
    target = b"../../../../etc/passwd"
    task, _, tip = _session(tmp_path, {"notes/link.md": target, "notes/x.md": b"# x\n"}, modes={"notes/link.md": "120000"})
    assert console.artifacts(task)["branches"] == [{"branch": f"task/{TASK_ID}", "tip": tip,
                                                      "paths": ["notes/link.md", "notes/x.md"]}]
    assert _git(Path(task.workspace), "ls-tree", tip, "--", "notes/link.md").startswith("120000 blob ")
    served = serve(task)
    status, kind, body = _get(served, f"/file?branch=task/{task.id}&path=notes/link.md")
    assert (status, kind, json.loads(body)) == (404, "application/json", {
        "ok": False, "error": f"notes/link.md on task/{task.id} is not a regular file (mode 120000)"})
    assert target not in body
    assert _get(served, f"/file?branch=task/{task.id}&path=notes/x.md")[::2] == (200, b"# x\n")  # the file beside it


def test_a_path_the_branch_deleted_is_not_listed_and_file_answers_404(serve, tmp_path: Path):
    """A deletion is no document (the operator, 2026-10-22: "delete it"): the task branch's paths leave out what it
    deleted (`--diff-filter=d`), a branch with nothing else is left out, and `/file` refuses the path as unlisted. A
    landed explore's recorded paths are filtered the same way, against what its tip holds."""
    task, _, tip = _session(tmp_path, {}, removed=("README.md",))
    assert console.artifacts(task)["branches"] == []
    assert _git(Path(task.workspace), "ls-tree", tip, "--", "README.md") == ""
    status, _, body = _get(serve(task), f"/file?branch=task/{task.id}&path=README.md")
    assert (status, json.loads(body)) == (404, {"ok": False, "error": f"README.md is not a document on task/{task.id}"})
    explore_tip = _explore(task, 1, tip, {"notes/draft.md": b"draft\n"})
    task.moves[-1].paths.append("notes/gone.md")  # recorded at landing, deleted on the branch since, or never there
    assert console.artifacts(task)["branches"] == [{"branch": f"explore/{TASK_ID}/1", "tip": explore_tip,
                                                     "paths": ["notes/draft.md"]}]


def test_a_repeated_branch_or_path_key_is_refused(desk: tuple[console.Console, Task, str]):
    """`parse_qs` keeps every value of a repeated key; `/file` takes exactly one of each, never the first."""
    served, task, _ = desk
    branch = f"task/{task.id}"
    for query in (f"branch={branch}&branch=main&path=notes/x.md", f"branch=main&branch={branch}&path=notes/x.md",
                  f"branch={branch}&path=notes/x.md&path=notes/x.md", f"branch={branch}&path=notes/none.md&path=notes/x.md"):
        status, kind, body = _get(served, f"/file?{query}")
        assert (status, kind, json.loads(body)) == (400, "application/json",
                                                     {"ok": False, "error": "/file takes one branch and one path"}), query
