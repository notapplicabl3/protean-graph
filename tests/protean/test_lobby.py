"""The lobby, a standing console for the brain root (`protean lobby`), driven over its real server thread, and the two
things a client of it reads from the brain's own record: `doing` in the snapshot and the `launch` event.

127.0.0.1 only, a private `state/lobby.json`, every request without the token refused with nothing spawned or written;
`/sessions` lists every record that loads, newest first, live exactly where a brain's console file names a pid that is
alive AND the session's lock is held (`loop.live_holder`; a lock the probe cannot open is skipped from the list, and
answers 500 on `/session` for that session and on `/start` / `/resume` while any session's lock is unprobeable); `/session` and `/file` read one session as the console would; `/start` and `/resume` validate, refuse while any
brain is live, and spawn exactly one detached `protean think` or `protean resume` with the argv the contract fixes (the
spawn injected, so no real brain runs, except one real child that exits at once)."""

from __future__ import annotations

import fcntl
import http.client
import json
import os
import re
import shutil
import signal
import socket
import stat
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest

from protean import dry, loop, state
from protean.dry import GIT_ID, make_workspace
from protean.graph import budget, console, graph, lobby
from protean.graph.budget import Caps
from protean.runner import RunResult
from protean.state import EXPLORE, Call, Held, Idea, Move, Root, Task, save, set_current

OWN = object()  # the lobby's own token
SUMMARY = {"task", "thought", "project", "workspace", "created", "tick", "cost_usd", "terminal", "stop_reason", "paused",
           "mode", "awaiting_thought", "open_questions", "held", "held_approved", "moves", "last_move", "last_ideas",
           "doing", "standing", "live", "pid"}
BYTES = b"# x\x00not text: \xff\xfe\x80\n"


class Child:
    """A spawned brain as the lobby sees it: a pid, and `poll` (None while it runs, else its exit code)."""

    def __init__(self, pid: int = 4242, code: int | None = None) -> None:
        self.pid, self.code = pid, code

    def poll(self) -> int | None:
        return self.code


class Spawner:
    """The injected spawn: records each argv and its keyword arguments, then does what `act` does (a fake brain)."""

    def __init__(self) -> None:
        self.calls: list[tuple[list[str], dict[str, Any]]] = []
        self.act: Callable[[list[str], dict[str, Any]], Child] = lambda argv, kwargs: Child()

    def __call__(self, argv: list[str], **kwargs: Any) -> Child:
        self.calls.append((list(argv), dict(kwargs)))
        return self.act(argv, kwargs)


HELD: list[Any] = []  # session locks this test holds as a live brain would, released after each test


@pytest.fixture(autouse=True)
def _release_locks() -> Iterator[None]:
    yield
    while HELD:
        HELD.pop().close()


def _console_up(root: Root, task_id: str, pid: int | None = None, locked: bool | None = None) -> None:
    """A brain's console file for `task_id`, naming `pid` (this process: alive). A live brain also holds the session's
    lock (`loop._task_lock`), so by default this process takes it when the pid is its own; `locked=False` leaves a
    console file naming a live pid with no brain behind it (pid reuse after a kill)."""
    console._write_private(root.console_file(task_id), {"host": "127.0.0.1", "port": 1, "token": "brain-token",
                                                        "pid": os.getpid() if pid is None else pid, "task": task_id})
    if locked if locked is not None else pid is None:
        handle = (root.state_dir(task_id) / "lock").open("a+", encoding="utf-8")
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        HELD.append(handle)


def _brain_up(root: Root, task_id: str, current: bool = True) -> Callable[[list[str], dict[str, Any]], Child]:
    """A fake brain that comes up: `state/current` names `task_id` (for a think) and its console file appears."""
    def act(argv: list[str], kwargs: dict[str, Any]) -> Child:
        if current:
            set_current(root, task_id)
        _console_up(root, task_id)
        return Child()
    return act


@pytest.fixture
def served(root: Root) -> Iterator[Callable[..., lobby.Lobby]]:
    """Start a lobby on `root` (an injected spawn unless `spawn` says otherwise), closed after the test."""
    opened: list[lobby.Lobby] = []

    def start(spawn: Any = None, echo: Callable[[str], None] | None = None) -> lobby.Lobby:
        root.path.mkdir(parents=True, exist_ok=True)
        desk = lobby.Lobby(root, spawn=Spawner() if spawn is None else spawn, echo=echo or (lambda _: None))
        desk.start()
        opened.append(desk)
        return desk

    yield start
    for desk in opened:
        desk.close()


def _call(desk: lobby.Lobby, method: str, path: str, body: dict | None = None, token: object = OWN
          ) -> tuple[int, str, bytes]:
    host, port = desk.address
    conn = http.client.HTTPConnection(host, port, timeout=30)
    headers = {} if token is None else {"Authorization": f"Bearer {desk.token if token is OWN else token}"}
    data = None if body is None else json.dumps(body).encode()
    try:
        conn.request(method, path, body=data, headers=headers)
        resp = conn.getresponse()
        return resp.status, resp.getheader("Content-Type") or "", resp.read()
    finally:
        conn.close()


def _json(desk: lobby.Lobby, method: str, path: str, body: dict | None = None, token: object = OWN
          ) -> tuple[int, dict]:
    status, _, raw = _call(desk, method, path, body, token)
    return status, json.loads(raw)


def _saved(root: Root, task_id: str, created: str, **fields: Any) -> Task:
    task = Task(id=task_id, thought=f"the thought of {task_id}", workspace="/nowhere", created=created, **fields)
    save(root, task)
    return task


def _git(workspace: Path, *args: str, data: bytes | None = None, index: Path | None = None) -> str:
    env = {**os.environ, "GIT_INDEX_FILE": str(index)} if index is not None else None
    proc = subprocess.run(["git", *GIT_ID, *args], cwd=workspace, input=data, capture_output=True, check=True, env=env)
    return proc.stdout.decode().strip()


def _documents(workspace: Path, task_id: str, files: dict[str, bytes]) -> str:
    """`task/<task_id>` one commit past HEAD adding `files`, built in a private index (as `test_artifacts` builds it):
    the working tree and HEAD never move. Returns the tip."""
    base, index = _git(workspace, "rev-parse", "HEAD"), workspace.parent / "index"
    _git(workspace, "read-tree", base, index=index)
    for rel, data in files.items():
        blob = _git(workspace, "hash-object", "-w", "--stdin", data=data)
        _git(workspace, "update-index", "--add", "--cacheinfo", f"100644,{blob},{rel}", index=index)
    tree = _git(workspace, "write-tree", index=index)
    index.unlink()
    tip = _git(workspace, "commit-tree", tree, "-p", base, "-m", "documents")
    _git(workspace, "update-ref", f"refs/heads/task/{task_id}", tip)
    return tip


def _files(root: Root) -> dict[str, bytes]:
    """Every file under the brain root, with its bytes."""
    return {str(p.relative_to(root.path)): p.read_bytes() for p in sorted(root.path.rglob("*")) if p.is_file()}


# --- DoD 2: the list ---


def test_sessions_lists_every_record_that_loads_newest_first_and_live_only_where_a_brain_answers(root: Root, served):
    closed = _saved(root, "task-20261020T100000", "2026-10-20T10:00:00+00:00", terminal="stopped",
                    stop_reason="closed", tick=4, cost_usd=1.25, standing="It is settled.",
                    moves=[Move("m1", "rest", "rest", 0.1, "rested")])
    parked = _saved(root, "task-20261021T100000", "2026-10-21T10:00:00+00:00", terminal="interrupted",
                    project="demo", mode="hold", held=Held(kind=EXPLORE, move="m2", brief="try b.txt"))
    state.add_question(parked, "director", "question", "which file?")
    save(root, parked)
    running = _saved(root, "task-20261022T100000", "2026-10-22T10:00:00+00:00", paused=True,
                     moves=[Move("m1", "ideas", "ideas i1, i2", 0.2, "proposed",
                                 ideas=[Idea("i1", "a", "one call", "o1"), Idea("i2", "b", "nothing", "o1")])],
                     calls=[Call("c1", "director", 1, 2.0, tick=1, outcome="ok"),
                            Call("c2", "worker", 1, 4.0, unit="u-1", tick=2, started="2026-10-22T10:05:00+00:00")])
    _console_up(root, running.id)  # this test process's pid: alive
    root.state_dir("task-20261023T100000").mkdir(parents=True)
    root.task_file("task-20261023T100000").write_text("{not json", encoding="utf-8")  # the newest, and corrupt
    set_current(root, parked.id)
    echoed: list[str] = []
    status, body = _json(served(echo=echoed.append), "GET", "/sessions")
    assert status == 200 and body["current"] == parked.id
    assert [s["task"] for s in body["sessions"]] == [running.id, parked.id, closed.id]  # the corrupt one skipped
    assert all(set(s) == SUMMARY for s in body["sessions"])
    by = {s["task"]: s for s in body["sessions"]}
    assert by[closed.id] == {
        "task": closed.id, "thought": f"the thought of {closed.id}", "project": "_root", "workspace": "/nowhere",
        "created": "2026-10-20T10:00:00+00:00", "tick": 4, "cost_usd": 1.25, "terminal": "stopped",
        "stop_reason": "closed", "paused": False, "mode": "auto", "awaiting_thought": False, "open_questions": 0,
        "held": None, "held_approved": None, "moves": 1, "last_move": "rest", "last_ideas": 0, "doing": None,
        "standing": "It is settled.", "live": False, "pid": None}
    assert {k: by[parked.id][k] for k in ("terminal", "project", "mode", "open_questions", "held", "held_approved",
                                           "live", "pid", "last_move", "last_ideas")} == {
        "terminal": "interrupted", "project": "demo", "mode": "hold", "open_questions": 1, "held": "explore m2",
        "held_approved": False, "live": False, "pid": None, "last_move": None, "last_ideas": 0}
    assert (by[running.id]["live"], by[running.id]["pid"], by[running.id]["paused"]) == (True, os.getpid(), True)
    # Decision 12's fields on a populated record: the latest move's kind and its ideas, the pending call as `doing`.
    assert (by[running.id]["last_move"], by[running.id]["last_ideas"]) == ("ideas", 2)
    assert by[running.id]["doing"] == {"call": "c2", "seat": "worker", "unit": "u-1", "attempt": 1, "tick": 2,
                                       "since": "2026-10-22T10:05:00+00:00"}
    assert [line for line in echoed if "task-20261023T100000" in line and "does not load" in line]


def test_an_approved_hold_reads_held_approved_true_in_the_list_and_in_the_record(root: Root, served):
    """Audit 2026-10-23, C5: `held_approved` was pinned only false. A released hold the runtime has not yet acted on
    reads true in the summary and in the record's `held`."""
    released = _saved(root, "task-20261022T100000", "2026-10-22T10:00:00+00:00", mode="hold",
                      held=Held(kind=EXPLORE, move="m2", brief="try b.txt", approved=True))
    desk = served()
    status, body = _json(desk, "GET", "/sessions")
    assert status == 200
    [row] = body["sessions"]
    assert (row["task"], row["held"], row["held_approved"]) == (released.id, "explore m2", True)
    status, record = _json(desk, "GET", f"/session?task={released.id}")
    assert status == 200 and record["held"]["approved"] is True


def test_a_skipped_session_is_echoed_once_until_it_lists_again(root: Root, served):
    """Audit 2026-10-23, H1: the skip echo fired on every `/sessions` poll, so one broken lock buried the hand-run
    lobby's stdout under a line every 2 s per window. Once per (session, message); forgotten when the session lists
    again, so a lock repaired and then broken anew is told again. The unloadable record shares the path."""
    odd = _saved(root, "task-20261022T100000", "2026-10-22T10:00:00+00:00", terminal="interrupted")
    _console_up(root, odd.id, locked=False)
    lock = root.state_dir(odd.id) / "lock"
    lock.mkdir()
    root.state_dir("task-20261023T100000").mkdir(parents=True)
    root.task_file("task-20261023T100000").write_text("{not json", encoding="utf-8")
    echoed: list[str] = []
    desk = served(echo=echoed.append)

    def told(task_id: str) -> int:
        return len([line for line in echoed if task_id in line and "skipped" in line])
    for _ in range(3):
        assert _json(desk, "GET", "/sessions")[0] == 200
    assert (told(odd.id), told("task-20261023T100000")) == (1, 1)
    lock.rmdir()
    assert odd.id in [s["task"] for s in _json(desk, "GET", "/sessions")[1]["sessions"]]  # listed again: forgotten
    lock.mkdir()
    for _ in range(2):
        _json(desk, "GET", "/sessions")
    assert (told(odd.id), told("task-20261023T100000")) == (2, 1)


class _PausingSet(set):
    """A set whose iteration pauses after its first element until released: holds a forget inside its pass."""

    def __init__(self, items: Iterator[Any] | list[Any]) -> None:
        super().__init__(items)
        self.entered, self.release = threading.Event(), threading.Event()

    def __iter__(self) -> Iterator[Any]:
        first = True
        for item in super().__iter__():
            yield item
            if first:
                first = False
                self.entered.set()
                self.release.wait(5)


def test_a_skip_told_while_a_forget_runs_neither_breaks_the_forget_nor_is_lost(root: Root, served):
    """Audit 2026-10-23, I2: `_told` is shared across the server's handler threads. A forget passing over it while
    another request's skip adds to it raised `Set changed size during iteration` (uncaught by `handle`: a dropped reply
    and a traceback on the lobby's stdout). Under the lock the add waits for the forget and lands in the new set."""
    desk = served()
    desk._told = _PausingSet([("task-a", "one"), ("task-b", "two")])
    failed: list[BaseException] = []

    def forget() -> None:
        try:
            desk._forget("task-a")
        except BaseException as exc:  # the race's RuntimeError, captured for the assertion
            failed.append(exc)
    forgetter = threading.Thread(target=forget)
    forgetter.start()
    assert desk._told.entered.wait(5)  # the forget is inside its pass
    teller = threading.Thread(target=lambda: desk._skip("task-c", "three"))
    teller.start()
    teller.join(0.3)  # unlocked, the add lands now, inside the pass
    desk._told.release.set()
    forgetter.join(5)
    teller.join(5)
    assert failed == []
    assert ("task-c", "three") in desk._told and ("task-a", "one") not in desk._told


def test_a_tie_on_created_goes_to_the_higher_id_and_a_console_file_whose_pid_is_dead_is_not_live(root: Root, served):
    first = _saved(root, "task-20261020T100000", "2026-10-20T10:00:00+00:00")
    second = _saved(root, "task-20261020T100001", "2026-10-20T10:00:00+00:00")
    gone = subprocess.Popen(["true"])
    gone.wait()  # exited and reaped: signal 0 no longer reaches it
    _console_up(root, second.id, pid=gone.pid)  # left behind by a kill
    root.lobby_log_dir.mkdir(parents=True)  # the lobby's own folder under state/ is no session
    (root.lobby_log_dir / "launch-x.log").write_text("x\n")
    status, body = _json(served(), "GET", "/sessions")
    assert status == 200 and body["current"] is None  # no state/current
    assert [(s["task"], s["live"], s["pid"]) for s in body["sessions"]] == [(second.id, False, gone.pid),
                                                                           (first.id, False, None)]


# --- DoD 3: one session, offline ---


def test_session_returns_the_record_as_the_console_snapshots_it(root: Root, served):
    closed = _saved(root, "task-20261020T100000", "2026-10-20T10:00:00+00:00", terminal="stopped",
                    stop_reason="closed", moves=[Move("m1", "rest", "rest", 0.1, "rested")])
    state.append_log(root, closed.id, {"event": "start"})
    desk = served()
    status, snap = _json(desk, "GET", f"/session?task={closed.id}")
    assert status == 200
    assert {"task", "thought", "moves", "units", "artifacts", "ceilings", "doing", "log_bytes", "transcript_bytes"} <= set(snap)
    assert (snap["task"], snap["live"], snap["terminal"], snap["stop_reason"]) == (closed.id, False, "stopped", "closed")
    assert snap["thought"]["thought"] == closed.thought and [m["id"] for m in snap["moves"]] == ["m1"]
    assert (snap["log_bytes"], snap["transcript_bytes"]) == (root.log_file(closed.id).stat().st_size, 0)
    assert snap == {**console.snapshot(state.load(root, closed.id), Caps(), False),
                    "log_bytes": snap["log_bytes"], "transcript_bytes": 0}
    for query, code in (("?task=task-20991231T000000", 404), ("", 400), ("?task=", 400),
                        (f"?task={closed.id}&task={closed.id}", 400), ("?task=../x", 404), ("?task=..", 404),
                        ("?task=.", 404), ("?task=state%2F..", 404)):
        status, body = _json(desk, "GET", f"/session{query}")
        assert (status, body["ok"]) == (code, False), query
    assert _json(desk, "POST", f"/session?task={closed.id}", {})[0] == 405
    assert _json(desk, "GET", "/nowhere")[0] == 404


def test_an_ended_sessions_record_read_offline_carries_the_registry_as_data(root: Root, served):
    """2026-10-26, the graph-view remake: `graph` is the node registry on the wire (the brain owns it, a client draws
    it), the same on a record read offline as in a live snapshot."""
    closed = _saved(root, "task-20261026T100000", "2026-10-26T10:00:00+00:00", terminal="stopped",
                    stop_reason="closed", moves=[Move("m1", "rest", "rest", 0.1, "rested")])
    status, snap = _json(served(), "GET", f"/session?task={closed.id}")
    assert (status, snap["live"]) == (200, False)
    view = snap["graph"]
    assert [n["id"] for n in view["nodes"]] == [n.key for n in graph.NODES]
    assert view["nodes"][1] == {"id": "block", "name": "block", "fired_by": "runtime", "always": True, "after": ["repair"],
                                "before": [], "brings": [], "within": [], "queued": False}
    assert view["fires"] == {kind: list(nodes) for kind, nodes in graph.FIRES.items()}
    assert view["shapes"] == {name: graph.walk(name) for name in graph.SHAPES}
    assert json.loads(json.dumps(view)) == view == console.snapshot(closed, Caps(), True)["graph"]


def test_session_is_live_while_a_brain_answers_for_it(root: Root, served):
    running = _saved(root, "task-20261022T100000", "2026-10-22T10:00:00+00:00")
    _console_up(root, running.id)
    assert _json(served(), "GET", f"/session?task={running.id}")[1]["live"] is True


# --- DoD 4: a session's documents ---


def test_file_serves_a_committed_document_byte_identical_and_nothing_off_the_list(root: Root, served, tmp_path: Path,
                                                                                  monkeypatch: pytest.MonkeyPatch):
    workspace = make_workspace(tmp_path / "ws")
    task = Task(id="task-20261022T120000", thought="t", workspace=str(workspace), created="2026-10-22T12:00:00+00:00",
                terminal="stopped", stop_reason="closed")
    save(root, task)
    _documents(workspace, task.id, {"notes/x.md": BYTES, "notes/y.md": b"y\n"})
    desk, branch = served(), f"task/{task.id}"
    assert _call(desk, "GET", f"/file?task={task.id}&branch={branch}&path=notes/x.md") == (
        200, "application/octet-stream", BYTES)
    for query in (f"branch={branch}&path=README.md", f"branch={branch}&path=../ws/notes/x.md",
                  "branch=main&path=README.md", f"branch={branch}&path=notes/none.md"):
        status, body = _json(desk, "GET", f"/file?task={task.id}&{query}")
        assert (status, body["ok"]) == (404, False), query
    for query, code in ((f"branch={branch}&path=notes/x.md", 400), (f"task=&branch={branch}&path=notes/x.md", 400),
                        (f"task=../x&branch={branch}&path=notes/x.md", 404),
                        (f"task=task-20991231T000000&branch={branch}&path=notes/x.md", 404),
                        (f"task={task.id}&branch={branch}", 400)):
        status, body = _json(desk, "GET", f"/file?{query}")
        assert (status, body["ok"]) == (code, False), query
    monkeypatch.setattr(console, "FILE_BOUND", len(b"y\n"))  # the bound the lobby serves under is the console's
    assert _json(desk, "GET", f"/file?task={task.id}&branch={branch}&path=notes/x.md") == (
        413, {"ok": False, "error": "notes/x.md is over the 2-byte bound"})
    assert _call(desk, "GET", f"/file?task={task.id}&branch={branch}&path=notes/y.md")[::2] == (200, b"y\n")


# --- DoD 5: start ---


def _start_argv(root: Root, thought: str, project: str = "_root", mode: str = "auto",
                workspace: str | None = None) -> list[str]:
    # Options first, then `--`, then the thought (audit 2026-10-23, B3): a thought starting with `-` stays a thought.
    return ([sys.executable, "-m", "protean.cli", "--brain", str(root.path), "think", "--project", project,
             "--mode", mode] + (["--workspace", workspace] if workspace is not None else []) + ["--", thought])


def test_start_refuses_a_bad_body_and_spawns_nothing(root: Root, served, tmp_path: Path):
    desk = served()
    for body in ({}, {"thought": ""}, {"thought": "   "}, {"thought": 7}, {"thought": "x" * 8001},
                 {"thought": "t", "mode": "fast"}, {"thought": "t", "mode": 1},
                 {"thought": "t", "workspace": str(tmp_path / "missing")}, {"thought": "t", "workspace": ""},
                 {"thought": "t", "workspace": 3}, {"thought": "t", "project": ""}, {"thought": "t", "project": 5}):
        status, answer = _json(desk, "POST", "/start", body)
        assert (status, answer["ok"]) == (400, False), body
    assert _call(desk, "POST", "/start")[0] == 400  # no body at all
    assert desk.spawn.calls == [] and not root.lobby_log_dir.exists()
    assert _json(desk, "GET", "/start")[0] == 405


def test_a_thought_the_os_cannot_pass_as_an_argument_is_a_400_and_the_lobby_keeps_serving(root: Root, served):
    """Audit 2026-10-23, B5: a NUL or a lone surrogate in the thought (or the project) made `Popen` raise ValueError,
    uncaught, so the request died without an answer."""
    desk = served()
    for body in ({"thought": "a\x00b"}, {"thought": "lone \ud800 half"}, {"thought": "t", "project": "p\x00"},
                 {"thought": "t", "project": "\udfff"}):
        status, answer = _json(desk, "POST", "/start", body)
        assert (status, answer["ok"]) == (400, False), body
        assert isinstance(answer["error"], str) and answer["error"]
    assert desk.spawn.calls == [] and not root.lobby_log_dir.exists()
    assert _json(desk, "GET", "/sessions")[0] == 200


def test_a_thought_that_starts_with_a_dash_reaches_the_child_as_a_thought(root: Root, served):
    """Audit 2026-10-23, B3 then C2: `-h`, `--help`, `--mode=hold` or `-x y` typed as a thought is a thought. The spawned
    argv carries the options first, then `--`, then the thought, and the brain's own parser reads it so."""
    from protean import cli

    desk = served()
    desk.wait_seconds, desk.poll_seconds = 0.1, 0.02  # the fake brain never comes up: each start answers after the wait
    for thought in ("-h", "--help", "--mode=hold", "-x y"):
        assert _json(desk, "POST", "/start", {"thought": thought})[0] == 200, thought
        argv, _ = desk.spawn.calls[-1]
        assert argv[-2:] == ["--", thought]
        args = cli.build_parser().parse_args(argv[3:])
        assert (args.verb, args.thought, args.mode) == ("think", thought, "auto"), thought


def test_a_spawn_that_refuses_its_argv_is_a_400_with_no_log_left_behind(root: Root, served):
    desk = served()

    def refuse(argv: list[str], kwargs: dict[str, Any]) -> Child:
        raise ValueError("embedded null byte")
    desk.spawn.act = refuse
    status, answer = _json(desk, "POST", "/start", {"thought": "fine"})
    assert (status, answer["ok"]) == (400, False) and "could not start" in answer["error"]
    assert list(root.lobby_log_dir.iterdir()) == [] and desk.children == []
    assert _json(desk, "GET", "/sessions")[0] == 200


def test_a_console_file_naming_a_live_pid_without_the_session_lock_is_not_live(root: Root, served):
    """Audit 2026-10-23, B2: a reused pid kept a dead session live, so every start and resume answered 409. Live is
    the console file's pid alive AND the session's lock held (`loop.live_holder`), which the kernel drops with the
    brain."""
    stale = _saved(root, "task-20261022T100000", "2026-10-22T10:00:00+00:00", terminal="interrupted")
    _console_up(root, stale.id, locked=False)  # this process's pid answers signal 0, yet no brain holds the lock
    desk = served()
    desk.wait_seconds = 0.2
    status, body = _json(desk, "GET", "/sessions")
    assert status == 200 and [(s["live"], s["pid"]) for s in body["sessions"]] == [(False, os.getpid())]
    assert _json(desk, "GET", f"/session?task={stale.id}")[1]["live"] is False
    assert _json(desk, "POST", "/start", {"thought": "a new one"})[0] == 200
    assert _json(desk, "POST", "/resume", {"task": stale.id})[0] == 200
    assert len(desk.spawn.calls) == 2


class _Blocking:
    """A spawned brain whose `poll` blocks until released, holding a reap inside its pass over the children."""

    pid = 4241

    def __init__(self) -> None:
        self.entered, self.release = threading.Event(), threading.Event()

    def poll(self) -> int | None:
        self.entered.set()
        self.release.wait(5)
        return None


def test_a_spawn_appended_while_a_reap_runs_is_kept(root: Root, served):
    """Audit 2026-10-23, E5 (B6's race itself): `_reap` rebinding the list outside a lock dropped a child a start
    appended while the reap was still polling — the rebind replaced the list the append had landed in. Under the
    children's lock the append waits for the reap and lands in the new list."""
    desk = served()
    blocker = _Blocking()
    desk.children = [blocker]
    reaper = threading.Thread(target=desk._reap)
    reaper.start()
    assert blocker.entered.wait(5)  # the reap is inside its pass, holding what it will rebind
    fresh = Child(pid=4343)
    desk.spawn.act = lambda argv, kwargs: fresh
    launcher = threading.Thread(target=lambda: desk._launch("think", ["x"], lambda: "task-x"))
    launcher.start()
    launcher.join(0.3)  # unlocked, the append lands now, in the list the reap is about to replace
    blocker.release.set()
    reaper.join(5)
    launcher.join(5)
    assert not reaper.is_alive() and not launcher.is_alive()
    assert fresh in desk.children and blocker in desk.children


def test_a_lock_file_left_on_disk_but_held_by_no_one_is_not_live(root: Root, served):
    """Audit 2026-10-23, E6: a SIGKILL leaves `state/<task>/lock` on disk with its pid in it, held by nobody (the
    kernel dropped the flock with the process). Beside a console file naming a live pid it is still not live."""
    stale = _saved(root, "task-20261022T100000", "2026-10-22T10:00:00+00:00", terminal="interrupted")
    _console_up(root, stale.id, locked=False)
    (root.state_dir(stale.id) / "lock").write_text("12345\n", encoding="utf-8")
    desk = served()
    desk.wait_seconds = 0.2
    status, body = _json(desk, "GET", "/sessions")
    assert status == 200 and [(s["live"], s["pid"]) for s in body["sessions"]] == [(False, os.getpid())]
    assert _json(desk, "POST", "/start", {"thought": "a new one"})[0] == 200
    assert len(desk.spawn.calls) == 1


def test_a_lock_that_cannot_be_probed_is_skipped_from_the_list_and_refuses_what_needs_it(root: Root, served):
    """Brief 7 item 5, amended by brief 8 F2 (decision 29): a lock path the probe cannot open (here a directory) is
    neither live nor not live. `/sessions` skips that one session and echoes it, as it skips a record that does not
    load, so Home still lists every healthy one; `/session` on that session, and `/start` / `/resume` while any
    session's lock is unprobeable, answer 500 (a start must not guess), nothing is spawned, and the lobby serves on."""
    odd = _saved(root, "task-20261022T100000", "2026-10-22T10:00:00+00:00", terminal="interrupted")
    fine = _saved(root, "task-20261021T100000", "2026-10-21T10:00:00+00:00", terminal="interrupted")
    _console_up(root, odd.id, locked=False)
    lock = root.state_dir(odd.id) / "lock"
    lock.mkdir()
    echoed: list[str] = []
    desk = served(echo=echoed.append)
    desk.wait_seconds = 0.2
    status, body = _json(desk, "GET", "/sessions")
    assert status == 200 and [s["task"] for s in body["sessions"]] == [fine.id]
    [line] = [line for line in echoed if odd.id in line]
    assert "skipped" in line and "cannot be probed" in line and "Is a directory" in line
    status, body = _json(desk, "GET", f"/session?task={odd.id}")
    assert (status, body["ok"]) == (500, False) and odd.id in body["error"]
    for route, request in (("/start", {"thought": "a new one"}), ("/resume", {"task": fine.id})):
        status, body = _json(desk, "POST", route, request)
        assert (status, body["ok"]) == (500, False), route
    assert desk.spawn.calls == []
    lock.rmdir()
    assert [s["task"] for s in _json(desk, "GET", "/sessions")[1]["sessions"]] == [odd.id, fine.id]
    assert _json(desk, "GET", f"/session?task={odd.id}")[0] == 200
    assert _json(desk, "POST", "/start", {"thought": "a new one"})[0] == 200


def test_the_list_answers_while_a_start_waits_for_its_brain(root: Root, served):
    """Audit 2026-10-23, B6: reaping takes the children's own lock, never the spawn lock a start holds for its whole
    wait, so a client polling the list during a launch is answered at once."""
    desk = served()
    desk.wait_seconds, desk.poll_seconds = 1.5, 0.05  # the fake brain never comes up: the start waits it out
    answered: list[int] = []
    starter = threading.Thread(target=lambda: answered.append(_json(desk, "POST", "/start", {"thought": "t"})[0]))
    starter.start()
    deadline = time.monotonic() + 2
    while not desk.spawn.calls and time.monotonic() < deadline:
        time.sleep(0.02)
    began = time.monotonic()
    assert _json(desk, "GET", "/sessions")[0] == 200
    assert time.monotonic() - began < 1.0 and starter.is_alive()
    starter.join()
    assert answered == [200] and len(desk.children) == 1


def test_start_is_refused_while_any_brain_is_live(root: Root, served):
    live = _saved(root, "task-20261022T100000", "2026-10-22T10:00:00+00:00")
    _console_up(root, live.id)
    desk = served()
    assert _json(desk, "POST", "/start", {"thought": "a new one"}) == (
        409, {"ok": False, "error": f"a brain is live: {live.id}"})
    assert desk.spawn.calls == [] and not root.lobby_log_dir.exists()


def test_start_spawns_exactly_one_detached_think_with_the_contracts_argv(root: Root, served, tmp_path: Path):
    workspace = make_workspace(tmp_path / "ws")
    set_current(root, "task-20261021T100000")  # an older session, not live: `current` must move off it
    desk = served()
    desk.spawn.act = _brain_up(root, "task-20261022T100000")
    status, answer = _json(desk, "POST", "/start", {"thought": "how should b.txt read?", "project": "demo",
                                                     "mode": "hold", "workspace": str(workspace)})
    assert (status, answer["ok"], answer["task"], answer["pid"]) == (200, True, "task-20261022T100000", 4242)
    (argv, kwargs), = desk.spawn.calls
    assert argv == _start_argv(root, "how should b.txt read?", "demo", "hold", workspace=str(workspace))
    assert argv[-2:] == ["--", "how should b.txt read?"]
    assert set(kwargs) == {"stdin", "stdout", "stderr", "start_new_session", "env"}  # no cwd: the lobby's own
    assert (kwargs["stdin"], kwargs["stderr"], kwargs["start_new_session"]) == (subprocess.DEVNULL, subprocess.STDOUT,
                                                                                True)
    assert kwargs["env"]["PYTHONUNBUFFERED"] == "1"
    assert {k: v for k, v in kwargs["env"].items() if k != "PYTHONUNBUFFERED"} == {
        k: v for k, v in os.environ.items() if k != "PYTHONUNBUFFERED"}  # inherited, plus PYTHONUNBUFFERED
    assert desk.token not in " ".join(argv) and desk.token not in " ".join(kwargs["env"].values())
    log = Path(answer["log"])
    assert kwargs["stdout"].name == str(log) and log.parent == root.lobby_log_dir and log.is_file()
    assert re.fullmatch(r"launch-\d{8}T\d{6}Z(-\d+)?\.log", log.name)

    root.console_file("task-20261022T100000").unlink()  # that brain exited: nothing is live
    desk.spawn.act = _brain_up(root, "task-20261022T100001")
    status, answer = _json(desk, "POST", "/start", {"thought": "and without a workspace"})
    assert (status, answer["task"]) == (200, "task-20261022T100001")
    assert desk.spawn.calls[1][0] == _start_argv(root, "and without a workspace")  # project _root, mode auto
    assert len(desk.spawn.calls) == 2 and Path(answer["log"]) != log


def test_start_answers_task_null_when_the_wait_runs_out_and_500_when_the_brain_exits_nonzero(root: Root, served):
    desk = served()
    desk.wait_seconds = 0.3
    started = time.monotonic()
    status, answer = _json(desk, "POST", "/start", {"thought": "a brain that never comes up"})
    assert (status, answer["ok"], answer["task"], answer["pid"]) == (200, True, None, 4242)
    assert time.monotonic() - started < 5 and len(desk.spawn.calls) == 1

    said = "".join(f"line {n}: protean: something refused\n" for n in range(40))

    def exits_2(argv: list[str], kwargs: dict[str, Any]) -> Child:
        kwargs["stdout"].write(said.encode())
        return Child(code=2)
    desk.spawn.act = exits_2
    assert _json(desk, "POST", "/start", {"thought": "a brain that fails"}) == (
        500, {"ok": False, "error": f"protean think exited 2: {said[-600:]}"})
    assert len(desk.spawn.calls) == 2


# --- DoD 6: resume ---


def test_resume_refuses_what_it_must_then_spawns_one_resume_of_that_session(root: Root, served):
    closed = _saved(root, "task-20261020T100000", "2026-10-20T10:00:00+00:00", terminal="stopped",
                    stop_reason="closed")
    parked = _saved(root, "task-20261021T100000", "2026-10-21T10:00:00+00:00", terminal="interrupted")
    other = _saved(root, "task-20261022T100000", "2026-10-22T10:00:00+00:00")
    desk = served()
    for body in ({}, {"task": ""}, {"task": 5}):
        assert _json(desk, "POST", "/resume", body)[0] == 400, body
    for task_id in ("task-20991231T000000", "../x", ".."):
        assert _json(desk, "POST", "/resume", {"task": task_id})[0] == 404, task_id
    assert _json(desk, "POST", "/resume", {"task": closed.id}) == (
        409, {"ok": False, "error": f"task {closed.id} is closed (closed)"})
    _console_up(root, other.id)  # another session's brain is live
    assert _json(desk, "POST", "/resume", {"task": parked.id}) == (
        409, {"ok": False, "error": f"a brain is live: {other.id}"})
    assert desk.spawn.calls == []

    root.console_file(other.id).unlink()
    _console_up(root, parked.id, pid=999_999)  # a console file a kill left behind: not live, and not the new one
    desk.spawn.act = _brain_up(root, parked.id, current=False)
    status, answer = _json(desk, "POST", "/resume", {"task": parked.id})
    assert (status, answer["ok"], answer["task"], answer["pid"]) == (200, True, parked.id, 4242)
    (argv, kwargs), = desk.spawn.calls
    assert argv == [sys.executable, "-m", "protean.cli", "--brain", str(root.path), "resume", parked.id]
    assert (kwargs["stdin"], kwargs["start_new_session"], kwargs["env"]["PYTHONUNBUFFERED"]) == (
        subprocess.DEVNULL, True, "1")


def test_a_real_spawn_that_exits_nonzero_answers_500_with_its_log(root: Root, served):
    """The default spawn, `subprocess.Popen`, end to end: `protean resume` of a session whose id it refuses exits 2 at
    once, and the lobby answers with the tail of the log it wrote."""
    _saved(root, "x.y", "2026-10-22T10:00:00+00:00")
    desk = served(spawn=subprocess.Popen)
    status, answer = _json(desk, "POST", "/resume", {"task": "x.y"})
    assert (status, answer["ok"]) == (500, False)
    assert answer["error"].startswith("protean resume exited 2: ") and "no session x.y" in answer["error"]
    (log,) = root.lobby_log_dir.iterdir()
    assert "no session x.y" in log.read_text()


# --- DoD 7: the token ---


def test_every_route_without_the_token_is_refused_and_nothing_is_spawned_or_written(root: Root, served,
                                                                                     tmp_path: Path):
    workspace = make_workspace(tmp_path / "ws")
    parked = _saved(root, "task-20261021T100000", "2026-10-21T10:00:00+00:00", terminal="interrupted")
    desk = served()
    desk.spawn.act = _brain_up(root, "task-20261022T100000")
    before = _files(root)
    requests = [("GET", "/sessions", None), ("GET", f"/session?task={parked.id}", None),
                ("GET", f"/file?task={parked.id}&branch=task/{parked.id}&path=a.md", None),
                ("POST", "/start", {"thought": "t", "workspace": str(workspace)}),
                ("POST", "/resume", {"task": parked.id}), ("PUT", "/start", {"thought": "t"}),
                ("POST", "/nowhere", {}), ("GET", "/nowhere", None)]
    for method, path, body in requests:
        for token in (None, "not-the-token", desk.token[:-1], desk.token + "x"):
            assert _json(desk, method, path, body, token) == (401, {"ok": False, "error": "unauthorized"}), (path, token)
    assert desk.spawn.calls == [] and _files(root) == before and not root.lobby_log_dir.exists()


# --- DoD 8: the lobby's file and its address ---


def _outward_address() -> str | None:
    """This host's address on its outward interface (no packet is sent), or None without a route."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
            probe.connect(("192.0.2.1", 9))
            address = probe.getsockname()[0]
    except OSError:
        return None
    return None if address.startswith("127.") else address


def test_the_lobby_listens_on_127_0_0_1_only_and_its_file_is_private_and_gone_after_close(root: Root, served):
    desk = served()
    assert desk.address[0] == "127.0.0.1" and desk.server.socket.family == socket.AF_INET
    assert json.loads(root.lobby_file.read_text()) == {"host": "127.0.0.1", "port": desk.address[1],
                                                        "token": desk.token, "pid": os.getpid()}
    assert stat.S_IMODE(root.lobby_file.stat().st_mode) == 0o600
    assert _json(desk, "GET", "/sessions")[0] == 200
    with pytest.raises(ConnectionRefusedError):
        socket.create_connection(("::1", desk.address[1]), timeout=2).close()
    outward = _outward_address()
    if outward is not None:
        with pytest.raises(OSError):
            socket.create_connection((outward, desk.address[1]), timeout=2).close()
    desk.close()
    assert not root.lobby_file.exists()


def test_close_leaves_a_lobby_file_that_is_no_longer_its_own(root: Root, served):
    desk = served()
    console._write_private(root.lobby_file, {"host": "127.0.0.1", "port": 1, "token": "another lobby's", "pid": 1})
    desk.close()
    assert json.loads(root.lobby_file.read_text())["token"] == "another lobby's"


def test_protean_lobby_serves_until_sigterm_then_closes_cleanly(root: Root):
    """The verb, as `python -m protean.cli` runs it: its two lines, a lobby that answers, and SIGTERM (the CLI's
    handler raises KeyboardInterrupt) closes it with exit 0 and its file removed."""
    root.path.mkdir(parents=True)
    proc = subprocess.Popen([sys.executable, "-m", "protean.cli", "--brain", str(root.path), "lobby"],
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                            env={**os.environ, "PYTHONUNBUFFERED": "1"})
    lines: list[str] = []
    reader = threading.Thread(target=lambda: lines.extend(proc.stdout), daemon=True)
    reader.start()
    try:
        deadline = time.monotonic() + 20
        while len(lines) < 2 and time.monotonic() < deadline and proc.poll() is None:
            time.sleep(0.05)
        assert [line.rstrip("\n") for line in lines] == [f"lobby served on {root.path.resolve()}",
                                                         "lobby: state/lobby.json"]
        info = json.loads(root.lobby_file.read_text())
        conn = http.client.HTTPConnection(info["host"], info["port"], timeout=10)
        conn.request("GET", "/sessions", headers={"Authorization": f"Bearer {info['token']}"})
        resp = conn.getresponse()
        assert (resp.status, json.loads(resp.read())) == (200, {"current": None, "sessions": [],
                                                                "capabilities": ["history"]})
        conn.close()
        proc.send_signal(signal.SIGTERM)
        assert proc.wait(timeout=20) == 0
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()
    reader.join(5)
    assert not root.lobby_file.exists() and info["pid"] == proc.pid


# --- DoD 9: `doing` ---


def test_doing_is_the_last_pending_call_and_none_once_every_call_is_resolved():
    task = Task(id="task-20261022T100000", thought="t", workspace="/nowhere", tick=3)
    thinker = budget.open_call(task, "director", 1, 2.0, "m")
    budget.resolve(thinker, 0.1, {"total_cost_usd": 0.1, "session_id": "s"})
    assert console.snapshot(task, Caps(), False)["doing"] is None
    worker = budget.open_call(task, "worker", 2, 4.0, "m", "u-1")
    assert re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d\+00:00", worker.started)
    assert console.snapshot(task, Caps(), False)["doing"] == {"call": "c2", "seat": "worker", "unit": "u-1",
                                                              "attempt": 2, "tick": 3, "since": worker.started}
    budget.resolve(worker, 4.0, None)  # ended without a receipt: resolved, no longer in flight
    assert console.snapshot(task, Caps(), False)["doing"] is None
    explore = budget.open_call(task, "worker", 1, 4.0, "m")
    assert console.snapshot(task, Caps(), False)["doing"]["unit"] is None  # an explore's call names no unit
    older = json.loads(state.to_json(task))
    for call in older["calls"]:
        del call["started"]  # a record from before the field
    loaded = state.from_json(json.dumps(older))
    assert loaded.calls[-1].started is None and console.snapshot(loaded, Caps(), False)["doing"] == {
        "call": explore.id, "seat": "worker", "unit": None, "attempt": 1, "tick": 3, "since": None}


# --- DoD 10: `launch` ---


def _seed(base: Path) -> Root:
    root = Root(base / "brain")
    root.path.mkdir(parents=True)
    for prompt in (root.thinker_prompt, root.editor_prompt, root.reader_prompt):
        prompt.write_text("p\n")
    return root


@pytest.mark.parametrize(("scenario", "seats"), [("think", {"director", "worker"}),
                                                  ("branch", {"director", "worker", "reader"}),
                                                  ("weaken", {"director", "worker", "reader"})])
def test_every_model_call_logs_one_launch_before_its_call_event(tmp_path: Path, scenario: str, seats: set[str]):
    root = _seed(tmp_path)
    task, _ = dry.run_scenario(scenario, root, make_workspace(tmp_path / "ws"), dry.scenario_runner(scenario),
                               lambda _: None)
    log = state.read_log(root, task.id)
    launches = {e["call"]: (i, e) for i, e in enumerate(log) if e["event"] == "launch"}
    calls = [(i, e) for i, e in enumerate(log) if e["event"] == "call"]
    assert calls and len(launches) == sum(e["event"] == "launch" for e in log)  # one launch per call id
    assert sorted(launches) == sorted(e["id"] for _, e in calls)  # exactly one per call event
    for at, call in calls:
        before, launch = launches[call["id"]]
        assert before < at, call["id"]
        assert {k: launch[k] for k in ("seat", "unit", "attempt", "tick")} == {
            k: call[k] for k in ("seat", "unit", "attempt", "tick")}
    assert {launch["seat"] for _, launch in launches.values()} == seats


def test_the_launch_lands_after_the_pending_save_so_a_refresh_on_it_sees_the_call(tmp_path: Path):
    root = _seed(tmp_path)
    task = Task(id="task-20261022T100000", thought="t", workspace=str(make_workspace(tmp_path / "ws")))
    save(root, task)
    seen: list[tuple[dict, dict | None]] = []

    def runner(argv: list[str], stdin: str | None, cwd: Path, timeout: int) -> RunResult:
        disk = state.load(root, task.id)  # what a client refreshing on the event reads
        seen.append((state.read_log(root, task.id)[-1], console.snapshot(disk, Caps(), False)["doing"]))
        move = {"rest": True, "state": {"standing": "It rests.", "open": []}}
        return RunResult(0, json.dumps({"type": "result", "is_error": False, "session_id": "s", "total_cost_usd": 0.1,
                                        "result": json.dumps(move)}), "")

    loop.run_tick(task, Caps(), root, runner)
    last, doing = seen[0]
    assert last["event"] == "launch" and doing is not None
    assert (doing["call"], doing["seat"]) == (last["call"], "director") == ("c1", "director")
    assert console.snapshot(task, Caps(), False)["doing"] is None  # resolved once it returned


# --- the lobby's token is as unreadable to the editor and a check command as the console's (audit 2026-10-23, B1) ---


def test_the_read_deny_names_the_lobby_file_beside_the_console_file(root: Root):
    from protean.graph import grade

    console_file = root.console_file("task-20261023T100000")
    clause = grade.read_deny(console_file)
    assert str(console_file.resolve()) in clause and str(root.lobby_file.resolve()) in clause
    assert clause.count("deny file-read*") == 4 and clause.count("(regex #\"^") == 2  # by name, and by temp name (C3)
    assert "/\\.lobby\\.json\\.[^/]*\\.tmp$" in clause and "/\\.console\\.json\\.[^/]*\\.tmp$" in clause
    assert grade.read_deny(None) == ""


@pytest.mark.skipif(shutil.which("sandbox-exec") is None,
                    reason="sandbox-exec is absent on this host, so the real profiles cannot be exercised")
def test_under_the_real_profiles_neither_the_lobby_file_nor_the_console_file_can_be_read(root: Root, tmp_path: Path):
    """The editor's, a check command's and the reader's profiles (audit 2026-10-23, B1, then C3/C4): the two token files
    by name and by the private temp name they are written through, with a sibling file as the positive control, so a
    profile that failed to compile could not pass this vacuously."""
    from protean.graph import grade, reader, worker

    task_id = "task-20261023T100000"
    root.state_dir(task_id).mkdir(parents=True, exist_ok=True)
    root.console_file(task_id).write_text('{"token": "console-secret"}', encoding="utf-8")
    root.lobby_file.write_text('{"token": "lobby-secret"}', encoding="utf-8")
    temps = (root.lobby_file.with_name(".lobby.json.k3q9z1.tmp"),
             root.console_file(task_id).with_name(".console.json.a7b2c3.tmp"))
    for temp in temps:
        temp.write_text('{"token": "temp-secret"}', encoding="utf-8")
    plain = root.state_dir(task_id) / "task.json"
    plain.write_text('{"id": "plain"}', encoding="utf-8")
    clone, workspace = tmp_path / "clone", tmp_path / "ws"
    clone.mkdir()
    workspace.mkdir()
    sandbox = shutil.which("sandbox-exec")
    for profile in (grade._sandbox_profile(clone, root.console_file(task_id)),
                    worker.sandbox_profile(clone, root.console_file(task_id)),
                    reader.sandbox_profile(clone, workspace, root.console_file(task_id))):
        for secret in (root.lobby_file, root.console_file(task_id), *temps):
            done = subprocess.run([sandbox, "-p", profile, "/bin/cat", str(secret)], capture_output=True, text=True)
            assert done.returncode != 0 and "secret" not in done.stdout, (secret, done.stdout, done.stderr)
            assert "Operation not permitted" in done.stderr, (secret, done.stderr)
        done = subprocess.run([sandbox, "-p", profile, "/bin/cat", str(plain)], capture_output=True, text=True)
        assert done.returncode == 0 and "plain" in done.stdout, (done.stdout, done.stderr)
