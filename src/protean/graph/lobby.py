"""The lobby: a standing console for the brain root (`protean lobby`), so a client lists the sessions, reads one offline,
reads its documents, and starts or wakes one, with no brain running.

One process, no model call ever: 127.0.0.1 only, an ephemeral port, a per-process token in `state/lobby.json` (mode
0600, written whole by rename, removed at exit while the token is still its own). The wire and the server are the
console's (`graph/console.py`: HTTP/1.0, one request per connection, the bearer token, JSON errors; its `_Server` and
`_Handler` call `Lobby.handle` as they call `Console.handle`). `GET /sessions`, `GET /session`, `GET /file` and
`GET /history` (one bounded page of a session's log or transcript, `graph/history.py`) read state; `POST /start` and
`POST /resume` spawn one detached `protean think` or `protean resume` of this interpreter, refused while any session's
brain is live, and answer once that brain's console is up or the wait runs out. The
operator typing the thought and pressing send is the go; the lobby itself spends nothing. The wire is
`docs/console-api.md` § The lobby.
"""

from __future__ import annotations

import hmac
import json
import os
import secrets
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlsplit

from protean import loop, state
from protean.graph import console, history
from protean.graph.budget import load_caps
from protean.state import AUTO, MODES, Root, Task

WAIT_SECONDS = 15.0  # how long a spawn waits for the new brain's console
WAIT_POLL_SECONDS = 0.2
LOG_TAIL = 600  # characters of a failed spawn's log in its 500
THOUGHT_LIMIT = console.TEXT_LIMIT
GETS = ("/sessions", "/session", "/file", "/history")
POSTS = ("/start", "/resume")
CAPABILITIES = ("history",)  # what the lobby serves beyond its frozen routes, in `/sessions` (record redesign)
# What a `task.json` that does not load raises (`state.from_json`): a skipped session, never a failed request.
UNLOADABLE = (OSError, ValueError, TypeError, KeyError, AttributeError)
Spawn = Callable[..., Any]  # `subprocess.Popen`'s shape: the argv, then its keyword arguments; returns a child
Echo = Callable[[str], None]


def _silent(_: str) -> None:
    pass


def _one_component(value: object) -> bool:
    """A session id names exactly one directory under `state/`: no `/`, never `.` or `..`."""
    return isinstance(value, str) and value not in ("", ".", "..") and "/" not in value and "\x00" not in value


def _passable(value: str) -> bool:
    """`value` can be one argument of a child process: no NUL, and UTF-8 encodable (no lone surrogate). `Popen`
    raises ValueError on either (audit 2026-10-23, B5)."""
    if "\x00" in value:
        return False
    try:
        value.encode("utf-8")
    except UnicodeEncodeError:
        return False
    return True


def _alive(pid: int) -> bool:
    """The process exists: signal 0 reaches it, or it exists under another user (PermissionError)."""
    try:
        os.kill(pid, 0)
    except PermissionError:
        return True
    except (OSError, OverflowError):
        return False
    return True


class Lobby:
    """The serving side of the lobby. `spawn` starts a brain (default `subprocess.Popen`; the battery injects one that
    records its argv); `wait_seconds` and `poll_seconds` bound the wait for that brain's console."""

    def __init__(self, root: Root, spawn: Spawn = subprocess.Popen, echo: Echo = _silent) -> None:
        self.root, self.spawn, self.echo = root, spawn, echo
        self.token = secrets.token_urlsafe(32)
        self.server: console._Server | None = None
        self.wait_seconds, self.poll_seconds = WAIT_SECONDS, WAIT_POLL_SECONDS
        self.children: list[Any] = []  # spawned brains not yet reaped
        self._spawning = threading.Lock()  # one start or resume at a time: the live check and the spawn are one step
        # `children` is appended by a start and pruned by every request: its own lock, never `_spawning`, which a start
        # holds for its whole wait (a list polled during a launch would block behind it) (audit 2026-10-23, B6).
        self._children = threading.Lock()
        # The skips `/sessions` has echoed, by (session, message): each told once, forgotten when it lists again (H1).
        self._told: set[tuple[str, str]] = set()
        # Handler threads share `_told`: its check-and-add and its forget hold this (audit 2026-10-23, I2).
        self._told_lock = threading.Lock()

    @property
    def address(self) -> tuple[str, int]:
        host, port = self.server.server_address[:2]
        return host, port

    def start(self) -> None:
        self.server = console._Server((console.HOST, 0), console._Handler)
        self.server.console = self  # the handler's duck type: `handle(req, method)`
        threading.Thread(target=self.server.serve_forever, kwargs={"poll_interval": console.POLL_SECONDS},
                         name="protean-lobby", daemon=True).start()
        host, port = self.address
        console._write_private(self.root.lobby_file, {"host": host, "port": port, "token": self.token,
                                                      "pid": os.getpid()})

    def close(self) -> None:
        """Stop serving, then remove `state/lobby.json` if it still carries this lobby's token."""
        if self.server is not None:
            self.server.shutdown()
            self.server.server_close()
        try:
            if json.loads(self.root.lobby_file.read_text(encoding="utf-8")).get("token") == self.token:
                self.root.lobby_file.unlink()
        except (OSError, ValueError, AttributeError):
            pass

    # --- the wire ---

    def handle(self, req: BaseHTTPRequestHandler, method: str) -> None:
        raw = console._raw_body(req)
        header = req.headers.get("Authorization") or ""
        if not (header.startswith("Bearer ") and hmac.compare_digest(header[7:].encode(), self.token.encode())):
            return console._reply(req, 401, {"ok": False, "error": "unauthorized"})
        path, query = req.path.split("?", 1)[0], urlsplit(req.path).query
        if path not in GETS + POSTS:
            return console._reply(req, 404, {"ok": False, "error": f"no endpoint {path}"})
        wants = "GET" if path in GETS else "POST"
        if method != wants:
            return console._reply(req, 405, {"ok": False, "error": f"{path} takes {wants}"})
        self._reap()
        try:
            if path == "/file":
                return console.serve_file(req, self._task(_query_task(path, query)), query)
            if path == "/history":
                status, payload = self.history_page(query)
            elif path == "/sessions":
                status, payload = 200, self.sessions()
            elif path == "/session":
                status, payload = 200, self.session(_query_task(path, query))
            elif path == "/start":
                status, payload = self.start_session(console._parse(raw))
            else:
                status, payload = self.resume_session(console._parse(raw))
        except (console.Refused, history.Refused) as refused:
            status, payload = refused.status, {"ok": False, "error": str(refused)}
        console._reply(req, status, payload)

    # --- reading state ---

    def _ids(self) -> list[str]:
        """Every directory under `state/`, by name."""
        try:
            return sorted(p.name for p in (self.root.path / "state").iterdir() if p.is_dir())
        except FileNotFoundError:
            return []

    def _current(self) -> str | None:
        """The session `state/current` names, or None."""
        try:
            return self.root.current_file.read_text(encoding="utf-8").strip() or None
        except OSError:
            return None

    def live(self, task_id: str) -> tuple[bool, int | None]:
        """Whether a brain serves the session now, and the pid its console file names: live exactly when
        `state/<task>/console.json` exists, its pid is alive, and the session's lock is held (`loop.live_holder`: a
        brain holds it for its whole drive, console included, and the kernel drops it with the process, so a reused
        pid never keeps a dead session live — audit 2026-10-23, B2); the pid is None when there is no such file (or it
        names none). The lock is probed only behind a live pid; when no brain holds it (a stale session whose pid was
        reused) the probe itself takes it for an instant and lets it go. A lock the probe cannot open (a directory, a
        permission error) is neither live nor not live, since reading it as not live could let a second brain start:
        `/sessions` skips that session (brief 8, F2), and `/session` on that session, and `/start` / `/resume` while any
        session's lock is unprobeable, answer `500` (brief 7, item 5)."""
        try:
            info = json.loads(self.root.console_file(task_id).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return False, None
        pid = info.get("pid") if isinstance(info, dict) else None
        if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0:
            return False, None
        if not _alive(pid):
            return False, pid
        try:
            holder = loop.live_holder(self.root, task_id)
        except OSError as exc:
            raise console.Refused(500, f"session {task_id}: its lock cannot be probed: {exc}") from None
        return holder is not None, pid

    def _any_live(self) -> str | None:
        return next((task_id for task_id in self._ids() if self.live(task_id)[0]), None)

    def summary(self, task: Task) -> dict[str, Any]:
        """One session in the list: the record's own facts, enough for a client to label the row without fetching
        the record — the latest move's kind (a pick wait after `ideas`, a thought wait after `rest`), how many ideas
        it offered, and the model call in flight (`doing`, as the snapshot carries it)."""
        live, pid = self.live(task.id)
        last = task.moves[-1] if task.moves else None
        return {"task": task.id, "thought": task.thought, "project": task.project, "workspace": task.workspace,
                "created": task.created, "tick": task.tick, "cost_usd": task.cost_usd, "terminal": task.terminal,
                "stop_reason": task.stop_reason, "paused": task.paused, "mode": task.mode,
                "awaiting_thought": task.awaiting_thought, "open_questions": len(state.open_questions(task)),
                "held": console.held_label(task.held) if task.held is not None else None,
                "held_approved": task.held.approved if task.held is not None else None, "moves": len(task.moves),
                "last_move": last.kind if last is not None else None,
                "last_ideas": len(last.ideas) if last is not None and last.kind == "ideas" else 0,
                "doing": console.doing(task),
                "standing": task.standing, "live": live, "pid": pid}

    def sessions(self) -> dict[str, Any]:
        """`GET /sessions`: every `state/<task>/task.json` that loads, newest `created` first (tie: id descending); one
        that does not load is skipped and echoed, and so is one whose lock cannot be probed (brief 8, F2: one broken
        session never blanks the list; `/session` on that session, and `/start` / `/resume` while any session's lock is
        unprobeable, still answer 500)."""
        found = []
        for task_id in self._ids():
            if not self.root.task_file(task_id).is_file():
                continue
            try:
                task = state.load(self.root, task_id)
            except UNLOADABLE as exc:
                self._skip(task_id, f"its task.json does not load ({exc})")
                continue
            try:
                found.append(self.summary(task))
            except console.Refused as refused:
                self._skip(task_id, str(refused))
                continue
            self._forget(task_id)
        found.sort(key=lambda s: (str(s["created"] or ""), str(s["task"])), reverse=True)
        return {"current": self._current(), "sessions": found, "capabilities": list(CAPABILITIES)}

    def _forget(self, task_id: str) -> None:
        """A session listed: its echoed skips are forgotten, so a later break is told again (H1)."""
        with self._told_lock:
            self._told = {told for told in self._told if told[0] != task_id}

    def _skip(self, task_id: str, why: str) -> None:
        """Echo a session `/sessions` skipped, once per (session, message) until it lists again (audit 2026-10-23, H1:
        a broken lock polled every 2 s per window would bury the lobby's stdout)."""
        with self._told_lock:
            if (task_id, why) in self._told:
                return
            self._told.add((task_id, why))
        self.echo(f"lobby: skipped {task_id}: {why}")

    def _task(self, task_id: str) -> Task:
        """The session `task_id` names, or a 404: not one path component, no `task.json`, or one that does not load."""
        if not _one_component(task_id) or not self.root.task_file(task_id).is_file():
            raise console.Refused(404, f"no session {task_id}")
        try:
            return state.load(self.root, task_id)
        except UNLOADABLE as exc:
            raise console.Refused(404, f"session {task_id} does not load: {exc}") from None

    def session(self, task_id: str) -> dict[str, Any]:
        """`GET /session?task=<id>`: the session's snapshot as the console builds it, `live` as `live` says, the caps
        reloaded for this request, with the log and transcript offsets."""
        task = self._task(task_id)
        try:
            caps = load_caps(self.root)
        except (OSError, ValueError) as exc:
            raise console.Refused(500, f"config: {exc}") from None
        live, _ = self.live(task.id)
        return {**console.snapshot(task, caps, live),
                "log_bytes": history.complete_size(self.root.log_file(task.id)),
                "transcript_bytes": history.complete_size(self.root.transcript_file(task.id))}

    def history_page(self, query: str) -> tuple[int, dict[str, Any]]:
        """`GET /history?task=<id>&source=…`: one page of the session's log or transcript, as the console's `/history`
        reads it (`graph/history.py`), for any session whose record loads, live or not. The query is checked first (a
        key it does not take, or one repeated, is a 400), then `task` as `/session` answers it (missing or empty: 400;
        no such session, or a record that does not load: 404)."""
        found = history.params(query, ("task", *history.KEYS))
        if not found.get("task"):
            raise console.Refused(400, "/history needs task")
        task = self._task(found.pop("task"))
        return history.answer(self.root, task.id, found)

    # --- starting a brain ---

    def start_session(self, body: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        """`POST /start`: validate, refuse while a brain is live, spawn `protean think`, wait for its console."""
        thought = body.get("thought")
        if not isinstance(thought, str) or not thought.strip():
            raise console.Refused(400, "thought must be a non-empty string")
        if len(thought) > THOUGHT_LIMIT:
            raise console.Refused(400, f"thought is longer than {THOUGHT_LIMIT} characters")
        if not _passable(thought):
            raise console.Refused(400, "thought must hold no NUL character and no lone surrogate")
        mode = AUTO if body.get("mode") is None else body["mode"]
        if mode not in MODES:
            raise console.Refused(400, f"mode must be one of {', '.join(MODES)}")
        workspace = body.get("workspace")
        if workspace is not None and not (isinstance(workspace, str) and workspace and _passable(workspace)
                                          and Path(workspace).is_dir()):
            raise console.Refused(400, f"workspace {workspace!r} is not an existing directory")
        project = "_root" if body.get("project") is None else body["project"]
        if not isinstance(project, str) or not project.strip():
            raise console.Refused(400, "project must be a non-empty string")
        if not _passable(project):
            raise console.Refused(400, "project must hold no NUL character and no lone surrogate")
        with self._spawning:
            self._refuse_live()
            before = self._current()
            # Options first, then `--`, then the thought: a thought that starts with `-` is a thought, never an
            # option of the child (audit 2026-10-23, B3).
            argv = ([sys.executable, "-m", "protean.cli", "--brain", str(self.root.path), "think",
                     "--project", project, "--mode", mode]
                    + (["--workspace", workspace] if workspace is not None else []) + ["--", thought])

            def started() -> str | None:  # `current` moved to a session whose console is up
                now = self._current()
                return now if now is not None and now != before and self.root.console_file(now).exists() else None
            return self._launch("think", argv, started)

    def resume_session(self, body: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        """`POST /resume`: the session must exist and be open, and no brain live; spawn `protean resume <task>` and
        wait for that session's console."""
        task_id = body.get("task")
        if not isinstance(task_id, str) or not task_id:
            raise console.Refused(400, "resume needs task")
        task = self._task(task_id)
        if state.is_closed(task):
            raise console.Refused(409, f"task {task.id} is closed ({task.stop_reason or task.terminal})")
        with self._spawning:
            self._refuse_live()
            before = self._console_text(task_id)
            argv = [sys.executable, "-m", "protean.cli", "--brain", str(self.root.path), "resume", task_id]

            def woke() -> str | None:  # a console file for this session, written after the spawn
                now = self._console_text(task_id)
                return task_id if now is not None and now != before else None
            return self._launch("resume", argv, woke)

    def _console_text(self, task_id: str) -> str | None:
        try:
            return self.root.console_file(task_id).read_text(encoding="utf-8")
        except OSError:
            return None

    def _refuse_live(self) -> None:
        live = self._any_live()
        if live is not None:
            raise console.Refused(409, f"a brain is live: {live}")

    def _log_path(self) -> Path:
        """A new `state/lobby/launch-<UTC>.log` (the directory created); a second spawn in the same second gets `-2`."""
        folder = self.root.lobby_log_dir
        folder.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        path, n = folder / f"launch-{stamp}.log", 2
        while path.exists():
            path, n = folder / f"launch-{stamp}-{n}.log", n + 1
        return path

    def _launch(self, verb: str, argv: list[str], ready: Callable[[], str | None]) -> tuple[int, dict[str, Any]]:
        """Spawn one detached brain (its own session, stdin closed, stdout and stderr to a new log, the environment
        plus `PYTHONUNBUFFERED=1`), then poll `ready` until it names the session, the child exits, or the wait runs
        out. A nonzero exit during the wait is a 500 with the log's tail; an argv the OS refuses (ValueError) is a 400
        with the empty log removed."""
        log = self._log_path()
        try:
            with log.open("xb") as handle:
                child = self.spawn(argv, stdin=subprocess.DEVNULL, stdout=handle, stderr=subprocess.STDOUT,
                                   start_new_session=True, env={**os.environ, "PYTHONUNBUFFERED": "1"})
        except ValueError as exc:
            log.unlink(missing_ok=True)
            return 400, {"ok": False, "error": f"protean {verb} could not start: {exc}"}
        except OSError as exc:
            return 500, {"ok": False, "error": f"protean {verb} could not start: {exc}"}
        with self._children:
            self.children.append(child)
        self.echo(f"lobby: protean {verb} spawned (pid {child.pid}), its output in {log}")
        deadline = time.monotonic() + self.wait_seconds
        task = ready()
        while task is None:
            code = child.poll()
            if code is not None:
                if code != 0:
                    tail = log.read_bytes().decode("utf-8", "replace")[-LOG_TAIL:]
                    return 500, {"ok": False, "error": f"protean {verb} exited {code}: {tail}"}
                task = ready()
                break
            if time.monotonic() >= deadline:
                break
            time.sleep(self.poll_seconds)
            task = ready()
        return 200, {"ok": True, "task": task, "pid": child.pid, "log": str(log)}

    def _reap(self) -> None:
        """Collect spawned brains that have exited, so none lingers as a zombie whose pid still answers signal 0."""
        # A snapshot polled and rebound under the lock (brief 8, F4): without the lock the rebind loses an append.
        with self._children:
            self.children = [child for child in list(self.children) if child.poll() is None]


def _query_task(route: str, query: str) -> str:
    """The one `task` value of a query: missing, empty or repeated is a 400."""
    values = parse_qs(query, keep_blank_values=True).get("task", [])
    if len(values) > 1:
        raise console.Refused(400, f"{route} takes one task")
    if not values or not values[0]:
        raise console.Refused(400, f"{route} needs task")
    return values[0]


def run(root: Root, echo: Echo = print) -> int:
    """`protean lobby`: serve until SIGINT or SIGTERM (the CLI's handlers raise KeyboardInterrupt), then close."""
    lobby = Lobby(root, echo=echo)
    try:
        lobby.start()
        echo(f"lobby served on {root.path}")
        echo("lobby: state/lobby.json")
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        pass
    finally:
        lobby.close()
    return 0
