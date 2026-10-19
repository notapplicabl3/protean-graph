"""The console, brain side: one local API per live task, served by a thread inside the brain process.

127.0.0.1 only, an ephemeral port, a per-run token in `state/<task>/console.json` (mode 0600, removed at exit;
neither the editor's sandbox nor a check command's can read it). A snapshot, then the stream (every `log.jsonl` line
as written, and the editor's transcript from `transcript.jsonl` while a worker runs), and operations in: pause,
resume, stop, think, answer, extend, mode, approve. An operation never reaches a node: it is validated, recorded in
`task.json`, logged with `actor: "console"`, and the loop reads that state at the tick boundary. `extend` is the only
operation that widens a ceiling, and it never touches a per-call cap. A `think` is the operator's thought; marked `decide`, it is
also their decision, the only way an entry enters `decided`. The wire is
`docs/console-api.md`.
"""

from __future__ import annotations

import hmac
import http.client
import json
import math
import os
import secrets
import select
import signal
import socket
import subprocess
import tempfile
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from protean import state
from protean.graph import budget
from protean.graph.budget import Caps
from protean.state import EXPLORE, HOLD, MODES, PLAN, Decision, Held, Move, Root, Task, Thought

HOST = "127.0.0.1"  # an invariant, not a setting: another device reaches the brain through tailscale, never a bind
UNATTENDED_SECONDS = 1800  # the session's default, not the operator's number: no client in that time -> park
OPS = ("pause", "resume", "stop", "think", "answer", "extend", "mode", "approve")
LIVE_ONLY = ("pause", "resume", "stop")
BODY_LIMIT = 64 * 1024
TEXT_LIMIT = 8000
POLL_SECONDS = 0.05
Echo = Callable[[str], None]
Notify = Callable[[str], None]


def _silent(_: str) -> None:
    pass


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Refused(ValueError):
    """An operation the task's state does not admit; `status` is the HTTP status it is answered with."""

    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status = status


class Unavailable(Exception):
    """No live brain answers for the task: no console file, or nothing listening behind it."""


def is_yes(text: str) -> bool:
    """The answer's first word is `yes`, in any case, trailing punctuation aside."""
    words = text.split()
    return bool(words) and words[0].rstrip(".,;:!").lower() == "yes"


def ceilings(task: Task, caps: Caps) -> dict[str, Any]:
    """The ceilings in force: the task totals with every extension; what active work has accrued and the active time
    it accrued over; the allowance (`max_usd` with every extension, plus what accrued), which
    the session's spend stops at; and the per-call limits, which never widen."""
    return {"max_usd": caps.max_usd + task.extra_usd, "max_ticks": caps.max_ticks + task.extra_ticks,
            "extra_usd": task.extra_usd, "extra_ticks": task.extra_ticks,
            "accrued_usd": budget.accrued_usd(task, caps), "allowance_usd": budget.allowance(task, caps),
            "active_seconds": task.active_seconds,
            "director_call_usd": caps.director_call_usd, "worker_call_usd": caps.worker_call_usd,
            "reader_call_usd": caps.reader_call_usd}


def prices(caps: Caps) -> dict[str, dict[str, float] | None]:
    """Each seat's prices in USD per million tokens, from the model it is pinned to; None for a model config prices
    nothing for. The brain prices nothing itself: the app estimates the call in flight from these."""
    models = {"director": caps.director_model, "worker": caps.worker_model, "reader": caps.reader_model}
    return {seat: dict(caps.prices[model]) if model in caps.prices else None for seat, model in models.items()}


def held_view(held: Held | None) -> dict[str, Any] | None:
    """What is held for `approve`, as a client sees it: a plan's units (id, intent, summary, needs, predicates), a
    retry's unit and attempt, or an explore's move and brief, with `thoughts`, how many the session had when it was
    held (the record's `Held.directions`); None when nothing is."""
    if held is None:
        return None
    view: dict[str, Any] = {"kind": held.kind, "approved": held.approved, "thoughts": held.directions}
    if held.kind == PLAN:
        view["units"] = [{"id": u.id, "intent": u.intent, "summary": u.plain_summary, "needs": list(u.needs),
                          "predicates": [state.predicate_view(p) for p in u.predicates]} for u in held.units]
    elif held.kind == EXPLORE:
        view.update(move=held.move, brief=held.brief)
    else:
        view.update(unit=held.unit, attempt=held.attempt)
    return view


def held_event(held: Held) -> dict[str, Any]:
    """A held progression's fields in a log event: `kind`, then the plan's unit ids, the retry's unit and attempt, or
    the explore's move."""
    if held.kind == PLAN:
        return {"kind": held.kind, "units": [u.id for u in held.units]}
    if held.kind == EXPLORE:
        return {"kind": held.kind, "move": held.move}
    return {"kind": held.kind, "unit": held.unit, "attempt": held.attempt}


def held_label(held: Held) -> str:
    """A held progression in a few words: `plan (u-1, u-2)`, `retry of u-1 (attempt 2)` or `explore m2`."""
    if held.kind == PLAN:
        return f"plan ({', '.join(u.id for u in held.units)})"
    if held.kind == EXPLORE:
        return f"explore {held.move}"
    return f"retry of {held.unit} (attempt {held.attempt})"


def move_view(move: Move) -> dict[str, Any]:
    """One move as a client sees it: id, kind, line, cost and outcome, then only the pointers it carries (an explore's
    branch, brief and report, a task's branch and units, an ask's question, an ideas move's items); a report always
    rides with `unverified: true`, since nothing grades an explore."""
    view: dict[str, Any] = {"id": move.id, "kind": move.kind, "line": move.line, "cost_usd": move.cost_usd,
                            "outcome": move.outcome}
    if move.branch:
        view["branch"] = move.branch
    if move.units:
        view["units"] = list(move.units)
    if move.question:
        view["question"] = move.question
    if move.ideas:
        view["ideas"] = [asdict(idea) for idea in move.ideas]
    if move.brief:
        view["brief"] = move.brief
    if move.report is not None:
        view.update(report=move.report, unverified=True)
    return view


def snapshot(task: Task, caps: Caps, live: bool) -> dict[str, Any]:
    """The session as a client first sees it: the thought record (the operator's thought verbatim, the standing, what is open,
    what they decided), the moves, their thoughts and whether one is awaited; the units with their statuses, the operator's plain
    summaries and predicates (`retain` shown only when false); cost, each seat's prices, tick, ceilings, open
    questions, the paused flag, the mode and what is held."""
    return {
        "task": task.id, "thought": {"thought": task.thought, "standing": task.standing,
                                     "open": [asdict(item) for item in list(task.open)],
                                     "decided": [asdict(d) for d in list(task.decided)]},
        "workspace": task.workspace, "project": task.project, "live": live, "tick": task.tick,
        "terminal": task.terminal, "stop_reason": task.stop_reason, "paused": task.paused,
        "mode": task.mode, "held": held_view(task.held), "awaiting_thought": task.awaiting_thought,
        "cost_usd": task.cost_usd, "ceilings": ceilings(task, caps), "prices": prices(caps),
        "units": [{"id": u.id, "intent": u.intent, "summary": u.plain_summary, "status": u.status,
                   "attempts": u.attempts, "needs": list(u.needs), "integrated": u.integrated,
                   "predicates": [state.predicate_view(p) for p in u.predicates]}
                  for u in list(task.units)],
        "moves": [move_view(move) for move in list(task.moves)],
        "questions": [asdict(q) for q in state.open_questions(task)],
        "thoughts": [asdict(t) for t in list(task.thoughts)],
    }


def _text(body: dict[str, Any], key: str) -> str:
    value = body.get(key)
    if not isinstance(value, str) or not value.strip():
        raise Refused(400, f"{key} must be a non-empty string")
    if len(value) > TEXT_LIMIT:
        raise Refused(413, f"{key} is longer than {TEXT_LIMIT} characters")
    return value.strip()


def _amounts(body: dict[str, Any]) -> tuple[float, int]:
    """Dollars and ticks one `extend` press adds; no upper wall, but at least one above zero."""
    usd, ticks = body.get("usd", 0), body.get("ticks", 0)
    if isinstance(usd, bool) or not isinstance(usd, (int, float)) or not math.isfinite(usd) or usd < 0:
        raise Refused(400, "usd must be a finite number of dollars, zero or more")
    if isinstance(ticks, bool) or not isinstance(ticks, int) or ticks < 0:
        raise Refused(400, "ticks must be a whole number, zero or more")
    if not usd and not ticks:
        raise Refused(400, "extend needs usd or ticks above zero")
    return float(usd), ticks


def check(task: Task, op: str, body: dict[str, Any], live: bool) -> None:
    """Refuse an operation the task does not admit; changes nothing."""
    if op not in OPS:
        raise Refused(404, f"no operation {op!r}")
    if task.terminal == "done" or task.stop_reason == "abandoned":
        raise Refused(409, f"task {task.id} is closed ({task.stop_reason or task.terminal})")
    if op in LIVE_ONLY and not live:
        raise Refused(409, f"{op} needs a live brain: no `protean think` or `protean resume` is driving {task.id}")
    if op == "pause" and task.paused:
        raise Refused(409, "already paused")
    if op == "resume" and not task.paused:
        raise Refused(409, "not paused")
    if op == "think":
        _text(body, "text")
        if not isinstance(body.get("decide", False), bool):
            raise Refused(400, "decide must be true or false")
    elif op == "mode":
        mode = body.get("mode")
        if mode not in MODES:
            raise Refused(400, f"mode must be one of {', '.join(MODES)}")
        if mode == task.mode:
            raise Refused(409, f"already {mode}")
    elif op == "approve":
        if task.held is None:
            raise Refused(409, "nothing is held")
        if task.held.approved:
            raise Refused(409, f"the held {held_label(task.held)} is already approved")
    elif op == "extend":
        usd, _ = _amounts(body)
        if not math.isfinite(task.extra_usd + usd):
            raise Refused(400, "that extension leaves no finite ceiling")
    elif op == "answer":
        qid = body.get("id")
        if not isinstance(qid, str) or not qid:
            raise Refused(400, "id must name a question")
        question = state.find_question(task, qid)
        if question is None:
            raise Refused(404, f"no question {qid}")
        if question.status != "open":
            raise Refused(409, f"{question.id} is already {question.status}")
        text = _text(body, "text")
        if question.reason == "cap_hit":
            raise Refused(409, f"{question.id} is a cap hit: `extend` releases it, `stop` ends the run")
        if question.reason == "unknown_spend" and not is_yes(text):
            raise Refused(409, f"{question.id} holds paid work under unknown spend: only `yes` continues it")
        if question.reason == "no_progress" and not is_yes(text):
            raise Refused(409, f"{question.id} holds the thinker on no progress: only `yes` calls it again")


def apply(task: Task, caps: Caps, op: str, body: dict[str, Any], live: bool, now: str | None = None) -> dict[str, Any]:
    """Check an operation, then record it in `task`; returns the log event, `actor: "console"`. Raises `Refused`."""
    check(task, op, body, live)
    now = now or _now()
    event: dict[str, Any] = {"event": op, "actor": "console", "live": live}
    if op == "pause":
        task.paused = True
    elif op == "resume":
        task.paused = False
    elif op == "think":  # reaches the thinker at its next call and ends a wait for one; `decide` is the operator's decision
        thought, decide = Thought(text=_text(body, "text"), created=now), body.get("decide", False)
        task.thoughts.append(thought)
        task.awaiting_thought = False
        if decide:  # the only writer of `decided`: their text as the thought holds it, its time, its index from 1
            task.decided.append(Decision(text=thought.text, created=now, think=len(task.thoughts)))
        event.update(text=thought.text, index=len(task.thoughts), decide=decide)
    elif op == "answer":
        question = state.find_question(task, body["id"])
        question.status, question.answer, question.actor = "answered", _text(body, "text"), "console"
        event.update(id=question.id, asker=question.asker, reason=question.reason, text=question.answer)
    elif op == "mode":
        event.update(mode=body["mode"], was=task.mode)
        task.mode = body["mode"]
    elif op == "approve":
        task.held.approved = True
        event.update(held_event(task.held))
    elif op == "extend":
        usd, ticks = _amounts(body)
        task.extra_usd += usd  # the allowance rises by the same X
        task.extra_ticks += ticks
        now_in_force = ceilings(task, caps)
        event.update(usd=usd, ticks=ticks, max_usd=now_in_force["max_usd"], max_ticks=now_in_force["max_ticks"],
                     allowance_usd=now_in_force["allowance_usd"])
        released = [q for q in state.open_questions(task) if q.reason == "cap_hit"]
        for question in released:
            question.status, question.actor = "answered", "console"
            question.answer = f"extend +${usd:.2f}, +{ticks} ticks"
        if released:
            event["released"] = [q.id for q in released]
    return event


def describe(event: dict[str, Any]) -> str:
    """One terminal line for an operation that landed."""
    op = event["event"]
    if op == "think":
        return f"console: think #{event['index']}: {event['text'][:120]}" + (" (decided)" if event["decide"] else "")
    if op == "answer":
        return f"console: answer {event['id']} ({event['asker']})"
    if op == "extend":
        return (f"console: extend +${event['usd']:.2f}, +{event['ticks']} ticks; ceilings now "
                f"${event['max_usd']:.2f}, {event['max_ticks']} ticks, allowance ${event['allowance_usd']:.2f}")
    if op == "stop":
        return "console: stop; the call in flight books its spend and the task saves"
    if op == "mode":
        return f"console: mode {event['mode']}" + (" (a task move's plan, each retry and each explore wait for "
                                                   "`approve`)" if event["mode"] == HOLD else "")
    if op == "approve":
        what = (f"plan ({', '.join(event['units'])})" if event["kind"] == PLAN
                else f"explore {event['move']}" if event["kind"] == EXPLORE
                else f"retry of {event['unit']} (attempt {event['attempt']})")
        return f"console: approve {what}; released at the tick boundary"
    return f"console: {op}" + (" (holds at the next tick boundary)" if op == "pause" else "")


def notify(text: str) -> None:
    """One macOS notification through `osascript`; any failure is swallowed, a missed banner never stops a run."""
    script = ["-e", "on run argv", "-e", 'display notification (item 1 of argv) with title "PROTEAN"', "-e", "end run"]
    try:
        subprocess.run(["osascript", *script, " ".join(text.split())[:240]], capture_output=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        pass


def _interrupt_main() -> None:
    """`stop`: the Ctrl-C path. SIGINT to the main thread wakes a blocked call, which books its spend and saves.
    Delivered only if the process handles SIGINT: `cli.main` installs the handler itself, since a process started in
    the background inherits SIGINT ignored; a wait at the tick boundary needs no signal (`Console.wait`)."""
    signal.pthread_kill(threading.main_thread().ident, signal.SIGINT)


def _write_private(path, data: dict[str, Any]) -> None:
    """Write the console file whole or not at all: a private (0600) temp file in the same directory, renamed over it,
    so a client never reads it empty or half written (the rename replaces a symlink there, never its target)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        try:
            os.fchmod(fd, 0o600)
            os.write(fd, (json.dumps(data) + "\n").encode("utf-8"))
        finally:
            os.close(fd)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _size(path) -> int:
    try:
        return path.stat().st_size
    except FileNotFoundError:
        return 0


def _read_from(path, offset: int) -> bytes:
    try:
        with path.open("rb") as handle:
            handle.seek(offset)
            return handle.read()
    except FileNotFoundError:
        return b""


def _gone(sock: socket.socket) -> bool:
    """The client closed its end: readable with nothing to read."""
    try:
        readable, _, _ = select.select([sock], [], [], 0)
        return bool(readable) and sock.recv(1, socket.MSG_PEEK) == b""
    except OSError:
        return True


class _Server(ThreadingHTTPServer):
    daemon_threads = True
    block_on_close = False
    console: Console


class _Handler(BaseHTTPRequestHandler):
    server_version = "protean-console/1"
    server: _Server

    def log_message(self, format: str, *args: Any) -> None:  # the run's terminal is not an access log
        pass

    def do_GET(self) -> None:
        self.server.console.handle(self, "GET")

    def do_POST(self) -> None:
        self.server.console.handle(self, "POST")

    def do_PUT(self) -> None:
        self.server.console.handle(self, "PUT")

    def do_DELETE(self) -> None:
        self.server.console.handle(self, "DELETE")

    def do_PATCH(self) -> None:
        self.server.console.handle(self, "PATCH")


def _reply(req: BaseHTTPRequestHandler, status: int, payload: dict[str, Any]) -> None:
    data = json.dumps(payload).encode("utf-8")
    req.send_response(status)
    req.send_header("Content-Type", "application/json")
    req.send_header("Content-Length", str(len(data)))
    req.end_headers()
    req.wfile.write(data)


def _raw_body(req: BaseHTTPRequestHandler) -> bytes | None:
    """The request body, read whole before any answer so the connection closes cleanly; None when over the limit."""
    try:
        length = max(int(req.headers.get("Content-Length") or 0), 0)
    except ValueError:
        length = 0
    if length > BODY_LIMIT:
        return None
    return req.rfile.read(length) if length else b""


def _parse(raw: bytes | None) -> dict[str, Any]:
    if raw is None:
        raise Refused(413, f"the body is over {BODY_LIMIT} bytes")
    try:
        body = json.loads(raw) if raw.strip() else {}
    except (json.JSONDecodeError, UnicodeDecodeError):
        raise Refused(400, "the body must be one JSON object") from None
    if not isinstance(body, dict):
        raise Refused(400, "the body must be one JSON object")
    return body


class Console:
    """The serving side of one live task, and the waits the loop makes on it at the tick boundary."""

    def __init__(self, root: Root, task: Task, caps: Caps, echo: Echo = _silent, unattended: float = 0.0,
                 notify: Notify | None = None, interrupt: Callable[[], None] | None = None) -> None:
        self.root, self.task, self.caps, self.echo = root, task, caps, echo
        self.unattended, self.notify, self.interrupt = unattended, notify, interrupt or _interrupt_main
        self.token = secrets.token_urlsafe(32)
        self.cond = threading.Condition()
        self.ops = 0  # operations landed; a wait ends when it moves
        self.clients = 0  # stream connections open now
        self.stopping = False  # a `stop` landed: a wait at the tick boundary ends with KeyboardInterrupt itself
        self.closing = False
        self.server: _Server | None = None
        self._episode: tuple[str, ...] | None = None
        self._deadline = 0.0
        self._notified = False

    @property
    def address(self) -> tuple[str, int]:
        host, port = self.server.server_address[:2]
        return host, port

    def start(self) -> None:
        self.server = _Server((HOST, 0), _Handler)
        self.server.console = self
        threading.Thread(target=self.server.serve_forever, kwargs={"poll_interval": POLL_SECONDS},
                         name="protean-console", daemon=True).start()
        host, port = self.address
        _write_private(self.root.console_file(self.task.id),
                       {"host": host, "port": port, "token": self.token, "pid": os.getpid(), "task": self.task.id})

    def close(self) -> None:
        """Stop serving: open streams send what is left of the log and end, then the console file goes."""
        with self.cond:
            self.closing = True
            self.cond.notify_all()
        if self.server is not None:
            self.server.shutdown()
            self.server.server_close()
        deadline = time.monotonic() + 1.0
        with self.cond:
            while self.clients and time.monotonic() < deadline:
                self.cond.wait(POLL_SECONDS)
        path = self.root.console_file(self.task.id)
        try:
            if json.loads(path.read_text(encoding="utf-8")).get("token") == self.token:
                path.unlink()
        except (OSError, json.JSONDecodeError):
            pass

    # --- the wire ---

    def handle(self, req: BaseHTTPRequestHandler, method: str) -> None:
        raw = _raw_body(req)
        header = req.headers.get("Authorization") or ""
        if not (header.startswith("Bearer ") and hmac.compare_digest(header[7:].encode(), self.token.encode())):
            return _reply(req, 401, {"ok": False, "error": "unauthorized"})
        path = req.path.split("?", 1)[0]
        name = path.removeprefix("/")
        if name in ("snapshot", "stream"):
            if method != "GET":
                return _reply(req, 405, {"ok": False, "error": f"{path} takes GET"})
            if name == "snapshot":
                return _reply(req, 200, self.snapshot())
            return self._stream(req)
        if name not in OPS:
            return _reply(req, 404, {"ok": False, "error": f"no endpoint {path}"})
        if method != "POST":
            return _reply(req, 405, {"ok": False, "error": f"{path} takes POST"})
        try:
            status, payload = self.operate(name, _parse(raw))
        except Refused as refused:
            status, payload = refused.status, {"ok": False, "error": str(refused)}
        _reply(req, status, payload)
        if status == 200 and name == "stop":
            with self.cond:
                self.stopping = True
                self.cond.notify_all()
            self.interrupt()

    def snapshot(self) -> dict[str, Any]:
        with state.LOCK:
            offsets = {"log_bytes": _size(self.root.log_file(self.task.id)),
                       "transcript_bytes": _size(self.root.transcript_file(self.task.id))}
            return {**snapshot(self.task, self.caps, True), **offsets}

    def operate(self, op: str, body: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        """Land one operation: checked against the live task, recorded in `task.json` on disk and in the loop's
        copy under the state lock, then logged. The loop acts on it at the tick boundary."""
        now = _now()
        with state.LOCK:
            if self.closing:
                return 409, {"ok": False, "error": "the run is ending"}
            try:
                check(self.task, op, body, True)
                if op != "stop":
                    disk = state.load(self.root, self.task.id)
                    apply(disk, self.caps, op, body, True, now)
                    state.save(self.root, disk)
                event = apply(self.task, self.caps, op, body, True, now)
            except Refused as refused:
                return refused.status, {"ok": False, "error": str(refused)}
            state.append_log(self.root, self.task.id, event)
        with self.cond:
            self.ops += 1
            self.cond.notify_all()
        self.echo(describe(event))
        return 200, {"ok": True, "event": event}

    def _stream(self, req: BaseHTTPRequestHandler) -> None:
        """The snapshot as the first line, then every log line and every transcript line appended after it, each
        file in its own order, until the run ends or the client leaves. A line may repeat a change the snapshot
        already shows; none is skipped. When the run ends, the last line before EOF is one more snapshot, of the
        terminal state: the console stops answering `/snapshot` as it closes, and a client that takes its state
        only from snapshots must still see how the task ended (gap found by the app's wire check, 2026-10-11)."""
        first = self.snapshot()
        tails = [[self.root.log_file(self.task.id), first["log_bytes"], b""],
                 [self.root.transcript_file(self.task.id), first["transcript_bytes"], b""]]
        req.send_response(200)
        req.send_header("Content-Type", "application/x-ndjson")
        req.send_header("Cache-Control", "no-store")
        req.end_headers()
        with self.cond:
            self.clients += 1
            self.cond.notify_all()
        try:
            req.wfile.write(json.dumps({"snapshot": first}).encode("utf-8") + b"\n")
            while True:
                closing, moved = self.closing, False
                for tail in tails:
                    chunk = _read_from(tail[0], tail[1])
                    if chunk:
                        moved, tail[1] = True, tail[1] + len(chunk)
                        *lines, tail[2] = (tail[2] + chunk).split(b"\n")
                        req.wfile.write(b"".join(line + b"\n" for line in lines if line.strip()))
                if not moved and closing:
                    req.wfile.write(json.dumps({"snapshot": self.snapshot()}).encode("utf-8") + b"\n")
                    break
                if not moved and _gone(req.connection):
                    break
                time.sleep(POLL_SECONDS)
        except OSError:
            pass
        finally:
            with self.cond:
                self.clients -= 1
                self.cond.notify_all()

    # --- the loop's side ---

    def mark(self) -> int:
        """The operation count before the loop reads state; a wait returns once it moves."""
        with self.cond:
            return self.ops

    def settled(self) -> None:
        """No hold: the next hold is a new episode, with its own notification and timer."""
        with self.cond:
            self._episode = None

    def wait(self, hold: tuple[str, ...], seen: int, line: str, unattended: bool) -> bool:
        """Hold at the tick boundary until an operation lands (True). An `unattended` hold with no stream client
        fires one notification per episode and, once no client has been attached for the timeout, returns False:
        the caller parks. With a client attached it waits indefinitely; an idle brain spends nothing. A `stop` that
        lands during the wait raises KeyboardInterrupt here, in the main thread, so the Ctrl-C path runs even when the
        process inherited SIGINT ignored (a background launch) and the stop's own SIGINT was dropped."""
        with self.cond:
            if hold != self._episode:
                self._episode, self._notified = hold, False
                self._deadline = time.monotonic() + self.unattended
                self.echo(line)
            while self.ops == seen and not self.closing:
                if self.stopping:
                    raise KeyboardInterrupt
                if self.clients or not unattended:
                    self._deadline = time.monotonic() + self.unattended
                    self.cond.wait(0.2)
                    continue
                if not self._notified:
                    self._notified = True
                    if self.notify is not None:
                        self.notify(f"{self.task.id}: {line}")
                left = self._deadline - time.monotonic()
                if left <= 0:
                    self._episode = None
                    return False
                self.cond.wait(min(left, 0.2))
            return True


@contextmanager
def serve(root: Root, task: Task, caps: Caps, echo: Echo = _silent, unattended: float = 0.0,
          notify: Notify | None = None, interrupt: Callable[[], None] | None = None) -> Iterator[Console]:
    """Serve the task's console while the block runs; the console file is removed however it ends."""
    desk = Console(root, task, caps, echo, unattended, notify, interrupt)
    desk.start()
    try:
        yield desk
    finally:
        desk.close()


# --- the client side, for `protean ctl` ---


def _endpoint(root: Root, task_id: str) -> tuple[int, str]:
    try:
        info = json.loads(root.console_file(task_id).read_text(encoding="utf-8"))
        return int(info["port"]), str(info["token"])
    except (OSError, ValueError, KeyError, TypeError):
        raise Unavailable(f"no console for {task_id}: no live brain serves it") from None


def request(root: Root, task_id: str, method: str, path: str, body: dict[str, Any] | None = None,
            timeout: float = 10.0) -> tuple[int, dict[str, Any]]:
    """One call to the live brain's console, with the token from its console file."""
    port, token = _endpoint(root, task_id)
    conn = http.client.HTTPConnection(HOST, port, timeout=timeout)
    headers = {"Authorization": f"Bearer {token}"}
    data = None
    if body is not None:
        data = json.dumps(body).encode("utf-8")
        headers["Content-Type"] = "application/json"
    try:
        conn.request(method, path, body=data, headers=headers)
        resp = conn.getresponse()
        return resp.status, json.loads(resp.read() or b"{}")
    except (OSError, http.client.HTTPException, json.JSONDecodeError) as exc:
        raise Unavailable(f"the console at {HOST}:{port} did not answer: {exc}") from None
    finally:
        conn.close()


def watch(root: Root, task_id: str, out: Echo) -> None:
    """Print the snapshot line, then the stream, until the brain ends it. Counts as an attached client."""
    port, token = _endpoint(root, task_id)
    conn = http.client.HTTPConnection(HOST, port)
    try:
        conn.request("GET", "/stream", headers={"Authorization": f"Bearer {token}"})
        resp = conn.getresponse()
        if resp.status != 200:
            raise Unavailable(f"the console refused the stream: {resp.status}")
        for line in resp:
            out(line.decode("utf-8").rstrip("\n"))
    except OSError as exc:
        raise Unavailable(f"the console at {HOST}:{port} did not answer: {exc}") from None
    finally:
        conn.close()
