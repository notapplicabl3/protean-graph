"""Console, brain side, driven by a scripted client over the real server thread.

127.0.0.1 only; every request without the per-run token refused with nothing changed; `extend` the only operation
that changes a ceiling, each press logged with its amount and the resulting ceiling, no per-call cap widened; the
snapshot first, then the stream; `pause` holds at the tick boundary and `resume` releases it; a think persisted and
carried whole in every later thinker message; an answer closes only its id and reaches only its asker; no runtime
question or answer in any thinker message; `stop` takes the Ctrl-C path; unattended, an open question notifies once
and parks, and so does a wait for a thought; the editor under its real sandbox profile cannot read the token file and
never carries the token.
"""

from __future__ import annotations

import http.client
import itertools
import json
import os
import shutil
import signal
import socket
import stat
import subprocess
import threading
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pytest

from protean import cli, loop, state
from protean.dry import ScriptedRunner, make_workspace
from protean.graph import console, director, worker
from protean.graph.budget import Caps
from protean.runner import real_runner
from protean.state import MOVE_KINDS, Predicate, Root, Task, Unit, load, open_questions, save, set_current

UNIT = {"id": "u-1", "intent": "write a.txt", "predicates": [{"kind": "file_exists", "args": {"path": "a.txt"}}]}
UNIT2 = {"id": "u-2", "intent": "write b.txt", "predicates": [{"kind": "file_exists", "args": {"path": "b.txt"}}]}
PASS = {"write": {"a.txt": "A\n"}, "commit": True, "exit_code": 0, "result": "wrote a.txt", "total_cost_usd": 0.5}
PASS2 = {"write": {"b.txt": "B\n"}, "commit": True, "exit_code": 0, "result": "wrote b.txt", "total_cost_usd": 0.5}
OWN = object()  # the client's own token


STANDINGS = itertools.count(1)


def _with_state(obj: object) -> object:
    """A thinker move without `state` gets one whose standing no earlier reply had."""
    if isinstance(obj, dict) and "state" not in obj and any(kind in obj for kind in MOVE_KINDS):
        return {**obj, "state": {"standing": f"Standing {next(STANDINGS)}.", "open": []}}
    return obj


def _reply(obj: dict | str, cost: float = 0.1) -> dict:
    obj = _with_state(obj)
    text = obj if isinstance(obj, str) else json.dumps(obj)
    return {"type": "result", "is_error": False, "session_id": "s-1", "total_cost_usd": cost, "result": text}


@pytest.fixture
def env(tmp_path: Path, root: Root) -> tuple[Root, Path]:
    root.path.mkdir(parents=True)
    root.thinker_prompt.write_text("m\n")
    root.editor_prompt.write_text("e\n")
    return root, make_workspace(tmp_path / "ws")


class Client:
    """The battery's scripted console client: plain HTTP to 127.0.0.1, the token read from the console file."""

    def __init__(self, root: Root, task_id: str) -> None:
        info = json.loads(root.console_file(task_id).read_text())
        self.port, self.token = info["port"], info["token"]

    def call(self, method: str, path: str, body: dict | None = None, token: Any = OWN) -> tuple[int, dict]:
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=10)
        headers = {}
        if token is not None:
            headers["Authorization"] = f"Bearer {self.token if token is OWN else token}"
        data = None if body is None else json.dumps(body).encode()
        try:
            conn.request(method, path, body=data, headers=headers)
            resp = conn.getresponse()
            raw = resp.read()
            return resp.status, json.loads(raw) if raw else {}
        finally:
            conn.close()


class Stream(threading.Thread):
    """One stream connection read on its own thread: the status line, the headers, then one JSON line at a time."""

    def __init__(self, client: Client) -> None:
        super().__init__(daemon=True)
        self.sock = socket.create_connection(("127.0.0.1", client.port), timeout=10)
        self.sock.settimeout(None)
        self.sock.sendall(f"GET /stream HTTP/1.0\r\nAuthorization: Bearer {client.token}\r\n\r\n".encode())
        self.lines: list[dict] = []
        self.status: int | None = None
        self.ended = False
        self.cond = threading.Condition()

    def run(self) -> None:
        handle = self.sock.makefile("rb")
        try:
            self.status = int(handle.readline().split()[1])
            while handle.readline().strip():
                pass
            for raw in handle:
                with self.cond:
                    self.lines.append(json.loads(raw))
                    self.cond.notify_all()
        except (OSError, ValueError):
            pass
        finally:
            with self.cond:
                self.ended = True
                self.cond.notify_all()

    def wait_for(self, match, timeout: float = 10.0) -> dict:
        deadline = time.monotonic() + timeout
        with self.cond:
            while True:
                found = next((line for line in self.lines if match(line)), None)
                if found is not None:
                    return found
                if self.ended or time.monotonic() > deadline:
                    raise AssertionError(f"no matching stream line in {self.lines}")
                self.cond.wait(0.05)

    def close(self) -> None:
        try:
            self.sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        self.sock.close()


def _client(root: Root, timeout: float = 10.0) -> Client:
    """The live brain's console, once its file exists."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        task = state.current(root)
        if task is not None and root.console_file(task.id).exists():
            return Client(root, task.id)
        time.sleep(0.02)
    raise AssertionError("the brain never wrote its console file")


def _director_messages(runner: ScriptedRunner) -> list[str]:
    return [stdin for argv, stdin, _, _ in runner.calls if "Write" not in argv]


def _events(root: Root, task_id: str, kind: str) -> list[dict]:
    return [e for e in director.read_log(root, task_id) if e["event"] == kind]


def _rested(line: dict) -> bool:
    """The last line the loop logs for a `rest` move, before it waits for a thought: its tick's `fired` event, which
    ends every tick (shape SPEC decision 8), after the record's change (`thought`)."""
    return line.get("event") == "fired" and "rest" in line.get("walked", [])


def _parked_on_a_thought(root: Root, task: Task) -> bool:
    """A session that rested and lost its client: parked `interrupted`, awaiting a thought."""
    parked = [(e["hold"], e["on"]) for e in _events(root, task.id, "parked")]
    on = [("thought", task.moves[-1].id)]
    return (task.terminal, task.awaiting_thought, parked[-1:]) == ("interrupted", True, on)


def _gated(scripted: ScriptedRunner, attached: threading.Event):
    """The scripted runner, its first call held until the client's stream is attached, so every event the test
    waits for comes after the snapshot."""
    def runner(argv, stdin, cwd, timeout):
        if not scripted.calls:
            assert attached.wait(10), "the client never attached"
        return scripted(argv, stdin, cwd, timeout)
    return runner


def _in_client_thread(script) -> tuple[threading.Thread, dict]:
    """Run the scripted client beside the loop, which stays on the main thread (where `stop` sends its SIGINT)."""
    seen: dict = {}

    def body() -> None:
        try:
            script(seen)
        except BaseException as exc:  # surfaced by the test after the run
            seen["error"] = exc
    thread = threading.Thread(target=body, daemon=True)
    thread.start()
    return thread, seen


# --- a dry run driven by the scripted client over the real server thread ---


def test_snapshot_then_stream_pause_holds_think_is_carried_whole_and_the_answer_reaches_the_thinker_once(
    env: tuple[Root, Path]
):
    root, workspace = env
    scripted = ScriptedRunner([_reply({"task": [UNIT]}), _reply({"ask": "which style?"}),
                               _reply({"task": [UNIT2]}), _reply({"rest": True})], [PASS, PASS2])
    in_call, release = threading.Event(), threading.Event()

    def runner(argv, stdin, cwd, timeout):
        if not scripted.calls:  # hold the first call in flight until the client has thought and paused
            in_call.set()
            assert release.wait(10), "the client never released the first call"
        return scripted(argv, stdin, cwd, timeout)

    def script(seen: dict) -> None:
        client = _client(root)
        stream = Stream(client)
        stream.start()
        try:
            seen["first"] = stream.wait_for(lambda line: True)
            assert in_call.wait(10)
            assert client.call("POST", "/think", {"text": "use tabs"})[0] == 200
            assert cli._main(["--brain", str(root.path), "ctl", "think", "keep", "it", "short"]) == 0  # live ctl
            assert client.call("POST", "/pause")[0] == 200
            release.set()
            stream.wait_for(lambda line: line.get("event") == "director")  # the call in flight finished
            time.sleep(0.3)
            seen["calls_while_held"] = len(scripted.calls)
            seen["held"] = client.call("GET", "/snapshot")[1]
            assert client.call("POST", "/resume")[0] == 200
            asked = stream.wait_for(lambda line: line.get("event") == "question")
            seen["unknown_id"] = client.call("POST", "/answer", {"id": "q9", "text": "tabs"})[0]
            seen["answer"] = client.call("POST", "/answer", {"id": asked["id"], "text": "terse"})
            seen["twice"] = client.call("POST", "/answer", {"id": asked["id"], "text": "verbose"})[0]
            stream.wait_for(_rested)  # the session rests and waits for a thought: the client leaves, so it parks
        finally:
            seen["lines"] = list(stream.lines)
            stream.close()

    thread, seen = _in_client_thread(script)
    task = loop.run("g", workspace, root, Caps(), runner, unattended=0.2)
    thread.join(15)
    assert "error" not in seen, repr(seen.get("error"))
    assert _parked_on_a_thought(root, task) and [u.status for u in task.units] == ["passed", "passed"]

    first, *rest = seen["lines"]  # the snapshot first, then nothing but log lines, exactly as written
    assert first is seen["first"] and set(first) == {"snapshot"} and first["snapshot"]["task"] == task.id
    log = root.log_file(task.id).read_bytes()[first["snapshot"]["log_bytes"]:]
    logged = [json.loads(line) for line in log.splitlines()]
    assert rest == logged[:len(rest)] and all("event" in line for line in rest) and _rested(rest[-1])
    assert [e["event"] for e in logged[len(rest):]] == ["parked"]  # after the client left

    assert seen["calls_while_held"] == 1  # no model call while held at the tick boundary
    held = seen["held"]
    assert held["paused"] is True and [t["text"] for t in held["thoughts"]] == ["use tabs", "keep it short"]
    order = [line["event"] for line in rest]
    assert order.index("pause") < order.index("director") < order.index("resume")
    assert all(line["actor"] == "console" for line in rest if line["event"] in ("think", "pause", "resume", "answer"))

    messages = [json.loads(m) for m in _director_messages(scripted)]
    assert [m["thoughts"] for m in messages] == [[], ["use tabs", "keep it short"],
                                                 ["use tabs", "keep it short"], ["use tabs", "keep it short"]]
    assert list(messages[1])[:5] == ["thought", "standing", "open", "decided", "thoughts"]  # beside the thought
    assert [m["notes"] for m in messages] == [[], [], [], []]  # the answer is the `answer` key's, never a note
    assert ["answer" in m for m in messages] == [False, False, True, True]
    assert messages[2]["answer"] == {"id": "q1", "question": "which style?", "answer": "terse"}

    assert (seen["unknown_id"], seen["answer"][0], seen["twice"]) == (404, 200, 409)
    (asked,) = task.questions
    assert (asked.asker, asked.status, asked.answer, asked.actor, asked.disposition) == (
        "director", "answered", "terse", "console", "delivered")
    saved = load(root, task.id)  # the thoughts survive a save and a load
    assert [t.text for t in saved.thoughts] == ["use tabs", "keep it short"] and saved.paused is False
    assert not root.console_file(task.id).exists()


def test_a_runtime_question_and_its_answer_never_reach_the_thinker(env: tuple[Root, Path]):
    root, workspace = env
    scripted = ScriptedRunner([_reply("no plan"), _reply("still no plan"), _reply({"task": [UNIT]}),
                               _reply({"rest": True})], [PASS])
    attached = threading.Event()

    def script(seen: dict) -> None:
        client = _client(root)
        stream = Stream(client)
        stream.start()
        try:
            stream.wait_for(lambda line: "snapshot" in line)
            attached.set()
            seen["asked"] = stream.wait_for(lambda line: line.get("event") == "question")
            seen["answer"] = client.call("POST", "/answer", {"id": seen["asked"]["id"], "text": "go smaller, carrot"})
            stream.wait_for(_rested)
        finally:
            stream.close()

    thread, seen = _in_client_thread(script)
    task = loop.run("g", workspace, root, Caps(), _gated(scripted, attached), unattended=0.2)
    thread.join(15)
    assert "error" not in seen, repr(seen.get("error"))
    asked = seen["asked"]
    assert (asked["asker"], asked["reason"], seen["answer"][0]) == ("runtime", "invalid_plan", 200)
    assert _parked_on_a_thought(root, task)
    messages = _director_messages(scripted)
    assert len(messages) == 4
    for message in messages:
        assert asked["text"] not in message and "invalid" not in message and "carrot" not in message
    assert json.loads(messages[2])["notes"] == [] and json.loads(messages[2])["thoughts"] == []
    assert "answer" not in json.loads(messages[2])
    (closed,) = task.questions
    assert (closed.status, closed.answer, closed.disposition) == ("answered", "go smaller, carrot", "continued")
    assert [e["disposition"] for e in _events(root, task.id, "runtime_answer")] == ["continued"]
    assert all("to_director" not in e for e in _events(root, task.id, "runtime_answer") + _events(root, task.id, "answer"))


def test_a_ceiling_stop_waits_for_extend_while_a_client_is_attached(env: tuple[Root, Path]):
    root, workspace = env
    scripted = ScriptedRunner([_reply({"task": [UNIT]}, cost=0.6), _reply({"rest": True})], [PASS])
    attached = threading.Event()

    def script(seen: dict) -> None:
        client = _client(root)
        stream = Stream(client)
        stream.start()
        try:
            stream.wait_for(lambda line: "snapshot" in line)
            attached.set()
            seen["stopped"] = stream.wait_for(lambda line: line.get("event") == "stopped")
            time.sleep(0.3)  # attached: the brain waits instead of exiting
            seen["calls_while_stopped"] = len(scripted.calls)
            seen["extend"] = client.call("POST", "/extend", {"usd": 2})
            stream.wait_for(_rested)
        finally:
            stream.close()

    thread, seen = _in_client_thread(script)
    task = loop.run("g", workspace, root, Caps(max_usd=0.5), _gated(scripted, attached), unattended=0.2)
    thread.join(15)
    assert "error" not in seen, repr(seen.get("error"))
    assert seen["stopped"]["reason"] == "max_usd: 0.60 >= 0.5" and seen["calls_while_stopped"] == 1
    assert seen["extend"][1]["event"]["max_usd"] == 2.5
    assert _parked_on_a_thought(root, task) and task.extra_usd == 2.0 and len(scripted.calls) == 3
    (released,) = _events(root, task.id, "released")
    assert released["reason"] == "max_usd: 0.60 >= 0.5"


def test_close_ends_the_live_session_for_good_books_the_call_in_flight_and_exits_11(
    env: tuple[Root, Path], monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    """2026-10-20: `close` records the end first (stopped/closed, saved), then takes stop's path out; the record admits
    nothing after and `resume` refuses it."""
    root, workspace = env
    in_call = threading.Event()

    def runner(argv, stdin, cwd, timeout):
        in_call.set()
        time.sleep(10)  # the close's SIGINT cuts this short
        raise AssertionError("close did not interrupt the call")

    monkeypatch.setattr(cli, "real_runner", runner)

    def script(seen: dict) -> None:
        assert in_call.wait(10)
        seen["close"] = _client(root).call("POST", "/close")

    thread, seen = _in_client_thread(script)
    code = cli.main(["--brain", str(root.path), "think", "g", "--workspace", str(workspace)])
    thread.join(10)
    assert "error" not in seen, repr(seen.get("error"))
    assert (code, seen["close"][0]) == (11, 200) and "session closed" in capsys.readouterr().err
    task = state.current(root)
    assert (task.cost_usd, task.terminal, task.stop_reason, task.tick) == (2.0, "stopped", "closed", 0)
    (call,) = task.calls
    assert (call.seat, call.outcome, call.charge_usd, call.booked) == ("director", "interrupted", 2.0, True)
    order = [e["event"] for e in director.read_log(root, task.id)]
    assert order.index("close") < order.index("interrupted")
    assert seen["close"][1]["event"] == {"event": "close", "actor": "console", "live": True, "superseded": [],
                                        "held": None} and not root.console_file(task.id).exists()
    assert not root.clones_dir(task.id).exists()
    code = cli.main(["--brain", str(root.path), "resume"])
    assert code == 11 and len(state.current(root).calls) == 1  # refused: nothing ran


def test_stop_takes_the_ctrl_c_path_books_the_call_in_flight_and_exits_130(
    env: tuple[Root, Path], monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    root, workspace = env
    in_call = threading.Event()

    def runner(argv, stdin, cwd, timeout):
        in_call.set()
        time.sleep(10)  # the stop's SIGINT cuts this short
        raise AssertionError("stop did not interrupt the call")

    monkeypatch.setattr(cli, "real_runner", runner)

    def script(seen: dict) -> None:
        assert in_call.wait(10)
        seen["stop"] = _client(root).call("POST", "/stop")

    thread, seen = _in_client_thread(script)
    code = cli.main(["--brain", str(root.path), "think", "g", "--workspace", str(workspace)])
    thread.join(10)
    assert "error" not in seen, repr(seen.get("error"))
    assert (code, seen["stop"][0]) == (130, 200) and "booked" in capsys.readouterr().err
    task = state.current(root)
    assert (task.cost_usd, task.terminal, task.tick) == (2.0, "interrupted", 0)  # a stop parks the record (decision 26)
    (call,) = task.calls
    assert (call.seat, call.outcome, call.charge_usd, call.booked) == ("director", "interrupted", 2.0, True)
    order = [e["event"] for e in director.read_log(root, task.id)]
    assert order.index("stop") < order.index("interrupted")
    assert _events(root, task.id, "stop")[0]["actor"] == "console" and not root.console_file(task.id).exists()


# --- access, and extend as the only widening ---


@pytest.fixture
def live(env: tuple[Root, Path]):
    """A console served for a task with three open questions (director, unknown spend, cap hit), no loop
    running; `stop` records its interrupt instead of signalling the test's own thread."""
    root, workspace = env
    task = Task(id="task-20261010T150000", goal="g", workspace=str(workspace),
                units=[Unit("u-1", "write a.txt", [Predicate("file_exists", {"path": "a.txt"})])])
    state.add_question(task, "director", "question", "which file?")
    state.add_question(task, "runtime", "unknown_spend", "killed mid-call; continue?", {"unknown_spend": ["c1"]})
    state.add_question(task, "runtime", "cap_hit", "the director's call hit its cap", {"cap_hit": "c2"})
    save(root, task)
    set_current(root, task.id)
    state.append_log(root, task.id, {"event": "start"})
    stops: list[bool] = []
    with console.serve(root, task, Caps(), interrupt=lambda: stops.append(True)) as desk:
        yield root, task, desk, Client(root, task.id), stops
    assert not root.console_file(task.id).exists()


def _outward_address() -> str | None:
    """This host's address on its outward interface (no packet is sent), or None without a route."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
            probe.connect(("192.0.2.1", 9))
            address = probe.getsockname()[0]
    except OSError:
        return None
    return None if address.startswith("127.") else address


def test_the_console_file_is_replaced_whole_by_a_rename_never_rewritten_in_place(tmp_path: Path):
    """Order 3's finding: a client read `console.json` empty between its truncation and its write. It is now a private
    temp file in the same directory renamed over it, so a reader holds the old copy or the new one whole, and a symlink
    there is replaced, never written through."""
    path = tmp_path / "state" / "t" / "console.json"
    console._write_private(path, {"token": "one"})
    with path.open(encoding="utf-8") as earlier:  # a client that opened the first copy
        console._write_private(path, {"token": "two"})
        assert json.loads(earlier.read()) == {"token": "one"}  # never truncated or rewritten under it
    assert json.loads(path.read_text()) == {"token": "two"} and stat.S_IMODE(path.stat().st_mode) == 0o600
    outside = tmp_path / "outside.json"
    outside.write_text("keep\n")
    path.unlink()
    path.symlink_to(outside)
    console._write_private(path, {"token": "three"})
    assert outside.read_text() == "keep\n" and not path.is_symlink()
    assert json.loads(path.read_text()) == {"token": "three"}
    assert [p.name for p in path.parent.iterdir()] == ["console.json"]  # no temp file left behind


def test_the_console_listens_on_127_0_0_1_only_and_its_file_is_private(live):
    root, task, desk, client, _ = live
    assert desk.address[0] == "127.0.0.1" and desk.server.socket.family == socket.AF_INET
    path = root.console_file(task.id)
    assert json.loads(path.read_text()) == {"host": "127.0.0.1", "port": desk.address[1], "token": desk.token,
                                            "pid": os.getpid(), "task": task.id}
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert client.call("GET", "/snapshot")[0] == 200
    with pytest.raises(ConnectionRefusedError):
        socket.create_connection(("::1", client.port), timeout=2).close()
    outward = _outward_address()
    if outward is not None:
        with pytest.raises(OSError):
            socket.create_connection((outward, client.port), timeout=2).close()


def test_every_request_without_the_token_is_refused_and_changes_nothing(live):
    root, task, desk, client, stops = live
    before = (root.task_file(task.id).read_bytes(), root.log_file(task.id).read_bytes())
    requests = [("GET", "/snapshot", None), ("GET", "/stream", None), ("POST", "/pause", {}), ("POST", "/resume", {}),
                ("POST", "/stop", {}), ("POST", "/think", {"text": "x"}), ("POST", "/answer", {"id": "q1", "text": "x"}),
                ("POST", "/extend", {"usd": 5}), ("PUT", "/extend", {"usd": 5}), ("POST", "/nowhere", {})]
    for method, path, body in requests:
        for token in (None, "not-the-token", desk.token[:-1], desk.token + "x"):
            assert client.call(method, path, body, token) == (401, {"ok": False, "error": "unauthorized"}), (path, token)
    assert (root.task_file(task.id).read_bytes(), root.log_file(task.id).read_bytes()) == before
    assert (stops, desk.ops, task.paused, task.thoughts, task.extra_usd, task.extra_ticks) == ([], 0, False, [], 0.0, 0)
    assert [q.status for q in task.questions] == ["open", "open", "open"]


def test_extend_is_the_only_operation_that_changes_a_ceiling_and_never_a_per_call_cap(live):
    root, task, desk, client, stops = live
    caps_before = asdict(desk.caps)

    def in_force() -> dict:
        return client.call("GET", "/snapshot")[1]["ceilings"]

    start = in_force()
    assert start == {"max_usd": 25.0, "max_ticks": 100, "extra_usd": 0.0, "extra_ticks": 0, "accrued_usd": 0.0,
                     "allowance_usd": 25.0, "active_seconds": 0.0, "director_call_usd": 2.0, "worker_call_usd": 4.0,
                     "reader_call_usd": 1.5, "reader_max_usd": 7.0, "reader_usd": 0.0, "counted_usd": 0.0}
    for path, body in [("/pause", {}), ("/resume", {}), ("/think", {"text": "use tabs", "decide": True}),
                       ("/answer", {"id": "q1", "text": "a.txt"}), ("/answer", {"id": "q2", "text": "yes"}),
                       ("/stop", {})]:
        assert client.call("POST", path, body)[0] == 200, path
        saved = load(root, task.id)
        assert in_force() == start and (saved.extra_usd, saved.extra_ticks) == (0.0, 0), path
    assert stops == [True]
    presses = [({"usd": 1.5}, 26.5, 100), ({"ticks": 5}, 26.5, 105), ({"usd": 2.25, "ticks": 1}, 28.75, 106),
               ({"usd": 1_000_000}, 1_000_028.75, 106)]  # no upper wall
    for body, max_usd, max_ticks in presses:
        status, payload = client.call("POST", "/extend", body)
        assert status == 200 and (payload["event"]["max_usd"], payload["event"]["max_ticks"]) == (max_usd, max_ticks)
        now = in_force()
        assert (now["max_usd"], now["allowance_usd"], now["max_ticks"], now["director_call_usd"],
                now["worker_call_usd"]) == (max_usd, max_usd, max_ticks, 2.0, 4.0)  # nothing accrued: no tick ran
    logged = [(e["actor"], e["usd"], e["ticks"], e["max_usd"], e["allowance_usd"], e["max_ticks"])
              for e in _events(root, task.id, "extend")]
    assert logged == [("console", 1.5, 0, 26.5, 26.5, 100), ("console", 0.0, 5, 26.5, 26.5, 105),
                      ("console", 2.25, 1, 28.75, 28.75, 106), ("console", 1_000_000.0, 0, 1_000_028.75, 1_000_028.75,
                                                                106)]
    assert _events(root, task.id, "extend")[0]["released"] == ["q3"]  # the cap hit, and only on the first press
    assert asdict(desk.caps) == caps_before
    saved = load(root, task.id)
    assert (saved.extra_usd, saved.extra_ticks) == (1_000_003.75, 6)
    argv = director.argv(saved, desk.caps, root)
    assert argv[argv.index("--max-budget-usd") + 1] == "2.00"  # the next director call keeps its per-call cap


def test_an_answer_closes_only_its_id_once_and_bad_operations_change_nothing(live):
    root, task, desk, client, _ = live
    assert client.call("POST", "/answer", {"id": "q2", "text": "maybe"})[0] == 409  # unknown spend takes `yes`
    assert client.call("POST", "/answer", {"id": "q3", "text": "yes"})[0] == 409  # a cap hit takes `extend`
    assert client.call("POST", "/answer", {"id": "q7", "text": "yes"})[0] == 404
    for path, body in [("/think", {"text": "  "}), ("/think", {}), ("/extend", {"usd": 0}), ("/extend", {"usd": -1}),
                       ("/extend", {"ticks": 1.5}), ("/extend", {"usd": True}), ("/extend", {"usd": "5"}),
                       ("/answer", {"id": "q1"}), ("/answer", {"text": "yes"})]:
        assert client.call("POST", path, body)[0] == 400, (path, body)
    assert client.call("POST", "/resume")[0] == 409  # not paused
    before = load(root, task.id)
    assert client.call("POST", "/answer", {"id": "q2", "text": "yes"})[0] == 200
    assert client.call("POST", "/answer", {"id": "q2", "text": "yes"})[0] == 409
    after = load(root, task.id)
    assert [(q.id, q.status, q.answer, q.actor) for q in after.questions] == [
        ("q1", "open", None, None), ("q2", "answered", "yes", "console"), ("q3", "open", None, None)]
    assert [(q.id, q.status) for q in task.questions] == [(q.id, q.status) for q in after.questions]
    assert (before.questions[0], before.questions[2]) == (after.questions[0], after.questions[2])
    assert [e["id"] for e in _events(root, task.id, "answer")] == ["q2"]


# --- unattended, the offline answer, and resume ---


def test_unattended_a_question_notifies_once_and_parks_then_ctl_answers_offline_and_resume_continues(
    env: tuple[Root, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    root, workspace = env
    notes = tmp_path / "osascript.log"
    monkeypatch.setenv("PROTEAN_OSASCRIPT_LOG", str(notes))
    runner = ScriptedRunner([_reply({"ask": "which file?"}), _reply({"task": [UNIT]}), _reply({"rest": True})],
                            [PASS])
    monkeypatch.setattr(cli, "real_runner", runner)
    brain = ["--brain", str(root.path)]
    assert cli.main([*brain, "think", "g", "--workspace", str(workspace)]) == 12
    out = capsys.readouterr().out.strip().splitlines()
    assert out[-1] == ("terminal: interrupted — open question q1 (director): answer it with "
                       "`protean ctl answer q1 <text>`, then `protean resume`")
    (notice,) = notes.read_text().splitlines()  # exactly one notification, through the fake osascript
    task = state.current(root)
    assert notice.startswith(f"{task.id}: waiting: question q1 (director)")
    assert task.terminal == "interrupted" and [q.id for q in open_questions(task)] == ["q1"]
    assert [(e["hold"], e["on"]) for e in _events(root, task.id, "parked")] == [("question", "q1")]
    assert not root.console_file(task.id).exists()

    assert cli.main([*brain, "ctl", "answer", "q1", "use", "a.txt"]) == 0  # no brain live: written to task.json
    assert cli.main([*brain, "ctl", "think", "keep", "it", "small"]) == 0
    assert cli.main([*brain, "ctl", "pause"]) == 2 and "pause needs a live brain" in capsys.readouterr().err
    saved = load(root, task.id)
    (asked,) = saved.questions
    assert (asked.status, asked.answer, asked.actor, asked.disposition) == ("answered", "use a.txt", "console", None)
    assert [t.text for t in saved.thoughts] == ["keep it small"] and saved.terminal == "interrupted"
    assert [(e["id"], e["live"]) for e in _events(root, task.id, "answer")] == [("q1", False)]

    assert cli.main([*brain, "resume"]) == 12  # the unit ran and the session rested: it parks awaiting a thought
    message = json.loads(_director_messages(runner)[1])
    assert message["notes"] == []
    assert message["answer"] == {"id": "q1", "question": "which file?", "answer": "use a.txt"}
    assert message["thoughts"] == ["keep it small"]
    first, second = notes.read_text().splitlines()  # one notification per unattended hold: the question, the thought
    assert first == notice and second.startswith(f"{task.id}: waiting: a thought after m3")
    assert _parked_on_a_thought(root, state.current(root))
    assert sorted(p.name for p in root.path.iterdir()) == ["editor.md", "state", "thinker.md"]  # no file flow


# --- the editor's real sandbox cannot read the token; no token in its env or argv ---


@pytest.mark.skipif(shutil.which("sandbox-exec") is None,
                    reason="sandbox-exec is absent on this host, so the real editor profile cannot be exercised")
def test_the_editor_under_its_real_profile_cannot_read_the_console_file_and_never_carries_the_token(
    env: tuple[Root, Path]
):
    root, workspace = env
    task = Task(id="task-20261010T160000", goal="g", workspace=str(workspace),
                units=[Unit("u1", "write a.txt", [Predicate("file_exists", {"path": "a.txt"})])])
    save(root, task)
    set_current(root, task.id)
    state.append_log(root, task.id, {"event": "start"})
    scripted, launches = ScriptedRunner([], [PASS]), []

    def recording(argv, stdin, cwd, timeout):
        launches.append((list(argv), stdin, dict(os.environ)))  # what a launched child would inherit
        return scripted(argv, stdin, cwd, timeout)

    with console.serve(root, task, Caps()) as desk:
        token, secret = desk.token, root.console_file(task.id).resolve()
        assert json.loads(secret.read_text())["token"] == token  # the brain itself can read it
        res = worker.run_unit(task, task.units[0], Caps(), root, recording)
        ((argv, stdin, launched_env),) = launches
        assert not any(token in part for part in argv) and token not in stdin
        assert not any(token in value for value in launched_env.values())
        sandbox, profile = argv[0], argv[2]
        assert argv[1] == "-p" and f'(deny file-read* (literal "{secret}"))' in profile

        def confined(*command: str) -> subprocess.CompletedProcess:
            return subprocess.run([sandbox, "-p", profile, *command], cwd=res.clone, capture_output=True, text=True)

        read = confined("/bin/cat", str(secret))
        assert read.returncode != 0 and read.stdout == "" and "Operation not permitted" in read.stderr
        (res.clone / "door").symlink_to(secret)
        through_link = confined("/bin/cat", "door")
        assert through_link.returncode != 0 and through_link.stdout == ""
        hard_link = confined("/bin/ln", str(secret), "copy")
        assert hard_link.returncode != 0 and not (res.clone / "copy").exists()
        beside = confined("/bin/cat", str(root.task_file(task.id).resolve()))
        assert beside.returncode == 0 and task.id in beside.stdout  # the deny is that one file
        child_env = real_runner([sandbox, "-p", profile, "/usr/bin/env"], None, res.clone, 10)
        assert child_env.returncode == 0 and child_env.stdout and token not in child_env.stdout
        assert token not in root.task_file(task.id).read_text() and token not in root.log_file(task.id).read_text()


# --- the stream's last line is the terminal snapshot (gap found by the app's real-wire check, 2026-10-11) ---


def test_the_stream_ends_with_a_snapshot_of_the_state_the_run_ended_in(env: tuple[Root, Path]):
    """A client that takes state only from snapshots (the app) must see how the run ended: the console stops
    answering `/snapshot` as it closes, so the stream itself carries one last snapshot before EOF. A session never
    ends `done`: with the client attached, a rest waits for a thought until the client's
    `stop` ends the run."""
    root, workspace = env
    runner = ScriptedRunner([_reply({"task": [UNIT]}), _reply({"rest": True})], [PASS])

    def script(seen: dict) -> None:
        client = _client(root)
        stream = Stream(client)
        stream.start()
        stream.wait_for(_rested)
        time.sleep(0.3)  # attached: the brain waits for a thought instead of parking
        seen["stop"] = client.call("POST", "/stop")
        stream.join(15)
        seen["lines"] = list(stream.lines)
        stream.close()

    thread, seen = _in_client_thread(script)
    with pytest.raises(KeyboardInterrupt):
        loop.run("g", workspace, root, Caps(), runner, unattended=0.2)
    thread.join(15)
    assert "error" not in seen, repr(seen.get("error"))
    task = state.current(root)
    first, *middle, last = seen["lines"]
    assert set(first) == {"snapshot"} and first["snapshot"]["terminal"] is None and seen["stop"][0] == 200
    assert all("event" in line for line in middle) and "director" in [line["event"] for line in middle]
    # A stop parks the record (decision 26, 2026-10-23): the `parked` event follows the `stop`, and the last snapshot
    # carries `interrupted`, so a client that takes state only from snapshots reads Parked, never a lost brain.
    assert [line["event"] for line in middle][-2:] == ["stop", "parked"]
    assert [(e["hold"], e["on"]) for e in _events(root, task.id, "parked")] == [("stop", "the operator")]
    assert set(last) == {"snapshot"} and (last["snapshot"]["terminal"], task.awaiting_thought) == ("interrupted", True)
    assert (last["snapshot"]["tick"], last["snapshot"]["cost_usd"]) == (task.tick, task.cost_usd)
    assert last["snapshot"]["log_bytes"] == root.log_file(task.id).stat().st_size  # nothing after it is missing


def _stop_during_the_clone_with_a_client_attached(root: Root, workspace: Path,
                                                  monkeypatch: pytest.MonkeyPatch) -> tuple[dict, Task]:
    """Run a session whose first unit's clone is cut by a stop while a client holds the stream, as a window would:
    what the client saw (`lines`) and the record on disk after the run."""
    runner = ScriptedRunner([_reply({"task": [UNIT]})], [])
    attached = threading.Event()

    def stop_during_the_clone(*args: object, **kwargs: object) -> None:
        attached.wait(10)  # the client holds the stream before the stop lands, as a window would
        raise KeyboardInterrupt

    monkeypatch.setattr(worker, "fresh_clone", stop_during_the_clone)

    def script(seen: dict) -> None:
        client = _client(root)
        stream = Stream(client)
        stream.start()
        stream.wait_for(lambda line: "snapshot" in line)
        attached.set()
        stream.join(15)
        seen["lines"] = list(stream.lines)
        stream.close()

    thread, seen = _in_client_thread(script)
    with pytest.raises(KeyboardInterrupt):
        loop.run("g", workspace, root, Caps(), runner, unattended=0.2)
    thread.join(15)
    assert "error" not in seen, repr(seen.get("error"))
    return seen, state.current(root)


def test_a_stop_during_a_clone_streams_the_record_on_disk_as_its_last_snapshot(env: tuple[Root, Path],
                                                                            monkeypatch: pytest.MonkeyPatch):
    """Audit 2026-10-23, H3 (F1's stream half): the mark writes the record on disk, but the stream's last snapshot was
    still built from the loop's in-memory task — the attached window's final page read "Attempts 1" for a unit that
    never ran while `task.json` and `/session` said 0. The console now serves the record the mark wrote."""
    root, workspace = env
    seen, task = _stop_during_the_clone_with_a_client_attached(root, workspace, monkeypatch)
    *_, last = seen["lines"]
    assert set(last) == {"snapshot"} and last["snapshot"]["terminal"] == "interrupted"
    assert [u["attempts"] for u in last["snapshot"]["units"]] == [u.attempts for u in task.units] == [0]


def test_a_second_interrupt_mid_mark_still_streams_the_record_on_disk(env: tuple[Root, Path],
                                                                     monkeypatch: pytest.MonkeyPatch):
    """Audit 2026-10-23, I3: the console took the on-disk record only once the mark had landed, so a second
    KeyboardInterrupt mid-mark (a double Ctrl-C, SIGTERM then SIGHUP) left it serving the dirty task — "Attempts 1",
    terminal None, a lost brain — while `task.json` said interrupted at 0. The console takes the record the moment it
    is read, before the mark; the `parked` event the second signal cuts is the one thing missing."""
    root, workspace = env
    real = state.append_log

    def second_interrupt(root_: Root, task_id: str, event: dict) -> None:
        if event.get("event") == "parked":
            raise KeyboardInterrupt  # the second signal lands after the save, before the parked event
        real(root_, task_id, event)

    monkeypatch.setattr(state, "append_log", second_interrupt)
    seen, task = _stop_during_the_clone_with_a_client_attached(root, workspace, monkeypatch)
    assert (task.terminal, [u.attempts for u in task.units], _events(root, task.id, "parked")) == ("interrupted", [0], [])
    *_, last = seen["lines"]
    assert set(last) == {"snapshot"} and last["snapshot"]["terminal"] == "interrupted"
    assert [u["attempts"] for u in last["snapshot"]["units"]] == [0]


# --- stop must end the run however the brain was launched (found by the app session, 2026-10-11) ---


def test_stop_during_a_wait_ends_the_run_even_when_sigint_is_ignored(env: tuple[Root, Path]):
    """A brain launched in the background (a shell's `&`, later launchd) inherits SIGINT ignored, so the stop's
    SIGINT alone did nothing while the loop waited on an open question. The wait itself must honor the stop."""
    root, workspace = env
    runner = ScriptedRunner([_reply({"ask": "which file?"})], [])
    previous = signal.signal(signal.SIGINT, signal.SIG_IGN)
    try:
        def script(seen: dict) -> None:
            client = _client(root)
            deadline = time.monotonic() + 10
            while not client.call("GET", "/snapshot")[1]["questions"] and time.monotonic() < deadline:
                time.sleep(0.05)
            seen["stop"] = client.call("POST", "/stop")

        thread, seen = _in_client_thread(script)
        with pytest.raises(KeyboardInterrupt):
            loop.run("g", workspace, root, Caps(), runner, unattended=30.0)
        thread.join(10)
    finally:
        signal.signal(signal.SIGINT, previous)
    assert "error" not in seen, repr(seen.get("error"))
    assert seen["stop"][0] == 200
    task = state.current(root)
    # A stop parks the record (`interrupted`, decision 26 of 2026-10-23) with the question open: resume continues it.
    assert task.terminal == "interrupted" and [q.status for q in task.questions] == ["open"]
    assert [e["event"] for e in director.read_log(root, task.id)][-2:] == ["stop", "parked"]
    assert not root.console_file(task.id).exists()


# --- what the operator reads beside the plan, and what the app prices a call in flight with ---


def test_the_snapshot_carries_each_unit_s_summary_for_the_operator_on_the_task_and_on_a_held_plan():
    """The operator's plain sentence rides beside the intent wherever a client shows a unit: "" when the director gave none,
    never the worker's report."""
    done = Unit("u-1", "write a.txt", [Predicate("file_exists", {"path": "a.txt"})], status="passed",
                summary="wrote a.txt exit_code: 0 [grade: passed]", plain_summary="Adds a.txt.")
    bare = Unit("u-2", "write c.txt", [Predicate("file_exists", {"path": "c.txt"})])
    held = state.Held(kind="plan", units=[Unit("u-3", "write b.txt", [Predicate("file_exists", {"path": "b.txt"})],
                                               plain_summary="Adds b.txt.")])
    snap = console.snapshot(Task("t", "g", "/w", units=[done, bare], held=held), Caps(), live=False)
    assert [(u["id"], u["summary"]) for u in snap["units"]] == [("u-1", "Adds a.txt."), ("u-2", "")]
    assert [(u["id"], u["summary"]) for u in snap["held"]["units"]] == [("u-3", "Adds b.txt.")]


def test_the_snapshot_prices_each_seat_from_its_model_and_null_when_unpinned():
    """The brain prices nothing itself: it hands the app each seat's pinned prices, null for a model config prices
    nothing for."""
    opus = {"input": 5.0, "output": 25.0, "cache_read": 0.5, "cache_creation": 10.0}
    caps = Caps(prices={"claude-opus-5": opus}, reader_model="made-up-reader-3")
    assert console.snapshot(Task("t", "g", "/w"), caps, live=False)["prices"] == {
        "director": opus, "worker": opus, "reader": None}
    assert console.snapshot(Task("t", "g", "/w"), Caps(), live=False)["prices"] == {
        "director": None, "worker": None, "reader": None}


def test_snapshot_names_the_project() -> None:
    """The snapshot carries `project`, the repo the workspace clones (a client may show it); the default `_root`
    passes through unchanged."""
    assert console.snapshot(Task("t", "g", "/w", project="demo"), Caps(), live=False)["project"] == "demo"
    assert console.snapshot(Task("t", "g", "/w"), Caps(), live=False)["project"] == "_root"
