"""The fixture brain (`protean dry <scenario> --serve`): a dry run served like a live brain for a console client to
develop against. It starts paused and makes no tick before a client presses Resume; holds wait on the console; the
scripted editor's lines reach the transcript; `--ticks` brings it to a ceiling that `extend` releases; `stuck` rests
once its question is answered; a rest waits for a thought while a client is attached and parks once it leaves; the
real root's `state/` is never touched; `think` waits for a think after its ideas, and the one the client sends runs it
on to its rest."""

from __future__ import annotations

import http.client
import json
import os
import socket
import subprocess
import threading
import time
from pathlib import Path

import pytest

from protean import cli, dry, state
from protean.dry import REPO
from protean.state import Root, Task, load

TEMPLATE = Root(REPO / "brain" / "protean")


class Served:
    """One served scenario on its own thread, and a plain HTTP client on its console."""

    def __init__(self, scenario: str, **kw: object) -> None:
        self.lines: list[str] = []
        self.result: list[Task] = []
        self.thread = threading.Thread(target=self._run, args=(scenario,), kwargs=kw, daemon=True)
        self.thread.start()
        self.root = Root(Path(self._line("served on ").split("served on ", 1)[1]))
        self.task_id = self._wait(lambda: state.current(self.root)).id
        # the console file is created, then written: read it until it parses (an empty read is a ValueError, retried)
        info = self._wait(lambda: json.loads(self.root.console_file(self.task_id).read_text()))
        self.port, self.token = info["port"], info["token"]

    def _run(self, scenario: str, **kw: object) -> None:
        self.result.append(dry.dry(scenario, TEMPLATE, echo=self.lines.append, serve=True, delay=0.0,
                                   unattended=0.2, **kw))

    def _line(self, text: str) -> str:
        return self._wait(lambda: next((l for l in self.lines if text in l), None))

    @staticmethod
    def _wait(get, timeout: float = 10.0):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                found = get()
            except (OSError, ValueError):
                found = None
            if found is not None:
                return found
            time.sleep(0.02)
        raise AssertionError("timed out waiting")

    def call(self, method: str, path: str, body: dict | None = None) -> tuple[int, dict]:
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=10)
        try:
            conn.request(method, path, body=None if body is None else json.dumps(body).encode(),
                         headers={"Authorization": f"Bearer {self.token}"})
            resp = conn.getresponse()
            raw = resp.read()
            return resp.status, json.loads(raw) if raw else {}
        finally:
            conn.close()

    def stream(self) -> "Stream":
        return Stream(self.port, self.token)

    def task(self) -> Task:
        return load(self.root, self.task_id)

    def finish(self, timeout: float = 30.0) -> Task:
        self.thread.join(timeout)
        assert not self.thread.is_alive(), f"the served run did not end; last lines: {self.lines[-3:]}"
        return self.result[0]


class Stream(threading.Thread):
    """One `/stream` connection: counts as an attached client; its JSON lines collect until EOF."""

    def __init__(self, port: int, token: str) -> None:
        super().__init__(daemon=True)
        self.sock = socket.create_connection(("127.0.0.1", port), timeout=10)
        self.sock.settimeout(None)
        self.sock.sendall(f"GET /stream HTTP/1.0\r\nAuthorization: Bearer {token}\r\n\r\n".encode())
        self.lines: list[dict] = []
        self.ended = False
        self.start()

    def run(self) -> None:
        handle = self.sock.makefile("rb")
        try:
            handle.readline()
            while handle.readline().strip():
                pass
            for raw in handle:
                self.lines.append(json.loads(raw))
        except (OSError, ValueError):
            pass
        finally:
            self.ended = True

    def wait_for(self, match, timeout: float = 15.0) -> dict:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            found = next((line for line in self.lines if match(line)), None)
            if found is not None:
                return found
            time.sleep(0.02)
        raise AssertionError(f"no matching stream line in {self.lines[-5:]}")

    def leave_after_the_rest(self) -> dict:
        """Wait for the rest's record change (the last line before the wait for a thought), then detach, so the
        brain parks once the unattended timeout passes."""
        rested = self.wait_for(lambda e: e.get("event") == "thought" and e.get("kind") == "rest")
        try:
            self.sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        self.sock.close()
        return rested


def _real_state_listing() -> list[str]:
    folder = TEMPLATE.path / "state"
    return sorted(os.listdir(folder)) if folder.is_dir() else []


def test_serve_starts_paused_and_makes_no_tick_before_a_client_presses_resume():
    before = _real_state_listing()
    served = Served("dry")
    assert "/protean-dry-" in str(served.root.path) and served.root.path != TEMPLATE.path
    assert any(line.startswith(f"attach: protean --brain {served.root.path} ctl watch") for line in served.lines)
    stream = served.stream()
    time.sleep(0.5)
    paused = served.task()
    assert (paused.tick, paused.paused, paused.calls, paused.terminal) == (0, True, [], None)
    assert [e["event"] for e in state.read_log(served.root, served.task_id)] == ["start", "pause"]
    status, payload = served.call("POST", "/resume", {})
    assert status == 200 and payload["event"]["event"] == "resume"
    stream.wait_for(lambda e: e.get("event") == "transcript")
    stream.leave_after_the_rest()
    task = served.finish()
    assert (task.terminal, task.awaiting_thought, task.tick) == ("interrupted", True, 4)
    served._wait(lambda: True if stream.ended else None)
    assert "snapshot" in stream.lines[0] and stream.lines[0]["snapshot"]["paused"] is True
    kinds = [e["event"] for e in stream.lines[1:]]
    assert kinds.count("transcript") == 6 and "verdict" in kinds and kinds[0] == "resume"  # two editor calls, 3 lines each
    first = next(e for e in stream.lines if e.get("event") == "transcript")
    assert json.loads(first["line"]) == {"type": "system", "subtype": "init", "session_id": "dry-worker"}
    assert _real_state_listing() == before and not served.root.path.exists()


def test_ticks_caps_the_scenario_at_a_ceiling_that_extend_releases():
    served = Served("dry", ticks=2)
    stream = served.stream()
    served.call("POST", "/resume", {})
    stopped = stream.wait_for(lambda e: e.get("event") == "stopped")
    assert stopped["reason"] == "max_ticks: 2 >= 2" and served.task().terminal == "stopped"
    status, payload = served.call("POST", "/extend", {"ticks": 10})
    assert status == 200 and payload["event"]["max_ticks"] == 12
    stream.leave_after_the_rest()
    assert served.finish().terminal == "interrupted"


def test_stuck_served_waits_for_the_answer_then_records_the_note_and_rests():
    served = Served("stuck")
    stream = served.stream()
    served.call("POST", "/resume", {})
    asked = stream.wait_for(lambda e: e.get("event") == "question" and e.get("reason") == "question")
    assert asked["id"] == "q1" and served.task().units[0].status == "failed"
    assert served.call("POST", "/answer", {"id": "q1", "text": "There is no loader; drop the unit."})[0] == 200
    stream.leave_after_the_rest()
    task = served.finish()
    assert (task.terminal, task.awaiting_thought) == ("interrupted", True)
    assert [(u.id, u.status) for u in task.units] == [("u-1", "failed"), ("u-2", "passed")]  # no descope: u-1 stays
    assert [m.kind for m in task.moves] == ["task", "ask", "task", "rest"] and task.moves[1].outcome == "answered"


def test_hold_mode_holds_the_first_plan_for_approve():
    served = Served("dry", mode="hold")
    stream = served.stream()
    served.call("POST", "/resume", {})
    held = stream.wait_for(lambda e: e.get("event") == "held")
    assert held["kind"] == "plan" and served.task().units == []
    assert served.call("POST", "/approve", {})[0] == 200
    stream.leave_after_the_rest()
    assert served.finish().terminal == "interrupted"


def test_think_served_waits_for_a_think_after_its_ideas_and_shows_the_thought_record():
    """`protean dry think --serve`, the app's thought page fixture: paused until Resume, then the ideas, and the wait
    for a think lasts while a client is attached (its own scripted client never runs); the snapshot shows the record,
    the moves and the wait; the think the client sends runs the explore, the task and the rest."""
    served = Served("think")
    stream = served.stream()
    assert served.call("POST", "/resume", {})[0] == 200
    ideas = stream.wait_for(lambda e: e.get("event") == "ideas")
    time.sleep(0.4)  # attached: the brain waits for a think instead of parking
    status, snap = served.call("GET", "/snapshot")
    assert status == 200 and snap["awaiting_thought"] is True and snap["thoughts"] == []
    assert [m["kind"] for m in snap["moves"]] == ["ideas"] and len(snap["moves"][0]["ideas"]) == len(ideas["ideas"])
    assert snap["thought"]["thought"] == "dry scenario: think" and snap["thought"]["open"][0]["id"] == "o1"
    assert [e["event"] for e in state.read_log(served.root, served.task_id)].count("think") == 0
    status, payload = served.call("POST", "/think", {"text": "Go with i1."})
    assert status == 200 and payload["event"]["index"] == 1
    stream.leave_after_the_rest()
    task = served.finish()
    assert [m.kind for m in task.moves] == ["ideas", "explore", "task", "rest"] and task.units[0].integrated
    assert (task.terminal, task.awaiting_thought, [t.text for t in task.thoughts]) == ("interrupted", True,
                                                                                      ["Go with i1."])


def test_think_served_lists_the_documents_its_branches_committed_and_serves_them_byte_identical():
    """The documents order's fixture row: `think`'s task unit commits `notes/hello.md` beside `hello.txt` and its
    explore commits `notes/hello-draft.md`; at the rest the snapshot's `artifacts` lists both branches, each with its
    tip, and `GET /file` returns the fixture's `write` text byte-identical."""
    served = Served("think")
    stream = served.stream()
    assert served.call("POST", "/resume", {})[0] == 200
    stream.wait_for(lambda e: e.get("event") == "ideas")
    assert served.call("POST", "/think", {"text": "Go with i1."})[0] == 200
    stream.wait_for(lambda e: e.get("event") == "thought" and e.get("kind") == "rest")
    status, snap = served.call("GET", "/snapshot")
    workspace, task_id = Path(snap["workspace"]), served.task_id

    def rev(ref: str) -> str:
        return subprocess.run(["git", "rev-parse", ref], cwd=workspace, capture_output=True, text=True,
                              check=True).stdout.strip()

    assert status == 200 and snap["artifacts"] == {"base": rev("HEAD"), "branches": [
        {"branch": f"task/{task_id}", "tip": rev(f"task/{task_id}"), "paths": ["hello.txt", "notes/hello.md"]},
        {"branch": f"explore/{task_id}/1", "tip": rev(f"explore/{task_id}/1"), "paths": ["notes/hello-draft.md"]}]}
    conn = http.client.HTTPConnection("127.0.0.1", served.port, timeout=10)
    try:
        conn.request("GET", f"/file?branch=task/{task_id}&path=notes/hello.md",
                     headers={"Authorization": f"Bearer {served.token}"})
        resp = conn.getresponse()
        got = (resp.status, resp.getheader("Content-Type"), resp.read())
    finally:
        conn.close()
    written = dry.read_jsonl(dry.FIXTURES / "think" / "worker.jsonl")[1]["write"]["notes/hello.md"]
    assert got == (200, "application/octet-stream", written.encode("utf-8"))
    stream.leave_after_the_rest()
    assert served.finish().terminal == "interrupted"


def test_the_dry_verb_takes_the_serve_flags():
    args = cli.build_parser().parse_args(["dry", "stuck", "--serve", "--mode", "hold", "--delay", "0.5", "--ticks", "3"])
    assert (args.serve, args.mode, args.delay, args.ticks) == (True, "hold", 0.5, 3)
    plain = cli.build_parser().parse_args(["dry", "dry"])
    assert (plain.serve, plain.mode, plain.delay, plain.ticks) == (False, "auto", 1.0, None)


def test_without_serve_a_scenario_still_parks_at_once(capsys: pytest.CaptureFixture[str]):
    assert cli.main(["dry", "stuck"]) == 12
    assert "served on" not in capsys.readouterr().out
