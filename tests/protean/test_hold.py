"""`mode` and `approve`, driven by the scripted client over the real server thread.

In `hold`, a `task` move's plan and a retry attempt each wait at the tick boundary for `approve`, which releases
exactly that one progression; `approve` with nothing held is refused. A think while something is held returns it to
the thinker, which moves again. The accepted-ref move and the thinker's own call are never held. In `auto` nothing
waits. (The explore's hold is `test_moves.py`'s.)
"""

from __future__ import annotations

import importlib.util
import json
import threading
import time
from pathlib import Path

import pytest

from protean import cli, loop, state
from protean.dry import ScriptedRunner, make_workspace
from protean.graph import console, worker
from protean.graph.budget import Caps
from protean.state import Held, Predicate, Root, Task, Unit, load, save, set_current


def _exemplar():
    """The battery's scripted console client lives in `test_console.py`; it is loaded from there, not copied (the
    battery imports test files by path, so it is not importable by name)."""
    spec = importlib.util.spec_from_file_location("protean_console_client", Path(__file__).with_name("test_console.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_c = _exemplar()
Client, Stream, _client, _director_messages, _events, _in_client_thread, _reply = (
    _c.Client, _c.Stream, _c._client, _c._director_messages, _c._events, _c._in_client_thread, _c._reply)
_rested, _parked_on_a_thought = _c._rested, _c._parked_on_a_thought

U1 = {"id": "u-1", "intent": "write a.txt", "predicates": [{"kind": "file_exists", "args": {"path": "a.txt"}}]}
U2 = {"id": "u-2", "intent": "write b.txt", "predicates": [{"kind": "file_exists", "args": {"path": "b.txt"}}]}
PASS1 = {"write": {"a.txt": "A\n"}, "commit": True, "exit_code": 0, "result": "wrote a.txt", "total_cost_usd": 0.5}
PASS2 = {"write": {"b.txt": "B\n"}, "commit": True, "exit_code": 0, "result": "wrote b.txt", "total_cost_usd": 0.5}
FAIL1 = {"write": {"c.txt": "wrong\n"}, "commit": True, "exit_code": 0, "result": "wrote c.txt", "total_cost_usd": 0.5}


@pytest.fixture
def env(tmp_path: Path, root: Root) -> tuple[Root, Path]:
    root.path.mkdir(parents=True)
    root.thinker_prompt.write_text("m\n")
    root.editor_prompt.write_text("e\n")
    return root, make_workspace(tmp_path / "ws")


def _held_first(scripted: ScriptedRunner, ready: threading.Event, in_call: threading.Event | None = None):
    """The scripted runner, its first call (its message already built) held until the client has attached and
    switched the task to `hold`; `in_call` is set once that call is in flight."""
    def runner(argv, stdin, cwd, timeout):
        if not scripted.calls:
            if in_call is not None:
                in_call.set()
            assert ready.wait(10), "the client never switched to hold"
        return scripted(argv, stdin, cwd, timeout)
    return runner


def _worker_calls(scripted: ScriptedRunner) -> int:
    return sum("Write" in argv for argv, _, _, _ in scripted.calls)


def _attach_and_hold(root: Root, ready: threading.Event, seen: dict, before) -> tuple[Client, Stream]:
    """Attach a stream, check that `approve` with nothing held is refused, switch to `hold`, run `before`, then let
    the first call go: whatever it decodes is held."""
    client = _client(root)
    stream = Stream(client)
    stream.start()
    stream.wait_for(lambda line: "snapshot" in line)
    seen["approve_nothing"] = client.call("POST", "/approve", {})
    seen["mode"] = client.call("POST", "/mode", {"mode": "hold"})
    if before is not None:
        before(client, seen)
    ready.set()
    return client, stream


def _run(env: tuple[Root, Path], scripted: ScriptedRunner, script, unattended: float = 0.2,
         before=None) -> tuple[Task, dict]:
    root, workspace = env
    ready, in_call = threading.Event(), threading.Event()
    seen_call = (lambda client, seen: before(client, seen, in_call)) if before is not None else None

    def body(seen: dict) -> None:
        client, stream = _attach_and_hold(root, ready, seen, seen_call)
        try:
            script(client, stream, seen)
        finally:
            seen["lines"] = list(stream.lines)
            stream.close()

    thread, seen = _in_client_thread(body)
    task = loop.run("g", workspace, root, Caps(), _held_first(scripted, ready, in_call), unattended=unattended)
    thread.join(15)
    assert "error" not in seen, repr(seen.get("error"))
    return task, seen


# --- a plan and a retry each wait for approve, which releases exactly one; nothing held -> 409 ---


def test_in_hold_a_plan_and_a_retry_each_wait_for_approve_which_releases_exactly_one(env: tuple[Root, Path]):
    root, _ = env
    scripted = ScriptedRunner([_reply({"task": [U1]}), _reply({"rest": True})], [FAIL1, PASS1])

    def script(client: Client, stream: Stream, seen: dict) -> None:
        stream.wait_for(lambda line: line.get("event") == "held" and line["kind"] == "plan")
        time.sleep(0.3)  # the loop waits at the boundary: no worker call while the plan is held
        seen["plan_held"] = (client.call("GET", "/snapshot")[1], _worker_calls(scripted), load(root, state.current(root).id))
        seen["approve_plan"] = client.call("POST", "/approve", {})
        seen["approve_again"] = client.call("POST", "/approve", {})  # exactly one: nothing else is held
        stream.wait_for(lambda line: line.get("event") == "held" and line["kind"] == "retry")
        time.sleep(0.3)
        seen["retry_held"] = (client.call("GET", "/snapshot")[1], _worker_calls(scripted))
        seen["approve_retry"] = client.call("POST", "/approve", {})
        stream.wait_for(_rested)

    task, seen = _run(env, scripted, script)
    assert seen["approve_nothing"] == (409, {"ok": False, "error": "nothing is held"})
    assert seen["mode"][0] == 200 and seen["mode"][1]["event"]["mode"] == "hold"
    snap, workers, on_disk = seen["plan_held"]
    assert (snap["mode"], snap["held"]["kind"], [u["id"] for u in snap["held"]["units"]]) == ("hold", "plan", ["u-1"])
    assert snap["units"] == [] and workers == 0 and on_disk.units == [] and on_disk.held.kind == "plan"
    assert seen["approve_plan"][0] == 200 and seen["approve_plan"][1]["event"]["units"] == ["u-1"]
    assert seen["approve_again"][0] == 409
    snap, workers = seen["retry_held"]
    assert snap["held"] == {"kind": "retry", "approved": False, "thoughts": 0, "unit": "u-1", "attempt": 2}
    assert workers == 1 and snap["units"][0]["attempts"] == 1  # attempt 2 waits: no second worker call
    assert seen["approve_retry"][0] == 200 and seen["approve_retry"][1]["event"]["attempt"] == 2

    assert _parked_on_a_thought(root, task) and task.held is None and _worker_calls(scripted) == 2
    order = [line["event"] for line in seen["lines"][1:]]
    assert [e for e in order if e in ("held", "approve", "approved", "verdict")] == [
        "held", "approve", "approved", "verdict", "held", "approve", "approved", "verdict"]
    assert [(e["kind"], e["by"]) for e in _events(root, task.id, "approved")] == [("plan", "approve"), ("retry", "approve")]


def test_in_auto_nothing_waits(env: tuple[Root, Path]):
    root, workspace = env
    scripted = ScriptedRunner([_reply({"task": [U1]}), _reply({"rest": True})], [FAIL1, PASS1])
    task = loop.run("g", workspace, root, Caps(), scripted, unattended=0.0, notify=None)  # no client: a wait would park
    assert (task.terminal, task.mode, task.held, _worker_calls(scripted)) == ("interrupted", "auto", None, 2)
    logged = [e["event"] for e in state.read_log(root, task.id)]
    assert not set(logged) & {"held", "approve", "approved", "returned"}
    assert logged[-1] == "parked" and logged.count("parked") == 1  # only the rest's wait for a thought


# --- a think while something is held returns it to the thinker ---


def test_a_think_on_a_held_plan_drops_it_and_the_thinker_gets_every_thought(env: tuple[Root, Path]):
    root, _ = env
    scripted = ScriptedRunner([_reply({"task": [U1]}), _reply({"ask": "smaller how?"})], [])

    def before(client: Client, seen: dict, in_call: threading.Event) -> None:
        assert in_call.wait(10)  # lands during the first thinker call: before the hold, so it returns nothing
        seen["think_before"] = client.call("POST", "/think", {"text": "use tabs"})

    def script(client: Client, stream: Stream, seen: dict) -> None:
        stream.wait_for(lambda line: line.get("event") == "held")
        seen["held_disk"] = load(root, state.current(root).id)
        seen["think"] = client.call("POST", "/think", {"text": "make it smaller"})
        stream.wait_for(lambda line: line.get("event") == "question")
        seen["snapshot"] = client.call("GET", "/snapshot")[1]

    task, seen = _run(env, scripted, script, unattended=0.5, before=before)
    assert seen["think_before"][0] == 200 and seen["think"][0] == 200
    assert seen["held_disk"].units == [] and [u.id for u in seen["held_disk"].held.units] == ["u-1"]
    (returned,) = _events(root, task.id, "returned")
    assert (returned["kind"], returned["units"], returned["thoughts"]) == ("plan", ["u-1"], 2)
    assert seen["held_disk"].held.directions == 1
    assert "returned with their thoughts and none of its units entered the session" in returned["to_director"]
    first, second = (json.loads(m) for m in _director_messages(scripted))
    assert first["thoughts"] == []  # built before either think
    assert second["thoughts"] == ["use tabs", "make it smaller"]  # the whole list, in order
    assert second["units"] == [] and second["notes"] == [returned["to_director"]]
    assert seen["snapshot"]["held"] is None and seen["snapshot"]["units"] == []
    saved = root.task_file(task.id).read_text()
    assert task.units == [] and task.held is None and '"u-1"' not in saved and "write a.txt" not in saved
    assert task.terminal == "interrupted" and _worker_calls(scripted) == 0
    assert [(m.kind, m.outcome) for m in task.moves] == [("task", "returned"), ("ask", "asked")]


def test_a_think_on_a_held_retry_cancels_it_and_the_thinker_is_told(env: tuple[Root, Path]):
    root, _ = env
    scripted = ScriptedRunner([_reply({"task": [U1]}), _reply({"ask": "retry it?"})], [FAIL1])

    def script(client: Client, stream: Stream, seen: dict) -> None:
        stream.wait_for(lambda line: line.get("event") == "held")
        assert client.call("POST", "/approve", {})[0] == 200
        stream.wait_for(lambda line: line.get("event") == "held" and line["kind"] == "retry")
        seen["think"] = client.call("POST", "/think", {"text": "write b.txt instead"})
        stream.wait_for(lambda line: line.get("event") == "question")

    task, seen = _run(env, scripted, script, unattended=0.5)
    assert seen["think"][0] == 200
    (returned,) = _events(root, task.id, "returned")
    assert (returned["kind"], returned["unit"], returned["attempt"]) == ("retry", "u-1", 2)
    assert "the retry was returned with their thoughts and did not run" in returned["to_director"]
    (unit,) = task.units
    assert (unit.status, unit.attempts, _worker_calls(scripted)) == ("failed", 1, 1)  # attempt 2 never ran
    second = json.loads(_director_messages(scripted)[1])
    assert second["thoughts"] == ["write b.txt instead"] and returned["to_director"] in second["notes"]
    assert [u["status"] for u in second["units"]] == ["failed"] and task.held is None


# --- in hold, the accepted-ref move and the thinker's own call are never held ---


def test_in_hold_the_ref_move_and_the_thinker_s_call_never_wait_for_approve(env: tuple[Root, Path]):
    root, workspace = env
    scripted = ScriptedRunner([_reply({"task": [U1, {**U2, "needs": ["u-1"]}]}), _reply({"rest": True})],
                              [PASS1, PASS2])

    def script(client: Client, stream: Stream, seen: dict) -> None:
        stream.wait_for(lambda line: line.get("event") == "held")
        seen["approve"] = client.call("POST", "/approve", {})
        stream.wait_for(_rested)

    task, seen = _run(env, scripted, script)
    assert seen["approve"][0] == 200 and _parked_on_a_thought(root, task) and task.mode == "hold"
    order = [line["event"] for line in seen["lines"][1:]]
    tail = order[order.index("approved"):]
    assert [e for e in tail if e in ("integrated", "held", "approve", "approved", "rest")] == [
        "approved", "integrated", "integrated", "rest"]  # both ref moves, then a rest: no further hold or approve
    assert order.count("approve") == 1 and order.count("held") == 1
    candidate = [u.candidate for u in task.units if u.id == "u-2"][0]
    assert worker.git(["rev-parse", f"task/{task.id}"], workspace).strip() == candidate == task.accepted


# --- mode auto releases what is held; the unattended park keeps it; offline approve + resume ---


def test_switching_to_auto_releases_a_held_plan(env: tuple[Root, Path]):
    root, _ = env
    scripted = ScriptedRunner([_reply({"task": [U1]}), _reply({"rest": True})], [PASS1])

    def script(client: Client, stream: Stream, seen: dict) -> None:
        stream.wait_for(lambda line: line.get("event") == "held")
        seen["auto"] = client.call("POST", "/mode", {"mode": "auto"})
        stream.wait_for(_rested)

    task, seen = _run(env, scripted, script)
    assert seen["auto"][0] == 200 and (seen["auto"][1]["event"]["mode"], seen["auto"][1]["event"]["was"]) == (
        "auto", "hold")
    assert _parked_on_a_thought(root, task) and task.mode == "auto"
    assert [(e["kind"], e["units"], e["by"]) for e in _events(root, task.id, "approved")] == [("plan", ["u-1"], "mode")]


def test_unattended_a_held_plan_parks_and_ctl_approve_offline_then_resume_releases_it(
    env: tuple[Root, Path], monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    root, workspace = env
    scripted = ScriptedRunner([_reply({"task": [U1]}), _reply({"rest": True})], [PASS1])
    ready = threading.Event()

    def switch(seen: dict) -> None:  # no stream: nobody stays attached
        seen["mode"] = _client(root).call("POST", "/mode", {"mode": "hold"})
        ready.set()

    thread, seen = _in_client_thread(switch)
    task = loop.run("g", workspace, root, Caps(), _held_first(scripted, ready), unattended=0.0, notify=None)
    thread.join(10)
    assert "error" not in seen, repr(seen.get("error"))
    assert task.terminal == "interrupted" and task.held.kind == "plan" and task.units == []
    assert loop.terminal_line(root, task) == (
        "terminal: interrupted — held plan (u-1): `protean ctl approve` releases it "
        "(`protean ctl think <text>` returns it to the thinker), then `protean resume`")
    assert [(e["hold"], e["on"]) for e in _events(root, task.id, "parked")] == [("held", "plan")]

    brain = ["--brain", str(root.path)]
    monkeypatch.setattr(cli, "real_runner", scripted)
    assert cli.main([*brain, "ctl", "approve"]) == 0  # no brain live: written to task.json
    assert "approve plan (u-1)" in capsys.readouterr().out
    parked = load(root, task.id)
    assert parked.held.approved is True and parked.units == []
    assert cli.main([*brain, "ctl", "approve"]) == 2  # exactly one: already approved
    assert cli.main([*brain, "resume"]) == 12  # the plan ran, then the rest parks awaiting a thought
    done = load(root, task.id)
    assert _parked_on_a_thought(root, done) and done.held is None and [u.status for u in done.units] == ["passed"]
    assert [(e["kind"], e["by"]) for e in _events(root, task.id, "approved")] == [("plan", "approve")]


# --- the two operations at the wire: token, validation, persistence ---


def test_mode_and_approve_need_the_token_are_checked_and_land_in_task_json(env: tuple[Root, Path]):
    root, workspace = env
    task = Task(id="task-20261010T180000", goal="g", workspace=str(workspace))
    save(root, task)
    set_current(root, task.id)
    with console.serve(root, task, Caps(), interrupt=lambda: None) as desk:
        client = Client(root, task.id)
        for path, body in (("/mode", {"mode": "hold"}), ("/approve", {})):
            assert client.call("POST", path, body, token=None) == (401, {"ok": False, "error": "unauthorized"})
            assert client.call("GET", path, None)[0] == 405
        for body in ({}, {"mode": "manual"}, {"mode": 1}):
            assert client.call("POST", "/mode", body)[0] == 400
        assert client.call("POST", "/mode", {"mode": "auto"}) == (409, {"ok": False, "error": "already auto"})
        assert client.call("POST", "/approve", {}) == (409, {"ok": False, "error": "nothing is held"})
        status, payload = client.call("POST", "/mode", {"mode": "hold"})
        assert status == 200 and load(root, task.id).mode == "hold" and desk.task.mode == "hold"
        assert {k: payload["event"][k] for k in ("event", "actor", "live", "mode", "was")} == {
            "event": "mode", "actor": "console", "live": True, "mode": "hold", "was": "auto"}
        with state.LOCK:  # what the loop does when a plan is held: saved first, then logged
            task.held = Held(kind="plan", units=[Unit("u-1", "write a.txt", [Predicate("file_exists", {"path": "a.txt"})])])
            save(root, task)
        snap = client.call("GET", "/snapshot")[1]
        assert snap["held"] == {"kind": "plan", "approved": False, "thoughts": 0, "units": [
            {"id": "u-1", "intent": "write a.txt", "summary": "", "needs": [],
             "predicates": [{"kind": "file_exists", "args": {"path": "a.txt"}}]}]}
        assert client.call("POST", "/approve", {}) == (200, {"ok": True, "event": {
            "event": "approve", "actor": "console", "live": True, "kind": "plan", "units": ["u-1"]}})
        assert load(root, task.id).held.approved is True and task.held.approved is True
        assert client.call("POST", "/approve", {})[0] == 409
        assert [e["event"] for e in state.read_log(root, task.id)] == ["mode", "approve"]
        assert [(e["kind"], e["units"]) for e in _events(root, task.id, "approve")] == [("plan", ["u-1"])]


def test_mode_and_held_round_trip_and_an_older_record_loads_as_auto_with_nothing_held():
    held = Held(kind="plan", units=[Unit("u-1", "write a.txt", [Predicate("file_exists", {"path": "a.txt"})],
                                         needs=["u-0"])], directions=2, approved=True)
    task = Task("t", "g", "/w", mode="hold", held=held)
    again = state.from_json(state.to_json(task))
    assert again == task and again.held.units[0].predicates[0] == Predicate("file_exists", {"path": "a.txt"})
    retry = Task("t", "g", "/w", mode="hold", held=Held(kind="retry", unit="u-1", attempt=2))
    assert state.from_json(state.to_json(retry)) == retry
    older = json.loads(state.to_json(Task("t", "g", "/w")))
    for key in ("mode", "held"):
        del older[key]
    loaded = state.from_json(json.dumps(older))
    assert (loaded.mode, loaded.held) == ("auto", None)
