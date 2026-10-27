"""`request_id` on `/think` and `/answer`, live and offline.

An optional body key of 1–64 characters from `[A-Za-z0-9_-]`; anything else is a 400. The first `200` records
`{request_id, op, payload, event}` in `task.json` in the same save that lands the operation, so the record exists before
the reply; the event (logged and replied) carries the id, and a think's thought does too. A known id with the same
operation and payload is answered `200` with the original event and `replayed: true`, before any state check, and
nothing is saved, logged or changed; with another operation or payload it is a `409` and nothing changes. Refusals and
failures are never recorded. A caller that sends no id keeps today's behaviour: two equal thoughts stay two thoughts.
"""

from __future__ import annotations

import copy
import json
import threading
from pathlib import Path

import pytest

from protean import loop, state
from protean.graph import console
from protean.graph.budget import Caps
from protean.state import Root, Task, load, save, set_current

TASK_ID = "task-20261026T120000"


def _seeded(root: Root, **fields: object) -> Task:
    """A parked session with one thinker question (`q1`) and one runtime `unknown_spend` (`q2`)."""
    root.path.mkdir(parents=True, exist_ok=True)
    task = Task(id=TASK_ID, thought="t", workspace="/nowhere", terminal="interrupted", **fields)
    state.add_question(task, "director", "question", "which file?")
    state.add_question(task, "runtime", "unknown_spend", "killed mid-call; continue?", {"unknown_spend": ["c1"]})
    save(root, task)
    set_current(root, task.id)
    state.append_log(root, task.id, {"event": "start"})
    return task


def _files(root: Root) -> tuple[bytes, bytes]:
    return root.task_file(TASK_ID).read_bytes(), root.log_file(TASK_ID).read_bytes()


def _post(root: Root, path: str, body: dict) -> tuple[int, dict]:
    return console.request(root, TASK_ID, "POST", path, body)


@pytest.fixture
def live(root: Root):
    """A console served for the seeded session, no loop running; `stop` and `close` record their interrupt."""
    task = _seeded(root)
    stops: list[bool] = []
    with console.serve(root, task, Caps(), interrupt=lambda: stops.append(True)) as desk:
        yield root, task, desk


def test_a_think_with_a_request_id_lands_once_and_records_the_ledger_in_the_same_save(live, monkeypatch):
    root, task, desk = live
    written: list[dict] = []
    real_write = state._write_atomic

    def write(path: Path, text: str) -> None:
        real_write(path, text)
        if path.name == "task.json":
            written.append(json.loads(text))

    monkeypatch.setattr(state, "_write_atomic", write)
    body = {"text": "  go with i1 ", "request_id": "r-1"}
    sent = copy.deepcopy(body)
    status, payload = _post(root, "/think", body)
    assert body == sent  # the caller's body is never rewritten
    event = {"event": "think", "actor": "console", "live": True, "text": "go with i1", "index": 1, "decide": False,
             "request_id": "r-1"}
    assert (status, payload) == (200, {"ok": True, "event": event})
    (one,) = written  # one save lands the thought and its ledger entry together
    assert one["requests"] == [{"request_id": "r-1", "op": "think", "payload": {"text": "  go with i1 ", "decide": False},
                                "event": event}]
    assert one["thoughts"][0]["request_id"] == "r-1"
    logged = [e for e in state.read_log(root, TASK_ID) if e["event"] == "think"]
    assert [{k: v for k, v in e.items() if k != "ts"} for e in logged] == [event]
    snap = console.request(root, TASK_ID, "GET", "/snapshot")[1]
    assert snap["thoughts"][0]["request_id"] == "r-1"
    assert task.requests == load(root, TASK_ID).requests  # the loop's copy holds the ledger too


def test_a_replay_returns_the_original_event_and_saves_logs_and_changes_nothing(live):
    root, task, desk = live
    status, first = _post(root, "/think", {"text": "use tabs", "request_id": "r-1"})
    assert status == 200
    before, ops, thoughts = _files(root), desk.ops, list(task.thoughts)
    for body in ({"text": "use tabs", "request_id": "r-1"},
                 {"text": "use tabs", "decide": False, "request_id": "r-1"},  # `decide` absent reads as false
                 {"text": "use tabs", "request_id": "r-1", "unknown": 1}):  # identity is the op's fields only
        assert _post(root, "/think", body) == (200, {"ok": True, "event": first["event"], "replayed": True}), body
    assert _files(root) == before and desk.ops == ops and task.thoughts == thoughts


def test_the_same_id_with_another_payload_or_operation_is_refused_and_nothing_changes(live):
    root, task, desk = live
    assert _post(root, "/think", {"text": "use tabs", "request_id": "r-1"})[0] == 200
    before, ops = _files(root), desk.ops
    refused = (409, {"ok": False, "error": "request_id r-1 was used for another operation"})
    for path, body in [("/think", {"text": "use spaces", "request_id": "r-1"}),
                       ("/think", {"text": "use tabs ", "request_id": "r-1"}),  # compared exactly, as received
                       ("/think", {"text": "use tabs", "decide": True, "request_id": "r-1"}),
                       ("/answer", {"id": "q1", "text": "use tabs", "request_id": "r-1"})]:
        assert _post(root, path, body) == refused, (path, body)
    assert _files(root) == before and desk.ops == ops
    assert [q.status for q in task.questions] == ["open", "open"] and len(task.thoughts) == 1


@pytest.mark.parametrize("rid", ["", "x" * 65, "a b", "a/b", "é", "r-1\n", 5, None, True, ["r-1"]])
def test_a_malformed_request_id_is_a_400_and_nothing_changes(live, rid):
    root, task, desk = live
    before = _files(root)
    status, payload = _post(root, "/think", {"text": "use tabs", "request_id": rid})
    assert status == 400 and "request_id" in payload["error"]
    assert _post(root, "/answer", {"id": "q1", "text": "a.txt", "request_id": rid})[0] == 400
    assert _files(root) == before and task.thoughts == [] and task.requests == []


def test_the_longest_id_and_every_allowed_character_are_accepted(live):
    root, task, _ = live
    rid = ("AZaz09_-" * 8)[:64]
    assert _post(root, "/think", {"text": "x", "request_id": rid})[0] == 200
    assert [r.request_id for r in task.requests] == [rid]


def test_the_replay_check_runs_before_every_state_check_and_an_answer_is_never_released_twice(live):
    root, task, desk = live
    status, first = _post(root, "/answer", {"id": "q2", "text": "yes", "request_id": "a-1"})
    assert status == 200 and first["event"]["request_id"] == "a-1"
    before, ops = _files(root), desk.ops
    # q2 is answered now: without the id this is a 409; with it, the original answer comes back and nothing moves.
    assert _post(root, "/answer", {"id": "q2", "text": "yes"})[0] == 409
    assert _post(root, "/answer", {"id": "q2", "text": "yes", "request_id": "a-1"}) == (
        200, {"ok": True, "event": first["event"], "replayed": True})
    assert _files(root) == before and desk.ops == ops
    assert [e["id"] for e in state.read_log(root, TASK_ID) if e["event"] == "answer"] == ["q2"]
    assert _post(root, "/think", {"text": "t", "request_id": "r-9"})[0] == 200
    assert _post(root, "/close", {})[0] == 200  # closed: every later operation is refused
    assert _post(root, "/think", {"text": "t", "request_id": "r-10"})[0] == 409
    assert _post(root, "/think", {"text": "t", "request_id": "r-9"})[1]["replayed"] is True


def test_refusals_and_failures_are_never_recorded(live, monkeypatch):
    root, task, desk = live
    assert _post(root, "/answer", {"id": "q7", "text": "x", "request_id": "a-1"})[0] == 404
    assert _post(root, "/answer", {"id": "q2", "text": "maybe", "request_id": "a-1"})[0] == 409
    assert _post(root, "/think", {"text": "  ", "request_id": "a-1"})[0] == 400
    assert task.requests == [] and load(root, TASK_ID).requests == []

    def fail(*_: object) -> None:
        raise OSError("disk full")

    real_save = state.save
    monkeypatch.setattr(state, "save", fail)
    status, payload = _post(root, "/think", {"text": "keep it", "request_id": "a-1"})
    assert status == 500 and "disk full" in payload["error"]  # never a 200 when the save failed
    assert task.thoughts == [] and task.requests == []
    monkeypatch.setattr(state, "save", real_save)
    status, payload = _post(root, "/answer", {"id": "q2", "text": "yes", "request_id": "a-1"})
    assert status == 200 and "replayed" not in payload  # the id was free: nothing refused or failed was kept
    assert [r.request_id for r in load(root, TASK_ID).requests] == ["a-1"]


def test_two_equal_requests_at_once_land_one_thought(live):
    root, task, _ = live
    gate, results = threading.Barrier(4), []

    def send() -> None:
        gate.wait()
        results.append(_post(root, "/think", {"text": "once", "request_id": "r-race"}))

    threads = [threading.Thread(target=send) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(10)
    assert [status for status, _ in results] == [200] * 4
    assert sum("replayed" not in payload for _, payload in results) == 1
    assert len({json.dumps(payload["event"], sort_keys=True) for _, payload in results}) == 1
    assert [t.text for t in load(root, TASK_ID).thoughts] == ["once"] == [t.text for t in task.thoughts]


def test_without_an_id_two_equal_thoughts_stay_two_thoughts_and_nothing_new_is_written(live):
    root, task, _ = live
    events = [_post(root, "/think", {"text": "again"})[1]["event"] for _ in range(2)]
    assert events == [{"event": "think", "actor": "console", "live": True, "text": "again", "index": n, "decide": False}
                      for n in (1, 2)]
    snap = console.request(root, TASK_ID, "GET", "/snapshot")[1]
    assert snap["thoughts"] == [{"text": "again", "created": t.created, "actor": "console"} for t in task.thoughts]
    assert len(snap["thoughts"]) == 2 and load(root, TASK_ID).requests == []
    answered = _post(root, "/answer", {"id": "q1", "text": "a.txt"})[1]["event"]
    assert "request_id" not in answered


def test_a_replay_works_after_the_record_reloads_from_disk(root: Root):
    task = _seeded(root)
    with console.serve(root, task, Caps(), interrupt=lambda: None):
        status, first = _post(root, "/think", {"text": "use tabs", "decide": True, "request_id": "r-1"})
        assert status == 200
    reloaded = load(root, TASK_ID)
    before = _files(root)
    with console.serve(root, reloaded, Caps(), interrupt=lambda: None) as desk:
        assert _post(root, "/think", {"text": "use tabs", "decide": True, "request_id": "r-1"}) == (
            200, {"ok": True, "event": first["event"], "replayed": True})
        assert desk.ops == 0
    assert _files(root) == before and len(reloaded.decided) == 1
    # Offline, the live record's id replays too: the original event, `live: true` and all.
    assert loop.offline_op(root, Caps(), "think", {"text": "use tabs", "decide": True, "request_id": "r-1"}) == \
        first["event"]
    assert _files(root) == before


def test_the_offline_path_follows_the_same_rules(root: Root):
    _seeded(root)
    body = {"id": "q1", "text": "a.txt", "request_id": "a-1"}
    sent = copy.deepcopy(body)
    event = loop.offline_op(root, Caps(), "answer", body)
    assert body == sent
    assert event == {"event": "answer", "actor": "console", "live": False, "id": "q1", "asker": "director",
                     "reason": "question", "text": "a.txt", "request_id": "a-1"}
    saved = load(root, TASK_ID)
    assert [(r.request_id, r.op, r.payload, r.event) for r in saved.requests] == [
        ("a-1", "answer", {"id": "q1", "text": "a.txt"}, event)]
    before = _files(root)
    assert loop.offline_op(root, Caps(), "answer", dict(sent)) == event  # replayed: q1 is no longer open
    assert _files(root) == before
    with pytest.raises(console.Refused) as refused:
        loop.offline_op(root, Caps(), "answer", {"id": "q1", "text": "b.txt", "request_id": "a-1"})
    assert (refused.value.status, str(refused.value)) == (409, "request_id a-1 was used for another operation")
    with pytest.raises(console.Refused) as refused:
        loop.offline_op(root, Caps(), "think", {"text": "t", "request_id": "bad id"})
    assert refused.value.status == 400
    assert _files(root) == before


def test_a_record_from_before_the_ledger_loads_with_none_and_saves_it_empty():
    task = Task("t", "g", "/w", thoughts=[state.Thought("a", "2026-10-26T00:00:00+00:00")])
    older = json.loads(state.to_json(task))
    del older["requests"]
    for thought in older["thoughts"]:
        del thought["request_id"]
    loaded = state.from_json(json.dumps(older))
    assert loaded.requests == [] and loaded.thoughts[0].request_id is None and loaded == task
    assert console.snapshot(loaded, Caps(), False)["thoughts"] == [
        {"text": "a", "created": "2026-10-26T00:00:00+00:00", "actor": "console"}]  # absent when none
    assert state.from_json(state.to_json(loaded)) == loaded
