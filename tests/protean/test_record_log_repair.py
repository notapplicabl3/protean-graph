"""F3 REPRO, written before the fix (independent review of the record contract, low finding): a think or an answer
lands in `task.json` (the commit point) and its log append then fails, so the accepted input has no line in the log
and `/history` never shows it. These tests use only names the pre-fix code has, so they run against it and fail there.

The repair: a replay of the same `request_id` (live and offline) and the next serve write the ledger entry's original
event to the log once, found by `request_id` and operation, never by text, reading the log a chunk at a time. Nothing
is applied again. A normal replay or serve, with the event already in the log, changes, saves and logs nothing. While
the log stays unwritable, the client is told neither "refused" nor "nothing landed": live, it gets no reply (its
delivery is unknown) until a replay can write the event; offline, the error says what landed.
"""

from __future__ import annotations

import json

import pytest

from protean import loop, state
from protean.graph import console, history
from protean.graph.budget import Caps
from protean.state import Root, Task, load, save, set_current

TASK_ID = "task-20261027T090000"


def _seeded(root: Root) -> Task:
    """A parked session with one thinker question (`q1`) and one runtime `unknown_spend` (`q2`)."""
    root.path.mkdir(parents=True, exist_ok=True)
    task = Task(id=TASK_ID, thought="t", workspace="/nowhere", terminal="interrupted")
    state.add_question(task, "director", "question", "which file?")
    state.add_question(task, "runtime", "unknown_spend", "killed mid-call; continue?", {"unknown_spend": ["c1"]})
    save(root, task)
    set_current(root, task.id)
    state.append_log(root, task.id, {"event": "start"})
    return task


def _files(root: Root) -> tuple[bytes, bytes, bytes]:
    return (root.task_file(TASK_ID).read_bytes(), root.log_file(TASK_ID).read_bytes(),
            root.thought_file(TASK_ID).read_bytes())


def _post(root: Root, path: str, body: dict) -> tuple[int, dict]:
    return console.request(root, TASK_ID, "POST", path, body)


def _events(root: Root, rid: str) -> list[dict]:
    """The logged events that carry `rid`, each without its `ts`."""
    return [{k: v for k, v in e.items() if k != "ts"} for e in state.read_log(root, TASK_ID)
            if e.get("request_id") == rid]


class BrokenLog:
    """While `on`, every append to a session's `log.jsonl` raises, as a full or failed disk would."""

    on = False


@pytest.fixture
def broken_log(monkeypatch: pytest.MonkeyPatch) -> BrokenLog:
    broken = BrokenLog()
    real = state._append

    def append(path, event):
        if broken.on and path.name == "log.jsonl":
            raise OSError("review fixture: the log cannot be written")
        real(path, event)

    monkeypatch.setattr(state, "_append", append)
    return broken


@pytest.fixture
def live(root: Root):
    task = _seeded(root)
    with console.serve(root, task, Caps(), interrupt=lambda: None) as desk:
        yield root, task, desk


@pytest.mark.parametrize("path,body", [
    ("/think", {"text": "keep this once", "request_id": "r-log"}),
    ("/answer", {"id": "q2", "text": "yes", "request_id": "a-log"}),  # releases paid work: never released twice
])
def test_f3_repro_a_failed_log_append_after_the_commit_is_written_once_by_the_replay(live, broken_log, path, body):
    root, task, desk = live
    broken_log.on = True
    with pytest.raises(console.Unavailable):  # landed, but no reply: its delivery is unknown, never refused or failed
        _post(root, path, body)
    broken_log.on = False
    saved = load(root, TASK_ID)
    (entry,) = saved.requests
    assert task.requests == saved.requests  # the canonical record and the loop's copy agree
    assert _events(root, entry.request_id) == []
    assert _post(root, path, body) == (200, {"ok": True, "event": entry.event, "replayed": True})
    assert _events(root, entry.request_id) == [entry.event]  # the original event, once, nothing applied again
    assert load(root, TASK_ID) == saved
    after, ops = _files(root), desk.ops
    assert _post(root, path, body) == (200, {"ok": True, "event": entry.event, "replayed": True})
    assert _files(root) == after and desk.ops == ops  # a normal replay changes, saves and logs nothing
    assert ops == 1  # the landed operation woke the loop once; neither replay did
    if body.get("id"):
        assert [q.status for q in saved.questions] == ["open", "answered"]
    else:
        assert [t.text for t in saved.thoughts] == ["keep this once"] == [t.text for t in task.thoughts]


def test_f3_repro_a_restart_writes_the_missing_event_once_and_a_normal_serve_writes_nothing(root: Root, broken_log):
    task = _seeded(root)
    body = {"id": "q2", "text": "yes", "request_id": "a-restart"}
    with console.serve(root, task, Caps(), interrupt=lambda: None):
        broken_log.on = True
        with pytest.raises(console.Unavailable):
            _post(root, "/answer", body)
        broken_log.on = False
    reloaded = load(root, TASK_ID)
    (entry,) = reloaded.requests
    assert _events(root, "a-restart") == []
    with console.serve(root, reloaded, Caps(), interrupt=lambda: None) as desk:
        assert _events(root, "a-restart") == [entry.event]  # written when the console serves again, before any request
        assert desk.ops == 0
        before = _files(root)
        assert _post(root, "/answer", body) == (200, {"ok": True, "event": entry.event, "replayed": True})
        assert _files(root) == before
    with console.serve(root, load(root, TASK_ID), Caps(), interrupt=lambda: None):
        pass
    assert _files(root) == before  # a serve with every event logged writes nothing
    assert load(root, TASK_ID) == reloaded and reloaded.questions[1].status == "answered"
    assert [e["id"] for e in state.read_log(root, TASK_ID) if e["event"] == "answer"] == ["q2"]


def test_f3_repro_offline_a_failed_append_after_the_save_is_written_by_the_offline_replay(root: Root, broken_log):
    _seeded(root)
    body = {"text": "offline once", "decide": True, "request_id": "r-off"}
    broken_log.on = True
    with pytest.raises(OSError):
        loop.offline_op(root, Caps(), "think", dict(body))
    broken_log.on = False
    saved = load(root, TASK_ID)
    (entry,) = saved.requests
    assert _events(root, "r-off") == []
    assert loop.offline_op(root, Caps(), "think", dict(body)) == entry.event
    assert _events(root, "r-off") == [entry.event]
    before = _files(root)
    assert loop.offline_op(root, Caps(), "think", dict(body)) == entry.event
    assert _files(root) == before
    assert load(root, TASK_ID) == saved and len(saved.thoughts) == 1 and len(saved.decided) == 1


def test_f3_repro_while_the_log_stays_unwritable_a_replay_is_never_a_200_and_changes_nothing(live, broken_log):
    root, task, desk = live
    body = {"text": "still unknown", "request_id": "r-down"}
    broken_log.on = True
    with pytest.raises(console.Unavailable):
        _post(root, "/think", body)
    record, ops = root.task_file(TASK_ID).read_bytes(), desk.ops
    with pytest.raises(console.Unavailable):  # the event still cannot be written: no reply, never a 200 or a refusal
        _post(root, "/think", body)
    assert root.task_file(TASK_ID).read_bytes() == record and desk.ops == ops and len(task.thoughts) == 1
    broken_log.on = False
    (entry,) = load(root, TASK_ID).requests
    assert _post(root, "/think", body) == (200, {"ok": True, "event": entry.event, "replayed": True})
    assert _events(root, "r-down") == [entry.event]


def test_f3_repro_the_repair_finds_events_by_request_id_never_by_text_and_reads_a_chunk_at_a_time(
        root: Root, monkeypatch: pytest.MonkeyPatch):
    task = _seeded(root)
    console.apply(task, Caps(), "think", {"text": "same words", "request_id": "r-missing"}, False)
    console.apply(task, Caps(), "think", {"text": "same words", "request_id": "r-logged"}, False)
    save(root, task)
    missing, logged = (r.event for r in task.requests)
    with root.log_file(TASK_ID).open("a", encoding="utf-8") as handle:  # about 2.4 MB ahead of the events
        for n in range(20_000):
            handle.write(json.dumps({"event": "transcript", "n": n, "line": "z" * 80}) + "\n")
        handle.write(json.dumps({"event": "think", "text": "same words", "index": 1}) + "\n")  # equal text, no id
        handle.write(json.dumps({**missing, "request_id": "r-other"}) + "\n")  # equal text, another id
    state.append_log(root, TASK_ID, logged)
    reads: list[int] = []
    real = history._read_at

    def spy(fd: int, count: int, offset: int) -> bytes:
        reads.append(count)
        return real(fd, count, offset)

    monkeypatch.setattr(history, "_read_at", spy)
    with console.serve(root, task, Caps(), interrupt=lambda: None):
        pass
    assert _events(root, "r-missing") == [missing]
    assert _events(root, "r-logged") == [logged]  # already there: not written again
    assert reads and max(reads) <= history.CHUNK
    before = _files(root)
    with console.serve(root, load(root, TASK_ID), Caps(), interrupt=lambda: None):
        pass
    assert _files(root) == before
