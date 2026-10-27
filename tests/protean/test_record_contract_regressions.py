"""Independent review regressions: one save commit point and a bound on the actual response bytes."""

import json

import pytest

from protean import state
from protean.graph import console, history
from protean.graph.budget import Caps


@pytest.mark.parametrize("op,body", [
    ("think", {"text": "Keep this input once.", "request_id": "review-think"}),
    ("answer", {"id": "q1", "text": "yes", "request_id": "review-answer"}),
])
@pytest.mark.parametrize("failed_file", ["thought.md", "task.json"])
def test_failed_save_leaves_the_canonical_record_and_live_state_unchanged(root, monkeypatch, op, body, failed_file):
    task = state.Task(id="task-review", thought="Test save failure", workspace="/nowhere")
    state.add_question(task, "runtime", "unknown_spend", "Continue after uncertain spend?", {"unknown_spend": []})
    state.save(root, task)
    before = root.task_file(task.id).read_bytes()
    real_write = state._write_atomic
    with console.serve(root, task, Caps(), interrupt=lambda: None) as desk:
        with monkeypatch.context() as patch:
            def broken_write(path, text):
                if path.name == failed_file:
                    raise OSError("review fixture write failure")
                real_write(path, text)
            patch.setattr(state, "_write_atomic", broken_write)
            status, _ = desk.operate(op, body)
            assert status == 500
            assert root.task_file(task.id).read_bytes() == before
            assert task.requests == []
            assert task.questions[0].status == "open"
        status, first = desk.operate(op, body)
        assert status == 200
        assert desk.operate(op, body) == (200, {"ok": True, "event": first["event"], "replayed": True})
        saved = state.load(root, task.id)
        assert len(saved.requests) == 1
        assert saved.requests == task.requests
        if op == "think":
            assert len(saved.thoughts) == 1
        else:
            assert saved.questions[0].status == "answered"


@pytest.mark.parametrize("raw", [False, True])
@pytest.mark.parametrize("forward", [False, True])
def test_history_bounds_the_encoded_response_including_escaping_and_metadata(root, raw, forward):
    path = root.transcript_file("task-review")
    path.parent.mkdir(parents=True, exist_ok=True)
    line = b"\xff" * 65000 if raw else json.dumps({"event": "transcript", "line": "é" * 32000}, ensure_ascii=False).encode()
    path.write_bytes((line + b"\n") * 4)
    status, page = history.page(path, "task-review", "transcript", limit=500, **({"after": 0} if forward else {}))
    assert status == 200
    assert len(json.dumps(page).encode()) <= history.PAGE_BOUND
    assert page["lines"]
    assert page["to"] > page["from"]
    assert page["to"] - page["from"] == sum(row["bytes"] for row in page["lines"])


def test_log_repair_preserves_a_torn_line_and_recovers_the_event_on_the_first_replay(root):
    task = state.Task(id="task-review", thought="Repair a torn log", workspace="/nowhere")
    event = console.apply(task, Caps(), "think", {"text": "Keep this once.", "request_id": "torn-log"}, True)
    state.save(root, task)
    path = root.log_file(task.id)
    partial = b'{"event":"unfinished"'
    path.write_bytes(partial)
    assert console.relog(root, task) == [event]
    _, page = history.page(path, task.id, "log", after=0)
    assert page["lines"][0].get("raw") == partial.decode()
    assert page["lines"][1]["line"]["request_id"] == "torn-log"
    before = path.read_bytes()
    assert console.relog(root, task) == []
    assert path.read_bytes() == before
