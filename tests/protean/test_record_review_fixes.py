"""The review batch's fixes beyond the parent's oracle (`test_record_contract_regressions.py`) and the F3 repro
(`test_record_log_repair.py`).

F1: `task.json`'s rename is a save's one commit point, on the offline path as on the live one: a save that raises has
landed nothing, and the `thought.md` a failed commit left ahead of the record is never read as state. F2: 262144 bytes
bound the whole encoded `/history` body (escapes, invalid bytes and the page's own keys counted) on the console and the
lobby, in both directions; a page is trimmed at the end it reads towards; a row too large for any page is an
`over_payload_bound` marker with its exact offset and length, so every byte span is covered and none is dropped silently.
"""

from __future__ import annotations

import http.client
import json
from collections.abc import Iterator

import pytest

from protean import loop, state
from protean.graph import console, history, lobby
from protean.graph.budget import Caps
from protean.state import Root, Task, load, save, set_current

TASK_ID = "task-20261027T100000"


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


# --- F1: one commit point ---


@pytest.mark.parametrize("op,body", [
    ("think", {"text": "keep this once", "decide": True, "request_id": "r-off"}),
    ("answer", {"id": "q2", "text": "yes", "request_id": "a-off"}),  # releases paid work under unknown spend
])
@pytest.mark.parametrize("failed_file", ["thought.md", "task.json"])
def test_offline_a_failed_save_lands_nothing_and_the_retry_lands_once(root: Root, monkeypatch: pytest.MonkeyPatch,
                                                                     op, body, failed_file):
    _seeded(root)
    record, log = root.task_file(TASK_ID).read_bytes(), root.log_file(TASK_ID).read_bytes()
    real = state._write_atomic

    def broken(path, text):
        if path.name == failed_file:
            raise OSError("fixture write failure")
        real(path, text)

    monkeypatch.setattr(state, "_write_atomic", broken)
    with pytest.raises(OSError) as failed:
        loop.offline_op(root, Caps(), op, dict(body))
    assert not isinstance(failed.value, console.Unlogged)  # a failed save: nothing landed, and it says so
    assert root.task_file(TASK_ID).read_bytes() == record and root.log_file(TASK_ID).read_bytes() == log
    monkeypatch.setattr(state, "_write_atomic", real)
    event = loop.offline_op(root, Caps(), op, dict(body))
    assert loop.offline_op(root, Caps(), op, dict(body)) == event  # the retry of the same request replays it
    saved = load(root, TASK_ID)
    assert [(r.request_id, r.event) for r in saved.requests] == [(body["request_id"], event)]
    logged = [e for e in state.read_log(root, TASK_ID) if e.get("request_id") == body["request_id"]]
    assert [{k: v for k, v in e.items() if k != "ts"} for e in logged] == [event]
    if op == "think":
        assert len(saved.thoughts) == 1 and len(saved.decided) == 1
    else:
        assert [q.status for q in saved.questions] == ["open", "answered"]
    assert root.thought_file(TASK_ID).read_text(encoding="utf-8") == state.render_thought(saved)


def test_a_view_written_ahead_of_a_failed_commit_is_never_read_as_state(root: Root, monkeypatch: pytest.MonkeyPatch):
    task = _seeded(root)
    body = {"text": "decide this", "decide": True, "request_id": "r-ahead"}
    real = state._write_atomic

    def broken(path, text):
        if path.name == "task.json":
            raise OSError("fixture write failure")
        real(path, text)

    with console.serve(root, task, Caps(), interrupt=lambda: None) as desk:
        monkeypatch.setattr(state, "_write_atomic", broken)
        assert desk.operate("think", body)[0] == 500
        monkeypatch.setattr(state, "_write_atomic", real)
        assert "decide this" in root.thought_file(TASK_ID).read_text(encoding="utf-8")  # the view is ahead
        disk = load(root, TASK_ID)
        assert (disk.thoughts, disk.decided, disk.requests) == ([], [], [])  # the record is not
        assert (task.thoughts, task.decided, task.requests) == ([], [], [])  # and neither is the loop's copy
        assert desk.snapshot()["thought"]["decided"] == [] and desk.ops == 0
        status, first = desk.operate("think", body)
        assert status == 200 and desk.operate("think", body) == (200, {"ok": True, "event": first["event"],
                                                                       "replayed": True})
    saved = load(root, TASK_ID)
    assert len(saved.decided) == 1 and saved.requests == task.requests
    assert root.thought_file(TASK_ID).read_text(encoding="utf-8") == state.render_thought(saved)


# --- F2: the bound on the whole encoded body ---


class Surface:
    """The live console's `/history`, or the lobby's `/history?task=<id>`, read as raw bytes."""

    def __init__(self, kind: str, port: int, token: str) -> None:
        self.kind, self.port, self.token = kind, port, token

    def get(self, query: str) -> tuple[int, bytes, int]:
        path = "/history?" + "&".join(([f"task={TASK_ID}"] if self.kind == "lobby" else []) + [query])
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=30)
        try:
            conn.request("GET", path, headers={"Authorization": f"Bearer {self.token}"})
            resp = conn.getresponse()
            return resp.status, resp.read(), int(resp.headers["Content-Length"])
        finally:
            conn.close()


@pytest.fixture(params=["console", "lobby"])
def surface(request: pytest.FixtureRequest, root: Root) -> Iterator[Surface]:
    root.path.mkdir(parents=True, exist_ok=True)
    task = Task(id=TASK_ID, thought="t", workspace="/nowhere")
    save(root, task)
    if request.param == "console":
        with console.serve(root, task, Caps(), interrupt=lambda: None) as desk:
            yield Surface("console", desk.address[1], desk.token)
        return
    desk = lobby.Lobby(root, spawn=lambda *a, **k: None, echo=lambda _: None)
    desk.start()
    try:
        yield Surface("lobby", desk.address[1], desk.token)
    finally:
        desk.close()


def _row(obj: object) -> bytes:
    return json.dumps(obj, ensure_ascii=False).encode("utf-8") + b"\n"


def _write(root: Root, rows: list[bytes]) -> list[int]:
    path = root.transcript_file(TASK_ID)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"".join(rows))
    offsets, at = [], 0
    for row in rows:
        offsets.append(at)
        at += len(row)
    return offsets


# Each disk row is at most 65536 bytes, so none is over the line bound but `OVER_LINE`, and each encodes to either at
# most about 200000 bytes (it fits a page alone) or at least about 300000 (it fits no page). A JSON object row grows at
# most threefold on the wire (its control characters are escaped on disk already), so only a `raw` row fits no page.
FITS = [
    _row({"event": "transcript", "line": "café ☕ — naïve"}),
    _row({"event": "transcript", "line": "é" * 30_000}),  # 2 bytes on disk, `é` (6) on the wire
    _row({"event": "transcript", "line": "中" * 20_000}),  # 3 bytes, 6 on the wire
    _row({"event": "transcript", "line": "😀" * 15_000}),  # 4 bytes, a surrogate pair (12) on the wire
]
TOO_LARGE = [
    b"\xff" * 60_000 + b"\n",  # not UTF-8: each byte is U+FFFD, `�` (6) on the wire
    b"\x01" * 60_000 + b"\n",  # a raw control byte, `\u0001` (6) on the wire
    b"\x01\xfe" * 30_000 + b"\n",  # both, mixed
]
OVER_LINE = b'{"event": "transcript", "line": "' + b"x" * 70_000 + b'"}\n'
ROWS = [FITS[0], FITS[1], TOO_LARGE[0], TOO_LARGE[1], FITS[2], FITS[3], TOO_LARGE[2], FITS[0], OVER_LINE, FITS[0]]


def _expected(offset: int, row: bytes) -> dict:
    if row in TOO_LARGE:
        return {"offset": offset, "bytes": len(row), "omitted": "over_payload_bound"}
    if row == OVER_LINE:
        return {"offset": offset, "bytes": len(row), "omitted": "over_line_bound"}
    return history.entry(offset, len(row), row[:-1])


def test_the_rows_are_what_the_test_says():
    for row in FITS:
        assert len(row) <= history.LINE_BOUND and len(json.dumps(history.entry(0, len(row), row[:-1]))) < 200_000
    for row in TOO_LARGE:
        assert len(row) <= history.LINE_BOUND and len(json.dumps(history.entry(0, len(row), row[:-1]))) > 300_000
    assert len(OVER_LINE) - 1 > history.LINE_BOUND


def _page(surface: Surface, query: str, limit: int = 500) -> dict:
    status, raw, length = surface.get(f"{query}&limit={limit}")
    assert status == 200 and length == len(raw) <= history.PAGE_BOUND, query  # the actual body, every byte
    page = json.loads(raw)
    assert page["lines"] and page["to"] > page["from"], query  # a page always moves on
    at = page["from"]
    for entry in page["lines"]:
        assert entry["offset"] == at
        at += entry["bytes"]
    assert at == page["to"]
    return page


def test_every_page_body_is_within_the_bound_both_ways_and_every_span_is_served_or_marked(surface: Surface, root):
    offsets = _write(root, ROWS)
    size = sum(len(r) for r in ROWS)
    every = [_expected(o, r) for o, r in zip(offsets, ROWS)]
    forward, cursor, pages = [], 0, 0
    while cursor < size:
        page = _page(surface, f"source=transcript&after={cursor}")
        assert page["from"] == cursor
        forward, cursor, pages = forward + page["lines"], page["to"], pages + 1
    assert forward == every and pages > 1
    backward, cursor = [], size
    while cursor > 0:
        page = _page(surface, f"source=transcript&before={cursor}" if cursor < size else "source=transcript")
        assert page["to"] == cursor
        backward, cursor = page["lines"] + backward, page["from"]
    assert backward == every
    marked = offsets[ROWS.index(TOO_LARGE[0])]
    page = _page(surface, f"source=transcript&after={marked}", limit=1)  # the marker alone moves the cursor on
    assert (page["lines"], page["to"]) == ([every[ROWS.index(TOO_LARGE[0])]], marked + len(TOO_LARGE[0]))
    assert every[ROWS.index(TOO_LARGE[0])] == {"offset": marked, "bytes": len(TOO_LARGE[0]),
                                               "omitted": "over_payload_bound"}


@pytest.mark.parametrize("forward", [True, False])
def test_a_page_is_trimmed_at_the_end_it_reads_towards_and_is_full_to_the_bound(root: Root, forward: bool):
    rows = [_row({"event": "transcript", "n": n, "line": "é" * 2_000}) for n in range(60)]  # about 12 KB each encoded
    offsets = _write(root, rows)
    size = sum(len(r) for r in rows)
    path = root.transcript_file(TASK_ID)
    status, page = history.page(path, TASK_ID, "transcript", limit=500, **({"after": 0} if forward else {}))
    body = len(json.dumps(page).encode("utf-8"))
    count = len(page["lines"])
    assert status == 200 and body <= history.PAGE_BOUND and 0 < count < len(rows)
    if forward:
        assert (page["from"], page["lines"][0]["line"]["n"], page["newer"]) == (0, 0, True)
        nxt = count
    else:
        assert (page["to"], page["lines"][-1]["line"]["n"], page["older"]) == (size, 59, True)
        nxt = len(rows) - count - 1
    following = history.entry(offsets[nxt], len(rows[nxt]), rows[nxt][:-1])
    assert body + len(json.dumps(following)) + 2 > history.PAGE_BOUND - 64  # the next one would not have fit


def test_the_offsets_stream_sends_whole_a_row_that_a_page_marks_over_payload():
    row = TOO_LARGE[0]
    whole = history.entry(10, len(row), row[:-1])
    assert "raw" in whole and json.loads(history.envelope("transcript", 10, len(row), row[:-1])) == {
        "source": "transcript", **whole}
