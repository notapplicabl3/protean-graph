"""`GET /history` on the console and the lobby, `GET /stream?offsets=1`, and `capabilities`.

A page reads `state/<task>/log.jsonl` or `transcript.jsonl` through `Root`, never a path the request names. `size` is the
end of the last complete line; entries are contiguous (`offset + bytes` is the next `offset`, newline counted, in raw
bytes); no cursor is the newest page, `after=A` reads forward, `before=B` is the page ending at B. `limit` is 1–500
(default 200), a page's whole encoded body is at most 262144 bytes (`test_record_review_fixes.py` holds the escaping
cases), a line over 65536 bytes is an `omitted` entry that still advances the page, and a row that is not a JSON
object is `raw`. A bad query is a `400`, a cursor past the end a `409`.
Reads are chunked: no request loads a whole file. The offsets stream's envelopes are the history entries, and the plain
stream is byte-identical to the file.
"""

from __future__ import annotations

import http.client
import json
import os
import socket
import threading
import time
from collections.abc import Iterator
from pathlib import Path

import pytest

from protean.graph import console, history, lobby
from protean.graph.budget import Caps
from protean.state import Root, Task, save

TASK_ID = "task-20261026T130000"
CAPABILITIES = ["request_id", "history", "stream_offsets", "walk"]


class Surface:
    """One of the two routes: the live console's `/history`, or the lobby's `/history?task=<id>`."""

    def __init__(self, kind: str, root: Root, port: int, token: str) -> None:
        self.kind, self.root, self.port, self.token = kind, root, port, token

    def call(self, method: str, path: str, token: object = True) -> tuple[int, bytes]:
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=30)
        headers = {}
        if token is not None:
            headers["Authorization"] = f"Bearer {self.token if token is True else token}"
        try:
            conn.request(method, path, headers=headers)
            resp = conn.getresponse()
            return resp.status, resp.read()
        finally:
            conn.close()

    def path(self, query: str) -> str:
        parts = ([f"task={TASK_ID}"] if self.kind == "lobby" else []) + ([query] if query else [])
        return "/history?" + "&".join(parts)

    def get(self, query: str) -> tuple[int, dict]:
        status, raw = self.call("GET", self.path(query))
        return status, json.loads(raw)

    def snapshot(self) -> dict:
        path = "/snapshot" if self.kind == "console" else f"/session?task={TASK_ID}"
        status, raw = self.call("GET", path)
        assert status == 200
        return json.loads(raw)


def _session(root: Root, **fields: object) -> Task:
    root.path.mkdir(parents=True, exist_ok=True)
    task = Task(id=TASK_ID, thought="t", workspace="/nowhere", created="2026-10-26T13:00:00+00:00", **fields)
    save(root, task)
    return task


@pytest.fixture(params=["console", "lobby"])
def surface(request: pytest.FixtureRequest, root: Root) -> Iterator[Surface]:
    task = _session(root)
    if request.param == "console":
        with console.serve(root, task, Caps(), interrupt=lambda: None) as desk:
            yield Surface("console", root, desk.address[1], desk.token)
        return
    desk = lobby.Lobby(root, spawn=lambda *a, **k: None, echo=lambda _: None)
    desk.start()
    try:
        yield Surface("lobby", root, desk.address[1], desk.token)
    finally:
        desk.close()


def _row(obj: object) -> bytes:
    return json.dumps(obj, ensure_ascii=False).encode("utf-8") + b"\n"


def _write(root: Root, source: str, rows: list[bytes], tail: bytes = b"") -> list[int]:
    """The file is exactly `rows` then `tail`; returns each row's offset."""
    path = root.log_file(TASK_ID) if source == "log" else root.transcript_file(TASK_ID)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"".join(rows) + tail)
    offsets, at = [], 0
    for row in rows:
        offsets.append(at)
        at += len(row)
    return offsets


def _entry(offset: int, row: bytes) -> dict:
    """The entry a row is served as: its JSON object, else its text as `raw`, else the omission marker."""
    content = row[:-1]
    if len(content) > history.LINE_BOUND:
        return {"offset": offset, "bytes": len(row), "omitted": "over_line_bound"}
    try:
        value = json.loads(content.decode("utf-8"))
    except ValueError:
        value = None
    if isinstance(value, dict):
        return {"offset": offset, "bytes": len(row), "line": value}
    return {"offset": offset, "bytes": len(row), "raw": content.decode("utf-8", "replace")}


def _contiguous(page: dict) -> None:
    at = page["from"]
    for entry in page["lines"]:
        assert entry["offset"] == at
        at += entry["bytes"]
    assert at == page["to"] and page["older"] == (page["from"] > 0) and page["newer"] == (page["to"] < page["size"])


ROWS = [_row({"ts": "2026-10-26T13:00:00+00:00", "event": "start", "thought": "t", "mode": "auto"}),
        _row({"ts": "2026-10-26T13:00:01+00:00", "event": "think", "text": "café ☕ — naïve", "index": 1}),
        _row({"ts": "2026-10-26T13:00:02+00:00", "event": "question", "id": "q1", "text": "which?"}),
        _row({"ts": "2026-10-26T13:00:03+00:00", "event": "answer", "id": "q1", "text": "this one"}),
        _row({"ts": "2026-10-26T13:00:04+00:00", "event": "future_kind", "nested": {"keep": [1, 2.5, None]}}),
        _row({"ts": "2026-10-26T13:00:05+00:00", "event": "think", "text": "again", "index": 2}),
        _row({"ts": "2026-10-26T13:00:06+00:00", "event": "think", "text": "again", "index": 2})]


# --- pages ---


@pytest.mark.parametrize("source", ["log", "transcript"])
def test_an_absent_file_is_an_empty_page(surface: Surface, source: str):
    assert surface.get(f"source={source}") == (200, {
        "ok": True, "task": TASK_ID, "source": source, "size": 0, "from": 0, "to": 0, "older": False, "newer": False,
        "lines": []})


@pytest.mark.parametrize("source", ["log", "transcript"])
def test_no_cursor_is_the_newest_page_and_the_cursors_walk_every_line_contiguously(surface: Surface, source: str):
    offsets = _write(surface.root, source, ROWS)
    size = sum(len(r) for r in ROWS)
    every = [_entry(o, r) for o, r in zip(offsets, ROWS)]
    status, page = surface.get(f"source={source}&limit=3")
    assert status == 200 and (page["task"], page["source"], page["size"], page["to"]) == (TASK_ID, source, size, size)
    assert page["lines"] == every[-3:] and (page["older"], page["newer"]) == (True, False)
    _contiguous(page)
    # raw bytes, not characters: the row with non-ASCII text counts its encoded length
    assert every[1]["bytes"] == len(ROWS[1]) != len(ROWS[1].decode("utf-8"))
    assert every[4]["line"]["nested"] == {"keep": [1, 2.5, None]}  # an unknown event and its fields kept whole
    backward, cursor = list(page["lines"]), page["from"]
    while True:
        status, older = surface.get(f"source={source}&before={cursor}&limit=3")
        assert status == 200 and older["to"] == cursor
        _contiguous(older)
        backward = older["lines"] + backward
        cursor = older["from"]
        if not older["older"]:
            break
    forward, cursor = [], 0
    while True:
        status, newer = surface.get(f"source={source}&after={cursor}&limit=3")
        assert status == 200 and newer["from"] == cursor
        _contiguous(newer)
        forward += newer["lines"]
        cursor = newer["to"]
        if not newer["newer"]:
            break
    assert backward == forward == every  # the two equal rows at the end stay two entries
    assert surface.get(f"source={source}")[1]["lines"] == every  # the default limit holds them all


def test_before_is_the_page_ending_there_and_after_reads_forward_from_there(surface: Surface):
    offsets = _write(surface.root, "log", ROWS)
    size = sum(len(r) for r in ROWS)
    every = [_entry(o, r) for o, r in zip(offsets, ROWS)]
    page = surface.get(f"source=log&before={offsets[3]}&limit=2")[1]
    assert (page["from"], page["to"], page["lines"]) == (offsets[1], offsets[3], every[1:3])
    page = surface.get(f"source=log&after={offsets[3]}&limit=2")[1]
    assert (page["from"], page["to"], page["lines"]) == (offsets[3], offsets[5], every[3:5])
    page = surface.get("source=log&before=0")[1]
    assert (page["from"], page["to"], page["lines"], page["older"], page["newer"]) == (0, 0, [], False, True)
    page = surface.get(f"source=log&after={size}")[1]
    assert (page["from"], page["to"], page["lines"], page["older"], page["newer"]) == (size, size, [], True, False)
    assert surface.get("source=log&limit=500")[1]["lines"] == every
    assert surface.get("source=log&limit=1")[1]["lines"] == every[-1:]


def test_a_partial_trailing_line_is_never_served_and_the_snapshot_offsets_stop_before_it(surface: Surface):
    _write(surface.root, "log", ROWS, tail=b'{"event": "half", "text": "not yet')
    _write(surface.root, "transcript", ROWS[:2], tail=b'{"event": "transcript"')
    size = sum(len(r) for r in ROWS)
    page = surface.get("source=log")[1]
    assert (page["size"], page["to"], len(page["lines"])) == (size, size, len(ROWS))
    assert surface.get(f"source=log&after={size}")[1]["lines"] == []
    assert surface.get(f"source=log&after={size + 3}") == (409, {
        "ok": False, "error": f"offset {size + 3} is past the end of log ({size} bytes)"})
    snap = surface.snapshot()
    assert (snap["log_bytes"], snap["transcript_bytes"]) == (size, len(ROWS[0]) + len(ROWS[1]))


def test_rows_that_are_not_json_objects_are_raw_and_keep_their_bytes(surface: Surface):
    rows = [b"not json\n", b"[1, 2]\n", b"\n", b"  \n", b"\xff\xfe broken\n", b'"just a string"\n', b"null\n",
            b'{"ok": 1, "event": "x"}\n']
    offsets = _write(surface.root, "transcript", rows)
    page = surface.get("source=transcript")[1]
    assert page["lines"] == [_entry(o, r) for o, r in zip(offsets, rows)]
    assert [e.get("raw") for e in page["lines"]] == ["not json", "[1, 2]", "", "  ", "�� broken",
                                                      '"just a string"', "null", None]
    _contiguous(page)


def test_a_line_over_the_bound_is_an_omission_marker_that_advances_the_page(surface: Surface):
    def sized(n: int) -> bytes:  # one JSON object whose line is exactly n bytes before its newline
        head = b'{"event": "transcript", "line": "'
        return head + b"x" * (n - len(head) - 2) + b'"}\n'

    rows = [ROWS[0], sized(history.LINE_BOUND), sized(history.LINE_BOUND + 1), ROWS[1]]
    assert [len(r) - 1 for r in rows[1:3]] == [65536, 65537]
    offsets = _write(surface.root, "log", rows)
    page = surface.get("source=log&after=0")[1]
    assert [sorted(set(e) - {"offset", "bytes"}) for e in page["lines"]] == [["line"], ["line"], ["omitted"], ["line"]]
    assert page["lines"][2] == {"offset": offsets[2], "bytes": 65538, "omitted": "over_line_bound"}
    assert page["lines"] == [_entry(o, r) for o, r in zip(offsets, rows)]
    page = surface.get(f"source=log&after={offsets[2]}&limit=1")[1]
    assert (page["lines"], page["to"]) == ([_entry(offsets[2], rows[2])], offsets[3])  # it advances the cursor
    page = surface.get("source=log&limit=2")[1]
    assert page["lines"] == [_entry(offsets[2], rows[2]), _entry(offsets[3], rows[3])]
    page = surface.get(f"source=log&before={offsets[3]}&limit=1")[1]
    assert (page["lines"], page["from"]) == ([_entry(offsets[2], rows[2])], offsets[2])


def test_a_page_holds_at_most_256_kib_of_lines(surface: Surface):
    rows = [_row({"event": "transcript", "n": n, "line": "y" * 60_000}) for n in range(6)]
    offsets = _write(surface.root, "transcript", rows)
    size = sum(len(r) for r in rows)
    assert 4 * len(rows[0]) <= history.PAGE_BOUND < 5 * len(rows[0])
    page = surface.get("source=transcript&after=0&limit=500")[1]
    assert [e["line"]["n"] for e in page["lines"]] == [0, 1, 2, 3] and page["newer"]
    assert sum(e["bytes"] for e in page["lines"]) <= history.PAGE_BOUND
    _contiguous(page)
    rest = surface.get(f"source=transcript&after={page['to']}&limit=500")[1]
    assert [e["line"]["n"] for e in rest["lines"]] == [4, 5] and rest["to"] == size and not rest["newer"]
    newest = surface.get("source=transcript&limit=500")[1]
    assert [e["line"]["n"] for e in newest["lines"]] == [2, 3, 4, 5] and newest["from"] == offsets[2]


# --- refusals ---


def test_bad_queries_are_400s_and_a_cursor_past_the_end_is_a_409(surface: Surface):
    offsets = _write(surface.root, "log", ROWS)
    size = sum(len(r) for r in ROWS)
    for query in ["", "source=", "source=nope", "source=LOG", "source=log&source=log", "source=log&source=transcript",
                  "source=../task.json", "source=log&after=0&before=0", "source=log&after=-1", "source=log&after=1.5",
                  "source=log&after=abc", "source=log&after=%201", "source=log&after=", "source=log&after=%2B1",
                  "source=log&after=0x1", "source=log&after=%D9%A1", "source=log&after=0&after=0",
                  "source=log&before=1&before=1", "source=log&limit=0", "source=log&limit=501",
                  "source=log&limit=abc", "source=log&limit=", "source=log&limit=1&limit=2", "source=log&limit=-1",
                  "source=log&path=..%2Ftask.json", "source=log&task=other", "source=log&offsets=1",
                  "source=log&after=1", f"source=log&before={offsets[1] + 1}", f"source=log&after={size - 1}"]:
        status, body = surface.get(query)
        assert (status, body["ok"]) == (400, False) and body["error"], query
    for query, at in [(f"source=log&after={size + 1}", size + 1), (f"source=log&before={10 ** 30}", 10 ** 30),
                      ("source=transcript&after=1", 1)]:
        source = "transcript" if "transcript" in query else "log"
        end = 0 if source == "transcript" else size
        assert surface.get(query) == (409, {"ok": False,
                                            "error": f"offset {at} is past the end of {source} ({end} bytes)"}), query


def test_history_needs_the_token_and_takes_get_only(surface: Surface):
    _write(surface.root, "log", ROWS)
    for token in (None, "not-the-token", surface.token[:-1]):
        status, raw = surface.call("GET", surface.path("source=log"), token=token)
        assert (status, json.loads(raw)) == (401, {"ok": False, "error": "unauthorized"}), token
    for method in ("POST", "PUT", "DELETE"):
        status, raw = surface.call(method, surface.path("source=log"))
        assert status == 405 and json.loads(raw)["ok"] is False, method


def test_the_lobby_answers_task_as_session_does_and_reads_an_ended_session(root: Root):
    _session(root, terminal="stopped", stop_reason="closed")
    _write(root, "log", ROWS)
    corrupt = "task-20261026T140000"
    root.state_dir(corrupt).mkdir(parents=True)
    root.task_file(corrupt).write_text("{not json", encoding="utf-8")
    (root.state_dir(corrupt) / "log.jsonl").write_bytes(ROWS[0])
    desk = lobby.Lobby(root, spawn=lambda *a, **k: None, echo=lambda _: None)
    desk.start()
    try:
        side = Surface("lobby", root, desk.address[1], desk.token)
        status, page = side.get("source=log")
        assert status == 200 and len(page["lines"]) == len(ROWS)  # no brain serves it: the record is read offline
        for path, code in [("/history?source=log", 400), ("/history?task=&source=log", 400),
                           (f"/history?task={TASK_ID}&task={TASK_ID}&source=log", 400),
                           ("/history?task=../x&source=log", 404), ("/history?task=..&source=log", 404),
                           ("/history?task=state%2F..&source=log", 404),
                           ("/history?task=task-20991231T000000&source=log", 404),
                           (f"/history?task={corrupt}&source=log", 404)]:
            status, raw = side.call("GET", path)
            assert (status, json.loads(raw)["ok"]) == (code, False), path
        status, raw = side.call("GET", "/sessions")
        assert status == 200 and json.loads(raw)["capabilities"] == ["history"]
    finally:
        desk.close()


def test_history_reads_only_the_session_s_two_files_and_never_through_a_link(surface: Surface, tmp_path: Path):
    secret = tmp_path / "outside.jsonl"
    secret.write_bytes(_row({"event": "secret", "text": "SECRET-OUTSIDE"}))
    log = surface.root.log_file(TASK_ID)
    log.parent.mkdir(parents=True, exist_ok=True)
    log.symlink_to(secret)
    status, raw = surface.call("GET", surface.path("source=log"))
    assert status == 404 and b"SECRET-OUTSIDE" not in raw and json.loads(raw)["ok"] is False
    (surface.root.state_dir(TASK_ID) / "other.jsonl").write_bytes(_row({"event": "other"}))
    status, page = surface.get("source=transcript")
    assert status == 200 and page["lines"] == []  # no transcript: the sibling files are never read


# --- bounded reads ---


def test_reads_are_chunked_and_never_load_a_whole_file(root: Root, monkeypatch: pytest.MonkeyPatch):
    rows = [_row({"event": "transcript", "n": n, "line": "z" * 90}) for n in range(30_000)]  # about 3.3 MB
    offsets = _write(root, "transcript", rows)
    path, size = root.transcript_file(TASK_ID), sum(len(r) for r in rows)
    reads: list[int] = []
    real = history._read_at

    def spy(fd: int, count: int, offset: int) -> bytes:
        reads.append(count)
        return real(fd, count, offset)

    monkeypatch.setattr(history, "_read_at", spy)
    for query, first in (({}, 29_800), ({"after": 0}, 0), ({"before": size}, 29_800), ({"after": offsets[15_000]}, 15_000),
                         ({"before": offsets[15_000]}, 14_800)):
        reads.clear()
        status, page = history.page(path, TASK_ID, "transcript", **query, limit=200)
        assert status == 200 and [e["line"]["n"] for e in page["lines"]] == list(range(first, first + 200)), query
        assert reads and max(reads) <= history.CHUNK, query
        assert sum(reads) < history.PAGE_BOUND + 4 * history.CHUNK + history.LINE_BOUND, query


def test_a_small_chunk_gives_the_same_pages(root: Root, monkeypatch: pytest.MonkeyPatch):
    rows = ROWS + [b"not json\n", b"\n", _row({"event": "x", "line": "w" * (history.LINE_BOUND + 5)}), ROWS[0]]
    offsets = _write(root, "log", rows)
    path, size = root.log_file(TASK_ID), sum(len(r) for r in rows)
    queries = [{}, {"after": 0}, {"before": size}, {"after": offsets[3], "limit": 2}, {"before": offsets[9], "limit": 3},
               {"limit": 1}]
    wide = [history.page(path, TASK_ID, "log", **q) for q in queries]
    monkeypatch.setattr(history, "CHUNK", 7)
    assert [history.page(path, TASK_ID, "log", **q) for q in queries] == wide
    assert wide[1][1]["lines"] == [_entry(o, r) for o, r in zip(offsets, rows)]


# --- the stream ---


class RawStream(threading.Thread):
    """One `/stream` connection read to its end as bytes, so a test can compare them with the file's."""

    def __init__(self, port: int, token: str, path: str) -> None:
        super().__init__(daemon=True)
        self.sock = socket.create_connection(("127.0.0.1", port), timeout=10)
        self.sock.sendall(f"GET {path} HTTP/1.0\r\nAuthorization: Bearer {token}\r\n\r\n".encode())
        self.data, self.ended, self.cond = bytearray(), False, threading.Condition()

    def run(self) -> None:
        try:
            while chunk := self.sock.recv(65536):
                with self.cond:
                    self.data += chunk
                    self.cond.notify_all()
        except OSError:
            pass
        finally:
            with self.cond:
                self.ended = True
                self.cond.notify_all()

    def lines(self) -> list[bytes]:
        with self.cond:
            body = bytes(self.data).split(b"\r\n\r\n", 1)[1] if b"\r\n\r\n" in self.data else b""
        return [line + b"\n" for line in body.split(b"\n")[:-1]]

    def wait_lines(self, count: int, timeout: float = 10.0) -> list[bytes]:
        deadline = time.monotonic() + timeout
        while len(self.lines()) < count:
            assert time.monotonic() < deadline and not self.ended, f"only {self.lines()}"
            time.sleep(0.01)
        return self.lines()

    def wait_end(self, timeout: float = 10.0) -> list[bytes]:
        with self.cond:
            self.cond.wait_for(lambda: self.ended, timeout)
        self.sock.close()
        return self.lines()


def _append(path: Path, rows: list[bytes]) -> None:
    with path.open("ab") as handle:
        for row in rows:
            handle.write(row)


def test_the_offsets_stream_carries_the_history_entries_and_the_plain_stream_is_unchanged(root: Root):
    task = _session(root)
    before = [_row({"event": "start", "thought": "t"}), _row({"event": "think", "text": "first"})]
    _write(root, "log", before)
    log_rows = [_row({"event": "think", "text": "naïve ☕", "unknown": {"x": 1}}), b"not a json object\n",
                _row({"event": "big", "line": "q" * (history.LINE_BOUND + 10)}), _row({"event": "think", "text": "x"})]
    transcript_rows = [_row({"event": "transcript", "seat": "worker", "call": "c1", "line": "{\"type\": \"x\"}"})]
    with console.serve(root, task, Caps(), interrupt=lambda: None) as desk:
        side = Surface("console", root, desk.address[1], desk.token)
        plain = RawStream(desk.address[1], desk.token, "/stream")
        offsets = RawStream(desk.address[1], desk.token, "/stream?offsets=1")
        for stream in (plain, offsets):
            stream.start()
        first_plain, first_offsets = plain.wait_lines(1)[0], offsets.wait_lines(1)[0]
        snap = json.loads(first_offsets)["snapshot"]
        assert set(json.loads(first_plain)["snapshot"]) == set(snap) and snap["capabilities"] == CAPABILITIES
        assert snap["log_bytes"] == sum(len(r) for r in before) and snap["transcript_bytes"] == 0
        _append(root.log_file(TASK_ID), log_rows)
        _append(root.transcript_file(TASK_ID), transcript_rows)
        received = offsets.wait_lines(1 + len(log_rows) + len(transcript_rows))
        plain.wait_lines(1 + len(log_rows) + len(transcript_rows))
        envelopes = [json.loads(line) for line in received[1:]]
        assert all(list(e)[:3] == ["source", "offset", "bytes"] for e in envelopes)
        by_source: dict[str, list[dict]] = {"log": [], "transcript": []}
        for envelope in envelopes:
            source = envelope.pop("source")
            by_source[source].append(envelope)
            status, page = side.get(f"source={source}&after={envelope['offset']}&limit=1")
            assert status == 200 and page["lines"] == [envelope], envelope
        at = sum(len(r) for r in before)
        assert [e["offset"] for e in by_source["log"]] == [at + sum(len(r) for r in log_rows[:n])
                                                           for n in range(len(log_rows))]
        assert [sorted(set(e) - {"offset", "bytes"}) for e in by_source["log"]] == [
            ["line"], ["raw"], ["omitted"], ["line"]]
        assert by_source["transcript"] == [_entry(0, transcript_rows[0])]
        assert side.get(f"source=log&after={at}")[1]["lines"] == by_source["log"]
    ended_plain, ended_offsets = plain.wait_end(), offsets.wait_end()
    for ended in (ended_plain, ended_offsets):
        last = json.loads(ended[-1])
        assert set(last) == {"snapshot"} and last["snapshot"]["capabilities"] == CAPABILITIES
    body = ended_plain[1:-1]  # the plain stream: each file's lines verbatim, byte for byte, in its own order
    assert [line for line in body if line in log_rows] == log_rows
    assert [line for line in body if line in transcript_rows] == transcript_rows and len(body) == 5
    assert len(ended_offsets) == len(ended_plain)


def test_capabilities_are_advertised_on_the_snapshot_and_the_lobby(root: Root):
    task = _session(root)
    with console.serve(root, task, Caps(), interrupt=lambda: None):
        status, snap = console.request(root, TASK_ID, "GET", "/snapshot")
        assert status == 200 and snap["capabilities"] == CAPABILITIES
    assert console.snapshot(task, Caps(), False)["capabilities"] == CAPABILITIES  # the offline snapshot too
    desk = lobby.Lobby(root, spawn=lambda *a, **k: None, echo=lambda _: None)
    desk.start()
    try:
        side = Surface("lobby", root, desk.address[1], desk.token)
        assert side.snapshot()["capabilities"] == CAPABILITIES
        status, raw = side.call("GET", "/sessions")
        listed = json.loads(raw)
        assert status == 200 and listed["capabilities"] == ["history"] and [s["task"] for s in listed["sessions"]] == [
            TASK_ID]
    finally:
        desk.close()
    assert not os.path.exists(root.lobby_file)


def test_large_live_transcript_preserves_usage_while_history_stays_bounded(root: Root):
    task = _session(root)
    line = {"event": "transcript", "seat": "worker", "call": "c1",
            "line": json.dumps({"type": "assistant", "message": {"content": [{"type": "text", "text": "x" * 70000}],
                               "usage": {"input_tokens": 12, "output_tokens": 6000}}})}
    payload = _row(line)
    with console.serve(root, task, Caps(), interrupt=lambda: None) as desk:
        stream = RawStream(desk.address[1], desk.token, "/stream?offsets=1")
        stream.start()
        stream.wait_lines(1)
        _append(root.transcript_file(TASK_ID), [payload])
        received = json.loads(stream.wait_lines(2)[1])
        assert received.get("line") == line, "live usage must survive the history row limit"
        status, page = history.page(root.transcript_file(TASK_ID), TASK_ID, "transcript")
        assert status == 200 and page["lines"][0]["omitted"] == "over_line_bound"
    stream.wait_end()
