"""Client review ruling 2, the brain's side: the offsets stream sends a live transcript line whole up to the stream's
line bound (16 MiB), so the usage it carries reaches the client, while `/history` keeps its 65536-byte line bound; both
give the line the same `offset` and `bytes`. An envelope whose encoding (JSON escapes counted) would pass the client's
stream line limit is an `over_stream_bound` omission with its true offset and length, never a line the client cannot
read. The log keeps the history bound on the stream too. The parent's own case is
`test_history.py::test_large_live_transcript_preserves_usage_while_history_stays_bounded`.
"""

from __future__ import annotations

import json
import socket
import threading
import time
from pathlib import Path

import pytest

from protean.graph import console, history
from protean.graph.budget import Caps
from protean.state import Root, Task, save

TASK_ID = "task-20261027T140000"


def _session(root: Root) -> Task:
    root.path.mkdir(parents=True, exist_ok=True)
    task = Task(id=TASK_ID, thought="t", workspace="/nowhere", created="2026-10-27T14:00:00+00:00")
    save(root, task)
    return task


def _row(obj: object) -> bytes:
    return json.dumps(obj, ensure_ascii=False).encode("utf-8") + b"\n"


def _append(path: Path, rows: list[bytes]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("ab") as handle:
        for row in rows:
            handle.write(row)


class _Stream(threading.Thread):
    """One `/stream` connection read to its end as bytes."""

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
            assert time.monotonic() < deadline and not self.ended, f"only {len(self.lines())} lines"
            time.sleep(0.01)
        return self.lines()

    def wait_end(self, timeout: float = 10.0) -> list[bytes]:
        with self.cond:
            self.cond.wait_for(lambda: self.ended, timeout)
        self.sock.close()
        return self.lines()


def test_lines_keep_the_history_bound_unless_given_another():
    assert history.LINE_BOUND == 65_536 and history.STREAM_LINE_BOUND == 16 * 1024 * 1024
    assert history.stream_bound("transcript") == history.STREAM_LINE_BOUND
    assert history.stream_bound("log") == history.LINE_BOUND
    big = b"x" * (history.LINE_BOUND + 10)
    assert history.Lines(0).feed(big + b"\n") == [(0, len(big) + 1, None)]
    assert history.Lines(0, bound=history.STREAM_LINE_BOUND).feed(big + b"\n") == [(0, len(big) + 1, big)]
    cut = history.Lines(5, bound=20)  # fed in pieces, the bound still holds and the offsets still count every byte
    assert cut.feed(b"a" * 15) == []
    assert cut.feed(b"b" * 10 + b"\nc\n") == [(5, 26, None), (31, 2, b"c")]


def test_an_envelope_over_the_wire_bound_is_an_omission_with_its_offset_and_length():
    assert history.STREAM_WIRE_BOUND == 16 * 1024 * 1024 - 65_536
    text = "☕" * 1000  # 3000 source bytes, 6000 as JSON escapes
    row = _row({"event": "transcript", "call": "c1", "line": text})
    content = row[:-1]
    whole = history.envelope("transcript", 7, len(row), content)
    assert json.loads(whole)["line"]["line"] == text and whole.endswith(b"\n")
    assert len(content) < 4000 < len(whole)
    small = history.envelope("transcript", 7, len(row), content, wire_bound=4000)
    assert json.loads(small) == {"source": "transcript", "offset": 7, "bytes": len(row), "omitted": "over_stream_bound"}
    assert small.endswith(b"\n") and len(small) <= 4000
    # A line over its cutter's bound (content None) names the bound that cut it: the stream's for the transcript.
    assert json.loads(history.envelope("transcript", 0, 99, None))["omitted"] == "over_stream_bound"
    assert json.loads(history.envelope("log", 0, 99, None))["omitted"] == "over_line_bound"


def test_the_live_transcript_keeps_a_long_line_whole_and_the_page_keeps_its_bound(root: Root,
                                                                                    monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(history, "STREAM_WIRE_BOUND", 150_000)  # the client's limit, made small enough to test
    task = _session(root)
    usage = {"event": "transcript", "seat": "worker", "call": "c1",
             "line": json.dumps({"type": "assistant", "message": {"id": "m1", "content": [
                 {"type": "text", "text": "y" * 80_000}], "usage": {"input_tokens": 3, "output_tokens": 4000}}})}
    wide = {"event": "transcript", "seat": "worker", "call": "c1", "line": "☕" * 30_000}  # 90 KB source, 180 KB sent
    rows = [_row(usage), _row(wide)]
    log_row = _row({"event": "big", "pad": "q" * (history.LINE_BOUND + 10)})
    assert history.LINE_BOUND < len(rows[0]) < 150_000 and history.LINE_BOUND < len(rows[1]) < 150_000
    with console.serve(root, task, Caps(), interrupt=lambda: None) as desk:
        stream = _Stream(desk.address[1], desk.token, "/stream?offsets=1")
        stream.start()
        stream.wait_lines(1)
        _append(root.transcript_file(TASK_ID), rows)
        _append(root.log_file(TASK_ID), [log_row])
        got = [json.loads(line) for line in stream.wait_lines(4)[1:4]]
    stream.wait_end()
    transcript = [e for e in got if e["source"] == "transcript"]
    assert transcript == [
        {"source": "transcript", "offset": 0, "bytes": len(rows[0]), "line": usage},
        {"source": "transcript", "offset": len(rows[0]), "bytes": len(rows[1]), "omitted": "over_stream_bound"}]
    assert [e for e in got if e["source"] == "log"] == [
        {"source": "log", "offset": 0, "bytes": len(log_row), "omitted": "over_line_bound"}]
    # The page: the same offsets and lengths, the history's bound.
    status, page = history.page(root.transcript_file(TASK_ID), TASK_ID, "transcript")
    assert status == 200 and page["lines"] == [
        {"offset": 0, "bytes": len(rows[0]), "omitted": "over_line_bound"},
        {"offset": len(rows[0]), "bytes": len(rows[1]), "omitted": "over_line_bound"}]
