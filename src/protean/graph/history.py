"""Bounded history: one page of a session's `log.jsonl` or `transcript.jsonl`, the console's `GET /history` and the
lobby's `GET /history?task=<id>`, and the line identity the offsets stream (`GET /stream?offsets=1`) shares with it.

A line is identified by its source and its byte offset: `offset` and `bytes` count raw encoded bytes, the newline
included, so entries are contiguous. Only complete lines are served: `size` is the end of the last newline at read
time. A row that is a JSON object is served whole as `line` (every field kept); any other row is `raw` text; a line
over `LINE_BOUND` bytes is an `omitted` entry with its true length, which still advances the page. A page holds at most
`limit` entries, and its whole response body, as `json.dumps` encodes it (escapes and the page's own keys counted), is
at most `PAGE_BOUND` bytes: each entry is charged its encoded size, the page ends before the first entry that would not
fit, and an entry that would not fit even alone is its `over_payload_bound` marker, with its offset and length, so a
page always moves on. Every read is a positional read of at most `CHUNK` bytes, so no request holds more than a page,
one chunk and one bounded line in memory, whatever the file's size. The file is opened only as `Root` names it, never
through a link, and nothing is written. The wire is `docs/console-api.md` § History.

The offsets stream gives each line the same `offset` and `bytes`, but its content bound is the stream's, not the page's
(client review ruling 2): the log keeps `LINE_BOUND`; the transcript keeps `STREAM_LINE_BOUND`, so a live line's usage
reaches the client whole while a page still omits it. An envelope whose encoding would pass `STREAM_WIRE_BOUND` (the
client's stream line limit, less one read) is its `over_stream_bound` marker instead.

`logged` reads the log the same way, a chunk at a time, for the console's repair of a request's missing log event.
"""

from __future__ import annotations

import json
import os
import re
import stat
from pathlib import Path
from typing import Any
from urllib.parse import unquote_plus

from protean.state import Root

SOURCES = ("log", "transcript")
KEYS = ("source", "before", "after", "limit")  # the console's query keys; the lobby adds `task`
LIMIT_DEFAULT, LIMIT_MAX = 200, 500
PAGE_BOUND = 262_144  # bytes of one page's whole response body, as JSON-encoded
LINE_BOUND = 65_536  # a line longer than this, its newline aside, is an `omitted` entry
CHUNK = 65_536  # the most one read takes
OMITTED = "over_line_bound"
OVER_PAYLOAD = "over_payload_bound"  # a line whose entry alone would put its page over `PAGE_BOUND`
# The offsets stream's transcript bound: the client reads a stream line of at most 16 MiB (`StreamConnection`), so a
# live transcript line is sent whole up to it; its envelope, as encoded with its newline, is held to that limit less
# one 64 KiB read, which the client may hold of the next line.
STREAM_LINE_BOUND = 16 * 1024 * 1024
STREAM_WIRE_BOUND = 16 * 1024 * 1024 - 65_536
OVER_STREAM = "over_stream_bound"  # a stream line over `STREAM_LINE_BOUND`, or whose envelope passes `STREAM_WIRE_BOUND`
_WHOLE = re.compile(r"[0-9]+")  # ASCII digits only: a sign, a space, a fraction or another script's digit is no cursor


class Refused(ValueError):
    """A history request refused; `status` is the HTTP status it is answered with."""

    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status = status


class _Shrunk(Exception):
    """The file was shorter than its own size said while it was read: it is append-only, so it was cut."""


def path_of(root: Root, task_id: str, source: str) -> Path:
    """The one file a source names: `state/<task>/log.jsonl` or `state/<task>/transcript.jsonl`."""
    return root.log_file(task_id) if source == "log" else root.transcript_file(task_id)


def params(query: str, keys: tuple[str, ...]) -> dict[str, str]:
    """The query's values by key: a key outside `keys`, a repeated key or a part without `=` is a 400."""
    found: dict[str, str] = {}
    for part in query.split("&"):
        if not part:
            continue
        if "=" not in part:
            raise Refused(400, f"/history: {unquote_plus(part)!r} has no value")
        key, value = (unquote_plus(p) for p in part.split("=", 1))
        if key not in keys:
            raise Refused(400, f"/history takes no {key!r} (it takes {', '.join(keys)})")
        if key in found:
            raise Refused(400, f"/history takes one {key}")
        found[key] = value
    return found


def _whole(name: str, value: str) -> int:
    if not _WHOLE.fullmatch(value):
        raise Refused(400, f"{name} must be a whole number of bytes, 0 or more")
    try:
        return int(value)
    except ValueError:  # more digits than Python converts: no file is that long, and no client sends it
        raise Refused(400, f"{name} has too many digits") from None


def answer(root: Root, task_id: str, found: dict[str, str]) -> tuple[int, dict[str, Any]]:
    """`GET /history` for one session, its query already read (`params`): checked, then one page (`page`)."""
    source = found.get("source")
    if source not in SOURCES:
        raise Refused(400, f"source must be one of {', '.join(SOURCES)}")
    if "before" in found and "after" in found:
        raise Refused(400, "/history takes before or after, not both")
    before = _whole("before", found["before"]) if "before" in found else None
    after = _whole("after", found["after"]) if "after" in found else None
    limit = LIMIT_DEFAULT
    if "limit" in found:
        digits = found["limit"].lstrip("0")
        if not _WHOLE.fullmatch(found["limit"]) or not digits or len(digits) > 3 or int(digits) > LIMIT_MAX:
            raise Refused(400, f"limit must be a whole number from 1 to {LIMIT_MAX}")
        limit = int(digits)
    return page(path_of(root, task_id, source), task_id, source, before=before, after=after, limit=limit)


def _read_at(fd: int, count: int, offset: int) -> bytes:
    """One positional read; the battery watches it to prove no request reads more than a chunk at a time."""
    return os.pread(fd, count, offset)


def _exact(fd: int, count: int, offset: int) -> bytes:
    data = _read_at(fd, count, offset)
    if len(data) != count:
        raise _Shrunk
    return data


def _complete(fd: int, end: int) -> int:
    """The end of the last complete line at or before `end`: just past its newline; 0 when there is none."""
    pos = end
    if pos and _exact(fd, 1, pos - 1) == b"\n":
        return pos
    while pos > 0:
        n = min(CHUNK, pos)
        data = _exact(fd, n, pos - n)
        k = data.rfind(b"\n")
        if k >= 0:
            return pos - n + k + 1
        pos -= n
    return 0


def complete_size(path: Path) -> int:
    """The end of the last complete line of `path` now (the snapshot's `log_bytes` and `transcript_bytes`, where the
    stream continues from); 0 when the file is absent. A partial last line is not counted. Anything but a regular file
    there (which the brain never writes) keeps the size `stat` gives, as before, and is never read."""
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK)
    except FileNotFoundError:
        return 0
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode):
            return info.st_size
        while True:
            try:
                return _complete(fd, os.fstat(fd).st_size)
            except _Shrunk:  # cut while it was read: read its new size again
                continue
    finally:
        os.close(fd)


def entry(offset: int, length: int, content: bytes | None) -> dict[str, Any]:
    """One line as history and the offsets stream serve it: its JSON object as `line`, any other row as `raw` text
    (bytes that are not UTF-8 decoded with replacement), or, for `content` None (over `LINE_BOUND`), the omission
    marker; with its `offset` and `bytes`, the newline counted. A page serves an entry too large for any page as its
    `over_payload_bound` marker instead (`_charged`); the offsets stream sends every entry whole."""
    if content is None:
        return {"offset": offset, "bytes": length, "omitted": OMITTED}
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError:
        return {"offset": offset, "bytes": length, "raw": content.decode("utf-8", "replace")}
    try:
        value = json.loads(text)
    except (ValueError, RecursionError):
        value = None
    if isinstance(value, dict):
        return {"offset": offset, "bytes": length, "line": value}
    return {"offset": offset, "bytes": length, "raw": text}


def stream_bound(source: str) -> int:
    """The content bound of `source` on the offsets stream: `STREAM_LINE_BOUND` for the transcript, `LINE_BOUND` (the
    page's) for the log."""
    return STREAM_LINE_BOUND if source == "transcript" else LINE_BOUND


def envelope(source: str, offset: int, length: int, content: bytes | None, wire_bound: int | None = None) -> bytes:
    """One line of the offsets stream: `source`, then the line's history entry, as one JSON line. `content` None (over
    its cutter's bound, `stream_bound`) is the omission marker: `over_stream_bound` for the transcript, `over_line_bound`
    for the log. An envelope longer than `wire_bound` (default `STREAM_WIRE_BOUND`), its newline counted, is its
    `over_stream_bound` marker with the same `offset` and `bytes`: escapes can make a line under its source bound too long
    for the client to read."""
    if content is None:
        item = {"offset": offset, "bytes": length, "omitted": OVER_STREAM if source == "transcript" else OMITTED}
    else:
        item = entry(offset, length, content)
    data = json.dumps({"source": source, **item}).encode("utf-8") + b"\n"
    if len(data) > (STREAM_WIRE_BOUND if wire_bound is None else wire_bound):
        data = json.dumps({"source": source, "offset": offset, "bytes": length, "omitted": OVER_STREAM}).encode() + b"\n"
    return data


class Lines:
    """Cuts bytes read forward from `offset` into complete lines, each `(offset, bytes, content)`: its newline counted
    in `bytes`, `content` without it, None once the line is over `bound` (default `LINE_BOUND`; its bytes are dropped as
    they arrive, so at most `bound` bytes of the line in progress are held)."""

    def __init__(self, offset: int, bound: int | None = None) -> None:
        self.offset, self.length, self.held = offset, 0, bytearray()
        self.bound = LINE_BOUND if bound is None else bound

    def feed(self, data: bytes) -> list[tuple[int, int, bytes | None]]:
        out: list[tuple[int, int, bytes | None]] = []
        i = 0
        while True:
            j = data.find(b"\n", i)
            part = data[i:] if j < 0 else data[i:j]
            self.length += len(part)
            if self.length <= self.bound:
                self.held += part
            elif self.held:
                self.held = bytearray()
            if j < 0:
                return out
            length = self.length + 1
            out.append((self.offset, length, bytes(self.held) if self.length <= self.bound else None))
            self.offset, self.length, self.held = self.offset + length, 0, bytearray()
            i = j + 1


def _room(task_id: str, source: str, size: int) -> int:
    """What a page's entries may take of `PAGE_BOUND`: the bound less the page's own keys at their widest (`from` and
    `to` as many digits as `size`, `older` and `newer` false, `lines` empty)."""
    shell = {"ok": True, "task": task_id, "source": source, "size": size, "from": size, "to": size, "older": False,
             "newer": False, "lines": []}
    return PAGE_BOUND - len(json.dumps(shell))


def _charged(item: dict[str, Any], room: int) -> tuple[dict[str, Any], int]:
    """An entry as a page serves it and what it adds to the page's body: its `json.dumps` length (ASCII, since every
    other character is escaped, so characters are bytes) and the `, ` that may precede it. An entry that would not fit
    `room` even alone is its `over_payload_bound` marker, its offset and length kept, so its span is never dropped
    silently and the page still moves on. Each entry is encoded once."""
    cost = len(json.dumps(item)) + 2
    if cost > room:
        item = {"offset": item["offset"], "bytes": item["bytes"], "omitted": OVER_PAYLOAD}
        cost = len(json.dumps(item)) + 2
    return item, cost


def _forward(fd: int, start: int, size: int, limit: int, room: int) -> list[dict[str, Any]]:
    """The page that starts at `start`: lines forward until `limit` entries, `size`, or the first entry that would take
    the page past `room` (that entry is the next page's first, so the page is trimmed at its newer end)."""
    cut, entries, used, pos = Lines(start), [], 0, start
    while pos < size and len(entries) < limit:
        n = min(CHUNK, size - pos)
        for offset, length, content in cut.feed(_exact(fd, n, pos)):
            if len(entries) >= limit:
                return entries
            item, cost = _charged(entry(offset, length, content), room)
            if entries and used + cost > room:
                return entries
            entries.append(item)
            used += cost
        pos += n
    return entries


def _line_start(fd: int, lo: int) -> tuple[int, bytes, int]:
    """Where the line holding the byte before `lo` starts, found without keeping it: `(start, the bytes [low, start),
    low)`, `low` where the read that found the start began."""
    while lo > 0:
        n = min(CHUNK, lo)
        data = _exact(fd, n, lo - n)
        k = data.rfind(b"\n")
        if k >= 0:
            return lo - n + k + 1, data[:k + 1], lo - n
        lo -= n
    return 0, b"", 0


def _backward(fd: int, end: int, limit: int, room: int) -> list[dict[str, Any]]:
    """The page that ends at `end`: lines backward until `limit` entries, the file's start, or the first entry that
    would take the page past `room` (trimmed at its older end), returned oldest first. `buf[:hi]` holds the bytes
    `[lo, line_end)`, the last of them the line's newline, and `buf[clean:hi - 1]` is known to hold no other; a line
    found over the bound is skipped to its start without keeping it."""
    entries: list[dict[str, Any]] = []
    used, lo, buf, hi, clean, line_end = 0, end, b"", 0, 0, end
    while line_end > 0 and len(entries) < limit:
        k = buf.rfind(b"\n", 0, clean) if hi else -1
        if k < 0 and lo > 0 and hi - 1 <= LINE_BOUND:
            n = min(CHUNK, lo)
            buf, lo = _exact(fd, n, lo - n) + buf[:hi], lo - n
            hi = len(buf)
            clean = min(n, hi - 1)  # what was held had no newline but its last byte: only the new bytes are searched
            continue
        content: bytes | None
        if k >= 0:
            start, content, hi = lo + k + 1, buf[k + 1:hi - 1], k + 1
        elif lo == 0:
            start, content, hi = 0, buf[:hi - 1], 0
        else:
            (start, buf, lo), content = _line_start(fd, lo), None
            hi = len(buf)
        clean = hi - 1
        if content is not None and len(content) > LINE_BOUND:
            content = None
        item, cost = _charged(entry(start, line_end - start, content), room)
        if entries and used + cost > room:
            break
        entries.append(item)
        used, line_end = used + cost, start
    entries.reverse()
    return entries


def page(path: Path, task_id: str, source: str, *, before: int | None = None, after: int | None = None,
         limit: int = LIMIT_DEFAULT) -> tuple[int, dict[str, Any]]:
    """One page of `path` (`source` of session `task_id`): with no cursor the newest, ending at `size`; `after=A` read
    forward from A; `before=B` the page that ends at B. A cursor past `size` is a 409; one inside a line a 400. An
    absent file is `size` 0 and no lines; a link or anything but a regular file there is a 404."""
    try:
        fd: int | None = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    except (FileNotFoundError, NotADirectoryError):
        fd = None
    except OSError as exc:  # a link where the file should be (ELOOP), or no permission
        return 404, {"ok": False, "error": f"the {source} of {task_id} cannot be read: {exc.strerror or exc}"}
    try:
        if fd is not None and not stat.S_ISREG(os.fstat(fd).st_mode):
            return 404, {"ok": False, "error": f"the {source} of {task_id} is not a regular file"}
        size = 0 if fd is None else _complete(fd, os.fstat(fd).st_size)
        cursor = after if after is not None else before
        if cursor is not None:
            if cursor > size:
                return 409, {"ok": False, "error": f"offset {cursor} is past the end of {source} ({size} bytes)"}
            if cursor and _exact(fd, 1, cursor - 1) != b"\n":
                return 400, {"ok": False, "error": f"offset {cursor} is not at the start of a line of {source}"}
        room = _room(task_id, source, size)
        if after is not None:
            entries = _forward(fd, after, size, limit, room) if fd is not None else []
            start = after
            stop = start + sum(e["bytes"] for e in entries)
        else:
            stop = size if before is None else before
            entries = _backward(fd, stop, limit, room) if fd is not None else []
            start = stop - sum(e["bytes"] for e in entries)
    except _Shrunk:
        return 409, {"ok": False, "error": f"the {source} of {task_id} was cut while it was read"}
    finally:
        if fd is not None:
            os.close(fd)
    return 200, {"ok": True, "task": task_id, "source": source, "size": size, "from": start, "to": stop,
                 "older": start > 0, "newer": stop < size, "lines": entries}


def logged(path: Path, ids: dict[str, str]) -> set[str]:
    """Which of `ids` (a `request_id` to its operation) the log at `path` holds an event for: a complete line that is
    a JSON object whose `request_id` is that id and whose `event` is that operation. Identity only, never text. Read
    forward a chunk at a time (`Lines`), so it holds one chunk and one bounded line whatever the log's size; only a
    line that names `"request_id"` is decoded, and the read stops once every id is found. An absent log holds none."""
    found: set[str] = set()
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK)
    except FileNotFoundError:
        return found
    try:
        cut, pos = Lines(0), 0
        while len(found) < len(ids) and (data := _read_at(fd, CHUNK, pos)):
            pos += len(data)
            for _, _, content in cut.feed(data):
                if content is None or b'"request_id"' not in content:
                    continue
                try:
                    value = json.loads(content)
                except (ValueError, RecursionError):
                    continue
                rid = value.get("request_id") if isinstance(value, dict) else None
                if isinstance(rid, str) and rid in ids and value.get("event") == ids[rid]:
                    found.add(rid)
    finally:
        os.close(fd)
    return found
