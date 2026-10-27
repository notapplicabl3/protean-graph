"""The brain's long-term memory: what each thinking session on a project left behind, handed to later sessions on it.

The operator, 2026-10-22: the brain owns the long-term memory, the thinking session the short-term (the thinker's own
CLI session, resumed within a warm stretch, `director.warm`). This is the brain's half, the smallest complete version:
one JSON line per remembered event in `memory/<project>.jsonl` under the brain (`Root.memory_file`), each a session's
thought, where it rested (`standing`), what stayed open and what the operator decided. A `rest` and a `close` write
one: a rest through the registry's memory node, walked right after the rest firing (`loop._remember`), a close
beside the save that already happens there (`loop.offline_op`, the live `close` in `console.Console.operate`);
`director.next_move` hands `recall` to the thinker as `memory`. A rest whose memory node a kill cut off gets its line
from the next tick's or the resume's repair, once (`owed`).

Deterministic and model-free, floating beside the ring like `graph/questions.py`: it never depends on
`graph/director.py`, and nothing in it reaches a worker's message or a reader's brief. A scratch session (project
`_root`) shares nothing. A failed write never stops the loop: `remember` returns None and the caller logs it.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from protean import state
from protean.state import Root, Task

SESSIONS = 5  # how many earlier sessions a thinker message carries
SCRATCH = "_root"  # the project of a session started without one: nothing remembered, nothing recalled


def remember(root: Root, task: Task, event: str) -> dict[str, Any] | None:
    """Append one line for `task` at a `rest` or a `close` and return it: the session, when (UTC, seconds), the event,
    the thought verbatim, the standing, the open lines and the operator's decisions. None, with nothing written, for a
    scratch session or when the write fails (an `OSError` never reaches the loop)."""
    if task.project == SCRATCH:
        return None
    entry = {"session": task.id, "when": datetime.now(timezone.utc).isoformat(timespec="seconds"), "event": event,
             "thought": task.thought, "standing": task.standing, "open": [item.line for item in task.open],
             "decided": [decision.text for decision in task.decided]}
    path = root.memory_file(task.project)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with state.LOCK, path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry) + "\n")
    except OSError:
        return None
    return entry


def logged(root: Root, task: Task, event: str) -> dict[str, Any] | None:
    """`remember`, as a hook calls it: the `memory` log event saying whether the line landed, or None for a scratch
    session (nothing was attempted, so nothing is logged)."""
    if task.project == SCRATCH:
        return None
    ok = remember(root, task, event) is not None
    return {"event": "memory", "project": task.project, "entry": event, "ok": ok}


def owed(root: Root, task: Task) -> dict[str, Any] | None:
    """The repair's net under the memory node (`loop._repair`): the `memory` event of a rest whose memory node a kill
    cut off, its line written once. Owed only when the session's last move is a rest, on a project, and no `memory`
    event of a rest follows that rest's `rest` event in the log. A line the node wrote before the kill cut off its
    event (the session has more rest lines in the file than `memory` events of a rest that say one landed) is not
    written again: only its event is returned. None when nothing is owed, when the log cannot be read, or when the
    rest's own `rest` event is not in it (no event to follow, so a later repair could not tell a line written)."""
    last = task.moves[-1] if task.moves else None
    if task.project == SCRATCH or last is None or last.kind != "rest":
        return None
    try:
        log = state.read_log(root, task.id)
    except (OSError, ValueError):
        return None
    rests = [i for i, e in enumerate(log) if e.get("event") == "memory" and e.get("entry") == "rest"]
    at = next((i for i in range(len(log) - 1, -1, -1)
               if log[i].get("event") == "rest" and log[i].get("move") == last.id), None)
    if at is None or any(i > at for i in rests):
        return None
    if _rest_lines(root, task) > sum(1 for i in rests if log[i].get("ok")):
        return {"event": "memory", "project": task.project, "entry": "rest", "ok": True}
    return logged(root, task, "rest")


def _rest_lines(root: Root, task: Task) -> int:
    """How many rest lines the session has in its project's memory file; 0 when the file cannot be read."""
    try:
        lines = root.memory_file(task.project).read_text(encoding="utf-8").splitlines()
    except OSError:
        return 0
    count = 0
    for line in lines:
        try:
            entry = json.loads(line)
        except ValueError:
            continue
        count += isinstance(entry, dict) and entry.get("session") == task.id and entry.get("event") == "rest"
    return count


def recall(root: Root, project: str, exclude_session: str, limit: int = SESSIONS) -> list[dict[str, Any]]:
    """The earlier sessions on `project` as the thinker is handed them: each session's latest line, never
    `exclude_session`'s (the asking session's own), the newest `limit` sessions by where that line sits in the file,
    oldest first. Empty for a scratch session, a missing or unreadable file, or nothing left; a line that does not
    parse as an entry is skipped."""
    if project == SCRATCH or limit <= 0:
        return []
    try:
        lines = root.memory_file(project).read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    latest: dict[str, dict[str, Any]] = {}
    for line in lines:
        try:
            entry = json.loads(line)
        except ValueError:
            continue
        session = entry.get("session") if isinstance(entry, dict) else None
        if not isinstance(session, str) or session == exclude_session:
            continue
        latest.pop(session, None)  # re-inserted, so dict order is the order of each session's latest line
        latest[session] = entry
    return list(latest.values())[-limit:]
