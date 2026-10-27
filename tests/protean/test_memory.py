"""The brain's long-term memory across thinking sessions on one project (the operator, 2026-10-22: "brain owns the
long-term memory, the thinking session owns the short-term memory"): a `rest` and a `close` each append one line to
`memory/<project>.jsonl`, and a later session's thinker message carries the earlier sessions' latest lines as `memory`,
never its own. A scratch session (`_root`) shares nothing."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from protean import cli, loop, state
from protean.dry import ScriptedRunner, make_workspace
from protean.graph import console, director, memory
from protean.graph.budget import Caps
from protean.state import Decision, OpenItem, Root, Task, save, set_current

FIELDS = ["session", "when", "event", "thought", "standing", "open", "decided"]


def _rest(standing: str, *lines: str) -> dict:
    move = {"rest": True, "state": {"standing": standing, "open": [{"id": f"o{n}", "line": line}
                                                                   for n, line in enumerate(lines, 1)]}}
    return {"type": "result", "is_error": False, "session_id": "s-1", "total_cost_usd": 0.1, "result": json.dumps(move)}


def _brain(root: Root) -> list[str]:
    root.path.mkdir(parents=True, exist_ok=True)
    root.thinker_prompt.write_text("t\n")
    root.editor_prompt.write_text("e\n")
    return ["--brain", str(root.path)]


def _messages(runner: ScriptedRunner) -> list[dict]:
    return [json.loads(stdin) for argv, stdin, _, _ in runner.calls if ScriptedRunner.seat(argv) == "director"]


def _lines(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def _events(root: Root, task_id: str) -> list[dict]:
    return [{k: v for k, v in e.items() if k != "ts"} for e in state.read_log(root, task_id) if e["event"] == "memory"]


@pytest.fixture
def ids(monkeypatch: pytest.MonkeyPatch):
    minted = iter(f"task-20261022T12{n:02d}00" for n in range(60))
    monkeypatch.setattr(state, "new_task_id", lambda now=None: next(minted))


def test_a_rest_on_a_project_writes_one_line_a_second_rest_appends_and_recall_keeps_the_latest(
    root: Root, tmp_path: Path, ids, monkeypatch: pytest.MonkeyPatch
):
    brain = _brain(root)
    workspace = make_workspace(tmp_path / "ws")
    runner = ScriptedRunner([_rest("A rests.", "which file"), _rest("A rests again.")], [])
    monkeypatch.setattr(cli, "real_runner", runner)
    assert cli.main([*brain, "think", "first", "--project", "demo", "--workspace", str(workspace)]) == 12
    task = state.current(root)
    path = root.path / "memory" / "demo.jsonl"
    (one,) = _lines(path)
    assert list(one) == FIELDS
    assert {k: v for k, v in one.items() if k != "when"} == {
        "session": task.id, "event": "rest", "thought": "first", "standing": "A rests.", "open": ["which file"],
        "decided": []}
    assert one["when"].endswith("+00:00") and len(one["when"]) == len("2026-10-22T12:00:00+00:00")
    assert _events(root, task.id) == [{"event": "memory", "project": "demo", "entry": "rest", "ok": True}]
    assert cli.main([*brain, "ctl", "think", "at", "the", "root", "--decide"]) == 0
    assert cli.main([*brain, "resume"]) == 12
    first, second = _lines(path)
    assert first == one and (second["session"], second["standing"], second["decided"]) == (
        task.id, "A rests again.", ["at the root"])
    assert memory.recall(root, "demo", "other") == [second]
    assert memory.recall(root, "demo", task.id) == []


def test_a_new_session_on_the_project_is_handed_the_earlier_one_never_its_own_and_scratch_shares_nothing(
    root: Root, tmp_path: Path, ids, monkeypatch: pytest.MonkeyPatch
):
    brain = _brain(root)
    workspace = make_workspace(tmp_path / "ws")
    runner = ScriptedRunner([_rest("A rests.", "which file"), _rest("A rests again.", "its name"),
                             _rest("B rests."), _rest("B rests again."), _rest("Scratch rests.")], [])
    monkeypatch.setattr(cli, "real_runner", runner)
    assert cli.main([*brain, "think", "first", "--project", "demo", "--workspace", str(workspace)]) == 12
    first = state.current(root)
    assert cli.main([*brain, "ctl", "think", "at", "the", "root", "--decide"]) == 0
    assert cli.main([*brain, "resume"]) == 12
    assert [("memory" in m) for m in _messages(runner)] == [False, False]  # nothing earlier, and never its own
    assert cli.main([*brain, "think", "second", "--project", "demo", "--workspace", str(workspace)]) == 12
    second = state.current(root)
    assert second.id != first.id
    (handed,) = _messages(runner)[2]["memory"]
    assert (handed["session"], handed["thought"], handed["standing"], handed["open"], handed["decided"]) == (
        first.id, "first", "A rests again.", ["its name"], ["at the root"])
    assert cli.main([*brain, "ctl", "think", "go", "on"]) == 0
    assert cli.main([*brain, "resume"]) == 12
    assert [entry["session"] for entry in _messages(runner)[3]["memory"]] == [first.id]  # its own rest is not in it
    assert cli.main([*brain, "think", "scratch"]) == 12
    scratch = state.current(root)
    assert scratch.project == "_root" and "memory" not in _messages(runner)[4]
    assert sorted(p.name for p in (root.path / "memory").iterdir()) == ["demo.jsonl"]
    assert _events(root, scratch.id) == []
    assert [e["session"] for e in _lines(root.memory_file("demo"))] == [first.id, first.id, second.id, second.id]


def _closable(root: Root, project: str = "demo") -> Task:
    root.path.mkdir(parents=True, exist_ok=True)
    task = Task(id="task-20261022T130000", thought="g", workspace="/w", project=project, terminal="interrupted",
                standing="It stands.", open=[OpenItem("o1", "still open")],
                decided=[Decision("keep it", "2026-10-22T13:00:00+00:00", 1)])
    save(root, task)
    set_current(root, task.id)
    return task


def test_an_offline_close_writes_a_close_entry(root: Root):
    task = _closable(root)
    assert loop.offline_op(root, Caps(), "close", {})["event"] == "close"
    (entry,) = _lines(root.memory_file("demo"))
    assert {k: v for k, v in entry.items() if k != "when"} == {
        "session": task.id, "event": "close", "thought": "g", "standing": "It stands.", "open": ["still open"],
        "decided": ["keep it"]}
    assert [e["event"] for e in state.read_log(root, task.id)][-2:] == ["close", "memory"]
    assert _events(root, task.id) == [{"event": "memory", "project": "demo", "entry": "close", "ok": True}]


def test_a_live_close_writes_a_close_entry_right_after_the_closed_record_is_saved(root: Root):
    task = _closable(root)
    with console.serve(root, task, Caps(), interrupt=lambda: None):
        status, payload = console.request(root, task.id, "POST", "/close", {})
    assert (status, payload["event"]["event"]) == (200, "close")
    (entry,) = _lines(root.memory_file("demo"))
    assert (entry["session"], entry["event"], entry["decided"]) == (task.id, "close", ["keep it"])
    assert [e["event"] for e in state.read_log(root, task.id)][-2:] == ["close", "memory"]
    assert _events(root, task.id) == [{"event": "memory", "project": "demo", "entry": "close", "ok": True}]


def test_a_failed_write_returns_none_and_the_hook_logs_it_not_ok(root: Root):
    task = _closable(root)
    (root.path / "memory").write_text("not a directory\n")  # the write's mkdir fails with an OSError
    assert memory.remember(root, task, "rest") is None
    assert memory.logged(root, task, "rest") == {"event": "memory", "project": "demo", "entry": "rest", "ok": False}
    assert memory.recall(root, "demo", "other") == []


def test_recall_hands_the_newest_five_sessions_oldest_first_and_skips_a_garbage_line(root: Root):
    sessions = [Task(id=f"task-20261022T14{n:02d}00", thought=f"t{n}", project="demo") for n in range(6)]
    for task in sessions[:3]:
        memory.remember(root, task, "rest")
    with root.memory_file("demo").open("a") as handle:
        handle.write("{not json\n[1, 2]\n\n")
    for task in sessions[3:]:
        memory.remember(root, task, "rest")
    assert memory.SESSIONS == 5
    assert [e["thought"] for e in memory.recall(root, "demo", "other")] == ["t1", "t2", "t3", "t4", "t5"]
    assert [e["thought"] for e in memory.recall(root, "demo", sessions[5].id)] == ["t0", "t1", "t2", "t3", "t4"]
    assert memory.recall(root, "_root", "other") == [] and memory.recall(root, "absent", "other") == []
    assert memory.remember(root, Task(id="s", thought="x"), "rest") is None  # `_root`: nothing written
    assert not root.memory_file("_root").exists()


def test_a_project_name_maps_to_one_safe_file_under_memory(root: Root):
    for project, name in [("a/b", "a_b.jsonl"), ("..", "_.jsonl"), ("x y", "x_y.jsonl"), ("demo", "demo.jsonl")]:
        path = root.memory_file(project)
        assert (path.name, path.parent) == (name, root.path / "memory"), project


def test_the_message_carries_memory_only_when_there_is_some():
    task = Task("t", "g", "/w", project="demo")
    entry = {"session": "s0", "standing": "It stood."}
    assert "memory" not in director.thinker_message(task, Caps(), [])
    assert "memory" not in director.thinker_message(task, Caps(), [], memory=[])
    assert director.thinker_message(task, Caps(), [], memory=[entry])["memory"] == [entry]


# --- a kill between the rest node and the memory node: the repair writes the line once ---


class Killed(BaseException):
    """A kill inside a tick: nothing after it runs, and the record stays as its last save left it."""


def _kill() -> None:
    raise Killed()


def _killed_rest(root: Root, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, project: str, where: str) -> Task:
    """A rest tick on `project` cut by a kill: `before` the memory node runs (right after the rest node's save), or
    inside it `after` its line is written and before its `memory` event. The record as the kill left it."""
    _brain(root)
    task = Task(id="task-20261026T150000", thought="first", workspace=str(make_workspace(tmp_path / "ws")),
                project=project)
    save(root, task)
    set_current(root, task.id)
    with monkeypatch.context() as patch:
        if where == "before":
            patch.setitem(loop.ACTS, "memory", lambda tick: _kill())
        else:
            real = memory.remember
            patch.setattr(memory, "remember", lambda *args: real(*args) and _kill())
        with pytest.raises(Killed):
            loop.run_tick(task, Caps(), root, ScriptedRunner([_rest("It rests.", "which file")], []))
    saved = state.load(root, task.id)
    assert ([m.kind for m in saved.moves], saved.awaiting_thought) == (["rest"], True)  # the rest node saved
    return saved


def _remembered(root: Root, task: Task) -> list[dict]:
    path = root.memory_file(task.project)
    return [{k: v for k, v in e.items() if k != "when"} for e in _lines(path)] if path.exists() else []


@pytest.mark.parametrize("where", ["before", "after"])
@pytest.mark.parametrize("then", ["the next tick", "resume"])
def test_a_rest_whose_memory_node_a_kill_cut_off_gets_its_line_once_from_the_repair(
    root: Root, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, where: str, then: str
):
    task = _killed_rest(root, tmp_path, monkeypatch, "demo", where)
    for _ in range(2):  # the repair writes it once; a second repair finds it written
        if then == "resume":
            loop.resume(root, Caps(), ScriptedRunner([], []), unattended=0.0, notify=None)
        else:
            loop.run_tick(state.load(root, task.id), Caps(), root, ScriptedRunner([], []))
        assert _remembered(root, task) == [{"session": task.id, "event": "rest", "thought": "first",
                                            "standing": "It rests.", "open": ["which file"], "decided": []}]
        assert _events(root, task.id) == [{"event": "memory", "project": "demo", "entry": "rest", "ok": True}]
    names = [e["event"] for e in state.read_log(root, task.id)]
    assert names.index("rest") < names.index("thought") < names.index("memory")


def test_a_scratch_session_s_cut_rest_owes_no_line(root: Root, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    task = _killed_rest(root, tmp_path, monkeypatch, memory.SCRATCH, "before")
    loop.run_tick(state.load(root, task.id), Caps(), root, ScriptedRunner([], []))
    loop.resume(root, Caps(), ScriptedRunner([], []), unattended=0.0, notify=None)
    assert not (root.path / "memory").exists() and _events(root, task.id) == []
