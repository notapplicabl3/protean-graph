"""The record: the thought record's fields in `task.json`, the goal-mode names as load aliases, the
six live records of runs 1–6 loading unchanged, all nine evidence records (sessions 7–9 too) loading with the shape's
fields at their defaults, and `thought.md` rendered from the record on every save."""

from __future__ import annotations

import copy
import json
from dataclasses import asdict
from pathlib import Path

import pytest

from protean import loop, state
from protean.dry import FIXTURES, REPO, ScriptedRunner, make_workspace, read_jsonl
from protean.graph import console, graph
from protean.graph.budget import Caps
from protean.state import (
    Decision, Direction, Idea, Move, OpenItem, Root, Task, Thought, from_json, load, render_thought, save, to_json,
)

LIVE = sorted((REPO / "evidence").glob("live-*/task.json"))


def _session() -> Task:
    """A session with every field of the thought record set."""
    return Task(
        "task-20261017T090000", "Can the README say what PROTEAN is in one paragraph?\nKeep it short.", "/w",
        tick=3, cost_usd=1.5,
        thoughts=[Thought("look at the old README first", "2026-10-17T09:01:00+00:00"),
                  Thought("go with the second idea", "2026-10-17T09:05:00+00:00")],
        standing="The README has no opening paragraph. Two ways to write one are open.",
        open=[OpenItem("o1", "which audience the paragraph is for"), OpenItem("o2", "whether to keep the diagram")],
        decided=[Decision("write it for a newcomer\nand drop the diagram", "2026-10-17T09:05:00+00:00", 2)],
        moves=[
            Move("m1", "ideas", "three ways into the paragraph", 0.12, "picked i2",
                 ideas=[Idea("i1", "lead with the loop", "one explore", "o1"),
                        Idea("i2", "lead with the problem", "one explore", "o1"),
                        Idea("i3", "lead with a run", "a task", "o2")]),
            Move("m2", "explore", "draft the problem-first opening", 0.4, "landed",
                 branch="explore/task-20261017T090000/1", report="Drafted two openings.\nThe second reads better."),
            Move("m3", "task", "write the opening", 0.9, "integrated", branch="task/task-20261017T090000",
                 units=["u-1", "u-2"]),
            Move("m4", "ask", "which tone", 0.05, "answered", question="q1"),
            Move("m5", "rest", "", 0.03, "parked"),
        ],
        awaiting_thought=True, active_seconds=420.5,
    )


def test_the_thought_record_round_trips_through_json_and_disk(root: Root):
    task = _session()
    again = from_json(to_json(task))
    assert again == task
    assert isinstance(again.thoughts[0], Thought) and isinstance(again.open[0], OpenItem)
    assert isinstance(again.decided[0], Decision) and isinstance(again.moves[0], Move)
    assert isinstance(again.moves[0].ideas[0], Idea) and again.moves[1].report == task.moves[1].report
    save(root, task)
    assert load(root, task.id) == task


def test_goal_and_directions_are_the_old_names_of_thought_and_thoughts_and_are_never_written():
    """The goal-mode names stay readable and writable on the dataclass (the loop and the console still use them until
    the loop switches), alias the same values, and the saved record carries only the new names, as `manager_session`
    became `director_session`."""
    task = Task(id="t", goal="make a.txt", workspace="/w", directions=[Direction("use tabs")])
    assert (task.thought, task.goal) == ("make a.txt", "make a.txt") and Direction is Thought
    assert task.directions is task.thoughts and [t.text for t in task.thoughts] == ["use tabs"]
    task.directions.append(Direction("keep it short", "2026-10-17T00:00:00+00:00"))
    task.goal = "make b.txt"
    assert task.thought == "make b.txt" and [t.text for t in task.thoughts] == ["use tabs", "keep it short"]
    assert Task("t", "make a.txt", "/w") == Task(id="t", goal="make a.txt", workspace="/w")
    saved = json.loads(to_json(task))
    assert saved["thought"] == "make b.txt" and [t["text"] for t in saved["thoughts"]] == ["use tabs", "keep it short"]
    assert "goal" not in saved and "directions" not in saved
    assert list(saved)[:3] == ["id", "thought", "workspace"]


def test_a_record_from_before_the_thought_record_loads_with_the_defaults():
    task = _session()
    task.cost_usd = 2.75
    older = json.loads(to_json(task))
    older["goal"], older["directions"] = older.pop("thought"), older.pop("thoughts")
    for key in ("standing", "open", "decided", "moves", "awaiting_thought", "active_seconds"):
        del older[key]
    loaded = from_json(json.dumps(older))
    assert loaded.thought == task.thought and loaded.thoughts == task.thoughts
    assert (loaded.standing, loaded.open, loaded.decided, loaded.moves, loaded.awaiting_thought) == ("", [], [], [], False)
    assert loaded.active_seconds == 0.0
    assert "goal" not in json.loads(to_json(loaded)) and "directions" not in json.loads(to_json(loaded))


def test_six_live_records_are_found():
    if not LIVE:
        pytest.skip("the six goal-mode receipts are not in this tree (withheld from the public mirror)")
    assert [path.parent.name.split("-")[1] for path in LIVE] == ["1", "2", "3", "4", "5", "6"]


@pytest.mark.parametrize("source", LIVE, ids=[path.parent.name for path in LIVE])
def test_each_live_record_of_runs_1_to_6_loads_unchanged_with_the_new_fields_at_their_defaults(source: Path, root: Root):
    """`thought` is its goal, `thoughts` its directions (text, created, actor), every new field at its
    default, `thought.md` renders, and so does its console snapshot, the record's thought
    and thoughts in it. The evidence file is read, never written: its bytes are the same after."""
    before = source.read_bytes()
    raw = json.loads(before)
    task = from_json(before.decode("utf-8"))
    assert task.thought == raw["goal"] and task.goal == raw["goal"]
    assert [asdict(t) for t in task.thoughts] == [{**d, "request_id": None} for d in raw.get("directions", [])]
    assert all(isinstance(t, Thought) for t in task.thoughts)
    assert (task.standing, task.open, task.decided, task.moves, task.awaiting_thought) == ("", [], [], [], False)
    assert task.active_seconds == 0.0
    view = render_thought(task)
    assert view.startswith("# Thought\n\n" + raw["goal"] + "\n")
    snap = json.loads(json.dumps(console.snapshot(task, Caps(), live=False)))  # renders, and is plain JSON
    assert (snap["thought"]["thought"], snap["thoughts"], snap["moves"]) == (raw["goal"], raw.get("directions", []), [])
    assert "goal" not in snap and len(snap["units"]) == len(raw["units"])
    save(root, task)
    assert root.thought_file(task.id).read_text(encoding="utf-8") == render_thought(load(root, task.id)) == view
    assert source.read_bytes() == before


EVIDENCE = sorted((REPO / "evidence").glob("*/task.json"))  # runs 1–6 (`live-*`) and sessions 7–9 (`think-*`)


def test_nine_evidence_records_are_found():
    if not EVIDENCE:
        pytest.skip("the receipts are not in this tree (withheld from the public mirror)")
    assert [path.parent.name.rsplit("-", 3)[0] for path in EVIDENCE] == [
        "live-1", "live-2", "live-3", "live-4", "live-5", "live-6", "think-1", "think-2", "think-3"]


@pytest.mark.parametrize("source", EVIDENCE, ids=[path.parent.name for path in EVIDENCE])
def test_each_evidence_record_loads_unchanged_with_the_shape_s_fields_at_their_defaults(source: Path):
    """Shape SPEC row 5: a record from before the registry has no `fired` and no move carries `fire`; it loads with
    `fired` None and each move's firing derived from its kind, and its snapshot renders with `fired` null. Read, never
    written: its bytes are the same after."""
    before = source.read_bytes()
    raw = json.loads(before)
    assert "fired" not in raw and all("fire" not in move for move in raw.get("moves", []))
    task = from_json(before.decode("utf-8"))
    assert task.fired is None
    assert [m.fire for m in task.moves] == [list(graph.FIRES[m["kind"]]) for m in raw.get("moves", [])]
    snap = json.loads(json.dumps(console.snapshot(task, Caps(), live=False)))
    assert snap["fired"] is None and len(snap["moves"]) == len(raw.get("moves", []))
    assert source.read_bytes() == before


def test_the_render_is_the_thought_verbatim_then_standing_open_decided_and_one_line_per_move():
    task = _session()
    untouched = copy.deepcopy(task)
    view = render_thought(task)
    assert view == render_thought(task) and task == untouched  # pure: the same text every time, the record unchanged
    assert view == (
        "# Thought\n\nCan the README say what PROTEAN is in one paragraph?\nKeep it short.\n\n"
        "## Standing\n\nThe README has no opening paragraph. Two ways to write one are open.\n\n"
        "## Open\n\n- o1: which audience the paragraph is for\n- o2: whether to keep the diagram\n\n"
        "## Decided\n\n- think 2, 2026-10-17T09:05:00+00:00: write it for a newcomer\n  and drop the diagram\n\n"
        "## Moves\n\n"
        "- m1 · ideas · three ways into the paragraph · $0.12 · picked i2 · ideas i1, i2, i3\n"
        "- m2 · explore · draft the problem-first opening · $0.40 · landed · branch explore/task-20261017T090000/1"
        " · report (unverified): Drafted two openings.\n"
        "- m3 · task · write the opening · $0.90 · integrated · branch task/task-20261017T090000 · units u-1, u-2\n"
        "- m4 · ask · which tone · $0.05 · answered · question q1\n"
        "- m5 · rest · $0.03 · parked\n"
    )


def test_an_empty_record_renders_every_section():
    assert render_thought(Task("t", "a thought", "/w")) == (
        "# Thought\n\na thought\n\n## Standing\n\n(none)\n\n## Open\n\n(none)\n\n## Decided\n\n(none)\n\n"
        "## Moves\n\n(none)\n")


def test_a_move_s_text_never_breaks_its_one_line():
    task = Task("t", "a thought", "/w", moves=[Move("m1", "ask", "which\nfile?", 0.1, "open\nstill", question="q1")])
    (line,) = [line for line in render_thought(task).splitlines() if line.startswith("- m1")]
    assert line == "- m1 · ask · which file? · $0.10 · open still · question q1"


def test_thought_md_equals_the_render_of_the_saved_record_after_every_save(tmp_path: Path, root: Root,
                                                                         monkeypatch: pytest.MonkeyPatch):
    """every save of a whole dry run (the `dry` scenario, through the loop) leaves `thought.md` equal to the
    render of the `task.json` that save wrote, checked at each write. The view is written first and the record's
    rename commits the save, so a failed view write never leaves the record saved."""
    root.path.mkdir(parents=True)
    root.thinker_prompt.write_text("m\n")
    root.editor_prompt.write_text("e\n")
    workspace = make_workspace(tmp_path / "ws")
    real_write, written, checked = state._write_atomic, [], []

    def write(path: Path, text: str) -> None:
        real_write(path, text)
        written.append(path.name)
        if path.name == "task.json":
            record = from_json(text)
            assert path.with_name("thought.md").read_text(encoding="utf-8") == render_thought(record)
            checked.append(record.tick)

    monkeypatch.setattr(state, "_write_atomic", write)
    runner = ScriptedRunner(read_jsonl(FIXTURES / "dry/director.jsonl"), read_jsonl(FIXTURES / "dry/worker.jsonl"))
    task = loop.run("dry", workspace, root, Caps(), runner)
    saves = [name for name in written if name in ("task.json", "thought.md")]
    assert (task.terminal, task.awaiting_thought) == ("interrupted", True) and len(checked) > 4
    assert checked[-1] == task.tick and [m.kind for m in task.moves] == ["task", "rest"]
    assert saves == ["thought.md", "task.json"] * len(checked)  # each save: the view, then the record (its commit)
    assert root.thought_file(task.id).read_text(encoding="utf-8") == render_thought(load(root, task.id))
    assert root.thought_file(task.id) == root.state_dir(task.id) / "thought.md"
    assert not list(root.state_dir(task.id).glob("*.tmp"))
