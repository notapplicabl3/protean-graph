"""State: the task round-trips through JSON and disk, the log appends, the unit helpers."""

from __future__ import annotations

import json
import shutil
from datetime import datetime

from protean.dry import REPO
from protean.state import (
    Direction, Held, Predicate, Question, Root, Task, Unit, add_question, append_log, block_dependents,
    current, find_question, from_json, load, new_task_id, open_questions, pending_unit, save, set_current, to_json,
)


def test_task_round_trips_through_json_and_disk(root: Root):
    units = [
        Unit("u1", "write a", [Predicate("file_exists", {"path": "a"})], "passed", 1, "unit/u1", "ok"),
        Unit("u2", "run it", [Predicate("exit_code", {"code": 0}), Predicate("command", {"cmd": "true"})]),
    ]
    task = Task("task-20260927T120000", "g", "/w", "p", 4, units, 1.25, "interrupted", "", "sess-1", "c",
                questions=[Question("q1", "director", "which?", "question", context={"x": [1]})],
                directions=[Direction("use tabs", "2026-10-10T00:00:00+00:00")], paused=True)
    again = from_json(to_json(task))
    assert again == task and isinstance(again.units[1].predicates[0], Predicate)
    assert isinstance(again.questions[0], Question) and isinstance(again.directions[0], Direction)
    save(root, task)
    assert load(root, task.id) == task
    task.tick = 5
    save(root, task)
    assert load(root, task.id).tick == 5
    assert not list(root.state_dir(task.id).glob("*.tmp"))


def test_a_unit_s_summary_for_the_operator_round_trips_and_a_record_without_it_loads_empty(live_receipt):
    """`plain_summary` rides in `task.json` on the task's units and a held plan's; every record saved
    before it loads with none."""
    a = Unit("u1", "write a", [Predicate("file_exists", {"path": "a"})], plain_summary="Adds a.")
    b = Unit("u2", "write b", [Predicate("file_exists", {"path": "b"})], plain_summary="Adds b.")
    task = Task("t", "g", "/w", units=[a], held=Held(kind="plan", units=[b]))
    assert from_json(to_json(task)) == task
    older = json.loads(to_json(task))
    for unit in (older["units"][0], older["held"]["units"][0]):
        del unit["plain_summary"]
    loaded = from_json(json.dumps(older))
    assert (loaded.units[0].plain_summary, loaded.held.units[0].plain_summary) == ("", "")
    record = from_json(live_receipt("live-6-2026-10-13").read_text())  # saved before the field
    assert record.units and all(u.plain_summary == "" for u in record.units)


def test_legacy_session_loads_and_saves_as_director():
    data = json.loads(to_json(Task("t", "g", "/w")))
    del data["director_session"]
    data["manager_session"] = "existing-session"
    task = from_json(json.dumps(data))
    assert task.director_session == "existing-session"
    saved = json.loads(to_json(task))
    assert saved["director_session"] == "existing-session" and "manager_session" not in saved


def test_new_task_id_format():
    assert new_task_id(datetime(2026, 9, 27, 8, 5, 3)) == "task-20260927T080503"
    assert len(new_task_id()) == len("task-YYYYMMDDTHHMMSS")


def test_root_layout(root: Root):
    p = root.path
    assert [root.state_dir("t"), root.task_file("t"), root.log_file("t"), root.seat_dir("t"), root.clones_dir("t")] == [
        p / "state/t", p / "state/t/task.json", p / "state/t/log.jsonl", p / "state/t/seat", p / "state/t/clones"
    ]
    assert [root.current_file, root.console_file("t")] == [p / "state/current", p / "state/t/console.json"]
    assert [root.config_file, root.thinker_prompt, root.editor_prompt] == [
        p / "config.json", p / "thinker.md", p / "editor.md"
    ]
    assert not hasattr(root, "director_prompt")  # director.md retired with goal mode


def test_current_is_none_on_an_empty_root_then_follows_set_current(root: Root, task: Task):
    assert current(root) is None
    save(root, task)
    set_current(root, task.id)
    assert root.current_file.read_text() == task.id + "\n"
    assert current(root) == task


def test_append_log_writes_one_stamped_line_per_event(root: Root):
    append_log(root, "t", {"event": "tick", "n": 1})
    append_log(root, "t", {"event": "verdict", "passed": False})
    lines = [json.loads(line) for line in root.log_file("t").read_text().splitlines()]
    assert [line["event"] for line in lines] == ["tick", "verdict"]
    assert all("ts" in line for line in lines) and lines[1]["passed"] is False


def test_pending_unit(task: Task):
    task.units.append(Unit("u2", "more", [Predicate("file_exists", {"path": "b"})]))
    assert pending_unit(Task("t", "g", "/w")) is None
    assert pending_unit(task).id == "u1"
    task.units[0].status = "passed"
    assert pending_unit(task).id == "u2"
    task.units[1].status = "failed"
    assert pending_unit(task) is None


def test_a_task_file_from_before_needs_loads_with_the_defaults():
    data = json.loads(to_json(Task("t", "g", "/w", units=[Unit("u1", "a", [Predicate("file_exists", {"path": "a"})])])))
    del data["accepted"]
    for key in ("needs", "candidate", "integrated"):
        del data["units"][0][key]
    task = from_json(json.dumps(data))
    assert task.accepted is None and (task.units[0].needs, task.units[0].candidate, task.units[0].integrated) == ([], None, False)


def test_pending_unit_is_the_first_whose_needs_have_passed_and_been_integrated():
    pred = [Predicate("file_exists", {"path": "a"})]
    task = Task("t", "g", "/w", units=[Unit("u3", "c", pred, needs=["u1", "u2"]), Unit("u1", "a", pred), Unit("u2", "b", pred)])
    assert pending_unit(task).id == "u1"
    task.units[1].status = task.units[2].status = "passed"
    assert pending_unit(task) is None
    task.units[1].integrated = task.units[2].integrated = True
    assert pending_unit(task).id == "u3"


def test_block_dependents_blocks_transitively_and_only_once():
    pred = [Predicate("file_exists", {"path": "a"})]
    task = Task("t", "g", "/w", units=[Unit("u3", "c", pred, needs=["u2"]), Unit("u2", "b", pred, needs=["u1"]),
                                       Unit("u1", "a", pred, "failed", 2), Unit("u4", "d", pred)])
    assert [(u.id, need.id) for u, need in block_dependents(task)] == [("u2", "u1"), ("u3", "u2")]
    assert [u.status for u in task.units] == ["blocked", "blocked", "failed", "pending"]
    assert block_dependents(task) == [] and pending_unit(task).id == "u4"


def test_a_task_file_from_before_the_console_loads_with_the_defaults_and_drops_the_old_question():
    data = json.loads(to_json(Task("t", "g", "/w")))
    for key in ("questions", "thoughts", "paused"):  # `directions` is written as `thoughts` now
        del data[key]
    data["question"] = None
    task = from_json(json.dumps(data))
    assert (task.questions, task.directions, task.paused) == ([], [], False)
    assert "question" not in json.loads(to_json(task))


def test_questions_get_the_next_id_and_an_open_one_is_never_replaced():
    task = Task("t", "g", "/w")
    first = add_question(task, "director", "question", "which file?")
    second = add_question(task, "runtime", "cap_hit", "extend?", {"cap_hit": "c1"})
    assert [q.id for q in open_questions(task)] == ["q1", "q2"] and second.context == {"cap_hit": "c1"}
    first.status = "answered"
    assert open_questions(task) == [second] and find_question(task, "q1") is first
    assert find_question(task, "q3") is None and first.text == "which file?"


def test_retain_persists_on_each_predicate_and_a_record_from_before_it_loads_retained(root: Root, live_receipt):
    task = Task("t", "g", "/w", units=[Unit("u1", "a", [Predicate("file_exists", {"path": "a"}),
                                                         Predicate("file_exists", {"path": "s"}, retain=False)])])
    data = json.loads(to_json(task))
    assert [p["retain"] for p in data["units"][0]["predicates"]] == [True, False] and from_json(to_json(task)) == task
    for predicate in data["units"][0]["predicates"]:
        del predicate["retain"]
    assert [p.retain for p in from_json(json.dumps(data)).units[0].predicates] == [True, True]
    source = live_receipt("live-5-2026-10-03")  # a live record from before the flag existed
    old = json.loads(source.read_text())
    target = root.task_file(old["id"])
    target.parent.mkdir(parents=True)
    shutil.copy2(source, target)
    assert all("retain" not in p for u in old["units"] for p in u["predicates"])
    loaded = load(root, old["id"]).units
    assert len(loaded[0].predicates) == 5 and all(p.retain is True for u in loaded for p in u.predicates)
