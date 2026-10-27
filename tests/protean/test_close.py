"""The operator ends a session for good (the operator, 2026-10-20: "i want the thinking sessions to end (because cache calls and
rereading)"): `close` supersedes every open question, drops held work, leaves the record `stopped`/`closed` and
readable, and from then on every console operation and `protean resume` are refused. Offline, `ctl close` lands in
`task.json`; the live path is `test_console.test_close_ends_the_live_session…`."""

from __future__ import annotations

import pytest

from protean import loop, state
from protean.dry import ScriptedRunner
from protean.graph import console, questions
from protean.graph.budget import Caps
from protean.state import DIRECTOR, PLAN, Held, Predicate, Root, Task, Unit, save, set_current

TASK_ID = "task-20261020T120000"


@pytest.fixture
def parked(tmp_path, root: Root) -> tuple[Root, Task]:
    """A session parked on the thinker's question with a plan held for approve and a clone directory on disk."""
    root.path.mkdir(parents=True)
    task = Task(id=TASK_ID, goal="g", workspace=str(tmp_path / "ws"), mode="hold", terminal="interrupted",
                awaiting_thought=False)
    task.held = Held(kind=PLAN, units=[Unit("u-1", "write a.txt", [Predicate("file_exists", {"path": "a.txt"})])])
    save(root, task)
    set_current(root, task.id)
    questions.ask(root, task, DIRECTOR, "question", "which file?")
    root.clones_dir(task.id).mkdir(parents=True, exist_ok=True)
    (root.clones_dir(task.id) / "u-1").mkdir()
    return root, state.load(root, task.id)


def test_an_offline_close_ends_the_parked_session_for_good(parked: tuple[Root, Task]):
    root, task = parked
    (open_question,) = state.open_questions(task)
    event = loop.offline_op(root, Caps(), "close", {})
    assert event == {"event": "close", "actor": "console", "live": False, "superseded": [open_question.id],
                     "held": "plan (u-1)"}
    saved = state.load(root, task.id)
    assert (saved.terminal, saved.stop_reason, saved.held, saved.awaiting_thought, saved.paused) == (
        "stopped", "closed", None, False, False)
    assert [(q.status, q.disposition) for q in saved.questions] == [("superseded", "closed")]
    assert state.is_closed(saved) and not root.clones_dir(task.id).exists()
    assert state.read_log(root, task.id)[-1]["event"] == "close"
    assert "ends for good" in console.describe(event) and "1 open question(s) superseded" in console.describe(event)
    assert "the held plan (u-1) dropped" in console.describe(event)


def test_a_closed_session_refuses_every_operation_and_a_resume(parked: tuple[Root, Task]):
    root, task = parked
    loop.offline_op(root, Caps(), "close", {})
    closed = state.load(root, task.id)
    for op, body in [("pause", {}), ("resume", {}), ("stop", {}), ("think", {"text": "more"}),
                     ("answer", {"id": "q1", "text": "a.txt"}), ("extend", {"usd": 1.0}), ("mode", {"mode": "auto"}),
                     ("approve", {}), ("close", {})]:
        with pytest.raises(console.Refused) as refused:
            console.check(closed, op, body, True)
        assert refused.value.status == 409 and "is closed (closed)" in str(refused.value), op
    runner = ScriptedRunner([], [], [])
    resumed = loop.resume(root, Caps(), runner, unattended=0.0, notify=None)
    assert (resumed.terminal, resumed.stop_reason, runner.calls) == ("stopped", "closed", [])
    assert state.read_log(root, task.id)[-1]["event"] == "close"  # a resume of a closed session logs nothing


def test_the_snapshot_of_a_closed_session_reads_closed():
    task = Task(id=TASK_ID, goal="g", workspace="w")
    console.apply(task, Caps(), "close", {}, False)
    snapshot = console.snapshot(task, Caps(), live=False)
    assert (snapshot["terminal"], snapshot["stop_reason"], snapshot["held"], snapshot["awaiting_thought"]) == (
        "stopped", "closed", None, False)
