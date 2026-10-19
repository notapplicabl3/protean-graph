"""The thinker's moves in the loop, with a scripted thinker and a scripted worker.

`ideas` and `rest` leave the session awaiting a thought: nothing is called until a console `think` lands, and with
no client attached it parks as a question does. An `explore` is one worker call in its own clone at the accepted tip,
on `explore/<session>/<n>`, fetched back, never graded, retried or integrated, its report stored unverified. A `task`
move's units run exactly as a plan's did. In `hold`, an explore and a task move each wait for `approve` before any
worker call, and a think while either is held returns it. A reply invalid twice parks on the runtime's question; a
standing left unchanged is returned once, and a second time opens `no_progress`.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import threading
from pathlib import Path

import pytest

from protean import cli, loop, state
from protean.dry import ScriptedRunner, make_workspace
from protean.graph import console, worker
from protean.graph.budget import Caps
from protean.state import Move, Root, Task, load, open_questions, save, set_current


def _exemplar():
    """The battery's scripted console client lives in `test_console.py`; it is loaded from there, not copied."""
    spec = importlib.util.spec_from_file_location("protean_console_client", Path(__file__).with_name("test_console.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_c = _exemplar()
Stream, _client, _in_client_thread, _rested = _c.Stream, _c._client, _c._in_client_thread, _c._rested

O1 = {"id": "o1", "line": "where b.txt goes"}
IDEAS = {"ideas": [{"id": "i1", "case": "b.txt at the root, beside the README", "cost": "one explore", "serves": "o1"},
                   {"id": "i2", "case": "b.txt under docs/", "cost": "one explore", "serves": "o1"},
                   {"id": "i3", "case": "no b.txt at all", "cost": "nothing", "serves": "o1"}],
         "state": {"standing": "There is no b.txt yet. Three places could hold it.", "open": [O1]}}
EXPLORE = {"explore": "Try b.txt at the root and say whether it fits.",
           "state": {"standing": "The operator picked the root. An explore tries b.txt there first.", "open": [O1]}}
U1 = {"id": "u-1", "intent": "write b.txt at the root",
      "predicates": [{"kind": "file_exists", "args": {"path": "b.txt"}}]}
TASK = {"task": [U1], "state": {"standing": "The explore says the root fits. A task writes b.txt there.", "open": []}}
REST = {"rest": True, "state": {"standing": "b.txt is written at the root and checked.", "open": []}}
FOUND = {"write": {"notes/b-root.md": "The root fits.\n"}, "commit": True, "exit_code": 0,
         "result": "The root fits: nothing else lives there.\nA draft note is on the branch.", "total_cost_usd": 0.4}
PASS = {"write": {"b.txt": "B\n"}, "commit": True, "exit_code": 0, "result": "wrote b.txt", "total_cost_usd": 0.5}


def _reply(obj: dict | str, cost: float = 0.1) -> dict:
    text = obj if isinstance(obj, str) else json.dumps(obj)
    return {"type": "result", "is_error": False, "session_id": "s-1", "total_cost_usd": cost, "result": text}


def _moved(standing: str, **move: object) -> dict:
    """A thinker reply: one move and a state of `standing`."""
    return _reply({**move, "state": {"standing": standing, "open": []}})


def _git(cwd: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True)


def _events(root: Root, task_id: str, name: str) -> list[dict]:
    return [e for e in state.read_log(root, task_id) if e["event"] == name]


def _seats(runner: ScriptedRunner) -> list[str]:
    return [ScriptedRunner.seat(argv) for argv, _, _, _ in runner.calls]


def _messages(runner: ScriptedRunner) -> list[dict]:
    return [json.loads(stdin) for argv, stdin, _, _ in runner.calls if ScriptedRunner.seat(argv) == "director"]


@pytest.fixture
def env(tmp_path: Path, root: Root) -> tuple[Root, Path]:
    root.path.mkdir(parents=True)
    root.thinker_prompt.write_text("t\n")
    root.editor_prompt.write_text("e\n")
    return root, make_workspace(tmp_path / "ws")


# --- ideas -> a thought -> explore -> task -> rest, driven by the scripted client ---


class Refs:
    """The scripted runner, recording before each call which `task/` refs the workspace holds."""

    def __init__(self, scripted: ScriptedRunner, workspace: Path, attached: threading.Event) -> None:
        self.scripted, self.workspace, self.attached, self.seen = scripted, workspace, attached, []

    def __call__(self, argv, stdin, cwd, timeout):
        if not self.scripted.calls:
            assert self.attached.wait(10), "the client never attached"
        refs = _git(self.workspace, "for-each-ref", "--format=%(refname:short)", "refs/heads/task/").stdout.split()
        self.seen.append((ScriptedRunner.seat(argv), refs))
        return self.scripted(argv, stdin, cwd, timeout)


def test_ideas_then_a_thought_then_an_explore_a_task_and_a_rest_each_move_the_record(
    env: tuple[Root, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    root, workspace = env
    shim_log = tmp_path / "shim.log"
    monkeypatch.setenv("PROTEAN_SHIM_LOG", str(shim_log))
    scripted = ScriptedRunner([_reply(IDEAS), _reply(EXPLORE), _reply(TASK), _reply(REST)], [FOUND, PASS])
    attached = threading.Event()
    runner = Refs(scripted, workspace, attached)

    def script(seen: dict) -> None:
        client = _client(root)
        stream = Stream(client)
        stream.start()
        try:
            stream.wait_for(lambda line: "snapshot" in line)
            attached.set()
            seen["ideas"] = stream.wait_for(lambda line: line.get("event") == "ideas")
            seen["waiting"] = (len(scripted.calls), load(root, state.current(root).id).awaiting_thought)
            seen["think"] = client.call("POST", "/think", {"text": "go with i1"})
            stream.wait_for(_rested)
        finally:
            seen["lines"] = list(stream.lines)
            stream.close()

    thread, seen = _in_client_thread(script)
    task = loop.run("Where should b.txt go?", workspace, root, Caps(), runner, unattended=0.2)
    thread.join(15)
    assert "error" not in seen, repr(seen.get("error"))
    explore = f"explore/{task.id}/1"

    # the moves, in order, and the standing after each one differs from the one before
    assert [m.kind for m in task.moves] == ["ideas", "explore", "task", "rest"]
    standings = [e["standing"] for e in _events(root, task.id, "thought")]
    assert standings == [reply["state"]["standing"] for reply in (IDEAS, EXPLORE, TASK, REST)]
    assert all(before != after for before, after in zip([""] + standings, standings))
    assert [e["kind"] for e in _events(root, task.id, "thought")] == ["ideas", "explore", "task", "rest"]

    # ideas: no further call until the thought landed; it reached the thinker's next call
    assert seen["waiting"] == (1, True) and seen["think"][0] == 200
    assert [i["id"] for i in seen["ideas"]["ideas"]] == ["i1", "i2", "i3"] and task.moves[0].outcome == "proposed"
    messages = _messages(scripted)
    assert [m["thoughts"] for m in messages] == [[], ["go with i1"], ["go with i1"], ["go with i1"]]

    # the explore: its branch exists in the workspace, nothing graded or integrated it, task/<session> unmoved by it
    assert _git(workspace, "rev-parse", "--verify", "--quiet", f"refs/heads/{explore}").returncode == 0
    assert runner.seen[:4] == [("director", []), ("director", []), ("worker", []), ("director", [])]
    found = _git(workspace, "rev-parse", explore).stdout.strip()
    assert all(v["branch"] != explore and v["commit"] != found for v in _events(root, task.id, "verdict"))
    assert all(e["accepted"] != found for e in _events(root, task.id, "integrated"))
    assert _git(workspace, "merge-base", "--is-ancestor", found, f"task/{task.id}").returncode == 1
    assert "notes/b-root.md" not in _git(workspace, "ls-tree", "-r", "--name-only", f"task/{task.id}").stdout
    launch, landed = _events(root, task.id, "explore")
    assert (launch["stage"], launch["brief"], launch["branch"]) == ("launch", EXPLORE["explore"], explore)
    assert (landed["stage"], landed["branch"], landed["exit_code"], landed["cost_usd"], landed["unverified"]) == (
        "landed", explore, 0, 0.4, True)
    assert landed["report"] == FOUND["result"] + "\nexit_code: 0" and landed["changed"] == ["notes/b-root.md"]
    move = task.moves[1]
    assert (move.branch, move.brief, move.report, move.outcome) == (explore, EXPLORE["explore"], landed["report"],
                                                                     "landed")
    assert messages[2]["moves"][1]["unverified_report"] == landed["report"]  # labelled unverified for the thinker
    (explore_call,) = [c for c in task.calls if c.seat == "worker" and c.unit is None]
    assert explore_call.outcome == "ok" and not (root.clones_dir(task.id) / worker.EXPLORE_CLONE).exists()

    # the task: its unit graded and integrated on task/<session>
    (unit,) = task.units
    (verdict,) = _events(root, task.id, "verdict")
    assert (unit.id, unit.status, unit.integrated, verdict["passed"]) == ("u-1", "passed", True, True)
    assert _git(workspace, "rev-parse", f"task/{task.id}").stdout.strip() == unit.candidate == task.accepted
    assert (task.moves[2].units, task.moves[2].outcome, task.moves[2].branch) == (["u-1"], "u-1 passed",
                                                                                   f"task/{task.id}")

    # the rest: the session awaits a thought, and with the client gone it parks as a question does
    assert (task.terminal, task.awaiting_thought) == ("interrupted", True)
    assert [(e["hold"], e["on"]) for e in _events(root, task.id, "parked")] == [("thought", "m4")]
    assert _seats(scripted) == ["director", "director", "worker", "director", "worker", "director"]
    assert not shim_log.exists()  # the shim recorded no call


def test_an_explore_after_a_task_starts_at_the_accepted_tip_and_leaves_it_there(env: tuple[Root, Path]):
    root, workspace = env
    runner = ScriptedRunner([_reply(TASK), _reply(EXPLORE), _reply(REST)], [PASS, FOUND])
    task = loop.run("g", workspace, root, Caps(), runner, unattended=0.0, notify=None)
    explore = f"explore/{task.id}/1"
    tip = _git(workspace, "rev-parse", f"task/{task.id}").stdout.strip()
    assert tip == task.units[0].candidate == task.accepted  # the explore did not move it
    assert _git(workspace, "merge-base", "--is-ancestor", tip, explore).returncode == 0  # it began at the tip
    assert _git(workspace, "rev-parse", f"{explore}~1").stdout.strip() == tip
    assert [m.kind for m in task.moves] == ["task", "explore", "rest"] and len(_events(root, task.id, "verdict")) == 1


# --- in hold, an explore and a task move each wait for approve; a think returns either ---


def _parked_held(root: Root, task: Task, kind: str) -> bool:
    return (task.terminal, task.held.kind, task.held.approved, _events(root, task.id, "parked")[-1]["on"]) == (
        "interrupted", kind, False, kind)


def test_in_hold_an_explore_and_a_task_move_each_wait_for_approve_before_any_worker_call(
    env: tuple[Root, Path], monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    root, workspace = env
    runner = ScriptedRunner([_reply(EXPLORE), _reply(TASK), _reply(REST)], [FOUND, PASS])
    task = loop.run("g", workspace, root, Caps(), runner, unattended=0.0, notify=None, mode="hold")
    assert _parked_held(root, task, "explore") and _seats(runner) == ["director"]  # no worker call while held
    assert (task.held.move, task.held.brief) == ("m1", EXPLORE["explore"]) and task.moves[0].outcome == "pending"
    (held,) = _events(root, task.id, "held")
    assert (held["kind"], held["move"], held["brief"]) == ("explore", "m1", EXPLORE["explore"])
    assert loop.resume(root, Caps(), runner).terminal == "interrupted" and _seats(runner) == ["director"]

    monkeypatch.setattr(cli, "real_runner", runner)
    brain = ["--brain", str(root.path)]
    assert cli.main([*brain, "ctl", "approve"]) == 0  # no brain live: written to task.json
    assert "approve explore m1" in capsys.readouterr().out
    assert cli.main([*brain, "resume"]) == 12  # the explore ran; the task move is held in its turn
    task = load(root, task.id)
    assert _parked_held(root, task, "plan") and _seats(runner) == ["director", "worker", "director"]
    assert task.units == [] and task.moves[1].outcome == "held"
    assert cli.main([*brain, "ctl", "approve"]) == 0 and cli.main([*brain, "resume"]) == 12
    task = load(root, task.id)
    assert _seats(runner) == ["director", "worker", "director", "worker", "director"]
    assert [(e["kind"], e["by"]) for e in _events(root, task.id, "approved")] == [("explore", "approve"),
                                                                                  ("plan", "approve")]
    assert [(m.kind, m.outcome) for m in task.moves] == [("explore", "landed"), ("task", "u-1 passed"),
                                                         ("rest", "rested")]
    assert (task.awaiting_thought, task.units[0].status, task.mode) == (True, "passed", "hold")


def test_a_think_while_an_explore_or_a_task_move_is_held_returns_it_and_none_of_it_runs(env: tuple[Root, Path]):
    root, workspace = env
    runner = ScriptedRunner([_reply(EXPLORE), _reply(TASK), _reply(REST)], [FOUND, PASS])
    task = loop.run("g", workspace, root, Caps(), runner, unattended=0.0, notify=None, mode="hold")
    assert _parked_held(root, task, "explore")
    loop.offline_op(root, Caps(), "think", {"text": "skip the explore, just write it"})
    task = loop.resume(root, Caps(), runner, unattended=0.0, notify=None)
    assert _parked_held(root, task, "plan")
    loop.offline_op(root, Caps(), "think", {"text": "not yet"})
    task = loop.resume(root, Caps(), runner, unattended=0.0, notify=None)
    assert _seats(runner) == ["director", "director", "director"]  # no worker call: nothing held ever ran
    returned = _events(root, task.id, "returned")
    assert [(e["kind"], e.get("move"), e.get("units"), e["thoughts"]) for e in returned] == [
        ("explore", "m1", None, 1), ("plan", None, ["u-1"], 2)]
    second, third = _messages(runner)[1:]
    assert returned[0]["to_director"] in second["notes"] and second["thoughts"] == ["skip the explore, just write it"]
    assert returned[1]["to_director"] in third["notes"] and len(third["thoughts"]) == 2
    assert [(m.kind, m.outcome, m.units, m.branch) for m in task.moves] == [
        ("explore", "returned", [], None), ("task", "returned", [], None), ("rest", "rested", [], None)]
    assert task.units == [] and task.held is None and task.awaiting_thought
    assert _git(workspace, "for-each-ref", "refs/heads/explore/", "refs/heads/task/").stdout == ""


# --- a think that lands during an ideas or rest call is never lost to the wait after it ---


@pytest.mark.parametrize("first", [IDEAS, {"rest": True, "state": {"standing": "Nothing to write yet.", "open": []}}],
                         ids=["ideas", "rest"])
def test_a_think_landing_during_an_ideas_or_rest_call_wakes_the_session_and_rides_in_the_next_call(
    env: tuple[Root, Path], first: dict
):
    root, workspace = env
    scripted, landed = ScriptedRunner([_reply(first), _reply(REST)], []), []

    def runner(argv, stdin, cwd, timeout):
        if not landed:  # the operator's think reaches the live console while the thinker's first call runs
            landed.append(console.request(root, state.current(root).id, "POST", "/think", {"text": "at the root"})[0])
        return scripted(argv, stdin, cwd, timeout)

    task = loop.run("b.txt", workspace, root, Caps(), runner, unattended=0.0, notify=None)
    assert _seats(scripted) == ["director", "director"]  # the thought woke it: the thinker moved again
    assert landed == [200] and [m["thoughts"] for m in _messages(scripted)] == [[], ["at the root"]]
    assert [(e["hold"], e["on"]) for e in _events(root, task.id, "parked")] == [("thought", "m2")]  # no wait after m1
    assert [m.kind for m in task.moves] == [next(iter(first)), "rest"] and task.awaiting_thought


# --- a reply invalid twice parks; an unchanged standing is returned once, then opens `no_progress` ---


def test_a_malformed_reply_is_returned_once_with_its_errors_then_the_session_parks(env: tuple[Root, Path]):
    root, workspace = env
    two_moves = _reply({"ask": "which?", "rest": True, "state": {"standing": "Two moves.", "open": []}})
    runner = ScriptedRunner([two_moves, _reply({"rest": True})], [])
    task = loop.run("g", workspace, root, Caps(), runner, unattended=0.0, notify=None)
    assert _seats(runner) == ["director", "director"] and task.moves == []
    retry = runner.calls[1][1]
    assert retry.startswith(runner.calls[0][1] + "\n\nYour previous answer was rejected:\n- the reply must carry "
                            "exactly one move of ideas, explore, task, ask, rest; it carried ['ask', 'rest']")
    (asked,) = open_questions(task)
    assert (asked.asker, asked.reason, task.terminal) == ("runtime", "invalid_plan", "interrupted")
    assert asked.text == 'The thinker returned an invalid reply twice: the reply must carry state: {"standing": ' \
                         '"...", "open": [{"id": "...", "line": "..."}]}'
    assert loop.resume(root, Caps(), runner).terminal == "interrupted" and len(runner.calls) == 2


def test_an_unchanged_standing_is_returned_once_then_no_progress_holds_every_paid_call_until_yes(
    env: tuple[Root, Path]
):
    root, workspace = env
    same = "b.txt is planned."
    runner = ScriptedRunner([_moved(same, task=[U1]), _moved(same, rest=True), _moved(same, rest=True),
                             _moved("b.txt is written and checked.", rest=True)], [PASS])
    task = loop.run("g", workspace, root, Caps(), runner, unattended=0.0, notify=None)
    assert _seats(runner) == ["director", "worker", "director", "director"]
    assert [m.kind for m in task.moves] == ["task"] and task.standing == same  # neither unchanged reply acted on
    first, second = _events(root, task.id, "no_progress")
    assert (first["to_director"], "to_director" in second) == ("no progress: standing unchanged", False)
    assert _messages(runner)[2]["notes"] == ["no progress: standing unchanged"]  # returned to the thinker once
    (asked,) = open_questions(task)
    assert (asked.asker, asked.reason, task.terminal, task.awaiting_thought) == (
        "runtime", "no_progress", "interrupted", False)
    assert loop.resume(root, Caps(), runner).terminal == "interrupted" and len(runner.calls) == 4  # no paid call

    loop.offline_op(root, Caps(), "think", {"text": "it is fine as it is"})
    loop.offline_op(root, Caps(), "answer", {"id": asked.id, "text": "yes, go on"})
    task = loop.resume(root, Caps(), runner)
    assert len(runner.calls) == 5 and [m.kind for m in task.moves] == ["task", "rest"]
    after = runner.calls[4][1]
    message = json.loads(after)
    assert message["thoughts"] == ["it is fine as it is"] and message["notes"] == []  # the thought rides
    assert "go on" not in after and asked.text not in after and "answer" not in message  # the answer never does
    assert (task.standing, task.awaiting_thought, [q.disposition for q in task.questions]) == (
        "b.txt is written and checked.", True, ["continued"])


# --- a kill mid-explore: the explore never runs again ---


def test_an_explore_a_kill_cut_mid_call_is_marked_lost_on_resume_and_never_runs_again(env: tuple[Root, Path]):
    root, workspace = env
    cut = Move("m1", "explore", "try b.txt", outcome="running", branch="explore/task-20261017T120000/1",
               brief="try b.txt")
    task = Task(id="task-20261017T120000", goal="g", workspace=str(workspace), moves=[cut],
                standing="An explore tries b.txt.")
    save(root, task)
    set_current(root, task.id)
    runner = ScriptedRunner([_moved("The explore was lost; b.txt waits.", rest=True)], [FOUND])
    task = loop.resume(root, Caps(), runner, unattended=0.0, notify=None)
    assert task.moves[0].outcome == "lost: the brain stopped during its call" and _seats(runner) == ["director"]
    assert _messages(runner)[0]["moves"][0]["outcome"] == "lost: the brain stopped during its call"


def test_an_interrupted_explore_is_booked_on_the_session_and_its_move_and_never_runs_again(env: tuple[Root, Path]):
    root, workspace = env
    task = Task(id="task-20261017T130000", goal="g", workspace=str(workspace), standing="An explore tries b.txt.",
                moves=[Move("m1", "explore", "try b.txt", 0.1, "pending", brief="try b.txt")])
    save(root, task)
    set_current(root, task.id)

    def interrupt(argv, stdin, cwd, timeout):
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        loop.run_tick(task, Caps(), root, interrupt)
    saved = load(root, task.id)
    (call,) = saved.calls
    assert (call.seat, call.unit, call.outcome, call.charge_usd, call.booked) == ("worker", None, "interrupted", 4.0,
                                                                                  True)
    move = saved.moves[0]
    assert (move.outcome, move.cost_usd, move.branch, saved.cost_usd) == (
        "interrupted", 4.1, f"explore/{task.id}/1", 4.0)
    assert [q.reason for q in open_questions(saved)] == ["unknown_spend"]  # paid work holds on the unknown spend
    loop.offline_op(root, Caps(), "answer", {"id": "q1", "text": "yes"})
    runner = ScriptedRunner([_moved("The explore was cut; b.txt waits.", rest=True)], [FOUND])
    task = loop.resume(root, Caps(), runner, unattended=0.0, notify=None)
    assert _seats(runner) == ["director"] and task.moves[0].outcome == "interrupted"  # no retry of an explore
