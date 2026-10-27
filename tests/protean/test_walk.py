"""Walk identity: each `run_tick` gets an opaque
`walk` id, unique within the session, carried on `Fired.walk` (snapshot `fired.walk`, the `fired` event), on
`Call.walk` (snapshot `doing.walk`, the `call` event) and on the `launch` event. `tick`, `task.tick` and every ceiling
keep their meaning: a walk cut before its save counts no tick, and the walk that runs that tick again has its own id.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from protean import dry, loop, state
from protean.dry import ScriptedRunner, make_workspace
from protean.graph import budget, console
from protean.graph.budget import Caps
from protean.state import Call, Fired, Predicate, Root, Task, Unit, current, load, open_questions, save, set_current

PASS = {"write": {"a.txt": "A\n"}, "commit": True, "exit_code": 0, "result": "wrote a.txt", "total_cost_usd": 0.5}


def _reply(obj: dict, cost: float = 0.1) -> dict:
    obj = {**obj, "state": {"standing": "It rests.", "open": []}}
    return {"type": "result", "is_error": False, "session_id": "s-1", "total_cost_usd": cost, "result": json.dumps(obj)}


@pytest.fixture
def env(tmp_path: Path, root: Root) -> tuple[Root, Path]:
    root.path.mkdir(parents=True)
    for prompt in (root.thinker_prompt, root.editor_prompt, root.reader_prompt):
        prompt.write_text("p\n")
    return root, make_workspace(tmp_path / "ws")


def _events(root: Root, task_id: str, name: str) -> list[dict]:
    return [e for e in state.read_log(root, task_id) if e["event"] == name]


def test_two_walks_of_one_tick_one_cut_before_its_save_carry_distinct_walks_and_the_tick_and_allowance_hold(
    env: tuple[Root, Path]
):
    root, workspace = env
    task = Task(id="task-20261026T140000", thought="t", workspace=str(workspace),
                units=[Unit("u-1", "write a.txt", [Predicate("file_exists", {"path": "a.txt"})])])
    save(root, task)
    set_current(root, task.id)
    seen: list[tuple] = []

    def cut(argv, stdin, cwd, timeout, **_):
        disk = load(root, task.id)  # what a client refreshing on the launch reads
        seen.append((disk.fired.walk, disk.calls[-1].walk, console.snapshot(disk, Caps(), False)["doing"],
                     state.read_log(root, task.id)[-1]))
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        loop.run_tick(task, Caps(), root, cut)
    after_cut = load(root, task.id)
    (first,) = after_cut.calls
    assert (after_cut.tick, first.tick, first.outcome) == (0, 0, "interrupted")  # the cut walk counted no tick
    assert isinstance(first.walk, str) and first.walk and after_cut.fired.walk == first.walk
    ((fired_walk, call_walk, doing, launch),) = seen
    assert fired_walk == call_walk == doing["walk"] == launch["walk"] == first.walk and launch["event"] == "launch"
    assert _events(root, task.id, "fired") == []  # a cut walk logs no `fired`
    (asked,) = open_questions(current(root))
    loop.offline_op(root, Caps(), "answer", {"id": asked.id, "text": "yes"})
    task = loop.resume(root, Caps(), ScriptedRunner([_reply({"rest": True})], [PASS]))
    assert [(c.id, c.seat, c.tick) for c in task.calls] == [("c1", "worker", 0), ("c2", "worker", 0),
                                                           ("c3", "director", 1)]
    walks = [c.walk for c in task.calls]
    assert all(isinstance(w, str) and w for w in walks) and len(set(walks)) == 3  # same tick, two walks
    assert walks[0] == first.walk  # the cut call keeps the walk it ran in
    fired = _events(root, task.id, "fired")
    assert [e["tick"] for e in fired] == list(range(task.tick)) == [0, 1]  # one counted tick each, as before
    assert [e["walk"] for e in fired] == walks[1:] and first.walk not in [e["walk"] for e in fired]
    assert [(e["call"], e["walk"]) for e in _events(root, task.id, "launch")] == list(zip(["c1", "c2", "c3"], walks))
    assert [(e["id"], e["walk"]) for e in _events(root, task.id, "call")] == list(zip(["c1", "c2", "c3"], walks))
    in_force = console.ceilings(task, Caps())
    assert (in_force["max_ticks"], in_force["allowance_usd"], in_force["accrued_usd"], task.active_seconds) == (
        100, 25.0, 0.0, 0.0)  # the allowance and the tick ceiling read as they did before walks
    snap = console.snapshot(task, Caps(), False)
    assert (snap["tick"], snap["fired"]["tick"], snap["fired"]["walk"]) == (2, 1, walks[2])
    assert load(root, task.id).fired == task.fired


def test_every_walk_of_a_scenario_has_its_own_id_and_each_call_names_the_walk_of_its_tick(tmp_path: Path, env):
    root, _ = env
    task, _ = dry.run_scenario("think", root, make_workspace(tmp_path / "ws2"), dry.scenario_runner("think"),
                               lambda _: None)
    fired = _events(root, task.id, "fired")
    walks = [e["walk"] for e in fired]
    assert len(walks) == task.tick and len(set(walks)) == len(walks) and all(isinstance(w, str) and w for w in walks)
    tick_of = {e["walk"]: e["tick"] for e in fired}
    calls = _events(root, task.id, "call")
    assert calls and all(tick_of[e["walk"]] == e["tick"] for e in calls)
    launches = {e["call"]: e["walk"] for e in _events(root, task.id, "launch")}
    assert launches == {e["id"]: e["walk"] for e in calls}
    assert [c.walk for c in task.calls] == [e["walk"] for e in calls]


def test_a_record_from_before_walks_loads_with_none_and_doing_names_none(root: Root):
    task = Task(id="task-20261026T150000", thought="t", workspace="/nowhere", tick=2,
                fired=Fired(tick=1, walked=["repair"]))
    budget.open_call(task, "director", 1, 2.0, "m")
    assert task.calls[-1].walk is None  # a call opened outside a walk names none (here: a record from before)
    older = json.loads(state.to_json(task))
    del older["fired"]["walk"]
    for call in older["calls"]:
        del call["walk"]
    loaded = state.from_json(json.dumps(older))
    assert (loaded.fired.walk, loaded.calls[0].walk) == (None, None) and loaded == task
    snap = console.snapshot(loaded, Caps(), False)
    assert snap["fired"]["walk"] is None and "walk" not in snap["doing"]  # absent when the call names none
    task.fired = Fired(tick=2, walk="w-x")
    assert budget.open_call(task, "worker", 1, 4.0, "m", "u-1").walk == "w-x"
    assert console.snapshot(task, Caps(), False)["doing"]["walk"] == "w-x"
    assert Call("c9", "reader", 1, 1.5).walk is None
