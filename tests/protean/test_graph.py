"""The node registry (`graph/graph.py`) and the walk the ring takes over it (shape SPEC, decisions 2–9).

Every node is declared once with who fires it and its edges, and the edges alone order a tick: two entries of one tick
that no edge orders, or a cycle, are refused, never settled by where they are declared. The required edges are proven
two ways: on the walk the registry gives every shape of tick, and on real ticks, where each model call is
seen against the nodes the walker ran before it. A reply fires exactly one node the thinker may fire; anything else
comes back once with its errors, then parks on the runtime's question (row 4). In `hold`, one `approve` releases the
held firing set, and a `think` returns it (row 4). Every tick of `protean dry think` logs its firing set (row 2).
"""

from __future__ import annotations

import itertools
import json
import subprocess
from dataclasses import asdict, replace
from pathlib import Path

import pytest

from protean import dry, loop, state
from protean.dry import ScriptedRunner, make_workspace, scenario_runner
from protean.graph import console, director, graph
from protean.graph.budget import Caps
from protean.state import HOLD, PLAN, Firing, Held, Predicate, Root, Task, Unit, open_questions, save, set_current

STANDINGS = itertools.count(1)
UNIT = {"id": "u-1", "intent": "write a.txt", "predicates": [{"kind": "file_exists", "args": {"path": "a.txt"}}]}
PASS = {"write": {"a.txt": "A\n"}, "commit": True, "exit_code": 0, "result": "wrote a.txt", "total_cost_usd": 0.5}
LANDED = {"write": {"notes.md": "n\n"}, "commit": True, "exit_code": 0, "result": "drafted", "total_cost_usd": 0.4}


def _reply(obj: dict, cost: float = 0.1) -> dict:
    """A thinker reply; a move without `state` gets one whose standing no earlier reply had."""
    if "state" not in obj:
        obj = {**obj, "state": {"standing": f"Standing {next(STANDINGS)}.", "open": []}}
    return {"type": "result", "is_error": False, "session_id": "s-1", "total_cost_usd": cost, "result": json.dumps(obj)}


def _names(keys: list[str], nodes=None) -> list[str]:
    table = graph.entries(nodes)
    return [table[key].name for key in keys]


# Every shape a tick can take is the registry's own list (`graph.SHAPES`, ordered by `graph.check` at import): by what
# it adds to the entries in every tick, the node the runtime fires (the thinker) or the firing pick draws, and what
# that firing brings.
SHAPES = graph.SHAPES
WALKS = {shape: _names(graph.walk(shape)) for shape in SHAPES}
PAID = ("director", "worker", "release", "gate")  # the thinker; a unit or an explore; a held plan's reader; weakening


# --- B1: the registry ---


def test_every_node_is_declared_once_with_who_fires_it_and_the_thinker_fires_only_worker_console_and_rest():
    names = sorted(node.name for node in graph.NODES)  # as a set with block twice: where a node is declared is moot
    assert names == sorted(["repair", "block", "budget", "release", "pick", "hold", "director", "worker", "grader",
                            "gate", "ref", "console", "rest", "memory", "block", "save", "reader"])
    assert len({node.key for node in graph.NODES}) == len(graph.NODES)  # block's two entries: `block`, `block2`
    fired_by = {node.name: node.fired_by for node in graph.NODES}
    assert sorted(name for name, by in fired_by.items() if by == graph.THINKER) == ["console", "rest", "worker"]
    required = ("repair", "budget", "grader", "gate", "reader", "block", "ref", "memory", "save")  # decision 3
    runtime_jobs = ("release", "pick", "hold", "director")  # A5: the runtime's other jobs, the thinker call included
    assert all(fired_by[name] == graph.RUNTIME for name in required + runtime_jobs)
    assert graph.FIRES == {"task": ("worker",), "explore": ("worker",), "ask": ("console",), "ideas": ("console",),
                           "rest": ("rest",)}
    assert [node.name for node in graph.NODES if node.queued] == ["worker"]  # a worker firing waits for a later tick
    assert graph.nested("gate", "director") and not graph.nested("gate", "release")  # the one nested edge


def test_a_bare_name_two_entries_share_is_refused_and_a_caller_names_the_entry_by_its_key():
    with pytest.raises(graph.Unordered, match="'block' names 2 entries"):
        graph.node("block")
    assert graph.entries()["block"].name == graph.entries()["block2"].name == "block"
    assert graph.node("rest").key == "rest" and graph.node("notice") is None
    assert graph.firing_errors(["block"]) == ["'block' is fired by the runtime, never by the thinker"]
    reply, errors = director.decode_move(json.dumps({"rest": True, "block": True, "state": STATE}), Task(id="t"))
    assert reply is None and "'block' is fired by the runtime, never by the thinker" in errors


def test_the_edges_give_each_shape_of_tick_one_walk():
    assert WALKS == {
        "no firing": ["repair", "block", "budget", "release", "pick", "hold", "block", "save"],
        "thinker": ["repair", "block", "budget", "release", "pick", "hold", "director", "block", "save"],
        "thinker, then its console firing": ["repair", "block", "budget", "release", "pick", "hold", "director",
                                             "console", "block", "save"],
        "thinker, then its rest firing": ["repair", "block", "budget", "release", "pick", "hold", "director", "rest",
                                          "memory", "block", "save"],
        "worker on a unit": ["repair", "block", "budget", "release", "pick", "hold", "worker", "grader", "gate", "ref",
                             "block", "save"],
        "worker on an explore": ["repair", "block", "budget", "release", "pick", "hold", "worker", "block", "save"],
    }


def test_the_edges_alone_order_a_tick_never_where_a_node_is_declared():
    for shuffled in (tuple(reversed(graph.NODES)), graph.NODES[1::2] + graph.NODES[::2]):
        assert {shape: _names(graph.walk(shape, shuffled), shuffled) for shape in SHAPES} == WALKS
    loose = tuple(replace(node, before=()) if node.name == "release" else node for node in graph.NODES)
    with pytest.raises(graph.Unordered, match="no edge orders"):  # release no longer before pick: a tie, refused
        graph.order(graph.always(loose), loose)
    looped = tuple(replace(node, after=("save",)) if node.name == "repair" else node for node in graph.NODES)
    with pytest.raises(graph.Unordered, match="cycle"):
        graph.order(graph.always(looped), looped)


@pytest.mark.parametrize("change, refusal", [
    (lambda node: replace(node, id="block") if node.key == "block2" else node, "declared twice"),
    (lambda node: replace(node, after=("nowhere",)) if node.name == "pick" else node, "no entry declares"),
    (lambda node: replace(node, fired_by="operator") if node.name == "pick" else node, "fired_by must be"),
    (lambda node: replace(node, fired_by=graph.THINKER) if node.name == "reader" else node, "exactly worker"),
])
def test_a_registry_that_cannot_be_walked_is_refused(change, refusal):
    with pytest.raises(graph.Unordered, match=refusal):
        graph.check(tuple(change(node) for node in graph.NODES))


@pytest.mark.parametrize("change, refusal", [
    # an added edge: the console firing before the thinker it follows, a cycle in that firing's tick alone
    (lambda node: replace(node, before=("director",)) if node.name == "console" else node, "cycle"),
    # an added `brings`: a rest brings the grader, which no edge orders without a worker, a tie in the rest's tick alone
    (lambda node: replace(node, brings=("memory", "grader")) if node.name == "rest" else node, "no edge orders"),
], ids=["a cycle", "a tie"])
def test_a_registry_that_breaks_one_shape_of_tick_beyond_the_entries_in_every_tick_is_refused_at_check(change, refusal):
    """Refused when the registry is checked (at import), never mid-walk after a paid call."""
    changed = tuple(change(node) for node in graph.NODES)
    graph.order(graph.always(changed), changed)  # the entries in every tick still order: one other shape breaks
    with pytest.raises(graph.Unordered, match=refusal):
        graph.check(changed)


# --- the required edges, on every shape of tick ---


@pytest.mark.parametrize("shape", list(SHAPES))
def test_repair_is_first_and_save_last_in_every_tick(shape: str):
    walk = WALKS[shape]
    assert walk[0] == "repair" and walk[-1] == "save" and walk.count("repair") == walk.count("save") == 1


@pytest.mark.parametrize("shape", list(SHAPES))
def test_budget_comes_before_every_node_that_makes_a_paid_call(shape: str):
    """The thinker, the worker (a unit or an explore), the release (a held plan's reader calls: ceilings before
    release) and the gate (the weakening reader call; in-call admission runs inside the thinker's call)."""
    walk = WALKS[shape]
    assert all(walk.index("budget") < walk.index(name) for name in PAID if name in walk)
    assert walk.index("budget") < walk.index("release")


def test_the_grader_follows_every_worker_unit_and_precedes_the_ref_move_and_an_explore_brings_neither():
    unit, explore = WALKS["worker on a unit"], WALKS["worker on an explore"]
    assert unit.index("worker") < unit.index("grader") < unit.index("gate") < unit.index("ref")
    assert not {"grader", "gate", "ref"} & set(explore)


@pytest.mark.parametrize("shape", list(SHAPES))
def test_block_runs_right_after_repair_and_again_after_the_fired_node(shape: str):
    walk = WALKS[shape]
    first, second = (i for i, name in enumerate(walk) if name == "block")
    assert first == walk.index("repair") + 1 and walk.count("block") == 2
    fired = [name for name in ("director", "worker", "ref", "console", "rest", "memory") if name in walk]
    assert bool(fired) == (SHAPES[shape] != graph.Shape())  # only the shape that fires nothing has no fired node
    assert all(walk.index(name) < second for name in fired) and second == len(walk) - 2


def test_memory_s_line_follows_every_rest_firing_and_no_other():
    assert graph.brought("rest") == ("memory",)
    rest = WALKS["thinker, then its rest firing"]
    assert rest.index("memory") == rest.index("rest") + 1
    assert all("memory" not in walk for shape, walk in WALKS.items() if "rest" not in walk)


# --- the same edges on real ticks: every model call against the nodes walked before it ---


class Tracer:
    """Every node act the walker calls and every model call, in one sequence; a tick starts at each `repair`."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch, runner) -> None:
        self.seen: list[tuple[str, str]] = []
        self.runner = runner
        acts = {name: (lambda tick, name=name, act=act: self.seen.append(("node", name)) or act(tick))
                for name, act in loop.ACTS.items()}
        monkeypatch.setattr(loop, "ACTS", acts)

    def __call__(self, argv, stdin, cwd, timeout):
        self.seen.append(("call", ScriptedRunner.seat(argv)))
        return self.runner(argv, stdin, cwd, timeout)

    def ticks(self) -> list[list[tuple[str, str]]]:
        ticks: list[list[tuple[str, str]]] = []
        for item in self.seen:
            if item == ("node", "repair"):
                ticks.append([])
            ticks[-1].append(item)
        return ticks


def _check_ticks(tracer: Tracer, root: Root, task: Task) -> list[list[tuple[str, str]]]:
    """Each traced tick: repair first, save last when it runs, budget walked before every model call in it, the
    nodes walked exactly its `fired` event's `walked`."""
    ticks = tracer.ticks()
    fired = [e for e in state.read_log(root, task.id) if e["event"] == "fired"]
    assert [[name for kind, name in tick if kind == "node"] for tick in ticks] == [e["walked"] for e in fired]
    for tick in ticks:
        nodes = [name for kind, name in tick if kind == "node"]
        assert nodes[0] == "repair" and ("save" not in nodes or nodes[-1] == "save")
        for i, (kind, _) in enumerate(tick):
            if kind == "call":
                assert ("node", "budget") in tick[:i], tick
    return ticks


@pytest.mark.parametrize("scenario", ["think", "weaken", "branch"])
def test_on_the_dry_scenarios_budget_is_walked_before_every_model_call_of_its_tick(
    scenario: str, tmp_path: Path, root: Root, monkeypatch: pytest.MonkeyPatch
):
    """`think`: the thinker, an explore and a unit; `weaken`: a weakening reader call, made only in the gate, after the
    grader; `branch`: in-call admission, the reader inside the thinker's call."""
    root.path.mkdir(parents=True)
    for prompt in (root.thinker_prompt, root.editor_prompt, root.reader_prompt):
        prompt.write_text("p\n")
    tracer = Tracer(monkeypatch, scenario_runner(scenario))
    task, _ = dry.run_scenario(scenario, root, make_workspace(tmp_path / "ws"), tracer, lambda _: None)
    ticks = _check_ticks(tracer, root, task)
    readers = [tick for tick in ticks if ("call", "reader") in tick]
    if scenario == "weaken":
        assert readers and all(tick.index(("node", "grader")) < tick.index(("node", "gate")) <
                               tick.index(("call", "reader")) < tick.index(("node", "ref")) for tick in readers)
    if scenario == "branch":
        (admission,) = readers
        assert admission.index(("node", "director")) < admission.index(("call", "director")) < admission.index(
            ("call", "reader")) < admission.index(("node", "block"), admission.index(("node", "director")))


def _accepted_with_a_held_branch(tmp_path: Path, root: Root) -> Task:
    """u-1 accepted at the tip with a retained `stats.py`; in `hold`, an approved plan branching it, not admitted."""
    root.path.mkdir(parents=True)
    for prompt in (root.thinker_prompt, root.editor_prompt, root.reader_prompt):
        prompt.write_text("p\n")
    workspace = make_workspace(tmp_path / "ws")
    (workspace / "stats.py").write_text("def mean(xs):\n    return sum(xs) / len(xs)\n")
    git = ["git", "-c", "user.name=t", "-c", "user.email=t@t.invalid"]
    subprocess.run([*git, "add", "stats.py"], cwd=workspace, check=True)
    subprocess.run([*git, "commit", "-qm", "stats"], cwd=workspace, check=True)
    tip = subprocess.run(["git", "rev-parse", "HEAD"], cwd=workspace, capture_output=True, text=True).stdout.strip()
    u1 = Unit("u-1", "write stats.py", [Predicate("file_exists", {"path": "stats.py"})], status="passed",
              candidate=tip, integrated=True)
    child = Unit("u-2", "rename stats.py to metrics.py",
                 [Predicate("file_contains", {"path": "metrics.py", "text": "def mean"}, parents=["u-1 p1"])],
                 rationale="metrics.py keeps what stats.py promised")
    task = Task(id="task-20261026T120000", goal="g", workspace=str(workspace), units=[u1], accepted=tip, mode=HOLD,
                held=Held(kind=PLAN, units=[child], approved=True))
    subprocess.run(["git", "update-ref", f"refs/heads/task/{task.id}", tip], cwd=workspace, check=True)
    save(root, task)
    set_current(root, task.id)
    return task


def test_a_released_plan_s_reader_call_is_made_after_the_budget_node_in_its_tick(
    tmp_path: Path, root: Root, monkeypatch: pytest.MonkeyPatch
):
    task = _accepted_with_a_held_branch(tmp_path, root)
    yes = {"type": "result", "is_error": False, "session_id": "r", "total_cost_usd": 0.05,
           "result": json.dumps({"verdict": "yes", "note": "metrics.py carries mean()."})}
    rename = {"write": {"metrics.py": "def mean(xs):\n    return 0\n"}, "delete": ["stats.py"], "commit": True,
              "exit_code": 0, "result": "renamed", "total_cost_usd": 0.5}
    tracer = Tracer(monkeypatch, ScriptedRunner([], [rename], [yes]))
    loop.run_tick(task, Caps(), root, tracer)
    (tick,) = _check_ticks(tracer, root, task)
    assert tick.index(("node", "budget")) < tick.index(("node", "release")) < tick.index(("call", "reader")) < tick.index(
        ("node", "pick")) < tick.index(("call", "worker"))
    assert task.fired.fire == [Firing("worker", unit="u-2")]  # the released unit is the tick's firing


def test_without_its_nested_edge_the_gate_admits_a_branch_at_the_release_not_inside_the_thinker_s_call(
    tmp_path: Path, root: Root, monkeypatch: pytest.MonkeyPatch
):
    """One declared edge, `gate.within`: dropped, the thinker's call in `auto` gets no admit hook, so its branching
    plan is held unadmitted and the next tick's release asks the reader. What runs, and in which tick, changes."""
    def the_branch_scenario() -> tuple[list[int], list[int], list[str]]:
        scratch = tmp_path / f"run-{len(list(tmp_path.iterdir()))}"
        brain = Root(scratch / "brain")
        brain.path.mkdir(parents=True)
        for prompt in (brain.thinker_prompt, brain.editor_prompt, brain.reader_prompt):
            prompt.write_text("p\n")
        task = loop.run("branch", make_workspace(scratch / "ws"), brain, Caps(), scenario_runner("branch"),
                        unattended=0.0, notify=None)
        ticks = {seat: [c.tick for c in task.calls if c.seat == seat] for seat in ("director", "reader")}
        approved = [e["by"] for e in state.read_log(brain, task.id) if e["event"] == "approved"]
        assert [(u.id, u.status) for u in task.units] == [("u-1", "passed"), ("u-2", "passed"), ("u-3", "passed")]
        return ticks["director"], ticks["reader"], approved

    directors, readers, approved = the_branch_scenario()
    assert readers == [directors[1]] and approved == []  # inside the thinker's own tick, in its call
    monkeypatch.setattr(graph, "NODES", tuple(replace(node, within=()) if node.name == "gate" else node
                                              for node in graph.NODES))
    directors, readers, approved = the_branch_scenario()
    assert readers == [directors[1] + 1] and approved == ["mode"]  # a tick later, in the release


# --- the firing set a reply may carry ---


@pytest.fixture
def env(tmp_path: Path, root: Root) -> tuple[Root, Path]:
    root.path.mkdir(parents=True)
    root.thinker_prompt.write_text("m\n")
    root.editor_prompt.write_text("e\n")
    return root, make_workspace(tmp_path / "ws")


def _parks_after_two_invalid_replies(root: Root, workspace: Path, reply: dict) -> tuple[Task, list[str]]:
    runner = ScriptedRunner([reply, reply], [])
    task = loop.run("g", workspace, root, Caps(), runner, unattended=0.0, notify=None)
    seats = [ScriptedRunner.seat(argv) for argv, _, _, _ in runner.calls]
    (asked,) = open_questions(task)
    assert (seats, task.moves, asked.asker, asked.reason, task.terminal) == (
        ["director", "director"], [], "runtime", "invalid_plan", "interrupted")
    retry = runner.calls[1][1]
    return task, retry[len(runner.calls[0][1]):].splitlines()


STATE = {"standing": "One line.", "open": []}


@pytest.mark.parametrize("obj, error", [
    ({"ask": "which?", "task": [UNIT], "state": STATE}, "exactly one move of ideas, explore, task, ask, rest; it carried "
                                                       "['task', 'ask']"),
    ({"notice": "a line for the operator", "state": STATE}, "drop 'notice'"),  # undeclared: no such node in v1
    ({"reader": {"verdict": "yes"}, "state": STATE}, "'reader' is fired by the runtime, never by the thinker"),
    ({"grader": {"passed": True}, "state": STATE}, "'grader' is fired by the runtime, never by the thinker"),
    ({"rest": True, "save": True, "state": STATE}, "'save' is fired by the runtime, never by the thinker"),
], ids=["two firings", "an undeclared node", "the reader", "the grader", "save"])
def test_a_reply_firing_two_nodes_an_undeclared_one_or_a_runtime_node_comes_back_once_then_parks(
    env: tuple[Root, Path], obj: dict, error: str
):
    root, workspace = env
    reply = {"type": "result", "is_error": False, "session_id": "s-1", "total_cost_usd": 0.1, "result": json.dumps(obj)}
    _, retry = _parks_after_two_invalid_replies(root, workspace, reply)
    assert retry[2] == "Your previous answer was rejected:" and any(error in line for line in retry), retry


@pytest.mark.parametrize("derived, error", [
    (("rest", "console"), "a reply fires exactly one node in this version; it fired ['rest', 'console']"),
    (("notice",), "'notice' is no declared node"),
    (("reader",), "'reader' is fired by the runtime, never by the thinker"),
])
def test_the_decoded_firing_set_is_checked_against_the_registry_on_the_retry_path(
    env: tuple[Root, Path], monkeypatch: pytest.MonkeyPatch, derived: tuple, error: str
):
    """The wire keeps one move; `fire` is derived from its kind (`graph.FIRES`) and checked at decode. Were a kind to
    derive two firings, an undeclared node or a runtime node, the reply would be invalid like any other."""
    root, workspace = env
    monkeypatch.setitem(graph.FIRES, "rest", derived)
    _, retry = _parks_after_two_invalid_replies(root, workspace, _reply({"rest": True}))
    assert f"- {error}" in retry, retry


def test_decode_accepts_exactly_one_firing_of_each_move_and_the_move_keeps_it(env: tuple[Root, Path]):
    root, workspace = env
    task = Task(id="t", workspace=str(workspace))
    replies = {"ideas": [{"id": f"i{n}", "case": "c", "cost": "one explore", "serves": "o1"} for n in (1, 2, 3)],
               "explore": "try it", "task": [UNIT], "ask": "which?", "rest": True}
    for kind, value in replies.items():
        state_ = {"standing": "s.", "open": [{"id": "o1", "line": "which"}]}
        reply, errors = director.decode_move(json.dumps({kind: value, "state": state_}), task)
        assert errors == [] and reply.fire == list(graph.FIRES[kind]), kind
    loop.run("g", workspace, root, Caps(), ScriptedRunner([_reply({"task": [UNIT]}), _reply({"rest": True})], [PASS]),
             unattended=0.0, notify=None)
    saved = state.current(root)
    assert [(m.kind, m.fire) for m in saved.moves] == [("task", ["worker"]), ("rest", ["rest"])]


# --- in `hold`, approve releases the held firing set and think returns it ---


def _held(root: Root, workspace: Path, kind: str) -> tuple[Task, ScriptedRunner]:
    """A session in `hold` after the tick that held one firing: a task move's plan, a unit's retry or an explore."""
    if kind == "retry":
        retry = Unit("u-1", "write a.txt", [Predicate("file_exists", {"path": "a.txt"})], attempts=1)
        task = Task(id="task-20261026T130000", goal="g", workspace=str(workspace), mode=HOLD, units=[retry])
        runner = ScriptedRunner([_reply({"rest": True})], [PASS])
    else:
        task = Task(id="task-20261026T130000", goal="g", workspace=str(workspace), mode=HOLD)
        move = {"task": [UNIT]} if kind == "plan" else {"explore": "draft notes.md"}
        runner = ScriptedRunner([_reply(move), _reply({"rest": True})], [PASS if kind == "plan" else LANDED])
    save(root, task)
    set_current(root, task.id)
    loop.run_tick(task, Caps(), root, runner)
    assert task.held is not None and task.held.kind == kind
    return task, runner


@pytest.mark.parametrize("kind", ["plan", "retry", "explore"])
def test_in_hold_one_approve_releases_the_held_firing_set_and_it_runs(env: tuple[Root, Path], kind: str):
    root, workspace = env
    task, runner = _held(root, workspace, kind)
    console.apply(task, Caps(), "approve", {}, False)
    loop.run_tick(task, Caps(), root, runner)
    (firing,) = task.fired.fire
    assert firing.node == "worker" and task.held is None
    assert "worker" in task.fired.walked and runner.calls[-1][0] and ScriptedRunner.seat(runner.calls[-1][0]) == "worker"
    approved = [e for e in state.read_log(root, task.id) if e["event"] == "approved"]
    assert [(e["kind"], e["by"]) for e in approved] == [(kind, "approve")]


@pytest.mark.parametrize("kind", ["plan", "retry", "explore"])
def test_in_hold_a_think_returns_the_held_firing_set_and_none_of_it_runs(env: tuple[Root, Path], kind: str):
    root, workspace = env
    task, runner = _held(root, workspace, kind)
    workers = sum(ScriptedRunner.seat(argv) == "worker" for argv, _, _, _ in runner.calls)
    console.apply(task, Caps(), "think", {"text": "smaller"}, False)
    loop._settle(task, Caps(), root, lambda _: None)
    loop.run_tick(task, Caps(), root, runner)
    assert task.held is None and "worker" not in task.fired.walked and task.fired.walked[6] == "director"
    assert sum(ScriptedRunner.seat(argv) == "worker" for argv, _, _, _ in runner.calls) == workers
    returned = [e for e in state.read_log(root, task.id) if e["event"] == "returned"]
    assert [e["kind"] for e in returned] == [kind]


# --- each tick of `protean dry think` logs its firing set ---


def test_each_tick_of_the_think_scenario_logs_its_firing_set_and_the_record_keeps_the_latest(tmp_path: Path, root: Root):
    root.path.mkdir(parents=True)
    for prompt in (root.thinker_prompt, root.editor_prompt, root.reader_prompt):
        prompt.write_text("p\n")
    task, _ = dry.run_scenario("think", root, make_workspace(tmp_path / "ws"), scenario_runner("think"), lambda _: None)
    fired = [e for e in state.read_log(root, task.id) if e["event"] == "fired"]
    assert [e["tick"] for e in fired] == list(range(task.tick)) == [0, 1, 2, 3, 4, 5]
    assert [[(f["node"], f["move"], f["unit"]) for f in e["fire"]] for e in fired] == [
        [("console", "m1", None)], [], [("worker", "m2", None)], [], [("worker", "m3", "u-1")], [("rest", "m4", None)]]
    assert all(e["walked"][0] == "repair" and e["walked"][-1] == "save" for e in fired)
    assert state.load(root, task.id).fired == task.fired and asdict(task.fired) == {
        k: v for k, v in fired[-1].items() if k not in ("ts", "event")}
    assert console.snapshot(task, Caps(), live=False)["fired"] == asdict(task.fired)
