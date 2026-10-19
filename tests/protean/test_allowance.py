"""The allowance, dry. A session may spend
`max_usd` (25.0), plus the operator's extensions, plus `accrual_usd` (10.0) for every `accrual_seconds` (1800) of active work,
pro rata. Active work is the wall time a tick runs, added to `active_seconds` at the tick's save; time spent waiting at
a tick boundary or parked adds nothing (the battery's loop clock stands still unless a test moves it: `conftest.clock`).
Spend that reaches the allowance stops the session on its `ceiling` in the save that books the call, and nothing paid
runs until `extend` (or more active time) clears it. Every call's `--max-budget-usd` is the least of its per-call cap
and what the allowance has left; under the $0.50 floor of that, no call starts. The sitting of 2026-10-17 is gone."""

from __future__ import annotations

import itertools
import json
import subprocess
import time
from dataclasses import asdict
from pathlib import Path

import pytest

from protean import loop, state
from protean.cli import DEFAULT_BRAIN
from protean.dry import ScriptedRunner, make_workspace
from protean.graph import budget, console, director, worker
from protean.graph.budget import Caps, load_caps
from protean.state import HOLD, MOVE_KINDS, PLAN, Held, Predicate, Root, Task, Unit, from_json, load, save, set_current

UNIT = {"id": "u-1", "intent": "write a.txt", "predicates": [{"kind": "file_exists", "args": {"path": "a.txt"}}]}
BRIEF = "Try b.txt at the root and say whether it fits."
STANDINGS = itertools.count(1)


def _reply(obj: dict, cost: float = 0.1) -> dict:
    """A thinker reply: one move, with a standing no earlier reply had."""
    if "state" not in obj and any(kind in obj for kind in MOVE_KINDS):
        obj = {**obj, "state": {"standing": f"Standing {next(STANDINGS)}.", "open": []}}
    return {"type": "result", "is_error": False, "session_id": "s-1", "total_cost_usd": cost, "result": json.dumps(obj)}


def _work(path: str, cost: float) -> dict:
    """A worker entry (a unit's or an explore's): one file written and committed, `exit_code: 0`."""
    return {"write": {path: f"{path}\n"}, "commit": True, "exit_code": 0, "result": f"wrote {path}",
            "total_cost_usd": cost}


def _verdict(cost: float, **answer: str) -> dict:
    """A reader receipt whose answer is `answer`."""
    return {"type": "result", "is_error": False, "session_id": "r", "total_cost_usd": cost,
            "result": json.dumps(answer)}


def _cap(argv: list[str]) -> str:
    return argv[argv.index("--max-budget-usd") + 1]


def _argv_of(runner: ScriptedRunner, seat: str) -> list[str]:
    return [_cap(argv) for argv, _, _, _ in runner.calls if ScriptedRunner.seat(argv) == seat]


def _seats(runner: ScriptedRunner) -> list[str]:
    return [ScriptedRunner.seat(argv) for argv, _, _, _ in runner.calls]


def _events(root: Root, task_id: str, name: str) -> list[dict]:
    return [e for e in state.read_log(root, task_id) if e["event"] == name]


def _reached(task: Task, caps: Caps) -> bool:
    """The session's spend has reached its allowance."""
    return task.cost_usd >= budget.allowance(task, caps) - 1e-9


class Timed:
    """A scripted runner whose calls take `seconds[seat]` of the scripted loop clock each."""

    def __init__(self, inner: ScriptedRunner, clock, seconds: dict[str, float]) -> None:
        self.inner, self.clock, self.seconds = inner, clock, seconds

    def __call__(self, argv, stdin, cwd, timeout):
        self.clock.now += self.seconds.get(ScriptedRunner.seat(argv), 0.0)
        return self.inner(argv, stdin, cwd, timeout)


@pytest.fixture
def env(tmp_path: Path, root: Root) -> tuple[Root, Path]:
    root.path.mkdir(parents=True)
    for prompt in (root.thinker_prompt, root.editor_prompt, root.reader_prompt):
        prompt.write_text("p\n")
    return root, make_workspace(tmp_path / "ws")


@pytest.fixture
def saves(monkeypatch: pytest.MonkeyPatch) -> list[Task]:
    """Every record the loop saved (`state.save`), read back from disk right after the write."""
    written: list[Task] = []
    real = state.save

    def recording(root: Root, task: Task) -> None:
        real(root, task)
        written.append(load(root, task.id))

    monkeypatch.setattr(state, "save", recording)
    return written


def _parked(root: Root, workspace: Path, **fields: object) -> Task:
    """A saved current session parked before its first move."""
    task = Task(id="task-20261018T120000", thought="t", workspace=str(workspace), terminal="interrupted", **fields)
    save(root, task)
    set_current(root, task.id)
    return task


def _git(cwd: Path, *args: str) -> str:
    return subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t.invalid", *args], cwd=cwd,
                          capture_output=True, text=True, check=True).stdout.strip()


def _accepted(root: Root, workspace: Path, files: dict[str, str], predicates: list[Predicate]) -> Task:
    """A saved current session whose u-1 wrote `files`, committed at the workspace tip, and is accepted there with
    `predicates`."""
    for rel, text in files.items():
        (workspace / rel).parent.mkdir(parents=True, exist_ok=True)
        (workspace / rel).write_text(text)
    _git(workspace, "add", "-A")
    _git(workspace, "commit", "-qm", "u-1")
    tip = _git(workspace, "rev-parse", "HEAD")
    u1 = Unit("u-1", "write u-1's files", predicates, status="passed", candidate=tip, integrated=True)
    task = Task(id="task-20261018T090000", thought="t", workspace=str(workspace), units=[u1], accepted=tip)
    _git(workspace, "update-ref", worker.accepted_ref(task), tip)
    save(root, task)
    set_current(root, task.id)
    return task


# --- config ---


def test_the_allowance_starts_at_25_accrues_10_per_1800_active_seconds_with_100_ticks_and_sitting_usd_is_unknown(
    root: Root, clock
):
    shipped = load_caps(Root(DEFAULT_BRAIN))
    for caps in (Caps(), shipped):
        assert (caps.max_usd, caps.accrual_usd, caps.accrual_seconds, caps.max_ticks) == (25.0, 10.0, 1800, 100)
    assert not hasattr(Caps(), "sitting_usd") and clock.real is time.monotonic  # outside the battery: the wall clock
    root.path.mkdir(parents=True)
    root.config_file.write_text(json.dumps({"accrual_usd": 2.5, "accrual_seconds": 60}))
    assert (load_caps(root).accrual_usd, load_caps(root).accrual_seconds) == (2.5, 60)
    for bad in ({"sitting_usd": 5.0}, {"accrual_seconds": 0}, {"accrual_seconds": 1.5}, {"accrual_usd": "10"}):
        root.config_file.write_text(json.dumps(bad))
        with pytest.raises(ValueError):
            load_caps(root)
    root.config_file.write_text(json.dumps({"sitting_usd": 5.0}))
    with pytest.raises(ValueError, match="unknown key"):
        load_caps(root)


def test_the_allowance_is_max_usd_plus_the_extensions_plus_what_active_work_accrued_pro_rata():
    task = Task("t", "g", "/w", extra_usd=1.5, active_seconds=450.0)
    assert budget.accrued_usd(task, Caps()) == pytest.approx(2.5)  # a quarter of 1800 s: a quarter of $10, not $0
    assert budget.allowance(task, Caps()) == pytest.approx(25.0 + 1.5 + 2.5)
    assert budget.allowance(task, Caps(max_usd=1.0, accrual_usd=6.0, accrual_seconds=900)) == pytest.approx(5.5)


# --- active time: each tick's elapsed time at its save; nothing while waiting or parked ---


def test_active_seconds_grow_by_each_tick_s_time_at_its_save_and_not_while_waiting_or_parked(
    env: tuple[Root, Path], saves: list[Task], clock, monkeypatch: pytest.MonkeyPatch
):
    root, workspace = env
    waited: list[tuple] = []
    real_wait = console.Console.wait

    def slow_wait(self, hold, seen, line, unattended):  # every wait at a tick boundary lasts 5000 s
        waited.append(hold)
        clock.now += 5000
        return real_wait(self, hold, seen, line, unattended)

    monkeypatch.setattr(console.Console, "wait", slow_wait)
    scripted = ScriptedRunner([_reply({"task": [UNIT]}), _reply({"rest": True}), _reply({"rest": True})],
                              [_work("a.txt", 0.5)])
    runner = Timed(scripted, clock, {"director": 60.0, "worker": 240.0})
    task = loop.run("g", workspace, root, Caps(), runner, unattended=0.0, notify=None)
    assert (task.terminal, task.tick, task.active_seconds) == ("interrupted", 3, 360.0)  # 60 + 240 + 60
    clock.now += 7000  # parked
    loop.offline_op(root, Caps(), "think", {"text": "go on"})
    task = loop.resume(root, Caps(), runner, unattended=0.0, notify=None)
    assert (task.terminal, task.tick, task.active_seconds) == ("interrupted", 4, 420.0)
    assert waited == [("thought", "m2"), ("thought", "m3")] and len(_events(root, task.id, "parked")) == 2
    ticked = [(s.tick, s.active_seconds) for s in saves]
    assert sorted(set(ticked)) == [(0, 0.0), (1, 60.0), (2, 300.0), (3, 360.0), (4, 420.0)]
    for before, after in zip(saves, saves[1:]):  # time is added only in the save that counts a tick
        assert (after.active_seconds != before.active_seconds) == (after.tick != before.tick)
    ceilings = console.snapshot(task, Caps(), live=False)["ceilings"]
    assert (ceilings["active_seconds"], ceilings["accrued_usd"]) == (420.0, pytest.approx(10.0 * 420 / 1800))
    assert ceilings["allowance_usd"] == pytest.approx(25.0 + 10.0 * 420 / 1800)


def test_a_tick_cut_before_its_save_forfeits_its_time(env: tuple[Root, Path], clock):
    root, workspace = env
    scripted = ScriptedRunner([_reply({"task": [UNIT]})], [])

    def runner(argv, stdin, cwd, timeout):
        if ScriptedRunner.seat(argv) == "worker":
            clock.now += 100
            raise KeyboardInterrupt  # Ctrl-C mid-call: the call books its cap, the tick never reaches its save
        clock.now += 60
        return scripted(argv, stdin, cwd, timeout)

    with pytest.raises(KeyboardInterrupt):
        loop.run("g", workspace, root, Caps(), runner, unattended=0.0, notify=None)
    task = state.current(root)
    assert (task.tick, task.active_seconds, [c.outcome for c in task.calls]) == (1, 60.0, ["ok", "interrupted"])


# --- the stop: the save that books the crossing call carries it; nothing paid runs after ---

CROSSING = {  # the seat whose call reaches a $1 allowance: the thinker's own, a task unit's worker, an explore's worker
    "thinker": ([_reply({"task": [UNIT]}, 1.0)], [_work("a.txt", 0.5)], 1),
    "worker": ([_reply({"task": [UNIT]}, 0.4)], [_work("a.txt", 0.6)], 2),
    "explore": ([_reply({"explore": BRIEF}, 0.4)], [_work("b.txt", 0.6)], 2),
}


@pytest.mark.parametrize("seat", list(CROSSING))
def test_spend_reaching_the_allowance_stops_the_session_on_its_ceiling_in_the_booking_save_and_nothing_paid_follows(
    env: tuple[Root, Path], saves: list[Task], seat: str
):
    root, workspace = env
    thinker, work, calls = CROSSING[seat]
    caps, runner = Caps(max_usd=1.0), ScriptedRunner(thinker + [_reply({"rest": True})], work)
    task = loop.run("g", workspace, root, caps, runner, unattended=0.0, notify=None)
    assert len(runner.calls) == calls  # the crossing call was the last one made
    assert (task.terminal, task.stop_reason, task.questions) == ("stopped", "max_usd: 1.00 >= 1.0", [])
    crossed = next(record for record in saves if _reached(record, caps))  # the first save at the allowance
    assert (crossed.terminal, crossed.stop_reason) == ("stopped", "max_usd: 1.00 >= 1.0")
    assert crossed.cost_usd == pytest.approx(1.0) and len(crossed.calls) == calls and crossed.calls[-1].booked
    names = [(e["event"], e.get("seat") or e.get("reason")) for e in state.read_log(root, task.id)]
    assert names.index(("stopped", "max_usd: 1.00 >= 1.0")) == names.index(("call", crossed.calls[-1].seat)) + 1
    parked = _events(root, task.id, "parked")[-1]
    assert (parked["hold"], parked["on"]) == ("ceiling", "max_usd: 1.00 >= 1.0")
    lines: list[str] = []
    again = loop.resume(root, caps, runner, echo=lines.append)  # resume alone: still stopped, no paid call
    assert (again.terminal, len(runner.calls)) == ("stopped", calls) and "protean ctl extend" in "\n".join(lines)


@pytest.mark.parametrize("seat", list(CROSSING))
def test_with_accrued_time_the_same_spend_does_not_stop_the_session(env: tuple[Root, Path], saves: list[Task],
                                                                     seat: str):
    root, workspace = env
    thinker, work, calls = CROSSING[seat]
    caps, runner = Caps(max_usd=1.0), ScriptedRunner(thinker + [_reply({"rest": True})], work)
    _parked(root, workspace, active_seconds=180.0)  # 180 s of active work accrued $1.00: the allowance is $2.00
    task = loop.resume(root, caps, runner, unattended=0.0, notify=None)
    assert (task.terminal, task.awaiting_thought, task.stop_reason) == ("interrupted", True, "")  # parked on a thought
    assert _events(root, task.id, "stopped") == [] and len(runner.calls) == 3  # the crossing call, then more
    booked = next(record for record in saves if len(record.calls) == calls and record.calls[-1].booked)
    assert booked.cost_usd == pytest.approx(1.0) and booked.terminal is None  # the same spend, booked, no stop
    assert budget.allowance(booked, caps) == pytest.approx(2.0)


def test_a_crossing_tick_s_own_time_accrues_at_its_save_and_the_boundary_lifts_the_stop(
    env: tuple[Root, Path], clock
):
    """The booking save reads the allowance the tick started under; the tick's own time is added at its save, so a
    tick that worked long enough clears the stop it hit, and the session goes on by itself (`released`)."""
    root, workspace = env
    scripted = ScriptedRunner([_reply({"task": [UNIT]}, 0.4), _reply({"rest": True})], [_work("a.txt", 0.6)])
    runner = Timed(scripted, clock, {"worker": 360.0})  # $2.00 of active work in the crossing tick
    task = loop.run("g", workspace, root, Caps(max_usd=1.0), runner, unattended=0.0, notify=None)
    assert [e["reason"] for e in _events(root, task.id, "stopped")] == ["max_usd: 1.00 >= 1.0"]
    assert [e["hold"] for e in _events(root, task.id, "released")] == ["ceiling"]
    assert (task.terminal, task.awaiting_thought, _seats(scripted)) == ("interrupted", True,
                                                                        ["director", "worker", "director"])


def test_extend_releases_the_stop_logs_x_and_the_resulting_allowance_and_the_thinker_sees_the_allowance_left(
    env: tuple[Root, Path]
):
    root, workspace = env
    caps = Caps(max_usd=1.0)
    runner = ScriptedRunner([_reply({"task": [UNIT]}, 0.4), _reply({"rest": True})], [_work("a.txt", 0.6)])
    task = loop.run("g", workspace, root, caps, runner, unattended=0.0, notify=None)
    assert (task.terminal, len(runner.calls)) == ("stopped", 2)
    event = loop.offline_op(root, caps, "extend", {"usd": 2.0})
    assert (event["usd"], event["ticks"], event["max_usd"], event["allowance_usd"]) == (2.0, 0, 3.0, 3.0)
    assert "released" not in event and "sitting_usd" not in event
    assert console.describe(event).endswith("allowance $3.00")
    task = loop.resume(root, caps, runner, unattended=0.0, notify=None)
    assert (task.terminal, task.awaiting_thought, len(runner.calls)) == ("interrupted", True, 3)
    assert [e["hold"] for e in _events(root, task.id, "released")] == ["ceiling"]
    messages = [json.loads(stdin) for argv, stdin, _, _ in runner.calls if ScriptedRunner.seat(argv) == "director"]
    assert [m["budget_left_usd"] for m in messages] == [1.0, 2.0]  # the allowance less the cost, each call
    assert not any("sitting" in key for m in messages for key in m)
    ceilings = console.snapshot(task, caps, live=False)["ceilings"]
    assert {k: ceilings[k] for k in ("max_usd", "extra_usd", "accrued_usd", "allowance_usd", "active_seconds",
                                     "max_ticks")} == {"max_usd": 3.0, "extra_usd": 2.0, "accrued_usd": 0.0,
                                                       "allowance_usd": 3.0, "active_seconds": 0.0, "max_ticks": 100}


# --- every call's cap: the least of its per-call cap and what the allowance has left ---


@pytest.mark.parametrize("caps, thinker_seconds, worker_cap", [(Caps(), 0.0, "4.00"), (Caps(max_usd=2.0), 0.0, "1.90"),
                                                                (Caps(max_usd=2.0), 90.0, "2.40")],
                         ids=["per-call", "allowance", "accrued"])
def test_a_task_unit_s_worker_cap_is_the_least_of_its_per_call_cap_and_the_allowance_s_left(
    env: tuple[Root, Path], clock, caps: Caps, thinker_seconds: float, worker_cap: str
):
    root, workspace = env
    scripted = ScriptedRunner([_reply({"task": [UNIT]}, 0.1), _reply({"rest": True})], [_work("a.txt", 0.5)])
    task = loop.run("g", workspace, root, caps, Timed(scripted, clock, {"director": thinker_seconds}), unattended=0.0,
                    notify=None)
    assert _argv_of(scripted, "worker") == [worker_cap]  # "accrued": the first tick's 90 s added $0.50 before it
    assert next(c.cap_usd for c in task.calls if c.seat == "worker") == float(worker_cap)


def test_an_explore_s_cap_and_the_thinker_s_are_clipped_to_the_allowance_s_left_accrual_included(
    env: tuple[Root, Path], clock
):
    root, workspace = env
    scripted = ScriptedRunner([_reply({"explore": BRIEF}, 0.1), _reply({"rest": True})], [_work("b.txt", 0.2)])
    loop.run("g", workspace, root, Caps(max_usd=1.0), Timed(scripted, clock, {"director": 90.0}), unattended=0.0,
             notify=None)
    # the first thinker call: $1.00 left; its tick accrued $0.50, so the explore has $1.40, then the thinker $1.20
    assert (_argv_of(scripted, "director"), _argv_of(scripted, "worker")) == (["1.00", "1.20"], ["1.40"])


def test_a_reader_s_cap_is_clipped_to_the_allowance_s_left_after_the_thinker_s_unbooked_spend(env: tuple[Root, Path]):
    root, workspace = env
    stats = {"id": "u-1", "intent": "write stats.py",
             "predicates": [{"kind": "file_exists", "args": {"path": "stats.py"}}]}
    branch = {"id": "u-2", "intent": "rename stats.py to metrics.py", "rationale": "the module moves whole",
              "predicates": [{"kind": "file_contains", "args": {"path": "metrics.py", "text": "def mean"},
                              "parents": ["u-1 p1"]}]}
    rename = {"write": {"metrics.py": "def mean(xs):\n    return sum(xs) / len(xs)\n"}, "delete": ["stats.py"],
              "commit": True, "exit_code": 0, "result": "renamed", "total_cost_usd": 0.3}
    runner = ScriptedRunner([_reply({"task": [stats]}, 0.1), _reply({"task": [branch]}, 0.1), _reply({"rest": True})],
                            [_work("stats.py", 0.5), rename], [_verdict(0.05, verdict="yes", note="carried whole")])
    task = loop.run("g", workspace, root, Caps(max_usd=1.5), runner, unattended=0.0, notify=None)
    # $1.50 less the thinker's 0.10, u-1's 0.50 and the second thinker call's unbooked 0.10; then less the reader's 0.05
    assert (_argv_of(runner, "reader"), _argv_of(runner, "worker")) == (["0.80"], ["1.40", "0.75"])
    assert (task.terminal, task.stop_reason) == ("stopped", "max_usd: remaining $0.45 below the $0.50 call floor")
    assert len(runner.director) == 1  # the rest was never asked for


@pytest.mark.parametrize("active, called", [(0.0, []), (36.0, ["0.60"])], ids=["under-the-floor", "accrued-past-it"])
def test_under_the_call_floor_of_the_allowance_s_left_no_call_starts_and_accrued_time_counts(
    env: tuple[Root, Path], active: float, called: list[str]
):
    root, workspace = env
    _parked(root, workspace, cost_usd=0.6, active_seconds=active)  # $0.40 left, or $0.60 with 36 s accrued ($0.20)
    runner = ScriptedRunner([_reply({"rest": True}, 0.1)], [])
    task = loop.resume(root, Caps(max_usd=1.0), runner, unattended=0.0, notify=None)
    assert _argv_of(runner, "director") == called
    if not called:
        assert (task.terminal, task.stop_reason) == ("stopped", "max_usd: remaining $0.40 below the $0.50 call floor")


# --- a reader's spend reaching the allowance: the gate books it in its own save, the loop stops before any call ---


def test_a_released_plan_s_reader_reaching_the_allowance_stops_the_session_before_its_worker_runs(
    env: tuple[Root, Path], saves: list[Task]
):
    root, workspace = env
    caps, mean = Caps(max_usd=1.0), "def mean(xs):\n    return sum(xs) / len(xs)\n"
    task = _accepted(root, workspace, {"stats.py": mean}, [Predicate("file_exists", {"path": "stats.py"})])
    rename = Unit("u-2", "rename stats.py to metrics.py",
                  [Predicate("file_contains", {"path": "metrics.py", "text": "def mean"}, parents=["u-1 p1"])],
                  rationale="the module moves whole")
    task.mode, task.held = HOLD, Held(kind=PLAN, units=[rename], approved=True)  # released: its reader may spend $1.00
    save(root, task)
    renamed = {"write": {"metrics.py": mean}, "delete": ["stats.py"], "commit": True, "exit_code": 0,
               "result": "renamed", "total_cost_usd": 0.3}
    runner = ScriptedRunner([_reply({"rest": True})], [renamed], [_verdict(1.0, verdict="yes", note="carried whole")])
    task = loop.resume(root, caps, runner, unattended=0.0, notify=None)
    assert _seats(runner) == ["reader"] and _cap(runner.calls[0][0]) == "1.00"  # no worker call after it
    assert (task.terminal, task.stop_reason) == ("stopped", "max_usd: 1.00 >= 1.0")
    crossed = next(record for record in saves if _reached(record, caps))
    assert (crossed.calls[-1].seat, crossed.terminal) == ("reader", None)  # the gate's own booking save
    stopped = next(record for record in saves if record.terminal == "stopped")  # the release step's ceiling save
    assert len(stopped.calls) == len(crossed.calls) == 1 and stopped.tick == 0  # nothing launched between them
    assert [(u.id, u.attempts) for u in task.units] == [("u-1", 0), ("u-2", 0)]


def test_a_weakening_reader_reaching_the_allowance_stops_the_session_at_the_next_ceiling_step(
    env: tuple[Root, Path]
):
    root, workspace = env
    caps = Caps(max_usd=1.1)  # the worker's $0.50 leaves $0.60: the reader's cap, and what it spends
    checks = Predicate("command", {"cmd": "true"}, protected=["tests"])
    task = _accepted(root, workspace, {"a.txt": "a\n", "tests/check.txt": "a\n"},
                     [Predicate("file_exists", {"path": "a.txt"}), checks])
    task.units.append(Unit("u-2", "write b.txt", [Predicate("file_exists", {"path": "b.txt"})]))
    save(root, task)
    touches = {"write": {"b.txt": "b\n", "tests/check.txt": "a\nb\n"}, "commit": True, "exit_code": 0,
               "result": "wrote b.txt", "total_cost_usd": 0.5}
    runner = ScriptedRunner([_reply({"rest": True})], [touches],
                            [_verdict(0.6, verdict="no", kind="strengthened", note="check.txt gains a line")])
    task = loop.resume(root, caps, runner, unattended=0.0, notify=None)
    assert _seats(runner) == ["worker", "reader"]  # no call after it
    assert _cap(runner.calls[1][0]) == "0.60" and (task.units[1].status, task.units[1].integrated) == ("passed", True)
    assert (task.terminal, task.stop_reason) == ("stopped", "max_usd: 1.10 >= 1.1")


# --- booking on the new seats: the record is in task.json before the runner runs ---


def test_a_thinker_call_and_an_explore_call_are_each_in_task_json_pending_before_the_runner_runs(
    env: tuple[Root, Path]
):
    root, workspace = env
    scripted = ScriptedRunner([_reply({"explore": BRIEF}, 0.1), _reply({"rest": True})], [_work("b.txt", 0.5)])
    seen: list[tuple] = []

    def runner(argv, stdin, cwd, timeout):
        call = load(root, state.current(root).id).calls[-1]
        seen.append((call.seat, call.outcome, call.booked, call.cap_usd == float(_cap(argv)), call.charge_usd))
        return scripted(argv, stdin, cwd, timeout)

    task = loop.run("g", workspace, root, Caps(), runner, unattended=0.0, notify=None)
    assert seen == [("director", "pending", False, True, None), ("worker", "pending", False, True, None),
                    ("director", "pending", False, True, None)]
    assert [(c.seat, c.unit, c.booked) for c in task.calls] == [("director", None, True), ("worker", None, True),
                                                                 ("director", None, True)]
    assert director.pending_notes(root, task.id) == []


# --- older records ---


def test_a_2026_10_17_record_carrying_the_sitting_keys_loads_with_them_ignored(env: tuple[Root, Path]):
    root, workspace = env
    record = json.loads(state.to_json(Task("task-20261017T200000", "t", str(workspace), cost_usd=3.0)))
    del record["active_seconds"]
    record.update(sitting_start_usd=1.0, sitting_extra_usd=2.0)  # what order 4 wrote, 2026-10-17
    task = from_json(json.dumps(record))
    assert task == Task("task-20261017T200000", "t", str(workspace), cost_usd=3.0) and task.active_seconds == 0.0
    assert not any(key.startswith("sitting") for key in asdict(task))
    assert not any(key.startswith("sitting") for key in json.loads(state.to_json(task)))
    assert budget.allowance(task, Caps()) == 25.0
