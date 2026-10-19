"""CLI: the dry scenarios' exit codes (each now rests or asks, and parks: 12), status on an empty and a live root,
the --extend grammar, `ctl` offline, `protean think --mode hold`. (`think`'s scratch repository, a session beside a
parked one, `resume <session>` and the retired `run` verb are `test_think.py`'s.)"""

from __future__ import annotations

import fcntl
import json
import os
import signal
import time
from pathlib import Path

import pytest

from protean import cli
from protean.dry import ScriptedRunner
from protean.state import Question, Root, Task, Thought, load, save, set_current


def test_dry_dry_rests_awaits_a_thought_and_exits_12(capsys: pytest.CaptureFixture[str]):
    assert cli.main(["dry", "dry"]) == 12
    lines = capsys.readouterr().out.strip().splitlines()
    assert lines[-3].startswith("tick 4: thinker m2 rest; awaiting a thought")
    assert lines[-2].startswith("waiting: a thought after m2") and "parks now" in lines[-2]
    assert lines[-1] == ("terminal: interrupted — awaiting a thought after m2: `protean ctl think <text>` adds one, "
                         "then `protean resume`")


def test_dry_stuck_ends_interrupted_and_exits_12(capsys: pytest.CaptureFixture[str]):
    assert cli.main(["dry", "stuck"]) == 12
    lines = capsys.readouterr().out.strip().splitlines()
    assert lines[-1] == ("terminal: interrupted — open question q1 (director): answer it with "
                         "`protean ctl answer q1 <text>`, then `protean resume`")
    assert lines[-2].startswith("waiting: question q1 (director)") and "parks now" in lines[-2]


def test_an_unknown_scenario_is_a_usage_error(capsys: pytest.CaptureFixture[str]):
    assert cli.main(["dry", "nope"]) == 2
    assert ("unknown scenario 'nope' (known: branch, deps, deps-blocked, deps-broken, deps-delivery-only, dry, stuck, "
            "think, think-rest, weaken)" in capsys.readouterr().err)


def test_status_on_an_empty_root_says_there_is_no_session(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    assert cli.main(["--brain", str(tmp_path / "empty"), "status"]) == 0
    assert capsys.readouterr().out.startswith("no session in ")


def test_status_shows_the_current_task(root: Root, task: Task, capsys: pytest.CaptureFixture[str]):
    task.cost_usd, task.terminal, task.stop_reason = 29.2, "stopped", "max_usd: 29.20 >= 29.0"
    task.extra_usd, task.active_seconds = 1.5, 450.0  # $2.50 accrued: the allowance is $29.00
    task.questions = [Question("q1", "director", "which file?", "question")]
    task.thoughts = [Thought("use tabs")]
    save(root, task)
    set_current(root, task.id)
    assert cli.main(["--brain", str(root.path), "status"]) == 0
    out = capsys.readouterr().out
    for piece in (task.id, "terminal: stopped — max_usd", "cost: $29.20 of $29.00",
                  "allowance: $25.00 + $1.50 extended + $2.50 accrued over 450s of active work", "u1", "pending",
                  "tick: 0 of 100", "think 1: use tabs", "question q1 (director): which file?"):
        assert piece in out


def test_resume_without_a_session_is_a_usage_error(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    assert cli.main(["--brain", str(tmp_path), "resume"]) == 2
    assert "no session in" in capsys.readouterr().err
    assert cli.main(["--brain", str(tmp_path), "resume", "--extend", "10m"]) == 2
    assert "no wall-clock ceiling" in capsys.readouterr().err


def test_resume_on_a_locked_task_is_a_usage_error(root: Root, task: Task, capsys: pytest.CaptureFixture[str]):
    task.terminal, task.questions = "interrupted", [Question("q1", "director", "q?", "question")]
    save(root, task)
    set_current(root, task.id)
    with (root.state_dir(task.id) / "lock").open("w") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        handle.write("4242\n")
        handle.flush()
        assert cli.main(["--brain", str(root.path), "resume"]) == 2
    assert f"task {task.id} is already running (pid 4242)" in capsys.readouterr().err


def test_extend_repeats_to_widen_ticks_and_dollars_together(root: Root, task: Task, monkeypatch: pytest.MonkeyPatch):
    task.tick, task.cost_usd, task.terminal, task.stop_reason = 100, 24.8, "stopped", "max_ticks: 100 >= 100"
    task.units[0].status = "failed"
    save(root, task)
    set_current(root, task.id)
    root.thinker_prompt.write_text("m\n")
    question = {"type": "result", "is_error": False, "session_id": "s", "total_cost_usd": 0.1,
                "result": json.dumps({"ask": "q?", "state": {"standing": "u1 failed.", "open": []}})}
    runner = ScriptedRunner([question], [])
    monkeypatch.setattr(cli, "real_runner", runner)
    assert cli.main(["--brain", str(root.path), "resume", "--extend", "5", "--extend", "$1"]) == 12
    saved = load(root, task.id)
    assert (saved.extra_ticks, saved.extra_usd, len(runner.calls)) == (5, 1.0, 1)


def _ctrl_c() -> None:
    raise KeyboardInterrupt


def _sigterm() -> None:
    os.kill(os.getpid(), signal.SIGTERM)
    time.sleep(1)


def _sighup() -> None:
    os.kill(os.getpid(), signal.SIGHUP)
    time.sleep(1)


@pytest.mark.parametrize("stop", [_ctrl_c, _sigterm, _sighup])
def test_a_ctrl_c_mid_call_exits_130_with_the_charge_on_disk(root: Root, task: Task, tmp_path: Path, stop,
                                                            capsys: pytest.CaptureFixture[str]):
    from protean.dry import make_workspace
    task.workspace = str(make_workspace(tmp_path / "ws"))
    task.terminal = "interrupted"
    task.questions = [Question("q1", "director", "q?", "question", status="answered", answer="go on", actor="console")]
    save(root, task)
    set_current(root, task.id)
    root.editor_prompt.write_text("e\n")
    def interrupt(argv, stdin, cwd, timeout):
        stop()
        raise AssertionError("the stop signal did not interrupt the call")
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(cli, "real_runner", interrupt)
    try:
        assert cli.main(["--brain", str(root.path), "resume"]) == 130
    finally:
        monkeypatch.undo()
    assert "booked" in capsys.readouterr().err
    saved = load(root, task.id)
    assert (saved.cost_usd, saved.units[0].attempts, saved.terminal) == (4.0, 1, None)


@pytest.mark.parametrize("text, parsed", [("5", (0.0, 5)), ("$2.5", (2.5, 0)), ("$3", (3.0, 0))])
def test_extend_takes_ticks_or_dollars(text: str, parsed: tuple[float, int]):
    assert cli.parse_extend(text) == parsed


@pytest.mark.parametrize("text", ["10m", "0", "$0", "five", "-3", "3$"])
def test_extend_refuses_anything_else(text: str):
    with pytest.raises(ValueError):
        cli.parse_extend(text)


def test_dry_deps_runs_the_needed_units_first_then_rests_and_exits_12(capsys: pytest.CaptureFixture[str]):
    assert cli.main(["dry", "deps"]) == 12
    lines = capsys.readouterr().out.strip().splitlines()
    assert [line.split(": ")[1].split(" ")[0] for line in lines if " attempt " in line] == ["u-1", "u-2", "u-3"]
    assert lines[-1].startswith("terminal: interrupted — awaiting a thought after m2")


@pytest.mark.parametrize("scenario, unit_line", [
    ("deps-broken", "tick 4: u-2 attempt 2: failed (broke u-1 p1)"),
    ("deps-blocked", "tick 3: u-1 attempt 2: failed (ungradeable p1 p2)"),
])
def test_dry_deps_failures_park_on_the_directors_question(scenario: str, unit_line: str,
                                                          capsys: pytest.CaptureFixture[str]):
    assert cli.main(["dry", scenario]) == 12
    lines = capsys.readouterr().out.strip().splitlines()
    assert any(line.startswith(unit_line) for line in lines), lines
    assert lines[-1].startswith("terminal: interrupted")


# --- `protean think --mode hold`: the first plan is held until `ctl approve`; `status` shows the mode and the hold ---


def test_think_in_hold_holds_the_first_plan_until_approve_and_status_shows_it(
    root: Root, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    from protean import state
    from protean.dry import make_workspace
    root.path.mkdir(parents=True)
    root.thinker_prompt.write_text("m\n")
    root.editor_prompt.write_text("e\n")
    workspace = make_workspace(tmp_path / "ws")
    plan = {"task": [{"id": "u-1", "intent": "write a.txt", "predicates": [
        {"kind": "file_exists", "args": {"path": "a.txt"}}]}], "state": {"standing": "a.txt is planned.", "open": []}}
    rest = {"rest": True, "state": {"standing": "a.txt is accepted.", "open": []}}
    runner = ScriptedRunner([{"type": "result", "is_error": False, "session_id": "s", "total_cost_usd": 0.1,
                              "result": json.dumps(reply)} for reply in (plan, rest)],
                            [{"write": {"a.txt": "A\n"}, "commit": True, "exit_code": 0, "result": "wrote a.txt",
                              "total_cost_usd": 0.5}])
    workers = lambda: sum("Write" in argv for argv, _, _, _ in runner.calls)  # noqa: E731
    monkeypatch.setattr(cli, "real_runner", runner)
    brain = ["--brain", str(root.path)]
    assert cli.build_parser().parse_args(["think", "g", "--workspace", "w"]).mode == "auto"

    assert cli.main([*brain, "think", "write a.txt", "--workspace", str(workspace), "--mode", "hold"]) == 12
    assert ("(mode hold: a task move's plan, each retry and each explore wait for approve)"
            in capsys.readouterr().out)
    task = state.current(root)
    assert (task.mode, task.held.kind, task.units, workers()) == ("hold", "plan", [], 0)
    (start,) = [e for e in state.read_log(root, task.id) if e["event"] == "start"]
    assert (start["thought"], start["mode"]) == ("write a.txt", "hold")
    assert cli.main([*brain, "ctl", "snapshot"]) == 0
    snap = json.loads(capsys.readouterr().out)
    assert (snap["mode"], snap["held"]["kind"], [u["id"] for u in snap["held"]["units"]]) == ("hold", "plan", ["u-1"])
    assert cli.main([*brain, "status"]) == 0
    out = capsys.readouterr().out
    assert "mode: hold\n" in out and "held: plan (u-1) — `protean ctl approve` releases it\n" in out

    assert cli.main([*brain, "resume"]) == 12 and workers() == 0  # still held: no worker call without approve
    capsys.readouterr()
    assert cli.main([*brain, "ctl", "approve"]) == 0
    assert cli.main([*brain, "status"]) == 0
    assert "held: plan (u-1) — approved, released at the next tick\n" in capsys.readouterr().out
    assert cli.main([*brain, "resume"]) == 12 and workers() == 1  # the unit ran, then the rest awaits a thought
    assert cli.main([*brain, "status"]) == 0
    out = capsys.readouterr().out
    assert "mode: hold\n" in out and "held:" not in out and "terminal: interrupted" in out


def test_main_installs_its_own_sigint_handler_so_a_backgrounded_brain_still_stops(tmp_path: Path):
    """A process started in the background inherits SIGINT ignored; `stop` and Ctrl-C both ride SIGINT, so the CLI
    takes the handler over, as it already did for SIGTERM and SIGHUP."""
    saved = {s: signal.getsignal(s) for s in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)}
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    try:
        assert cli.main(["--brain", str(tmp_path / "empty"), "status"]) == 0
        assert all(signal.getsignal(s) is cli._interrupt for s in saved)
    finally:
        for signum, handler in saved.items():
            signal.signal(signum, handler)
