"""The channel, the CLI and the wire of a thinking session, and the `think` and `think-rest`
fixtures (B6).

`think` is the operator's way in, live and offline (`protean ctl think <text> [--decide]`): each one lands in `thoughts`, ends a
wait for one, and with `decide` adds their decision to `decided`, verbatim, in the same save; no seat writes `decided`,
and no `steer` operation remains. `protean think` starts a session, on a fresh scratch repository under the brain when
no workspace is named; a parked session never refuses a new one, a live brain does; `protean resume [<session>]`
wakes the named session or the current one; `protean run` is no verb. The snapshot carries the thought record, the
moves, the thoughts and the wait. The `think` fixture runs ideas -> a console `think` -> explore -> task -> rest,
driven by its scripted client; `think-rest` parks unattended, and an offline `ctl think` plus `resume` reach the
thinker with that thought.
"""

from __future__ import annotations

import fcntl
import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

import pytest

from protean import cli, dry, state
from protean.dry import REPO, ScriptedRunner, make_workspace
from protean.graph import console, director
from protean.graph.budget import Caps
from protean.state import (EXPLORE, RUNTIME, Decision, Held, Idea, Move, OpenItem, Root, Task, Thought, add_question,
                           load, save, set_current)

TEMPLATE = REPO / "brain" / "protean"
SNAPSHOT_KEYS = {"task", "thought", "workspace", "project", "live", "tick", "terminal", "stop_reason", "paused", "mode",
                 "held", "awaiting_thought", "cost_usd", "ceilings", "prices", "units", "moves", "questions", "thoughts"}


def _reply(obj: dict, cost: float = 0.1) -> dict:
    return {"type": "result", "is_error": False, "session_id": "s-1", "total_cost_usd": cost, "result": json.dumps(obj)}


def _rest(standing: str) -> dict:
    return _reply({"rest": True, "state": {"standing": standing, "open": []}})


def _brain(root: Root) -> list[str]:
    """A brain root with the thinker's and the editor's prompts and no `config.json` (the caps' defaults, so a hold
    with no client parks at once); the CLI's `--brain` arguments for it."""
    root.path.mkdir(parents=True, exist_ok=True)
    root.thinker_prompt.write_text("t\n")
    root.editor_prompt.write_text("e\n")
    return ["--brain", str(root.path)]


def _seeded(root: Root) -> None:
    """The fixture brain's prompts, copied from the tracked brain (no `config.json`)."""
    root.path.mkdir(parents=True, exist_ok=True)
    for name in ("thinker.md", "editor.md", "reader.md"):
        shutil.copy2(TEMPLATE / name, root.path / name)


def _events(root: Root, task_id: str, name: str) -> list[dict]:
    return [e for e in state.read_log(root, task_id) if e["event"] == name]


def _git(cwd: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True)


def _messages(runner: ScriptedRunner) -> list[dict]:
    return [json.loads(stdin) for argv, stdin, _, _ in runner.calls if ScriptedRunner.seat(argv) == "director"]


@pytest.fixture
def ids(monkeypatch: pytest.MonkeyPatch):
    """Session ids one second apart, so a test may start several sessions in one root within a second."""
    minted = iter(f"task-20261017T15{n:02d}00" for n in range(60))
    monkeypatch.setattr(state, "new_task_id", lambda now=None: next(minted))


# --- the channel: `think` on the console, live and offline (decisions 9, 10) ---


def test_a_think_lands_a_thought_ends_the_wait_for_one_and_logs_its_event():
    task = Task("t", "Where should b.txt go?", "/w", awaiting_thought=True)
    event = console.apply(task, Caps(), "think", {"text": "  go with i1 "}, False, now="2026-10-17T10:00:00+00:00")
    assert event == {"event": "think", "actor": "console", "live": False, "text": "go with i1", "index": 1,
                     "decide": False}
    assert [asdict(t) for t in task.thoughts] == [{"text": "go with i1", "created": "2026-10-17T10:00:00+00:00",
                                                   "actor": "console"}]
    assert (task.awaiting_thought, task.decided) == (False, [])
    assert console.describe(event) == "console: think #1: go with i1"


def test_a_think_marked_decide_adds_the_operators_decision_verbatim_in_the_same_save_live_and_offline(
    root: Root, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    brain = _brain(root)
    task = Task(id="task-20261017T140000", thought="Where should b.txt go?", workspace="/w")
    save(root, task)
    set_current(root, task.id)
    said = "Put it at the root,\nbeside the README."
    with console.serve(root, task, Caps(), interrupt=lambda: None):
        status, payload = console.request(root, task.id, "POST", "/think", {"text": said, "decide": True})
        assert status == 200 and payload["event"] == {"event": "think", "actor": "console", "live": True,
                                                      "text": said, "index": 1, "decide": True}
        disk = load(root, task.id)
        assert [t.text for t in disk.thoughts] == [said]
        assert [asdict(d) for d in disk.decided] == [{"text": said, "created": disk.thoughts[0].created, "think": 1}]
        assert task.decided == disk.decided  # the loop's copy holds it too
        assert console.request(root, task.id, "POST", "/think", {"text": "keep it short"})[0] == 200
        assert len(load(root, task.id).decided) == 1  # a thought without `decide` is direction, never a decision

    real_write, written = state._write_atomic, []

    def write(path: Path, text: str) -> None:
        real_write(path, text)
        if path.name == "task.json":
            written.append(json.loads(text))

    monkeypatch.setattr(state, "_write_atomic", write)
    assert cli.main([*brain, "ctl", "think", "make", "it", "plain", "text", "--decide"]) == 0  # no brain live
    out = capsys.readouterr().out
    assert "console: think #3: make it plain text (decided)" in out and "written to task.json" in out
    (one,) = written  # one save carries the thought and the decision together
    assert [t["text"] for t in one["thoughts"]][-1] == "make it plain text"
    assert [(d["text"], d["think"]) for d in one["decided"]] == [(said, 1), ("make it plain text", 3)]
    saved = load(root, task.id)
    assert saved.decided[1].created == saved.thoughts[2].created
    assert [(e["live"], e["index"], e["decide"]) for e in _events(root, task.id, "think")] == [
        (True, 1, True), (True, 2, False), (False, 3, True)]


def test_think_bodies_are_checked_and_no_steer_operation_remains(root: Root, capsys: pytest.CaptureFixture[str]):
    brain = _brain(root)
    task = Task(id="task-20261017T140100", thought="t", workspace="/w")
    save(root, task)
    set_current(root, task.id)
    with console.serve(root, task, Caps(), interrupt=lambda: None):
        for body in ({}, {"text": "  "}, {"text": 3}, {"text": "x", "decide": "yes"}, {"text": "x", "decide": 1}):
            assert console.request(root, task.id, "POST", "/think", body)[0] == 400, body
        assert console.request(root, task.id, "POST", "/steer", {"text": "use tabs"}) == (
            404, {"ok": False, "error": "no endpoint /steer"})
    assert load(root, task.id).thoughts == [] and state.read_log(root, task.id) == []
    with pytest.raises(console.Refused) as refused:
        console.apply(task, Caps(), "steer", {"text": "use tabs"}, False)
    assert refused.value.status == 404 and "think" in console.OPS and "steer" not in console.OPS
    with pytest.raises(SystemExit) as exited:
        cli.main([*brain, "ctl", "steer", "use", "tabs"])
    assert exited.value.code == 2 and "invalid choice: 'steer'" in capsys.readouterr().err


def test_the_no_progress_question_is_released_only_by_yes(root: Root, capsys: pytest.CaptureFixture[str]):
    brain = _brain(root)
    task = Task(id="task-20261017T140200", thought="t", workspace="/w", terminal="interrupted")
    add_question(task, RUNTIME, "no_progress", "standing unchanged twice")
    add_question(task, RUNTIME, "no_progress", "again")
    save(root, task)
    set_current(root, task.id)
    with console.serve(root, task, Caps(), interrupt=lambda: None):
        status, payload = console.request(root, task.id, "POST", "/answer", {"id": "q2", "text": "go on"})
        assert status == 409 and "only `yes`" in payload["error"]
        assert console.request(root, task.id, "POST", "/answer", {"id": "q2", "text": "yes"})[0] == 200
    assert cli.main([*brain, "ctl", "answer", "q1", "keep", "going"]) == 2
    assert "only `yes`" in capsys.readouterr().err
    assert cli.main([*brain, "ctl", "answer", "q1", "yes"]) == 0
    assert [(q.status, q.answer) for q in load(root, task.id).questions] == [("answered", "yes"), ("answered", "yes")]


# --- the wire: the snapshot (decision 10) ---


def test_the_snapshot_carries_the_thought_record_the_moves_the_thoughts_and_the_wait():
    ideas = [Idea("i1", "at the root", "one explore", "o1"), Idea("i2", "under docs", "one explore", "o1"),
             Idea("i3", "nowhere", "nothing", "o1")]
    task = Task(
        "task-20261017T090000", "Where should b.txt go?", "/w", awaiting_thought=True,
        thoughts=[Thought("go with i1", "2026-10-17T09:01:00+00:00")], standing="b.txt is at the root.",
        open=[OpenItem("o1", "where b.txt goes")], decided=[Decision("at the root", "2026-10-17T09:01:00+00:00", 1)],
        moves=[Move("m1", "ideas", "ideas i1, i2, i3", 0.1, "proposed", ideas=ideas),
               Move("m2", "explore", "try the root", 0.5, "landed", branch="explore/task-20261017T090000/1",
                    brief="try the root", report="It fits."),
               Move("m3", "task", "task u-1", 0.6, "u-1 passed", branch="task/task-20261017T090000", units=["u-1"]),
               Move("m4", "ask", "which name?", 0.1, "answered", question="q1"),
               Move("m5", "rest", "rest", 0.1, "rested")],
        held=Held(kind=EXPLORE, move="m6", brief="try docs", directions=1))
    snap = console.snapshot(task, Caps(), live=False)
    assert set(snap) == SNAPSHOT_KEYS and "goal" not in snap and "directions" not in snap
    assert snap["thought"] == {"thought": "Where should b.txt go?", "standing": "b.txt is at the root.",
                               "open": [{"id": "o1", "line": "where b.txt goes"}],
                               "decided": [{"text": "at the root", "created": "2026-10-17T09:01:00+00:00", "think": 1}]}
    assert snap["thoughts"] == [{"text": "go with i1", "created": "2026-10-17T09:01:00+00:00", "actor": "console"}]
    assert snap["awaiting_thought"] is True
    assert snap["held"] == {"kind": "explore", "approved": False, "thoughts": 1, "move": "m6", "brief": "try docs"}
    assert snap["moves"] == [
        {"id": "m1", "kind": "ideas", "line": "ideas i1, i2, i3", "cost_usd": 0.1, "outcome": "proposed",
         "ideas": [asdict(idea) for idea in ideas]},
        {"id": "m2", "kind": "explore", "line": "try the root", "cost_usd": 0.5, "outcome": "landed",
         "branch": "explore/task-20261017T090000/1", "brief": "try the root", "report": "It fits.",
         "unverified": True},
        {"id": "m3", "kind": "task", "line": "task u-1", "cost_usd": 0.6, "outcome": "u-1 passed",
         "branch": "task/task-20261017T090000", "units": ["u-1"]},
        {"id": "m4", "kind": "ask", "line": "which name?", "cost_usd": 0.1, "outcome": "answered", "question": "q1"},
        {"id": "m5", "kind": "rest", "line": "rest", "cost_usd": 0.1, "outcome": "rested"}]


# --- the CLI (decision 11) ---


def test_protean_run_is_an_unknown_verb(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    with pytest.raises(SystemExit) as exited:
        cli.main(["--brain", str(tmp_path / "b"), "run", "make a.txt", "--workspace", str(tmp_path)])
    assert exited.value.code == 2 and "invalid choice: 'run'" in capsys.readouterr().err
    proc = subprocess.run([sys.executable, "-m", "protean.cli", "--brain", str(tmp_path / "b"), "run", "make a.txt",
                           "--workspace", str(tmp_path)], capture_output=True, text=True)
    assert proc.returncode == 2 and "invalid choice: 'run'" in proc.stderr and not (tmp_path / "b").exists()


def test_think_without_a_workspace_starts_on_a_fresh_scratch_repo_under_the_brain(
    root: Root, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    brain = _brain(root)
    runner = ScriptedRunner([_rest("Nothing is asked of the tree yet.")], [])
    monkeypatch.setattr(cli, "real_runner", runner)
    assert cli.main([*brain, "think", "What could this repo become?"]) == 12
    task = state.current(root)
    scratch = (root.path / "workspaces" / task.id).resolve()
    assert (Path(task.workspace), task.thought, task.project, task.mode) == (
        scratch, "What could this repo become?", "_root", "auto")
    assert _git(scratch, "rev-list", "--count", "HEAD").stdout.strip() == "1"  # one commit, and it is empty
    assert _git(scratch, "ls-tree", "-r", "HEAD").stdout == "" and _git(scratch, "status", "--porcelain").stdout == ""
    assert _git(scratch, "branch", "--show-current").stdout.strip() == "main"
    (start,) = _events(root, task.id, "start")
    assert {k: v for k, v in start.items() if k != "ts"} == {
        "event": "start", "thought": "What could this repo become?", "workspace": task.workspace, "mode": "auto"}
    argv = runner.calls[0][0]
    assert argv[argv.index("--add-dir") + 1] == task.workspace  # the thinker reads the scratch tree
    out = capsys.readouterr().out
    assert f"session {task.id}: What could this repo become?" in out and f"workspace: {scratch}" in out


def test_a_new_think_while_another_session_is_parked_is_not_refused_and_the_old_one_stays_as_it_was(
    root: Root, tmp_path: Path, ids, monkeypatch: pytest.MonkeyPatch
):
    brain = _brain(root)
    workspace = make_workspace(tmp_path / "ws")
    asks = _reply({"ask": "which file?", "state": {"standing": "A file is wanted.", "open": []}})
    runner = ScriptedRunner([asks, _rest("B rests."), _rest("C rests.")], [])
    monkeypatch.setattr(cli, "real_runner", runner)
    assert cli.main([*brain, "think", "first", "--workspace", str(workspace)]) == 12  # interrupted on its question
    first = state.current(root)
    asked = root.task_file(first.id).read_bytes()
    assert cli.main([*brain, "think", "second", "--workspace", str(workspace), "--mode", "hold"]) == 12
    second = state.current(root)
    assert (second.id != first.id, second.awaiting_thought, second.mode) == (True, True, "hold")
    rested = root.task_file(second.id).read_bytes()
    assert cli.main([*brain, "think", "third", "--project", "demo"]) == 12  # beside a session awaiting a thought
    third = state.current(root)
    assert third.id not in (first.id, second.id) and (third.thought, third.project) == ("third", "demo")
    assert root.task_file(first.id).read_bytes() == asked and root.task_file(second.id).read_bytes() == rested
    assert [q.status for q in load(root, first.id).questions] == ["open"]


def test_a_live_brain_refuses_a_second_think_and_a_resume_of_another_session(
    root: Root, capsys: pytest.CaptureFixture[str]
):
    brain = _brain(root)
    parked = Task(id="task-20261017T150000", thought="parked", workspace="/w", terminal="interrupted",
                  awaiting_thought=True)
    live = Task(id="task-20261017T150100", thought="live", workspace="/w")
    for task in (parked, live):
        save(root, task)
    set_current(root, live.id)
    with (root.state_dir(live.id) / "lock").open("w") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        handle.write("4242\n")
        handle.flush()
        for argv in (["think", "another thought"], ["resume", parked.id]):
            assert cli.main([*brain, *argv]) == 2, argv
            assert f"session {live.id} is live (pid 4242); a second brain is refused" in capsys.readouterr().err
    assert state.current(root).id == live.id and not (root.path / "workspaces").exists()
    assert sorted(os.listdir(root.path / "state")) == ["current", parked.id, live.id]


def test_resume_wakes_the_named_session_and_bare_resume_the_current_one(
    root: Root, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    brain = _brain(root)
    workspace = make_workspace(tmp_path / "ws")
    older = Task(id="task-20261017T160000", thought="thought A", workspace=str(workspace), terminal="interrupted",
                 standing="A stands.")
    current = Task(id="task-20261017T160100", thought="thought B", workspace=str(workspace), terminal="interrupted",
                   awaiting_thought=True, standing="B rests.")
    for task in (older, current):
        save(root, task)
    set_current(root, current.id)
    untouched = root.task_file(current.id).read_bytes()
    runner = ScriptedRunner([_rest("A rests."), _rest("A rests again.")], [])
    monkeypatch.setattr(cli, "real_runner", runner)
    assert cli.main([*brain, "resume", older.id]) == 12
    assert [m["thought"] for m in _messages(runner)] == ["thought A"] and state.current(root).id == older.id
    assert root.task_file(current.id).read_bytes() == untouched
    assert cli.main([*brain, "ctl", "think", "go", "on"]) == 0  # lands on the current session, now A
    assert cli.main([*brain, "resume"]) == 12
    assert [(m["thought"], m["thoughts"]) for m in _messages(runner)] == [("thought A", []), ("thought A", ["go on"])]
    capsys.readouterr()
    for name in ("task-20261017T999999", "../task-20261017T160100"):
        assert cli.main([*brain, "resume", name]) == 2
        assert f"no session {name} in " in capsys.readouterr().err
    assert cli.main([*brain, "resume", current.id, "--abandon"]) == 11
    assert (load(root, current.id).stop_reason, state.current(root).id) == ("abandoned", current.id)
    assert len(runner.calls) == 2


def test_status_shows_the_thought_its_record_the_moves_and_the_budget(root: Root, capsys: pytest.CaptureFixture[str]):
    task = Task("task-20261017T170000", "Where should b.txt go?", "/w", tick=3, cost_usd=1.25, terminal="interrupted",
                standing="b.txt is planned at the root.", open=[OpenItem("o1", "its name")],
                decided=[Decision("at the root", "2026-10-17T17:01:00+00:00", 1)],
                thoughts=[Thought("go with i1"), Thought("keep it short")], awaiting_thought=True,
                moves=[Move("m1", "ideas", "ideas i1, i2, i3", 0.1, "proposed"), Move("m2", "rest", "rest", 0.1,
                                                                                         "rested")])
    save(root, task)
    set_current(root, task.id)
    assert cli.main(["--brain", str(root.path), "status"]) == 0
    out = capsys.readouterr().out
    for piece in (f"session: {task.id} (_root)\n", "tick: 3 of 100\n", "cost: $1.25 of $25.00\n",
                  "allowance: $25.00 + $0.00 extended + $0.00 accrued over 0s of active work\n", "mode: auto\n",
                  "awaiting a thought after m2: `protean ctl think <text>` adds one\n",
                  "# Thought\n\nWhere should b.txt go?\n", "## Standing\n\nb.txt is planned at the root.\n",
                  "- o1: its name\n", "- think 1, 2026-10-17T17:01:00+00:00: at the root\n",
                  "- m1 · ideas · ideas i1, i2, i3 · $0.10 · proposed\n", "think 1: go with i1\n",
                  "think 2: keep it short\n"):
        assert piece in out, piece
    assert cli.main(["--brain", str(root.path / "empty"), "status"]) == 0
    assert capsys.readouterr().out.startswith("no session in ")


# --- `decided` is the operator's (decision 9) ---


WRITES_DECIDED = re.compile(r"\.decided\s*(?:=(?!=)|\+=|\.append|\.extend|\.insert)")


def test_decided_is_written_only_by_a_console_think_marked_decide_never_by_a_seat_or_the_loop():
    writers = [(path.relative_to(REPO / "src").as_posix(), line.strip())
               for path in sorted((REPO / "src").rglob("*.py"))
               for line in path.read_text(encoding="utf-8").splitlines() if WRITES_DECIDED.search(line)]
    assert [path for path, _ in writers] == ["protean/graph/console.py"], writers
    carried = '{"rest": true, "state": {"standing": "It rests.", "open": [], "decided": [{"text": "x"}]}}'
    move, errors = director.decode_move(carried, Task("t", "g", "/w"))
    assert move is None and errors  # a thinker reply that carries `decided` is a decode error


# --- the fixtures ---


class Refs:
    """A runner recording, before each call, which `task/` refs the workspace holds."""

    def __init__(self, scripted: ScriptedRunner, workspace: Path) -> None:
        self.scripted, self.workspace, self.seen = scripted, workspace, []

    def __call__(self, argv, stdin, cwd, timeout):
        refs = _git(self.workspace, "for-each-ref", "--format=%(refname:short)", "refs/heads/task/").stdout.split()
        self.seen.append((ScriptedRunner.seat(argv), refs))
        return self.scripted(argv, stdin, cwd, timeout)


def test_dry_think_driven_by_its_scripted_client_runs_ideas_a_think_an_explore_a_task_and_a_rest(
    root: Root, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    shim_log = tmp_path / "shim.log"
    monkeypatch.setenv("PROTEAN_SHIM_LOG", str(shim_log))
    _seeded(root)
    workspace = make_workspace(tmp_path / "ws")
    scripted = dry.scenario_runner("think")
    runner = Refs(scripted, workspace)
    task, client = dry.run_scenario("think", root, workspace, runner, echo=lambda _: None)
    explore = f"explore/{task.id}/1"

    # the moves, in order, the standing changed by each, and the think the client sent after the ideas
    assert [m.kind for m in task.moves] == ["ideas", "explore", "task", "rest"]
    standings = [e["standing"] for e in _events(root, task.id, "thought")]
    assert len(standings) == 4 and all(before != after for before, after in zip([""] + standings, standings))
    (think,) = _events(root, task.id, "think")
    text = "Go with i1: a hello.txt at the root."
    assert (think["actor"], think["live"], think["text"], think["index"], think["decide"]) == (
        "console", True, text, 1, False)
    assert [e["event"] for e in client.sent] == ["think"] and client.error is None
    order = [e["event"] for e in state.read_log(root, task.id)]
    assert order.index("ideas") < order.index("think") < order.index("explore")
    assert [m["thoughts"] for m in _messages(scripted)] == [[], [text], [text], [text]]
    assert len(task.moves[0].ideas) == 3 and task.decided == []  # a pick is direction, not a decision

    # the explore: its branch is in the workspace, no verdict names it, task/<session> unmoved by it
    assert _git(workspace, "rev-parse", "--verify", "--quiet", f"refs/heads/{explore}").returncode == 0
    found = _git(workspace, "rev-parse", explore).stdout.strip()
    assert all(explore not in json.dumps(v) and found not in json.dumps(v) for v in _events(root, task.id, "verdict"))
    assert runner.seen[:4] == [("director", []), ("director", []), ("worker", []), ("director", [])]
    assert _git(workspace, "merge-base", "--is-ancestor", found, f"task/{task.id}").returncode == 1
    assert "notes/hello-draft.md" not in _git(workspace, "ls-tree", "-r", "--name-only", f"task/{task.id}").stdout

    # the task: its unit graded and integrated on task/<session>
    (unit,) = task.units
    (verdict,) = _events(root, task.id, "verdict")
    assert (unit.id, unit.status, unit.integrated, verdict["passed"], verdict["unit"]) == ("u-1", "passed", True,
                                                                                          True, "u-1")
    assert _git(workspace, "rev-parse", f"task/{task.id}").stdout.strip() == unit.candidate == task.accepted
    assert [(e["unit"], e["accepted"]) for e in _events(root, task.id, "integrated")] == [("u-1", unit.candidate)]

    # the rest: nobody is attached any more, so the wait for a thought parks
    assert (task.terminal, task.awaiting_thought) == ("interrupted", True)
    assert [(e["hold"], e["on"]) for e in _events(root, task.id, "parked")] == [("thought", "m4")]

    # and the verb itself, on its scratch root
    assert cli.main(["dry", "think"]) == 12
    lines = capsys.readouterr().out.strip().splitlines()
    assert lines[0] == "dry think: scripted seats on a scratch root, driven by its scripted client"
    assert lines[-1] == ("terminal: interrupted — awaiting a thought after m4: `protean ctl think <text>` adds one, "
                         "then `protean resume`")
    assert not shim_log.exists()  # the shim recorded no call


def test_dry_think_rest_parks_unattended_then_an_offline_think_and_resume_reach_the_thinker(
    root: Root, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    shim_log = tmp_path / "shim.log"
    monkeypatch.setenv("PROTEAN_SHIM_LOG", str(shim_log))
    assert cli.main(["dry", "think-rest"]) == 12  # the verb: a single rest, nobody attached, parked at once
    lines = capsys.readouterr().out.strip().splitlines()
    assert lines[-3].startswith("tick 1: thinker m1 rest; awaiting a thought")
    assert lines[-1].startswith("terminal: interrupted — awaiting a thought after m1")

    # the same fixture's seats on a kept root, through the CLI verbs
    _seeded(root)
    brain = ["--brain", str(root.path)]
    runner = dry.scenario_runner("think-rest")
    monkeypatch.setattr(cli, "real_runner", runner)
    assert cli.main([*brain, "think", "dry scenario: think-rest"]) == 12
    task = state.current(root)
    assert (task.terminal, task.awaiting_thought, [m.kind for m in task.moves]) == ("interrupted", True, ["rest"])
    assert [(e["hold"], e["on"]) for e in _events(root, task.id, "parked")] == [("thought", "m1")]
    assert len(runner.calls) == 1
    capsys.readouterr()
    assert cli.main([*brain, "ctl", "think", "write", "a", "README", "first"]) == 0
    assert "(no brain is live: written to task.json; `protean resume` continues)" in capsys.readouterr().out
    assert cli.main([*brain, "resume"]) == 12
    first, second = _messages(runner)
    assert (first["thoughts"], second["thoughts"]) == ([], ["write a README first"])  # the thought reached the thinker
    task = load(root, task.id)
    assert [m.kind for m in task.moves] == ["rest", "rest"] and task.awaiting_thought
    assert [(e["hold"], e["on"]) for e in _events(root, task.id, "parked")] == [("thought", "m1"), ("thought", "m2")]
    assert not shim_log.exists()
