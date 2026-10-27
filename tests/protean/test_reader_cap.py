"""A reader call that reaches its dollar cap refuses nothing on its own (run 6: u-2 passed every check twice and was
rejected both times on the weakening reader's cost alone).

The first cap hit is asked again once, on the same brief, under twice `reader_call_usd` (clipped by what the allowance
has left, as every call is), and that verdict is applied as the first one's would have been. A second cap hit, or no
room under the call floor for the wider call, holds the attempt (the weakening check) or the plan (branch admission) on
a runtime `reader_cap` question, saved with the booking: `yes` asks once more under the wider cap (a cap hit there opens
the question again), `no` applies today's cap-hit path exactly, and neither `resume` nor `extend` alone releases it.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from protean import loop, state
from protean.dry import ScriptedRunner, make_workspace
from protean.graph import console, gate, questions
from protean.graph.budget import Caps
from protean.state import HOLD, PLAN, RUNTIME, Held, Predicate, Root, Task, Unit, load, save, set_current

TASK_ID = "task-20261018T120000"
CAPPED = {"type": "result", "is_error": True, "subtype": "error_max_budget_usd", "session_id": "dry-reader",
          "total_cost_usd": 1.52, "result": ""}
CAPPED_WIDE = {**CAPPED, "total_cost_usd": 3.04}
NOT_WEAKENED = {"type": "result", "is_error": False, "session_id": "dry-reader", "total_cost_usd": 0.05,
                "result": json.dumps({"verdict": "no", "kind": "legitimate refactor", "note": "t.txt reworded"})}
YES = {"type": "result", "is_error": False, "session_id": "dry-reader", "total_cost_usd": 0.05,
       "result": json.dumps({"verdict": "yes", "note": "metrics.py carries mean()."})}
W_TOUCH = {"write": {"tests/t.txt": "two\n", "b.txt": "b\n"}, "commit": True, "exit_code": 0,
           "result": "wrote b.txt, reworded tests/t.txt", "total_cost_usd": 0.5}
W_UNTOUCHED = {"write": {"b.txt": "b\n"}, "commit": True, "exit_code": 0, "result": "wrote b.txt",
               "total_cost_usd": 0.5}
REST = {"type": "result", "is_error": False, "session_id": "dry-director", "total_cost_usd": 0.1,
        "result": json.dumps({"rest": True, "state": {"standing": "b.txt is accepted.", "open": []}})}
MEAN = "def mean(xs):\n    return sum(xs) / len(xs)\n"
W_RENAME = {"write": {"metrics.py": MEAN}, "delete": ["stats.py"], "commit": True, "exit_code": 0,
            "result": "renamed stats.py to metrics.py", "total_cost_usd": 0.5}
CAP_NOTE = "weakening check could not run: the reader call reached its $3.00 cap"


def _git(cwd: Path, *args: str) -> str:
    return subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t.invalid", *args], cwd=cwd,
                          capture_output=True, text=True, check=True).stdout.strip()


def _events(root: Root, name: str) -> list[dict]:
    return [e for e in state.read_log(root, TASK_ID) if e["event"] == name]


def _seats(runner: ScriptedRunner) -> list[str]:
    return [ScriptedRunner.seat(argv) for argv, _, _, _ in runner.calls]


def _caps(runner: ScriptedRunner) -> list[str]:
    """The `--max-budget-usd` each reader call carried."""
    return [argv[argv.index("--max-budget-usd") + 1] for argv, _, _, _ in runner.calls
            if ScriptedRunner.seat(argv) == "reader"]


def _seed(tmp_path: Path, root: Root, *files: tuple[str, str]) -> tuple[Path, str]:
    root.path.mkdir(parents=True)
    for prompt in (root.thinker_prompt, root.editor_prompt, root.reader_prompt):
        prompt.write_text("p\n")
    workspace = make_workspace(tmp_path / "ws")
    for rel, text in files:
        (workspace / rel).parent.mkdir(parents=True, exist_ok=True)
        (workspace / rel).write_text(text)
    _git(workspace, "add", "-A")
    _git(workspace, "commit", "-qm", "accepted")
    return workspace, _git(workspace, "rev-parse", "HEAD")


@pytest.fixture
def guarded(tmp_path: Path, root: Root) -> tuple[Root, Task]:
    """u-1 accepted at the tip, its retained command protecting `tests`; u-2 pending, whose editor rewords
    tests/t.txt, so its candidate touches a protected path and the weakening reader is called."""
    workspace, tip = _seed(tmp_path, root, ("tests/t.txt", "one\n"))
    u1 = Unit("u-1", "write tests/t.txt", [Predicate("command", {"cmd": "test -f tests/t.txt"}, protected=["tests"])],
              status="passed", candidate=tip, integrated=True)
    u2 = Unit("u-2", "write b.txt", [Predicate("file_exists", {"path": "b.txt"})])
    task = Task(id=TASK_ID, goal="g", workspace=str(workspace), units=[u1, u2], accepted=tip)
    _git(workspace, "update-ref", f"refs/heads/task/{TASK_ID}", tip)
    save(root, task)
    set_current(root, task.id)
    return root, task


# --- (a) the first cap hit is asked again once, under twice the cap, and its verdict applies ---


def test_a_weakening_reader_at_its_cap_is_asked_once_more_under_twice_the_cap_and_that_verdict_applies(
    guarded: tuple[Root, Task]
):
    root, task = guarded
    scripted, seen = ScriptedRunner([], [W_TOUCH], [CAPPED, NOT_WEAKENED]), []

    def runner(argv, stdin, cwd, timeout):
        if ScriptedRunner.seat(argv) == "reader":  # what task.json held when each reader call launched
            seen.append((stdin, [(c.id, c.outcome, c.booked) for c in load(root, TASK_ID).calls]))
        return scripted(argv, stdin, cwd, timeout)

    loop.run_tick(task, Caps(), root, runner)
    assert _caps(scripted) == ["1.50", "3.00"] and seen[0][0] == seen[1][0]  # the same brief, twice the cap
    assert seen[0][1][-1] == ("c2", "pending", False)  # recorded before it launched
    assert seen[1][1][-2:] == [("c2", "error", True), ("c3", "pending", False)]  # booked before the re-ask launched
    u2 = task.units[1]
    assert (u2.status, u2.attempts, u2.integrated) == ("passed", 1, True)  # not rejected on the first cap hit
    calls = [e for e in _events(root, "call") if e["seat"] == "reader"]
    assert [(e["id"], e["attempt"], e["cap_usd"], e["charge_usd"], e["booked"], e.get("retry_of")) for e in calls] == [
        ("c2", 1, 1.5, 1.52, True, None), ("c3", 1, 3.0, 0.05, True, "c2")]
    (verdict,) = [v for v in _events(root, "verdict") if v["unit"] == "u-2"]
    assert verdict["weakening"] == {"protected": ["tests"], "touched": ["tests/t.txt"], "result": "not_weakened",
                                    "checks": ["u-1 p1"], "reason": "no", "kind": "legitimate refactor",
                                    "note": "t.txt reworded", "call": "c3"}
    assert task.cost_usd == pytest.approx(0.5 + 1.52 + 0.05) and state.open_questions(task) == []


# --- (b)-(d) a second cap hit holds the attempt on `reader_cap` until the operator answers ---


@pytest.mark.parametrize("answer", ["yes", "no"])
def test_a_weakening_reader_at_its_cap_twice_holds_the_attempt_until_the_operator_answers(
    guarded: tuple[Root, Task], answer: str
):
    root, task = guarded
    # the reader's own budget (2026-10-20) is widened here so two re-asks under $3.00 fit; `test_the_admission_re_ask…`
    # pins the default $7.00 clipping them
    caps = Caps(reader_max_usd=12.0)
    readers = [CAPPED, CAPPED_WIDE] + ([CAPPED_WIDE, NOT_WEAKENED] if answer == "yes" else [])
    workers = [W_TOUCH] + ([] if answer == "yes" else [W_UNTOUCHED])
    runner = ScriptedRunner([REST], workers, readers)
    loop.run_tick(task, caps, root, runner)

    # (b) the recorded save: the question opened with the booking, the attempt neither graded nor rejected
    saved = load(root, TASK_ID)
    (held,) = state.open_questions(saved)
    assert (held.asker, held.reason) == (RUNTIME, "reader_cap")
    info = held.context["reader_cap"]
    assert (info["check"], info["unit"], info["attempt"], info["calls"], info["cost_usd"]) == (
        "weakening", "u-2", 1, ["c2", "c3"], pytest.approx(4.56))
    for named in ("u-2", "weakening", "$1.50", "$3.00", "$4.56", "yes", "no"):
        assert named in held.text, named
    u2 = saved.units[1]
    assert (u2.status, u2.attempts, u2.integrated, u2.candidate) == ("pending", 1, False, None)
    assert [(c.id, c.cap_usd, c.charge_usd, c.booked) for c in saved.calls if c.seat == "reader"] == [
        ("c2", 1.5, 1.52, True), ("c3", 3.0, 3.04, True)] and saved.cost_usd == pytest.approx(0.5 + 1.52 + 3.04)
    names = [(e["event"], e.get("id")) for e in state.read_log(root, TASK_ID)]
    assert names.index(("call", "c3")) < names.index(("question", held.id))
    assert [v["unit"] for v in _events(root, "verdict")] == [] and _events(root, "weakening") == []
    assert _caps(runner) == ["1.50", "3.00"]

    # neither `resume` nor `extend` alone releases it, and no further paid call starts
    task = loop.resume(root, caps, runner, unattended=0.0, notify=None)
    assert task.terminal == "interrupted" and [q.id for q in state.open_questions(task)] == [held.id]
    loop.offline_op(root, caps, "extend", {"usd": 5.0})
    task = loop.resume(root, caps, runner, unattended=0.0, notify=None)
    assert [q.id for q in state.open_questions(task)] == [held.id] and len(runner.calls) == 3
    assert (task.units[1].status, task.units[1].attempts) == ("pending", 1)

    loop.offline_op(root, caps, "answer", {"id": held.id, "text": answer})
    task = loop.resume(root, caps, runner, unattended=0.0, notify=None)
    if answer == "yes":  # (c) once more under the wider cap; a cap hit there opens the question again
        assert _caps(runner) == ["1.50", "3.00", "3.00"] and load(root, TASK_ID).questions[0].disposition == "reasked"
        (again,) = state.open_questions(task)
        assert (again.reason, again.context["reader_cap"]["calls"]) == ("reader_cap", ["c2", "c3", "c4"])
        assert again.context["reader_cap"]["cost_usd"] == pytest.approx(7.60) and "$7.60" in again.text
        assert (task.units[1].status, task.units[1].attempts) == ("pending", 1)
        assert [e.get("retry_of") for e in _events(root, "call") if e["id"] == "c4"] == ["c3"]
        loop.offline_op(root, caps, "answer", {"id": again.id, "text": "yes"})
        task = loop.resume(root, caps, runner, unattended=0.0, notify=None)
        assert _caps(runner) == ["1.50", "3.00", "3.00", "3.00"]
        u2 = task.units[1]
        assert (u2.status, u2.attempts, u2.integrated) == ("passed", 1, True)  # the verdict graded the attempt
        (verdict,) = [v for v in _events(root, "verdict") if v["unit"] == "u-2"]
        weakening = verdict["weakening"]
        assert (verdict["attempt"], verdict["passed"], weakening["result"], weakening["call"]) == (
            1, True, "not_weakened", "c5")
        assert _seats(runner) == ["worker", "reader", "reader", "reader", "reader", "director"]
        return

    # (d) `no`: the attempt is rejected exactly as a cap-hit rejection is, and the retry runs as it always did
    assert load(root, TASK_ID).questions[0].disposition == "rejected" and _caps(runner) == ["1.50", "3.00"]
    first, second = [v for v in _events(root, "verdict") if v["unit"] == "u-2"]
    assert (first["attempt"], first["passed"], first["status"], first["passed_ids"]) == (1, False, "pending", ["p1"])
    assert first["weakening"] == {"protected": ["tests"], "touched": ["tests/t.txt"], "result": "unchecked",
                                  "checks": ["u-1 p1"], "reason": "cap_hit", "kind": None, "note": CAP_NOTE,
                                  "call": "c3"}
    (event,) = _events(root, "weakening")
    assert (event["attempt"], event["result"], event["reason"], event["note"], event["call"]) == (
        1, "unchecked", "cap_hit", CAP_NOTE, "c3")
    retry = [stdin for argv, stdin, _, _ in runner.calls if ScriptedRunner.seat(argv) == "worker"][1]
    assert "could not run" in retry and f"- note: {json.dumps(CAP_NOTE)}\n" in retry
    u2 = task.units[1]
    assert (second["attempt"], u2.status, u2.attempts, u2.integrated) == (2, "passed", 2, True)
    assert _seats(runner) == ["worker", "reader", "reader", "worker", "director"]


def test_the_tick_whose_release_node_acts_on_the_answer_records_save_as_the_last_node_it_walked(
    guarded: tuple[Root, Task], monkeypatch: pytest.MonkeyPatch
):
    """The release node acts on the answered check and runs save's act inside it; the tick is counted, and its `fired`
    walk ends on save, as every counted tick's does."""
    root, task = guarded
    caps = Caps(reader_max_usd=12.0)
    runner = ScriptedRunner([REST], [W_TOUCH], [CAPPED, CAPPED_WIDE])
    loop.run_tick(task, caps, root, runner)
    (held,) = state.open_questions(task)
    console.apply(task, caps, "answer", {"id": held.id, "text": "no"}, False)
    loop._settle(task, caps, root, lambda _: None)
    visited: list[str] = []
    acts = {name: (lambda tick, name=name, act=act: visited.append(name) or act(tick))
            for name, act in loop.ACTS.items()}
    monkeypatch.setattr(loop, "ACTS", acts)
    counted = task.tick
    loop.run_tick(task, caps, root, runner)
    assert task.tick == counted + 1 and visited == ["repair", "block", "budget", "release"]  # save ran inside release
    (verdict,) = _events(root, "verdict")
    assert (verdict["unit"], verdict["passed"], verdict["weakening"]["reason"]) == ("u-2", False, "cap_hit")
    fired = _events(root, "fired")[-1]
    walked = ["repair", "block", "budget", "release", "save"]
    assert task.fired.walked == fired["walked"] == load(root, TASK_ID).fired.walked == walked


# --- (e) branch admission: the same re-ask, and a held plan instead of a refused one ---


@pytest.fixture
def released(tmp_path: Path, root: Root) -> tuple[Root, Task]:
    """u-1 (stats.py) accepted at the tip; a branching plan (u-2 takes over u-1 p1) held in `hold` and approved."""
    workspace, tip = _seed(tmp_path, root, ("stats.py", MEAN))
    u1 = Unit("u-1", "write stats.py", [Predicate("file_exists", {"path": "stats.py"})], status="passed",
              candidate=tip, integrated=True)
    task = Task(id=TASK_ID, goal="g", workspace=str(workspace), units=[u1], accepted=tip, mode=HOLD)
    _git(workspace, "update-ref", f"refs/heads/task/{TASK_ID}", tip)
    u2 = Unit("u-2", "rename stats.py to metrics.py",
              [Predicate("file_contains", {"path": "metrics.py", "text": "def mean"}, parents=["u-1 p1"])],
              rationale="u-1 p1 pinned stats.py; metrics.py takes it over.")
    task.held = Held(kind=PLAN, units=[u2], approved=True, move="m1")
    save(root, task)
    set_current(root, task.id)
    return root, task


@pytest.mark.parametrize("answer", ["yes", "no"])
def test_an_admission_reader_at_its_cap_twice_holds_the_plan_and_the_answer_re_asks_or_refuses(
    released: tuple[Root, Task], answer: str
):
    root, task = released
    caps, silent = Caps(), lambda _: None
    runner = ScriptedRunner([REST], [W_RENAME], [CAPPED, CAPPED_WIDE] + ([YES] if answer == "yes" else []))
    loop.run_tick(task, caps, root, runner)
    assert [q.id for q in state.open_questions(load(root, TASK_ID))] == ["q1"]  # saved
    (held,) = state.open_questions(task)
    info = held.context["reader_cap"]
    assert (held.reason, info["check"], info["unit"], info["calls"]) == ("reader_cap", "admission", "u-2",
                                                                          ["c1", "c2"])
    assert (task.held.kind, task.held.approved, [u.id for u in task.units]) == (PLAN, True, ["u-1"])  # still held
    assert _events(root, "branch_refused") == [] and _events(root, "branch_admitted") == []
    assert _seats(runner) == ["reader", "reader"] and _caps(runner) == ["1.50", "3.00"]  # never back to the thinker

    console.apply(task, caps, "answer", {"id": held.id, "text": answer}, True)
    loop._settle(task, caps, root, silent)
    loop.run_tick(task, caps, root, runner)
    if answer == "yes":
        assert _caps(runner) == ["1.50", "3.00", "2.44"] and [u.id for u in task.units] == ["u-1", "u-2"]  # $7 − $4.56
        (admitted,) = _events(root, "branch_admitted")
        assert admitted["call"] == "c3" and [e.get("retry_of") for e in _events(root, "call")
                                             if e["seat"] == "reader"] == [None, "c1", "c2"]
        assert held.disposition == "reasked" and _seats(runner) == ["reader", "reader", "reader", "worker"]
        return
    line = "unit 'u-2': branch refused, the reader gave no verdict (cap_hit): the reader call reached its $3.00 cap"
    (refused,) = _events(root, "branch_refused")
    assert (refused["reason"], refused["call"], refused["to_director"]) == ("cap_hit", "c2", line)
    assert task.held is None and [u.id for u in task.units] == ["u-1"] and held.disposition == "refused"
    assert _caps(runner) == ["1.50", "3.00"] and _seats(runner) == ["reader", "reader", "director"]
    assert line in json.loads(runner.calls[-1][1])["notes"]


def test_the_admission_re_ask_is_clipped_by_the_reader_s_budget_and_no_room_for_it_holds_the_plan(
    released: tuple[Root, Task]
):
    """2026-10-20: the reader's calls are clipped by `reader_max_usd` ($7.00) less the reader's spend, never by the
    allowance (the session's $25 is untouched here)."""
    root, task = released
    u2 = task.held.units[0]
    from protean.state import Call  # noqa: PLC0415 — the reader has spent $4.00 of its $7.00 already
    task.calls.append(Call(id="c1", seat="reader", attempt=1, cap_usd=3.0, outcome="ok", charge_usd=4.0, booked=True))
    runner = ScriptedRunner([], [], [CAPPED, YES, CAPPED])
    admission = gate.admit(task, [u2], Caps(), root, runner)
    # the first call charged 1.52 leaves 1.48 of the reader's budget: the wider cap is clipped to it
    assert (admission.refused, admission.waits, _caps(runner)) == ([], False, ["1.50", "1.48"])
    assert [e["unit"] for e in _events(root, "branch_admitted")] == ["u-2"]
    # $5.57 spent: the next first call is clipped to 1.43, and its cap hit leaves nothing: no wider call starts
    admission = gate.admit(task, [u2], Caps(), root, runner)
    assert (admission.refused, admission.waits, _caps(runner)) == ([], True, ["1.50", "1.48", "1.43"])
    (held,) = state.open_questions(load(root, TASK_ID))
    assert (held.reason, held.context["reader_cap"]["calls"]) == ("reader_cap", ["c4"])
    assert "below the $0.50 call floor" in held.text and _events(root, "branch_refused") == []


# --- the console's rules for the answer, and what the terminal says ---


def test_a_reader_cap_question_takes_only_yes_or_no_and_extend_never_releases_it():
    task = Task(id=TASK_ID, goal="g", workspace="/w")
    question = state.add_question(task, RUNTIME, "reader_cap", "held", {"reader_cap": {"check": "weakening"}})
    with pytest.raises(console.Refused) as refused:
        console.check(task, "answer", {"id": question.id, "text": "maybe"}, True)
    assert refused.value.status == 409
    for text in ("yes", "No."):
        console.check(task, "answer", {"id": question.id, "text": text}, True)
    console.apply(task, Caps(), "extend", {"usd": 5.0}, True)
    assert question.status == "open"
    assert "answer q1 yes" in questions.how(question) and "no" in questions.how(question)
    question.status, question.answer = "answered", "no"
    assert questions.consume(question)["disposition"] == "rejected"
