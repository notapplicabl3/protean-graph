"""Protected paths and the pre-integration weakening check (reader-first; the gate decides the order).

Each retained `command` names the `protected` paths that control its result; one that names none, or names one
outside the tree, goes back to the director at decode, before any worker call. Before a candidate integrates, and only
once it passed every free check (its own grade, the re-checks, the clean clone), the gate diffs it against the accepted
tip it branched from. The protected set is the union over every integrated unit's command still in the re-check set,
less the parents excused on this candidate; the candidate's own commands never count. Nothing touched: no reader call.
Touched: exactly one fresh reader call at the candidate. `yes` (weakened) rejects the attempt like a failed predicate
and its kind, note and touched paths reach the next attempt as evidence; a call that gives no verdict never integrates
either.
"""

from __future__ import annotations

import itertools
import json
import subprocess
from pathlib import Path

import pytest

from protean import loop, state
from protean.dry import FIXTURES, REPO, ScriptedRunner, make_workspace, read_jsonl, scenario_runner
from protean.graph import director, gate, grade, reader, worker
from protean.graph.budget import Caps
from protean.state import Predicate, Root, Task, Unit, load, save, set_current

TASK_ID = "task-20261011T120000"
PLAN = json.loads(read_jsonl(FIXTURES / "weaken" / "director.jsonl")[0]["result"])
U1, U2, U3 = PLAN["task"]
W_CALC, W_WEAK, W_SUB, W_TEST_SUB = read_jsonl(FIXTURES / "weaken" / "worker.jsonl")
READ_YES, READ_NO = read_jsonl(FIXTURES / "weaken" / "reader.jsonl")
YES, NO = json.loads(READ_YES["result"]), json.loads(READ_NO["result"])
TEST_CALC = W_CALC["write"]["tests/test_calc.py"]
NEEDS = ('a retained command needs "protected": the paths that control its result (its tests, their config), or '
         '"retain": false when it only proves this unit\'s delivery')
FINDING = ("weakening check (a reader's second look at the protected paths of accepted checks this commit touched): "
           "{}\n")
KIND_ANSWER = '{"verdict": "yes"|"no", "kind": "<kind>", "note": "<one line>"}'
KINDS = ("deleted tests", "weakened assertions", "skipped or expected-fail", "legitimate refactor", "strengthened",
         "unrelated")


STANDINGS = itertools.count(1)


def _reply(obj: dict, cost: float = 0.2) -> dict:
    """A thinker reply; a move without `state` gets one whose standing no earlier reply had."""
    if "state" not in obj and any(kind in obj for kind in state.MOVE_KINDS):
        obj = {**obj, "state": {"standing": f"Standing {next(STANDINGS)}.", "open": []}}
    return {"type": "result", "is_error": False, "session_id": "dry-director", "total_cost_usd": cost,
            "result": json.dumps(obj)}


def _rested(task: Task) -> bool:
    return (task.terminal, task.awaiting_thought, task.moves[-1].kind) == ("interrupted", True, "rest")


def _read(result: str, cost: float = 0.05) -> dict:
    return {"type": "result", "is_error": False, "session_id": "dry-reader", "total_cost_usd": cost, "result": result}


def _git(cwd: Path, *args: str) -> str:
    return subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t.invalid", *args], cwd=cwd,
                          capture_output=True, text=True, check=True).stdout.strip()


def _write(tree: Path, files: dict[str, str]) -> None:
    for rel, text in files.items():
        (tree / rel).parent.mkdir(parents=True, exist_ok=True)
        (tree / rel).write_text(text)


def _events(root: Root, task_id: str, name: str | None = None) -> list[dict]:
    return [e for e in state.read_log(root, task_id) if name is None or e["event"] == name]


def _seats(runner: ScriptedRunner) -> list[str]:
    return [ScriptedRunner.seat(argv) for argv, _, _, _ in runner.calls]


def _messages(runner: ScriptedRunner, seat: str) -> list[str]:
    return [stdin for argv, stdin, _, _ in runner.calls if ScriptedRunner.seat(argv) == seat]


class Spy:
    """The scripted runner, recording the commit and status each reader call's working directory held."""

    def __init__(self, scripted: ScriptedRunner) -> None:
        self.scripted, self.reader_cwds = scripted, []

    @property
    def calls(self):
        return self.scripted.calls

    def __call__(self, argv, stdin, cwd, timeout):
        if ScriptedRunner.seat(argv) == "reader":
            self.reader_cwds.append((Path(cwd), _git(Path(cwd), "rev-parse", "HEAD"),
                                     _git(Path(cwd), "status", "--porcelain")))
        return self.scripted(argv, stdin, cwd, timeout)


@pytest.fixture
def env(tmp_path: Path, root: Root) -> tuple[Root, Path]:
    root.path.mkdir(parents=True)
    root.thinker_prompt.write_text("m\n")
    root.editor_prompt.write_text("e\n")
    root.reader_prompt.write_text("# reader prompt\n")
    return root, make_workspace(tmp_path / "ws")


def _saved_task(root: Root, workspace: Path) -> Task:
    task = Task(id=TASK_ID, goal="g", workspace=str(workspace))
    save(root, task)
    set_current(root, task.id)
    return task


# --- decode requires protected paths on a retained command and confines them, before any worker call ---


def _plan(*predicates: dict) -> str:
    return json.dumps({"task": [{"id": "u-2", "intent": "i", "predicates": list(predicates)}],
                       "state": {"standing": "s.", "open": []}})


def test_decode_requires_protected_paths_on_a_retained_command_and_confines_each(tmp_path: Path):
    task = Task(id="t", goal="g", workspace=str(make_workspace(tmp_path / "ws")))
    outside, in_git = str(tmp_path), str(Path(".git", "info"))
    run = {"cmd": "true"}
    reply, errors = director.decode_move(_plan(
        {"kind": "command", "args": run},  # retained by default, no protected
        {"kind": "command", "args": run, "protected": []},
        {"kind": "command", "args": run, "protected": ["../x", outside, in_git, "tests"]},
        {"kind": "command", "args": run, "protected": "tests"},
        {"kind": "file_exists", "args": {"path": "a.txt"}, "protected": ["tests"]},
        {"kind": "command", "args": run, "protected": ["tests", " "]},
    ), task)
    assert reply is None and errors == [
        f"unit 'u-2': p1 command: {NEEDS}",
        f"unit 'u-2': p2 command: {NEEDS}",
        "unit 'u-2': p4 command: protected must be a list of paths relative to the tree",
        "unit 'u-2': p5 file_exists: protected is only for a command predicate",
        "unit 'u-2': p6 command: protected must be a list of paths relative to the tree",
        "unit 'u-2': p3 command: protected path climbs out of the tree: ../x",
        f"unit 'u-2': p3 command: protected absolute path, not relative to the tree: {outside}",
        f"unit 'u-2': p3 command: protected path is inside .git, not on the branch: {in_git}",
    ]
    reply, errors = director.decode_move(_plan(
        {"kind": "command", "args": run, "retain": False},  # delivery-only: may omit it
        {"kind": "command", "args": run, "retain": False, "protected": ["tests"]},  # kept, never consulted
        {"kind": "command", "args": run, "protected": ["tests/", "tests/", "conftest.py"]},
    ), task)
    assert errors == [] and [p.protected for p in reply.units[0].predicates] == [
        [], ["tests"], ["tests/", "conftest.py"]]


def test_a_retained_command_without_protected_paths_goes_back_to_the_thinker_before_any_worker_call(
    env: tuple[Root, Path]
):
    root, workspace = env
    bare = {**U1, "predicates": [U1["predicates"][0],
                                 {k: v for k, v in U1["predicates"][1].items() if k != "protected"}]}
    runner = ScriptedRunner([_reply({"task": [bare]}), _reply({"task": [U1]}), _reply({"rest": True})], [W_CALC])
    task = loop.run("g", workspace, root, Caps(), runner, unattended=0.0, notify=None)
    assert _seats(runner) == ["director", "director", "worker", "director"]  # no worker call until corrected
    first, retry = _messages(runner, "director")[:2]
    assert retry == first + (f"\n\nYour previous answer was rejected:\n- unit 'u-1': p2 command: {NEEDS}\n"
                             "Answer again with exactly one JSON object.")
    assert _rested(task) and task.units[0].predicates[1].protected == ["tests"]


# --- the `weaken` scenario, on real git diffs ---


def test_the_weaken_scenario_rejects_the_weakened_candidate_and_pays_only_where_a_protected_path_is_touched(
    env: tuple[Root, Path]
):
    root, workspace = env
    spy = Spy(scenario_runner("weaken"))
    task = loop.run("weaken", workspace, root, Caps(), spy, unattended=0.0, notify=None)
    assert _rested(task) and [(u.id, u.status, u.attempts, u.integrated) for u in task.units] == [
        ("u-1", "passed", 1, True), ("u-2", "passed", 2, True), ("u-3", "passed", 1, True)]
    # one reader call per candidate that touched a protected path; none for u-2 attempt 2, which touched none
    assert _seats(spy.scripted) == ["director", "worker", "worker", "reader", "worker", "worker", "reader", "director"]
    assert spy.scripted.reader == [] and task.cost_usd == pytest.approx(2.40)
    verdicts = {(v["unit"], v["attempt"]): v for v in _events(root, task.id, "verdict")}
    u1, a1, a2, u3 = (verdicts[key]["commit"] for key in (("u-1", 1), ("u-2", 1), ("u-2", 2), ("u-3", 1)))
    assert verdicts[("u-1", 1)]["weakening"] == {"protected": [], "touched": [], "result": "untouched"}
    assert verdicts[("u-2", 2)]["weakening"] == {"protected": ["tests"], "touched": [], "result": "untouched"}
    assert verdicts[("u-2", 2)]["changed"] == ["calc.py"]

    # row 4: exactly one reader call for the candidate that touched one protected path, recorded and charged
    calls = [c for c in task.calls if c.seat == "reader"]
    assert [(c.unit, c.attempt, c.outcome, c.charge_usd, c.receipt_cost_usd, c.booked, c.cap_usd) for c in calls] == [
        ("u-2", 1, "ok", 0.05, 0.05, True, 1.5), ("u-3", 1, "ok", 0.05, 0.05, True, 1.5)]
    first, second = _events(root, task.id, "weakening")
    assert first == {**first, "unit": "u-2", "attempt": 1, "accepted_tip": u1, "candidate": a1,
                     "touched": ["tests/test_calc.py"], "checks": ["u-1 p2"], "result": "weakened", "reason": "yes",
                     "kind": "weakened assertions", "note": YES["note"], "call": calls[0].id}
    assert second == {**second, "unit": "u-3", "attempt": 1, "accepted_tip": a2, "candidate": u3,
                      "touched": ["tests/test_sub.py"], "checks": ["u-1 p2"], "result": "not_weakened",
                      "reason": "no", "kind": "strengthened", "note": NO["note"], "call": calls[1].id}
    names = [(e["event"], e.get("seat"), e.get("unit"), e.get("attempt")) for e in _events(root, task.id)]
    call_at = names.index(("call", "reader", "u-2", 1))
    assert call_at < names.index(("weakening", None, "u-2", 1)) < names.index(("verdict", None, "u-2", 1))
    argv, stdin, cwd, timeout = next(c for c in spy.calls if ScriptedRunner.seat(c[0]) == "reader")
    clone = reader.clone_dir(root, task)
    assert argv[:3] == [worker.sandbox_binary(), "-p", reader.sandbox_profile(clone, workspace,
                                                                              root.console_file(task.id))]
    assert argv[argv.index("--max-budget-usd") + 1] == "1.50" and argv[-2:] == ["--system-prompt", "# reader prompt\n"]
    assert spy.reader_cwds == [(clone, a1, ""), (clone, u3, "")]  # fresh, clean clones at the candidates
    assert cwd == clone and timeout == 300 and not clone.exists()
    brief = json.loads(stdin)
    assert brief == {"check": "weakening", "unit": "u-2", "intent": U2["intent"], "accepted_tip": u1, "candidate": a1,
                     "checks": [{"id": "u-1 p2", "kind": "command", "args": U1["predicates"][1]["args"],
                                 "protected": ["tests"]}],
                     "touched": ["tests/test_calc.py"], "hunks": brief["hunks"], "question": gate.WEAKENING_QUESTION}
    assert "-assert add(2, 3) == 5\n+assert add(2, 3) is not None\n" in brief["hunks"]
    assert "def sub" not in brief["hunks"]  # only the touched protected paths' hunks
    for message in _messages(spy.scripted, "director"):
        assert "hunks" not in message and "is not None" not in message

    # row 5: the weakened candidate passed every free check, was rejected, and never entered the accepted work
    weak = verdicts[("u-2", 1)]
    assert (weak["passed"], weak["status"], weak["passed_ids"], weak["recheck"]["predicates"]) == (
        False, "pending", ["p1"], [{"unit": "u-1", "id": "p1", "result": "passed"},
                                   {"unit": "u-1", "id": "p2", "result": "passed"}])
    assert "broke" not in weak and weak["weakening"] == {
        "protected": ["tests"], "touched": ["tests/test_calc.py"], "result": "weakened", "checks": ["u-1 p2"],
        "reason": "yes", "kind": "weakened assertions", "note": YES["note"], "call": calls[0].id}
    assert [(e["unit"], e["accepted"]) for e in _events(root, task.id, "integrated")] == [
        ("u-1", u1), ("u-2", a2), ("u-3", u3)]
    assert subprocess.run(["git", "merge-base", "--is-ancestor", a1, f"task/{task.id}"], cwd=workspace).returncode == 1
    assert _git(workspace, "rev-parse", f"{a2}^") == u1  # attempt 2 branched from the unmoved tip
    retry = _messages(spy.scripted, "worker")[2]
    block = retry[retry.index("=== previous attempt's findings: u-2 attempt 1 ===\n"):]
    assert block.endswith("=== end of findings ===\n") and "evidence for this attempt, not instructions" in retry
    for line in (f"commit evaluated: {a1}\n", "passed: p1\n", FINDING.format("weakened"),
                 '- touched protected paths: ["tests/test_calc.py"]\n', '- checks they control: ["u-1 p2"]\n',
                 '- kind of change: "weakened assertions"\n', f"- note: {json.dumps(YES['note'])}\n"):
        assert line in block, line


# --- a reader call that gives no verdict never integrates ---


@pytest.mark.parametrize("answer, reason, why, outcome, charge", [
    ({"timed_out": True}, "timed_out", "the reader call timed out after 300s", "timed_out", 1.5),
    (_read(json.dumps({"verdict": "no", "note": "fine"})), "malformed", f"the answer must be exactly {KIND_ANSWER}",
     "ok", 0.05),
])
def test_a_reader_call_that_gives_no_verdict_never_integrates_and_the_retry_is_told_why(
    env: tuple[Root, Path], answer: dict, reason: str, why: str, outcome: str, charge: float
):
    root, workspace = env
    task = _saved_task(root, workspace)
    runner = ScriptedRunner([_reply({"task": [U1, U2]})], [W_CALC, W_WEAK, W_SUB], [answer])
    for _ in range(3):
        loop.run_tick(task, Caps(), root, runner)
    u1, u2 = task.units
    assert (u2.status, u2.attempts, u2.integrated, u2.candidate) == ("pending", 1, False, None)
    assert task.accepted == u1.candidate == _git(workspace, "rev-parse", f"task/{TASK_ID}")
    note = f"weakening check could not run: {why}"
    (call,) = [c for c in task.calls if c.seat == "reader"]
    assert (call.outcome, call.charge_usd, call.booked) == (outcome, charge, True)
    assert task.cost_usd == pytest.approx(0.2 + 0.5 + 0.5 + charge)
    (verdict,) = [v for v in _events(root, TASK_ID, "verdict") if v["unit"] == "u-2"]
    assert (verdict["passed"], verdict["passed_ids"], verdict["weakening"]) == (False, ["p1"], {
        "protected": ["tests"], "touched": ["tests/test_calc.py"], "result": "unchecked", "checks": ["u-1 p2"],
        "reason": reason, "kind": None, "note": note, "call": call.id})
    assert u2.summary.endswith(f"[grade: {note}]")
    loop.run_tick(task, Caps(), root, runner)  # attempt 2 touches calc.py only: no reader call, integrated
    assert (u2.status, u2.attempts, u2.integrated) == ("passed", 2, True) and _seats(runner).count("reader") == 1
    retry = _messages(runner, "worker")[2]
    assert FINDING.format("could not run") in retry and f"- note: {json.dumps(note)}\n" in retry
    assert '- touched protected paths: ["tests/test_calc.py"]\n' in retry and "kind of change" not in retry


def test_an_interrupted_weakening_call_books_its_cap_before_the_interruption_propagates(env: tuple[Root, Path]):
    root, workspace = env
    task = _saved_task(root, workspace)
    scripted = ScriptedRunner([_reply({"task": [U1, U2]})], [W_CALC, W_WEAK], [])

    def runner(argv, stdin, cwd, timeout):
        if ScriptedRunner.seat(argv) == "reader":
            raise KeyboardInterrupt
        return scripted(argv, stdin, cwd, timeout)

    for _ in range(2):
        loop.run_tick(task, Caps(), root, runner)
    with pytest.raises(KeyboardInterrupt):
        loop.run_tick(task, Caps(), root, runner)
    saved = load(root, TASK_ID)
    assert saved.cost_usd == pytest.approx(0.2 + 0.5 + 0.5 + 1.5)  # director, two workers, the reader's cap
    assert (saved.calls[-1].seat, saved.calls[-1].outcome, saved.calls[-1].booked) == ("reader", "interrupted", True)
    interrupted = _events(root, TASK_ID, "interrupted")[-1]
    assert (interrupted["seat"], interrupted["charged_usd"]) == ("reader", 1.5)
    assert not saved.units[1].integrated and saved.accepted == saved.units[0].candidate


# --- the order (ruled 2026-10-11): a candidate a free check rejects never pays for the reader ---


def test_the_reader_is_never_paid_for_a_candidate_a_free_check_already_rejects(env: tuple[Root, Path]):
    root, workspace = env
    mul = {**U2, "predicates": [{"kind": "file_contains", "args": {"path": "calc.py", "text": "def mul"}}]}
    breaks = {**W_WEAK, "write": {"calc.py": W_SUB["write"]["calc.py"].replace("sub", "mul"),
                                  "tests/test_calc.py": TEST_CALC.replace("== 5", "== 6")}}
    runner = ScriptedRunner([_reply({"task": [U1, mul]}), _reply({"ask": "u-2 failed: replan?"})],
                            [W_CALC, W_WEAK, breaks], [])  # any reader call would find no answer and raise
    task = loop.run("g", workspace, root, Caps(), runner, unattended=0.0, notify=None)
    assert task.terminal == "interrupted" and [u.status for u in task.units] == ["passed", "failed"]
    assert "reader" not in _seats(runner) and _events(root, task.id, "weakening") == []
    accepted, own, broke = _events(root, task.id, "verdict")
    assert accepted["weakening"] == {"protected": [], "touched": [], "result": "untouched"}  # u-1: free checks passed
    assert own["failed_ids"] == ["p1"] and "recheck" not in own and "weakening" not in own  # its own grade failed
    assert list(broke["broke"]) == ["u-1 p2"] and "weakening" not in broke  # it broke an accepted command


# --- an excused parent's protected paths are not in the set on the branching candidate ---


def test_the_protected_set_counts_accepted_retained_commands_only_and_never_an_excused_parent():
    def command(*protected: str, **flags) -> Predicate:
        return Predicate("command", {"cmd": "true"}, protected=list(protected), **flags)

    u1 = Unit("u-1", "i", [command("tests"), command("docs", "tests"), command("scaffold", retain=False),
                           Predicate("file_exists", {"path": "a.txt"}), command("old", children=["u-3 p1"])],
              status="passed", candidate="c1", integrated=True)
    u2 = Unit("u-2", "i", [command("mine")])  # the candidate's own unit: not integrated, never counted
    task = Task("t", "g", "/w", units=[u1, u2])
    assert gate.protected_set(task, {}) == {"tests": ["u-1 p1", "u-1 p2"], "docs": ["u-1 p2"]}
    assert gate.protected_set(task, {"u-1 p1": ["u-2 p1"]}) == {"docs": ["u-1 p2"], "tests": ["u-1 p2"]}
    assert gate.protected_set(task, {"u-1 p1": ["u-2 p1"], "u-1 p2": ["u-2 p1"]}) == {}


def test_a_branching_candidate_is_not_checked_against_its_excused_parents_protected_paths(env: tuple[Root, Path]):
    root, workspace = env
    move = {"id": "u-2", "intent": "rename tests/test_calc.py to tests/test_math.py",
            "rationale": "u-1 p2 ran the add test; the same test runs under its new name as u-2 p1.",
            "predicates": [{"kind": "command", "args": {"cmd": "PYTHONPATH=. python3 -B tests/test_math.py"},
                            "parents": ["u-1 p2"], "protected": ["tests"]}]}
    w_move = {"write": {"tests/test_math.py": TEST_CALC}, "delete": ["tests/test_calc.py"], "commit": True,
              "exit_code": 0, "result": "renamed the test", "total_cost_usd": 0.5}
    admit = _read(json.dumps({"verdict": "yes", "note": "the same test under a new name"}))
    runner = ScriptedRunner([_reply({"task": [U1]}), _reply({"task": [move]}), _reply({"rest": True})],
                            [W_CALC, w_move], [admit])
    task = loop.run("g", workspace, root, Caps(), runner, unattended=0.0, notify=None)
    assert _rested(task) and [(u.id, u.status) for u in task.units] == [("u-1", "passed"), ("u-2", "passed")]
    # the branch admission is the only reader call: the rename touched tests/, but only the excused parent named it
    assert _seats(runner) == ["director", "worker", "director", "reader", "worker", "director"]
    (verdict,) = [v for v in _events(root, task.id, "verdict") if v["unit"] == "u-2"]
    assert verdict["recheck"]["predicates"] == [{"unit": "u-1", "id": "p1", "result": "passed"},
                                                {"unit": "u-1", "id": "p2", "result": "excused",
                                                 "children": ["u-2 p1"]}]
    tip = next(v["commit"] for v in _events(root, task.id, "verdict") if v["unit"] == "u-1")
    assert gate.touched(workspace, tip, verdict["commit"], ["tests"]) == ["tests/test_calc.py", "tests/test_math.py"]
    assert verdict["weakening"] == {"protected": [], "touched": [], "result": "untouched"}
    assert _events(root, task.id, "weakening") == []


# --- what counts as touched, the hunks' bound, the reader's cap and floor ---


def test_touched_matches_whole_components_case_folded_and_sees_a_protected_file_moved_out(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    table = {"tests": True, "tests/test_x.py": True, "tests/a/b.py": True, "Tests/x.py": True, "tests_old.py": False,
             "test": False, "src/tests/x.py": False}
    assert {path: gate.covers("tests", path) for path in table} == table
    assert gate.covers("tests/", "tests/x.py") and gate.covers("./conftest.py", "conftest.py")
    assert gate.covers(".", "any/where.py") and not gate.covers("conftest.py", "conftest.pyc")
    repo = make_workspace(tmp_path / "ws")
    _write(repo, {"tests/test_a.py": "assert True\n", "tests_old.py": "x = 1\n"})
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "tip")
    tip = _git(repo, "rev-parse", "HEAD")
    (repo / "old").mkdir()
    _git(repo, "mv", "tests/test_a.py", "old/test_a.py")
    (repo / "tests_old.py").write_text("x = 2\n")
    _git(repo, "commit", "-qam", "candidate")
    candidate = _git(repo, "rev-parse", "HEAD")
    assert gate.touched(repo, tip, candidate, ["tests"]) == ["tests/test_a.py"]  # the move out is a touch
    monkeypatch.setattr(gate, "HUNK_BOUND", 40)
    text = gate.hunks(repo, tip, candidate, ["tests/test_a.py"])
    assert text.startswith("diff --git a/tests/test_a.py b/tests/test_a.py\n"[:40])
    assert text.endswith("files themselves are in your working directory ...]\n") and "cut here at 40 of" in text


@pytest.fixture
def candidate(env: tuple[Root, Path]) -> tuple[Root, Task, Unit, str, str]:
    """u-1 accepted at the workspace tip (its retained command protects `tests`), and u-2's attempt-2 candidate on its
    unit branch, weakening tests/test_calc.py."""
    root, workspace = env
    _write(workspace, W_CALC["write"])
    _git(workspace, "add", "-A")
    _git(workspace, "commit", "-qm", "u-1")
    tip = _git(workspace, "rev-parse", "HEAD")
    u1 = Unit("u-1", U1["intent"], [Predicate("file_exists", {"path": "calc.py"}),
                                    Predicate("command", U1["predicates"][1]["args"], protected=["tests"])],
              status="passed", candidate=tip, integrated=True)
    task = Task(id=TASK_ID, goal="g", workspace=str(workspace), units=[u1], accepted=tip)
    _git(workspace, "update-ref", worker.accepted_ref(task), tip)
    save(root, task)
    set_current(root, task.id)
    _git(workspace, "checkout", "-qb", f"unit/{TASK_ID}/u-2")
    _write(workspace, W_WEAK["write"])
    _git(workspace, "commit", "-qam", "u-2")
    candidate = _git(workspace, "rev-parse", "HEAD")
    _git(workspace, "checkout", "-q", "main")
    return root, task, Unit("u-2", U2["intent"], [Predicate("file_contains", {"path": "calc.py", "text": "def sub"})],
                            attempts=2), tip, candidate


def test_the_weakening_call_is_clipped_and_booked_like_any_reader_call_and_none_starts_under_the_floor(
    candidate: tuple[Root, Task, Unit, str, str]
):
    root, task, u2, tip, cand = candidate
    workspace = Path(task.workspace)
    task.cost_usd = 23.77
    runner = ScriptedRunner([], [], [READ_YES])
    check = gate.weakening(task, u2, workspace, tip, cand, {}, Caps(reader_call_usd=2.0), root, runner)
    argv = runner.calls[0][0]
    # the reader's own budget (2026-10-20): the $2.00 per-call cap stands whole, though the allowance has $1.23 left
    assert argv[argv.index("--max-budget-usd") + 1] == "2.00" and (check.result, check.kind) == (
        "weakened", "weakened assertions")
    assert task.cost_usd == pytest.approx(23.82) and load(root, task.id).calls[-1].booked  # booked at once, saved
    assert (check.call.seat, check.call.unit, check.call.attempt) == ("reader", "u-2", 2)
    from protean.state import Call  # noqa: PLC0415 — one booked reader call that leaves $0.40 of the reader's budget
    task.calls.append(Call(id="c-r", seat="reader", attempt=1, cap_usd=3.0, outcome="ok", charge_usd=6.55, booked=True))
    floor = gate.weakening(task, u2, workspace, tip, cand, {}, Caps(), root, runner)
    assert len(runner.calls) == 1 and floor.rejects and (floor.result, floor.reason, floor.call, floor.note) == (
        "unchecked", "floor", None,
        "weakening check could not run: no reader call: reader_max_usd: remaining $0.40 below the $0.50 call floor")
    assert _events(root, task.id, "weakening")[-1]["call"] is None


# --- the answer, the records, the prompts and the contract ---


@pytest.mark.parametrize("text, expected", [
    ('{"verdict": "yes", "kind": "deleted tests", "note": "test_a\\n went"}', ("yes", "deleted tests", "test_a went")),
    ('```json\n{"verdict": "no", "kind": "legitimate refactor", "note": "moved whole"}\n```',
     ("no", "legitimate refactor", "moved whole")),
    ('{"verdict": "no", "note": "fine"}', (None, None, f"the answer must be exactly {KIND_ANSWER}")),
    ('{"verdict": "no", "kind": "cosmetic", "note": "fine"}',
     (None, None, f"the kind must be one of {', '.join(KINDS)}, not 'cosmetic'")),
    ('{"verdict": "maybe", "kind": "unrelated", "note": "x"}',
     (None, None, "the verdict must be yes or no, not 'maybe'")),
    ('{"verdict": "no", "kind": "unrelated", "note": " "}', (None, None, "the note must be a non-empty string")),
])
def test_the_weakening_answer_names_exactly_one_kind(text: str, expected: tuple):
    assert reader.KINDS == KINDS and reader.parse_answer(text, reader.KINDS) == expected


def test_the_branch_answer_still_takes_no_kind():
    kinded = '{"verdict": "yes", "kind": "strengthened", "note": "x"}'
    assert reader.parse(kinded) == (None, f"the answer must be exactly {reader.ANSWER}")
    assert reader.parse_answer(kinded) == (None, None, f"the answer must be exactly {reader.ANSWER}")
    assert reader.parse('{"verdict": "yes", "note": "x"}') == ("yes", "x")


def test_protected_round_trips_older_records_load_without_it_and_the_prompts_and_contract_carry_it():
    command = Predicate("command", {"cmd": "true"}, protected=["tests"])
    task = Task("t", "g", "/w", units=[Unit("u-1", "i", [command])])
    assert state.from_json(state.to_json(task)) == task
    old = json.loads(state.to_json(task))
    del old["units"][0]["predicates"][0]["protected"]
    assert state.from_json(json.dumps(old)).units[0].predicates[0].protected == []
    for record in (REPO / "evidence").glob("*/task.json"):  # historical records, never edited: a pre-alpha one
        raw = record.read_text()  # loads with none; one the alpha saved (live-6 on) keeps the paths it wrote
        loaded = state.from_json(raw)
        if '"protected"' not in raw:
            assert all(p.protected == [] for u in loaded.units for p in u.predicates)
        else:
            assert any(p.protected for u in loaded.units for p in u.predicates)
    assert state.predicate_view(command) == {"kind": "command", "args": {"cmd": "true"}, "protected": ["tests"]}
    assert "protected" not in state.predicate_view(Predicate("command", {"cmd": "true"}, retain=False))
    assert grade.rechecked(command)
    prompt = (REPO / "brain" / "protean" / "thinker.md").read_text()
    (paragraph,) = [line for line in prompt.splitlines() if line.startswith("Each predicate is a lasting promise")]
    assert ('{"kind": "command", "args": {"cmd": "uv run pytest -q"}, "protected": ["tests", "conftest.py", '
            '"pyproject.toml"]}') in paragraph
    seat = (REPO / "brain" / "protean" / "reader.md").read_text()
    assert '"check": "weakening"' in seat
    assert "does this change weaken the acceptance checks the protected paths control?" in seat
    assert all(f"`{kind}`" in seat for kind in KINDS)
    contract = (REPO / "docs" / "console-api.md").read_text()
    assert "## The weakening check" in contract and "| `weakening` |" in contract
    assert "`weakening` {`protected` (the set in force)" in contract
