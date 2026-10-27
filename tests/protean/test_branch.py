"""Predicate branching, the reader seat and the gate's admission of a branch.

A new predicate may name `parents`: accepted, retained predicates it inherits, one-to-many or many-to-one, with the
unit's `rationale`. Before the unit enters the task, inside the same tick, the runtime makes one fresh, read-only
`reader` call (never through the director): `yes` admits; `no`, or any call that fails or answers anything else,
returns the whole plan to the director as one line on the retry path (a call that reaches its cap is asked again,
`test_reader_cap.py`; one that times out or leaves no receipt holds paid work). The
parents stay in force until the branching unit is integrated, except on that unit's own candidate: once every child
passes there, the parents of those children are excused from its re-check, so the unit may rename or remove what they
pinned. At integration each gains its children, keeps its record and the last commit it passed on, and leaves the
re-check set.
"""

from __future__ import annotations

import itertools
import json
import subprocess
from pathlib import Path

import pytest

from protean import cli, loop, state
from protean.dry import REPO, ScriptedRunner, make_workspace, scenario_runner
from protean.graph import console, director, gate, grade, reader, worker
from protean.graph.budget import Caps
from protean.state import Predicate, Root, Task, Unit, load, save, set_current

RATIONALE = "u-1 p1 pinned the stats module; mean() moves to metrics.py, which u-2 p1 pins."
U1 = {"id": "u-1", "intent": "write stats.py", "predicates": [{"kind": "file_exists", "args": {"path": "stats.py"}}]}
BRANCH = {"id": "u-2", "intent": "move mean() to metrics.py", "rationale": RATIONALE, "predicates": [
    {"kind": "file_contains", "args": {"path": "metrics.py", "text": "def mean"}, "parents": ["u-1 p1"]}]}
PLAIN = {"id": "u-2", "intent": "add metrics.py beside stats.py", "predicates": [
    {"kind": "file_contains", "args": {"path": "metrics.py", "text": "def mean"}}]}
NOTES = {"id": "u-3", "intent": "write NOTES.md", "predicates": [{"kind": "file_exists", "args": {"path": "NOTES.md"}}]}
MEAN = "def mean(xs):\n    return sum(xs) / len(xs)\n"
W_STATS = {"write": {"stats.py": MEAN}, "commit": True, "exit_code": 0, "result": "wrote stats.py",
           "total_cost_usd": 0.5}
W_SHIM = {"write": {"metrics.py": MEAN, "stats.py": "from metrics import mean\n"}, "commit": True, "exit_code": 0,
          "result": "moved mean() to metrics.py", "total_cost_usd": 0.5}
W_RENAME = {"write": {"metrics.py": MEAN}, "delete": ["stats.py"], "commit": True, "exit_code": 0,
            "result": "renamed stats.py to metrics.py", "total_cost_usd": 0.5}
W_NOTES = {"write": {"NOTES.md": "notes\n"}, "commit": True, "exit_code": 0, "result": "wrote NOTES.md",
           "total_cost_usd": 0.5}
W_NO_MEAN = {"write": {"metrics.py": "def average(xs):\n    return sum(xs) / len(xs)\n"}, "delete": ["stats.py"],
             "commit": True, "exit_code": 0, "result": "renamed stats.py to metrics.py", "total_cost_usd": 0.5}
W_PAIR = {"write": {"stats.py": MEAN, "keep.txt": "keep\n"}, "commit": True, "exit_code": 0,
          "result": "wrote stats.py and keep.txt", "total_cost_usd": 0.5}
W_RENAME_CLOBBER = {"write": {"metrics.py": MEAN, "keep.txt": "gone\n"}, "delete": ["stats.py"], "commit": True,
                    "exit_code": 0, "result": "renamed stats.py, rewrote keep.txt", "total_cost_usd": 0.5}
W_NOTES_DROP = {"write": {"NOTES.md": "notes\n"}, "delete": ["stats.py"], "commit": True, "exit_code": 0,
                "result": "wrote NOTES.md, removed stats.py", "total_cost_usd": 0.5}
NO_NOTE = "metrics.py drops the module's other functions that stats.py held."


STANDINGS = itertools.count(1)


def _reply(obj: object, cost: float = 0.2) -> dict:
    """A thinker reply; a move without `state` gets one whose standing no earlier reply had."""
    if isinstance(obj, dict) and "state" not in obj and any(kind in obj for kind in state.MOVE_KINDS):
        obj = {**obj, "state": {"standing": f"Standing {next(STANDINGS)}.", "open": []}}
    return {"type": "result", "is_error": False, "session_id": "dry-director", "total_cost_usd": cost,
            "result": obj if isinstance(obj, str) else json.dumps(obj)}


def _decode(units: list, task: Task) -> tuple[list[Unit] | None, list[str]]:
    """A `task` move's units through the thinker's decode: the units, or None and every error."""
    reply, errors = director.decode_move(json.dumps({"task": units, "state": {"standing": "s.", "open": []}}), task)
    return (None if reply is None else reply.units), errors


def _rested(task: Task) -> bool:
    return (task.terminal, task.awaiting_thought, task.moves[-1].kind) == ("interrupted", True, "rest")


def _read(result: str, cost: float = 0.05, **extra: object) -> dict:
    return {"type": "result", "is_error": False, "session_id": "dry-reader", "total_cost_usd": cost, "result": result,
            **extra}


YES = _read(json.dumps({"verdict": "yes", "note": "metrics.py carries mean()."}))
NO = _read(json.dumps({"verdict": "no", "note": NO_NOTE}))


def _git(cwd: Path, *args: str) -> str:
    return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, check=True).stdout.strip()


def _events(root: Root, task_id: str, name: str | None = None) -> list[dict]:
    return [e for e in state.read_log(root, task_id) if name is None or e["event"] == name]


def _seats(runner: ScriptedRunner) -> list[str]:
    return [ScriptedRunner.seat(argv) for argv, _, _, _ in runner.calls]


def _messages(runner: ScriptedRunner, seat: str) -> list[str]:
    return [stdin for argv, stdin, _, _ in runner.calls if ScriptedRunner.seat(argv) == seat]


@pytest.fixture
def env(tmp_path: Path, root: Root) -> tuple[Root, Path]:
    root.path.mkdir(parents=True)
    root.thinker_prompt.write_text("m\n")
    root.editor_prompt.write_text("e\n")
    root.reader_prompt.write_text("# reader prompt\n")
    return root, make_workspace(tmp_path / "ws")


# --- decode takes one-to-many and many-to-one, and refuses each bad parent on the retry path ---


def _decode_task(workspace: Path) -> Task:
    """u-1 accepted (p1, p3, p4 retained; p2 delivery-only; p5 a legacy exit_code); u-4 pending; u-5 accepted with p1
    already branched to u-6 p1; u-7 pending, claiming u-1 p3."""
    u1 = Unit("u-1", "i", [Predicate("file_exists", {"path": "stats.py"}),
                           Predicate("file_exists", {"path": "scaffold.txt"}, retain=False),
                           Predicate("file_contains", {"path": "w.py", "text": "w"}),
                           Predicate("command", {"cmd": "uv run pytest -q"}),
                           Predicate("exit_code", {"code": 0})],
              status="passed", candidate="c1", integrated=True)
    u4 = Unit("u-4", "i", [Predicate("file_exists", {"path": "x.py"})])
    u5 = Unit("u-5", "i", [Predicate("file_exists", {"path": "old.py"}, children=["u-6 p1"])],
              status="passed", candidate="c5", integrated=True)
    u6 = Unit("u-6", "i", [Predicate("file_exists", {"path": "new.py"}, parents=["u-5 p1"])], status="passed",
              candidate="c6", integrated=True, rationale="r")
    u7 = Unit("u-7", "i", [Predicate("file_exists", {"path": "y.py"}, parents=["u-1 p3"])], rationale="r")
    return Task(id="t", goal="g", workspace=str(workspace), units=[u1, u4, u5, u6, u7])


def _unit(uid: str, *parents: str, rationale: str | None = "r", retain: bool = True) -> dict:
    predicate: dict = {"kind": "file_exists", "args": {"path": f"{uid}.py"}, "parents": list(parents)}
    if not retain:
        predicate["retain"] = False
    unit = {"id": uid, "intent": "i", "predicates": [predicate]}
    if rationale is not None:
        unit["rationale"] = rationale
    return unit


def test_decode_takes_a_one_to_many_and_a_many_to_one_branch(tmp_path: Path):
    task = _decode_task(tmp_path)
    split = {"id": "u-8", "intent": "split stats.py", "rationale": "  stats.py becomes a.py and b.py  ",
             "predicates": [{"kind": "file_exists", "args": {"path": "a.py"}, "parents": ["u-1 p1"]},
                            {"kind": "file_exists", "args": {"path": "b.py"}, "parents": ["u-1 p1", "u-1 p1"]},
                            {"kind": "file_exists", "args": {"path": "c.py"}}]}
    units, errors = _decode([split], task)
    assert errors == []
    (unit,) = units
    assert [p.parents for p in unit.predicates] == [["u-1 p1"], ["u-1 p1"], []]
    assert unit.rationale == "stats.py becomes a.py and b.py" and gate.children_of(unit) == ["u-8 p1", "u-8 p2"]
    merge = {"id": "u-9", "intent": "merge", "rationale": "w.py and the test command merge into one check",
             "predicates": [{"kind": "command", "args": {"cmd": "uv run pytest -q tests/w"},
                             "parents": ["u-1 p4", "u-1 p1"], "protected": ["tests/w"]}]}
    units, errors = _decode([merge], task)
    assert errors == [] and units[0].predicates[0].parents == ["u-1 p4", "u-1 p1"]
    assert gate.parents_of(units[0]) == ["u-1 p4", "u-1 p1"]
    plain, _ = _decode([PLAIN | {"rationale": "ignored"}], task)
    assert plain[0].rationale == "" and gate.branching(plain) == []


def test_decode_refuses_each_bad_parent_naming_the_unit_and_the_predicate(tmp_path: Path):
    task = _decode_task(tmp_path)
    plan = {"task": [
        _unit("u-10", "u-9 p1", "u-1 p9", "u-1p1"),  # unknown
        _unit("u-11", "u-1 p2"),  # delivery-only parent
        _unit("u-12", "u-4 p1"),  # its unit is not integrated
        _unit("u-13", "u-5 p1"),  # already branched
        _unit("u-14", "u-1 p1", rationale=None),  # no rationale
        _unit("u-15", "u-1 p1", retain=False),  # parents on a delivery-only predicate
        _unit("u-16", "u-1 p3"),  # claimed by pending u-7
        _unit("u-17", "u-1 p1", rationale="  "),  # blank rationale
        _unit("u-18", "u-1 p1"),  # claimed by u-14 in this plan
        _unit("u-19", "u-1 p1"),
        _unit("u-20", "u-1 p5"),  # a legacy kind never re-checked
    ]}
    plan["task"][-2]["predicates"][0]["parents"] = "u-1 p1"  # not a list
    reply, errors = _decode(plan["task"], task)
    unknown = "is no predicate of a unit in the task (parents name accepted predicates as \"<unit> <pid>\")"
    assert reply is None and errors == [
        "unit 'u-14': a unit whose predicates carry parents needs a non-empty rationale (what the parents promised "
        "and how the children together keep it)",
        "unit 'u-17': a unit whose predicates carry parents needs a non-empty rationale (what the parents promised "
        "and how the children together keep it)",
        "unit 'u-19': p1 file_exists: parents must be a list of \"<unit> <pid>\" ids",
        f"unit 'u-10': p1 file_exists: parent 'u-9 p1' {unknown}",
        f"unit 'u-10': p1 file_exists: parent 'u-1 p9' {unknown}",
        f"unit 'u-10': p1 file_exists: parent 'u-1p1' {unknown}",
        "unit 'u-11': p1 file_exists: parent 'u-1 p2' is delivery-only (retain: false), not a lasting promise",
        "unit 'u-12': p1 file_exists: parent 'u-4 p1' belongs to u-4, which is not accepted (pending, not integrated)",
        "unit 'u-13': p1 file_exists: parent 'u-5 p1' already has children (u-6 p1)",
        "unit 'u-15': p1 file_exists: a delivery-only predicate (retain: false) cannot carry parents",
        "unit 'u-16': p1 file_exists: parent 'u-1 p3' is already claimed by unit 'u-7', whose branch is not settled",
        "unit 'u-17': p1 file_exists: parent 'u-1 p1' is already claimed by unit 'u-14', whose branch is not settled",
        "unit 'u-18': p1 file_exists: parent 'u-1 p1' is already claimed by unit 'u-14', whose branch is not settled",
        "unit 'u-20': p1 file_exists: parent 'u-1 p5' is never re-checked (kind exit_code)",
    ]


def test_a_bad_parent_goes_back_on_the_retry_path_before_any_reader_call(env: tuple[Root, Path]):
    root, workspace = env
    task = _decode_task(workspace)
    bad = _unit("u-10", "u-9 p1")
    runner = ScriptedRunner([_reply({"task": [bad]}), _reply({"task": [_unit("u-10", "u-1 p1")]})], [])
    asked: list[list[str]] = []

    def admit(units: list[Unit], spent: float) -> list[str]:
        asked.append([u.id for u in units])
        return []

    reply = director.next_move(task, Caps(), root, runner, admit=admit)
    assert [u.id for u in reply.move.units] == ["u-10"] and asked == [["u-10"]]  # the gate saw the valid move only
    first, retry = runner.calls[0][1], runner.calls[1][1]
    assert retry == first + ("\n\nYour previous answer was rejected:\n- unit 'u-10': p1 file_exists: parent 'u-9 p1' "
                             "is no predicate of a unit in the task (parents name accepted predicates as "
                             "\"<unit> <pid>\")\nAnswer again with exactly one JSON object.")


# --- the `branch` scenario, admitted by a scripted yes; the reader's argv, profile and clone ---


class Spy:
    """The scripted runner, recording what each reader call's working directory held while it ran."""

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


def test_the_branch_scenario_admits_grades_the_children_and_retires_the_parent(env: tuple[Root, Path]):
    root, workspace = env
    spy = Spy(scenario_runner("branch"))
    task = loop.run("branch", workspace, root, Caps(), spy, unattended=0.0, notify=None)
    assert _rested(task) and [m.kind for m in task.moves] == ["task", "task", "rest"]
    assert [(u.id, u.status, u.integrated) for u in task.units] == [
        ("u-1", "passed", True), ("u-2", "passed", True), ("u-3", "passed", True)]
    assert _seats(spy.scripted) == ["director", "worker", "director", "reader", "worker", "worker", "director"]
    log = _events(root, task.id)
    names = [(e["event"], e.get("seat"), e.get("unit")) for e in log]
    admitted = names.index(("branch_admitted", None, "u-2"))
    reader_call = log[names.index(("call", "reader", "u-2"))]
    assert names.index(("call", "reader", "u-2")) < admitted < names.index(("call", "worker", "u-2"))
    assert (reader_call["outcome"], reader_call["charge_usd"], reader_call["booked"], reader_call["cap_usd"]) == (
        "ok", 0.05, True, 1.5)
    event = log[admitted]
    assert (event["parents"], event["children"], event["call"], event["note"]) == (
        ["u-1 p1"], ["u-2 p1"], reader_call["id"],
        "metrics.py is the stats module under its new name, mean() included: nothing u-1 p1 verified is lost.")
    assert "to_director" not in event and task.cost_usd == pytest.approx(2.05)
    verdicts = {v["unit"]: v for v in _events(root, task.id, "verdict")}
    u1_candidate, u2_candidate = verdicts["u-1"]["commit"], verdicts["u-2"]["commit"]
    # a true rename: u-2's candidate has no stats.py, its child passed, the parent is excused (neither broke nor passed)
    assert _git(workspace, "ls-tree", "--name-only", u2_candidate).split("\n") == ["README.md", "metrics.py"]
    assert (verdicts["u-2"]["passed"], verdicts["u-2"]["passed_ids"], verdicts["u-2"]["recheck"]["predicates"]) == (
        True, ["p1"], [{"unit": "u-1", "id": "p1", "result": "excused", "children": ["u-2 p1"]}])
    assert "broke" not in verdicts["u-2"]
    accepted = _events(root, task.id, "branch_accepted")
    assert accepted == [{**accepted[0], "unit": "u-2", "parents": ["u-1 p1"], "children": ["u-2 p1"],
                         "candidate": u2_candidate, "last_passed": {"u-1 p1": u1_candidate}}]  # excused is not passed
    assert names.index(("integrated", None, "u-2")) + 1 == names.index(("branch_accepted", None, "u-2"))
    saved = load(root, task.id)
    parent = saved.units[0].predicates[0]
    assert (parent.kind, parent.args, parent.children, parent.last_passed_commit) == (
        "file_exists", {"path": "stats.py"}, ["u-2 p1"], u1_candidate)
    assert grade.rechecked(parent) is False and grade.rechecked(saved.units[1].predicates[0]) is True
    assert verdicts["u-3"]["recheck"]["predicates"] == [{"unit": "u-2", "id": "p1", "result": "passed"}]
    assert _git(workspace, "ls-tree", "--name-only", f"task/{task.id}").split("\n") == [
        "NOTES.md", "README.md", "metrics.py"]  # stats.py is gone from the accepted work
    assert console.snapshot(saved, Caps(), live=False)["units"][0]["predicates"] == [
        {"kind": "file_exists", "args": {"path": "stats.py"}, "children": ["u-2 p1"]}]
    assert console.snapshot(saved, Caps(), live=False)["units"][1]["predicates"][0]["parents"] == ["u-1 p1"]

    # row 5: the reader's argv, its sandbox profile, its fresh clone at the accepted tip
    argv, stdin, cwd, timeout = next(c for c in spy.calls if ScriptedRunner.seat(c[0]) == "reader")
    clone = reader.clone_dir(root, task)
    assert argv[:2] == [worker.sandbox_binary(), "-p"]
    assert argv[2] == reader.sandbox_profile(clone, workspace, root.console_file(task.id))
    assert argv[3:] == ["claude", "-p", "--setting-sources", "", "--strict-mcp-config", "--mcp-config",
                        '{"mcpServers":{}}', "--output-format", "json", "--max-budget-usd", "1.50",
                        "--model", "claude-opus-5", "--permission-mode", "dontAsk", "--tools", "Read", "Grep", "Glob",
                        "--allowedTools", "Read", "Grep", "Glob", "--system-prompt", "# reader prompt\n"]
    profile = argv[2]
    assert profile.startswith("(version 1)(allow default)(deny file-write*)(allow file-write* ")
    denied = f"(subpath {grade.quoted(str(clone))}) (subpath {grade.quoted(str(workspace.resolve()))})"
    assert f"(deny file-write* {denied})" in profile
    assert f"(deny file-read* (literal {grade.quoted(str(root.console_file(task.id).resolve()))}))" in profile
    assert "network" not in profile  # the model call needs its transport
    assert spy.reader_cwds == [(clone, u1_candidate, "")]  # a fresh, clean clone at the accepted tip
    assert cwd == clone != root.clones_dir(task.id).resolve() / "u-2" and timeout == 300 and not clone.exists()
    brief = json.loads(stdin)
    assert brief == {"unit": "u-2", "rationale": brief["rationale"], "accepted_tip": u1_candidate,
                     "question": gate.QUESTION,
                     "parents": [{"id": "u-1 p1", "kind": "file_exists", "args": {"path": "stats.py"},
                                  "last_passed_commit": u1_candidate}],
                     "children": [{"id": "u-2 p1", "kind": "file_contains",
                                   "args": {"path": "metrics.py", "text": "def mean"}, "parents": ["u-1 p1"]}]}
    for message in _messages(spy.scripted, "director"):
        assert stdin not in message and "accepted_tip" not in message and "dry-reader" not in message


# --- a no, or a reader call that gives no verdict, returns the plan; no worker call ---


@pytest.mark.parametrize("answer, reason, line", [
    (NO, "no", f"unit 'u-2': branch refused by the reader: {NO_NOTE}"),
    (_read("", 0.1, is_error=True, subtype="error_during_execution"), "error",
     "unit 'u-2': branch refused, the reader gave no verdict (error): the reader call ended in error: "
     "error_during_execution"),
])
def test_a_refused_branch_returns_the_plan_with_the_line_and_no_worker_call(
    env: tuple[Root, Path], answer: dict, reason: str, line: str
):
    root, workspace = env
    runner = ScriptedRunner(
        [_reply({"task": [U1]}), _reply({"task": [BRANCH]}), _reply({"task": [PLAIN]}), _reply({"rest": True})],
        [W_STATS, W_SHIM], [answer])
    task = loop.run("g", workspace, root, Caps(), runner, unattended=0.0, notify=None)
    assert _seats(runner) == ["director", "worker", "director", "reader", "director", "worker", "director"]
    assert _rested(task) and [(u.id, u.status) for u in task.units] == [("u-1", "passed"), ("u-2", "passed")]
    assert task.units[1].predicates[0].parents == [] and task.units[0].predicates[0].children == []
    first, retry = _messages(runner, "director")[1:3]
    rejected = f"\n\nYour previous answer was rejected:\n- {line}\nAnswer again with exactly one JSON object."
    assert retry == first + rejected
    (refused,) = _events(root, task.id, "branch_refused")
    assert (refused["unit"], refused["reason"], refused["parents"], refused["children"]) == (
        "u-2", reason, ["u-1 p1"], ["u-2 p1"]) and "to_director" not in refused
    reader_stdin = _messages(runner, "reader")[0]
    for message in _messages(runner, "director"):
        assert reader_stdin not in message and "accepted_tip" not in message and RATIONALE not in message
    assert sum(line in message for message in _messages(runner, "director")) == 1  # the retry only, never a note
    (call,) = [c for c in task.calls if c.seat == "reader"]
    assert call.booked and call.outcome == ("ok" if reason == "no" else "error")
    assert call.charge_usd == (0.05 if reason == "no" else 0.1) and refused["call"] == call.id
    assert _events(root, task.id, "branch_admitted") == [] and _events(root, task.id, "branch_accepted") == []


@pytest.fixture
def accepted(env: tuple[Root, Path]) -> tuple[Root, Task, Unit]:
    """A saved task whose u-1 (stats.py) is accepted at the workspace tip, and a branching u-2 proposing to take over
    u-1 p1."""
    root, workspace = env
    (workspace / "stats.py").write_text(MEAN)
    subprocess.run(["git", "add", "stats.py"], cwd=workspace, check=True)
    subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t.invalid", "commit", "-qm", "stats"],
                   cwd=workspace, check=True, capture_output=True)
    tip = _git(workspace, "rev-parse", "HEAD")
    u1 = Unit("u-1", "write stats.py", [Predicate("file_exists", {"path": "stats.py"})], status="passed",
              candidate=tip, integrated=True)
    task = Task(id="task-20261010T200000", goal="g", workspace=str(workspace), units=[u1], accepted=tip)
    _git(workspace, "update-ref", worker.accepted_ref(task), tip)
    save(root, task)
    set_current(root, task.id)
    u2 = Unit("u-2", "move mean()", [Predicate("file_contains", {"path": "metrics.py", "text": "def mean"},
                                               parents=["u-1 p1"])], rationale=RATIONALE)
    return root, task, u2


ANSWER = '{"verdict": "yes"|"no", "note": "<one line>"}'
OUTCOMES = [  # (reader entry, reason, note, call outcome, charge)
    (_read("", 0.1, is_error=True, subtype="error_during_execution"), "error",
     "the reader call ended in error: error_during_execution", "error", 0.1),
    ({"type": "result", "is_error": False, "total_cost_usd": 0.05}, "malformed",
     "the reader's receipt carries no result text", "ok", 0.05),
    (_read('Sure: {"verdict": "yes", "note": "fine"}'), "malformed", f"the answer is not one JSON object {ANSWER}",
     "ok", 0.05),
]


def test_every_reader_outcome_but_yes_refuses_and_is_charged_and_a_no_carries_its_note(
    accepted: tuple[Root, Task, Unit]
):
    root, task, u2 = accepted
    runner = ScriptedRunner([], [], [entry for entry, *_ in OUTCOMES] + [NO])
    total = 0.0
    for entry, reason, note, outcome, charge in OUTCOMES:
        admission = gate.admit(task, [u2], Caps(), root, runner)
        assert admission.refused == [f"unit 'u-2': branch refused, the reader gave no verdict ({reason}): {note}"]
        call, total = task.calls[-1], total + charge
        assert (call.seat, call.unit, call.outcome, call.charge_usd, call.booked) == (
            "reader", "u-2", outcome, charge, True), reason
        assert task.cost_usd == pytest.approx(total) and load(root, task.id).calls[-1].booked
        refused = _events(root, task.id, "branch_refused")[-1]
        assert (refused["reason"], refused["note"], refused["call"]) == (reason, note, call.id)
    line = f"unit 'u-2': branch refused by the reader: {NO_NOTE}"
    assert gate.admit(task, [u2], Caps(), root, runner, to_director=True).refused == [line]
    assert _events(root, task.id, "branch_refused")[-1]["to_director"] == line
    assert _events(root, task.id, "branch_admitted") == [] and len(runner.calls) == len(OUTCOMES) + 1


@pytest.mark.parametrize("text, verdict, note", [
    ('{"verdict": "yes", "note": "carried\\n  whole"}', "yes", "carried whole"),
    ('```json\n{"verdict": "no", "note": "lost"}\n```', "no", "lost"),
    ("Yes, they do.", None, f"the answer is not one JSON object {ANSWER}"),
    ('{"verdict": "yes"} {"verdict": "yes"}', None, f"the answer is not one JSON object {ANSWER}"),
    ('["yes"]', None, f"the answer must be exactly {ANSWER}"),
    ('{"verdict": "yes"}', None, f"the answer must be exactly {ANSWER}"),
    ('{"verdict": "yes", "note": "fine", "also": 1}', None, f"the answer must be exactly {ANSWER}"),
    ('{"verdict": "Yes", "note": "fine"}', None, "the verdict must be yes or no, not 'Yes'"),
    ('{"verdict": true, "note": "fine"}', None, "the verdict must be yes or no, not True"),
    ('{"verdict": "yes", "note": " "}', None, "the note must be a non-empty string"),
    ('{"verdict": "yes", "note": 1}', None, "the note must be a non-empty string"),
])
def test_only_exactly_one_verdict_object_is_an_answer(text: str, verdict: str | None, note: str):
    assert reader.parse(text) == (verdict, note)


def test_a_fenced_yes_admits(accepted: tuple[Root, Task, Unit]):
    root, task, u2 = accepted
    fenced = _read('```json\n{"verdict": "yes", "note": "carried whole"}\n```')
    assert gate.admit(task, [u2], Caps(), root, ScriptedRunner([], [], [fenced])).refused == []
    (admitted,) = _events(root, task.id, "branch_admitted")
    assert (admitted["note"], admitted["call"], task.calls[0].outcome) == ("carried whole", task.calls[0].id, "ok")


def test_the_reader_cap_is_clipped_like_any_seat_and_no_call_starts_under_the_floor(
    accepted: tuple[Root, Task, Unit]
):
    root, task, u2 = accepted
    task.cost_usd = 23.77
    runner = ScriptedRunner([], [], [YES, YES])
    gate.admit(task, [u2], Caps(reader_call_usd=2.0), root, runner)
    gate.admit(task, [u2], Caps(reader_call_usd=2.0), root, runner, spent=0.2)
    caps = [argv[argv.index("--max-budget-usd") + 1] for argv, _, _, _ in runner.calls]
    # the reader's own budget (2026-10-20): $7.00 less nothing, then less the first call's 0.05 and the unbooked 0.20,
    # both well above the $2.00 per-call cap; the allowance's $1.23 left never clips a reader call
    assert caps == ["2.00", "2.00"]
    from protean.state import Call  # noqa: PLC0415 — one booked reader call that leaves $0.40 of the reader's budget
    task.calls.append(Call(id="c-r", seat="reader", attempt=1, cap_usd=3.0, outcome="ok", charge_usd=6.5, booked=True))
    floor = gate.admit(task, [u2], Caps(), root, runner)
    assert len(runner.calls) == 2 and floor.refused == [
        "unit 'u-2': branch refused, the reader gave no verdict (floor): no reader call: reader_max_usd: remaining "
        "$0.40 below the $0.50 call floor"]


def test_an_interrupted_reader_call_books_the_unbooked_spend_with_its_cap(accepted: tuple[Root, Task, Unit]):
    root, task, u2 = accepted
    charged: list[float] = []

    def runner(argv, stdin, cwd, timeout):
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        gate.admit(task, [u2], Caps(), root, runner, spent=0.3, charge=charged.append)
    assert charged == [pytest.approx(1.8)] and task.calls[0].outcome == "interrupted"
    assert not reader.clone_dir(root, task).exists()


def test_the_reader_profile_on_the_real_sandbox_denies_writes_and_the_console_file(accepted: tuple[Root, Task, Unit]):
    root, task, _ = accepted
    clone = reader.clone_dir(root, task)
    worker.fresh_clone(task, clone, task.accepted)
    secret = root.console_file(task.id)
    secret.write_text('{"token": "t0ken"}')
    profile = reader.sandbox_profile(clone, Path(task.workspace), secret)
    sandbox = worker.sandbox_binary()  # the real binary; raises (fails the test) when absent

    def confined(*command: str) -> subprocess.CompletedProcess:
        return subprocess.run([sandbox, "-p", profile, *command], cwd=clone, capture_output=True, text=True)

    assert confined("/bin/cat", "stats.py").stdout == MEAN  # reads the clone
    write = confined("/usr/bin/touch", "new.txt")
    assert write.returncode != 0 and not (clone / "new.txt").exists()
    into_workspace = confined("/usr/bin/touch", str(Path(task.workspace) / "new.txt"))
    assert into_workspace.returncode != 0 and not (Path(task.workspace) / "new.txt").exists()
    token = confined("/bin/cat", str(secret))
    assert token.returncode != 0 and "t0ken" not in token.stdout


# --- parents stay in force until the branching unit is accepted; on its candidate, passing children excuse them ---


def _seen(root: Root, task_id: str) -> list[tuple]:
    return [(v["unit"], v["attempt"], v["passed_ids"], v["failed_ids"], v.get("broke"),
             (v.get("recheck") or {}).get("predicates")) for v in _events(root, task_id, "verdict")]


def test_a_branching_candidate_whose_child_fails_is_rejected_and_its_parents_are_untouched(env: tuple[Root, Path]):
    root, workspace = env
    runner = ScriptedRunner(
        [_reply({"task": [U1]}), _reply({"task": [BRANCH]}), _reply({"task": [NOTES]}),
         _reply({"ask": "u-2 failed: rename again?"})],
        [W_STATS, W_NO_MEAN, W_NO_MEAN, W_NOTES], [YES])
    task = loop.run("g", workspace, root, Caps(), runner, unattended=0.0, notify=None)
    assert task.terminal == "interrupted" and [(u.id, u.status) for u in task.units] == [
        ("u-1", "passed"), ("u-2", "failed"), ("u-3", "passed")]
    missing = {"p1": "file_contains: metrics.py does not contain 'def mean'"}
    assert _seen(root, task.id) == [
        ("u-1", 1, ["p1"], [], None, []),
        # the child fails: rejected on its own grade, nothing re-checked, nothing excused
        ("u-2", 1, [], ["p1"], None, None),
        ("u-2", 2, [], ["p1"], None, None),
        # the failed branch changed nothing: u-1 p1 is still re-checked on the next candidate
        ("u-3", 1, ["p1"], [], None, [{"unit": "u-1", "id": "p1", "result": "passed"}]),
    ]
    assert all(v["notes"]["p1"] == missing["p1"] for v in _events(root, task.id, "verdict") if v["unit"] == "u-2")
    parent = load(root, task.id).units[0].predicates[0]
    assert parent.children == [] and parent.last_passed_commit is None and grade.rechecked(parent)
    assert _events(root, task.id, "branch_accepted") == [] and len(_events(root, task.id, "branch_admitted")) == 1


def test_passing_children_do_not_excuse_an_unrelated_accepted_predicate(env: tuple[Root, Path]):
    root, workspace = env
    pair = {"id": "u-1", "intent": "write stats.py and keep.txt", "predicates": [
        {"kind": "file_exists", "args": {"path": "stats.py"}},
        {"kind": "file_contains", "args": {"path": "keep.txt", "text": "keep"}}]}
    runner = ScriptedRunner([_reply({"task": [pair]}), _reply({"task": [BRANCH]}), _reply({"rest": True})],
                            [W_PAIR, W_RENAME_CLOBBER, W_RENAME], [YES])
    task = loop.run("g", workspace, root, Caps(), runner, unattended=0.0, notify=None)
    assert _rested(task) and [(u.id, u.status) for u in task.units] == [("u-1", "passed"), ("u-2", "passed")]
    excused = {"unit": "u-1", "id": "p1", "result": "excused", "children": ["u-2 p1"]}
    assert _seen(root, task.id) == [
        ("u-1", 1, ["p1", "p2"], [], None, []),
        # the child passes and its parent is excused, but u-1 p2 is no parent: it is re-checked and breaks
        ("u-2", 1, ["p1"], [], {"u-1 p2": "file_contains: keep.txt does not contain 'keep'"},
         [excused, {"unit": "u-1", "id": "p2", "result": "failed"}]),
        ("u-2", 2, ["p1"], [], None, [excused, {"unit": "u-1", "id": "p2", "result": "passed"}]),
    ]
    rejected = next(v for v in _events(root, task.id, "verdict") if v["unit"] == "u-2" and v["attempt"] == 1)
    assert rejected["passed"] is False and "u-1 p1" not in rejected["to_director"]
    saved = load(root, task.id)
    assert [p.children for p in saved.units[0].predicates] == [["u-2 p1"], []]


def test_another_units_candidate_still_breaks_a_parent_while_the_branching_unit_is_pending(env: tuple[Root, Path]):
    root, workspace = env
    notes, branch = NOTES | {"id": "u-2"}, BRANCH | {"id": "u-3"}
    runner = ScriptedRunner([_reply({"task": [U1]}), _reply({"task": [notes, branch]}), _reply({"rest": True})],
                            [W_STATS, W_NOTES_DROP, W_NOTES, W_RENAME], [YES])
    task = loop.run("g", workspace, root, Caps(), runner, unattended=0.0, notify=None)
    assert _rested(task) and [(u.id, u.status) for u in task.units] == [
        ("u-1", "passed"), ("u-2", "passed"), ("u-3", "passed")]
    assert _seats(runner) == ["director", "worker", "director", "reader", "worker", "worker", "worker", "director"]
    assert _seen(root, task.id) == [
        ("u-1", 1, ["p1"], [], None, []),
        # u-3's branch is admitted but pending: u-2 is no branching unit, so removing stats.py breaks u-1 p1
        ("u-2", 1, ["p1"], [], {"u-1 p1": "file_exists: missing: stats.py"},
         [{"unit": "u-1", "id": "p1", "result": "failed"}]),
        ("u-2", 2, ["p1"], [], None, [{"unit": "u-1", "id": "p1", "result": "passed"}]),
        ("u-3", 1, ["p1"], [], None, [{"unit": "u-1", "id": "p1", "result": "excused", "children": ["u-3 p1"]},
                                      {"unit": "u-2", "id": "p1", "result": "passed"}]),
    ]
    u2_candidate = next(v["commit"] for v in _events(root, task.id, "verdict") if v["unit"] == "u-2" and v["passed"])
    parent = load(root, task.id).units[0].predicates[0]
    assert (parent.children, parent.last_passed_commit) == (["u-3 p1"], u2_candidate)


def test_excused_names_every_parent_once_all_children_pass_and_nothing_otherwise():
    split = Unit("u-8", "split", [Predicate("file_exists", {"path": "a.py"}, parents=["u-1 p1"]),
                                  Predicate("file_exists", {"path": "b.py"}, parents=["u-1 p1", "u-1 p1", "u-4 p2"]),
                                  Predicate("file_exists", {"path": "c.py"})], rationale="r")
    assert gate.excused(split, ["p1", "p2", "p3"]) == {"u-1 p1": ["u-8 p1", "u-8 p2"], "u-4 p2": ["u-8 p2"]}
    assert gate.excused(split, ["p1", "p2"]) == {"u-1 p1": ["u-8 p1", "u-8 p2"], "u-4 p2": ["u-8 p2"]}
    assert gate.excused(split, ["p1", "p3"]) == {} and gate.excused(split, []) == {}
    assert gate.excused(Unit("u-9", "plain", [Predicate("file_exists", {"path": "c.py"})]), ["p1"]) == {}


def test_the_thinker_may_rename_in_the_branching_unit_and_the_contract_documents_excused():
    prompt = (REPO / "brain" / "protean" / "thinker.md").read_text()
    (paragraph,) = [line for line in prompt.splitlines() if line.startswith("A retained promise can only be handed on")]
    assert "re-export" not in paragraph and "keep the old file" not in paragraph
    assert "That unit may itself rename or remove what the parents pinned, provided every child passes" in paragraph
    contract = (REPO / "docs" / "console-api.md").read_text()
    assert "or `excused`: a parent of the branching unit's passing children" in contract
    assert "records each as `excused` with the `children` that carry it" in contract


# --- hold: the plan is held first, admission runs after approve, a refusal reaches the next director message ---


def test_in_hold_admission_runs_after_approve_and_a_refusal_drops_the_plan_with_a_note(env: tuple[Root, Path]):
    root, workspace = env
    caps = Caps()
    runner = ScriptedRunner(
        [_reply({"task": [U1]}), _reply({"task": [BRANCH]}), _reply({"task": [PLAIN]})], [W_STATS], [NO])

    def approve_and_resume() -> Task:
        assert loop.offline_op(root, caps, "approve", {})["event"] == "approve"
        return loop.resume(root, caps, runner, unattended=0.0, notify=None)

    task = loop.run("g", workspace, root, caps, runner, unattended=0.0, notify=None, mode="hold")
    assert task.held.kind == "plan" and _seats(runner) == ["director"]
    task = approve_and_resume()  # u-1 runs; the branch plan is held, no reader call yet
    assert task.held.units[0].id == "u-2" and _seats(runner) == ["director", "worker", "director"]
    task = approve_and_resume()  # approved: the reader refuses in this tick, the director is told in the same tick
    assert _seats(runner) == ["director", "worker", "director", "reader", "director"]
    line = f"unit 'u-2': branch refused by the reader: {NO_NOTE}"
    (refused,) = _events(root, task.id, "branch_refused")
    assert refused["to_director"] == line and [u.id for u in task.units] == ["u-1"]
    assert json.loads(_messages(runner, "director")[2])["notes"] == [line]
    assert task.held.units[0].predicates[0].parents == []  # the corrected plan, held again
    assert [e["kind"] for e in _events(root, task.id, "approved")] == ["plan"]  # the refused plan never released


# --- records: the fields round-trip, an older record loads with the defaults ---


def test_branch_fields_round_trip_and_an_older_record_loads_without_them():
    child = Predicate("file_exists", {"path": "b.py"}, parents=["u-1 p1"])
    parent = Predicate("file_exists", {"path": "a.py"}, children=["u-2 p1"], last_passed_commit="abc")
    task = Task("t", "g", "/w", units=[Unit("u-1", "i", [parent]), Unit("u-2", "i", [child], rationale="r")])
    assert state.from_json(state.to_json(task)) == task
    old = json.loads(state.to_json(Task("t", "g", "/w", units=[Unit("u-1", "i", [Predicate("file_exists", {})])])))
    for key in ("parents", "children", "last_passed_commit"):
        del old["units"][0]["predicates"][0][key]
    del old["units"][0]["rationale"]
    loaded = state.from_json(json.dumps(old))
    assert loaded.units[0].predicates[0] == Predicate("file_exists", {}) and loaded.units[0].rationale == ""
    assert state.predicate_view(child) == {"kind": "file_exists", "args": {"path": "b.py"}, "parents": ["u-1 p1"]}
    assert state.predicate_view(parent) == {"kind": "file_exists", "args": {"path": "a.py"}, "children": ["u-2 p1"]}


# --- a reader call that leaves no receipt holds paid work (ruled 2026-10-11), it never refuses the branch ---


@pytest.mark.parametrize("entry, outcome", [({"stdout": "Error: boom"}, "no_receipt"), ({"timed_out": True}, "timed_out")])
def test_a_reader_call_without_a_receipt_holds_paid_work_instead_of_refusing(
    accepted: tuple[Root, Task, Unit], entry: dict, outcome: str
):
    root, task, u2 = accepted
    admission = gate.admit(task, [u2], Caps(), root, ScriptedRunner([], [], [entry]))
    assert (admission.refused, admission.waits) == ([], True)
    (call,) = [c for c in task.calls if c.seat == "reader"]
    assert (call.outcome, call.charge_usd, call.actual_usd, call.booked) == (outcome, 1.5, "UNKNOWN", True)
    assert [(q.reason, q.context) for q in state.open_questions(task)] == [("unknown_spend", {"unknown_spend": [call.id]})]
    assert _events(root, task.id, "branch_refused") == [] and task.cost_usd == pytest.approx(1.5)


def test_a_timed_out_admission_reader_holds_the_plan_and_the_run_until_yes(env: tuple[Root, Path]):
    root, workspace = env
    runner = ScriptedRunner([_reply({"task": [U1]}), _reply({"task": [BRANCH]}), _reply({"rest": True})],
                            [W_STATS, W_SHIM], [{"timed_out": True}, YES])
    task = loop.run("g", workspace, root, Caps(), runner, unattended=0.0, notify=None)
    assert _seats(runner) == ["director", "worker", "director", "reader"]
    assert task.terminal == "interrupted" and [u.id for u in task.units] == ["u-1"]
    assert (task.held.kind, task.held.admitted, [u.id for u in task.held.units]) == ("plan", False, ["u-2"])
    (held,) = state.open_questions(task)
    assert held.reason == "unknown_spend"
    loop.offline_op(root, Caps(), "answer", {"id": held.id, "text": "yes"})
    task = loop.resume(root, Caps(), runner, unattended=0.0, notify=None)
    assert _seats(runner) == ["director", "worker", "director", "reader", "reader", "worker", "director"]
    assert _rested(task) and [(u.id, u.status) for u in task.units] == [("u-1", "passed"), ("u-2", "passed")]
    assert [c.outcome for c in task.calls if c.seat == "reader"] == ["timed_out", "ok"]
