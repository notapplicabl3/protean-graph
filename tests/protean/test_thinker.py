"""The thinker: its message, the decode of one move plus `state`, the no-progress check and its
prompt. Goal mode's `plan`, `decode`, `view` and `director.md` are gone (order 2): the loop calls the thinker."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from protean import loop
from protean.graph import director, gate, worker
from protean.graph.budget import Caps
from protean.dry import REPO, ScriptedRunner, make_workspace
from protean.state import (
    DIRECTOR, HOLD, MOVE_KINDS, PLAN, RUNTIME, Decision, Held, Idea, Move, OpenItem, Predicate, Root, Task, Thought,
    Unit, add_question, append_log, append_transcript, save, set_current,
)

UNIT = {"id": "u-2", "intent": "write b.txt", "predicates": [{"kind": "file_exists", "args": {"path": "b.txt"}}]}
STATE = {"standing": "There is no b.txt yet. One way to add it is open.", "open": [{"id": "o1", "line": "where b.txt goes"}]}
IDEA = {"id": "i1", "case": "put it at the root", "cost": "one small explore", "serves": "o1"}
PROMPT = REPO / "brain" / "protean" / "thinker.md"


def _ideas(n: int) -> list[dict]:
    return [{**IDEA, "id": f"i{k}"} for k in range(1, n + 1)]


def _decode(obj: object, task: Task) -> tuple[director.MoveReply | None, list[str]]:
    return director.decode_move(obj if isinstance(obj, str) else json.dumps(obj), task)


@pytest.fixture
def session(task: Task) -> Task:
    """The conftest task carried as a session: its thought, two thoughts, the state the last move left, a decision
    and two moves."""
    task.thoughts = [Thought("keep it small", "2026-10-17T09:00:00+00:00"),
                     Thought("b.txt is the point", "2026-10-17T09:02:00+00:00")]
    task.standing = "a.txt is planned. Nothing has run yet."
    task.open = [OpenItem("o1", "whether b.txt is needed too")]
    task.decided = [Decision("b.txt is the point", "2026-10-17T09:02:00+00:00", 2)]
    task.units[0].summary = "wrote a.txt [grade: passed]"
    task.moves = [
        Move("m1", "ideas", "two ways to start", 0.1, "picked i2", ideas=[Idea("i1", "a first", "a task", "o1"),
                                                                       Idea("i2", "b first", "a task", "o1")]),
        Move("m2", "explore", "try b.txt at the root", 0.4, "landed", branch="explore/s/1", report="b.txt fits."),
        Move("m3", "task", "write a.txt", 0.6, "integrated", branch="task/s", units=["u1"]),
    ]
    task.cost_usd = 1.25
    return task


# The message.

def test_the_message_carries_the_record_every_thought_the_units_the_moves_the_budget_and_the_notes(session: Task):
    message = director.thinker_message(session, Caps(), ["u1 broke accepted work"])
    assert list(message) == ["thought", "standing", "open", "decided", "thoughts", "operator", "workspace", "tick",
                             "budget_left_usd", "ticks_left", "max_attempts_per_unit", "units", "moves", "notes"]
    assert (message["thought"], message["standing"]) == ("make a.txt", "a.txt is planned. Nothing has run yet.")
    assert message["open"] == [{"id": "o1", "line": "whether b.txt is needed too"}]
    assert message["decided"] == [{"text": "b.txt is the point", "created": "2026-10-17T09:02:00+00:00", "think": 2}]
    assert message["thoughts"] == ["keep it small", "b.txt is the point"]  # every thought, in order
    assert (message["budget_left_usd"], message["ticks_left"], message["notes"]) == (
        23.75, 100, ["u1 broke accepted work"])
    assert message["units"] == director._units_view(session)  # the units as the seat has always seen them
    assert message["units"][0]["summary"] == "wrote a.txt [grade: passed]" and message["units"][0]["status"] == "pending"
    assert message["moves"] == [
        {"id": "m1", "kind": "ideas", "line": "two ways to start", "cost_usd": 0.1, "outcome": "picked i2",
         "ideas": [{"id": "i1", "case": "a first", "cost": "a task", "serves": "o1"},
                   {"id": "i2", "case": "b first", "cost": "a task", "serves": "o1"}]},
        {"id": "m2", "kind": "explore", "line": "try b.txt at the root", "cost_usd": 0.4, "outcome": "landed",
         "branch": "explore/s/1", "unverified_report": "b.txt fits."},
        {"id": "m3", "kind": "task", "line": "write a.txt", "cost_usd": 0.6, "outcome": "integrated",
         "branch": "task/s", "units": ["u1"]},
    ]


def test_the_budget_left_is_the_allowance_less_the_cost_and_no_other_budget_figure_rides(session: Task):
    """The allowance: `max_usd`, plus the extensions, plus what active work accrued."""
    session.extra_usd, session.active_seconds = 2.0, 900.0  # $5.00 accrued
    message = director.thinker_message(session, Caps(), [])
    assert message["budget_left_usd"] == round(25.0 + 2.0 + 5.0 - 1.25, 2) == 30.75
    assert [key for key in message if "usd" in key] == ["budget_left_usd"]  # the 2026-10-17 sitting figure is gone


def test_the_answer_to_its_last_ask_rides_and_no_other_question_does(session: Task):
    assert "answer" not in director.thinker_message(session, Caps(), [])
    first = add_question(session, DIRECTOR, "question", "which folder?")
    first.status, first.answer = "answered", "the root"
    runtime = add_question(session, RUNTIME, "unknown_spend", "a call left no receipt; continue?")
    runtime.status, runtime.answer = "answered", "yes"
    assert director.thinker_message(session, Caps(), [])["answer"] == {
        "id": "q1", "question": "which folder?", "answer": "the root"}
    later = add_question(session, DIRECTOR, "question", "which name?")
    assert "answer" not in director.thinker_message(session, Caps(), [])  # its last ask is still open
    later.status, later.answer = "answered", "b.txt"
    assert director.thinker_message(session, Caps(), [])["answer"] == {
        "id": "q3", "question": "which name?", "answer": "b.txt"}


NOTE = "the children carry only half of what u-1 p1 promised"
# Seeded where the reader's exchange lives (its brief, the hunks it read, its receipt and call record) and in the
# runtime's and the editor's records; none may reach a thinker message.
SEEDS = ("READER-SESSION-MARK", "READER-SUBTYPE-MARK", "777123", "0.0317", "RUNTIME-QUESTION-MARK",
         "RUNTIME-ANSWER-MARK", "TRANSCRIPT-LINE-MARK")
BRIEF_SEEDS = {"branch": ("BRIEF-RATIONALE-MARK", gate.QUESTION, '"last_passed_commit"'),
               "weakening": ("HUNK-MARK", gate.WEAKENING_QUESTION, '"hunks"')}
# The shapes of the reader's records (its raw answer, the gate's verdict events, a call record), never in a message.
SHAPES = ('"verdict"', '"reason"', '"weakening"', '"branch_refused"', '"accepted_tip"', '"rationale"', '"calls"',
          '"charge_usd"', '"receipt_cost_usd"', '"message_bytes"')
BRANCH = {"id": "u-2", "intent": "rename stats.py to metrics.py", "rationale": "BRIEF-RATIONALE-MARK: it moves whole",
          "predicates": [{"kind": "file_contains", "args": {"path": "metrics.py", "text": "def mean"},
                          "parents": ["u-1 p1"]}]}
REST = {"rest": True, "state": {"standing": "The rename is off for now.", "open": []}}


def _git(cwd: Path, *args: str) -> str:
    return subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t.invalid", *args], cwd=cwd,
                          capture_output=True, text=True, check=True).stdout.strip()


def _said(obj: dict) -> dict:
    return {"type": "result", "is_error": False, "session_id": "s-1", "total_cost_usd": 0.1, "result": json.dumps(obj)}


def _reader_says(verdict: str, **kind: str) -> dict:
    """The reader's receipt, every field of it seeded: its session, subtype, tokens and cost."""
    return {"type": "result", "is_error": False, "subtype": "READER-SUBTYPE-MARK", "session_id": "READER-SESSION-MARK",
            "total_cost_usd": 0.0317, "usage": {"input_tokens": 777123},
            "result": json.dumps({"verdict": verdict, **kind, "note": NOTE})}


def _reader_world(tmp_path: Path, root: Root) -> Task:
    """A saved session whose u-1 is accepted at the workspace tip (p1: stats.py exists; p2: a retained command that
    protects `tests`), with an answered runtime question and an editor transcript line on record."""
    root.path.mkdir(parents=True)
    for prompt in (root.thinker_prompt, root.editor_prompt, root.reader_prompt):
        prompt.write_text("p\n")
    workspace = make_workspace(tmp_path / "ws")
    (workspace / "tests").mkdir()
    (workspace / "stats.py").write_text("def mean(xs):\n    return sum(xs) / len(xs)\n")
    (workspace / "tests" / "check.txt").write_text("a\n")
    _git(workspace, "add", "-A")
    _git(workspace, "commit", "-qm", "u-1")
    tip = _git(workspace, "rev-parse", "HEAD")
    u1 = Unit("u-1", "write stats.py", [Predicate("file_exists", {"path": "stats.py"}),
                                        Predicate("command", {"cmd": "true"}, protected=["tests"])],
              status="passed", candidate=tip, integrated=True)
    task = Task(id="task-20261018T100000", thought="t", workspace=str(workspace), units=[u1], accepted=tip)
    _git(workspace, "update-ref", worker.accepted_ref(task), tip)
    asked = add_question(task, RUNTIME, "no_progress", "RUNTIME-QUESTION-MARK")
    asked.status, asked.answer, asked.actor = "answered", "RUNTIME-ANSWER-MARK", "console"
    asked.disposition = "continued"
    save(root, task)
    set_current(root, task.id)
    for event in ({"event": "question", "id": asked.id, "asker": RUNTIME, "reason": "no_progress",
                   "text": "RUNTIME-QUESTION-MARK"},
                  {"event": "answer", "id": asked.id, "asker": RUNTIME, "text": "RUNTIME-ANSWER-MARK",
                   "actor": "console"},
                  {"event": "runtime_answer", "id": asked.id, "reason": "no_progress", "disposition": "continued"}):
        append_log(root, task.id, event)
    append_transcript(root, task.id, {"event": "transcript", "line": "TRANSCRIPT-LINE-MARK"})
    return task


@pytest.mark.parametrize("path", ["refused in its call", "refused at its release", "weakened"])
def test_the_reader_s_one_line_note_reaches_the_thinker_and_its_brief_hunks_receipt_and_call_never_do(
        tmp_path: Path, root: Root, path: str):
    """the reader's one-line refusal or weakening note is grade feedback and reaches the
    thinker on each of its three real paths (an `auto` plan's retry stdin, a released held plan's `to_director` note,
    a failed unit's summary); the reader's brief, the hunks it read, its receipt and its call record never do, and
    neither does a runtime question or answer or an editor transcript line."""
    task, refused = _reader_world(tmp_path, root), f"unit 'u-2': branch refused by the reader: {NOTE}"
    if path == "refused in its call":  # auto: the gate refuses inside the thinker's call, which retries once
        runner = ScriptedRunner([_said({"task": [BRANCH], "state": {"standing": "u-2 renames it.", "open": []}}),
                                 _said(REST)], [], [_reader_says("no")])
        loop.run_tick(task, Caps(), root, runner)
        kind, arrived = "branch", lambda text: f"\n\nYour previous answer was rejected:\n- {refused}\n" in text
    elif path == "refused at its release":  # hold: the approved plan's admission, then the thinker in the same tick
        task.mode = HOLD
        task.held = Held(kind=PLAN, approved=True, units=[Unit(
            "u-2", BRANCH["intent"], [Predicate("file_contains", {"path": "metrics.py", "text": "def mean"},
                                                parents=["u-1 p1"])], rationale=BRANCH["rationale"])])
        runner = ScriptedRunner([_said(REST)], [], [_reader_says("no")])
        loop.run_tick(task, Caps(), root, runner)
        kind, arrived = "branch", lambda text: json.loads(text)["notes"] == [refused]
    else:  # the candidate weakened a protected path: rejected on its last attempt, the thinker moves next tick
        task.units.append(Unit("u-2", "write b.txt", [Predicate("file_exists", {"path": "b.txt"})]))
        weakens = {"write": {"b.txt": "b\n", "tests/check.txt": "HUNK-MARK\n"}, "commit": True, "exit_code": 0,
                   "result": "wrote b.txt", "total_cost_usd": 0.5}
        runner = ScriptedRunner([_said(REST)], [weakens], [_reader_says("yes", kind="weakened assertions")])
        for _ in range(2):
            loop.run_tick(task, Caps(max_attempts_per_unit=1), root, runner)
        kind, arrived = "weakening", lambda text: json.loads(text)["units"][1]["summary"].endswith(
            f" [grade: weakened tests/check.txt: weakened assertions: {NOTE}]")
    stdins = {ScriptedRunner.seat(argv): stdin for argv, stdin, _, _ in runner.calls}  # each seat's last message
    text, call = stdins["director"], next(c for c in task.calls if c.seat == "reader")
    world = stdins["reader"] + "".join(record.read_text(encoding="utf-8") for record in (
        root.task_file(task.id), root.log_file(task.id), root.transcript_file(task.id)))
    assert arrived(text)  # the one line arrives
    assert [seed for seed in SEEDS + BRIEF_SEEDS[kind] if seed not in world] == []  # every seed is on record
    assert [mark for mark in SEEDS + BRIEF_SEEDS["branch"] + BRIEF_SEEDS["weakening"] + SHAPES + (f'"{call.id}"',)
            if mark in text] == []  # and none of it reached the thinker
    assert (call.session_id, call.tokens["input"]) == ("READER-SESSION-MARK", 777123)


# Decode: exactly one of the five moves plus `state`.

def test_decode_accepts_each_move_with_its_state(task: Task):
    ideas, errors = _decode({"ideas": _ideas(3), "state": STATE}, task)
    assert errors == [] and ideas.kind == "ideas" and ideas.ideas[0] == Idea("i1", "put it at the root",
                                                                              "one small explore", "o1")
    assert (ideas.standing, ideas.open) == (STATE["standing"], [OpenItem("o1", "where b.txt goes")])
    assert len(_decode({"ideas": _ideas(7), "state": STATE}, task)[0].ideas) == 7
    explore = _decode({"explore": "  try b.txt at the root  ", "state": STATE}, task)[0]
    assert (explore.kind, explore.brief) == ("explore", "try b.txt at the root")
    planned = _decode(json.dumps({"task": [UNIT], "state": STATE}), task)[0]
    (unit,) = planned.units
    assert planned.kind == "task" and (unit.id, unit.status, unit.predicates) == (
        "u-2", "pending", [Predicate("file_exists", {"path": "b.txt"})])
    asked = _decode({"ask": "which folder?", "state": STATE}, task)[0]
    assert (asked.kind, asked.question) == ("ask", "which folder?")
    rest = _decode('Resting. {"rest": true, "state": {"standing": " All done for now. ", "open": []}}', task)[0]
    assert (rest.kind, rest.standing, rest.open) == ("rest", "All done for now.", [])
    assert MOVE_KINDS == ("ideas", "explore", "task", "ask", "rest")


def test_a_status_carried_in_a_task_move_is_ignored(task: Task):
    claimed = {**UNIT, "status": "passed", "attempts": 3, "branch": "main"}
    (unit,) = _decode({"task": [claimed], "state": STATE}, task)[0].units
    assert (unit.status, unit.attempts, unit.branch) == ("pending", 0, None)


SIX = "One. Two. Three. Four. Five. Six."


@pytest.mark.parametrize("obj, error", [
    ({"ask": "which?", "rest": True, "state": STATE}, "the reply must carry exactly one move of ideas, explore, task, "
                                                      "ask, rest; it carried ['ask', 'rest']"),
    ({"state": STATE}, "it carried none"),
    ({"rest": True}, "the reply must carry state"),
    ({"rest": True, "state": STATE, "done": True}, "drop 'done'"),
    ({"rest": True, "state": STATE, "units": [UNIT]}, "drop 'units'"),
    ({"rest": True, "state": {**STATE, "standing": SIX}}, "state.standing must be at most 5 sentences; it has 6"),
    ({"rest": True, "state": {**STATE, "decided": ["b.txt"]}}, "state cannot carry decided: only the brain writes it"),
    ({"rest": True, "state": {**STATE, "moves": []}}, "state cannot carry moves: only the brain writes it"),
    ({"rest": True, "state": {**STATE, "notes": "x"}}, "state holds only standing and open; it carried ['notes']"),
    ({"rest": True, "state": {"open": []}}, "state.standing must be a non-empty string"),
    ({"rest": True, "state": {**STATE, "standing": "  "}}, "state.standing must be a non-empty string"),
    ({"rest": True, "state": {"standing": "Fine."}}, "state.open must be a list"),
    ({"rest": True, "state": {**STATE, "open": [{"id": "o1"}]}}, "open 'o1': line must be one non-empty line"),
    ({"rest": True, "state": {**STATE, "open": [{"id": "o1", "line": "a\nb"}]}}, "open 'o1': line must be one"),
    ({"rest": True, "state": {**STATE, "open": [{"id": "o1", "line": "a"}, {"id": "o1", "line": "b"}]}},
     "open 'o1': id is already used"),
    ({"rest": True, "state": {**STATE, "open": [{"line": "a"}]}}, "open 1: id must match"),
    ({"rest": True, "state": {**STATE, "open": ["a"]}}, "open 1: must be an object"),
    ({"rest": True, "state": "fine"}, "state must be an object"),
    ({"ideas": _ideas(2), "state": STATE}, "ideas must be a list of 3 to 7 items; it has 2"),
    ({"ideas": _ideas(8), "state": STATE}, "ideas must be a list of 3 to 7 items; it has 8"),
    ({"ideas": "three", "state": STATE}, "ideas must be a list of 3 to 7 items"),
    ({"ideas": [*_ideas(2), {**IDEA, "id": "i3", "serves": "o9"}], "state": STATE},
     "idea 'i3': serves 'o9', which is no open id in this state (o1)"),
    ({"ideas": [*_ideas(2), {**IDEA, "id": "i3", "cost": ""}], "state": STATE}, "idea 'i3': cost must be one"),
    ({"ideas": [*_ideas(2), {**IDEA, "id": "i2"}], "state": STATE}, "idea 'i2': id is already used"),
    ({"ideas": [*_ideas(2), "a third"], "state": STATE}, "idea 3: must be an object"),
    ({"explore": "", "state": STATE}, "explore must be one non-empty brief"),
    ({"explore": ["a", "b"], "state": STATE}, "explore must be one non-empty brief"),
    ({"ask": "  ", "state": STATE}, "ask must be one non-empty question"),
    ({"rest": False, "state": STATE}, "rest must be true"),
    ({"task": [], "state": STATE}, "units must be a non-empty list"),
    ({"task": [{**UNIT, "predicates": [{"kind": "file_exists"}]}], "state": STATE}, "predicates must be a list"),
    ({"task": [{**UNIT, "needs": ["u-2"]}], "state": STATE}, "unit 'u-2': needs itself"),
    ("I think we should rest now.", "the reply holds no JSON object"),
])
def test_decode_returns_every_other_shape_with_its_errors(task: Task, obj: object, error: str):
    reply, errors = _decode(obj, task)
    assert reply is None and any(error in e for e in errors), errors


def test_every_error_of_one_reply_comes_back_together(task: Task):
    reply, errors = _decode({"ask": "?", "explore": "x", "state": {"standing": SIX, "decided": []}}, task)
    assert reply is None and len(errors) == 4, errors


@pytest.mark.parametrize("units", [
    [UNIT],
    [UNIT, UNIT],
    [{**UNIT, "id": "../escape"}],
    [{**UNIT, "predicates": []}],
    [{**UNIT, "predicates": [{"kind": "exit_code", "args": {"expected": 0}}]}],
    [{**UNIT, "predicates": [{"kind": "file_exists", "args": {"path": "/etc/passwd"}}]}],
    [{**UNIT, "predicates": [{"kind": "command", "args": {"cmd": "true"}}]}],
    [{**UNIT, "predicates": [{**UNIT["predicates"][0], "retain": "yes"}]}],
    [{**UNIT, "predicates": [{**UNIT["predicates"][0], "parents": ["u1 p1"]}]}],
    [{**UNIT, "needs": ["u-9"]}],
])
def test_a_task_move_s_units_are_validated_by_the_plan_s_own_code(task: Task, units: list,
                                                                  monkeypatch: pytest.MonkeyPatch):
    """Unit validation inside a `task` move is `_decode_units` itself, never a copy: the same units give the same
    errors as a plan, and the call goes through it."""
    plan_units, plan_errors = director._decode_units(json.loads(json.dumps(units)), task)
    plan_reply = None if plan_errors else plan_units
    seen = []
    real = director._decode_units
    monkeypatch.setattr(director, "_decode_units", lambda items, t: seen.append(items) or real(items, t))
    move, move_errors = _decode({"task": units, "state": STATE}, task)
    assert seen == [units] and move_errors == plan_errors
    assert (move is None) == (plan_reply is None)
    if move is not None:
        assert move.units == plan_reply


# No progress.

@pytest.mark.parametrize("last, standing, same", [
    ("Two ways are open.", "Two ways are open.", True),
    ("Two ways are open.", "Two ways are open. ", False),
    ("Two ways are open.", "two ways are open.", False),
    ("The café is open.", "The café is open.", False),  # the same to the eye, not the same bytes
    ("", "Two ways are open.", False),
])
def test_no_progress_is_a_byte_identical_standing(last: str, standing: str, same: bool):
    assert director.no_progress(last, standing) is same


# The prompt.

def test_the_thinker_prompt_names_the_five_moves_state_and_the_rules_of_a_task_move():
    prompt = PROMPT.read_text(encoding="utf-8")
    assert not (REPO / "brain" / "protean" / "director.md").exists()  # retired with goal mode (order 2)
    lines = prompt.splitlines()
    for kind in MOVE_KINDS:
        assert any(line.startswith(f'- `{{"{kind}"') for line in lines), kind
    assert "exactly ONE JSON object" in prompt and "one move plus `state`" in prompt
    (state_rule,) = [line for line in lines if line.startswith('`state` is `{"standing"')]
    assert "at most 5 plain sentences" in state_rule and "no jargon" in state_rule and "no file paths" in state_rule
    assert "never holds `decided` or `moves`" in state_rule
    assert "no progress" in prompt and "unchanged" in prompt
    (ideas_rule,) = [line for line in lines if line.startswith('- `{"ideas"')]
    (rest_rule,) = [line for line in lines if line.startswith('- `{"rest"')]
    assert "3 to 7" in ideas_rule and "waits for the operator" in ideas_rule and "waits for the operator" in rest_rule
    (thoughts_rule,) = [line for line in lines if line.startswith("The operator's `thoughts`")]
    assert "direction, an answer, a possibility or a decision" in thoughts_rule
    assert "never" in thoughts_rule and "`decided`" in thoughts_rule
    assert "moves the thought forward" in prompt


KINDS = [  # the three predicate lines `director.md` carried when it retired (order 2), word for word
    '- `{"kind": "file_exists", "args": {"path": "relative/path"}}`',
    '- `{"kind": "file_contains", "args": {"path": "relative/path", "text": "exact substring"}}`',
    '- `{"kind": "command", "args": {"cmd": "uv run pytest -q", "expect_exit": 0}, "protected": ["tests"]}` — run in '
    'the clone after the worker finishes; `protected` is explained below',
]


def test_the_task_move_keeps_the_director_s_predicate_rules_word_for_word():
    """The `task` move's units are a plan's, so the predicate kinds, retain, parents, protected and needs read as the
    retired director prompt had them; the goal-mode words (descope, done, the goal) are not carried over."""
    prompt = PROMPT.read_text(encoding="utf-8")
    lines = prompt.splitlines()
    assert [line for line in lines if line.startswith('- `{"kind"')] == KINDS
    for start in ("Every unit needs at least one predicate.", "A retained promise can only be handed on."):
        (paragraph,) = [line for line in lines if line.startswith(start)]
        assert len(paragraph) > len(start) + 200, start
    assert prompt.count('"retain": false') == 1 and '"protected"' in prompt and '"parents"' in prompt
    assert '"needs"' in prompt and "`blocked`" in prompt and "`max_attempts_per_unit`" in prompt
    assert "descope" not in prompt and '"done"' not in prompt and "goal" not in prompt
