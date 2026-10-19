"""The director seat in its thinker role: argv and seat cwd, mint-then-resume, the one retry, the call floor, a status
it cannot set, the receipt-less and capped calls, an unchanged standing returned at once, and the `task` move's units
decoded by the plan's own validation (needs, retain, a cut-off reply)."""

from __future__ import annotations

import json

import pytest

from protean.graph import director
from protean.graph.budget import Caps
from protean.dry import REPO, ScriptedRunner
from protean.runner import RunResult
from protean.state import Predicate, Root, Task

UNIT = {"id": "u-2", "intent": "write b.txt", "predicates": [{"kind": "file_exists", "args": {"path": "b.txt"}}]}
STATE = {"standing": "b.txt is planned.", "open": []}
PROMPT = REPO / "brain" / "protean" / "thinker.md"


def _reply(obj: object, cost: float = 0.25, session: str = "sess-1") -> dict:
    text = obj if isinstance(obj, str) else json.dumps(obj)
    return {"type": "result", "is_error": False, "session_id": session, "total_cost_usd": cost, "result": text}


def _task_move(*units: dict, standing: str = STATE["standing"]) -> dict:
    return {"task": list(units), "state": {**STATE, "standing": standing}}


@pytest.fixture
def seeded(root: Root) -> Root:
    root.path.mkdir(parents=True)
    root.thinker_prompt.write_text("# thinker prompt\n")
    return root


def test_first_call_mints_with_the_prompt_and_later_calls_resume(seeded: Root, task: Task):
    runner = ScriptedRunner([_reply(_task_move(UNIT)), _reply({"rest": True, "state": {**STATE, "standing": "x."}})],
                            [])
    first = director.next_move(task, Caps(), seeded, runner)
    assert first.session_id == "sess-1" and first.kind == "task" and first.cost_usd == 0.25
    assert [u.id for u in first.move.units] == ["u-2"] and first.move.standing == "b.txt is planned."
    argv, stdin, cwd, timeout = runner.calls[0]
    assert argv == [
        "claude", "-p", "--setting-sources", "", "--output-format", "json", "--max-budget-usd", "2.00",
        "--model", "claude-opus-5",
        "--permission-mode", "dontAsk", "--tools", "Read", "Grep", "Glob", "--allowedTools", "Read", "Grep", "Glob",
        "--add-dir", "/tmp/ws", "--system-prompt", "# thinker prompt\n",
    ]
    assert cwd == seeded.seat_dir(task.id) and cwd.is_dir() and timeout == 300
    message = json.loads(stdin)
    assert (message["thought"], message["units"][0]["id"], message["budget_left_usd"], message["notes"]) == (
        "make a.txt", "u1", 25.0, [])
    task.director_session, task.standing = first.session_id, first.move.standing
    assert director.next_move(task, Caps(), seeded, runner).kind == "rest"
    argv2, _, cwd2, _ = runner.calls[1]
    assert argv2[-4:] == ["--resume", "sess-1", "--system-prompt", "# thinker prompt\n"] and cwd2 == cwd


def test_a_malformed_reply_is_retried_once_then_becomes_the_runtime_s_question(seeded: Root, task: Task):
    runner = ScriptedRunner([_reply("Let us start with a.txt."), _reply({"task": [], "state": STATE}, cost=0.5)], [])
    reply = director.next_move(task, Caps(), seeded, runner)
    assert len(runner.calls) == 2 and reply.cost_usd == 0.75 and reply.move is None and reply.kind == "invalid"
    assert reply.question == "The thinker returned an invalid reply twice: units must be a non-empty list"
    assert "Your previous answer was rejected:\n- the reply holds no JSON object" in runner.calls[1][1]
    assert runner.calls[1][0][-4:-2] == ["--resume", "sess-1"]


def test_the_retry_recovers_a_misspelled_predicate(seeded: Root, task: Task):
    bad = _task_move({**UNIT, "predicates": [{"kind": "exit_code", "args": {"expected": 0}}]})
    fenced = "Move:\n```json\n" + json.dumps(_task_move(UNIT)) + "\n```"
    runner = ScriptedRunner([_reply(bad), _reply(fenced)], [])
    reply = director.next_move(task, Caps(), seeded, runner)
    assert [u.id for u in reply.move.units] == ["u-2"] and reply.question is None
    assert "unit 'u-2': p1 exit_code: unknown arg 'expected'" in runner.calls[1][1]


def test_calls_are_clipped_to_what_is_left_and_no_retry_goes_below_the_floor(seeded: Root, task: Task):
    task.cost_usd = 24.0
    runner = ScriptedRunner([_reply("no move here", cost=0.6)], [])
    reply = director.next_move(task, Caps(), seeded, runner)
    argv = runner.calls[0][0]
    assert argv[argv.index("--max-budget-usd") + 1] == "1.00" and len(runner.calls) == 1
    assert (reply.stop, reply.errors, reply.kind) == (
        "max_usd: remaining $0.40 below the $0.50 call floor", ["the reply holds no JSON object"], "stopped")
    assert reply.cost_usd == 0.6 and reply.question is None


def test_a_status_carried_in_a_task_move_is_ignored(seeded: Root, task: Task):
    claimed = {**UNIT, "status": "passed", "attempts": 3, "summary": "already done", "branch": "main"}
    reply = director.next_move(task, Caps(), seeded, ScriptedRunner([_reply(_task_move(claimed))], []))
    (unit,) = reply.move.units
    assert (unit.status, unit.attempts, unit.summary, unit.branch) == ("pending", 0, "", None)


def test_a_unit_s_summary_for_the_operator_is_kept_stripped_and_never_required(seeded: Root, task: Task):
    """The operator's plain sentence beside the editor's intent is optional, so a move without one decodes as
    every plan before it did; it lands in `plain_summary`, never in the worker's `summary`."""
    plain = {**UNIT, "summary": "  Adds b.txt so the demo has a second file.  "}
    bare, odd = {**UNIT, "id": "u-3"}, {**UNIT, "id": "u-4", "summary": None}
    reply = director.next_move(task, Caps(), seeded, ScriptedRunner([_reply(_task_move(plain, bare, odd))], []))
    assert (reply.kind, reply.errors) == ("task", [])
    assert [(u.id, u.plain_summary, u.summary) for u in reply.move.units] == [
        ("u-2", "Adds b.txt so the demo has a second file.", ""), ("u-3", "", ""), ("u-4", "", "")]


def test_an_unchanged_standing_comes_back_at_once_unadmitted_and_unretried(seeded: Root, task: Task):
    """the call returns the reply marked `unchanged`, with no retry and no admission, so no
    reader is paid for a move the loop will not act on."""
    task.standing = STATE["standing"]
    admitted: list[list[str]] = []
    runner = ScriptedRunner([_reply(_task_move(UNIT))], [])
    reply = director.next_move(task, Caps(), seeded, runner, admit=lambda units, spent: admitted.append(units) or [])
    assert (reply.kind, reply.unchanged, reply.move.kind, len(runner.calls), admitted) == (
        "no_progress", True, "task", 1, [])


@pytest.mark.parametrize("stdout, timed_out, why", [
    ("", True, "the call timed out after 300s"),
    ("Error: not logged in", False, "stdout was not one JSON object (exit 1): "),
])
def test_timeouts_and_garbage_leave_no_receipt_so_the_cap_counts_and_nothing_is_retried(
    seeded: Root, task: Task, stdout: str, timed_out: bool, why: str
):
    calls = []
    def runner(argv, stdin, cwd, timeout):
        calls.append(argv)
        return RunResult(-9 if timed_out else 1, stdout, "", timed_out)
    reply = director.next_move(task, Caps(), seeded, runner)
    assert len(calls) == 1 and reply.kind == "unreceipted" and reply.unreceipted == why
    assert reply.cost_usd == pytest.approx(2.0) and reply.question is None


def test_a_confirmed_cap_hit_is_never_retried_and_names_its_cap_and_constraint(seeded: Root, task: Task):
    calls = []
    def runner(argv, stdin, cwd, timeout):
        calls.append(argv)
        return RunResult(1, '{"is_error": true, "subtype": "error_max_budget_usd", "total_cost_usd": 2.05}', "")
    reply = director.next_move(task, Caps(), seeded, runner)
    assert len(calls) == 1 and reply.kind == "ceiling" and reply.question is None and reply.cost_usd == 2.05
    assert reply.ceiling == {"seat": "director", "call": "c1", "cap_usd": 2.0, "bound_by": "per_call_limit",
                             "per_call_limit_usd": 2.0, "task_left_usd": 25.0, "charged_usd": 2.05,
                             "subtype": "error_max_budget_usd"}
    task.cost_usd = 24.25
    assert director.next_move(task, Caps(), seeded, runner).ceiling["bound_by"] == "task_budget" and len(calls) == 2
    assert calls[1][calls[1].index("--max-budget-usd") + 1] == "0.75"


def test_a_call_with_no_receipt_is_charged_its_clipped_cap_and_never_retried(seeded: Root, task: Task):
    task.cost_usd = 24.0
    calls: list[list[str]] = []
    def timeout(argv, *_):
        calls.append(argv)
        return RunResult(-9, "", "", True)
    reply = director.next_move(task, Caps(), seeded, timeout)
    assert (reply.cost_usd, reply.kind, len(calls)) == (1.0, "unreceipted", 1)
    assert reply.unreceipted == "the call timed out after 300s"
    (record,) = task.calls
    assert (record.outcome, record.charge_usd, record.actual_usd) == ("timed_out", 1.0, "UNKNOWN")


def test_the_message_carries_the_attempt_cap_the_prompt_names(task: Task):
    assert director.thinker_message(task, Caps(max_attempts_per_unit=3), [])["max_attempts_per_unit"] == 3
    prompt = PROMPT.read_text()
    assert "`max_attempts_per_unit`" in prompt and "fails twice" not in prompt


def _decode(obj: object, task: Task):
    return director.decode_move(obj if isinstance(obj, str) else json.dumps(obj), task)


@pytest.mark.parametrize("units, error", [
    ([UNIT, UNIT], "id is already used"),
    ([{**UNIT, "id": "u1"}], "id is already used"),
    ([{**UNIT, "id": "../escape"}], "id must match"),
    ([{**UNIT, "predicates": []}], "a unit needs at least one predicate"),
    ([{**UNIT, "predicates": [{"kind": "file_exists"}]}], "predicates must be a list"),
])
def test_decode_rejects_bad_units_in_a_task_move(task: Task, units: list, error: str):
    reply, errors = _decode(_task_move(*units), task)
    assert reply is None and any(error in e for e in errors), errors


# live-2 (2026-09-27): the director twice ended a valid plan one `}` short; the decoder then took the
# first nested object that parsed (a unit) and reported "carried none", so the retry was told the wrong error.
TRUNCATED = ('{"task": [{"id": "u-2", "intent": "write b.txt", "predicates": [{"kind": "file_exists", "args": '
             '{"path": "b.txt"}}]}], "state": {"standing": "b.txt is planned.", "open": []}')


def test_a_reply_cut_off_before_its_closing_brackets_is_completed(task: Task):
    reply, errors = _decode(TRUNCATED, task)
    assert errors == [] and [u.id for u in reply.units] == ["u-2"]
    reply, errors = _decode(TRUNCATED[: TRUNCATED.rindex("[]}")] + "[]", task)  # two closers short
    assert errors == [] and reply.units[0].predicates[0].args == {"path": "b.txt"}
    assert _decode('{"ask": "which?', task)[0] is None  # cut inside a string stays unreadable


def test_a_broken_outer_object_names_its_parse_error_instead_of_a_nested_object(task: Task):
    broken = '{"task": [{"id": "u-2", "intent": "write b.txt", "predicates": []}] oops}'
    reply, errors = _decode(broken, task)
    assert reply is None and "carried none" in errors[0] and "did not parse" in errors[0], errors


def test_pending_notes_cross_the_role_rename(root: Root):
    from protean.state import append_log
    for event in [{"to_manager": "stale"}, {"event": "manager"},
                  {"to_manager": "the operator answered"}, {"to_director": "retry"}]:
        append_log(root, "t", event)
    assert director.pending_notes(root, "t") == ["the operator answered", "retry"]
    append_log(root, "t", {"event": "director"})
    assert director.pending_notes(root, "t") == []


def _unit(uid: str, *needs: str) -> dict:
    return {**UNIT, "id": uid, "needs": list(needs)}


@pytest.mark.parametrize("units, error", [
    ([_unit("u-2", "u-9")], "unit 'u-2': needs 'u-9', which is no unit in the task or this plan"),
    ([_unit("u-2", "u-2")], "unit 'u-2': needs itself"),
    ([_unit("u-2", "u-3"), _unit("u-3", "u-4"), _unit("u-4", "u-2")], "unit 'u-2': needs form a cycle: u-2 -> u-3 -> u-4 -> u-2"),
    ([{**UNIT, "needs": "u1"}], "unit 'u-2': needs must be a list of unit ids"),
])
def test_decode_rejects_bad_needs(task: Task, units: list, error: str):
    reply, errors = _decode(_task_move(*units), task)
    assert reply is None and error in errors, errors


def test_decode_takes_needs_on_the_session_and_later_in_the_same_move(task: Task):
    reply, errors = _decode(_task_move(_unit("u-3", "u-2", "u1"), _unit("u-2")), task)
    assert errors == [] and [(u.id, u.needs) for u in reply.units] == [("u-3", ["u-2", "u1"]), ("u-2", [])]
    assert _decode(_task_move(UNIT), task)[0].units[0].needs == []


def test_the_message_shows_needs_and_the_prompt_documents_them(task: Task):
    task.units[0].needs = ["u0"]
    assert director.thinker_message(task, Caps(), [])["units"][0]["needs"] == ["u0"]
    prompt = PROMPT.read_text()
    assert '"needs"' in prompt and "`blocked`" in prompt


def test_decode_refuses_a_need_on_a_unit_that_passed_before_the_accepted_branch(task: Task):
    task.units[0].status = "passed"
    reply, errors = _decode(_task_move(_unit("u-2", "u1")), task)
    assert reply is None and errors == [
        "unit 'u-2': needs 'u1', which passed before the accepted branch existed and cannot be built on"], errors
    task.units[0].candidate, task.units[0].integrated = "abc", True
    assert _decode(_task_move(_unit("u-2", "u1")), task)[1] == []


# Retention: `retain` per predicate, default retained, a boolean or a decode error.

def test_decode_reads_retain_per_predicate_and_an_absent_flag_is_retained(task: Task):
    preds = [{"kind": "file_exists", "args": {"path": "b.txt"}},
             {"kind": "file_exists", "args": {"path": "stub.txt"}, "retain": False},
             {"kind": "command", "args": {"cmd": "true"}, "retain": True, "protected": ["tests"]}]
    reply, errors = _decode(_task_move({**UNIT, "predicates": preds}), task)
    assert errors == [] and [p.retain for p in reply.units[0].predicates] == [True, False, True]


@pytest.mark.parametrize("value", ["yes", 1, None, "false"])
def test_decode_rejects_a_retain_that_is_not_a_boolean_naming_unit_and_predicate(task: Task, value: object):
    preds = [{"kind": "file_exists", "args": {"path": "b.txt"}},
             {"kind": "file_exists", "args": {"path": "stub.txt"}, "retain": value}]
    reply, errors = _decode(_task_move({**UNIT, "predicates": preds}), task)
    assert reply is None and errors == [f"unit 'u-2': p2 file_exists: retain must be true or false, not {value!r}"]


def test_a_non_boolean_retain_goes_back_to_the_thinker_on_the_retry(seeded: Root, task: Task):
    bad = _task_move({**UNIT, "predicates": [{**UNIT["predicates"][0], "retain": "yes"}]})
    runner = ScriptedRunner([_reply(bad), _reply(_task_move(UNIT))], [])
    reply = director.next_move(task, Caps(), seeded, runner)
    assert [(u.id, [p.retain for p in u.predicates]) for u in reply.move.units] == [("u-2", [True])]
    assert "- unit 'u-2': p1 file_exists: retain must be true or false, not 'yes'" in runner.calls[1][1]


def test_the_message_shows_retain_only_on_a_delivery_only_predicate_and_the_prompt_documents_it(task: Task):
    task.units[0].predicates.append(Predicate("file_exists", {"path": "stub.txt"}, retain=False))
    assert director.thinker_message(task, Caps(), [])["units"][0]["predicates"] == [
        {"kind": "file_exists", "args": {"path": "a.txt"}},
        {"kind": "file_exists", "args": {"path": "stub.txt"}, "retain": False}]
    prompt = PROMPT.read_text()
    (paragraph,) = [line for line in prompt.splitlines() if line.startswith("Each predicate is a lasting promise")]
    assert prompt.count('"retain": false') == 1 and '"retain": false' in paragraph
    assert "re-checked" in paragraph and "cannot be dropped" in paragraph
    assert "accepted unit's retained predicates" in prompt
