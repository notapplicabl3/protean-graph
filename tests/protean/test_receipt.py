"""Receipt and recovery: a record per model call, written before launch; idempotent recovery of a call cut
off before its receipt; unknown spend parks paid work; successful re-checks recorded; old records load."""

from __future__ import annotations

import itertools
import json
import shutil
import signal
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

import pytest

from protean import loop
from protean.graph import console, director
from protean.graph.budget import Caps, tokens_of
from protean.dry import FIXTURES, REPO, ScriptedRunner, make_workspace, read_jsonl
from protean.runner import RunResult
from protean.state import (MOVE_KINDS, Call, Move, Predicate, Root, Task, Unit, current, from_json, load,
                           open_questions, save, set_current, to_json)

UNIT = {"id": "u-1", "intent": "write a.txt", "predicates": [{"kind": "file_exists", "args": {"path": "a.txt"}}]}
PASS = {"write": {"a.txt": "A\n"}, "commit": True, "exit_code": 0, "result": "wrote a.txt", "total_cost_usd": 0.5}
RECORD_FIELDS = {"id", "seat", "attempt", "cap_usd", "charge_usd", "receipt_cost_usd", "outcome", "tokens",
                 "message_bytes", "session_id"}
NO_TOKENS = {"input": None, "output": None, "cache_read": None, "cache_creation": None}
CAPTURES = REPO / "evidence" / "captures"


STANDINGS = itertools.count(1)


def _with_state(obj: object) -> object:
    """A thinker move without `state` gets one whose standing no earlier reply had."""
    if isinstance(obj, dict) and "state" not in obj and any(kind in obj for kind in MOVE_KINDS):
        return {**obj, "state": {"standing": f"Standing {next(STANDINGS)}.", "open": []}}
    return obj


def _reply(obj: dict, cost: float = 0.1) -> dict:
    obj = _with_state(obj)
    return {"type": "result", "is_error": False, "session_id": "s-1", "total_cost_usd": cost, "result": json.dumps(obj)}


@pytest.fixture
def env(tmp_path: Path, root: Root) -> tuple[Root, Path]:
    root.path.mkdir(parents=True)
    root.thinker_prompt.write_text("m\n")
    root.editor_prompt.write_text("e\n")
    return root, make_workspace(tmp_path / "ws")


def _task(root: Root, workspace: Path, **fields: object) -> Task:
    task = Task(id="task-20261010T120000", goal="g", workspace=str(workspace), **fields)
    save(root, task)
    set_current(root, task.id)
    return task


def _answer(root: Root, text: str) -> None:
    """`protean ctl answer` on the parked task's open question, no brain live."""
    (asked,) = open_questions(current(root))
    loop.offline_op(root, Caps(), "answer", {"id": asked.id, "text": text})


def _rested(task: Task) -> bool:
    return (task.terminal, task.awaiting_thought, task.moves[-1].kind) == ("interrupted", True, "rest")


def _calls(root: Root, task_id: str) -> list[dict]:
    return [e for e in director.read_log(root, task_id) if e["event"] == "call"]


# A separate process runs one tick and is SIGKILLed: either inside the model call, after serving the
# listed director replies, or right after the director seat returned, before the loop booked its cost.
KILL = """
import json, os, signal, sys
from pathlib import Path
from protean import loop, state
from protean.dry import ScriptedRunner
from protean.graph import director
from protean.graph.budget import Caps

root, when, replies = state.Root(Path(sys.argv[1])), sys.argv[2], json.loads(sys.argv[3])

def die(*_):
    os.kill(os.getpid(), signal.SIGKILL)

scripted = ScriptedRunner(replies, [])
if when == "after-plan":
    move = director.next_move
    director.next_move = lambda *a, **k: (move(*a, **k), die())[0]
    runner = scripted
else:
    def runner(argv, stdin, cwd, timeout):
        if not scripted.director or "Write" in argv:
            die()
        return scripted(argv, stdin, cwd, timeout)
loop.run_tick(state.current(root), Caps(), root, runner)
"""


def _hard_kill(root: Root, when: str = "call", replies: list[dict] | None = None) -> None:
    proc = subprocess.run([sys.executable, "-c", KILL, str(root.path), when, json.dumps(replies or [])],
                          capture_output=True, text=True)
    assert proc.returncode == -signal.SIGKILL, proc.stderr


@pytest.mark.parametrize("seat, cap", [("worker", 4.0), ("director", 2.0), ("explore", 4.0)])
def test_a_call_killed_before_its_receipt_is_booked_once_at_its_reserved_cap_across_resumes(
    env: tuple[Root, Path], seat: str, cap: float
):
    """The thinker's call (seat `director`), a task unit's worker and an explore's worker (seat `worker`, no unit):
    each record is in `task.json` before the call runs, so a SIGKILL inside it leaves it `pending`; the resume books
    it once at its reserved cap, its actual spend `UNKNOWN`, and paid work holds on `unknown_spend`."""
    root, workspace = env
    units = [Unit("u-1", "write a.txt", [Predicate("file_exists", {"path": "a.txt"})])] if seat == "worker" else []
    moves = [Move("m1", "explore", "try b.txt", 0.1, "pending", brief="Try b.txt.")] if seat == "explore" else []
    task = _task(root, workspace, cost_usd=1.0, units=units, moves=moves)
    _hard_kill(root)
    killed = load(root, task.id)
    (call,) = killed.calls
    assert (call.seat, call.outcome, call.cap_usd, call.booked, killed.cost_usd) == (
        "worker" if seat == "explore" else seat, "pending", cap, False, 1.0)
    assert call.unit == ("u-1" if seat == "worker" else None)
    runner = ScriptedRunner([], [])
    first = loop.resume(root, Caps(), runner)
    (lost,) = first.calls
    assert (lost.outcome, lost.actual_usd, [(q.reason, q.context) for q in open_questions(first)]) == (
        "lost", "UNKNOWN", [("unknown_spend", {"unknown_spend": [call.id]})])
    after_first = first.cost_usd
    after_second = loop.resume(root, Caps(), runner).cost_usd
    assert after_first == after_second == 1.0 + cap
    assert [c.charge_usd for c in load(root, task.id).calls] == [cap]
    assert [e["id"] for e in _calls(root, task.id)] == [call.id]
    assert runner.calls == []


def test_lost_spend_is_unknown_and_paid_work_waits_on_a_runtime_question_until_the_operator_says_yes(env: tuple[Root, Path]):
    root, workspace = env
    task = _task(root, workspace, cost_usd=1.0,
                 units=[Unit("u-1", "write a.txt", [Predicate("file_exists", {"path": "a.txt"})])])
    _hard_kill(root)
    runner = ScriptedRunner([_reply({"rest": True})], [PASS])
    task = loop.resume(root, Caps(), runner)
    (call,) = task.calls
    assert (call.outcome, call.actual_usd, call.receipt_cost_usd, call.charge_usd) == ("lost", "UNKNOWN", None, 4.0)
    (held,) = open_questions(task)
    assert (task.terminal, held.asker, held.reason, runner.calls) == ("interrupted", "runtime", "unknown_spend", [])
    assert held.context == {"unknown_spend": [call.id]} and load(root, task.id).questions == [held]
    with pytest.raises(console.Refused, match="only `yes`"):
        _answer(root, "not sure yet\n")
    assert loop.resume(root, Caps(), runner).terminal == "interrupted" and runner.calls == []
    _answer(root, "yes, continue")
    task = loop.resume(root, Caps(), runner)
    assert _rested(task) and len(runner.calls) == 2
    assert round(task.cost_usd, 2) == round(1.0 + 4.0 + 0.5 + 0.1, 2)
    assert [c.charge_usd for c in task.calls if c.id == call.id] == [4.0]
    message = runner.calls[1][1]
    assert json.loads(message)["notes"] == [] and "UNKNOWN" not in message and "yes, continue" not in message
    assert held.text not in message and [q.disposition for q in task.questions] == ["continued"]


def test_abandoning_a_killed_task_still_books_the_lost_call_and_asks_nothing(env: tuple[Root, Path]):
    root, workspace = env
    task = _task(root, workspace, cost_usd=1.0)
    _hard_kill(root)
    task = loop.resume(root, Caps(), ScriptedRunner([], []), abandon=True)
    assert (task.terminal, task.stop_reason, task.questions, task.cost_usd) == ("stopped", "abandoned", [], 3.0)
    assert [(c.outcome, c.actual_usd, c.booked) for c in task.calls] == [("lost", "UNKNOWN", True)]
    assert loop.resume(root, Caps(), ScriptedRunner([], [])).cost_usd == 3.0


def test_a_kill_during_the_retry_recovers_the_first_call_from_its_receipt_and_holds_on_the_second(
    env: tuple[Root, Path]
):
    root, workspace = env
    task = _task(root, workspace, cost_usd=1.0)
    _hard_kill(root, "call", [_reply({"nonsense": 1}, cost=0.3)])
    first, second = load(root, task.id).calls
    assert (first.outcome, first.charge_usd, first.booked, second.outcome) == ("ok", 0.3, False, "pending")
    task = loop.resume(root, Caps(), ScriptedRunner([], []))
    assert round(task.cost_usd, 2) == round(1.0 + 0.3 + 2.0, 2)
    assert [(c.outcome, c.actual_usd) for c in task.calls] == [("ok", 0.3), ("lost", "UNKNOWN")]
    assert task.terminal == "interrupted" and [q.reason for q in open_questions(task)] == ["unknown_spend"]


def test_a_call_killed_after_its_receipt_is_reconciled_from_it_without_holding_paid_work(env: tuple[Root, Path]):
    root, workspace = env
    task = _task(root, workspace, cost_usd=1.0)
    _hard_kill(root, "after-plan", [_reply({"task": [UNIT]}, cost=0.3)])
    (call,) = load(root, task.id).calls
    assert (call.outcome, call.receipt_cost_usd, call.booked, load(root, task.id).cost_usd) == ("ok", 0.3, False, 1.0)
    # the resumed warm session's receipt reports its whole spend so far, 0.3 + 0.1: the call is charged the 0.1 it added
    runner = ScriptedRunner([_reply({"ask": "which file?"}, cost=0.4)], [])
    task = loop.resume(root, Caps(), runner)
    assert [q.text for q in open_questions(task)] == ["which file?"] and len(runner.calls) == 1
    assert round(task.cost_usd, 2) == round(1.0 + 0.3 + 0.1, 2)
    assert [(c.charge_usd, c.booked) for c in task.calls] == [(0.3, True), (0.1, True)]


def test_a_ctrl_c_call_is_booked_once_and_holds_paid_work_until_yes(env: tuple[Root, Path]):
    root, workspace = env
    task = _task(root, workspace, units=[Unit("u-1", "write a.txt", [Predicate("file_exists", {"path": "a.txt"})])])
    def interrupt(argv, stdin, cwd, timeout):
        raise KeyboardInterrupt
    with pytest.raises(KeyboardInterrupt):
        loop.run_tick(task, Caps(), root, interrupt)
    saved = load(root, task.id)
    (call,) = saved.calls
    assert (call.outcome, call.charge_usd, call.booked, call.receipt_cost_usd) == ("interrupted", 4.0, True, None)
    assert [(q.reason, q.context) for q in open_questions(saved)] == [("unknown_spend", {"unknown_spend": [call.id]})]
    runner = ScriptedRunner([_reply({"rest": True})], [PASS])
    assert loop.resume(root, Caps(), runner).terminal == "interrupted" and runner.calls == []
    _answer(root, "yes")
    task = loop.resume(root, Caps(), runner)
    assert _rested(task) and round(task.cost_usd, 2) == 4.6
    assert [c.charge_usd for c in task.calls if c.id == call.id] == [4.0]
    assert [q.reason for q in task.questions] == ["unknown_spend"]  # booked once: no second question on resume


def test_every_call_record_carries_the_receipt_fields_and_missing_tokens_are_null(env: tuple[Root, Path]):
    root, workspace = env
    runner = ScriptedRunner(read_jsonl(FIXTURES / "dry/director.jsonl"), read_jsonl(FIXTURES / "dry/worker.jsonl"))
    task = loop.run("dry", workspace, root, Caps(), runner)
    assert _rested(task) and len(task.calls) == len(runner.calls) == 4
    for call, (argv, stdin, _, _) in zip(task.calls, runner.calls):
        assert RECORD_FIELDS <= asdict(call).keys()
        assert call.cap_usd == float(argv[argv.index("--max-budget-usd") + 1])
        assert call.message_bytes == len(stdin.encode("utf-8"))
        assert call.tokens == NO_TOKENS
        assert (call.outcome, call.booked, call.charge_usd) == ("ok", True, call.actual_usd)
        assert '"cap_usd"' not in stdin and "receipt_cost_usd" not in stdin
    assert [(c.seat, c.unit, c.attempt) for c in task.calls] == [
        ("director", None, 1), ("worker", "u-1", 1), ("worker", "u-2", 1), ("director", None, 1)]
    # the last call resumed the thinker's warm session: its receipt reports the session's 0.2 + 0.1, its charge the 0.1
    assert [c.receipt_cost_usd for c in task.calls] == [0.2, 0.5, 0.5, 0.3]
    assert [c.charge_usd for c in task.calls] == [0.2, 0.5, 0.5, 0.1]
    assert [c.session_id for c in task.calls] == ["dry-director", "dry-worker", "dry-worker", "dry-director"]
    assert sum(c.charge_usd for c in task.calls) == pytest.approx(task.cost_usd)
    logged = _calls(root, task.id)
    assert [e["id"] for e in logged] == [c.id for c in task.calls] and logged[0]["tokens"] == NO_TOKENS


def test_a_call_with_no_receipt_records_its_cap_charge_and_nothing_it_could_not_measure(env: tuple[Root, Path]):
    root, workspace = env
    task = _task(root, workspace)
    loop.run_tick(task, Caps(), root, lambda *_: RunResult(-9, "", "", True))
    records = [(c.attempt, c.outcome, c.charge_usd, c.receipt_cost_usd, c.actual_usd, c.session_id) for c in task.calls]
    assert records == [(1, "timed_out", 2.0, None, "UNKNOWN", None)]  # never retried: paid work holds instead
    assert all(c.tokens == NO_TOKENS for c in task.calls) and task.cost_usd == 2.0


def test_a_capped_call_records_its_receipt_cost_beside_its_cap(env: tuple[Root, Path]):
    root, workspace = env
    task = _task(root, workspace)
    capped = json.loads((CAPTURES / "p4-at-cap.stdout.json").read_text())
    replies = [RunResult(1, json.dumps(capped), ""), RunResult(0, json.dumps(_reply({"ask": "q?"})), "")]
    loop.run_tick(task, Caps(), root, lambda *_: replies.pop(0))
    first = task.calls[0]
    assert (first.outcome, first.subtype, first.cap_usd) == ("error", "error_max_budget_usd", 2.0)
    assert first.receipt_cost_usd == first.charge_usd == capped["total_cost_usd"]
    assert first.tokens == {"input": 914, "output": 62, "cache_read": 17693, "cache_creation": 4481}
    assert first.session_id == capped["session_id"]


def test_tokens_are_read_from_the_receipt_and_an_absent_count_is_null_never_zero():
    receipt = json.loads((CAPTURES / "p1-call1.stdout.json").read_text())
    assert tokens_of(receipt) == {"input": 914, "output": 63, "cache_read": 0, "cache_creation": 22170}
    assert tokens_of({"usage": {"input_tokens": 7, "output_tokens": 3}}) == {
        "input": 7, "output": 3, "cache_read": None, "cache_creation": None}
    assert tokens_of({}) == NO_TOKENS
    assert tokens_of({"modelUsage": "garbled", "usage": None}) == NO_TOKENS
    garbled = {"modelUsage": {"m": {"inputTokens": True, "outputTokens": -1, "cacheReadInputTokens": 2.5}}}
    assert tokens_of(garbled) == NO_TOKENS


def _verdicts(root: Root, task_id: str) -> list[dict]:
    return [e for e in director.read_log(root, task_id) if e["event"] == "verdict"]


def test_a_graded_candidate_records_its_rechecks_and_an_empty_set_when_nothing_is_accepted(env: tuple[Root, Path]):
    root, workspace = env
    runner = ScriptedRunner(read_jsonl(FIXTURES / "deps/director.jsonl"), read_jsonl(FIXTURES / "deps/worker.jsonl"))
    task = loop.run("deps", workspace, root, Caps(), runner)
    by_id = {u.id: u for u in task.units}
    assert [(v["unit"], v["recheck"]) for v in _verdicts(root, task.id)] == [
        ("u-1", {"candidate": by_id["u-1"].candidate, "predicates": []}),
        ("u-2", {"candidate": by_id["u-2"].candidate, "predicates": [{"unit": "u-1", "id": "p1", "result": "passed"}]}),
        ("u-3", {"candidate": by_id["u-3"].candidate, "predicates": [
            {"unit": "u-1", "id": "p1", "result": "passed"}, {"unit": "u-2", "id": "p1", "result": "passed"}]}),
    ]


def test_a_broken_recheck_is_recorded_with_its_result(env: tuple[Root, Path]):
    root, workspace = env
    runner = ScriptedRunner(read_jsonl(FIXTURES / "deps-broken/director.jsonl"),
                            read_jsonl(FIXTURES / "deps-broken/worker.jsonl"))
    task = loop.run("deps-broken", workspace, root, Caps(), runner)
    first, *broken = _verdicts(root, task.id)
    assert first["recheck"]["predicates"] == []
    assert [v["recheck"]["predicates"] for v in broken] == [[{"unit": "u-1", "id": "p1", "result": "failed"}]] * 2
    assert all(v["recheck"]["candidate"] and v["recheck"]["candidate"] != task.accepted for v in broken)


def test_an_attempt_whose_own_grade_never_ran_records_no_recheck(env: tuple[Root, Path]):
    root, workspace = env
    runner = ScriptedRunner(read_jsonl(FIXTURES / "stuck/director.jsonl"), read_jsonl(FIXTURES / "stuck/worker.jsonl"))
    task = loop.run("stuck", workspace, root, Caps(), runner)
    verdicts = _verdicts(root, task.id)
    assert len(verdicts) == 2 and all("recheck" not in v for v in verdicts)


@pytest.mark.parametrize("run", ["live-5-2026-10-03", "live-4-2026-10-02"])
def test_an_old_task_json_loads_with_the_new_fields_at_their_defaults(root: Root, run: str, live_receipt):
    source = live_receipt(run)
    task_id = json.loads(source.read_text())["id"]
    target = root.task_file(task_id)
    target.parent.mkdir(parents=True)
    shutil.copy2(source, target)
    assert "calls" not in json.loads(target.read_text())
    task = load(root, task_id)
    assert task.calls == [] and task.units and from_json(to_json(task)) == task
    older = Call(**{"id": "c1", "seat": "director", "attempt": 1, "cap_usd": 2.0})
    assert (older.outcome, older.charge_usd, older.receipt_cost_usd, older.tokens, older.booked) == (
        "pending", None, None, NO_TOKENS, False)


# --- every receipt-less call holds paid work (ruled 2026-10-11): timeout, no receipt and Ctrl-C, not only a kill ---


def test_a_timed_out_director_call_holds_paid_work_and_is_not_retried(env: tuple[Root, Path]):
    root, workspace = env
    task = _task(root, workspace, cost_usd=1.0)
    runner = ScriptedRunner([{"timed_out": True}, _reply({"task": [UNIT]}), _reply({"rest": True})], [PASS])
    loop.run_tick(task, Caps(), root, runner)
    (call,) = task.calls
    assert (call.outcome, call.charge_usd, call.actual_usd, call.booked) == ("timed_out", 2.0, "UNKNOWN", True)
    assert len(runner.calls) == 1 and round(task.cost_usd, 2) == 3.0 and task.units == []
    (held,) = open_questions(task)
    assert (held.asker, held.reason, held.context) == ("runtime", "unknown_spend", {"unknown_spend": [call.id]})
    assert "timed_out" in held.text and load(root, task.id).questions == [held]
    assert loop.resume(root, Caps(), runner).terminal == "interrupted" and len(runner.calls) == 1
    _answer(root, "yes")
    task = loop.resume(root, Caps(), runner)
    assert _rested(task) and len(runner.calls) == 4 and round(task.cost_usd, 2) == 3.7
    assert [c.outcome for c in task.calls] == ["timed_out", "ok", "ok", "ok"]
    message = runner.calls[1][1]
    assert "UNKNOWN" not in message and held.text not in message  # the hold is the runtime's, the thinker never sees it


def test_a_timed_out_worker_call_holds_paid_work_before_the_retry(env: tuple[Root, Path]):
    root, workspace = env
    task = _task(root, workspace, units=[Unit("u-1", "write a.txt", [Predicate("file_exists", {"path": "a.txt"})])])
    runner = ScriptedRunner([_reply({"rest": True})], [{"timed_out": True}, PASS])
    loop.run_tick(task, Caps(), root, runner)
    (call,) = task.calls
    assert (call.seat, call.outcome, call.charge_usd, call.actual_usd) == ("worker", "timed_out", 4.0, "UNKNOWN")
    assert (task.units[0].status, task.units[0].attempts) == ("pending", 1)
    (held,) = open_questions(task)
    assert (held.reason, held.context) == ("unknown_spend", {"unknown_spend": [call.id]})
    assert loop.resume(root, Caps(), runner).terminal == "interrupted" and len(runner.calls) == 1
    _answer(root, "yes")
    task = loop.resume(root, Caps(), runner)
    assert _rested(task) and [c.outcome for c in task.calls] == ["timed_out", "ok", "ok"]
    assert round(task.cost_usd, 2) == round(4.0 + 0.5 + 0.1, 2)


def test_a_worker_call_without_a_receipt_holds_paid_work(env: tuple[Root, Path]):
    root, workspace = env
    task = _task(root, workspace, units=[Unit("u-1", "write a.txt", [Predicate("file_exists", {"path": "a.txt"})])])
    def no_receipt(argv, stdin, cwd, timeout):
        return RunResult(0, "", "", False)
    loop.run_tick(task, Caps(), root, no_receipt)
    (call,) = task.calls
    assert (call.outcome, call.actual_usd) == ("no_receipt", "UNKNOWN")
    assert [q.reason for q in open_questions(task)] == ["unknown_spend"]
