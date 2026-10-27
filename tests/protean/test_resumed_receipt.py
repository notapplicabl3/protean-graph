"""A resumed seat's receipt reports the whole session: the call is charged the difference."""

from __future__ import annotations

import json
from pathlib import Path

from protean import loop
from protean.graph.budget import Caps
from protean.dry import make_workspace
from protean.runner import RunResult
from protean.state import Root, Task, save, set_current


def _receipt(result: str, cost: float, tokens: tuple[int, int, int, int], session: str = "s-1") -> RunResult:
    cache_read, cache_creation, inp, out = tokens
    receipt = {"type": "result", "is_error": False, "session_id": session, "total_cost_usd": cost, "result": result,
               "usage": {"input_tokens": inp, "output_tokens": out, "cache_read_input_tokens": cache_read,
                         "cache_creation_input_tokens": cache_creation}}
    return RunResult(0, json.dumps(receipt), "")


def test_a_resumed_session_receipt_is_cumulative_and_the_call_is_charged_its_difference(tmp_path: Path, root: Root):
    """Session 7, tick 12: two thinker calls on one resumed session reported $8.64 then $8.88; the brain booked $17.52
    for a second call that cost $0.24. The receipt's figure is the session's total so far; the call's charge is what it
    added, and so are its tokens."""
    root.path.mkdir(parents=True)
    root.thinker_prompt.write_text("m\n")
    root.editor_prompt.write_text("e\n")
    task = Task(id="task-20261020T120000", goal="g", workspace=str(make_workspace(tmp_path / "ws")))
    save(root, task)
    set_current(root, task.id)
    rest = json.dumps({"rest": True, "state": {"standing": "Resting.", "open": []}})
    replies = [_receipt("not a move", 1.0, (100, 400, 10, 50)),  # rejected, retried on the same session
               _receipt(rest, 1.5, (300, 450, 14, 80))]  # the session's total: this call added $0.50
    loop.run_tick(task, Caps(), root, lambda *_: replies.pop(0))
    first, second = task.calls
    assert (first.charge_usd, first.actual_usd, first.receipt_cost_usd) == (1.0, 1.0, 1.0)
    assert (second.charge_usd, second.actual_usd, second.receipt_cost_usd) == (0.5, 0.5, 1.5)
    assert second.tokens == {"input": 4, "output": 30, "cache_read": 200, "cache_creation": 50}
    assert task.cost_usd == 1.5
    assert task.moves[-1].cost_usd == 1.5
