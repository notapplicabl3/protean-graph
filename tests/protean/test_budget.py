"""Budget: caps from config, the stop test at >=, and dollars from receipts only."""

from __future__ import annotations

import json
import re

import pytest

from protean.cli import DEFAULT_BRAIN
from protean.dry import REPO
from protean.graph.budget import Caps, cost_of, load_caps, stop_reason
from protean.state import Root, Task, from_json, to_json

CONFIG = {
    "max_usd": 25.0, "max_ticks": 100, "director_call_usd": 2.0, "worker_call_usd": 4.0,
    "director_wall_seconds": 300, "worker_wall_seconds": 900, "max_attempts_per_unit": 2, "accrual_usd": 10.0,
    "accrual_seconds": 1800,
}


def _load(root: Root, data: object) -> Caps:
    root.path.mkdir(parents=True, exist_ok=True)
    root.config_file.write_text(json.dumps(data))
    return load_caps(root)


def test_config_loading_defaults(root: Root):
    assert load_caps(root) == Caps()
    assert _load(root, CONFIG) == Caps()
    caps = _load(root, {"max_usd": 5, "max_ticks": 12})
    assert (caps.max_usd, caps.max_ticks, caps.worker_call_usd) == (5, 12, 4.0)


@pytest.mark.parametrize(
    "data", [{"max_usd": 10.0, "max_dollars": 5}, {"max_ticks": 3.5}, {"max_usd": "10"}, {"max_usd": True}, [1]]
)
def test_a_bad_config_raises(root: Root, data: object):
    with pytest.raises(ValueError):
        _load(root, data)


OPUS = {"input": 5.0, "output": 25.0, "cache_read": 0.5, "cache_creation": 10.0}


def test_prices_load_per_model_and_default_to_none(root: Root):
    assert Caps().prices == {} and _load(root, {"max_usd": 5}).prices == {}
    cheap = {"input": 1, "output": 2, "cache_read": 0, "cache_creation": 3}
    assert _load(root, {"prices": {"claude-opus-5": OPUS, "made-up-2": cheap}}).prices == {
        "claude-opus-5": OPUS, "made-up-2": {"input": 1.0, "output": 2.0, "cache_read": 0.0, "cache_creation": 3.0}}


@pytest.mark.parametrize("prices", [
    [OPUS], {"claude-opus-5": 5.0}, {"": OPUS}, {"claude-opus-5": {k: v for k, v in OPUS.items() if k != "cache_read"}},
    {"claude-opus-5": {**OPUS, "cache_write": 10.0}}, {"claude-opus-5": {**OPUS, "input": -1}},
    {"claude-opus-5": {**OPUS, "output": "25"}}, {"claude-opus-5": {**OPUS, "output": True}},
    {"claude-opus-5": {**OPUS, "output": None}},
])
def test_a_malformed_price_entry_raises_like_an_unknown_key(root: Root, prices: object):
    with pytest.raises(ValueError, match=rf"^{re.escape(str(root.path / 'config.json'))}: prices"):
        _load(root, {"prices": prices})


def test_the_tracked_prices_reproduce_run_6_s_receipts_to_the_cent(live_receipt):
    """input·5 + output·25 + cache_read·0.5 + cache_creation·10 per million tokens (the 1-hour cache write at $10)
    gives back calls c2 and c3 of run 6 (2026-10-16)."""
    prices = load_caps(Root(DEFAULT_BRAIN)).prices["claude-opus-5"]
    record = from_json(live_receipt("live-6-2026-10-13").read_text())
    calls = [c for c in record.calls if c.id in ("c2", "c3")]
    estimates = [sum(c.tokens[k] * prices[k] for k in OPUS) / 1_000_000 for c in calls]
    assert [c.id for c in calls] == ["c2", "c3"]
    assert estimates == pytest.approx([c.receipt_cost_usd for c in calls], abs=0.005)


def test_the_reader_cap_defaults_to_1_50():
    """Run 6: the weakening reader on a unit that edits one protected test file cost $0.65 twice
    against 0.50 and burned both attempts. The shipped config doubled it to 3.00 with every other dollar cap on
    2026-10-20 (`test_reader_budget`); the default is the battery's fixture number."""
    assert Caps().reader_call_usd == 1.5 and load_caps(Root(DEFAULT_BRAIN)).reader_call_usd == 3.0


def test_legacy_director_limits_are_preserved(root: Root):
    caps = _load(root, {"manager_call_usd": 1.25, "manager_wall_seconds": 42})
    assert (caps.director_call_usd, caps.director_wall_seconds) == (1.25, 42)
    with pytest.raises(ValueError, match="conflicting keys"):
        _load(root, {"manager_call_usd": 1.25, "director_call_usd": 2.0})


def test_stop_reason_crosses_at_greater_or_equal():
    assert stop_reason(Task("t", "g", "/w", tick=99, cost_usd=24.99), Caps()) is None
    assert stop_reason(Task("t", "g", "/w", cost_usd=25.0), Caps()) == "max_usd: 25.00 >= 25.0"
    assert stop_reason(Task("t", "g", "/w", cost_usd=25.36), Caps()) == "max_usd: 25.36 >= 25.0"
    assert stop_reason(Task("t", "g", "/w", tick=100), Caps()) == "max_ticks: 100 >= 100"
    assert stop_reason(Task("t", "g", "/w", tick=3), Caps(max_ticks=3)) == "max_ticks: 3 >= 3"


def test_task_extras_default_to_zero_and_round_trip():
    task = Task("t", "g", "/w")
    assert (task.extra_usd, task.extra_ticks) == (0.0, 0)
    task.extra_usd, task.extra_ticks = 2.5, 4
    assert from_json(to_json(task)) == task
    older = json.loads(to_json(Task("t", "g", "/w")))
    del older["extra_usd"], older["extra_ticks"]
    assert from_json(json.dumps(older)) == Task("t", "g", "/w")


def test_stop_reason_widens_by_the_task_extras():
    assert stop_reason(Task("t", "g", "/w", cost_usd=25.36, extra_usd=5.0), Caps()) is None
    assert stop_reason(Task("t", "g", "/w", cost_usd=30.0, extra_usd=5.0), Caps()) == "max_usd: 30.00 >= 30.0"
    assert stop_reason(Task("t", "g", "/w", tick=100, extra_ticks=5), Caps()) is None
    assert stop_reason(Task("t", "g", "/w", tick=105, extra_ticks=5), Caps()) == "max_ticks: 105 >= 105"
    assert stop_reason(Task("t", "g", "/w", cost_usd=29.99, active_seconds=900.0), Caps()) is None  # $5.00 accrued
    assert stop_reason(Task("t", "g", "/w", cost_usd=30.0, active_seconds=900.0), Caps()) == "max_usd: 30.00 >= 30.0"


def test_cost_of_a_capped_call_is_its_receipt_not_its_tokens():
    capped = {
        "is_error": True,
        "total_cost_usd": 2.51,
        "usage": {"input_tokens": 0, "output_tokens": 0},
        "modelUsage": {"claude-opus-5": {"inputTokens": 0, "outputTokens": 0, "costUSD": 2.51}},
    }
    assert cost_of(capped) == 2.51


def test_cost_of_falls_back_to_the_model_usage_sum():
    receipt = {"modelUsage": {"a": {"costUSD": 1.25}, "b": {"costUSD": 0.5}, "c": {"inputTokens": 900}}}
    assert cost_of(receipt) == 1.75
    assert cost_of({"total_cost_usd": None, **receipt}) == 1.75


@pytest.mark.parametrize("total", [float("nan"), float("inf"), -1.0, True, "2.51"])
def test_cost_of_ignores_a_total_that_is_not_a_real_amount(total: object):
    assert cost_of({"total_cost_usd": total, "modelUsage": {"m": {"costUSD": 0.25}}}) == 0.25


def test_cost_of_with_no_dollar_figure_is_zero_never_estimated():
    assert cost_of({}) == 0.0
    assert cost_of({"usage": {"input_tokens": 50_000, "output_tokens": 8_000}}) == 0.0
    assert cost_of({"modelUsage": "garbled"}) == 0.0
