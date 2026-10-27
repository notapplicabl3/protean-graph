"""The reader's own budget (the operator, 2026-10-20: "make reader calls not count towards the spend cap, but be limited to $7
itself"; and every cap doubled the same day). Reader spend is booked on the session like every call's, but it never
counts toward the allowance: the allowance stops the session on `counted_usd` (the booked spend less the reader's), and
reader calls are clipped by `reader_max_usd` (7.0) less what the reader has spent, down to the call floor, under which
no reader call starts — the attempt or plan then holds as a floor-stopped check does today."""

from __future__ import annotations

import pytest

from protean.cli import DEFAULT_BRAIN
from protean.graph import budget, console, reader
from protean.graph.budget import Caps, load_caps
from protean.runner import floor_reason, usd_cap, usd_left
from protean.state import Predicate, Root, Task, Unit


def _booked(task: Task, seat: str, usd: float, cap: float = 10.0) -> None:
    """One resolved, booked call of `seat` that spent `usd`, through the same functions the loop uses."""
    call = budget.open_call(task, seat, 1, cap, None)
    budget.resolve(call, usd, {"type": "result", "is_error": False, "total_cost_usd": usd})
    budget.book(task, call)


def test_the_shipped_config_doubles_every_dollar_cap_and_the_reader_has_its_own():
    """`config.json` is the operator's numbers: every dollar cap twice the battery's fixture defaults (`Caps()`), the
    tick ceiling the same, and the reader's own budget beside them."""
    caps, shipped = Caps(), load_caps(Root(DEFAULT_BRAIN))
    for name in ("max_usd", "accrual_usd", "director_call_usd", "worker_call_usd", "reader_call_usd"):
        assert getattr(shipped, name) == 2 * getattr(caps, name), name
    assert (shipped.max_usd, shipped.accrual_usd, shipped.director_call_usd, shipped.worker_call_usd,
            shipped.reader_call_usd) == (50.0, 20.0, 4.0, 8.0, 3.0)
    assert (shipped.reader_max_usd, caps.reader_max_usd) == (7.0, 7.0)
    assert (shipped.max_ticks, shipped.accrual_seconds) == (caps.max_ticks, caps.accrual_seconds) == (100, 1800)


def test_reader_spend_is_booked_but_never_counted_toward_the_allowance():
    task = Task(id="task-20261020T000000", goal="g", workspace="w")
    _booked(task, "director", 2.0)
    _booked(task, "reader", 1.5)
    _booked(task, "reader", 3.0)
    assert task.cost_usd == pytest.approx(6.5)  # the ledger is whole
    assert budget.reader_usd(task) == pytest.approx(4.5) and budget.counted_usd(task) == pytest.approx(2.0)
    caps = Caps(max_usd=5.0)
    assert budget.allowance_reason(task, caps) is None  # 2.00 counted under 5.00, though 6.50 is booked
    assert usd_left(task, caps) == pytest.approx(3.0)
    _booked(task, "worker", 3.0)
    assert budget.allowance_reason(task, caps) == "max_usd: 5.00 >= 5.0"


def test_a_reader_call_is_clipped_by_the_reader_budget_not_the_allowance():
    task = Task(id="task-20261020T000000", goal="g", workspace="w")
    caps = Caps(max_usd=100.0, reader_call_usd=3.0, reader_max_usd=7.0)
    _booked(task, "reader", 5.0)
    assert usd_left(task, caps, reader=True) == pytest.approx(2.0)
    assert usd_cap(caps.reader_call_usd, task, caps, reader=True) == "2.00"
    assert usd_cap(caps.reader_call_usd, task, caps) == "3.00"  # the allowance alone would not clip it
    assert floor_reason(task, caps, reader=True) is None
    _booked(task, "reader", 1.6)
    assert floor_reason(task, caps, reader=True) == "reader_max_usd: remaining $0.40 below the $0.50 call floor"
    assert floor_reason(task, caps) is None  # the session itself may still call


def test_a_reader_check_at_the_reader_budget_makes_no_call(tmp_path, root: Root):
    task = Task(id="task-20261020T000000", goal="g", workspace=str(tmp_path))
    caps = Caps(reader_max_usd=1.0)
    _booked(task, "reader", 0.8)
    launched: list = []
    unit = Unit("u-1", "write a.txt", [Predicate("file_exists", {"path": "a.txt"})])
    check = reader.check(task, unit, {"question": "fits?"}, caps, root, lambda *args: launched.append(args))
    assert check.reason == "floor" and "reader_max_usd" in check.note and launched == []


def test_the_snapshot_ceilings_carry_the_reader_budget():
    task = Task(id="task-20261020T000000", goal="g", workspace="w")
    _booked(task, "director", 1.0)
    _booked(task, "reader", 0.5)
    ceilings = console.ceilings(task, Caps())
    assert (ceilings["reader_max_usd"], ceilings["reader_usd"], ceilings["counted_usd"]) == (7.0, 0.5, 1.0)
    assert (ceilings["max_usd"], ceilings["allowance_usd"], ceilings["reader_call_usd"]) == (25.0, 25.0, 1.5)
