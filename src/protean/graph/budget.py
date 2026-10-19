"""Ceilings and spend: the caps, the stop test, the dollar cost of one call's receipt, and the per-call record.

Every model call's record is saved before its process starts and booked once: in the step that made it,
in the interruption that cut it, or by the resume after a kill. A call that left no receipt is charged its
reserved cap and its actual spend stays UNKNOWN.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field, fields, replace

from protean.state import Call, Root, Task

UNKNOWN = "UNKNOWN"
CAP_HIT = "error_max_budget_usd"  # the receipt subtype of a call the CLI ended at its --max-budget-usd
# (record key, `modelUsage` per-model key, `usage` key) for each token count a receipt can carry.
TOKEN_FIELDS = (
    ("input", "inputTokens", "input_tokens"),
    ("output", "outputTokens", "output_tokens"),
    ("cache_read", "cacheReadInputTokens", "cache_read_input_tokens"),
    ("cache_creation", "cacheCreationInputTokens", "cache_creation_input_tokens"),
)
PRICE_KEYS = tuple(name for name, _, _ in TOKEN_FIELDS)  # a model's prices, USD per million tokens of each count


@dataclass
class Caps:
    """Per-task ceilings, per-call limits and the model pinned to each seat. `max_usd` is where the session's dollar
    allowance starts; `accrual_usd` is what every `accrual_seconds` of active work adds to it, pro rata (`allowance`).

    `prices` maps a model to its USD per million tokens for each of `PRICE_KEYS`, for the console to hand the app; the
    brain prices nothing with them. claude-opus-5's are pinned from Anthropic's price table (input, output) and proven
    on run 6's receipts: with the 1-hour cache write at $10, input·5 + output·25 + cache_read·0.5 + cache_creation·10
    gives calls c2 ($2.795631) and c3 ($1.82465) to the cent."""

    max_usd: float = 25.0
    max_ticks: int = 100
    accrual_usd: float = 10.0
    accrual_seconds: int = 1800
    director_call_usd: float = 2.0
    worker_call_usd: float = 4.0
    director_wall_seconds: int = 300
    worker_wall_seconds: int = 900
    max_attempts_per_unit: int = 2
    unattended_seconds: int | None = None  # console: no client attached this long after a hold opens -> notify once and park; None -> console.UNATTENDED_SECONDS (1800, the session's default)
    # The reader seat's per-call cap, charged to the task; 0.5 until run 6, where the weakening reader
    # on a unit that edits one protected test file cost $0.65 twice against 0.50 and burned both attempts.
    reader_call_usd: float = 1.5
    reader_wall_seconds: int = 300
    director_model: str = "claude-opus-5"  # each seat's `--model`
    worker_model: str = "claude-opus-5"
    reader_model: str = "claude-opus-5"
    prices: dict[str, dict[str, float]] = field(default_factory=dict)


def _is_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _prices(config: object, value: object) -> dict[str, dict[str, float]]:
    """`prices` as `Caps` holds it: each model (a non-empty string) mapped to exactly `PRICE_KEYS`, each a non-negative
    number; anything else raises as an unknown key does."""
    if not isinstance(value, dict):
        raise ValueError(f"{config}: prices must map each model to its prices, got {value!r}")
    for model, entry in value.items():
        if (not model.strip() or not isinstance(entry, dict) or set(entry) != set(PRICE_KEYS)
                or any(_dollars(entry[key]) is None for key in PRICE_KEYS)):
            raise ValueError(f"{config}: prices for {model!r} must map exactly {', '.join(PRICE_KEYS)} to "
                             f"non-negative numbers (USD per million tokens), got {entry!r}")
    return {model: {key: float(entry[key]) for key in PRICE_KEYS} for model, entry in value.items()}


def load_caps(root: Root) -> Caps:
    """Caps from `config.json`; a missing file or key keeps the default, an unknown key raises. A model is a non-empty
    string, `prices` as `_prices` reads it, every other value a number (`accrual_seconds` one above zero)."""
    try:
        data = json.loads(root.config_file.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return Caps()
    if not isinstance(data, dict):
        raise ValueError(f"{root.config_file}: expected a JSON object")
    for suffix in ("call_usd", "wall_seconds"):
        old, new = f"manager_{suffix}", f"director_{suffix}"
        if old in data:  # Accept existing custom configs without changing their limits.
            if new in data and data[new] != data[old]:
                raise ValueError(f"{root.config_file}: conflicting keys {old} and {new}")
            data[new] = data.pop(old)
    known = {f.name: f for f in fields(Caps)}
    unknown = sorted(data.keys() - known.keys())
    if unknown:
        raise ValueError(f"{root.config_file}: unknown key(s) {', '.join(unknown)}")
    if "prices" in data:
        data["prices"] = _prices(root.config_file, data["prices"])
    for name, value in data.items():
        if name == "prices":
            continue
        if isinstance(known[name].default, str):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{root.config_file}: {name} must be a non-empty string, got {value!r}")
            continue
        wants_int = isinstance(known[name].default, int)
        if not _is_number(value) or (wants_int and not isinstance(value, int)):
            kind = "an integer" if wants_int else "a number"
            raise ValueError(f"{root.config_file}: {name} must be {kind}, got {value!r}")
    if data.get("accrual_seconds", 1) <= 0:
        raise ValueError(f"{root.config_file}: accrual_seconds must be above zero, got {data['accrual_seconds']!r}")
    return Caps(**data)


def accrued_usd(task: Task, caps: Caps) -> float:
    """What active work has added to the allowance: `accrual_usd` for every `accrual_seconds` of `active_seconds`, pro
    rata (continuous, never in steps)."""
    return caps.accrual_usd * task.active_seconds / caps.accrual_seconds


def allowance(task: Task, caps: Caps) -> float:
    """What the session may spend in all: `max_usd`, plus every `extend`, plus what active
    work has accrued (`accrued_usd`)."""
    return caps.max_usd + task.extra_usd + accrued_usd(task, caps)


def call_caps(task: Task, caps: Caps) -> Caps:
    """`caps` as a call is clipped by: `max_usd` raised by what has accrued, so what the runner counts as left
    (`runner.usd_left`: `max_usd` plus the task's extensions, less its cost and any `spent`) is what the allowance has
    left. Every per-call cap, wall, model and price stays, so `usd_cap` gives the least of the per-call cap and the
    allowance's left, and the call floor applies to that. An `extend` landing mid-step still counts (the runner reads
    the task's extensions live); active time moves only at a tick's save. Only a call's argv and its floor read it."""
    return replace(caps, max_usd=caps.max_usd + accrued_usd(task, caps))


def allowance_reason(task: Task, caps: Caps) -> str | None:
    """The allowance the session's spend has reached, named with its numbers, or None while it is under it (a sum of
    receipts may land a hair under the allowance it reached)."""
    allowed = allowance(task, caps)
    if task.cost_usd >= allowed - 1e-9:
        return f"max_usd: {task.cost_usd:.2f} >= {round(allowed, 2)}"
    return None


def stop_reason(task: Task, caps: Caps) -> str | None:
    """The ceiling the task has reached, named with its numbers, or None while under every one: the allowance
    (`allowance_reason`), or `max_ticks` with every extension."""
    max_ticks = caps.max_ticks + task.extra_ticks
    reached = allowance_reason(task, caps)
    if reached:
        return reached
    if task.tick >= max_ticks:
        return f"max_ticks: {task.tick} >= {max_ticks}"
    return None


def _dollars(value: object) -> float | None:
    """A usable dollar amount: a finite, non-negative number that is not a bool."""
    if _is_number(value) and math.isfinite(value) and value >= 0:
        return float(value)
    return None


def reported_cost(receipt: dict) -> float | None:
    """Dollars the receipt reports: `total_cost_usd`, else the sum of `modelUsage[*].costUSD`; None when it
    carries no dollar figure at all."""
    total = _dollars(receipt.get("total_cost_usd"))
    if total is not None:
        return total
    usage = receipt.get("modelUsage")
    if not isinstance(usage, dict):
        return None
    costs = [c for c in (_dollars(m.get("costUSD")) for m in usage.values() if isinstance(m, dict)) if c is not None]
    return float(sum(costs)) if costs else None


def cost_of(receipt: dict) -> float:
    """Dollars one call spent: `total_cost_usd`, else the sum of `modelUsage[*].costUSD`, else 0.

    Capped calls report their dollars but zero tokens, so cost is never estimated from tokens.
    """
    reported = reported_cost(receipt)
    return 0.0 if reported is None else reported


def cap_hit(receipt: dict) -> bool:
    """The receipt confirms the call ended because it reached its dollar cap: its `subtype` is `CAP_HIT`
    (`evidence/captures/p4-at-cap.stdout.json`). An incomplete reply alone is never a cap hit."""
    return receipt.get("subtype") == CAP_HIT


def _count(value: object) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None


def tokens_of(receipt: dict) -> dict[str, int | None]:
    """The token counts a receipt reports, each summed over `modelUsage` (a capped call's `usage` block reads
    zero while `modelUsage` keeps the real counts), else `usage`'s own count, else None. Never 0 for a count
    the receipt does not carry."""
    usage = receipt.get("modelUsage")
    models = [m for m in usage.values() if isinstance(m, dict)] if isinstance(usage, dict) else []
    flat = receipt.get("usage") if isinstance(receipt.get("usage"), dict) else {}
    tokens: dict[str, int | None] = {}
    for name, per_model, overall in TOKEN_FIELDS:
        counts = [c for c in (_count(m.get(per_model)) for m in models) if c is not None]
        tokens[name] = sum(counts) if counts else _count(flat.get(overall))
    return tokens


def open_call(task: Task, seat: str, attempt: int, cap: float, message: str | None, unit: str | None = None) -> Call:
    """Add a pending record for a call about to start, with its reserved cap; the caller saves the task
    before the process launches."""
    call = Call(id=f"c{len(task.calls) + 1}", seat=seat, attempt=attempt, cap_usd=cap, unit=unit, tick=task.tick,
                message_bytes=None if message is None else len(message.encode("utf-8")))
    task.calls.append(call)
    return call


def resolve(call: Call, charge: float, receipt: dict | None = None, *, timed_out: bool = False,
            interrupted: bool = False) -> None:
    """Fill a record from how its call ended: the charge the seat books and the receipt, None when the call
    left no usable one. What the receipt does not carry stays None."""
    if interrupted:
        call.outcome = "interrupted"
    elif timed_out:
        call.outcome = "timed_out"
    elif receipt is None:
        call.outcome = "no_receipt"
    else:
        call.outcome = "error" if receipt.get("is_error") else "ok"
    call.charge_usd = charge
    if receipt is not None:
        call.receipt_cost_usd = reported_cost(receipt)
        call.subtype = receipt.get("subtype") if isinstance(receipt.get("subtype"), str) else None
        call.tokens = tokens_of(receipt)
        call.session_id = receipt.get("session_id") if isinstance(receipt.get("session_id"), str) else None
    call.actual_usd = UNKNOWN if call.receipt_cost_usd is None else call.receipt_cost_usd


def book(task: Task, call: Call) -> None:
    """Add one resolved call's charge to `task.cost_usd` and mark it booked, leaving every other record as it is (the
    gate's reader call, booked while the director's own records still wait for their step)."""
    if not call.booked and call.outcome != "pending":
        task.cost_usd += call.charge_usd or 0.0
        call.booked = True


def mark_booked(task: Task) -> list[Call]:
    """Mark booked every resolved record not yet booked, once the caller has added those calls' charges to
    `task.cost_usd`; returns them for the log."""
    newly = [call for call in task.calls if not call.booked and call.outcome != "pending"]
    for call in newly:
        call.booked = True
    return newly


def recover(task: Task) -> tuple[list[Call], list[Call]]:
    """Book, once, every record a kill left unbooked: one holding its receipt is charged what the receipt
    said; one still pending left no receipt, so it is charged its reserved cap, marked lost, and its actual
    spend is UNKNOWN. Returns (booked, lost); a booked record is never charged again."""
    booked: list[Call] = []
    lost: list[Call] = []
    for call in task.calls:
        if call.booked:
            continue
        if call.outcome == "pending":
            call.outcome, call.charge_usd, call.actual_usd = "lost", call.cap_usd, UNKNOWN
            lost.append(call)
        task.cost_usd += call.charge_usd or 0.0
        call.booked = True
        booked.append(call)
    return booked, lost
