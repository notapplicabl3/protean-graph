"""The node registry: every node of the graph declared once, as data, and the order a tick walks them in.

Each node has a name, who fires it (`runtime` or `thinker`) and its edges: `after` names the entries that run before
it in a tick and `before` the entries that run after it, whenever both are in the tick; `brings` names the nodes its
firing adds to the tick after it; `within` names the nodes inside whose call it also runs (the one nested edge: the
gate inside the thinker call, as its admit hook). An entry in every tick is `always`; `queued` marks a firing that waits
for a later tick, one per tick. `id` tells apart the two entries of one node (block runs after repair and again after
the fired node), and an edge names an `id`.

The ring (`loop.py`) is the walker: it takes the tick's firing set, adds the required nodes, orders the set along these
edges (`order`) and runs it. The edges alone order every tick: two entries of one tick that no edge orders, or a cycle,
are refused (`Unordered`), never settled by where they are declared. Every legal shape of a tick is declared too
(`SHAPES`), and `check` orders each one at import, so a registry that leaves one unordered is refused before any tick
runs, never mid-walk after a paid call. Nothing here imports the rest of the brain, so the record (`state.py`) and the
decoder (`graph/director.py`) read it too.

The thinker fires `worker` (a `task` or an `explore` move), `console` (an `ask` or an `ideas` move) and `rest`, one
firing per reply in this version (`FIRES`, `firing_errors`). Every other node is the runtime's, and the thinker can
neither fire, skip nor reorder it.
"""

from __future__ import annotations

from collections.abc import Collection, Sequence
from dataclasses import dataclass

RUNTIME, THINKER = "runtime", "thinker"


@dataclass(frozen=True)
class Node:
    """One declared entry of the registry; see the module docstring for each field."""

    name: str
    fired_by: str
    after: tuple[str, ...] = ()
    before: tuple[str, ...] = ()
    brings: tuple[str, ...] = ()
    within: tuple[str, ...] = ()
    always: bool = False
    queued: bool = False
    id: str = ""

    @property
    def key(self) -> str:
        """The id an edge names: `id`, else the node's name."""
        return self.id or self.name


NODES: tuple[Node, ...] = (
    # Finish a pass cut between its two writes, and write once a rest's memory line a kill cut off; end the tick on a
    # terminal session, an open question, something held for approve, or a wait for a thought. First in every tick.
    Node("repair", RUNTIME, always=True),
    # Block the pending units whose needs died: after repair, and again after the fired node.
    Node("block", RUNTIME, always=True, after=("repair",)),
    # Every ceiling and the call floor, before every node that makes a paid call: the release (a held plan's reader
    # calls), the thinker, the worker (a unit or an explore) and the gate (the weakening reader call).
    Node("budget", RUNTIME, always=True, after=("block",), before=("release", "director", "worker", "gate")),
    # The held plan `approve` (or `auto`) released passes the gate's admission, and its units enter the session; an
    # answered `reader_cap` weakening check is finished here.
    Node("release", RUNTIME, always=True, before=("pick",)),
    # Draw the tick's firing from the queue: the runnable unit, else the queued explore (a worker firing either way);
    # with neither, the runtime fires the thinker.
    Node("pick", RUNTIME, always=True, before=("hold",)),
    # In `hold`, the picked retry or explore waits for `approve`.
    Node("hold", RUNTIME, always=True, before=("director", "worker")),
    # The thinker's call (the seat key stays `director`), fired by the runtime when the tick drew no worker firing.
    Node("director", RUNTIME),
    # One unit's attempt, or one explore (the worker's explore form, never graded: it brings nothing).
    Node("worker", THINKER, brings=("grader", "gate", "ref"), queued=True),
    # The free checks of the attempt, in the grading clone: its own grade, the re-checks, the clean clone.
    Node("grader", RUNTIME, after=("worker",)),
    # The weakening check once every free check passed (one reader call when a protected path was touched); inside
    # the thinker call, branch admission.
    Node("gate", RUNTIME, after=("grader",), within=("director",)),
    # The ref move: the verdict, and on a pass its two durable writes (`passed`, then `task/<id>` and `integrated`).
    Node("ref", RUNTIME, after=("gate",)),
    # An ask opens the thinker's question; an ideas move proposes its options.
    Node("console", THINKER, after=("director",)),
    # The session waits for a thought.
    Node("rest", THINKER, after=("director",), brings=("memory",)),
    # The brain's long-term memory: one line after every rest (a close writes its own, outside any tick).
    Node("memory", RUNTIME, after=("rest",)),
    Node("block", RUNTIME, id="block2", always=True,
         after=("hold", "director", "worker", "ref", "console", "rest", "memory")),
    # Count the tick and its active time, save, echo the tick's line. Last when it runs.
    Node("save", RUNTIME, always=True, after=("block2",)),
    # The reader seat: one fresh call inside the gate (a branch, a weakening check) or the release; never walked on
    # its own and never the thinker's to fire.
    Node("reader", RUNTIME),
)

# What each of the five moves fires: one node, so a reply fires exactly one in this version.
FIRES: dict[str, tuple[str, ...]] = {
    "task": ("worker",), "explore": ("worker",), "ask": ("console",), "ideas": ("console",), "rest": ("rest",),
}


@dataclass(frozen=True)
class Shape:
    """One legal shape of a tick, by what it adds to the entries in every tick: `adds`, the nodes the runtime fires
    into it (the thinker), and `fire`, its one firing, with `unit` for a worker on a task unit (a sample id; none for
    an explore). What the firing brings follows from the registry (`brought`)."""

    adds: tuple[str, ...] = ()
    fire: str | None = None
    unit: str | None = None


# Every legal shape of a tick. A held plan's release and a hold for `approve` add no shape: release and hold are in
# every tick, and a held firing keeps the shape it was drawn in.
SHAPES: dict[str, Shape] = {
    "no firing": Shape(),  # a tick that ends before pick draws a firing or the runtime fires the thinker
    "thinker": Shape(adds=("director",)),  # its move fires the worker, which waits in the queue, or nothing runs now
    "thinker, then its console firing": Shape(adds=("director",), fire="console"),
    "thinker, then its rest firing": Shape(adds=("director",), fire="rest"),
    "worker on a unit": Shape(fire="worker", unit="u-1"),
    "worker on an explore": Shape(fire="worker"),
}


class Unordered(ValueError):
    """The edges leave two entries of one tick unordered, form a cycle, or a node entered the tick behind one that
    must follow it; or the registry is refused another way (`check`), or a bare name two entries share was asked for
    (`node`)."""


def entries(nodes: Sequence[Node] | None = None) -> dict[str, Node]:
    """Each entry by the id its edges are named by."""
    return {node.key: node for node in (NODES if nodes is None else nodes)}


def named(name: str, nodes: Sequence[Node] | None = None) -> list[Node]:
    """Every entry of the node called `name`: none for an undeclared name, two for block."""
    return [n for n in (NODES if nodes is None else nodes) if n.name == name]


def node(name: str, nodes: Sequence[Node] | None = None) -> Node | None:
    """The one entry of the node called `name`, or None when no node has that name. A name that two entries share
    (block, whose second entry is `block2`) is refused (`Unordered`): a caller names the entry it means by its key
    (`entries`), or reads every entry of the name (`named`)."""
    found = named(name, nodes)
    if len(found) > 1:
        raise Unordered(f"{name!r} names {len(found)} entries ({', '.join(n.key for n in found)}); name one by its key")
    return found[0] if found else None


def always(nodes: Sequence[Node] | None = None) -> list[str]:
    """The ids of the entries in every tick."""
    return [n.key for n in (NODES if nodes is None else nodes) if n.always]


def brought(name: str, unit: str | None = None, nodes: Sequence[Node] | None = None) -> tuple[str, ...]:
    """The nodes a firing of `name` adds to its tick. A worker firing without a unit is an explore: never graded, so
    it brings nothing."""
    if name == "worker" and unit is None:
        return ()
    found = node(name, nodes)
    return found.brings if found is not None else ()


def nested(name: str, host: str, nodes: Sequence[Node] | None = None) -> bool:
    """Whether node `name` runs inside the call of node `host` (a `within` edge)."""
    found = node(name, nodes)
    return found is not None and host in found.within


def order(ids: Collection[str], nodes: Sequence[Node] | None = None) -> list[str]:
    """The ids, in the one order their edges allow. An edge binds only when both its ends are in `ids`. Two entries
    ready at the same step, or a cycle, raise `Unordered`: the edges alone order a tick."""
    table = entries(nodes)
    present = set(ids)
    unknown = sorted(present - set(table))
    if unknown:
        raise Unordered(f"no entry {', '.join(unknown)} is declared")
    preds: dict[str, set[str]] = {key: set() for key in present}
    for key in present:
        entry = table[key]
        preds[key] |= {a for a in entry.after if a in present}
        for b in entry.before:
            if b in present:
                preds[b].add(key)
    walk: list[str] = []
    while len(walk) < len(present):
        ready = sorted(key for key in present if key not in walk and not preds[key] - set(walk))
        if not ready:
            raise Unordered(f"the edges form a cycle among {', '.join(sorted(present - set(walk)))}")
        if len(ready) > 1:
            raise Unordered(f"no edge orders {', '.join(ready)}")
        walk.append(ready[0])
    return walk


def walk(shape: str, nodes: Sequence[Node] | None = None) -> list[str]:
    """The ids of a tick of `shape` (`SHAPES`), in the one order their edges allow: the entries in every tick, the
    nodes the runtime fires into it, its firing and the nodes that firing brings."""
    found = SHAPES[shape]
    ids = always(nodes) + list(found.adds)
    if found.fire is not None:
        ids += [found.fire, *brought(found.fire, found.unit, nodes)]
    return order(ids, nodes)


def fires(kind: str) -> list[str]:
    """The nodes a move of `kind` fires (`FIRES`); none for a kind no move has (a goal-mode record's)."""
    return list(FIRES.get(kind, ()))


def firing_errors(fire: Sequence[str], nodes: Sequence[Node] | None = None) -> list[str]:
    """Why a decoded firing set is refused: not exactly one firing, a node no entry declares, or a node only the
    runtime fires. Empty when the thinker may fire it."""
    errors = []
    if len(fire) != 1:
        errors.append(f"a reply fires exactly one node in this version; it fired {list(fire) or 'none'}")
    for name in fire:
        found = named(name, nodes)
        if not found:
            errors.append(f"{name!r} is no declared node")
        elif any(n.fired_by != THINKER for n in found):
            errors.append(f"{name!r} is fired by the runtime, never by the thinker")
    return errors


def check(nodes: Sequence[Node] | None = None) -> None:
    """Refuse a registry that cannot be walked: a repeated id, an edge, a `brings` or a `within` naming nothing
    declared, an unknown `fired_by`, a thinker node that is not exactly worker, console and rest, or a shape of tick
    (`SHAPES`, the entries in every tick among them) that the edges do not order."""
    nodes = NODES if nodes is None else nodes
    keys = [n.key for n in nodes]
    if len(keys) != len(set(keys)):
        raise Unordered(f"an id is declared twice: {sorted(k for k in set(keys) if keys.count(k) > 1)}")
    names = {n.name for n in nodes}
    for n in nodes:
        if n.fired_by not in (RUNTIME, THINKER):
            raise Unordered(f"{n.key}: fired_by must be {RUNTIME} or {THINKER}, not {n.fired_by!r}")
        missing = [k for k in n.after + n.before if k not in keys] + [k for k in n.brings + n.within if k not in names]
        if missing:
            raise Unordered(f"{n.key} names {', '.join(missing)}, which no entry declares")
    if sorted(n.name for n in nodes if n.fired_by == THINKER) != ["console", "rest", "worker"]:
        raise Unordered("the thinker fires exactly worker, console and rest")
    for shape in SHAPES:
        try:
            walk(shape, nodes)
        except Unordered as exc:
            raise Unordered(f"a tick of shape {shape!r}: {exc}") from None


check()
