"""The console contract file names every event the brain can put on the stream (log event
shapes are part of the contract), and every field it lists for an event or the snapshot is one the code emits: the fields are harvested from the logs of four dry scenarios and the `think` and `think-rest`
fixtures, plus one emission of each event they do not reach, each through the code path that logs
it. The two thinking fixtures are checked the other way too: every event they log is in the
contract's Events table, and their snapshots carry exactly the fields its Snapshot section lists."""

from __future__ import annotations

import json
import os
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pytest

from protean import dry, loop, state
from protean.dry import REPO, ScriptedRunner, make_workspace, scenario_runner
from protean.graph import console, director, gate, questions, worker
from protean.graph.budget import Caps
from protean.state import DIRECTOR, EXPLORE, PLAN, RETRY, RUNTIME, Held, Idea, OpenItem, Predicate, Root, Task, Unit

EMITTED = re.compile(r"""["']event["']\s*:\s*["']([a-z_]+)["']""")
DOC_ROW = re.compile(r"^\|\s*(`[a-z_]+`(?:\s*,\s*`[a-z_]+`)*)\s*\|", re.MULTILINE)
TOKEN = re.compile(r"`([^`]*)`|[(\[{]|[)\]}]")
FIELD = re.compile(r"[a-z_]+")
SCENARIOS = ("branch", "weaken", "deps-broken", "deps-blocked")
THINKING = ("think", "think-rest")
# Goal mode's vocabulary: `goal` as a word; `delivered.to_director` its delivered note.
RETIRED = re.compile(r"steer|directions|\bgoal\b|done_refused|delivered.to_director")


def emitted() -> set[str]:
    """Every event name a source file writes as a literal, plus the console's operations (logged under their own
    names)."""
    names = set(console.OPS)
    for path in (REPO / "src").rglob("*.py"):
        names |= set(EMITTED.findall(path.read_text(encoding="utf-8")))
    return names


def _contract() -> str:
    return (REPO / "docs" / "console-api.md").read_text(encoding="utf-8")


def documented() -> set[str]:
    """The names in the first column of `docs/console-api.md`'s events table."""
    table = _contract()[_contract().index("## Events on the stream"):]
    return {name for cell in DOC_ROW.findall(table) for name in re.findall(r"`([a-z_]+)`", cell)} - {"event"}


def test_every_event_the_brain_emits_is_documented_and_nothing_documented_is_invented():
    assert {"transcript", "held", "approved", "returned", "mode", "approve", "think", "ideas", "explore", "rest",
            "thought", "no_progress"} <= emitted()
    assert emitted() == documented()


def test_the_contract_names_no_steer_no_directions_no_goal_no_done_refused_and_no_delivered_note():
    """goal mode's words do not drift back into the doc."""
    assert RETIRED.findall(_contract()) == []


# --- F5: the fields ---


def fields(text: str) -> list[str]:
    """The backticked field names at bracket depth 0 of a contract cell or paragraph: what sits inside (), [] or {} is
    a value list or a nested record, never a field of this level."""
    names, depth = [], 0
    for token in TOKEN.finditer(text):
        if token.group(1) is not None:
            if depth == 0 and FIELD.fullmatch(token.group(1)):
                names.append(token.group(1))
        elif token.group(0) in "([{":
            depth += 1
        else:
            depth -= 1
    return names


def group(text: str, name: str) -> str:
    """The bracketed group right after `` `name` `` in `text`, without its brackets."""
    start = text.index(f"`{name}` (") + len(f"`{name}` (")
    depth, end = 1, start
    while depth:
        depth += {"(": 1, "[": 1, "{": 1, ")": -1, "]": -1, "}": -1}.get(text[end], 0)
        end += 1
    return text[start:end - 1]


def event_rows() -> dict[str, list[str]]:
    """Each documented event name with the top-level fields its row lists (besides `ts` and `event`)."""
    table = _contract()[_contract().index("## Events on the stream"):]
    rows = {}
    for line in table.splitlines():
        cells = line.split("|")
        if len(cells) < 4 or not DOC_ROW.match(line) or "`event`" in cells[1]:
            continue
        for name in re.findall(r"`([a-z_]+)`", cells[1]):
            rows[name] = fields(cells[2])
    return rows


def snapshot_paragraph() -> str:
    text = _contract()
    return text[text.index("## Snapshot"):].split("\n\n")[1]


def snapshot_fields(name: str | None = None) -> set[str]:
    """The fields the Snapshot section lists at its top level, or inside the group of field `name`."""
    text = snapshot_paragraph()
    return set(fields(text if name is None else group(text, name)))


def _seed(base: Path) -> Root:
    root = Root(base / "brain")
    root.path.mkdir(parents=True)
    for prompt in (root.thinker_prompt, root.editor_prompt, root.reader_prompt):
        prompt.write_text("p\n")
    return root


def _answer(result: dict | str, cost: float = 0.2, **extra: Any) -> dict:
    return {"type": "result", "is_error": False, "session_id": "s", "total_cost_usd": cost,
            "result": result if isinstance(result, str) else json.dumps(result), **extra}


def _emit_the_rest(base: Path) -> tuple[Root, Task]:
    """One emission of each event the dry scenarios never reach, through the function that logs it (a console
    operation is logged as `Console.operate` logs it: the event `console.apply` returns)."""
    root, caps, said = _seed(base), Caps(), []
    task = Task(id="task-20261011T170000", goal="g", workspace=str(base / "unused"))
    state.save(root, task)

    def op(name: str, body: dict, live: bool = True) -> None:
        state.append_log(root, task.id, console.apply(task, caps, name, body, live))

    def interrupt(argv, stdin, cwd, timeout):
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        loop.run_tick(task, caps, root, interrupt)  # interrupted, question (q1, unknown_spend: no receipt)
    op("answer", {"id": "q1", "text": "yes"})
    loop._settle(task, caps, root, said.append)  # runtime_answer (q1)
    capped = _answer("", 2.0, is_error=True, subtype="error_max_budget_usd")
    loop.run_tick(task, caps, root, ScriptedRunner([capped], []))  # ceiling, question (q2, cap_hit)
    for name, body in (("pause", {}), ("resume", {}), ("think", {"text": "use tabs", "decide": True}), ("extend", {"usd": 1.0}),
                       ("mode", {"mode": "hold"}), ("stop", {})):
        op(name, body)
    questions.ask(root, task, DIRECTOR, "question", "which file?")
    op("answer", {"id": "q3", "text": "a.txt"})
    loop._settle(task, caps, root, said.append)  # delivered (q3), runtime_answer (q2, released by extend)
    u1 = Unit("u-1", "write a.txt", [Predicate("file_exists", {"path": "a.txt"})])
    for held in (Held(kind=PLAN, units=[u1]), Held(kind=RETRY, unit="u-1", attempt=2),
                 Held(kind=EXPLORE, move="m9", brief="try b.txt")):
        loop._hold_for_approve(task, held, root)  # held
        op("approve", {})
        loop._release(task, root, said.append)  # approved
    for held in (Held(kind=PLAN, units=[Unit("u-2", "i", [Predicate("file_exists", {"path": "b.txt"})])]),
                 Held(kind=RETRY, unit="u-1", attempt=2), Held(kind=EXPLORE, move="m9", brief="try b.txt")):
        loop._hold_for_approve(task, held, root)
        op("think", {"text": "smaller"})
        loop._settle(task, caps, root, said.append)  # returned
    loop._apply(director.Reply(stop="max_usd: remaining $0.40", errors=["bad"]), task, caps, root, admitted=True)
    unchanged = director.Reply(move=director.MoveReply(kind="rest", standing=task.standing), unchanged=True)
    loop._apply(unchanged, task, caps, root, admitted=True)  # no_progress, its note
    loop._apply(unchanged, task, caps, root, admitted=True)  # no_progress again, the runtime's question
    ideas = [Idea(f"i{n}", "a case", "one explore", "o1") for n in (1, 2, 3)]
    loop._accept(director.MoveReply(kind="ideas", standing="Three ways are open.", open=[OpenItem("o1", "which")],
                                    ideas=ideas), 0.1, task, root, admitted=True)  # ideas, thought
    loop._accept(director.MoveReply(kind="ask", standing="Two ways are left.", question="which one?"), 0.1, task, root,
                 admitted=True)  # an ask move, its question
    loop._accept(director.MoveReply(kind="rest", standing="It rests."), 0.1, task, root, admitted=True)  # rest
    loop._park(task, ("thought", task.moves[-1].id), caps, root)  # parked, on a thought
    task.terminal, task.stop_reason = "stopped", "max_ticks: 100 >= 100"
    loop._settle(task, caps, root, said.append)  # released
    loop._park(task, ("question", "q9"), caps, root)  # parked
    questions.ask(root, task, RUNTIME, "stale_tip", "moved", stale_tip="u-1")
    questions.repaired(root, task)  # repaired
    worker._transcript(root, task, u1, task.calls[0])('{"type": "system"}')  # transcript
    loop._prepare(task, root, caps, 0.0, 0, True, said.append)  # abandoned
    return root, task


def _thinking(base: Path, scenario: str) -> tuple[Root, Task, dict[str, Any]]:
    """One thinking fixture, its seats scripted, on a kept root: `think` driven by its scripted client; `think-rest`
    parked, then an offline `think` and a resume. The session, and its snapshot as a client attaching to it sees one
    (the scripted client's, for `think`)."""
    root, workspace, silent = _seed(base), make_workspace(base / "ws"), lambda _: None
    runner = scenario_runner(scenario)
    if scenario == "think":
        task, client = dry.run_scenario(scenario, root, workspace, runner, silent)
        return root, task, client.snapshot
    loop.run(f"dry scenario: {scenario}", workspace, root, Caps(), runner, unattended=0.0, notify=None)
    loop.offline_op(root, Caps(), "think", {"text": "write a README first"})
    task = loop.resume(root, Caps(), runner, unattended=0.0, notify=None)
    return root, task, console.snapshot(task, Caps(), live=False) | {"log_bytes": 0, "transcript_bytes": 0}


def _scenario(base: Path, scenario: str) -> tuple[Root, Task]:
    """One dry scenario's session on its own root; `branch` also logs a released held plan's refused branch (the
    `to_director` variant)."""
    root = _seed(base / scenario)
    task = loop.run(scenario, make_workspace(base / scenario / "ws"), root, Caps(), scenario_runner(scenario),
                    unattended=0.0, notify=None)
    if scenario == "branch":
        child = Unit("u-9", "i", [Predicate("file_exists", {"path": "x.py"}, parents=["u-2 p1"])], rationale="r")
        no = _answer({"verdict": "no", "note": "lost"}, 0.05)
        gate.admit(task, [child], Caps(), root, ScriptedRunner([], [], [no]), to_director=True)
    return root, task


def _keys(root: Root, task_id: str, into: dict[str, set[str]]) -> None:
    for path in (root.log_file(task_id), root.transcript_file(task_id)):
        for line in path.read_text(encoding="utf-8").splitlines() if path.exists() else []:
            event = json.loads(line)
            into.setdefault(event["event"], set()).update(event)


@pytest.fixture(scope="module")
def harvest(tmp_path_factory: pytest.TempPathFactory) -> tuple[dict[str, set[str]], dict[str, Any]]:
    """Every event name the harvest emitted with the union of its keys, and the snapshots: each thinking fixture's
    as a client attached to it, its final one, and one of a session with a unit, a thought and a decision. The six
    sessions run side by side, each on its own root and workspace (they wait on git and the sandbox, not on each
    other; 2026-10-18 trim), and are harvested once all have ended."""
    base, keys, snaps = tmp_path_factory.mktemp("contract"), {}, {}
    with pytest.MonkeyPatch.context() as patch:  # the battery's failing `claude` shim, as in every test
        patch.setenv("PATH", f"{Path(__file__).resolve().parent / 'shim'}{os.pathsep}{os.environ['PATH']}")
        with ThreadPoolExecutor(len(SCENARIOS) + len(THINKING)) as pool:
            scenarios = [pool.submit(_scenario, base, scenario) for scenario in SCENARIOS]
            thinking = {scenario: pool.submit(_thinking, base / scenario, scenario) for scenario in THINKING}
        for run in scenarios:
            root, task = run.result()
            _keys(root, task.id, keys)
        for scenario, run in thinking.items():
            root, task, attached = run.result()
            _keys(root, task.id, keys)
            snaps[scenario] = attached
            snaps[f"{scenario} final"] = console.snapshot(task, Caps(), live=False) | {"log_bytes": 0,
                                                                                         "transcript_bytes": 0}
            snaps[f"{scenario} events"] = {event["event"] for event in state.read_log(root, task.id)}
        root, task = _emit_the_rest(base / "rest")
        _keys(root, task.id, keys)
    snaps["rest"] = console.snapshot(task, Caps(), live=False) | {"log_bytes": 0, "transcript_bytes": 0}
    return keys, snaps


def test_every_field_the_contract_lists_for_an_event_is_one_the_brain_emits(harvest):
    keys, _ = harvest
    rows = event_rows()
    assert sorted(rows) == sorted(documented()) and all(rows.values())
    missing = {name: sorted(set(listed) - keys.get(name, set())) for name, listed in rows.items()}
    assert {name: gone for name, gone in missing.items() if gone or name not in keys} == {}


def _union(items: list[dict[str, Any]]) -> set[str]:
    return set().union(*(set(item) for item in items))


def test_every_field_the_contract_lists_for_the_snapshot_is_in_it(harvest):
    _, snaps = harvest
    snap, think = snaps["rest"], snaps["think final"]
    assert sorted(snapshot_fields() - set(snap)) == []
    assert sorted(snapshot_fields("ceilings") - set(snap["ceilings"])) == []
    assert "reader_call_usd" in snapshot_fields("ceilings")
    assert sorted(snapshot_fields("units") - set(think["units"][0])) == []
    assert sorted(snapshot_fields("thoughts") - set(snap["thoughts"][0])) == []
    assert sorted(snapshot_fields("thought") - set(snap["thought"])) == []
    assert sorted(snapshot_fields("moves") - _union(think["moves"] + snap["moves"])) == []
    unit = Unit("u-1", "write a.txt", [Predicate("file_exists", {"path": "a.txt"})])
    held = [console.held_view(h) for h in (Held(kind=PLAN, units=[unit]), Held(kind=RETRY, unit="u-1", attempt=2),
                                           Held(kind=EXPLORE, move="m2", brief="try b.txt"))]
    assert snapshot_fields("held") == _union(held)


@pytest.mark.parametrize("scenario", THINKING)
def test_the_thinking_fixtures_log_only_documented_events_and_show_exactly_the_documented_snapshot(harvest, scenario):
    """every event `think` and `think-rest` log is in the Events table, and the snapshot a client attaching
    to each sees, and its final one, carry exactly the top-level fields the Snapshot section lists, the thought record
    and each thought exactly theirs, and each move only fields the section lists."""
    _, snaps = harvest
    assert sorted(snaps[f"{scenario} events"] - documented()) == []
    for snap in (snaps[scenario], snaps[f"{scenario} final"]):
        assert set(snap) == snapshot_fields() and "goal" not in snap and "directions" not in snap
        assert set(snap["thought"]) == snapshot_fields("thought")
        assert all(set(thought) == snapshot_fields("thoughts") for thought in snap["thoughts"])
        assert _union(snap["moves"]) <= snapshot_fields("moves")
    final = snaps[f"{scenario} final"]
    assert final["thoughts"] and final["moves"]  # the final snapshot shows the checks something to check
    if scenario == "think":
        assert {"ideas", "think", "explore", "verdict", "integrated", "rest", "thought", "parked"} <= snaps[
            "think events"]
        assert [m["kind"] for m in final["moves"]] == ["ideas", "explore", "task", "rest"]
