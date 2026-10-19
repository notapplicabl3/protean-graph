"""The `protean` command: think, resume, status, dry and ctl.

Exit codes: parked 12 (`interrupted`: a question, a held progression or a wait for a thought, with no client
attached), stopped 11 (a ceiling, or `--abandon`), `stop` or Ctrl-C 130, a usage error or a refusal 2 (an unknown verb
such as the retired `run` included); 0 for a command that only reads or writes state (`status`, `ctl`), and for a
record of runs 1–6 that ended `done` (a session never does)."""

from __future__ import annotations

import argparse
import json
import re
import signal
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

from protean import dry, loop, state
from protean.graph import budget, console
from protean.graph.budget import Caps, load_caps
from protean.operator import load_operator
from protean.runner import real_runner
from protean.state import Root

DEFAULT_BRAIN = dry.REPO / "brain" / "protean"
EXIT = {"done": 0, "stopped": 11, "interrupted": 12}
EXIT_USAGE = 2
EXIT_INTERRUPTED = 130


def parse_extend(text: str) -> tuple[float, int]:
    """`N` more ticks or `$X` more dollars, as (dollars, ticks)."""
    if re.fullmatch(r"[1-9]\d*", text):
        return 0.0, int(text)
    match = re.fullmatch(r"\$(\d+(?:\.\d+)?)", text)
    if match and float(match.group(1)) > 0:
        return float(match.group(1)), 0
    if re.fullmatch(r"\d+m", text):
        raise ValueError(f"--extend {text}: there is no wall-clock ceiling to extend; use N (ticks) or $X")
    raise ValueError(f"--extend {text!r}: expected N (ticks) or $X (dollars)")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="protean", description="A thinker moves the operator's thought forward one move at a "
                                     "time; workers do its tasks and explores, a grader decides.")
    parser.add_argument("--brain", metavar="PATH", type=Path, default=DEFAULT_BRAIN,
                        help="the brain root (default: <repo>/brain/protean)")
    verbs = parser.add_subparsers(dest="verb", metavar="VERB", required=True)
    think = verbs.add_parser("think", help="start a session on a thought and run it until it parks or stops")
    think.add_argument("thought")
    think.add_argument("--workspace", metavar="PATH",
                       help="the git clone tasks and explores branch from (default: the operator profile's `workspace`, "
                            "else a fresh scratch repository under the brain, <brain>/workspaces/<session>/)")
    think.add_argument("--project", default="_root", metavar="NAME", help="the repo the workspace clones (default _root)")
    think.add_argument("--mode", choices=("auto", "hold"), default="auto",
                       help="auto (default): progress on its own; hold: a task move's plan, each retry and each explore "
                            "wait for approve")
    resume = verbs.add_parser("resume", help="continue a session: the named one, else the current one")
    resume.add_argument("session", nargs="?", help="the session to wake (default: the current one); it becomes current")
    resume.add_argument("--extend", metavar="N|$X", action="append",
                        help="`ctl extend` before resuming: widen by N ticks or $X; repeat it to widen both")
    resume.add_argument("--abandon", action="store_true", help="close its questions and stop it for good")
    verbs.add_parser("status", help="show the current session: its thought record, moves, units and budget")
    scenario = verbs.add_parser("dry", help="run a scripted scenario on a scratch root; no model is called")
    scenario.add_argument("scenario")
    scenario.add_argument("--serve", action="store_true",
                          help="the fixture brain: serve the console on the scratch root, start paused, wait on holds")
    scenario.add_argument("--mode", choices=sorted(loop.MODES), default=loop.AUTO, help="start in auto (default) or hold")
    scenario.add_argument("--delay", type=float, default=1.0, metavar="SECONDS",
                          help="with --serve: the pause before each scripted call (default 1.0)")
    scenario.add_argument("--ticks", type=int, metavar="N", help="cap the scenario at N ticks (a ceiling to extend)")
    ctl = verbs.add_parser("ctl", help="operate the current session's console (live brain, or task.json when parked)")
    ops = ctl.add_subparsers(dest="op", metavar="OP", required=True)
    ops.add_parser("pause", help="hold at the next tick boundary (live brain)")
    ops.add_parser("resume", help="release a pause (live brain)")
    ops.add_parser("stop", help="the Ctrl-C path: book the call in flight, save, exit 130 (live brain)")
    think_op = ops.add_parser("think", help="add a thought for the thinker; it ends a wait for one and returns what is "
                                            "held")
    think_op.add_argument("text", nargs="+")
    think_op.add_argument("--decide", action="store_true", help="the thought is also the operator's decision: it enters `decided`")
    answer = ops.add_parser("answer", help="answer one open question by id")
    answer.add_argument("id")
    answer.add_argument("text", nargs="+")
    extend = ops.add_parser("extend", help="widen the session by N ticks or $X; the only widening")
    extend.add_argument("amount", nargs="+", metavar="N|$X")
    mode = ops.add_parser("mode", help="auto: the session progresses on its own; hold: a task move's plan, each retry "
                                       "and each explore wait for approve")
    mode.add_argument("mode", choices=("auto", "hold"))
    ops.add_parser("approve", help="release the one plan, retry or explore held in hold mode")
    ops.add_parser("snapshot", help="print the session as the console sees it")
    ops.add_parser("watch", help="print the snapshot, then the stream, until the run ends (an attached client)")
    return parser


def status(root: Root) -> int:
    """Print the current session, or say there is none: where it stands and what it may spend (its ceilings: the
    allowance and what it is made of), then its thought record as `thought.md` renders it
    (the thought, standing, open, decided and the moves), the units, the operator's thoughts and the open questions."""
    task = state.current(root)
    if task is None:
        print(f"no session in {root.path}; start one with `protean think \"<thought>\"`")
        return 0
    caps = load_caps(root)
    print(f"session: {task.id} ({task.project})\nworkspace: {task.workspace}")
    print(f"tick: {task.tick} of {caps.max_ticks + task.extra_ticks}")
    print(f"terminal: {task.terminal or 'running'}" + (f" — {task.stop_reason}" if task.stop_reason else "")
          + (" (paused)" if task.paused else ""))
    print(f"cost: ${task.cost_usd:.2f} of ${budget.allowance(task, caps):.2f}")
    print(f"allowance: ${caps.max_usd:.2f} + ${task.extra_usd:.2f} extended + ${budget.accrued_usd(task, caps):.2f} "
          f"accrued over {task.active_seconds:.0f}s of active work")
    print(f"mode: {task.mode}")
    if task.held is not None:
        print(f"held: {console.held_label(task.held)} — " + ("approved, released at the next tick" if task.held.approved
                                                              else "`protean ctl approve` releases it"))
    if task.awaiting_thought:
        print(f"awaiting a thought after {loop.awaited(task)}: `protean ctl think <text>` adds one")
    print(state.render_thought(task), end="")
    print("units:" if task.units else "units: none yet")
    for unit in task.units:
        print(f"  {unit.id:<10} {unit.status:<9} attempts {unit.attempts}  {unit.intent}")
    for number, thought in enumerate(task.thoughts, start=1):
        print(f"think {number}: {thought.text}")
    for question in state.open_questions(task):
        print(f"question {question.id} ({question.asker}): {question.text}")
    return 0


def _ctl_body(args: argparse.Namespace) -> dict:
    if args.op == "think":
        return {"text": " ".join(args.text), "decide": args.decide}
    if args.op == "answer":
        return {"id": args.id, "text": " ".join(args.text)}
    if args.op == "extend":
        amounts = [parse_extend(text) for text in args.amount]
        return {"usd": sum((u for u, _ in amounts), 0.0), "ticks": sum(t for _, t in amounts)}
    if args.op == "mode":
        return {"mode": args.mode}
    return {}


def ctl(root: Root, args: argparse.Namespace) -> int:
    """One console operation on the current session: POSTed to the live brain, or, when none drives it, written
    straight into its `task.json` (answer, think, extend, mode, approve; pause, resume and stop need a live brain)."""
    task = state.current(root)
    if task is None:
        raise ValueError(f"no session in {root.path}")
    if args.op == "watch":
        console.watch(root, task.id, print)
        return 0
    if args.op == "snapshot":
        try:
            status, payload = console.request(root, task.id, "GET", "/snapshot")
        except console.Unavailable:
            status, payload = 200, console.snapshot(task, load_caps(root), False)
        print(json.dumps(payload, indent=2))
        return 0 if status == 200 else EXIT_USAGE
    body = _ctl_body(args)
    event = loop.offline_op(root, load_caps(root), args.op, body)
    if event is not None:
        print(f"{console.describe(event)} (no brain is live: written to task.json; `protean resume` continues)")
        return 0
    status, payload = console.request(root, task.id, "POST", f"/{args.op}", body)
    if status != 200:
        print(f"protean: {args.op} refused ({status}): {payload.get('error', payload)}", file=sys.stderr)
        return EXIT_USAGE
    print(console.describe(payload["event"]))
    return 0


def workspace_for(named: str | None, root: Root) -> str | None:
    """The workspace a `think` runs on: `--workspace` when given, else the operator profile's `workspace`, else None
    (the loop then makes a scratch repository under the brain)."""
    return named if named is not None else load_operator(root).workspace


def _unattended(caps: Caps) -> float:
    """The unattended wait: `config.json`'s `unattended_seconds` when set, else the console's default."""
    return console.UNATTENDED_SECONDS if caps.unattended_seconds is None else caps.unattended_seconds


def _main(argv: Sequence[str] | None) -> int:
    args = build_parser().parse_args(argv)
    root = Root(args.brain.expanduser().resolve())
    try:
        if args.verb == "status":
            return status(root)
        if args.verb == "ctl":
            return ctl(root, args)
        if args.verb == "dry":
            task = dry.dry(args.scenario, root, serve=args.serve, mode=args.mode,
                           delay=args.delay if args.serve else 0.0, ticks=args.ticks)
        elif args.verb == "think":
            caps = load_caps(root)
            task = loop.run(args.thought, workspace_for(args.workspace, root), root, caps, real_runner, args.project, echo=print,
                            unattended=_unattended(caps), notify=console.notify, mode=args.mode)
        else:
            if args.extend and args.abandon:
                raise ValueError("--extend and --abandon do not go together")
            extends = [parse_extend(text) for text in args.extend or []]
            usd, ticks = sum((u for u, _ in extends), 0.0), sum(t for _, t in extends)
            caps = load_caps(root)
            task = loop.resume(root, caps, real_runner, usd, ticks, args.abandon, echo=print,
                               unattended=_unattended(caps), notify=console.notify, session=args.session)
    except (ValueError, OSError, RuntimeError, subprocess.CalledProcessError, console.Unavailable) as exc:
        print(f"protean: {exc}", file=sys.stderr)
        return EXIT_USAGE
    except KeyboardInterrupt:
        print("protean: interrupted; any spend in flight is booked on the session. `protean resume` continues it.",
              file=sys.stderr)
        return EXIT_INTERRUPTED
    return EXIT.get(task.terminal or "", EXIT_USAGE)


def _interrupt(signum: int, frame: object) -> None:
    """SIGINT, SIGTERM and SIGHUP take the Ctrl-C path, so a closed terminal, a kill or the console's `stop` still
    books the spend. SIGINT is installed explicitly: a process started in the background (a shell's `&`, launchd)
    inherits it ignored, which left `stop` and Ctrl-C without effect (found 2026-10-11)."""
    raise KeyboardInterrupt


def main(argv: Sequence[str] | None = None) -> int:
    for signum in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(signum, _interrupt)
    return _main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
