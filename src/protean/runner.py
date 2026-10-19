"""The one subprocess seam for model calls, and the dollar cap each call may carry.

A Runner takes argv, stdin, cwd and a wall timeout and returns a RunResult. A runner that can also hand over its
stdout line by line while the process runs says so with `streams_lines` and takes an `on_line` callback (the editor's
transcript); `launch` passes the callback only to such a runner. Every call's `--max-budget-usd` is its configured
cap clipped to what the task has left; below the call floor no call is made at all.
"""

from __future__ import annotations

import math
import os
import signal
import subprocess
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from protean.graph.budget import Caps
from protean.state import Task

KILL_DRAIN_SECONDS = 5
CALL_FLOOR_USD = 0.50


@dataclass
class RunResult:
    """What one process left behind. `timed_out` means the wall cap killed its process group."""

    returncode: int
    stdout: str
    stderr: str
    timed_out: bool = False


Runner = Callable[[list[str], str | None, Path, int], RunResult]
Charge = Callable[[float], None]
LineSink = Callable[[str], None]


def launch(runner: Runner, argv: list[str], stdin: str | None, cwd: Path, timeout: int,
           on_line: LineSink) -> RunResult:
    """One call through `runner`. A runner with `streams_lines` hands each stdout line to `on_line` while the
    process runs; any other runner is called as it always was and leaves no transcript."""
    if getattr(runner, "streams_lines", False):
        return runner(argv, stdin, cwd, timeout, on_line=on_line)
    return runner(argv, stdin, cwd, timeout)


def no_charge(usd: float) -> None:
    """The default `charge`: a caller that keeps no books."""


def usd_left(task: Task, caps: Caps, spent: float = 0.0) -> float:
    """Dollars left under the task's ceiling, after `spent` more."""
    return caps.max_usd + task.extra_usd - task.cost_usd - spent


def _cents(value: float) -> str:
    """Dollars rounded down to the cent, never below zero."""
    return f"{max(math.floor(value * 100 + 1e-9) / 100, 0.0):.2f}"


def floor_reason(task: Task, caps: Caps, spent: float = 0.0) -> str | None:
    """Why no call may be made: less left than the call floor. None while a call may be made."""
    left = usd_left(task, caps, spent)
    if left < CALL_FLOOR_USD:
        return f"max_usd: remaining ${_cents(left)} below the ${CALL_FLOOR_USD:.2f} call floor"
    return None


def usd_cap(configured: float, task: Task, caps: Caps, spent: float = 0.0) -> str:
    """The `--max-budget-usd` value: the configured cap clipped to what is left, down to the cent."""
    return _cents(min(configured, usd_left(task, caps, spent)))


PER_CALL_LIMIT, TASK_BUDGET = "per_call_limit", "task_budget"


def cap_bound(configured: float, task: Task, caps: Caps, spent: float = 0.0) -> str:
    """Which constraint set `usd_cap`'s value: the configured per-call limit, or what the task had left."""
    return PER_CALL_LIMIT if configured <= usd_left(task, caps, spent) else TASK_BUDGET


def charged_cap(argv: list[str]) -> float:
    """The `--max-budget-usd` a call carried: what it costs when it leaves no usable receipt."""
    return float(argv[argv.index("--max-budget-usd") + 1])


def _kill_group(proc: subprocess.Popen) -> None:
    """SIGKILL the child's whole process group, so the tools it started die with it. The child leads its own
    session, so the group id is its pid, which still names the group after the leader itself has been reaped."""
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except OSError:
        proc.kill()


def real_runner(argv: list[str], stdin: str | None, cwd: Path, timeout: int,
                on_line: LineSink | None = None) -> RunResult:
    """Run `argv` in its own session; on timeout kill the group and report `timed_out`.

    Any other interruption (Ctrl-C included, which the child's own session never sees) kills the
    group too before it propagates. With `on_line`, stdout is read line by line as the process writes it and
    each line (without its newline) goes to `on_line` before the call returns; the result still carries the
    whole stdout, and the wall covers the drain.
    """
    try:
        proc = subprocess.Popen(
            argv, cwd=cwd, text=True, start_new_session=True,
            stdin=subprocess.DEVNULL if stdin is None else subprocess.PIPE,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, errors="replace" if on_line else None,
        )
    except OSError as exc:
        return RunResult(127, "", f"could not start {argv[0]}: {exc}")
    if on_line is not None:
        return _streamed(proc, stdin, timeout, on_line)
    try:
        out, err = proc.communicate(stdin, timeout=timeout)
    except subprocess.TimeoutExpired:
        _kill_group(proc)
        try:
            out, err = proc.communicate(timeout=KILL_DRAIN_SECONDS)
        except subprocess.TimeoutExpired:
            proc.kill()
            out, err = "", ""
        return RunResult(proc.returncode if proc.returncode is not None else -9, out or "", err or "", True)
    except BaseException:
        _kill_group(proc)
        try:
            proc.wait(timeout=KILL_DRAIN_SECONDS)
        except subprocess.TimeoutExpired:
            pass
        raise
    return RunResult(proc.returncode, out, err)


real_runner.streams_lines = True  # type: ignore[attr-defined]


def _streamed(proc: subprocess.Popen, stdin: str | None, timeout: int, on_line: LineSink) -> RunResult:
    """`real_runner` with its stdout handed over line by line: one thread reads stdout and calls `on_line`, one
    reads stderr, one feeds stdin, while this thread waits on the process under the wall. A failing `on_line`
    never stops the read: the transcript is presentation, the stdout is the receipt."""
    out: list[str] = []
    err: list[str] = []

    def read_out() -> None:
        for line in proc.stdout:
            out.append(line)
            try:
                on_line(line.rstrip("\n"))
            except Exception:  # noqa: BLE001 - a transcript write never costs the receipt
                pass

    def read_err() -> None:
        err.append(proc.stderr.read())

    def feed() -> None:
        try:
            proc.stdin.write(stdin)
        except (BrokenPipeError, OSError, ValueError):
            pass
        finally:
            try:
                proc.stdin.close()
            except (BrokenPipeError, OSError):
                pass

    pumps = [threading.Thread(target=read_out, daemon=True), threading.Thread(target=read_err, daemon=True)]
    if stdin is not None:
        pumps.append(threading.Thread(target=feed, daemon=True))
    for pump in pumps:
        pump.start()
    deadline = time.monotonic() + timeout
    try:
        proc.wait(timeout=timeout)
        for pump in pumps:  # the pipes drain inside the same wall; a child still holding one is killed
            pump.join(max(deadline - time.monotonic(), 0))
        timed_out = any(pump.is_alive() for pump in pumps)
    except subprocess.TimeoutExpired:
        timed_out = True
    except BaseException:
        _kill_group(proc)
        try:
            proc.wait(timeout=KILL_DRAIN_SECONDS)
        except subprocess.TimeoutExpired:
            pass
        raise
    if timed_out:
        _kill_group(proc)
        for pump in pumps:
            pump.join(KILL_DRAIN_SECONDS)
        try:
            proc.wait(timeout=KILL_DRAIN_SECONDS)
        except subprocess.TimeoutExpired:
            proc.kill()
        return RunResult(proc.returncode if proc.returncode is not None else -9, "".join(out), "".join(err), True)
    return RunResult(proc.returncode, "".join(out), "".join(err))
