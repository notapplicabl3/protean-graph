"""Scripted seats for the dry path, so the battery and `protean dry` never reach a model.

A scenario is `fixtures/protean/<scenario>/`: `director.jsonl` (the thinker's replies, in its seat's key), `worker.jsonl`
(the worker's, for task units and explores alike, in call order), `reader.jsonl` when a reader is asked, and
`client.jsonl` when the scenario has a scripted console client (`ScriptedClient`). Each runs as a thinking session on a
scratch root and workspace. `think` is the thinking session end to end: ideas, then its scripted client sends a console
`think` that picks one, an explore, a task and a rest; `think-rest` is a single rest that parks unattended (an offline
`protean ctl think` and `protean resume` wake it on a kept root). The others re-drive the goal-mode scenarios as
sessions whose first move is a `task`: `dry`, `stuck`, `deps`, `deps-broken`, `deps-blocked`, `deps-delivery-only`,
`branch` and `weaken`. With `--serve`, any scenario is the fixture brain a console client develops against (`dry think
--serve` for the thought page): it starts paused, waits on holds, and its own scripted client stays out of the way.
"""

from __future__ import annotations

import http.client
import json
import shutil
import subprocess
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

from protean import loop
from protean.graph import console
from protean.graph.budget import load_caps
from protean.runner import RunResult
from protean.state import Root, Task

REPO = Path(__file__).resolve().parents[2]
FIXTURES = REPO / "fixtures" / "protean"
SEED_FILES = ("thinker.md", "editor.md", "reader.md", "config.json")
GIT_ID = ["-c", "user.name=protean dry", "-c", "user.email=dry@protean.invalid"]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """One object per non-blank line; a missing file is an empty script."""
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _git(args: list[str], cwd: Path) -> None:
    subprocess.run(["git", *GIT_ID, *args], cwd=cwd, check=True, capture_output=True, text=True)


class ScriptedRunner:
    """A Runner that answers from three queues and records every call it is given.

    A director entry (the seat's key; the thinker in its role, `director.jsonl` in a fixture) is the whole stdout object
    the CLI would print, its `result` one move plus `state`. A worker entry acts inside the clone it is started in, a
    unit's or an explore's: `write` files, `delete` paths, `commit` them, then report `result` with an `exit_code: N`
    line and `total_cost_usd`. A reader entry (the sandboxed seat without a write tool) is the stdout object too. Any
    seat's entry may instead be `{"timed_out": true}`, or `{"stdout": "<raw text>"}` for a call that leaves no receipt.
    """

    def __init__(self, director: list[dict[str, Any]], worker: list[dict[str, Any]],
                 reader: list[dict[str, Any]] | None = None) -> None:
        self.director, self.worker, self.reader = list(director), list(worker), list(reader or [])
        self.calls: list[tuple[list[str], str | None, Path, int]] = []
        self.spent: dict[str, float] = {}  # per session id: what its receipts have reported so far

    @staticmethod
    def seat(argv: list[str]) -> str:
        """Which seat an argv is: the worker carries the Write tool, the reader is sandboxed without it, the director is
        neither."""
        if "Write" in argv:
            return "worker"
        return "reader" if argv and argv[0].endswith("sandbox-exec") else "director"

    def __call__(self, argv: list[str], stdin: str | None, cwd: Path, timeout: int) -> RunResult:
        self.calls.append((argv, stdin, cwd, timeout))
        seat = self.seat(argv)
        if argv and argv[0].endswith("sandbox-exec") and argv[1:2] == ["-p"]:
            argv = argv[3:]
        if not argv or argv[0] != "claude":
            raise ValueError(f"the scripted runner only answers model calls, got {argv[:1]}")
        queue = {"worker": self.worker, "reader": self.reader, "director": self.director}[seat]
        if not queue:
            raise RuntimeError(f"the scripted {seat} has no reply left")
        entry = queue.pop(0)
        if seat == "worker":
            return self._work(entry, Path(cwd))
        if entry.get("timed_out"):
            return RunResult(-9, "", "", True)
        if "stdout" in entry:
            return RunResult(entry.get("returncode", 1), entry["stdout"], "scripted: no receipt")
        return RunResult(0, json.dumps(self._cumulative(entry, argv)), "")

    def _cumulative(self, entry: dict[str, Any], argv: list[str]) -> dict[str, Any]:
        """What the CLI prints: an entry's `total_cost_usd` is the call's own spend, but a call that resumes a session
        (`--resume <id>`, the thinker's within a warm stretch) reports the session's whole spend so far, as
        `claude -p --resume` does."""
        session, cost = entry.get("session_id"), float(entry.get("total_cost_usd", 0.0))
        if not isinstance(session, str):
            return entry
        resumed = "--resume" in argv and argv[argv.index("--resume") + 1] == session
        cost = self.spent.get(session, 0.0) + cost if resumed else cost
        self.spent[session] = cost
        return {**entry, "total_cost_usd": round(cost, 7)}

    @staticmethod
    def _work(entry: dict[str, Any], clone: Path) -> RunResult:
        for rel, content in entry.get("write", {}).items():
            path = clone / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
        for rel in entry.get("delete", []):
            (clone / rel).unlink()
        if entry.get("commit"):
            _git(["add", "-A"], clone)
            _git(["commit", "--quiet", "--allow-empty", "-m", "dry: scripted worker"], clone)
        if entry.get("timed_out"):
            return RunResult(-9, "", "", True)
        result = entry.get("result", "")
        if "exit_code" in entry:
            result += f"\nexit_code: {entry['exit_code']}"
        receipt = {"type": "result", "is_error": False, "session_id": "dry-worker", "result": result,
                   "total_cost_usd": entry.get("total_cost_usd", 0.0)}
        return RunResult(entry.get("returncode", 0), json.dumps(receipt), "")


CLIENT_SECONDS = 30.0  # how long the scripted client waits for the console, or for one line of the stream


class ScriptedClient(threading.Thread):
    """A fixture's scripted console client, from its `client.jsonl`: one step per line, `{"on": "<event>", "op":
    "<operation>", "body": {...}}`. It attaches to the live brain's stream, so a hold waits for it as for any client,
    and sends each step's operation through the console the first time the stream shows the step's event after the
    step before; after the last step it leaves, so the next hold parks as an unattended one does. `attached` is set once
    the brain counts it as attached (or once it fails: `error`); `snapshot` is the stream's first line, the snapshot a
    client attaches to; `sent` holds each operation's event as the console answered it; `finished` tells it the run
    ended."""

    def __init__(self, root: Root, steps: list[dict[str, Any]], timeout: float = CLIENT_SECONDS) -> None:
        super().__init__(name="protean-dry-client", daemon=True)
        self.root, self.steps, self.timeout = root, steps, timeout
        self.attached, self.finished = threading.Event(), threading.Event()
        self.snapshot: dict[str, Any] | None = None
        self.sent: list[dict[str, Any]] = []
        self.error: BaseException | None = None

    def run(self) -> None:
        try:
            self._drive()
        except BaseException as exc:  # surfaced by `run_scenario` once the run ends
            self.error = exc
        finally:
            self.attached.set()

    def _console(self) -> tuple[str, int, str]:
        """The session the brain serves, its port and its token, once its console file exists."""
        deadline = time.monotonic() + self.timeout
        while time.monotonic() < deadline and not self.finished.is_set():
            try:
                task_id = self.root.current_file.read_text(encoding="utf-8").strip()
                info = json.loads(self.root.console_file(task_id).read_text(encoding="utf-8"))
                return task_id, int(info["port"]), str(info["token"])
            except (OSError, ValueError, KeyError):
                time.sleep(0.02)
        raise RuntimeError("the brain served no console")

    def _drive(self) -> None:
        task_id, port, token = self._console()
        conn = http.client.HTTPConnection(console.HOST, port, timeout=self.timeout)
        try:
            conn.request("GET", "/stream", headers={"Authorization": f"Bearer {token}"})
            stream = conn.getresponse()
            first = stream.readline() if stream.status == 200 else b""
            if not first:
                raise RuntimeError(f"the console refused the stream: {stream.status}")
            self.snapshot = json.loads(first)["snapshot"]  # written once the brain counts this client as attached
            self.attached.set()
            for step in self.steps:
                while True:
                    line = stream.readline()
                    if not line:
                        raise RuntimeError(f"the stream ended before a {step['on']!r} event")
                    if json.loads(line).get("event") == step["on"]:
                        break
                status, payload = console.request(self.root, task_id, "POST", f"/{step['op']}", step.get("body", {}))
                if status != 200:
                    raise RuntimeError(f"{step['op']} refused ({status}): {payload.get('error', payload)}")
                self.sent.append(payload["event"])
        finally:
            conn.close()


class _Gated:
    """A scenario's runner while its scripted client drives it: the first call waits until the client is attached, so
    the client sees every event and the first hold waits for it instead of parking."""

    def __init__(self, inner: Any, client: ScriptedClient) -> None:
        self.inner, self.client = inner, client

    def __call__(self, argv: list[str], stdin: str | None, cwd: Path, timeout: int) -> RunResult:
        if not self.client.attached.wait(self.client.timeout) or self.client.error is not None:
            raise RuntimeError(f"the scripted client is not attached: {self.client.error or 'timed out'}")
        return self.inner(argv, stdin, cwd, timeout)


def run_scenario(scenario: str, root: Root, workspace: Path, runner: Any, echo: loop.Echo = print, *,
                 mode: str = loop.AUTO) -> tuple[Task, ScriptedClient | None]:
    """One scenario as a session on `root` and `workspace`, with nobody attached but the scenario's own scripted client
    when its `client.jsonl` names one (then `runner` is gated on its attach): a hold nobody releases parks at once, and
    nothing is shown on the desktop. Returns the session and the client, whose `error` is raised once the run ends."""
    steps = read_jsonl(FIXTURES / scenario / "client.jsonl")
    client = ScriptedClient(root, steps) if steps else None
    if client is not None:
        client.start()
        runner = _Gated(runner, client)
    try:
        task = loop.run(f"dry scenario: {scenario}", workspace, root, load_caps(root), runner, echo=echo,
                        unattended=0.0, notify=None, mode=mode)
    finally:
        if client is not None:
            client.finished.set()
            client.join(client.timeout)
    if client is not None and client.error is not None:
        raise RuntimeError(f"the scripted client of {scenario!r} failed: {client.error}")
    return task, client


def scenario_runner(scenario: str) -> ScriptedRunner:
    """The scripted runner for `fixtures/protean/<scenario>/`."""
    folder = FIXTURES / scenario
    if not (folder / "director.jsonl").is_file():
        known = sorted(p.name for p in FIXTURES.iterdir() if (p / "director.jsonl").is_file())
        raise ValueError(f"unknown scenario {scenario!r} (known: {', '.join(known)})")
    return ScriptedRunner(read_jsonl(folder / "director.jsonl"), read_jsonl(folder / "worker.jsonl"),
                          read_jsonl(folder / "reader.jsonl"))


def make_workspace(path: Path) -> Path:
    """A one-commit git repository to run a scenario against."""
    path.mkdir(parents=True)
    (path / "README.md").write_text("# dry workspace\n", encoding="utf-8")
    _git(["init", "--quiet", "-b", "main"], path)
    _git(["add", "-A"], path)
    _git(["commit", "--quiet", "-m", "dry: seed"], path)
    return path


class Serving:
    """A scripted runner wrapped for `--serve`: a pause before every model call so a person can press between ticks,
    and the editor's lines streamed into the transcript (`streams_lines`, as the real runner does): an init line, the
    editor's text, then the receipt line itself. The scripted answers are untouched."""

    streams_lines = True

    def __init__(self, inner: ScriptedRunner, delay: float) -> None:
        self.inner, self.delay = inner, delay

    def __call__(self, argv: list[str], stdin: str | None, cwd: Path, timeout: int,
                 on_line: Any = None) -> RunResult:
        if self.delay:
            time.sleep(self.delay)
        res = self.inner(argv, stdin, cwd, timeout)
        if on_line is not None and ScriptedRunner.seat(argv) == "worker" and res.stdout:
            receipt = json.loads(res.stdout)
            on_line(json.dumps({"type": "system", "subtype": "init", "session_id": receipt.get("session_id")}))
            on_line(json.dumps({"type": "assistant", "message": {"role": "assistant", "content": [
                {"type": "text", "text": receipt.get("result", "")}]}}))
            on_line(res.stdout)
        return res


def dry(scenario: str, root_template: Root, echo: loop.Echo = print, *, serve: bool = False, mode: str = loop.AUTO,
        delay: float = 0.0, ticks: int | None = None, unattended: float | None = None) -> Task:
    """Run a scenario on a scratch root and workspace, both removed afterwards.

    Without `serve`, nobody attaches but the scenario's own scripted client, when it has one (`run_scenario`). `serve`
    is the fixture brain a console client develops against: the session starts paused and serves its console on the
    scratch root (printed, with the attach line) until a client presses Resume; holds wait on the console with the
    configured unattended timeout (`unattended` overrides it) instead of parking at once; the scenario's scripted client
    never runs (the attached client is the driver); scripted calls are paced by `delay` seconds; `ticks` caps the
    scenario at that many ticks so `extend` can be exercised. The real brain root's `state/` is never touched: only the
    seed files are copied out of it."""
    runner = scenario_runner(scenario)
    scratch = Path(tempfile.mkdtemp(prefix="protean-dry-"))
    try:
        root = Root(scratch / "brain")
        root.path.mkdir()
        for name in SEED_FILES:
            shutil.copy2(root_template.path / name, root.path / name)
        if ticks is not None:
            config = json.loads(root.config_file.read_text(encoding="utf-8"))
            config["max_ticks"] = ticks
            root.config_file.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
        caps = load_caps(root)
        workspace = make_workspace(scratch / "workspace")
        if not serve:
            driven = (FIXTURES / scenario / "client.jsonl").is_file()
            echo(f"dry {scenario}: scripted seats on a scratch root" + (", driven by its scripted client" if driven
                                                                         else ""))
            return run_scenario(scenario, root, workspace, runner, echo, mode=mode)[0]
        if unattended is None:
            unattended = console.UNATTENDED_SECONDS if caps.unattended_seconds is None else caps.unattended_seconds
        echo(f"dry {scenario}: scripted seats served on {root.path}")
        echo(f"attach: protean --brain {root.path} ctl watch  (the task starts paused; `ctl resume` runs it)")
        return loop.run(f"dry scenario: {scenario}", workspace, root, caps, Serving(runner, delay), echo=echo,
                        unattended=unattended, notify=console.notify, mode=mode, paused=True)
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
