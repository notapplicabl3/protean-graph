"""Shared fixtures for the protean battery: the model-binary and `osascript` shims on PATH, no unattended wait,
a loop clock that stands still unless a test moves it, a scratch brain root, and a check that a killed process is
gone. For speed only: no machine-global git hook in a scratch repository, a 5 ms console poll, and
`dry.make_workspace` as a copy of one workspace the real function made for the run (`_copied_workspace`)."""

from __future__ import annotations

import os
import shutil
import subprocess
import time
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest

from protean import dry, loop
from protean.dry import REPO
from protean.graph import console
from protean.state import Predicate, Root, Task, Unit

SHIM_DIR = Path(__file__).resolve().parent / "shim"
_REAL_MAKE_WORKSPACE = dry.make_workspace
_TEMPLATE: list[Path] = []  # the run's one real workspace, once `workspace_template` made it


def _copied_workspace(path: Path) -> Path:
    """`dry.make_workspace` in the battery: a copy of the run's one real workspace (`workspace_template`), the same
    bytes, commit and refs, with its index refreshed against the copied files so git sees it as it sees a fresh one.
    It saves the `init`, `add` and `commit` of each of a few hundred workspaces; no test needs its own seed commit.
    Before the template exists, the real function runs."""
    if not _TEMPLATE:
        return _REAL_MAKE_WORKSPACE(path)
    shutil.copytree(_TEMPLATE[0], path, symlinks=True)
    subprocess.run(["git", "update-index", "-q", "--refresh"], cwd=path, check=True, capture_output=True)
    return path


dry.make_workspace = _copied_workspace  # set before any test module binds the name (`from protean.dry import ...`)


@pytest.fixture(scope="session", autouse=True)
def workspace_template(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Path]:
    """The run's one workspace from the real `dry.make_workspace`, which every later one copies."""
    _TEMPLATE.append(_REAL_MAKE_WORKSPACE(tmp_path_factory.mktemp("template") / "ws"))
    yield _TEMPLATE[0]
    _TEMPLATE.clear()


@pytest.fixture(autouse=True)
def shim_on_path(monkeypatch: pytest.MonkeyPatch) -> Path:
    """Put the failing `claude` shim and the recording `osascript` shim first on PATH, so no test can reach a real
    model or show a real notification; the CLI's unattended wait is 0, so a hold parks at once."""
    monkeypatch.setenv("PATH", f"{SHIM_DIR}{os.pathsep}{os.environ.get('PATH', '')}")
    monkeypatch.delenv("PROTEAN_OSASCRIPT_LOG", raising=False)
    monkeypatch.setattr(console, "UNATTENDED_SECONDS", 0)
    return SHIM_DIR


@pytest.fixture(scope="session", autouse=True)
def no_machine_git_hooks() -> Iterator[None]:
    """For the whole run (module fixtures too), every git command a test runs, or the code under test runs, sees
    `core.hooksPath` as an empty place: a machine-global hook (a secret scan on each commit) never runs in a scratch
    repository. No test pins a git hook, and the hook cost the battery about 30 ms per commit. Added after any
    `GIT_CONFIG_*` pairs the environment already carries."""
    count = int(os.environ.get("GIT_CONFIG_COUNT") or 0)
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv(f"GIT_CONFIG_KEY_{count}", "core.hooksPath")
        patch.setenv(f"GIT_CONFIG_VALUE_{count}", os.devnull)
        patch.setenv("GIT_CONFIG_COUNT", str(count + 1))
        yield


@pytest.fixture(scope="session", autouse=True)
def fast_console_poll() -> Iterator[None]:
    """For the whole run, the console's and the lobby's poll (`console.POLL_SECONDS`: the server loop, a stream's tail,
    the close's wait for clients) at 5 ms, not 50: each close waits up to one poll for the server thread to stop, and
    the battery closes a console a few hundred times. What a poll sees does not change, only how soon."""
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(console, "POLL_SECONDS", 0.005)
        yield


class Clock:
    """The loop's wall clock as the battery scripts it (`loop.clock`): it reads `now`, which moves only when a test
    moves it, so no session's allowance depends on how fast the machine runs (active time
    accrues dollars). `real` is the clock the loop reads outside the battery."""

    def __init__(self, real: Callable[[], float]) -> None:
        self.now, self.real = 1000.0, real

    def __call__(self) -> float:
        return self.now


@pytest.fixture(autouse=True)
def clock(monkeypatch: pytest.MonkeyPatch) -> Clock:
    """The scripted loop clock, for every test; a test that needs time to pass moves `clock.now`."""
    scripted = Clock(loop.clock)
    monkeypatch.setattr(loop, "clock", scripted)
    return scripted


@pytest.fixture
def root(tmp_path: Path) -> Root:
    """An empty brain root under the test's tmp dir."""
    return Root(tmp_path / "brain")


@pytest.fixture
def reaped() -> Callable[[Path], bool]:
    """Whether the process whose pid a test's shell wrote to a file (`echo $! > kid`) is gone, killed and reaped,
    within 3 s: a direct check that a group kill took it, where a fixed sleep once waited for its side effect."""
    def gone(path: Path) -> bool:
        deadline = time.monotonic() + 3.0
        while not (path.exists() and path.read_text().strip()) and time.monotonic() < deadline:
            time.sleep(0.01)
        pid = int(path.read_text())
        while time.monotonic() < deadline:
            try:
                os.kill(pid, 0)
            except (ProcessLookupError, PermissionError):  # gone (or the pid already names another user's process)
                return True
            time.sleep(0.01)
        return False
    return gone


@pytest.fixture
def task() -> Task:
    """A small task with one pending unit."""
    unit = Unit(id="u1", intent="write a.txt", predicates=[Predicate("file_exists", {"path": "a.txt"})])
    return Task(id="task-20260927T120000", goal="make a.txt", workspace="/tmp/ws", units=[unit])


@pytest.fixture
def live_receipt() -> Callable[[str], Path]:
    """`evidence/<run>/task.json`, a historical record read and never written; the test skips when it is not in this
    tree (the goal-mode receipts are withheld from the public mirror, so its battery skips these rather than failing)."""
    def receipt(run: str) -> Path:
        path = REPO / "evidence" / run / "task.json"
        if not path.exists():
            pytest.skip(f"{run}'s receipt is not in this tree (withheld from the public mirror)")
        return path
    return receipt
