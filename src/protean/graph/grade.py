"""The grader: predicate vocabulary, validation before execution, and the verdict.

Only a grade can mark a unit passed. A predicate whose evidence is missing is ungradeable, and an
ungradeable predicate blocks a pass exactly as a failed one does.

The clone a candidate is graded in is the grading clone (`worker.grading_clone`): a fresh clone of the WORKSPACE at the
candidate SHA, never the editor's own clone, whose `.git` the editor wrote (a replace ref there would let a file
predicate read substitute objects while the real commit integrates; an ignored `.venv` file would run inside every
`uv run` check). Check commands still write in the grading clone, so every git command the brain runs in a repository
the editor or a check command wrote carries `GIT_HARDENING` and `git_env()`: no fsmonitor hook, replace objects ignored,
no system config.
"""

from __future__ import annotations

import os
import shutil
import signal
import subprocess
import tempfile
from collections.abc import Collection
from dataclasses import dataclass, field
from pathlib import Path

from protean.state import Predicate, Unit

PREDICATE_KINDS: dict[str, tuple[frozenset[str], frozenset[str]]] = {
    "file_exists": (frozenset({"path"}), frozenset()),
    "file_contains": (frozenset({"path", "text"}), frozenset()),
    "exit_code": (frozenset({"code"}), frozenset()),  # still decodes for old records; the loop requires exit 0 before grading
    "command": (frozenset({"cmd"}), frozenset({"expect_exit"})),
}
INT_ARGS = frozenset({"code", "expect_exit"})
# Kinds that read the tree, so they can be graded again on a later clone; `exit_code` is one worker's evidence.
RECHECKABLE = frozenset({"file_exists", "file_contains", "command"})
# Kinds whose `path` the confinement rule (`confined`) governs, at decode and at grading.
PATH_KINDS = frozenset({"file_exists", "file_contains"})
FILE_BOUND = 262_144
COMMAND_TIMEOUT = 120
KILL_DRAIN_SECONDS = 5
OUTPUT_TAIL = 400
SHIM_DIR = Path(__file__).resolve().parents[1] / "shim"  # the package root's shim; grade.py lives in graph/
SHIM_MESSAGE = "protean: model calls are not allowed from a command predicate"
SANDBOX = "sandbox-exec"
# Where a check command may write besides the clone: tool caches and temp dirs, never the home tree.
SANDBOX_WRITABLE = ("/private/tmp", "/private/var/folders", "/dev")
# Prefixed to every brain-side git command in a repository the editor or a check command wrote (with `git_env()`):
# a `core.fsmonitor` hook planted in its config never runs.
GIT_HARDENING = ("-c", "core.fsmonitor=false")


def _problems(predicate: Predicate) -> list[str]:
    """What is wrong with one predicate's kind and arguments; empty when it is well formed."""
    if not isinstance(predicate.kind, str) or predicate.kind not in PREDICATE_KINDS:
        return [f"unknown kind {predicate.kind!r} (known: {', '.join(PREDICATE_KINDS)})"]
    if not isinstance(predicate.args, dict):
        return ["args must be an object"]
    required, optional = PREDICATE_KINDS[predicate.kind]
    problems = [f"missing required arg {name!r}" for name in sorted(required - predicate.args.keys())]
    for name in sorted(predicate.args):
        value = predicate.args[name]
        if name not in required | optional:
            problems.append(f"unknown arg {name!r}")
        elif name in INT_ARGS:
            if not isinstance(value, int) or isinstance(value, bool):
                problems.append(f"arg {name!r} must be an integer")
        elif not isinstance(value, str) or not value.strip():
            problems.append(f"arg {name!r} must be a non-empty string")
    return problems


def rechecked(predicate: Predicate) -> bool:
    """Whether an accepted unit's predicate is in the re-check set every later candidate is graded against: a kind that
    reads the tree (`RECHECKABLE`), retained, and not yet replaced by children. A delivery-only predicate passed its own
    unit's grade and is done; a parent whose branch was accepted is carried by its children from then on."""
    return predicate.kind in RECHECKABLE and predicate.retain is not False and not predicate.children


def predicate_ids(unit: Unit) -> list[str]:
    """Positional ids for a unit's predicates: p1, p2, ..."""
    return [f"p{index}" for index in range(1, len(unit.predicates) + 1)]


def validate(predicates: list[Predicate]) -> list[str]:
    """One message per problem across a unit's predicates; an empty list means valid."""
    if not predicates:
        return ["a unit needs at least one predicate"]
    messages = []
    for index, predicate in enumerate(predicates, start=1):
        messages += [f"p{index} {predicate.kind}: {problem}" for problem in _problems(predicate)]
    return messages


@dataclass
class Verdict:
    """The grade of one unit. `passed` is true only with no failed and no ungradeable predicate."""

    passed_ids: list[str] = field(default_factory=list)
    failed_ids: list[str] = field(default_factory=list)
    ungradeable_ids: list[str] = field(default_factory=list)
    passed: bool = False
    notes: dict[str, str] = field(default_factory=dict)


def git_env() -> dict[str, str]:
    """The environment for brain-side git: the inherited one, with replace objects ignored (`git replace` in a clone
    never substitutes what the brain reads) and the system config skipped."""
    return {**os.environ, "GIT_NO_REPLACE_OBJECTS": "1", "GIT_CONFIG_NOSYSTEM": "1"}


def _git(clone: Path, *args: str) -> tuple[bytes | None, str]:
    """One hardened git command in the clone: its stdout, or None with git's own message."""
    try:
        proc = subprocess.run(["git", *GIT_HARDENING, *args], cwd=clone, capture_output=True, env=git_env())
    except OSError as exc:
        return None, str(exc)
    if proc.returncode != 0:
        return None, proc.stderr.decode("utf-8", "replace").strip()[-200:]
    return proc.stdout, ""


def confined(tree: Path, rel: str) -> str:
    """Why a file predicate's path `rel` is not confined to `tree`, or "" when it is: the one confinement rule,
    applied by the grader to the clone it grades and by the director's decode to the workspace before dispatch, so the
    two never disagree. The path must not name `.git` in any case, must be relative, must not climb out with `..`, and
    must stay inside `tree` once every symlink on it is resolved."""
    if "\0" in rel:
        return f"path is not a file name (it holds a NUL byte): {rel!r}"
    if any(part.lower() == ".git" for part in Path(rel).parts):
        return f"path is inside .git, not on the branch: {rel}"
    if os.path.isabs(rel):
        return f"absolute path, not relative to the tree: {rel}"
    if os.path.normpath(rel).split(os.sep)[0] == os.pardir:
        return f"path climbs out of the tree: {rel}"
    try:
        top = os.path.realpath(tree)
        real = os.path.realpath(os.path.join(top, rel))
    except (OSError, ValueError) as exc:
        return f"path cannot be resolved ({exc}): {rel}"
    if os.path.commonpath([top, real]) != top:
        return f"path leaves the tree through a symlink: {rel}"
    return ""


def _branch_entry(clone: Path, rel: str) -> tuple[str | None, str]:
    """What `rel` is on the checked-out commit: file, symlink, directory, submodule or missing.

    File predicates grade the branch's tree objects, never the working tree, so nothing a worker or
    a check command left uncommitted, ignored, inside `.git` or inside a nested repository can count.
    None with the reason when the path cannot be looked up (`confined` refuses it, or there is no repo).
    """
    outside = confined(clone, rel)
    if outside:
        return None, outside
    out, err = _git(clone, "ls-tree", "-z", "HEAD", "--", rel)
    if out is None:
        return None, f"cannot read the branch for {rel}: {err}"
    entries = [entry for entry in out.split(b"\0") if entry]
    if not entries:
        ignored, _ = _git(clone, "check-ignore", "-q", "--", rel)
        if ignored is not None:
            return None, f"path is gitignored, not on the branch: {rel}"
        return "missing", ""
    if len(entries) > 1:
        return "directory", ""
    mode, kind = entries[0].split(b"\t", 1)[0].split(b" ")[:2]
    if kind == b"tree":
        return "directory", ""
    if kind == b"commit":
        return "submodule", ""
    return ("symlink" if mode == b"120000" else "file"), ""


def _file_exists(clone: Path, args: dict) -> tuple[bool | None, str]:
    entry, why_not = _branch_entry(clone, args["path"])
    if entry is None:
        return None, why_not
    if entry == "symlink":
        return None, f"symlink on the branch, not a regular file: {args['path']}"
    if entry == "file":
        return True, f"exists on the branch: {args['path']}"
    return False, (f"missing: {args['path']}" if entry == "missing" else f"not a regular file: {args['path']}")


def _file_contains(clone: Path, args: dict) -> tuple[bool | None, str]:
    entry, why_not = _branch_entry(clone, args["path"])
    if entry is None:
        return None, why_not
    if entry == "symlink":
        return None, f"symlink on the branch, not a regular file: {args['path']}"
    if entry != "file":
        return None, f"no regular file at {args['path']}"
    size, err = _git(clone, "cat-file", "-s", f"HEAD:{args['path']}")
    if size is None or int(size) > FILE_BOUND:
        return None, f"{args['path']} is over the {FILE_BOUND}-byte bound" if size else f"unreadable {args['path']}: {err}"
    raw, err = _git(clone, "cat-file", "-p", f"HEAD:{args['path']}")
    if raw is None:
        return None, f"unreadable {args['path']}: {err}"
    try:
        content = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        return None, f"unreadable {args['path']}: {exc}"
    if args["text"] in content:
        return True, f"{args['path']} contains the text"
    return False, f"{args['path']} does not contain {args['text']!r}"


def _exit_code(exit_code: int | None, args: dict) -> tuple[bool | None, str]:
    if exit_code is None:
        return None, "no exit code was reported"
    return exit_code == args["code"], f"exit {exit_code}, expected {args['code']}"


def _command_env() -> dict[str, str]:
    """The inherited environment without `ANTHROPIC_*` or `CLAUDE_*` variables, the refusing shim first on PATH."""
    env = {name: value for name, value in os.environ.items() if not name.startswith(("ANTHROPIC_", "CLAUDE_"))}
    env["PATH"] = f"{SHIM_DIR}{os.pathsep}{os.environ.get('PATH', '')}"
    return env


def _kill_group(proc: subprocess.Popen) -> None:
    """SIGKILL the command's whole process group."""
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except OSError:
        pass


def quoted(path: str) -> str:
    """A path as a sandbox-profile string literal."""
    return '"' + path.replace("\\", "\\\\").replace('"', '\\"') + '"'


def writable_trees(clone: Path) -> list[str]:
    """Where a confined process may write: the clone, uv's caches, the temp dirs and /dev, as realpaths
    (sandbox-exec matches a subpath against the realpath, so a symlink that leaves these trees is denied).
    Shared by a check command's profile and the editor's."""
    home = Path.home()
    writable = [
        os.path.realpath(clone),
        os.path.realpath(os.environ.get("UV_CACHE_DIR") or home / ".cache" / "uv"),
        os.path.realpath(home / ".local" / "share" / "uv"),
        os.path.realpath(tempfile.gettempdir()),
        *SANDBOX_WRITABLE,
    ]
    return list(dict.fromkeys(writable))


def write_clauses(clone: Path) -> str:
    """The `(allow file-write* ...)` body for `writable_trees`."""
    return " ".join(f"(subpath {quoted(path)})" for path in writable_trees(clone))


def read_deny(console_file: Path | None) -> str:
    """The clause that denies reading the task's console file (its token operates the run): shared by the editor's
    profile and a check command's, since code the editor wrote runs under the second. Empty without a path."""
    if console_file is None:
        return ""
    return f"(deny file-read* (literal {quoted(str(console_file.resolve()))}))"


def _sandbox_profile(clone: Path, console_file: Path | None = None) -> str:
    """No network at all; writes only inside the clone, the uv caches and the temp dirs; reading `console_file`
    denied."""
    return (f"(version 1)(allow default)(deny network*)(deny file-write*)(allow file-write* {write_clauses(clone)})"
            + read_deny(console_file))


def _command(clone: Path, args: dict, console_file: Path | None = None) -> tuple[bool | None, str]:
    """Run `cmd` in the clone, sandboxed, in its own group; a timeout or any interruption kills the group.

    The sandbox denies every network operation, every write outside the clone, the tool caches and the
    temp dirs, and reading the task's console file, so a check command can neither call a model, touch the
    rest of the machine, nor carry the console token out. Without the sandbox binary the predicate is
    ungradeable: a command never runs unconfined.
    """
    expect = args.get("expect_exit", 0)
    sandbox = shutil.which(SANDBOX)
    if sandbox is None:
        return None, f"no sandbox: {SANDBOX} is not on this host, so the command did not run"
    argv = [sandbox, "-p", _sandbox_profile(clone, console_file), "/bin/sh", "-c", args["cmd"]]
    try:
        proc = subprocess.Popen(
            argv, cwd=clone, env=_command_env(), start_new_session=True,
            stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
    except OSError as exc:
        return None, f"could not start the command: {exc}"
    try:
        out, err = proc.communicate(timeout=COMMAND_TIMEOUT)
    except subprocess.TimeoutExpired:
        _kill_group(proc)
        try:
            proc.communicate(timeout=KILL_DRAIN_SECONDS)
        except subprocess.TimeoutExpired:
            pass
        return None, f"timed out after {COMMAND_TIMEOUT}s"
    except BaseException:
        _kill_group(proc)
        try:
            proc.wait(timeout=KILL_DRAIN_SECONDS)
        except subprocess.TimeoutExpired:
            pass
        raise
    if SHIM_MESSAGE in err.decode("utf-8", "replace"):
        return None, f"{SHIM_MESSAGE} (exit {proc.returncode})"
    tail = (out + err).decode("utf-8", "replace").strip()[-OUTPUT_TAIL:]
    return proc.returncode == expect, f"exit {proc.returncode}, expected {expect}: {tail}"


def ungradeable(unit: Unit, note: str) -> Verdict:
    """Every predicate ungradeable for one reason, when there is nothing fit to grade."""
    ids = predicate_ids(unit)
    return Verdict(ungradeable_ids=ids, notes={pid: f"{p.kind}: {note}" for pid, p in zip(ids, unit.predicates)})


def grade(unit: Unit, clone: Path, exit_code: int | None, recheck: bool = False, *,
          console_file: Path | None = None, skip: Collection[str] = ()) -> Verdict:
    """Grade every predicate of `unit` against the clone's checked-out branch and the worker's exit code.

    File kinds read the branch's tree objects; `command` runs sandboxed in the working tree, denied reading
    `console_file` (the live task's console file). With `recheck`, only the predicates in the re-check set
    (`rechecked`) are graded, under their own ids. A pid in `skip` is not graded at all (a parent excused on the
    candidate of the unit whose children inherit it) and appears in no list.
    """
    verdict = Verdict()
    for pid, predicate in zip(predicate_ids(unit), unit.predicates):
        if (recheck and not rechecked(predicate)) or pid in skip:
            continue
        problems = _problems(predicate)
        if problems:
            result, note = None, "invalid predicate: " + "; ".join(problems)
        elif predicate.kind == "file_exists":
            result, note = _file_exists(clone, predicate.args)
        elif predicate.kind == "file_contains":
            result, note = _file_contains(clone, predicate.args)
        elif predicate.kind == "exit_code":
            result, note = _exit_code(exit_code, predicate.args)
        else:
            result, note = _command(clone, predicate.args, console_file)
        if result is None:
            verdict.ungradeable_ids.append(pid)
        elif result:
            verdict.passed_ids.append(pid)
        else:
            verdict.failed_ids.append(pid)
        verdict.notes[pid] = f"{predicate.kind}: {note}"
    verdict.passed = bool(unit.predicates) and not verdict.failed_ids and not verdict.ungradeable_ids
    return verdict
