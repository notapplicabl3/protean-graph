"""The editor's live transcript, proven without a model: the real runner launches the
real sandbox around the battery's `claude` shim, which emits stream-json when a test asks for it.

The editor is launched with `--output-format stream-json --verbose`; each stdout line reaches the console stream as a
`transcript` event while the call runs, kept in `transcript.jsonl`, never in `task.json`, the log or a director
message. The receipt is the stream's final `result` event: cost, tokens and session come from it, a capped final
event resolves as a cap always has, and a stream with no final event is charged its cap.
"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import threading
import time
from pathlib import Path

import pytest

from protean import loop, state
from protean.dry import ScriptedRunner, make_workspace
from protean.graph import worker
from protean.graph.budget import Caps
from protean.runner import RunResult, launch, real_runner
from protean.state import Predicate, Root, Task, Unit, load


def _exemplar():
    """The battery's scripted console client lives in `test_console.py`; loaded from there, not copied."""
    spec = importlib.util.spec_from_file_location("protean_console_client", Path(__file__).with_name("test_console.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_c = _exemplar()
Stream, _client, _director_messages, _in_client_thread, _reply = (
    _c.Stream, _c._client, _c._director_messages, _c._in_client_thread, _c._reply)
_rested, _parked_on_a_thought = _c._rested, _c._parked_on_a_thought

U1 = {"id": "u-1", "intent": "write a.txt", "predicates": [{"kind": "file_exists", "args": {"path": "a.txt"}}]}
COMMIT_A = ("SH printf 'A\\n' > a.txt && git add a.txt && "
            "git -c user.name=shim -c user.email=shim@protean.invalid commit -qm shim")
MARK = "MARK-3B-TRANSCRIPT-ONLY"
INIT = {"type": "system", "subtype": "init", "session_id": "shim-editor", "model": "claude-opus-5"}
SAYS = {"type": "assistant", "session_id": "shim-editor",
        "message": {"content": [{"type": "text", "text": f"{MARK}: writing a.txt"}]}}
TOOL = {"type": "user", "session_id": "shim-editor", "message": {"content": [{"type": "tool_result", "content": "ok"}]}}
RESULT = {"type": "result", "subtype": "success", "is_error": False, "session_id": "shim-editor", "num_turns": 3,
          "total_cost_usd": 0.42, "result": "wrote a.txt\nexit_code: 0",
          "usage": {"input_tokens": 11, "output_tokens": 22, "cache_read_input_tokens": 33,
                    "cache_creation_input_tokens": 44}}


@pytest.fixture
def env(tmp_path: Path, root: Root) -> tuple[Root, Path]:
    root.path.mkdir(parents=True)
    root.thinker_prompt.write_text("m\n")
    root.editor_prompt.write_text("e\n")
    return root, make_workspace(tmp_path / "ws")


def _script(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *lines: object) -> list[str]:
    """Hand the shim its stream (dicts are JSON lines, strings are directives); returns the lines it prints."""
    path = tmp_path / "stream.txt"
    text = [json.dumps(line) if isinstance(line, dict) else str(line) for line in lines]
    path.write_text("\n".join(text) + "\n")
    monkeypatch.setenv("PROTEAN_SHIM_STREAM", str(path))
    return [line for line in text if not line.startswith(("WAIT ", "SH "))]


def _editor_through_the_shim(scripted: ScriptedRunner, launches: list[list[str]]):
    """Director calls go to the scripted seat; the editor's go through the real runner: the real sandbox around the
    shim, its stdout read line by line."""
    def runner(argv, stdin, cwd, timeout, on_line=None):
        if "Write" in argv:
            launches.append(list(argv))
            return real_runner(argv, stdin, cwd, timeout, on_line=on_line)
        return scripted(argv, stdin, cwd, timeout)
    runner.streams_lines = True
    return runner


# --- the transcript streams to the console and nowhere else ---


def test_the_editor_streams_its_transcript_to_the_console_during_the_call_and_nowhere_else(
    env: tuple[Root, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    root, workspace = env
    go = tmp_path / "go"
    printed = _script(tmp_path, monkeypatch, INIT, SAYS, f"WAIT {go}", COMMIT_A, TOOL, RESULT)
    scripted, launches = ScriptedRunner([_reply({"task": [U1]}), _reply({"rest": True})], []), []
    attached = threading.Event()
    scripted_first = _editor_through_the_shim(scripted, launches)

    def runner(argv, stdin, cwd, timeout, on_line=None):
        if not scripted.calls:
            assert attached.wait(10), "the client never attached"
        return scripted_first(argv, stdin, cwd, timeout, on_line=on_line)
    runner.streams_lines = True

    def script(seen: dict) -> None:
        stream = Stream(_client(root))
        stream.start()
        try:
            stream.wait_for(lambda line: "snapshot" in line)
            attached.set()
            seen["mark"] = stream.wait_for(lambda line: MARK in line.get("line", ""))
            task = load(root, state.current(root).id)
            seen["while_running"] = ([c.outcome for c in task.calls], go.exists())  # the call is still in flight
            go.touch()
            stream.wait_for(_rested)
        finally:
            seen["lines"] = list(stream.lines)
            stream.close()

    thread, seen = _in_client_thread(script)
    task = loop.run("g", workspace, root, Caps(), runner, unattended=0.2)
    thread.join(15)
    assert "error" not in seen, repr(seen.get("error"))
    assert _parked_on_a_thought(root, task) and [u.status for u in task.units] == ["passed"]

    (argv,) = launches
    assert argv[argv.index("--output-format") + 1:argv.index("--output-format") + 3] == ["stream-json", "--verbose"]
    director_argv = scripted.calls[0][0]
    assert director_argv[director_argv.index("--output-format") + 1] == "json"  # the director keeps json

    assert seen["while_running"] == (["ok", "pending"], False)  # director booked, the editor's call not yet back
    editor_call = next(c for c in task.calls if c.seat == "worker")
    streamed = [line for line in seen["lines"][1:] if line.get("event") == "transcript"]
    assert [line["line"] for line in streamed] == printed  # every stdout line, in order, as it was written
    assert {(line["seat"], line["call"], line["unit"]) for line in streamed} == {("worker", editor_call.id, "u-1")}
    kept = [json.loads(line) for line in root.transcript_file(task.id).read_text().splitlines()]
    assert kept == streamed

    assert MARK not in root.task_file(task.id).read_text() and MARK not in root.log_file(task.id).read_text()
    assert all(MARK not in message for message in _director_messages(scripted))
    assert all(line.get("event") != "transcript" for line in state.read_log(root, task.id))

    assert (editor_call.outcome, editor_call.subtype, editor_call.session_id) == ("ok", "success", "shim-editor")
    assert editor_call.receipt_cost_usd == editor_call.charge_usd == editor_call.actual_usd == 0.42
    assert editor_call.tokens == {"input": 11, "output": 22, "cache_read": 33, "cache_creation": 44}


def test_a_capped_final_event_resolves_as_a_cap_always_has(
    env: tuple[Root, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    root, workspace = env
    capped = {"type": "result", "subtype": "error_max_budget_usd", "is_error": True, "session_id": "shim-capped",
              "total_cost_usd": 4.07, "errors": [], "usage": {"input_tokens": 0, "output_tokens": 0},
              "modelUsage": {"claude-opus-5": {"inputTokens": 900, "outputTokens": 60, "cacheReadInputTokens": 17000,
                                               "cacheCreationInputTokens": 4400, "costUSD": 4.07}}}
    printed = _script(tmp_path, monkeypatch, INIT, SAYS, capped)
    task = Task(id="task-20261010T190000", goal="g", workspace=str(workspace),
                units=[Unit("u-1", "write a.txt", [Predicate("file_exists", {"path": "a.txt"})])])
    res = worker.run_unit(task, task.units[0], Caps(), root, real_runner)
    (call,) = task.calls
    assert (call.outcome, call.subtype, call.cap_usd, call.session_id) == ("error", "error_max_budget_usd", 4.0,
                                                                          "shim-capped")
    assert call.receipt_cost_usd == call.charge_usd == call.actual_usd == res.cost_usd == 4.07
    assert call.tokens == {"input": 900, "output": 60, "cache_read": 17000, "cache_creation": 4400}
    assert res.exit_code is None and res.summary.startswith("ended in error: error_max_budget_usd")
    assert [json.loads(line)["line"] for line in root.transcript_file(task.id).read_text().splitlines()] == printed


def test_a_stream_cut_before_its_final_event_is_charged_its_cap_with_its_spend_unknown(
    env: tuple[Root, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    root, workspace = env
    _script(tmp_path, monkeypatch, INIT, SAYS, {"type": "result_pending", "total_cost_usd": 0.01})
    task = Task(id="task-20261010T191000", goal="g", workspace=str(workspace),
                units=[Unit("u-1", "write a.txt", [Predicate("file_exists", {"path": "a.txt"})])])
    res = worker.run_unit(task, task.units[0], Caps(), root, real_runner)
    (call,) = task.calls
    assert (call.outcome, call.charge_usd, call.receipt_cost_usd, call.actual_usd) == ("no_receipt", 4.0, None, "UNKNOWN")
    assert res.cost_usd == 4.0 and res.summary.startswith("no receipt (exit 0)")


# --- the shim speaks like the real CLI, and stays a failure when nobody asked for a stream ---


def test_the_shim_refuses_stream_json_without_verbose_and_fails_when_no_stream_was_asked_for(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    _script(tmp_path, monkeypatch, INIT)
    refused = subprocess.run(["claude", "-p", "--output-format", "stream-json"], input="", capture_output=True, text=True)
    assert (refused.returncode, refused.stdout) == (1, "")
    assert refused.stderr == "Error: When using --print, --output-format=stream-json requires --verbose\n"
    streamed = subprocess.run(["claude", "-p", "--output-format", "stream-json", "--verbose"], input="x",
                              capture_output=True, text=True)
    assert (streamed.returncode, streamed.stdout) == (0, json.dumps(INIT) + "\n")
    unasked = {name: value for name, value in os.environ.items() if name != "PROTEAN_SHIM_STREAM"}
    for argv, env in ((["claude", "-p", "--output-format", "json"], None),  # the director's format: never a stream
                      (["claude", "-p", "--output-format", "stream-json", "--verbose"], unasked)):
        reached = subprocess.run(argv, input="", capture_output=True, text=True, env=env)
        assert reached.returncode == 89 and "SHIM: the real binary was reached" in reached.stderr


# --- the seam: one runner, a line callback ---


def test_real_runner_hands_over_each_line_while_the_process_runs(tmp_path: Path):
    gate, seen = tmp_path / "gate", []

    def on_line(line: str) -> None:
        seen.append((line, gate.exists()))
        if line == "one":
            gate.touch()  # the process waits for this file before it prints its second line

    res = real_runner(["sh", "-c", f"cat >/dev/null; echo one; while [ ! -e {gate} ]; do sleep 0.02; done; "
                                   "echo two; echo err >&2; exit 3"], "stdin", tmp_path, 10, on_line=on_line)
    assert seen == [("one", False), ("two", True)]
    assert (res.returncode, res.stdout, res.stderr, res.timed_out) == (3, "one\ntwo\n", "err\n", False)


def test_a_streamed_call_still_kills_its_whole_group_on_timeout_and_on_interrupt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, reaped
):
    lines: list[str] = []
    started = time.monotonic()
    res = real_runner(["sh", "-c", "echo first; (sleep 30; touch late) & echo $! > kid1; sleep 30"], None, tmp_path,
                      0.5, on_line=lines.append)
    assert res.timed_out and lines == ["first"] and res.stdout == "first\n" and time.monotonic() - started < 5

    def interrupted(self, *args, **kwargs):  # Ctrl-C lands once the child is running
        deadline = time.monotonic() + 5
        while "ready" not in lines and time.monotonic() < deadline:
            time.sleep(0.01)
        raise KeyboardInterrupt
    held_pipe = real_runner(["sh", "-c", "(sleep 30; touch orphan) & echo $! > kid2; echo gone; exit 0"], None,
                            tmp_path, 0.5, on_line=lines.append)  # the leader exits at once; its child holds stdout
    assert held_pipe.timed_out and lines == ["first", "gone"] and time.monotonic() - started < 5
    with monkeypatch.context() as patch:
        patch.setattr(subprocess.Popen, "wait", interrupted)
        with pytest.raises(KeyboardInterrupt):
            real_runner(["sh", "-c", "(sleep 30; touch later) & echo $! > kid3; echo ready; sleep 30"], None, tmp_path,
                        10, on_line=lines.append)
    assert lines == ["first", "gone", "ready"]
    assert all(reaped(tmp_path / kid) for kid in ("kid1", "kid2", "kid3"))  # every child died with its group
    assert not any((tmp_path / name).exists() for name in ("late", "orphan", "later"))


def test_launch_hands_the_callback_only_to_a_runner_that_streams(tmp_path: Path):
    plain_calls, lines = [], []

    def plain(argv, stdin, cwd, timeout):  # every landed test double has this shape
        plain_calls.append(argv)
        return RunResult(0, "a\nb\n", "")
    assert launch(plain, ["x"], None, tmp_path, 1, lines.append).stdout == "a\nb\n" and lines == []
    assert launch(real_runner, ["sh", "-c", "echo a; echo b"], None, tmp_path, 5, lines.append).stdout == "a\nb\n"
    assert lines == ["a", "b"] and plain_calls == [["x"]]
