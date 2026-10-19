"""The operator: one profile file, read at runtime; the example beside it runs a fresh clone; nothing inline."""

from __future__ import annotations

import json
import re

import pytest

from protean.dry import REPO
from protean.operator import EXAMPLE, PROFILE, Operator, example_file, load_operator, profile_file
from protean.state import Root


def _write(root: Root, name: str, data: object) -> None:
    root.path.mkdir(parents=True, exist_ok=True)
    (root.path / name).write_text(json.dumps(data), encoding="utf-8")


def test_no_file_gives_the_default_operator(root: Root):
    assert load_operator(root) == Operator()
    assert Operator().name == "the operator" and Operator().workspace is None


def test_the_profile_wins_over_the_example(root: Root):
    _write(root, EXAMPLE, {"name": "Example", "preferences": "short sentences"})
    assert load_operator(root) == Operator(name="Example", preferences="short sentences")
    _write(root, PROFILE, {"name": "Real", "workspace": "/tmp/ws"})
    assert load_operator(root) == Operator(name="Real", workspace="/tmp/ws")  # a missing key keeps the default
    assert profile_file(root).name == PROFILE and example_file(root).name == EXAMPLE


@pytest.mark.parametrize("data, text", [
    ([], "expected a JSON object"),
    ({"name": "A", "home": "/x"}, "unknown key(s) home"),
    ({"name": "   "}, "name must be a non-empty string"),
    ({"name": 3}, "name must be a string"),
    ({"preferences": None}, "preferences must be a string"),
])
def test_a_bad_profile_raises_naming_the_file(root: Root, data: object, text: str):
    _write(root, PROFILE, data)
    with pytest.raises(ValueError, match=re.escape(text)) as info:
        load_operator(root)
    assert PROFILE in str(info.value)


def test_a_seat_is_told_the_name_and_preferences_never_the_workspace():
    operator = Operator(name="A", preferences="p", workspace="/secret")
    assert operator.as_message() == {"name": "A", "preferences": "p"}


def test_the_tracked_example_loads_and_names_no_real_person(root: Root):
    example = json.loads(example_file(Root(REPO / "brain" / "protean")).read_text(encoding="utf-8"))
    assert example["name"] == Operator.name and example.get("workspace") is None  # the generic operator, no path
    assert not any(isinstance(v, str) and v.startswith("/") for v in example.values())  # no machine path anywhere
    _write(root, EXAMPLE, example)  # a fresh clone: the example alone, no profile beside it
    assert load_operator(root).workspace is None


# --- the operator reaches the seats through the message, never the prompt ---

def test_the_thinker_message_carries_the_operator(task):
    from protean.graph.budget import Caps
    from protean.graph.director import thinker_message
    assert thinker_message(task, Caps(), [])["operator"] == Operator().as_message()
    given = Operator(name="A", preferences="p", workspace="/w").as_message()
    assert thinker_message(task, Caps(), [], given)["operator"] == {"name": "A", "preferences": "p"}


def test_the_thinker_call_reads_the_profile_beside_the_prompts(tmp_path, root: Root):
    from protean.dry import ScriptedRunner, make_workspace
    from protean import loop
    from protean.graph.budget import Caps
    import shutil
    root.path.mkdir(parents=True, exist_ok=True)
    for prompt in ("thinker.md", "editor.md", "reader.md"):
        shutil.copy(REPO / "brain" / "protean" / prompt, root.path / prompt)
    _write(root, PROFILE, {"name": "A", "preferences": "short", "workspace": "/never-sent"})
    reply = {"type": "result", "subtype": "success", "is_error": False, "session_id": "s", "total_cost_usd": 0.1,
             "result": json.dumps({"rest": True, "state": {"standing": "Resting.", "open": []}})}
    runner = ScriptedRunner([reply], [])
    loop.run("g", str(make_workspace(tmp_path / "ws")), root, Caps(), runner)
    sent = json.loads(runner.calls[0][1])
    assert sent["operator"] == {"name": "A", "preferences": "short"} and "/never-sent" not in runner.calls[0][1]


def test_think_takes_the_workspace_from_the_profile_when_none_is_named(root: Root):
    from protean.cli import build_parser, workspace_for
    assert build_parser().parse_args(["think", "g"]).workspace is None
    assert workspace_for(None, root) is None
    _write(root, PROFILE, {"workspace": "/w"})
    assert workspace_for(None, root) == "/w" and workspace_for("/named", root) == "/named"
