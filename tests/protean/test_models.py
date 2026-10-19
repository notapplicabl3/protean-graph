"""Models: one pinned per seat in `config.json`, carried into that seat's argv; the default is the only name in code."""

from __future__ import annotations

import json

import pytest

from protean.cli import DEFAULT_BRAIN
from protean.graph import director, reader, worker
from protean.graph.budget import Caps, load_caps
from protean.state import Root, Task

SEATS = ("director_model", "worker_model", "reader_model")
PINNED = {"director_model": "made-up-director-1", "worker_model": "made-up-editor-2",
          "reader_model": "made-up-reader-3"}


def _load(root: Root, data: object) -> Caps:
    root.path.mkdir(parents=True, exist_ok=True)
    root.config_file.write_text(json.dumps(data))
    return load_caps(root)


def _models(caps: Caps) -> list[str]:
    return [getattr(caps, seat) for seat in SEATS]


def _argv_models(root: Root, caps: Caps) -> list[list[str]]:
    """The `--model` values in the director's, the editor's and the reader's argv, in that order."""
    task = Task(id="task-20260927T120000", goal="g", workspace="/tmp/ws")
    clone = root.clones_dir(task.id) / "u1"
    argvs = (director.argv(task, caps, root, None), worker.argv(task, caps, root, clone, "sandbox-exec"),
             reader.argv(task, caps, root, clone, "sandbox-exec"))
    return [[argv[i + 1] for i, arg in enumerate(argv) if arg == "--model"] for argv in argvs]


def test_each_seat_s_argv_carries_the_model_its_config_pins(root: Root):
    root.path.mkdir(parents=True)
    for prompt in (root.thinker_prompt, root.editor_prompt, root.reader_prompt):
        prompt.write_text("# prompt\n")
    assert _argv_models(root, load_caps(root)) == [["claude-opus-5"]] * 3
    assert _argv_models(root, _load(root, PINNED)) == [[PINNED[seat]] for seat in SEATS]


def test_a_config_pins_the_seats_models_and_defaults_them_when_it_names_none(root: Root):
    assert _models(load_caps(root)) == ["claude-opus-5"] * 3
    assert _models(_load(root, {"max_usd": 5})) == ["claude-opus-5"] * 3
    caps = _load(root, {"max_usd": 5, **PINNED})
    assert (caps.max_usd, _models(caps)) == (5, [PINNED[seat] for seat in SEATS])
    with pytest.raises(ValueError, match=r"unknown key\(s\) editor_model$"):
        _load(root, {**PINNED, "editor_model": "made-up"})


@pytest.mark.parametrize("seat", SEATS)
@pytest.mark.parametrize("value", ["", "   ", 5, None, True, ["claude-opus-5"]])
def test_a_model_that_is_not_a_non_empty_string_raises_naming_its_key(root: Root, seat: str, value: object):
    with pytest.raises(ValueError, match=rf": {seat} must be a non-empty string, got "):
        _load(root, {seat: value})


def test_the_tracked_config_pins_every_seat():
    data = json.loads((DEFAULT_BRAIN / "config.json").read_text(encoding="utf-8"))
    assert _models(load_caps(Root(DEFAULT_BRAIN))) == [data[seat] for seat in SEATS]
