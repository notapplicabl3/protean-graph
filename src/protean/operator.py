"""The operator: the person a brain runs for, read from one profile file at runtime and never written inline.

`brain/protean/operator.json` (gitignored) names the operator; `operator.example.json` beside it (tracked) is
what a fresh clone runs on. Prompts, runtime strings, docs and tests say "the operator" and read this.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, fields
from pathlib import Path

from protean.state import Root

PROFILE = "operator.json"
EXAMPLE = "operator.example.json"


@dataclass(frozen=True)
class Operator:
    name: str = "the operator"
    preferences: str = ""  # free text the thinker reads: how the operator wants to be written to, what they care about
    workspace: str | None = None  # the default workspace when `protean think` names none

    def as_message(self) -> dict[str, str]:
        """What a seat is told: name and preferences only, never the workspace path."""
        return {"name": self.name, "preferences": self.preferences}


def profile_file(root: Root) -> Path:
    return root.path / PROFILE


def example_file(root: Root) -> Path:
    return root.path / EXAMPLE


def load_operator(root: Root) -> Operator:
    """The operator from `operator.json`, else from `operator.example.json`, else the default. A missing key keeps the
    default; an unknown key, an empty name or a non-string value raises, naming the file."""
    for file in (profile_file(root), example_file(root)):
        try:
            data = json.loads(file.read_text(encoding="utf-8"))
        except FileNotFoundError:
            continue
        return _decode(file, data)
    return Operator()


def _decode(file: Path, data: object) -> Operator:
    if not isinstance(data, dict):
        raise ValueError(f"{file}: expected a JSON object")
    known = {f.name for f in fields(Operator)}
    unknown = sorted(data.keys() - known)
    if unknown:
        raise ValueError(f"{file}: unknown key(s) {', '.join(unknown)}")
    for key, value in data.items():
        if key == "workspace" and value is None:
            continue
        if not isinstance(value, str):
            raise ValueError(f"{file}: {key} must be a string, got {value!r}")
    name = data.get("name", Operator.name)
    if not name.strip():
        raise ValueError(f"{file}: name must be a non-empty string")
    return Operator(**data)
