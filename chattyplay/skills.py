from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class Skill:
    name: str
    description: str
    instructions: str
    path: Path


def _parse(path: Path) -> Skill:
    text = path.read_text(encoding="utf-8", errors="replace")
    meta: dict[str, str] = {}
    body = text
    if text.startswith("---\n"):
        end = text.find("\n---\n", 4)
        if end != -1:
            for line in text[4:end].splitlines():
                if ":" in line:
                    key, value = line.split(":", 1)
                    meta[key.strip()] = value.strip().strip("'\"")
            body = text[end + 5 :]
    return Skill(meta.get("name", path.parent.name), meta.get("description", ""), body.strip(), path)


def discover(workspace: Path, dirs: Iterable[str]) -> dict[str, Skill]:
    found: dict[str, Skill] = {}
    workspace = workspace.resolve()
    roots: list[Path] = []
    for directory in dirs:
        root = (workspace / directory).resolve()
        if root.is_relative_to(workspace):
            roots.append(root)
    roots.extend((Path.home() / ".chattyplay" / "skills", Path.home() / ".agents" / "skills"))
    for root in roots:
        if not root.exists():
            continue
        for path in root.glob("*/SKILL.md"):
            try:
                if path.stat().st_size > 500_000:
                    continue
                skill = _parse(path)
                if re.fullmatch(r"[\w.-]+", skill.name):
                    found.setdefault(skill.name, skill)
            except OSError:
                continue
    return found


def enabled_prompt(skills: dict[str, Skill], enabled: Iterable[str]) -> str:
    selected = [skills[name] for name in enabled if name in skills]
    return "\n\n".join(f"# Skill: {skill.name}\n{skill.instructions}" for skill in selected)
