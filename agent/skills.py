"""
Loading and validation of the non-engineer-editable files in skills/.

Owns: reading skills/*.yaml into platform and content-type skills, reading the
content pillars (skills/content_pillars.yaml or a per-org override in the
profile), and reporting problems in those files as plain-English messages.
A broken file is skipped and reported, never raised, so one typo by a
non-engineer cannot take every conversation down. Does not own how skills are
rendered into the prompt (assistant.py) or used by tools (tools.py).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

SKILLS_DIR: Path = Path(__file__).parent.parent / "skills"
PILLARS_FILE: str = "content_pillars.yaml"

# Fields each kind of skill must define; everything else is optional.
REQUIRED_FIELDS: dict[str, tuple[str, ...]] = {
    "platform": ("id", "name", "max_length"),
    "content_type": ("id", "name", "purpose", "structure"),
}
# Fields that must be text, and fields that must be lists of text, when present. The prompt
# builder relies on these types; "format_hints:" followed by "- " lines is a YAML list.
TEXT_FIELDS: tuple[str, ...] = (
    "id",
    "name",
    "tone",
    "hashtag_style",
    "format_hints",
    "purpose",
    "structure",
)
LIST_FIELDS: tuple[str, ...] = ("must_include", "avoid")

logger = logging.getLogger(__name__)


@dataclass
class SkillSet:
    """Valid skills found in skills/, split by kind, plus any problems found."""

    platforms: list[dict[str, Any]] = field(default_factory=list)
    content_types: list[dict[str, Any]] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)


def _skill_problem(skill: Any, file_name: str) -> str | None:
    """Return a plain-English problem with one parsed skill file, or None if it is valid."""
    if not isinstance(skill, dict):
        return f"{file_name}: expected 'field: value' lines, found something else"
    kind = skill.get("kind", "platform")
    if kind not in REQUIRED_FIELDS:
        return f"{file_name}: kind must be one of {', '.join(REQUIRED_FIELDS)} (found {kind!r})"
    missing = [name for name in REQUIRED_FIELDS[kind] if not skill.get(name)]
    if missing:
        return f"{file_name}: missing required field(s) {', '.join(missing)}"
    if kind == "platform" and not isinstance(skill["max_length"], int):
        return f"{file_name}: max_length must be a whole number (found {skill['max_length']!r})"
    for name in TEXT_FIELDS:
        if name in skill and not isinstance(skill[name], str):
            return f"{file_name}: {name} must be text; for several lines, write '{name}: |'"
    for name in LIST_FIELDS:
        value = skill.get(name, [])
        if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
            return f"{file_name}: {name} must be a list of '- ' lines"
    return None


def load_skills() -> SkillSet:
    """
    Load every skill file in skills/ except the content pillars file.

    A skill's `kind` defaults to "platform" so existing platform files need no
    change. Invalid files and duplicate ids are skipped and reported in
    SkillSet.problems (and logged), not raised.

    Returns:
        SkillSet with platforms and content_types sorted by file name.
    """
    skills = SkillSet()
    seen_ids: set[str] = set()
    for path in sorted(SKILLS_DIR.glob("*.yaml")):
        if path.name == PILLARS_FILE:
            continue
        try:
            skill = yaml.safe_load(path.read_text())
        except yaml.YAMLError as e:
            skills.problems.append(f"{path.name}: not valid YAML ({e})")
            continue
        problem = _skill_problem(skill, path.name)
        if problem is None and skill["id"] in seen_ids:
            problem = f"{path.name}: id {skill['id']!r} is already used by another skill file"
        if problem:
            skills.problems.append(problem)
            continue
        seen_ids.add(skill["id"])
        if skill.get("kind", "platform") == "platform":
            skills.platforms.append(skill)
        else:
            skills.content_types.append(skill)
    for problem in skills.problems:
        logger.warning("Skipped skill file — %s", problem)
    return skills


def load_content_pillars(profile: dict[str, Any]) -> list[dict[str, Any]]:
    """
    Return the content pillars for an org.

    A per-org "content_pillars" list in profile.json replaces the defaults in
    skills/content_pillars.yaml. Entries without a name are dropped; an
    unreadable pillars file yields no pillars rather than an error.

    Args:
        profile: Already-loaded org profile dict.

    Returns:
        List of {"name", "description"} dicts (possibly empty).
    """
    pillars = profile.get("content_pillars")
    if not pillars:
        path = SKILLS_DIR / PILLARS_FILE
        try:
            pillars = (yaml.safe_load(path.read_text()) or {}).get("pillars", [])
        except (OSError, yaml.YAMLError, AttributeError) as e:
            logger.warning("Could not read %s — %s", PILLARS_FILE, e)
            return []
    return [
        {"name": p["name"], "description": p.get("description", "")}
        for p in pillars or []
        if isinstance(p, dict) and p.get("name")
    ]


def hashtag_range(skill: dict[str, Any]) -> tuple[int, int] | None:
    """
    Parse a platform's hashtag_count ("3-5", "2", 0) into (min, max).

    Returns None when the field is missing or not understood, so hashtag
    checks are skipped rather than wrong.
    """
    value = skill.get("hashtag_count")
    if isinstance(value, int):
        return value, value
    if not isinstance(value, str):
        return None
    parts = [p.strip() for p in value.split("-")]
    if len(parts) == 1 and parts[0].isdigit():
        return int(parts[0]), int(parts[0])
    if len(parts) == 2 and all(p.isdigit() for p in parts):
        return int(parts[0]), int(parts[1])
    return None
