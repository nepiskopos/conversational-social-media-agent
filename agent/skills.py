"""
Loading and validation of the non-engineer-editable files in skills/.

Owns: reading skills/*.yaml into platform, content-type and format skills,
reading the content pillars (skills/content_pillars.yaml or a per-org override
in the profile), reading the content guardrails (skills/guardrails.yaml: rules
for the prompt, and blocked terms with Guardrails.blocked_term to find them),
and reporting problems in those files as plain-English messages.
A broken file is skipped and reported, never raised, so one typo by a
non-engineer cannot take every conversation down. Does not own how skills are
rendered into the prompt (assistant.py) or used by tools (tools.py).
"""

from __future__ import annotations

import copy
import functools
import logging
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

SKILLS_DIR: Path = Path(__file__).parent.parent / "skills"
# The skill files are read on every prompt build and tool call; PyYAML's C parser (in its
# wheels) reads them about 7x faster than the pure-Python one, with the same result.
_YAML_LOADER: type = getattr(yaml, "CSafeLoader", yaml.SafeLoader)
PILLARS_FILE: str = "content_pillars.yaml"
GUARDRAILS_FILE: str = "guardrails.yaml"
# The two lists guardrails.yaml may have; each is a list of '- ' text lines.
GUARDRAIL_LISTS: tuple[str, ...] = ("rules", "blocked_terms")

# Fields each kind of skill must define; everything else is optional.
REQUIRED_FIELDS: dict[str, tuple[str, ...]] = {
    "platform": ("id", "name", "max_length"),
    "content_type": ("id", "name", "purpose", "structure"),
    "format": ("id", "name", "purpose", "structure"),
}
# Fields that must be text, lists of text, or true/false, when present. The prompt builder
# and the post checks rely on these types; "format_hints:" followed by "- " lines is a list.
TEXT_FIELDS: tuple[str, ...] = (
    "id",
    "name",
    "tone",
    "hashtag_style",
    "format_hints",
    "purpose",
    "structure",
)
LIST_FIELDS: tuple[str, ...] = ("must_include", "avoid", "platforms")
BOOL_FIELDS: tuple[str, ...] = ("has_caption", "needs_visual")
# Ids, and the platform ids a format lists, are compared with what the model sends, lowercased.
ID_PATTERN: re.Pattern[str] = re.compile(r"[a-z0-9_]+")

logger = logging.getLogger(__name__)
# Skill-file problems already logged by this process.
_logged_problems: set[str] = set()


def _load_yaml(text: str) -> Any:
    """yaml.safe_load, through the C parser when PyYAML has it."""
    return yaml.load(text, Loader=_YAML_LOADER)


def _read_yaml(path: Path) -> tuple[Any, str]:
    """
    A YAML file's contents and "", or None and what is wrong with the file.

    Besides bad YAML, a file saved in another encoding than UTF-8 (Windows
    editors do) or a folder named *.yaml is a problem to report, not a crash.
    The files are read on every prompt build and tool call, so each text is
    parsed once (_parse_yaml); keyed by the text, not the file's timestamps,
    an edit is read on the next load however soon it follows the last one.
    Callers get a copy, so changing it can't change the cache.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return None, f"{path.name}: not saved as UTF-8 text; save it again as UTF-8"
    except OSError as e:
        return None, f"{path.name}: can't be read ({e.strerror})"
    data, problem = _parse_yaml(text)
    if problem:
        return None, f"{path.name}: {problem}"
    return copy.deepcopy(data), ""


@functools.lru_cache(maxsize=256)
def _parse_yaml(text: str) -> tuple[Any, str]:
    """_read_yaml's parse of one file's text: the data and "", or None and the problem."""
    try:
        return _load_yaml(text), ""
    except yaml.YAMLError as e:
        return None, f"not valid YAML ({e})"


@dataclass
class SkillSet:
    """Valid skills found in skills/, split by kind, plus any problems found."""

    platforms: list[dict[str, Any]] = field(default_factory=list)
    content_types: list[dict[str, Any]] = field(default_factory=list)
    formats: list[dict[str, Any]] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)


def _skill_problem(skill: Any, file_name: str) -> str | None:
    """Return a plain-English problem with one parsed skill file, or None if it is valid."""
    if not isinstance(skill, dict):
        return f"{file_name}: expected 'field: value' lines, found something else"
    kind = skill.get("kind", "platform")
    if not isinstance(kind, str) or kind not in REQUIRED_FIELDS:
        return f"{file_name}: kind must be one of {', '.join(REQUIRED_FIELDS)} (found {kind!r})"
    missing = [name for name in REQUIRED_FIELDS[kind] if not skill.get(name)]
    if missing:
        return f"{file_name}: missing required field(s) {', '.join(missing)}"
    max_length = skill.get("max_length")
    # bool is a kind of int in Python, but "max_length: true" is no length.
    if kind == "platform" and (isinstance(max_length, bool) or not isinstance(max_length, int)):
        return f"{file_name}: max_length must be a whole number (found {skill['max_length']!r})"
    for name in TEXT_FIELDS:
        if name in skill and not isinstance(skill[name], str):
            return f"{file_name}: {name} must be text; for several lines, write '{name}: |'"
    for name in LIST_FIELDS:
        value = skill.get(name, [])
        if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
            return f"{file_name}: {name} must be a list of '- ' lines"
    if "platforms" in skill and not skill["platforms"]:
        return f"{file_name}: list the platforms under platforms, or leave it out for all"
    for value in [skill["id"], *skill.get("platforms", [])]:
        if not ID_PATTERN.fullmatch(value):
            return (
                f"{file_name}: {value!r} must be written in lowercase letters, digits and "
                "underscores only (e.g. feed_post)"
            )
    for name in BOOL_FIELDS:
        if name in skill and not isinstance(skill[name], bool):
            return f"{file_name}: {name} must be true or false"
    return None


def _pillars_problem() -> str | None:
    """
    Return a plain-English problem with skills/content_pillars.yaml, or None.

    A missing file is not a problem (the org simply has no default pillars).
    load_content_pillars copes with a broken one; this says what to fix.
    """
    path = SKILLS_DIR / PILLARS_FILE
    if not path.exists():
        return None
    data, problem = _read_yaml(path)
    if problem:
        return problem
    pillars = data.get("pillars") if isinstance(data, dict) else None
    if not isinstance(pillars, list):
        return f"{PILLARS_FILE}: expected a 'pillars:' list of '- name: ...' entries"
    if not all(isinstance(pillar, dict) and pillar.get("name") for pillar in pillars):
        return f"{PILLARS_FILE}: every pillar needs a name (entries without one are skipped)"
    return None


@dataclass(frozen=True)
class Guardrails:
    """The org-wide content guardrails from skills/guardrails.yaml."""

    rules: tuple[str, ...] = ()  # plain-English rules every post follows (the prompt)
    blocked_terms: tuple[str, ...] = ()  # words and phrases no post may contain (checked in code)

    def blocked_term(self, text: str) -> str | None:
        """
        The first blocked term in text, as written in the file, or None.

        Matched as whole words, ignoring case, spacing, accents and look-alike
        forms (NFKC: fullwidth letters, ligatures); a term ending in * also
        matches longer words that start with it.
        """
        folded = _folded(text)
        return next(
            (term for term in self.blocked_terms if _term_pattern(term).search(folded)), None
        )


def _folded(text: str) -> str:
    """
    Text as blocked terms are compared: NFKC-normalized, without combining marks, case-folded.

    Without the marks, an accent ("dámn") or a mark drawn over or under a
    letter, which would otherwise split the word in two, hides no term.
    """
    decomposed = unicodedata.normalize("NFKD", text)
    unmarked = "".join(char for char in decomposed if not unicodedata.combining(char))
    return unicodedata.normalize("NFKC", unmarked).casefold()


@functools.lru_cache(maxsize=1024)
def _term_pattern(term: str) -> re.Pattern[str]:
    """The pattern for one blocked term: whole words, any spacing, a trailing * for any ending."""
    stem, any_ending = (term[:-1], True) if term.endswith("*") else (term, False)
    words = r"\s+".join(re.escape(word) for word in _folded(stem).split())
    return re.compile(rf"(?<!\w){words}{r'\w*' if any_ending else ''}(?!\w)")


def _guardrail_entries(data: Any) -> tuple[dict[str, tuple[str, ...]], str | None]:
    """
    The usable rules and blocked terms in parsed guardrails.yaml, and its first problem or None.

    A wrong entry (not text, or blank) is skipped and reported, and the rest
    still apply: one typo must not switch every guardrail off.
    """
    if not isinstance(data, dict):
        return {name: () for name in GUARDRAIL_LISTS}, (
            f"{GUARDRAILS_FILE}: expected 'rules:' and 'blocked_terms:' lists of '- ' lines"
        )
    entries: dict[str, tuple[str, ...]] = {}
    problem = None
    for name in GUARDRAIL_LISTS:
        value = data.get(name) or []
        items = value if isinstance(value, list) else []
        # A blank entry, or a lone "*" (which would block every word), is no entry.
        entries[name] = tuple(
            item.strip() for item in items if isinstance(item, str) and item.strip("* \t\n")
        )
        all_usable = isinstance(value, list) and len(entries[name]) == len(value)
        if problem is None and not all_usable:
            problem = (
                f"{GUARDRAILS_FILE}: {name} must be a list of '- ' lines of text "
                "(entries that aren't are skipped)"
            )
    return entries, problem


def _read_guardrails() -> tuple[Guardrails, str | None]:
    """The guardrails in skills/guardrails.yaml and the file's problem, if any; none if it's missing."""
    path = SKILLS_DIR / GUARDRAILS_FILE
    if not path.exists():
        return Guardrails(), None
    data, problem = _read_yaml(path)
    if problem:
        return Guardrails(), problem
    entries, problem = _guardrail_entries(data)
    return Guardrails(**entries), problem


def load_guardrails() -> Guardrails:
    """
    Load the content guardrails from skills/guardrails.yaml.

    A missing file means no guardrails. A broken one yields whatever entries
    are usable (none for a file that isn't valid YAML); load_skills reports
    and logs what is wrong with it.
    """
    return _read_guardrails()[0]


def load_skills() -> SkillSet:
    """
    Load every skill file in skills/, and check the content pillars file.

    A skill's `kind` defaults to "platform" so existing platform files need no
    change. Invalid files and duplicate ids are skipped, and they and any
    pillars problem are reported in SkillSet.problems (and logged), not raised.

    Returns:
        SkillSet with platforms, content_types and formats sorted by file name.
    """
    skills = SkillSet()
    seen_ids: set[str] = set()
    skills_of_kind = {
        "platform": skills.platforms,
        "content_type": skills.content_types,
        "format": skills.formats,
    }
    for path in sorted(SKILLS_DIR.glob("*.yaml")):
        if path.name in (PILLARS_FILE, GUARDRAILS_FILE):
            continue
        skill, problem = _read_yaml(path)
        if problem:
            skills.problems.append(problem)
            continue
        problem = _skill_problem(skill, path.name)
        if problem is None and skill["id"] in seen_ids:
            problem = f"{path.name}: id {skill['id']!r} is already used by another skill file"
        if problem:
            skills.problems.append(problem)
            continue
        seen_ids.add(skill["id"])
        skills_of_kind[skill.get("kind", "platform")].append(skill)
    for problem in (_pillars_problem(), _read_guardrails()[1]):
        if problem:
            skills.problems.append(problem)
    for problem in skills.problems:
        # Skills load on every model call: each problem is logged once, not per call.
        if problem not in _logged_problems:
            _logged_problems.add(problem)
            logger.warning("Skipped skill file — %s", problem)
    return skills


def load_content_pillars(profile: dict[str, Any]) -> list[dict[str, Any]]:
    """
    Return the content pillars for an org.

    A per-org "content_pillars" list in the profile replaces the defaults in
    skills/content_pillars.yaml. Entries without a name are dropped; a
    missing, unreadable or malformed pillars file yields no pillars rather
    than an error (load_skills reports what is wrong with it).

    Args:
        profile: Already-loaded org profile dict.

    Returns:
        List of {"name", "description"} dicts (possibly empty).
    """
    pillars = profile.get("content_pillars")
    if not pillars:
        path = SKILLS_DIR / PILLARS_FILE
        data = _read_yaml(path)[0] if path.exists() else None
        pillars = data.get("pillars") if isinstance(data, dict) else None
    if not isinstance(pillars, list):
        return []
    return [
        {"name": p["name"], "description": p.get("description", "")}
        for p in pillars
        if isinstance(p, dict) and p.get("name")
    ]


def hashtag_range(skill: dict[str, Any]) -> tuple[int, int] | None:
    """
    Parse a platform's hashtag_count ("3-5", "2", 0) into (min, max).

    Returns None when the field is missing or not understood, so hashtag
    checks are skipped rather than wrong.
    """
    value = skill.get("hashtag_count")
    if isinstance(value, bool):
        return None
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
