"""
Check the files a non-engineer edits: skills/*.yaml and data/*/profile.json.

Owns: a plain-English report of every problem that would make Mackie skip a
skill file or fail to read an org's memory. Changes nothing on disk. Does not
own the loading logic itself (agent/skills.py, agent/memory.py).

Usage (from the project root):
    python scripts/validate.py
Exit code is 0 when everything is valid, 1 otherwise.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from agent import memory as mem  # needs the path set above
from agent import skills


def profile_problems() -> list[str]:
    """Return one message per data/*/profile.json that Mackie cannot read."""
    problems: list[str] = []
    for profile_path in sorted(mem.DATA_DIR.glob("*/profile.json")):
        org_id = profile_path.parent.name
        if not mem.ORG_ID_PATTERN.fullmatch(org_id):
            problems.append(
                f"{profile_path}: folder name {org_id!r} is not an org id "
                "(lowercase letters, digits and dashes only), so Mackie never opens it"
            )
            continue
        try:
            mem.load_profile(org_id)
        except mem.ProfileError as e:
            problems.append(str(e))
    return problems


def main() -> int:
    """Print a report and return the process exit code."""
    skill_set = skills.load_skills()
    print(f"Platforms:     {', '.join(s['id'] for s in skill_set.platforms) or '(none)'}")
    print(f"Content types: {', '.join(s['id'] for s in skill_set.content_types) or '(none)'}")
    print(f"Formats:       {', '.join(s['id'] for s in skill_set.formats) or '(none)'}")
    problems = skill_set.problems + profile_problems()
    if not problems:
        print("All skill files and org profiles are valid.")
        return 0
    print(f"\n{len(problems)} problem(s):")
    for problem in problems:
        print(f"- {problem}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
