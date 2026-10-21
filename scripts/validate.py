"""
Check the files a non-engineer edits: skills/*.yaml and exported org profiles.

Owns: a plain-English report of every problem that would make Mackie skip a
skill file, or refuse to import a profile file given on the command line.
Changes nothing. Does not own the loading logic itself (agent/skills.py,
agent/memory.py).

Usage (from the project root):
    python scripts/validate.py                    # skill files
    python scripts/validate.py brca-strong.json   # skill files and an edited export
Exit code is 0 when everything is valid, 1 otherwise.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from agent import memory as mem  # needs the path set above
from agent import skills


def profile_problems(paths: list[Path]) -> list[str]:
    """
    Return one message per profile file that scripts/import_profiles.py would refuse.
    """
    problems: list[str] = []
    for path in paths:
        try:
            mem.profile_org_id(path, mem.read_profile_file(path))
        except mem.ProfileError as e:
            problems.append(str(e))
    return problems


def main(argv: list[str]) -> int:
    """Print a report and return the process exit code."""
    parser = argparse.ArgumentParser(
        description="Check skills/*.yaml, and any exported profile files named, without changing them."
    )
    parser.add_argument("profiles", nargs="*", type=Path, help="exported profile JSON files")
    paths = parser.parse_args(argv).profiles
    # The report below lists every problem; the skills module's log lines would repeat each.
    logging.disable(logging.WARNING)
    try:
        skill_set = skills.load_skills()
        guardrails = skills.load_guardrails()
    finally:
        logging.disable(logging.NOTSET)
    print(f"Platforms:     {', '.join(s['id'] for s in skill_set.platforms) or '(none)'}")
    print(f"Content types: {', '.join(s['id'] for s in skill_set.content_types) or '(none)'}")
    print(f"Formats:       {', '.join(s['id'] for s in skill_set.formats) or '(none)'}")
    print(
        f"Guardrails:    {len(guardrails.rules)} rule(s), "
        f"{len(guardrails.blocked_terms)} blocked term(s)"
    )
    problems = skill_set.problems + profile_problems(paths)
    if not problems:
        print("All files are valid.")
        return 0
    print(f"\n{len(problems)} problem(s):")
    for problem in problems:
        print(f"- {problem}")
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
