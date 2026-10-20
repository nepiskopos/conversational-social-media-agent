"""
Write an organization's memory to a JSON file a person can read and edit.

Owns: the export command. The file holds what Mackie learned (preferences
with the user's words, voice examples, research and brief, any content
pillars of its own) and the post ledger with reported results;
scripts/import_profiles.py loads it back. An org with nothing stored, such as
a misspelled name, is refused. Does not own the format (agent/memory.py).

Usage (from the project root):
    python scripts/export_profile.py "BRCA Strong" > brca-strong.json
    python scripts/export_profile.py brca-strong -o brca-strong.json
Exit code is 0 on success, 1 otherwise.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


from agent.config import load_settings  # noqa: E402 - needs the path set above

load_settings(Path(__file__).parent.parent / ".env")

import psycopg

from agent import db  # after load_settings and the path change
from agent import memory as mem


async def export(org: str, output: Path | None) -> int:
    """Export one org to output (stdout if None); return the exit code."""
    org_id = mem.slugify(org)
    try:
        if not org_id:
            raise ValueError(f"{org!r} is not an organization name")
        profile = await mem.load_profile(org_id)
        # A misspelled name would export an empty profile, which an import turns into a new org.
        if profile == mem.default_profile(org_id):
            raise ValueError(
                f"nothing is stored for {org_id} yet; check the name "
                "(scripts/users.py list shows each user's organizations)"
            )
    except (ValueError, db.DatabaseError, psycopg.Error) as e:
        print(f"Export failed: {e}", file=sys.stderr)
        return 1
    finally:
        await db.close_pool()
    text = json.dumps(profile, indent=2, ensure_ascii=False) + "\n"
    if output is None:
        sys.stdout.write(text)
    else:
        output.write_text(text, encoding="utf-8")  # what read_profile_file reads
        print(f"Wrote {org_id} to {output}.", file=sys.stderr)
    return 0


def main(argv: list[str]) -> int:
    """Run the command line and return the exit code."""
    parser = argparse.ArgumentParser(description="Export an organization's memory to JSON.")
    parser.add_argument("org", help="organization name or id")
    parser.add_argument("-o", "--output", type=Path, help="file to write (default: stdout)")
    args = parser.parse_args(argv)
    return asyncio.run(export(args.org, args.output))


if __name__ == "__main__":
    from agent.logs import configure_logging

    configure_logging("WARNING")
    sys.exit(main(sys.argv[1:]))
