"""
Load organization memory from JSON files into the database.

Owns: the import command, for a file edited after scripts/export_profile.py
and for moving the old data/{org}/profile.json files into PostgreSQL. Each
file replaces everything stored for its org, in one transaction; a file with
a mistake is reported in plain English and changes nothing. Conversation
history (data/checkpoints.db) is not moved: it starts fresh, and everything
Mackie learned is in the profiles. Does not own the checks or the format
(agent/memory.py).

Usage (from the project root):
    python scripts/import_profiles.py brca-strong.json
    python scripts/import_profiles.py data/*/profile.json
    python scripts/import_profiles.py --check brca-strong.json   # check only, change nothing
The org is the file's "org_id", or else the name of the folder holding a profile.json.
Exit code is 0 when every file was imported (or is valid, with --check), 1 otherwise.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


from agent.config import load_settings  # noqa: E402 - needs the path set above

load_settings(Path(__file__).parent.parent / ".env")

import psycopg

from agent import db  # after load_settings and the path change
from agent import memory as mem


async def import_files(paths: list[Path], check_only: bool) -> int:
    """Check, and unless check_only import, each file; print one line per file; return the exit code."""
    failures = 0
    try:
        for path in paths:
            try:
                profile = mem.read_profile_file(path)
                org_id = mem.profile_org_id(path, profile)
                if not check_only:
                    await mem.save_profile(org_id, profile)
            except mem.ProfileError as e:
                print(f"Not imported: {e}")
                failures += 1
                continue
            except (psycopg.DataError, psycopg.IntegrityError) as e:
                # The database refused something the checks let through; the file's
                # transaction rolled back, and the next file is still imported.
                print(f"Not imported: {path}: the database refused it ({e})")
                failures += 1
                continue
            posts = len(profile.get("post_ledger", []))
            verb = "Valid" if check_only else "Imported"
            print(f"{verb}: {path} → {org_id} ({posts} posts)")
    except (db.DatabaseError, psycopg.Error) as e:
        print(f"Import stopped: {e}")
        return 1
    finally:
        await db.close_pool()
    return 1 if failures else 0


def main(argv: list[str]) -> int:
    """Run the command line and return the exit code."""
    parser = argparse.ArgumentParser(description="Import organization memory from JSON files.")
    parser.add_argument("files", nargs="+", type=Path)
    parser.add_argument("--check", action="store_true", help="check the files; change nothing")
    args = parser.parse_args(argv)
    return asyncio.run(import_files(args.files, args.check))


if __name__ == "__main__":
    from agent.logs import configure_logging

    configure_logging("WARNING")
    sys.exit(main(sys.argv[1:]))
