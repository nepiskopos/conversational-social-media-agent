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

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

import psycopg

from agent import db  # after load_dotenv and the path change
from agent import memory as mem


def org_id_for(path: Path, profile: dict) -> str:
    """
    The org a file belongs to: its "org_id", or the folder of an old data/{org}/profile.json.

    Raises:
        mem.ProfileError: neither is an org id.
    """
    org_id = str(profile.get("org_id") or (path.parent.name if path.name == "profile.json" else ""))
    if not db.ORG_ID_PATTERN.fullmatch(org_id):
        raise mem.ProfileError(
            f"{path}: {org_id!r} is not an org id (lowercase letters, digits and dashes); "
            'set "org_id" in the file'
        )
    return org_id


async def import_files(paths: list[Path], check_only: bool) -> int:
    """Check, and unless check_only import, each file; print one line per file; return the exit code."""
    failures = 0
    try:
        for path in paths:
            try:
                profile = mem.read_profile_file(path)
                org_id = org_id_for(path, profile)
                if not check_only:
                    await mem.save_profile(org_id, profile)
            except OSError as e:
                print(f"Not imported: {path}: can't be read ({e.strerror})")
                failures += 1
                continue
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
