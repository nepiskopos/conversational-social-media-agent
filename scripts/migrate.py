"""
Create or update Mackie's database schema.

Owns: running agent.db.migrate() as the owner role (MACKIE_MIGRATION_URL),
granting the app's role (the user in MACKIE_DATABASE_URL) only reading and
writing rows, and making sure the default account exists, so there is always
someone who can log in. The default account is MACKIE_DEFAULT_USERNAME
("admin" if unset) with the password MACKIE_DEFAULT_PASSWORD; without one, a
random password is made and printed once. An existing account is never
changed. Safe to run again: it applies only what is missing. Does not own the
migrations themselves (migrations/*.sql, agent/db.py) or the account rules
(agent/accounts.py).

Usage (from the project root):
    python scripts/migrate.py
Exit code is 0 on success, 1 if something is missing or fails.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import secrets
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


from agent.config import load_settings  # noqa: E402 - needs the path set above

load_settings(Path(__file__).parent.parent / ".env")

import psycopg

from agent import accounts, db  # after load_settings and the path change

DEFAULT_USERNAME: str = "admin"
# Bytes of randomness in a generated default password (24 URL-safe characters).
GENERATED_PASSWORD_BYTES: int = 18


async def ensure_default_user() -> str:
    """
    Create the default account if it doesn't exist, and say what happened.

    Raises:
        accounts.AccountError: the configured username or password is invalid.
    """
    username = accounts.normalize_username(
        os.getenv("MACKIE_DEFAULT_USERNAME", "").strip() or DEFAULT_USERNAME
    )
    password = os.getenv("MACKIE_DEFAULT_PASSWORD", "")
    generated = not password
    if generated:
        password = secrets.token_urlsafe(GENERATED_PASSWORD_BYTES)
    try:
        created = await accounts.ensure_user(username, password)
    finally:
        await db.close_pool()
    if not created:
        return f"The default account {username} exists."
    if generated:
        return (
            f"Created the default account {username} with the password {password} "
            f"(shown only now). Change it with: python scripts/users.py passwd {username}"
        )
    return f"Created the default account {username} with the password in MACKIE_DEFAULT_PASSWORD."


def parse_args(argv: list[str]) -> argparse.Namespace:
    """No options: parsing makes --help print usage instead of running the migration."""
    parser = argparse.ArgumentParser(
        description="Create or update Mackie's database schema, and the default account."
    )
    return parser.parse_args(argv)


async def run() -> int:
    """Apply the migrations and print what was done; return the exit code."""
    owner_url = os.getenv("MACKIE_MIGRATION_URL", "").strip()
    if not owner_url:
        print("MACKIE_MIGRATION_URL is not set: the connection string of the schema's owner role.")
        return 1
    try:
        app_role = db.role_of(db.database_url())
        if app_role == db.role_of(owner_url):
            print(
                "MACKIE_DATABASE_URL and MACKIE_MIGRATION_URL use the same role. The app needs its "
                "own role, which can't change the schema (see Configuration in the README)."
            )
            return 1
        applied = await db.migrate(owner_url, app_role)
        print(f"Applied: {', '.join(applied)}" if applied else "The schema is up to date.")
        print(f"Role {app_role} can read and write rows.")
    except (db.DatabaseError, psycopg.Error) as e:
        print(f"Migration failed: {e}")
        return 1
    try:
        print(await ensure_default_user())
    except (db.DatabaseError, psycopg.Error, accounts.AccountError) as e:
        print(f"The schema is ready, but the default account wasn't created: {e}")
        return 1
    return 0


if __name__ == "__main__":
    from agent.logs import configure_logging

    parse_args(sys.argv[1:])
    configure_logging("INFO")  # account changes are on record
    sys.exit(asyncio.run(run()))
