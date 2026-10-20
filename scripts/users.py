"""
Manage who can log in to Mackie and which organizations each user can open.

Owns: the admin commands (add, passwd, which also lifts a lockout, remove,
grant, revoke, list) and reading a password without echoing it. Does not own the
account rules (agent/accounts.py).

Usage (from the project root):
    python scripts/users.py add alice@example.org
    python scripts/users.py grant alice@example.org "BRCA Strong"
    python scripts/users.py revoke alice@example.org "BRCA Strong"
    python scripts/users.py passwd alice@example.org    # also lifts a lockout
    python scripts/users.py remove alice@example.org
    python scripts/users.py list
add and passwd ask for the password twice; with --password-stdin they read one line instead.
Exit code is 0 on success, 1 otherwise.
"""

from __future__ import annotations

import argparse
import asyncio
import getpass
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

import psycopg

from agent import accounts, db  # after load_dotenv and the path change
from agent.memory import slugify


def read_password(from_stdin: bool) -> str:
    """
    Read a new password: one line from stdin, or typed twice without echo.

    Raises:
        accounts.AccountError: the two typed passwords differ.
    """
    if from_stdin:
        return sys.stdin.readline().rstrip("\n")
    password = getpass.getpass("New password: ")
    if getpass.getpass("Repeat it: ") != password:
        raise accounts.AccountError("The passwords don't match.")
    return password


def _org_id(org: str) -> str:
    """An org given by name or id, as its id."""
    org_id = slugify(org)
    if not org_id:
        raise accounts.AccountError(f"{org!r} is not an organization name.")
    return org_id


async def run(args: argparse.Namespace) -> str:
    """
    Carry out one command and return what to print.

    Raises:
        accounts.AccountError: the command can't be carried out; the message says why.
    """
    if args.command == "add":
        user = await accounts.create_user(args.username, read_password(args.password_stdin))
        return f"Created {user.username}. Give them access with: grant {user.username} ORG"
    if args.command == "passwd":
        if not await accounts.set_password(args.username, read_password(args.password_stdin)):
            raise accounts.AccountError(f"There is no user {args.username}.")
        return f"Password changed for {args.username}."
    if args.command == "remove":
        if not await accounts.delete_user(args.username):
            raise accounts.AccountError(f"There is no user {args.username}.")
        return f"Removed {args.username}."
    if args.command == "grant":
        await accounts.grant(args.username, _org_id(args.org))
        return f"{args.username} can open {_org_id(args.org)}."
    if args.command == "revoke":
        if not await accounts.revoke(args.username, _org_id(args.org)):
            raise accounts.AccountError(f"{args.username} can't open {_org_id(args.org)} anyway.")
        return f"{args.username} can no longer open {_org_id(args.org)}."
    users = await accounts.list_users()
    lines = [f"{name}: {', '.join(orgs) or '(no organizations)'}" for name, orgs in users]
    return "\n".join(lines) or "No users yet. Create one with: add USERNAME"


def parse_args(argv: list[str]) -> argparse.Namespace:
    """The command line as a Namespace with command, username, org and password_stdin."""
    parser = argparse.ArgumentParser(description="Manage Mackie's users.")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("add", "passwd"):
        command = commands.add_parser(name)
        command.add_argument("username")
        command.add_argument("--password-stdin", action="store_true")
    commands.add_parser("remove").add_argument("username")
    for name in ("grant", "revoke"):
        command = commands.add_parser(name)
        command.add_argument("username")
        command.add_argument("org", help="organization name or id")
    commands.add_parser("list")
    return parser.parse_args(argv)


async def main(argv: list[str]) -> int:
    """Run the command line; print the result or the problem and return the exit code."""
    args = parse_args(argv)
    try:
        print(await run(args))
        return 0
    except (accounts.AccountError, db.DatabaseError, psycopg.Error) as e:
        print(e)
        return 1
    finally:
        await db.close_pool()


if __name__ == "__main__":
    from agent.logs import configure_logging

    configure_logging("INFO")  # account changes are on record
    sys.exit(asyncio.run(main(sys.argv[1:])))
