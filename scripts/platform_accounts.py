"""
Connect an organization's social media accounts, so Mackie can publish its approved posts.

Owns: the admin commands (connect, disconnect, list). Does not own the accounts'
storage or the publishers (agent/publishing.py), or when a post may be published
(agent/tools.py). Publishing is simulated for now; when a platform gets a live
publisher, its access token comes from a secret store, never from this command.

Usage (from the project root):
    python scripts/platform_accounts.py connect "BRCA Strong" linkedin urn:li:organization:123 --name "BRCA Strong"
    python scripts/platform_accounts.py connect "BRCA Strong" facebook 1234567890 --scopes pages_manage_posts
    python scripts/platform_accounts.py disconnect "BRCA Strong" facebook
    python scripts/platform_accounts.py list "BRCA Strong"
connect grants the permissions the platform's publisher needs unless --scopes lists
the ones the org actually granted (comma-separated). Exit code is 0 on success, 1 otherwise.
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

from agent import db, publishing  # after load_settings and the path change
from agent.memory import slugify


def _org_id(org: str) -> str:
    """An org given by name or id, as its id."""
    org_id = slugify(org)
    if not org_id:
        raise publishing.PublishingError(f"{org!r} is not an organization name.")
    return org_id


def _describe(account: publishing.PlatformAccount) -> str:
    """One account as a line: platform, id, name, permissions and any missing ones."""
    name = f" ({account.name})" if account.name else ""
    line = f"{account.platform}: {account.account_id}{name}; permissions: {', '.join(account.scopes) or 'none'}"
    if missing := publishing.missing_scopes(account):
        line += f"; can't publish without {', '.join(missing)}"
    return line


async def run(args: argparse.Namespace) -> str:
    """
    Carry out one command and return what to print.

    Raises:
        publishing.PublishingError: the command can't be carried out; the message says why.
    """
    org_id = _org_id(args.org)
    if args.command == "connect":
        platform = args.platform.lower()
        publisher = publishing.PUBLISHERS.get(platform)
        scopes = (
            tuple(s.strip() for s in args.scopes.split(",") if s.strip())
            if args.scopes is not None
            else publisher.required_scopes
            if publisher
            else ()
        )
        account = publishing.PlatformAccount(platform, args.account_id, args.name, scopes)
        await publishing.connect_account(org_id, account)
        return f"Connected {org_id}'s {_describe(account)}"
    if args.command == "disconnect":
        if not await publishing.disconnect_account(org_id, args.platform.lower()):
            raise publishing.PublishingError(f"{org_id} has no {args.platform} account connected.")
        return f"Disconnected {org_id}'s {args.platform.lower()} account."
    accounts = await publishing.get_accounts(org_id)
    return "\n".join(_describe(account) for account in accounts.values()) or (
        f"{org_id} has no accounts connected. Connect one with: connect ORG PLATFORM ACCOUNT_ID"
    )


def parse_args(argv: list[str]) -> argparse.Namespace:
    """The command line as a Namespace with command, org, platform, account_id, name and scopes."""
    parser = argparse.ArgumentParser(description="Connect an organization's social media accounts.")
    commands = parser.add_subparsers(dest="command", required=True)
    connect = commands.add_parser("connect", help="connect (or replace) an account for a platform")
    connect.add_argument("org", help="organization name or id")
    connect.add_argument("platform", help=", ".join(publishing.PUBLISHERS))
    connect.add_argument("account_id", help="the platform's id of the page or profile")
    connect.add_argument("--name", default="", help="the account's name, for people")
    connect.add_argument(
        "--scopes", help="the permissions granted, comma-separated (default: all that are needed)"
    )
    disconnect = commands.add_parser("disconnect", help="stop publishing to a platform")
    disconnect.add_argument("org", help="organization name or id")
    disconnect.add_argument("platform")
    commands.add_parser("list", help="list an organization's accounts").add_argument(
        "org", help="organization name or id"
    )
    return parser.parse_args(argv)


async def main(argv: list[str]) -> int:
    """Run the command line; print the result or the problem and return the exit code."""
    args = parse_args(argv)
    try:
        print(await run(args))
        return 0
    except (publishing.PublishingError, db.DatabaseError, psycopg.Error) as e:
        print(e)
        return 1
    finally:
        await db.close_pool()


if __name__ == "__main__":
    from agent.logs import configure_logging

    configure_logging("INFO")  # account changes are on record
    sys.exit(asyncio.run(main(sys.argv[1:])))
