"""
Publishing approved posts to the org's social media accounts.

Owns: the connected platform accounts (platform_accounts: the account's id and name
and the permissions it was granted, never an access token), the publishers
(PUBLISHERS: one per platform, with the permissions each needs) and the record
of what went out (publications), which makes publishing a post to a platform
happen at most once, also from sessions running at once. The publishers are
dummies for now: each records a simulated publication and calls no platform; a
live publisher for a platform replaces its entry in PUBLISHERS. Does not own
when a post may be published (tools.py: approval, the user's request, the
platform rules and the guardrails).
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Protocol

from agent import db
from agent.logs import safe

# A publisher's call to its platform gives up after this long, so a turn can't hang on it.
PUBLISH_TIMEOUT_S: float = 30.0
# A claim this old was left by a process that stopped mid-publish, so it may be tried again.
STALE_CLAIM_S: int = 300

logger = logging.getLogger(__name__)


class PublishingError(Exception):
    """An account change can't be made; the message says why in plain words."""


@dataclass(frozen=True)
class PlatformAccount:
    """An org's account on one platform: where its posts go and what it may do there."""

    platform: str
    account_id: str  # the platform's id of the page or profile, e.g. a LinkedIn organization URN
    name: str = ""
    scopes: tuple[str, ...] = ()  # the permissions (OAuth scopes) the org granted Mackie


@dataclass(frozen=True)
class PublishRequest:
    """What goes out: an approved post's stored text, never text the model wrote for the call."""

    post_id: str
    content: str
    visual: str = ""  # the photo or graphic idea; a live Instagram publisher needs the media itself


@dataclass(frozen=True)
class Receipt:
    """What a platform said about a published post."""

    external_id: str
    url: str = ""
    simulated: bool = False


@dataclass(frozen=True)
class Outcome:
    """What happened on one platform, for the tool result."""

    platform: str
    published: bool  # newly published by this call
    message: str


class Publisher(Protocol):
    """Publishes to one platform. A live one reads the account's access token from a secret store."""

    platform: str
    required_scopes: tuple[str, ...]

    async def publish(self, request: PublishRequest, account: PlatformAccount) -> Receipt:
        """Publish the post to the account; raise on failure."""
        ...


@dataclass(frozen=True)
class DummyPublisher:
    """Records a simulated publication and calls no platform, until a live publisher replaces it."""

    platform: str
    required_scopes: tuple[str, ...]

    async def publish(self, request: PublishRequest, account: PlatformAccount) -> Receipt:
        """A receipt with a made-up id that says it was simulated; nothing is sent."""
        external_id = f"simulated-{self.platform}-{account.account_id}-{request.post_id}"
        return Receipt(external_id=external_id, simulated=True)


# The permissions are the ones each platform's API needs to post as a page or organization.
PUBLISHERS: dict[str, Publisher] = {
    publisher.platform: publisher
    for publisher in (
        DummyPublisher("linkedin", ("w_organization_social",)),
        DummyPublisher("facebook", ("pages_manage_posts",)),
        DummyPublisher("instagram", ("instagram_basic", "instagram_content_publish")),
    )
}


async def connect_account(org_id: str, account: PlatformAccount) -> None:
    """
    Save the org's account for a platform, replacing any earlier one.

    Raises:
        PublishingError: the platform has no publisher, the account id is blank or
            has spaces, or the org doesn't exist.
    """
    if account.platform not in PUBLISHERS:
        raise PublishingError(
            f"Mackie can't publish to {account.platform!r}; it publishes to "
            f"{', '.join(PUBLISHERS)}."
        )
    if not account.account_id or any(char.isspace() for char in account.account_id):
        raise PublishingError("The account id must be the platform's id, without spaces.")
    async with db.org_transaction(org_id) as conn:
        if not await (
            await conn.execute("SELECT 1 FROM orgs WHERE org_id = %s", (org_id,))
        ).fetchone():
            raise PublishingError(f"There is no organization {org_id}.")
        await conn.execute(
            "INSERT INTO platform_accounts (org_id, platform, account_id, account_name, scopes) "
            "VALUES (%s, %s, %s, %s, %s) ON CONFLICT (org_id, platform) DO UPDATE SET "
            "account_id = EXCLUDED.account_id, account_name = EXCLUDED.account_name, "
            "scopes = EXCLUDED.scopes, connected_at = now()",
            (org_id, account.platform, account.account_id, account.name, list(account.scopes)),
        )
    logger.info("Connected %s's %s account %s", org_id, account.platform, safe(account.account_id))


async def disconnect_account(org_id: str, platform: str) -> bool:
    """Remove the org's account for a platform; False if it had none."""
    async with db.org_transaction(org_id) as conn:
        cursor = await conn.execute(
            "DELETE FROM platform_accounts WHERE org_id = %s AND platform = %s", (org_id, platform)
        )
    if cursor.rowcount:
        logger.info("Disconnected %s's %s account", org_id, safe(platform))
    return bool(cursor.rowcount)


async def get_accounts(org_id: str) -> dict[str, PlatformAccount]:
    """The org's connected accounts, by platform."""
    async with db.org_transaction(org_id) as conn:
        rows = await (
            await conn.execute(
                "SELECT platform, account_id, account_name, scopes FROM platform_accounts "
                "WHERE org_id = %s ORDER BY platform",
                (org_id,),
            )
        ).fetchall()
    return {
        row["platform"]: PlatformAccount(
            row["platform"], row["account_id"], row["account_name"], tuple(row["scopes"])
        )
        for row in rows
    }


def missing_scopes(account: PlatformAccount) -> list[str]:
    """The permissions the platform's publisher needs that the account wasn't granted."""
    return [
        scope
        for scope in PUBLISHERS[account.platform].required_scopes
        if scope not in account.scopes
    ]


async def published_platforms(org_id: str, post_id: str) -> list[str]:
    """The platforms Mackie published the post to."""
    async with db.org_transaction(org_id) as conn:
        rows = await (
            await conn.execute(
                "SELECT platform FROM publications WHERE org_id = %s AND post_id = %s "
                "AND state = 'published' ORDER BY platform",
                (org_id, post_id),
            )
        ).fetchall()
    return [row["platform"] for row in rows]


async def _claim(org_id: str, post_id: str, account: PlatformAccount) -> dict | None:
    """
    Claim publishing the post to the account's platform: None if claimed, else the row in the way.

    One statement, so of two sessions claiming at once only one gets the row. A failed
    attempt, or a claim left by a process that stopped mid-publish, can be claimed again.
    """
    async with db.org_transaction(org_id) as conn:
        claimed = await (
            await conn.execute(
                "INSERT INTO publications (org_id, post_id, platform, state, account_id) "
                "VALUES (%s, %s, %s, 'publishing', %s) "
                "ON CONFLICT (org_id, post_id, platform) DO UPDATE SET state = 'publishing', "
                "account_id = EXCLUDED.account_id, error = '', updated_at = now() "
                "WHERE publications.state = 'failed' OR (publications.state = 'publishing' "
                "AND publications.updated_at < now() - make_interval(secs => %s)) "
                "RETURNING state",
                (org_id, post_id, account.platform, account.account_id, STALE_CLAIM_S),
            )
        ).fetchone()
        if claimed:
            return None
        return await (
            await conn.execute(
                "SELECT state, external_id, simulated, updated_at FROM publications "
                "WHERE org_id = %s AND post_id = %s AND platform = %s",
                (org_id, post_id, account.platform),
            )
        ).fetchone()


async def _record(
    org_id: str, post_id: str, platform: str, receipt: Receipt | None, error: str = ""
) -> None:
    """Record how a claimed publication ended: published with the receipt, or failed."""
    async with db.org_transaction(org_id) as conn:
        await conn.execute(
            "UPDATE publications SET state = %s, external_id = %s, url = %s, error = %s, "
            "simulated = %s, updated_at = now() WHERE org_id = %s AND post_id = %s "
            "AND platform = %s",
            (
                "published" if receipt else "failed",
                receipt.external_id if receipt else "",
                receipt.url if receipt else "",
                error,
                bool(receipt and receipt.simulated),
                org_id,
                post_id,
                platform,
            ),
        )


async def _publish_one(org_id: str, request: PublishRequest, account: PlatformAccount) -> Outcome:
    """Claim, publish and record one platform; never raises for the publisher's failure."""
    platform = account.platform
    if existing := await _claim(org_id, request.post_id, account):
        if existing["state"] == "published":
            when = existing["updated_at"].date().isoformat()
            return Outcome(
                platform, False, f"already published on {when} ({existing['external_id']})"
            )
        return Outcome(platform, False, "another session is publishing it right now")
    try:
        receipt = await asyncio.wait_for(
            PUBLISHERS[platform].publish(request, account), PUBLISH_TIMEOUT_S
        )
    except Exception as e:  # the publisher's failure is reported, not raised (tools never raise)
        error = type(e).__name__
        logger.warning(
            "Publishing %s for %s to %s failed: %s", request.post_id, org_id, platform, error
        )
        await _record(org_id, request.post_id, platform, None, error)
        return Outcome(
            platform,
            False,
            f"failed ({error}); nothing is recorded as published. Check the account on the "
            "platform before trying again",
        )
    await _record(org_id, request.post_id, platform, receipt)
    logger.info(
        "Published %s for %s to %s%s",
        request.post_id,
        org_id,
        platform,
        " (simulated)" if receipt.simulated else "",
    )
    where = f"; {receipt.url}" if receipt.url else ""
    if receipt.simulated:
        return Outcome(
            platform,
            True,
            f"published (simulated: publishing is a dummy for now, so nothing was sent; "
            f"id {receipt.external_id})",
        )
    return Outcome(platform, True, f"published (id {receipt.external_id}{where})")


async def publish(
    org_id: str, request: PublishRequest, accounts: list[PlatformAccount]
) -> list[Outcome]:
    """
    Publish the post to each account's platform at once; one Outcome per account, in order.

    A platform the post already went out on, or one another session is publishing to,
    is skipped; one platform failing doesn't stop the others.
    """
    return list(
        await asyncio.gather(*(_publish_one(org_id, request, account) for account in accounts))
    )
