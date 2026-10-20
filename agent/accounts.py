"""
User accounts and which orgs each user may open.

Owns: password hashing (scrypt from the standard library, a random salt per
password, compared in constant time), logging in with a lockout after
repeated failures, creating and removing users, and org memberships: a user
opens only the orgs they belong to, and becomes a member of an org they
create. Usernames are case-insensitive.
Does not own the login screens (app.py, main.py), the admin commands
(scripts/users.py) or org memory (memory.py).
"""

from __future__ import annotations

import asyncio
import base64
import functools
import hashlib
import hmac
import logging
import re
import secrets
from dataclasses import dataclass
from datetime import timedelta

from psycopg import errors

from . import db
from .logs import safe

USERNAME_PATTERN: re.Pattern[str] = re.compile(r"[a-z0-9._@+-]{1,64}")
MIN_PASSWORD_CHARS: int = 12
# scrypt cost: 16 MB and tens of milliseconds per check, so guessing offline is slow
# (the scrypt paper's setting for interactive logins).
SCRYPT_N: int = 2**14
SCRYPT_R: int = 8
SCRYPT_P: int = 1
SALT_BYTES: int = 16
# After this many wrong passwords in a row, the account refuses logins for a while.
MAX_FAILED_LOGINS: int = 5
LOCKOUT: timedelta = timedelta(minutes=15)
# One more wrong password: the MAX_FAILED_LOGINS-th locks the account and starts the count again.
_COUNT_FAILED_LOGIN_SQL: str = """
UPDATE users SET
    failed_logins = CASE WHEN failed_logins + 1 >= %(max)s THEN 0 ELSE failed_logins + 1 END,
    locked_until = CASE WHEN failed_logins + 1 >= %(max)s THEN now() + %(lockout)s
                        ELSE locked_until END
WHERE user_id = %(user_id)s
RETURNING locked_until > now() AS locked
"""


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class User:
    """A logged-in user."""

    user_id: int
    username: str


class AccountError(Exception):
    """An account change that can't be made; the message says why."""


class AccessRevoked(AccountError):
    """The user's account or their access to the org was removed during a session."""


def normalize_username(username: str) -> str:
    """
    The username as stored: trimmed and lowercase.

    Raises:
        AccountError: it has characters other than a-z, 0-9 and . _ @ + -, or is too long.
    """
    name = username.strip().lower()
    if not USERNAME_PATTERN.fullmatch(name):
        raise AccountError(
            "A username is 1 to 64 characters: letters, digits and . _ @ + - (an email works)."
        )
    return name


def hash_password(password: str) -> str:
    """Hash a password as "scrypt$n$r$p$salt$hash" (base64), with a new random salt."""
    salt = secrets.token_bytes(SALT_BYTES)
    digest = _scrypt(password, salt, SCRYPT_N, SCRYPT_R, SCRYPT_P)
    encoded = (base64.b64encode(part).decode() for part in (salt, digest))
    return "$".join(["scrypt", str(SCRYPT_N), str(SCRYPT_R), str(SCRYPT_P), *encoded])


def verify_password(password: str, stored: str) -> bool:
    """True if the password matches a hash_password() hash; False for any other hash."""
    try:
        scheme, n, r, p, salt, digest = stored.split("$")
        expected = base64.b64decode(digest)
        actual = _scrypt(password, base64.b64decode(salt), int(n), int(r), int(p))
    except ValueError:
        return False
    return scheme == "scrypt" and hmac.compare_digest(actual, expected)


def _scrypt(password: str, salt: bytes, n: int, r: int, p: int) -> bytes:
    """The scrypt digest of a password."""
    return hashlib.scrypt(
        password.encode(), salt=salt, n=n, r=r, p=p, maxmem=2 * 128 * r * n, dklen=32
    )


@functools.cache
def _unknown_user_hash() -> str:
    """
    A hash checked when a username doesn't exist.

    A wrong name then takes as long as a wrong password, so the timing doesn't
    tell an attacker which usernames are real.
    """
    return hash_password(secrets.token_urlsafe())


async def _verify(password: str, stored: str | None) -> bool:
    """
    verify_password in a worker thread, so a login doesn't stall other sessions.

    stored None checks the unknown-user hash, for a username that doesn't exist.
    """
    return await asyncio.to_thread(
        lambda: verify_password(password, stored or _unknown_user_hash()) and stored is not None
    )


def _check_password_strength(password: str) -> None:
    """Raise AccountError for a password shorter than MIN_PASSWORD_CHARS."""
    if len(password) < MIN_PASSWORD_CHARS:
        raise AccountError(f"A password needs at least {MIN_PASSWORD_CHARS} characters.")


async def create_user(username: str, password: str) -> User:
    """
    Create an account.

    Raises:
        AccountError: the username is taken or invalid, or the password too short.
    """
    name = normalize_username(username)
    _check_password_strength(password)
    password_hash = await asyncio.to_thread(hash_password, password)
    try:
        async with db.transaction() as conn:
            row = await (
                await conn.execute(
                    "INSERT INTO users (username, password_hash) VALUES (%s, %s) RETURNING user_id",
                    (name, password_hash),
                )
            ).fetchone()
    except errors.UniqueViolation as e:
        raise AccountError(f"The username {name} is taken.") from e
    logger.info("Account created: %s", safe(name))
    return User(row["user_id"], name)


async def ensure_user(username: str, password: str) -> bool:
    """
    Create the account unless one with this username exists; an existing one is left as it is.

    Returns:
        True if the account was created.

    Raises:
        AccountError: the username is invalid or the password too short.
    """
    name = normalize_username(username)
    _check_password_strength(password)
    password_hash = await asyncio.to_thread(hash_password, password)
    async with db.transaction() as conn:
        row = await (
            await conn.execute(
                "INSERT INTO users (username, password_hash) VALUES (%s, %s) "
                "ON CONFLICT (username) DO NOTHING RETURNING user_id",
                (name, password_hash),
            )
        ).fetchone()
    if row is not None:
        logger.info("Account created: %s", safe(name))
    return row is not None


async def set_password(username: str, password: str) -> bool:
    """
    Change a user's password and lift any lockout. Returns False if there is no such user.

    Raises:
        AccountError: the password is too short.
    """
    _check_password_strength(password)
    password_hash = await asyncio.to_thread(hash_password, password)
    name = normalize_username(username)
    async with db.transaction() as conn:
        cursor = await conn.execute(
            "UPDATE users SET password_hash = %s, failed_logins = 0, locked_until = NULL "
            "WHERE username = %s",
            (password_hash, name),
        )
        changed = cursor.rowcount > 0
    if changed:
        logger.info("Password changed (and any lockout lifted): %s", safe(name))
    return changed


async def delete_user(username: str) -> bool:
    """Delete a user and their memberships. Returns False if there is no such user."""
    name = normalize_username(username)
    async with db.transaction() as conn:
        cursor = await conn.execute("DELETE FROM users WHERE username = %s", (name,))
        deleted = cursor.rowcount > 0
    if deleted:
        logger.info("Account deleted: %s", safe(name))
    return deleted


async def authenticate(username: str, password: str) -> User | None:
    """
    Return the user if the password is right and the account isn't locked, else None.

    The answer is the same for an unknown user, a wrong password and a locked
    account, and each costs one password check, so neither the answer nor its
    timing reveals which usernames exist. MAX_FAILED_LOGINS wrong passwords in
    a row lock the account for LOCKOUT. The check runs with no database
    connection held, and the count of wrong passwords goes up in one
    statement, so a flood of guesses neither ties up the pool nor goes
    uncounted.
    """
    try:
        name = normalize_username(username)
    except AccountError:
        await _verify(password, None)
        # Not the typed text: it may be a password typed into the wrong field.
        logger.warning("Login refused: the username typed is not a valid username")
        return None
    async with db.transaction() as conn:
        row = await (
            await conn.execute(
                "SELECT user_id, password_hash, coalesce(locked_until > now(), false) AS locked "
                "FROM users WHERE username = %s",
                (name,),
            )
        ).fetchone()
    if row is None or row["locked"]:
        await _verify(password, None)
        reason = "no such user" if row is None else "the account is locked"
        logger.warning("Login refused for %s: %s", safe(name), reason)
        return None
    if not await _verify(password, row["password_hash"]):
        async with db.transaction() as conn:
            counted = await (
                await conn.execute(
                    _COUNT_FAILED_LOGIN_SQL,
                    {"max": MAX_FAILED_LOGINS, "lockout": LOCKOUT, "user_id": row["user_id"]},
                )
            ).fetchone()
        logger.warning("Login refused for %s: wrong password", safe(name))
        if counted and counted["locked"]:
            logger.warning(
                "Account %s locked for %s after %d wrong passwords",
                safe(name),
                LOCKOUT,
                MAX_FAILED_LOGINS,
            )
        return None
    async with db.transaction() as conn:
        # Not if guesses running alongside this login locked the account meanwhile.
        cleared = await (
            await conn.execute(
                "UPDATE users SET failed_logins = 0, locked_until = NULL WHERE user_id = %s "
                "AND (locked_until IS NULL OR locked_until <= now()) RETURNING user_id",
                (row["user_id"],),
            )
        ).fetchone()
    if not cleared:
        logger.warning("Login refused for %s: the account was locked meanwhile", safe(name))
        return None
    logger.info("Login: %s", safe(name))
    return User(row["user_id"], name)


async def get_user(user_id: int) -> User | None:
    """The user with this id, or None if the account has been deleted."""
    async with db.transaction() as conn:
        row = await (
            await conn.execute("SELECT user_id, username FROM users WHERE user_id = %s", (user_id,))
        ).fetchone()
    return User(row["user_id"], row["username"]) if row else None


async def user_orgs(user_id: int) -> list[str]:
    """The ids of the orgs a user may open, alphabetically."""
    async with db.transaction() as conn:
        rows = await (
            await conn.execute(
                "SELECT org_id FROM memberships WHERE user_id = %s ORDER BY org_id", (user_id,)
            )
        ).fetchall()
    return [row["org_id"] for row in rows]


async def has_access(user_id: int, org_id: str) -> bool:
    """
    True if the user's account exists and they belong to the org.

    Checked before every message, so removing a user or their access also
    stops chat sessions they already have open.
    """
    async with db.transaction() as conn:
        row = await (
            await conn.execute(
                "SELECT 1 FROM memberships WHERE user_id = %s AND org_id = %s",
                (user_id, db.check_org_id(org_id)),
            )
        ).fetchone()
    return row is not None


async def open_org(user_id: int, org_id: str) -> bool:
    """
    True if the user may open the org: they are a member, or it is new and they become its first.

    An org that exists and that the user doesn't belong to is refused; an
    admin adds members with scripts/users.py grant.

    Raises:
        ValueError: org_id is not a slugify() id.
    """
    member = None
    async with db.org_transaction(org_id) as conn:
        created = await (
            await conn.execute(
                "INSERT INTO orgs (org_id) VALUES (%s) ON CONFLICT (org_id) DO NOTHING "
                "RETURNING org_id",
                (org_id,),
            )
        ).fetchone()
        if created:
            await conn.execute(
                "INSERT INTO memberships (user_id, org_id) VALUES (%s, %s)", (user_id, org_id)
            )
        else:
            member = await (
                await conn.execute(
                    "SELECT 1 FROM memberships WHERE user_id = %s AND org_id = %s",
                    (user_id, org_id),
                )
            ).fetchone()
    if created:
        logger.info("User %d created org %s", user_id, org_id)
    elif member is None:
        logger.warning("User %d refused org %s: not a member", user_id, org_id)
    return bool(created) or member is not None


async def grant(username: str, org_id: str) -> None:
    """
    Let a user open an org, creating the org if it doesn't exist yet.

    Raises:
        AccountError: there is no such user.
        ValueError: org_id is not a slugify() id.
    """
    async with db.org_transaction(org_id) as conn:
        user = await (
            await conn.execute(
                "SELECT user_id FROM users WHERE username = %s", (normalize_username(username),)
            )
        ).fetchone()
        if user is None:
            raise AccountError(f"There is no user {username}.")
        await conn.execute(
            "INSERT INTO orgs (org_id) VALUES (%s) ON CONFLICT (org_id) DO NOTHING", (org_id,)
        )
        await conn.execute(
            "INSERT INTO memberships (user_id, org_id) VALUES (%s, %s) ON CONFLICT DO NOTHING",
            (user["user_id"], org_id),
        )
    logger.info("Access granted: %s to %s", safe(normalize_username(username)), org_id)


async def revoke(username: str, org_id: str) -> bool:
    """Stop a user opening an org. Returns False if they weren't a member."""
    async with db.transaction() as conn:
        cursor = await conn.execute(
            "DELETE FROM memberships WHERE org_id = %s AND user_id = "
            "(SELECT user_id FROM users WHERE username = %s)",
            (db.check_org_id(org_id), normalize_username(username)),
        )
        revoked = cursor.rowcount > 0
    if revoked:
        logger.info("Access revoked: %s from %s", safe(normalize_username(username)), org_id)
    return revoked


async def list_users() -> list[tuple[str, list[str]]]:
    """Every username with the orgs that user may open, alphabetically."""
    async with db.transaction() as conn:
        rows = await (
            await conn.execute(
                "SELECT u.username, coalesce(array_agg(m.org_id ORDER BY m.org_id) "
                "FILTER (WHERE m.org_id IS NOT NULL), '{}') AS orgs "
                "FROM users u LEFT JOIN memberships m USING (user_id) "
                "GROUP BY u.username ORDER BY u.username"
            )
        ).fetchall()
    return [(row["username"], list(row["orgs"])) for row in rows]
