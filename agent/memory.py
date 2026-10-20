"""
Persistent storage for org profiles, post ledgers, preferences and the tool-call trace.

Every org's memory lives in PostgreSQL (schema in migrations/), one row per
post, preference and voice example. Each function that reaches the database
runs in one db.org_transaction(), so the database refuses any other org's rows. A
read-change-write first locks the org's row, so the web UI, the CLI and
scripts on any number of servers can write to one org without overwriting
each other. A profile can be exported to JSON for a non-engineer to edit and
imported again (scripts/export_profile.py, scripts/import_profiles.py); read_profile_file()
checks such a file with plain-English messages.

Owns: assembling the profile dict the prompt and tools read, saving or
replacing it, the org-id guard (slugify), assigning post ids, searching the
ledger, the agenda of posts that need the user's attention, performance
results reported for posted posts, learned preferences with their
provenance, voice examples, research, and the tool-call trace.
Does not own the connection pool or migrations (db.py), accounts
(accounts.py), LLM interaction, tool dispatch (tools.py), or fetching
research (research.py).
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Callable
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Any

from psycopg import AsyncConnection
from psycopg.rows import tuple_row
from psycopg.types.json import Jsonb

from . import db

MAX_VOICE_EXAMPLES: int = 5
POST_STATUSES: tuple[str, ...] = ("suggested", "draft", "approved", "posted", "planned", "rejected")
# Engagement metrics a user can report for a post; reach is recorded too but is not engagement.
ENGAGEMENT_METRICS: tuple[str, ...] = ("reactions", "comments", "shares", "clicks", "saves")
PERFORMANCE_METRICS: tuple[str, ...] = ("reach", *ENGAGEMENT_METRICS)
# The agenda shows planned posts due within this many days, and asks for results this
# many days after a post went out (platforms need a few days to collect them).
AGENDA_LOOKAHEAD_DAYS: int = 7
RESULTS_AFTER_DAYS: int = 3
POST_ID_PATTERN: re.Pattern[str] = re.compile(r"post_(\d+)")
# Years a stored time may have: within what every time zone can show and PostgreSQL holds.
MIN_YEAR: int = 2
MAX_YEAR: int = 9998
# The largest post number the database's integer counter holds.
MAX_POST_NUMBER: int = 2**31 - 1
# Post fields with a column of their own; every other field is kept in posts.details.
_POST_COLUMNS: tuple[str, ...] = (
    "id",
    "status",
    "platform",
    "content",
    "created_at",
    "performance",
)
_INSERT_POSTS_SQL: str = """
INSERT INTO posts (org_id, post_id, status, platform, content, created_at, details, performance)
SELECT %s, p.post_id, p.status, p.platform, p.content, p.created_at, p.details, p.performance
FROM ROWS FROM (
    jsonb_to_recordset(%s) AS (
        post_id text, status text, platform text, content text,
        created_at timestamptz, details jsonb, performance jsonb
    )
) WITH ORDINALITY AS p (
    post_id, status, platform, content, created_at, details, performance, n
)
ORDER BY p.n
"""


class ProfileError(Exception):
    """A profile file, or a profile to save, has a mistake; the message says which and where."""


def slugify(text: str) -> str:
    """
    Convert a display name to an org ID ([a-z0-9-] only).

    A name with no Latin letters or digits (e.g. Greek or Japanese) gets a
    stable "org-" + hash ID instead of an empty one, which would put every
    such org under one id. Returns "" for a name with no letters or digits in
    any script ("", "!!!"), so the UIs ask for the name again.
    """
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    if slug or not any(char.isalnum() for char in text):
        return slug
    return "org-" + hashlib.sha256(text.strip().casefold().encode()).hexdigest()[:10]


def _storable(value: Any) -> Any:
    """
    The value with what PostgreSQL can't store replaced, in every string and number in it.

    NUL characters are removed, a lone UTF-16 surrogate (half of an emoji, as
    a model or a JSON file can send) becomes "?", and an infinite or NaN number
    becomes null. Refusing the whole save over one of these would lose the
    post, and the database's error would quote the value into the logs.
    """
    if isinstance(value, str):
        return value.replace("\x00", "").encode("utf-8", "replace").decode("utf-8")
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, list | tuple):
        return type(value)(_storable(item) for item in value)
    if isinstance(value, dict):
        return {_storable(str(key)): _storable(item) for key, item in value.items()}
    return value


def _now() -> datetime:
    """The current time with the local time zone, for timestamptz columns."""
    return datetime.now().astimezone()


def _iso(moment: datetime | None, timespec: str = "auto") -> str:
    """
    A stored time as ISO text in the local zone, with its UTC offset.

    The offset makes an export mean the same moment wherever it is imported;
    the first 10 characters are still the local date.
    """
    if moment is None:
        return ""
    return moment.astimezone().isoformat(timespec=timespec)


def _parse_time(value: Any) -> datetime | None:
    """
    An ISO date-time; a time without a zone is read as local. None if it isn't one.

    Also None for a year PostgreSQL or the local time zone can't represent
    (year 1 and 9999 overflow in some zones), which would otherwise import fine
    and then make the org's profile unloadable.
    """
    try:
        moment = datetime.fromisoformat(str(value))
        moment = moment if moment.tzinfo else moment.astimezone()
        moment.astimezone()  # its local form must exist too: load_profile shows that
    except (ValueError, OverflowError):
        return None
    return moment if MIN_YEAR <= moment.year <= MAX_YEAR else None


def default_profile(org_id: str) -> dict[str, Any]:
    """The profile of an org Mackie knows nothing about yet."""
    return {
        "org_id": org_id,
        "name": "",
        "website": "",
        "research": None,
        "org_brief": "",
        "research_updated_at": "",
        "content_pillars": None,
        "preferences": {},
        "preference_notes": {},
        "voice_examples": [],
        "post_ledger": [],
    }


def _post_from_row(row: dict[str, Any]) -> dict[str, Any]:
    """A posts row as the post dict the prompt and tools read."""
    post = {
        "id": row["post_id"],
        "platform": row["platform"],
        "content": row["content"],
        "status": row["status"],
        **row["details"],
        "created_at": _iso(row["created_at"]),
    }
    if row["performance"] is not None:
        post["performance"] = row["performance"]
    return post


def _post_record(post: dict[str, Any]) -> dict[str, Any]:
    """A post dict as a record for _INSERT_POSTS_SQL."""
    created = _parse_time(post.get("created_at")) or _now()
    return _storable(
        {
            "post_id": post["id"],
            "status": post.get("status") or "suggested",
            "platform": str(post.get("platform", "")),
            "content": str(post.get("content", "")),
            "created_at": created.isoformat(),
            "details": {k: v for k, v in post.items() if k not in _POST_COLUMNS},
            "performance": post.get("performance"),
        }
    )


async def _insert_posts(conn: AsyncConnection, org_id: str, posts: list[dict[str, Any]]) -> None:
    """Insert posts that already have ids, in ledger order, in one statement."""
    if posts:
        records = [_post_record(post) for post in posts]
        await conn.execute(_INSERT_POSTS_SQL, (org_id, Jsonb(records)))


async def _fetch_posts(conn: AsyncConnection, query: str, params: Any) -> list[dict[str, Any]]:
    """
    Run a query on posts and return its rows as post dicts.

    Rows come back as tuples and are paired with the column names here: the
    prompt reads every post on every turn, and psycopg's dict rows took half
    the time of loading a 20,000-post ledger.
    """
    cursor = conn.cursor(row_factory=tuple_row)
    await cursor.execute(query, params)
    names = [column.name for column in cursor.description or []]
    return [_post_from_row(dict(zip(names, row, strict=True))) for row in await cursor.fetchall()]


async def _lock_org(conn: AsyncConnection, org_id: str) -> dict[str, Any]:
    """
    Create the org's row if it is new, lock it until the transaction ends, and return it.

    Every read-change-write takes this lock first, so two sessions writing to
    one org take turns instead of overwriting each other.
    """
    await conn.execute(
        "INSERT INTO orgs (org_id) VALUES (%s) ON CONFLICT (org_id) DO NOTHING", (org_id,)
    )
    cursor = await conn.execute("SELECT * FROM orgs WHERE org_id = %s FOR UPDATE", (org_id,))
    row = await cursor.fetchone()
    assert row is not None  # inserted above, and only this org's transaction can delete it
    return row


async def load_profile(org_id: str) -> dict[str, Any]:
    """
    Load the org's profile from one snapshot of the database.

    Args:
        org_id: The org identifier (from slugify).

    Returns:
        Dict with keys: org_id, name, website, research, org_brief,
        research_updated_at, content_pillars, preferences, preference_notes,
        voice_examples, post_ledger (oldest post first). An org with nothing
        saved yet gets default_profile().

    Raises:
        ValueError: org_id is not a slugify() id.
    """
    profile = default_profile(org_id)
    async with db.org_transaction(org_id, read_only=True) as conn:
        cursors = [
            await conn.execute(
                f"SELECT * FROM {table} WHERE org_id = %s ORDER BY {order}", (org_id,)
            )
            for table, order in (
                ("orgs", "org_id"),
                ("preferences", "seq"),
                ("voice_examples", "id"),
            )
        ]
        org = await cursors[0].fetchone()
        prefs, examples = [await cursor.fetchall() for cursor in cursors[1:]]
        ledger = await _fetch_posts(
            conn, "SELECT * FROM posts WHERE org_id = %s ORDER BY seq", (org_id,)
        )
    if org is None:
        return profile
    profile.update(
        {key: org[key] for key in ("name", "website", "research", "org_brief", "content_pillars")},
        research_updated_at=org["research_updated_at"],
        preferences={row["key"]: row["value"] for row in prefs},
        preference_notes={
            row["key"]: {
                "learned_at": _iso(row["learned_at"], "seconds"),
                "quote": row["quote"] or "",
            }
            for row in prefs
            if row["learned_at"] is not None or row["quote"]
        },
        voice_examples=[
            {
                "content": row["content"],
                "source": row["source"],
                "added_at": _iso(row["added_at"], "seconds"),
            }
            for row in examples
        ],
        post_ledger=ledger,
    )
    return profile


def read_profile_file(path: Path) -> dict[str, Any]:
    """
    Read a profile exported with scripts/export_profile.py, after a person may have edited it.

    Raises:
        ProfileError: the file is not valid JSON, not a JSON object, or a
            section has the wrong type; the message names the file and the fix.
    """
    try:
        data = json.loads(
            path.read_text(), parse_constant=_refuse_constant, parse_float=_finite_float
        )
    except ProfileError as e:
        raise ProfileError(f"{path}: {e}") from e
    except json.JSONDecodeError as e:
        raise ProfileError(
            f"{path} has a formatting error at line {e.lineno}, column {e.colno} ({e.msg})."
        ) from e
    if not isinstance(data, dict):
        raise ProfileError(f"{path} must be a JSON object ({{ ... }}) at the top level.")
    if problem := profile_problem(data):
        raise ProfileError(f"{path}: {problem}.")
    return data


def _finite_float(text: str) -> float:
    """A JSON number as a float, refusing one too large to store (1e999 reads as infinity)."""
    number = float(text)
    if not math.isfinite(number):
        raise ProfileError(f"{text} is too large a number to store; use a smaller one.")
    return number


def _refuse_constant(name: str) -> None:
    """Refuse NaN and Infinity, which Python's JSON reader accepts and PostgreSQL doesn't."""
    raise ProfileError(f"{name} is not a number PostgreSQL can store; use a plain number or null.")


def profile_problem(data: dict[str, Any]) -> str:
    """
    Describe the first thing in a profile that can't be saved as it is, or return "".

    Checks section types, the fields the prompt relies on (research sources
    are a list, a post's parent_id is a post id), and what the database
    constrains: ids look like post_001, fit its counter and are unique,
    statuses are POST_STATUSES, and dates and times are dates and times.
    """
    for key in ("post_ledger", "voice_examples"):
        items = data.get(key, [])
        if not isinstance(items, list) or not all(isinstance(item, dict) for item in items):
            return f"{key} must be a list of {{ ... }} entries"
    for key in ("preferences", "preference_notes"):
        if not isinstance(data.get(key, {}), dict):
            return f"{key} must be a {{ ... }} object"
    if not isinstance(data.get("research"), dict | None):
        return "research must be a { ... } object, or null to research the org again"
    if not isinstance(data.get("content_pillars"), list | None):
        return "content_pillars must be a list of { ... } entries, or null for the defaults"
    if not isinstance((data.get("research") or {}).get("sources", []), list):
        return "research sources must be a list of { ... } entries"
    for key, note in data.get("preference_notes", {}).items():
        if not isinstance(note, dict):
            return f"the note for preference {key!r} must be a {{ ... }} object"
        if note.get("learned_at") and _parse_time(note["learned_at"]) is None:
            return f"preference {key!r} has learned_at {note['learned_at']!r}, which is not a date and time"
    for number, example in enumerate(data.get("voice_examples", []), 1):
        if example.get("added_at") and _parse_time(example["added_at"]) is None:
            return f"voice example {number} has added_at {example['added_at']!r}, which is not a date and time"
    seen: set[str] = set()
    for number, post in enumerate(data.get("post_ledger", []), 1):
        post_id = post.get("id")
        if post_id is not None and not POST_ID_PATTERN.fullmatch(str(post_id)):
            return f"post {number} has id {post_id!r}; ids look like post_001, or leave it out"
        if post_id is not None and int(str(post_id).removeprefix("post_")) > MAX_POST_NUMBER:
            return f"post {number} has id {post_id!r}; the number is too large"
        if not isinstance(post.get("parent_id", ""), str):
            return f"post {number} has parent_id {post['parent_id']!r}; it must be a post id like post_001"
        if post_id is not None and post_id in seen:
            return f"two posts have the id {post_id}"
        seen.add(post_id)
        if post.get("status", "suggested") not in POST_STATUSES:
            return f"post {number} has status {post.get('status')!r}; use one of {', '.join(POST_STATUSES)}"
        if post.get("created_at") is not None and _parse_time(post["created_at"]) is None:
            return (
                f"post {number} has created_at {post['created_at']!r}, which is not a date and time"
            )
    return ""


async def save_profile(org_id: str, profile: dict[str, Any]) -> None:
    """
    Replace everything stored for the org with this profile, in one transaction.

    For scripts/import_profiles.py and tests; the assistant changes one thing at a time
    through the other functions here. A post without an id gets the next one.
    Post numbers already handed out are never handed out again.

    Args:
        org_id:  The org identifier (from slugify).
        profile: A profile dict, as load_profile returns or an export file holds.

    Raises:
        ProfileError: profile_problem() found a mistake; nothing is changed.
        ValueError: org_id is not a slugify() id.
    """
    if problem := profile_problem(profile):
        raise ProfileError(problem)
    async with db.org_transaction(org_id) as conn:
        org = await _lock_org(conn, org_id)
        ledger = [dict(post) for post in profile.get("post_ledger", [])]
        numbers = [int(POST_ID_PATTERN.fullmatch(p["id"])[1]) for p in ledger if p.get("id")]
        last = max([org["last_post_number"], *numbers])
        for post in ledger:
            if not post.get("id"):
                last += 1
                post["id"] = f"post_{last:03d}"
        await conn.execute(
            "UPDATE orgs SET name = %s, website = %s, research = %s, org_brief = %s, "
            "research_updated_at = %s, content_pillars = %s, last_post_number = %s "
            "WHERE org_id = %s",
            _storable(
                (
                    str(profile.get("name") or ""),
                    str(profile.get("website") or ""),
                    _jsonb_or_null(profile.get("research")),
                    str(profile.get("org_brief") or ""),
                    str(profile.get("research_updated_at") or ""),
                    _jsonb_or_null(profile.get("content_pillars")),
                    last,
                    org_id,
                )
            ),
        )
        for table in ("posts", "preferences", "voice_examples"):
            await conn.execute(f"DELETE FROM {table} WHERE org_id = %s", (org_id,))
        await _insert_posts(conn, org_id, ledger)
        notes = profile.get("preference_notes", {})
        for key, value in profile.get("preferences", {}).items():
            note = notes.get(key) if isinstance(notes.get(key), dict) else {}
            await conn.execute(
                "INSERT INTO preferences (org_id, key, value, quote, learned_at) "
                "VALUES (%s, %s, %s, %s, %s)",
                _storable(
                    (
                        org_id,
                        str(key),
                        Jsonb(_storable(value)),
                        str(note.get("quote") or ""),
                        _parse_time(note.get("learned_at")) if note else None,
                    )
                ),
            )
        for example in profile.get("voice_examples", []):
            await conn.execute(
                "INSERT INTO voice_examples (org_id, content, source, added_at) "
                "VALUES (%s, %s, %s, %s)",
                _storable(
                    (
                        org_id,
                        str(example.get("content", "")),
                        str(example.get("source", "")),
                        _parse_time(example.get("added_at")) or _now(),
                    )
                ),
            )


def _jsonb_or_null(value: Any) -> Jsonb | None:
    """A value for a nullable jsonb column."""
    return None if value is None else Jsonb(_storable(value))


async def fill_org_details(org_id: str, name: str, website: str = "") -> None:
    """
    Record the org's name and website where the profile has none yet.

    What is already there is kept, so a corrected name is never overwritten
    by what someone typed at onboarding.
    """
    async with db.org_transaction(org_id) as conn:
        org = await _lock_org(conn, org_id)
        await conn.execute(
            "UPDATE orgs SET name = %s, website = %s WHERE org_id = %s",
            _storable((org["name"] or name, org["website"] or website, org_id)),
        )


async def save_research(
    org_id: str, research: dict[str, Any], brief: str, updated_at: str = ""
) -> None:
    """
    Store the org's research and the brief written from it.

    Args:
        org_id:     The org identifier.
        research:   What research.research_org returned.
        brief:      The short org brief for the prompt.
        updated_at: When a refresh ran, shown to the user; "" keeps the earlier value.
    """
    async with db.org_transaction(org_id) as conn:
        await _lock_org(conn, org_id)
        await conn.execute(
            "UPDATE orgs SET research = %s, org_brief = %s, "
            "research_updated_at = coalesce(nullif(%s, ''), research_updated_at) WHERE org_id = %s",
            (Jsonb(_storable(research)), _storable(brief), updated_at, org_id),
        )


async def update_research(org_id: str, change: Callable[[dict[str, Any]], bool]) -> None:
    """
    Change the org's research in place, with the org locked, and save it if change returns True.

    Nothing happens for an org that has no research yet. change runs inside
    the transaction, so it must not wait on anything.
    """
    async with db.org_transaction(org_id) as conn:
        research = (await _lock_org(conn, org_id))["research"]
        if isinstance(research, dict) and change(research):
            await conn.execute(
                "UPDATE orgs SET research = %s WHERE org_id = %s",
                (Jsonb(_storable(research)), org_id),
            )


async def save_preference(
    org_id: str, key: str, value: str, quote: str = "", append: bool = False
) -> None:
    """
    Persist a brand preference for the org, with a note of where it came from.

    The value stays a plain value (a string, or a list when append is used);
    the date and the user's own words are stored beside it.

    Args:
        org_id: The org identifier.
        key:    Preference name (e.g. "voice", "banned_topics").
        value:  Preference value.
        quote:  The user's words that expressed the preference, if any.
        append: Add value to a list under key (e.g. one more banned topic)
                instead of replacing what is there.
    """
    key, value, quote = _storable(key), _storable(value), _storable(quote)
    async with db.org_transaction(org_id) as conn:
        await _lock_org(conn, org_id)
        new_value: Any = value
        if append:
            row = await (
                await conn.execute(
                    "SELECT value FROM preferences WHERE org_id = %s AND key = %s", (org_id, key)
                )
            ).fetchone()
            current = row["value"] if row else []
            items = current if isinstance(current, list) else [current]
            new_value = items + [value] if value not in items else items
        await conn.execute(
            "INSERT INTO preferences (org_id, key, value, quote, learned_at) "
            "VALUES (%s, %s, %s, %s, %s) ON CONFLICT (org_id, key) DO UPDATE SET "
            "value = EXCLUDED.value, quote = EXCLUDED.quote, learned_at = EXCLUDED.learned_at",
            (org_id, key, Jsonb(new_value), quote, _now()),
        )


async def forget_preference(org_id: str, key: str, value: str | None = None) -> bool:
    """
    Remove a learned preference, or one item from a list-valued preference.

    Args:
        org_id: The org identifier.
        key:    Preference name.
        value:  If given, remove only this item of a list preference, or the
                preference itself only if its value matches; both ignore case.

    Returns:
        True if something was removed; False if the key (or value) was not found.
    """
    # Matched as save_preference stored them (NUL removed, a lone surrogate as "?").
    key, value = _storable(key), _storable(value)
    async with db.org_transaction(org_id) as conn:
        await _lock_org(conn, org_id)
        row = await (
            await conn.execute(
                "SELECT value FROM preferences WHERE org_id = %s AND key = %s", (org_id, key)
            )
        ).fetchone()
        if row is None:
            return False
        current = row["value"]
        if value is not None and isinstance(current, list):
            match = next(
                (item for item in current if str(item).casefold() == value.casefold()), None
            )
            if match is None:
                return False
            current.remove(match)
            if current:
                await conn.execute(
                    "UPDATE preferences SET value = %s WHERE org_id = %s AND key = %s",
                    (Jsonb(current), org_id, key),
                )
                return True
        elif value is not None and str(current).casefold() != value.casefold():
            return False
        await conn.execute("DELETE FROM preferences WHERE org_id = %s AND key = %s", (org_id, key))
        return True


async def add_voice_example(org_id: str, content: str, source: str) -> None:
    """
    Remember a post that shows the org's real voice, keeping the newest few.

    Voice examples come from posts the org approved, wrote themselves, or
    rewrote from a draft; the prompt shows them as the style to imitate.

    Args:
        org_id:  The org identifier.
        content: The post text.
        source:  Where it came from, e.g. "approved post_004" or "pasted by user".
    """
    async with db.org_transaction(org_id) as conn:
        await _lock_org(conn, org_id)
        await conn.execute(
            "INSERT INTO voice_examples (org_id, content, source, added_at) VALUES (%s, %s, %s, %s)",
            (org_id, _storable(content), _storable(source), _now()),
        )
        await conn.execute(
            "DELETE FROM voice_examples WHERE org_id = %s AND id NOT IN "
            "(SELECT id FROM voice_examples WHERE org_id = %s ORDER BY id DESC LIMIT %s)",
            (org_id, org_id, MAX_VOICE_EXAMPLES),
        )


async def add_post(org_id: str, post: dict[str, Any]) -> str:
    """
    Append a post to the org's ledger, assigning an ID and timestamp.

    The next post number is taken from the org's counter, so ids are never
    reused, even after posts are deleted or a profile is replaced. post is
    changed in place: it gets its id, created_at and a default status.

    Args:
        org_id: The org identifier.
        post:   Post dict with at minimum a 'platform' and 'content' key.
                'status' defaults to "suggested" if not provided.

    Returns:
        The assigned post ID string (e.g. "post_001").
    """
    async with db.org_transaction(org_id) as conn:
        await _lock_org(conn, org_id)
        row = await (
            await conn.execute(
                "UPDATE orgs SET last_post_number = last_post_number + 1 WHERE org_id = %s "
                "RETURNING last_post_number",
                (org_id,),
            )
        ).fetchone()
        post_id = f"post_{row['last_post_number']:03d}"
        post["id"] = post_id
        post["created_at"] = _now().isoformat()
        post.setdefault("status", "suggested")
        await _insert_posts(conn, org_id, [post])
    return post_id


async def update_post_status(org_id: str, post_id: str, status: str) -> bool:
    """
    Update the status of an existing ledger entry.

    Args:
        org_id:  The org identifier.
        post_id: ID of the post to update (e.g. "post_001").
        status:  New status, one of POST_STATUSES.

    Returns:
        True if the post was found and updated; False if post_id does not exist.
    """
    async with db.org_transaction(org_id) as conn:
        cursor = await conn.execute(
            "UPDATE posts SET status = %s WHERE org_id = %s AND post_id = %s RETURNING post_id",
            (status, org_id, _storable(post_id)),
        )
        return await cursor.fetchone() is not None


async def record_performance(
    org_id: str, post_id: str, metrics: dict[str, int], note: str = ""
) -> bool:
    """
    Store the results the user reported for a published post, and mark it posted.

    A post only has results once it is live, so its status becomes "posted".
    Reporting again replaces the earlier numbers: platforms show running
    totals, so the latest report is the most complete.

    Args:
        org_id:  The org identifier.
        post_id: ID of the post the results are for.
        metrics: Counts keyed by names from PERFORMANCE_METRICS.
        note:    Anything else the user said, e.g. "brought in 3 donations".

    Returns:
        True if the post was found and updated; False if post_id does not exist.
    """
    performance = {
        **metrics,
        **({"note": note} if note else {}),
        "recorded_at": _now().isoformat(timespec="seconds"),
    }
    async with db.org_transaction(org_id) as conn:
        cursor = await conn.execute(
            "UPDATE posts SET performance = %s, status = 'posted' "
            "WHERE org_id = %s AND post_id = %s RETURNING post_id",
            (Jsonb(_storable(performance)), org_id, _storable(post_id)),
        )
        return await cursor.fetchone() is not None


def engagement(post: dict[str, Any]) -> int:
    """Total engagement (reactions, comments, shares, clicks, saves) recorded for a post."""
    performance = post.get("performance")
    if not isinstance(performance, dict):
        return 0
    # isinstance check: an imported count such as "40" or null must not break every turn.
    return sum(
        value
        for name in ENGAGEMENT_METRICS
        if isinstance(value := performance.get(name), int) and not isinstance(value, bool)
    )


def top_performing(ledger: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    """
    Return up to limit posts that have recorded results, highest engagement first.

    Args:
        ledger: Post dicts, as in profile["post_ledger"].
        limit:  Maximum number of posts to return.
    """
    measured = [post for post in ledger if isinstance(post.get("performance"), dict)]
    return sorted(measured, key=engagement, reverse=True)[:limit]


def _post_day(post: dict[str, Any]) -> str:
    """The post's scheduled date, or the day it was created, as YYYY-MM-DD ("" if neither)."""
    return str(post.get("scheduled_for") or post.get("created_at") or "")[:10]


def agenda(ledger: list[dict[str, Any]], today: date) -> dict[str, list[dict[str, Any]]]:
    """
    Group the posts that need the user's attention by the next step each needs.

    A draft or approved post that has a newer version on the same platform is
    left out: the newer version is the one to act on. A version for another
    platform, or one the user rejected, does not replace it.

    Args:
        ledger: Post dicts, as in profile["post_ledger"].
        today:  The date to measure "due" from.

    Returns:
        Dict with lists of posts under "to_draft" (planned and due within
        AGENDA_LOOKAHEAD_DAYS, or overdue), "awaiting_approval" (drafts),
        "to_publish" (approved) and "needs_results" (posted at least
        RESULTS_AFTER_DAYS ago with no results recorded).
    """
    horizon = (today + timedelta(days=AGENDA_LOOKAHEAD_DAYS)).isoformat()
    results_due = (today - timedelta(days=RESULTS_AFTER_DAYS)).isoformat()
    replaced = {
        (post.get("parent_id"), post.get("platform"))
        for post in ledger
        if post.get("status") != "rejected"
    }

    def current(post: dict[str, Any], status: str) -> bool:
        return (
            post.get("status") == status and (post.get("id"), post.get("platform")) not in replaced
        )

    return {
        "to_draft": [p for p in ledger if current(p, "planned") and _post_day(p) <= horizon],
        "awaiting_approval": [p for p in ledger if current(p, "draft")],
        "to_publish": [p for p in ledger if current(p, "approved")],
        "needs_results": [
            p
            for p in ledger
            if p.get("status") == "posted"
            and not isinstance(p.get("performance"), dict)
            and _post_day(p) <= results_due
        ],
    }


async def search_ledger(
    org_id: str,
    status: str | None = None,
    platform: str | None = None,
    since: str | None = None,
    until: str | None = None,
    text: str | None = None,
) -> list[dict[str, Any]]:
    """
    Return ledger posts matching every filter given (all filters optional).

    Args:
        org_id:   The org identifier.
        status:   Exact status, e.g. "draft".
        platform: Exact platform id, e.g. "instagram" (case-insensitive).
        since:    ISO date "YYYY-MM-DD"; posts created on or after it (local time).
        until:    ISO date "YYYY-MM-DD"; posts created on or before it (local time).
        text:     Case-insensitive substring of the title or content.

    Returns:
        Matching post dicts, oldest first.

    Raises:
        ValueError: since or until is not a YYYY-MM-DD date.
    """
    conditions, params = ["org_id = %s"], [org_id]
    if status is not None:
        conditions.append("status = %s")
        params.append(_storable(status))
    if platform is not None:
        conditions.append("lower(platform) = lower(%s)")
        params.append(_storable(platform))
    if since is not None:
        conditions.append("created_at >= %s")
        params.append(datetime.combine(date.fromisoformat(since), time.min).astimezone())
    if until is not None:
        next_day = date.fromisoformat(until) + timedelta(days=1)
        conditions.append("created_at < %s")
        params.append(datetime.combine(next_day, time.min).astimezone())
    if text is not None:
        # strpos, not LIKE: the user's text may contain % or _.
        conditions.append(
            "strpos(lower(coalesce(details->>'title', '') || ' ' || content), lower(%s)) > 0"
        )
        params.append(_storable(text))
    async with db.org_transaction(org_id, read_only=True) as conn:
        return await _fetch_posts(
            conn, f"SELECT * FROM posts WHERE {' AND '.join(conditions)} ORDER BY seq", params
        )


async def get_ledger(org_id: str) -> list[dict[str, Any]]:
    """Return the org's posts, oldest first; an empty list if none have been saved."""
    return await search_ledger(org_id)


async def get_post(org_id: str, post_id: str) -> dict[str, Any] | None:
    """Return one post of the org's ledger, or None if there is no post with that id."""
    async with db.org_transaction(org_id, read_only=True) as conn:
        row = await (
            await conn.execute(
                "SELECT * FROM posts WHERE org_id = %s AND post_id = %s",
                (org_id, _storable(post_id)),
            )
        ).fetchone()
    return _post_from_row(row) if row else None


async def append_trace(org_id: str, entry: dict[str, Any]) -> None:
    """
    Record one tool call: entry has ts (epoch seconds), tool, input, result_preview, latency_ms.
    """
    async with db.org_transaction(org_id) as conn:
        await conn.execute(
            "INSERT INTO orgs (org_id) VALUES (%s) ON CONFLICT (org_id) DO NOTHING", (org_id,)
        )
        await conn.execute(
            "INSERT INTO tool_calls (org_id, called_at, tool, input, result_preview, latency_ms) "
            "VALUES (%s, to_timestamp(%s), %s, %s, %s, %s)",
            (
                org_id,
                entry["ts"],
                _storable(entry["tool"]),
                Jsonb(_storable(entry["input"])),
                _storable(entry["result_preview"]),
                entry["latency_ms"],
            ),
        )


async def read_trace(org_id: str) -> list[dict[str, Any]]:
    """The org's recorded tool calls, oldest first, in the shape append_trace takes."""
    async with db.org_transaction(org_id, read_only=True) as conn:
        rows = await (
            await conn.execute(
                "SELECT extract(epoch FROM called_at)::float8 AS ts, tool, input, "
                "result_preview, latency_ms FROM tool_calls WHERE org_id = %s ORDER BY id",
                (org_id,),
            )
        ).fetchall()
    return [dict(row) for row in rows]
