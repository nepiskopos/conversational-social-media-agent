"""
Persistent storage for org profiles, post ledgers, and preferences.

All org state is stored as JSON on disk at data/{org_id}/profile.json, so a
non-engineer can read and edit what the assistant has learned in any text
editor, with no database setup or migrations.

Every load-modify-save runs under locked(org_id), an exclusive lock file per
org, so the web UI, the CLI and scripts can write to one org from separate
processes without overwriting each other. It works on one machine only;
several servers sharing data/ need a database with row-level locking.

Owns: reading and writing profile.json for each org (atomic writes with a
profile.json.bak copy, the per-org lock, a readable error for hand-editing
mistakes), the org-id guard, assigning post IDs, searching the ledger, the
agenda of posts that need the user's attention, performance results reported
for posted posts, learned preferences with their provenance, voice examples,
and the path of the trace file.
Does not own LLM interaction, tool dispatch, or research.
"""

from __future__ import annotations

import fcntl
import functools
import hashlib
import json
import os
import re
import shutil
import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

DATA_DIR: Path = Path(__file__).parent.parent / "data"
MAX_VOICE_EXAMPLES: int = 5
POST_STATUSES: tuple[str, ...] = ("suggested", "draft", "approved", "posted", "planned", "rejected")
# Engagement metrics a user can report for a post; reach is recorded too but is not engagement.
ENGAGEMENT_METRICS: tuple[str, ...] = ("reactions", "comments", "shares", "clicks", "saves")
PERFORMANCE_METRICS: tuple[str, ...] = ("reach", *ENGAGEMENT_METRICS)
# The agenda shows planned posts due within this many days, and asks for results this
# many days after a post went out (platforms need a few days to collect them).
AGENDA_LOOKAHEAD_DAYS: int = 7
RESULTS_AFTER_DAYS: int = 3
ORG_ID_PATTERN: re.Pattern[str] = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")
LOCK_FILE_NAME: str = ".lock"
# Lock files the current thread holds; locked() is reentrant through this. Per thread,
# because flock already excludes other threads, which must not skip it.
_held_locks = threading.local()


class ProfileError(Exception):
    """profile.json exists but cannot be read — usually a hand-editing mistake."""


def slugify(text: str) -> str:
    """
    Convert a display name to a filesystem-safe org ID ([a-z0-9-] only).

    A name with no Latin letters or digits (e.g. Greek or Japanese) gets a
    stable "org-" + hash ID instead of an empty one, which would put every
    such org in the same data/ folder. Returns "" for a name with no letters
    or digits in any script ("", "!!!"), so the UIs ask for the name again.
    """
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    if slug or not any(char.isalnum() for char in text):
        return slug
    return "org-" + hashlib.sha256(text.strip().casefold().encode()).hexdigest()[:10]


def _org_dir(org_id: str) -> Path:
    """
    Return (and create) the per-org data directory.

    Raises:
        ValueError: org_id is not a slugify() id, so it could point outside data/.
    """
    if not ORG_ID_PATTERN.fullmatch(org_id):
        raise ValueError(f"invalid org id {org_id!r}; org ids come from slugify()")
    d = DATA_DIR / org_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def _profile_path(org_id: str) -> Path:
    """Return the path to the org's profile.json file."""
    return _org_dir(org_id) / "profile.json"


@contextmanager
def locked(org_id: str) -> Iterator[None]:
    """
    Hold the org's exclusive lock while loading, changing and saving its profile.

    The web UI, the CLI and scripts can run as separate processes on one org;
    without the lock, two load-change-save sequences interleave and one
    silently overwrites the other. Holders only do quick file I/O and the OS
    releases the lock when a process dies, so waiting needs no timeout.
    Reentrant within a thread, so a locked caller can call add_post and the like.
    Never await while holding it: the lock blocks the thread, not the coroutine.

    Raises:
        ValueError: org_id is not a slugify() id.
    """
    lock_path = _org_dir(org_id) / LOCK_FILE_NAME
    held: set[Path] = _held_locks.__dict__.setdefault("paths", set())
    if lock_path in held:
        yield
        return
    with open(lock_path, "w") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)  # released when the file closes
        held.add(lock_path)
        try:
            yield
        finally:
            held.discard(lock_path)


def _under_org_lock(function: Callable[..., Any]) -> Callable[..., Any]:
    """Run a function whose first argument is an org id under locked(org_id)."""

    @functools.wraps(function)
    def wrapper(org_id: str, *args: Any, **kwargs: Any) -> Any:
        with locked(org_id):
            return function(org_id, *args, **kwargs)

    return wrapper


def load_profile(org_id: str) -> dict[str, Any]:
    """
    Load the org profile from disk, returning a default skeleton if none exists.

    Args:
        org_id: Filesystem-safe org identifier.

    Returns:
        Dict with keys: org_id, name, website, research, preferences,
        preference_notes, voice_examples, post_ledger. Keys missing from an
        older or hand-edited file are filled in with their defaults.

    Raises:
        ProfileError: profile.json is not valid JSON, is not a JSON object, or
            one of its sections has the wrong type (e.g. a post_ledger that is
            not a list of posts). The message says what is wrong and points at
            the profile.json.bak backup.
    """
    path = _profile_path(org_id)
    profile: dict[str, Any] = {
        "org_id": org_id,
        "name": "",
        "website": "",
        "research": None,
        "preferences": {},
        "preference_notes": {},
        "voice_examples": [],
        "post_ledger": [],
    }
    if not path.exists():
        return profile
    restore_hint = (
        f"Fix it in a text editor, or replace it with the last good copy, {path.name}.bak."
    )
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError as e:
        raise ProfileError(
            f"{path} has a formatting error at line {e.lineno}, column {e.colno} ({e.msg}). {restore_hint}"
        ) from e
    if not isinstance(data, dict):
        raise ProfileError(
            f"{path} must be a JSON object ({{ ... }}) at the top level. {restore_hint}"
        )
    if problem := _section_problem(data):
        raise ProfileError(f"{path}: {problem}. {restore_hint}")
    profile.update(data)
    return profile


def _section_problem(data: dict[str, Any]) -> str:
    """
    Describe the first section of a loaded profile with the wrong type, or return "".

    Every turn reads these sections, so a hand edit such as "preferences": null
    would otherwise fail later with an error that names no file.
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
    return ""


def save_profile(org_id: str, profile: dict[str, Any]) -> None:
    """Write the org profile to disk, keeping the previous version as profile.json.bak.

    The new content is written to a temporary file and swapped in with
    os.replace, so a crash mid-write never leaves a half-written profile.

    Callers that loaded the profile first hold locked(org_id) across both
    steps, so another process can't write in between. The temporary file is
    named per process, so even an unlocked writer can't take another's.

    Args:
        org_id: Filesystem-safe org identifier.
        profile: Full profile dict as returned by load_profile.
    """
    path = _profile_path(org_id)
    tmp_path = path.with_suffix(f".json.{os.getpid()}.tmp")
    tmp_path.write_text(json.dumps(profile, indent=2))
    if path.exists():
        shutil.copyfile(path, path.with_suffix(".json.bak"))
    os.replace(tmp_path, path)


@_under_org_lock
def fill_org_details(org_id: str, name: str, website: str = "") -> None:
    """
    Record the org's name and website where the profile has none yet.

    What is already there is kept, so a hand-corrected name is never
    overwritten by what someone typed at onboarding.
    """
    profile = load_profile(org_id)
    if profile["name"] and (profile["website"] or not website):
        return
    profile["name"] = profile["name"] or name
    profile["website"] = profile["website"] or website
    save_profile(org_id, profile)


@_under_org_lock
def save_preference(
    org_id: str, key: str, value: str, quote: str = "", append: bool = False
) -> None:
    """Persist a brand preference for the org, with a note of where it came from.

    preferences[key] stays a plain value (a string, or a list when append is
    used) so profile.json remains easy to read and edit; the date and the
    user's own words are stored beside it in preference_notes[key].

    Args:
        org_id: Filesystem-safe org identifier.
        key:    Preference name (e.g. "voice", "banned_topics").
        value:  Preference value.
        quote:  The user's words that expressed the preference, if any.
        append: Add value to a list under key (e.g. one more banned topic)
                instead of replacing what is there.
    """
    profile = load_profile(org_id)
    prefs = profile.setdefault("preferences", {})
    if append:
        current = prefs.get(key, [])
        items = current if isinstance(current, list) else [current]
        prefs[key] = items + [value] if value not in items else items
    else:
        prefs[key] = value
    profile.setdefault("preference_notes", {})[key] = {
        "learned_at": datetime.now().isoformat(timespec="seconds"),
        "quote": quote,
    }
    save_profile(org_id, profile)


@_under_org_lock
def forget_preference(org_id: str, key: str, value: str | None = None) -> bool:
    """
    Remove a learned preference, or one item from a list-valued preference.

    Args:
        org_id: Filesystem-safe org identifier.
        key:    Preference name.
        value:  If given, remove only this item of a list preference, or the
                preference itself only if its value matches; both ignore case.

    Returns:
        True if something was removed; False if the key (or value) was not found.
    """
    profile = load_profile(org_id)
    prefs = profile["preferences"]
    if key not in prefs:
        return False
    if value is not None and isinstance(prefs[key], list):
        match = next(
            (item for item in prefs[key] if str(item).casefold() == value.casefold()), None
        )
        if match is None:
            return False
        prefs[key].remove(match)
        if prefs[key]:
            save_profile(org_id, profile)
            return True
    elif value is not None and str(prefs[key]).casefold() != value.casefold():
        return False
    del prefs[key]
    profile["preference_notes"].pop(key, None)
    save_profile(org_id, profile)
    return True


@_under_org_lock
def add_voice_example(org_id: str, content: str, source: str) -> None:
    """
    Remember a post that shows the org's real voice, keeping the newest few.

    Voice examples come from posts the org approved, wrote themselves, or
    rewrote from a draft; the prompt shows them as the style to imitate.

    Args:
        org_id:  Filesystem-safe org identifier.
        content: The post text.
        source:  Where it came from, e.g. "approved post_004" or "pasted by user".
    """
    profile = load_profile(org_id)
    examples = profile["voice_examples"]
    examples.append(
        {
            "content": content,
            "source": source,
            "added_at": datetime.now().isoformat(timespec="seconds"),
        }
    )
    profile["voice_examples"] = examples[-MAX_VOICE_EXAMPLES:]
    save_profile(org_id, profile)


def _next_post_id(ledger: list[dict[str, Any]]) -> str:
    """
    Return the next free post ID, one above the highest existing number.

    Counting entries would reuse an ID after a non-engineer deletes a post
    from profile.json, so the highest existing number is used instead.
    """
    numbers = [
        int(match.group(1))
        for post in ledger
        if (match := re.fullmatch(r"post_(\d+)", str(post.get("id", ""))))
    ]
    return f"post_{max(numbers, default=0) + 1:03d}"


@_under_org_lock
def add_post(org_id: str, post: dict[str, Any]) -> str:
    """
    Append a post to the org's ledger, assigning an ID and timestamp.

    Args:
        org_id: Filesystem-safe org identifier.
        post:   Post dict with at minimum a 'platform' and 'content' key.
                'status' defaults to "suggested" if not provided.

    Returns:
        The assigned post ID string (e.g. "post_001").
    """
    profile = load_profile(org_id)
    post_id = _next_post_id(profile["post_ledger"])
    post["id"] = post_id
    post["created_at"] = datetime.now().isoformat()
    post.setdefault("status", "suggested")
    profile["post_ledger"].append(post)
    save_profile(org_id, profile)
    return post_id


@_under_org_lock
def update_post_status(org_id: str, post_id: str, status: str) -> bool:
    """
    Update the status of an existing ledger entry.

    Args:
        org_id:  Filesystem-safe org identifier.
        post_id: ID of the post to update (e.g. "post_001").
        status:  New status, one of POST_STATUSES.

    Returns:
        True if the post was found and updated; False if post_id does not exist.
    """
    profile = load_profile(org_id)
    for post in profile["post_ledger"]:
        if post.get("id") == post_id:
            post["status"] = status
            save_profile(org_id, profile)
            return True
    return False


@_under_org_lock
def record_performance(org_id: str, post_id: str, metrics: dict[str, int], note: str = "") -> bool:
    """
    Store the results the user reported for a published post, and mark it posted.

    A post only has results once it is live, so its status becomes "posted".
    Reporting again replaces the earlier numbers: platforms show running
    totals, so the latest report is the most complete.

    Args:
        org_id:  Filesystem-safe org identifier.
        post_id: ID of the post the results are for.
        metrics: Counts keyed by names from PERFORMANCE_METRICS.
        note:    Anything else the user said, e.g. "brought in 3 donations".

    Returns:
        True if the post was found and updated; False if post_id does not exist.
    """
    profile = load_profile(org_id)
    for post in profile["post_ledger"]:
        if post.get("id") == post_id:
            post["performance"] = {
                **metrics,
                **({"note": note} if note else {}),
                "recorded_at": datetime.now().isoformat(timespec="seconds"),
            }
            post["status"] = "posted"
            save_profile(org_id, profile)
            return True
    return False


def engagement(post: dict[str, Any]) -> int:
    """Total engagement (reactions, comments, shares, clicks, saves) recorded for a post."""
    performance = post.get("performance")
    if not isinstance(performance, dict):
        return 0
    # isinstance check: a hand-edited count such as "40" or null must not break every turn.
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


def search_ledger(
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
        org_id:   Filesystem-safe org identifier.
        status:   Exact status, e.g. "draft".
        platform: Exact platform id, e.g. "instagram" (case-insensitive).
        since:    ISO date "YYYY-MM-DD"; posts created on or after it.
        until:    ISO date "YYYY-MM-DD"; posts created on or before it.
        text:     Case-insensitive substring of the title or content.

    Returns:
        Matching post dicts, oldest first.
    """

    def matches(post: dict[str, Any]) -> bool:
        created = str(post.get("created_at", ""))[:10]
        searchable = f"{post.get('title', '')} {post.get('content', '')}".lower()
        return (
            (status is None or post.get("status") == status)
            and (platform is None or str(post.get("platform", "")).lower() == platform.lower())
            and (since is None or created >= since)
            and (until is None or created <= until)
            and (text is None or text.lower() in searchable)
        )

    return [post for post in get_ledger(org_id) if matches(post)]


def get_ledger(org_id: str) -> list[dict[str, Any]]:
    """Return the full post ledger list for an org.

    Args:
        org_id: Filesystem-safe org identifier.

    Returns:
        List of post dicts; empty list if no posts have been saved.
    """
    return load_profile(org_id)["post_ledger"]


def trace_path(org_id: str) -> Path:
    """Return the path to the JSONL tool-call trace file for an org."""
    return _org_dir(org_id) / "trace.jsonl"
