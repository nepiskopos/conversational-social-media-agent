"""
Persistent storage for org profiles, post ledgers, and preferences.

All org state is stored as JSON on disk at data/{org_id}/profile.json, so a
non-engineer can read and edit what the assistant has learned in any text
editor, with no database setup or migrations.

Consequence: the load-modify-save pattern used by save_preference, add_post,
and update_post_status is not atomic. Two concurrent sessions writing to the
same org profile race, and the last write wins. This is acceptable for one
user per org at a time; a multi-user deployment needs a database with
row-level locking.

Owns: reading and writing profile.json for each org (atomic writes with a
profile.json.bak copy, a readable error for hand-editing mistakes), assigning
post IDs, searching the ledger, learned preferences with their provenance,
voice examples, and returning file paths for trace output.
Does not own LLM interaction, tool dispatch, or research.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

DATA_DIR: Path = Path(__file__).parent.parent / "data"
MAX_VOICE_EXAMPLES: int = 5
POST_STATUSES: tuple[str, ...] = ("suggested", "draft", "approved", "posted", "planned", "rejected")
ORG_ID_PATTERN: re.Pattern[str] = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")


class ProfileError(Exception):
    """profile.json exists but cannot be read — usually a hand-editing mistake."""


def slugify(text: str) -> str:
    """
    Convert a display name to a filesystem-safe org ID ([a-z0-9-] only).

    A name with no Latin letters or digits (e.g. Greek or Japanese) gets a
    stable "org-" + hash ID instead of an empty one, which would put every
    such org in the same data/ folder. Returns "" only for a blank name.
    """
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    if slug or not text.strip():
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

    No concurrency control: two writers for the same org race and the last
    one wins (see the module docstring).

    Args:
        org_id: Filesystem-safe org identifier.
        profile: Full profile dict as returned by load_profile.
    """
    path = _profile_path(org_id)
    tmp_path = path.with_suffix(".json.tmp")
    tmp_path.write_text(json.dumps(profile, indent=2))
    if path.exists():
        shutil.copyfile(path, path.with_suffix(".json.bak"))
    os.replace(tmp_path, path)


def save_preference(
    org_id: str, key: str, value: str, quote: str = "", append: bool = False
) -> None:
    """Persist a brand preference for the org, with a note of where it came from.

    preferences[key] stays a plain value (a string, or a list when append is
    used) so profile.json remains easy to read and edit; the date and the
    user's own words are stored beside it in preference_notes[key].
    Uses a load-modify-save pattern. Not atomic — see module docstring.

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


def forget_preference(org_id: str, key: str, value: str | None = None) -> bool:
    """
    Remove a learned preference, or one item from a list-valued preference.

    Args:
        org_id: Filesystem-safe org identifier.
        key:    Preference name.
        value:  If given, remove only this item of a list preference, or the
                preference itself only if its value matches (ignoring case).

    Returns:
        True if something was removed; False if the key (or value) was not found.
    """
    profile = load_profile(org_id)
    prefs = profile["preferences"]
    if key not in prefs:
        return False
    if value is not None and isinstance(prefs[key], list):
        if value not in prefs[key]:
            return False
        prefs[key].remove(value)
        if prefs[key]:
            save_profile(org_id, profile)
            return True
    elif value is not None and str(prefs[key]).casefold() != value.casefold():
        return False
    del prefs[key]
    profile["preference_notes"].pop(key, None)
    save_profile(org_id, profile)
    return True


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


def add_post(org_id: str, post: dict[str, Any]) -> str:
    """
    Append a post to the org's ledger, assigning an ID and timestamp.

    Uses a load-modify-save pattern. Not atomic — see module docstring.

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
