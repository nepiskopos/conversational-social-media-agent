"""
Unit tests for agent/memory.py.

Owns: slugify; profiles (defaults, saving and replacing all or nothing, post
ids, a lossless round trip, times across time zones, profile_problem and
read_profile_file for a hand-edited export); text and numbers PostgreSQL
can't store; research; the post ledger, its statuses and its search; post
results, engagement and top_performing; the agenda; preferences; voice
examples; fill_org_details; and the tool-call trace. Does not own concurrent
writes to one org (test_sessions.py), row-level security and SQL injection
(test_security.py) or the import and export scripts (test_admin_scripts.py).
"""

import json
import os
import time
import unicodedata
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import pytest

# An exported time: local, with its offset, so it means one moment wherever it is imported.
NINE_AM = datetime(2026, 10, 1, 9).astimezone().isoformat()


@pytest.mark.parametrize(
    "name, org_id",
    [
        ("BRCA Strong", "brca-strong"),
        ("Myra's Kids", "myra-s-kids"),
        ("Vets & Drones", "vets-drones"),
        ("--Leading Dashes--", "leading-dashes"),
        # Accented letters are separators, not dropped: the id stays ASCII.
        ("Café Ñoño", "caf-o-o"),
        # Compatibility characters count as the letters they stand for.
        ("ﬁsh Club", "fish-club"),
        ("Tokyo ２ Org", "tokyo-2-org"),
    ],
)
def test_slugify_makes_an_id_from_a_name(name, org_id):
    from agent.memory import slugify

    assert slugify(name) == org_id


def test_slugify_gives_non_latin_names_distinct_stable_ids():
    from agent.memory import slugify

    greek, japanese = slugify("Ελληνική Ένωση"), slugify("日本")
    assert greek.startswith("org-") and japanese.startswith("org-")
    assert greek != japanese
    assert slugify("Ελληνική Ένωση") == greek
    # Regression: Latin letters beside them used to be the whole id, so these were one org.
    mixed = {slugify(name) for name in ("日本 Org", "中国 Org", "Org")}
    assert len(mixed) == 3 and slugify("日本 Org").startswith("org-")
    assert slugify("Ένωση 2024").startswith("2024-")
    # One org however the name is spaced, punctuated or Unicode-normalised.
    assert slugify("日本  Org.") == slugify("日本 Org")
    assert slugify(unicodedata.normalize("NFD", "Ελληνική Ένωση")) == greek
    # Ids stored before stay as they were: Latin letters separate, other scripts hash the same.
    assert (slugify("Røde Kors"), slugify("Straße e.V.")) == ("r-de-kors", "stra-e-e-v")
    assert greek == "org-f6adf79fa7"
    for name, stored in (
        ("Σύλλογος «Ελπίδα»", "org-c6b590d102"),  # punctuation: the name as typed, as before
        ("Hawaiʻi Foodbank", "hawai-i-foodbank"),  # a modifier letter is no other script
    ):
        assert slugify(name) == stored


@pytest.mark.parametrize("name", ["!!!", "  ", "—", "🎉🎉", "-_-", "word " * 21])
def test_slugify_gives_no_id_to_a_name_without_letters_or_digits_or_too_long(name):
    """Nor to a pasted paragraph: its id didn't fit the database index, which looked like an outage."""
    from agent.memory import slugify

    assert slugify(name) == ""


async def test_load_profile_returns_defaults_for_new_org(database):
    from agent.memory import default_profile, load_profile

    assert await load_profile("new-org") == default_profile("new-org")


async def test_save_profile_replaces_everything_stored_for_the_org(database):
    from agent.memory import default_profile, load_profile, save_profile

    await save_profile(
        "org",
        {
            "name": "Old",
            "research": {"summary": "x"},
            "preferences": {"voice": "warm"},
            "voice_examples": [{"content": "Hi", "source": "pasted"}],
            "post_ledger": [{"platform": "linkedin", "content": "x"}],
        },
    )
    await save_profile("org", {"name": "New"})
    assert await load_profile("org") == default_profile("org") | {"name": "New"}


async def test_save_profile_gives_posts_without_an_id_the_next_unused_one(database):
    from agent.memory import add_post, get_ledger, save_profile

    for i in range(3):
        await add_post("org", {"platform": "linkedin", "content": f"post {i}"})
    # Every post is removed, then three come back, one numbered by hand.
    await save_profile(
        "org",
        {
            "post_ledger": [
                {"title": "Typed by hand"},
                {"id": "post_010", "content": "numbered"},
                {"title": "Another"},
            ]
        },
    )
    assert [p["id"] for p in await get_ledger("org")] == ["post_011", "post_010", "post_012"]
    assert await add_post("org", {"platform": "linkedin", "content": "new"}) == "post_013"


async def test_add_post_does_not_reuse_id_after_deletion(database):
    from agent.memory import add_post, load_profile, save_profile

    for i in range(3):
        await add_post("org", {"platform": "linkedin", "content": f"post {i}"})
    # A non-engineer deletes post_003 from an exported profile and imports it again.
    profile = await load_profile("org")
    profile["post_ledger"] = [p for p in profile["post_ledger"] if p["id"] != "post_003"]
    await save_profile("org", profile)

    assert await add_post("org", {"platform": "linkedin", "content": "new"}) == "post_004"


async def test_save_profile_refuses_a_problem_and_changes_nothing(database):
    from agent.memory import ProfileError, load_profile, save_profile

    await save_profile("org", {"name": "Kept", "post_ledger": [{"content": "x"}]})
    before = await load_profile("org")
    with pytest.raises(ProfileError, match="status 'live'"):
        await save_profile("org", {"name": "Lost", "post_ledger": [{"status": "live"}]})
    assert await load_profile("org") == before


async def test_save_profile_is_all_or_nothing_when_the_database_refuses_part(database, monkeypatch):
    """A failure after the old rows are deleted must not leave the org half-replaced."""
    from psycopg.errors import DataError

    from agent import memory
    from agent.memory import load_profile, save_profile

    await save_profile(
        "org",
        {
            "name": "A",
            "preferences": {"voice": "warm"},
            "voice_examples": [{"content": "Hi", "source": "pasted", "added_at": NINE_AM}],
            "post_ledger": [{"content": "kept", "created_at": NINE_AM}],
        },
    )
    before = await load_profile("org")

    async def refuse(*args, **kwargs):
        raise DataError("value out of range")

    monkeypatch.setattr(memory, "_insert_posts", refuse)
    with pytest.raises(DataError):
        await save_profile("org", {"name": "B", "post_ledger": [{"content": "lost"}]})
    assert await load_profile("org") == before


def _rich_profile() -> dict:
    """A profile using every section and nested values, for the round-trip test."""
    return {
        "name": "Food Bank",
        "website": "https://food.example",
        "research": {
            "summary": "Feeds 4,000 families",
            "sources": [{"url": "https://food.example/about", "facts": ["a", {"b": [1, 2.5]}]}],
        },
        "org_brief": "Brief",
        "research_updated_at": "2026-10-01",
        "content_pillars": [{"name": "Impact", "examples": ["meals served"]}],
        "preferences": {
            "voice": "warm",
            "banned_topics": ["politics", "religion"],
            "audience": "donors",
        },
        "preference_notes": {
            "voice": {"learned_at": NINE_AM, "quote": "be warm"},
            "audience": {"learned_at": "", "quote": "talk to donors"},
        },
        "voice_examples": [
            {"content": f"Example {i}", "source": "pasted by user", "added_at": NINE_AM}
            for i in range(3)
        ],
        "post_ledger": [
            {
                "id": "post_001",
                "platform": "linkedin",
                "content": "Hello",
                "status": "posted",
                "created_at": datetime(2026, 10, 1, 9, 30, 15, 123456).astimezone().isoformat(),
                "details": {"carousel": [{"slide": 1, "text": "x"}], "alt": None},
                "performance": {"reach": 900, "reactions": 40, "note": "2 donations"},
            },
            {
                "id": "post_002",
                "platform": "instagram",
                "content": "Hello again",
                "status": "planned",
                "parent_id": "post_001",
                "scheduled_for": "2026-10-26",
                "created_at": NINE_AM,
            },
        ],
    }


async def test_save_profile_round_trip_is_lossless_and_stable(database):
    """Export, import and export again gives the same file, preferences in their saved order."""
    from agent.memory import load_profile, save_profile

    await save_profile("org", _rich_profile())
    first = await load_profile("org")
    t1 = json.dumps(first)
    await save_profile("org", json.loads(t1))
    t2 = json.dumps(await load_profile("org"))
    assert t1 == t2
    assert list(first["preferences"]) == ["voice", "banned_topics", "audience"]
    expected = _rich_profile()
    expected["preference_notes"]["audience"]["learned_at"] = ""
    assert first == {"org_id": "org", **expected}


@pytest.fixture
def set_zone():
    """Set the process's local time zone for the test; restore the run's (conftest sets it)."""
    saved = os.environ["TZ"]

    def set_zone(name: str) -> None:
        os.environ["TZ"] = name
        time.tzset()

    yield set_zone
    set_zone(saved)


async def test_export_and_import_in_another_time_zone_keep_the_same_moments(database, set_zone):
    from agent.memory import (
        add_post,
        add_voice_example,
        load_profile,
        save_preference,
        save_profile,
    )

    set_zone("UTC")
    await add_post("org", {"platform": "linkedin", "content": "x"})
    await save_preference("org", "voice", "warm", quote="be warm")
    await add_voice_example("org", "Hi", "pasted by user")
    exported = json.loads(json.dumps(await load_profile("org")))

    set_zone("Asia/Tokyo")
    await save_profile("org", exported)
    set_zone("UTC")
    imported = await load_profile("org")

    def moments(profile: dict) -> list[datetime]:
        times = [
            profile["post_ledger"][0]["created_at"],
            profile["preference_notes"]["voice"]["learned_at"],
            profile["voice_examples"][0]["added_at"],
        ]
        return [datetime.fromisoformat(t) for t in times]

    assert all(moment.tzinfo for moment in moments(imported))
    assert moments(imported) == moments(exported)


async def test_a_time_without_an_offset_is_read_as_the_importing_zone(database, set_zone):
    """A time without an offset (an older export, a hand edit) is read as local time."""
    from agent.memory import load_profile, save_profile

    set_zone("Asia/Tokyo")
    await save_profile("org", {"post_ledger": [{"created_at": "2026-10-01T09:00:00"}]})
    set_zone("UTC")
    created = (await load_profile("org"))["post_ledger"][0]["created_at"]
    tokyo_nine_am = datetime(2026, 10, 1, 9, tzinfo=ZoneInfo("Asia/Tokyo"))
    assert datetime.fromisoformat(created) == tokyo_nine_am
    assert created == "2026-10-01T00:00:00+00:00"


@pytest.mark.parametrize(
    "profile, expected",
    [
        ({"post_ledger": ["post_001"]}, "post_ledger must be a list of { ... } entries"),
        ({"post_ledger": {"id": "post_001"}}, "post_ledger must be a list"),
        ({"voice_examples": {"content": "x"}}, "voice_examples must be a list"),
        ({"voice_examples": ["just text"]}, "voice_examples must be a list"),
        ({"preferences": None}, "preferences must be a { ... } object"),
        ({"preferences": ["warm"]}, "preferences must be a { ... } object"),
        ({"preference_notes": "x"}, "preference_notes must be a { ... } object"),
        ({"research": "done"}, "research must be a { ... } object, or null"),
        ({"content_pillars": {"name": "x"}}, "content_pillars must be a list"),
        (
            {"post_ledger": [{"id": "spring-campaign"}]},
            "post 1 has id 'spring-campaign'; ids look like post_001",
        ),
        (
            {"post_ledger": [{"id": "post_001"}, {"id": "post_001"}]},
            "two posts have the id post_001",
        ),
        ({"post_ledger": [{}, {"status": "live"}]}, "post 2 has status 'live'; use one of"),
        (
            {"post_ledger": [{"created_at": "yesterday"}]},
            "post 1 has created_at 'yesterday', which is not a date and time",
        ),
        # Years PostgreSQL or the local zone can't show: imported, they'd make load_profile fail.
        *[
            (
                {"post_ledger": [{"created_at": time}]},
                f"post 1 has created_at {time!r}, which is not",
            )
            for time in (
                "0001-01-01T00:00:00",
                "9999-12-31T23:59:59",
                "9999-12-31T23:59:59-14:00",
                "0001-01-01T00:00:00+14:00",
            )
        ],
        # Python's \d matches any script's digits; the database's CHECK takes only 0-9.
        ({"post_ledger": [{"id": "post_\u0663"}]}, "ids look like post_001"),
        (
            {"post_ledger": [{"scheduled_for": "next week"}]},
            "post 1 has scheduled_for 'next week'; use YYYY-MM-DD",
        ),
        (
            {"research": {"sources": [{"url": "https://a.org"}, {"url": 5}]}},
            "research source 2 has url 5; it must be text",
        ),
        (
            {"post_ledger": [{"id": "post_99999999999"}]},
            "post 1 has id 'post_99999999999'; the number is too large",
        ),
        (
            {"post_ledger": [{"parent_id": ["post_009"]}]},
            "post 1 has parent_id ['post_009']; it must be a post id",
        ),
        (
            {"post_ledger": [{"parent_id": {"x": 1}}]},
            "post 1 has parent_id {'x': 1}; it must be a post id",
        ),
        ({"research": {"sources": 5}}, "research sources must be a list"),
        (
            {"preferences": {"voice": "warm"}, "preference_notes": {"voice": "be warm"}},
            "the note for preference 'voice' must be a { ... } object",
        ),
        (
            {"preference_notes": {"voice": {"learned_at": "last week"}}},
            "preference 'voice' has learned_at 'last week', which is not a date and time",
        ),
        (
            {"voice_examples": [{"content": "Hi", "added_at": "yesterday"}]},
            "voice example 1 has added_at 'yesterday', which is not a date and time",
        ),
        # Regression: each of these passed, then broke every turn or every calendar.
        (
            {"research": {"sources": [{"type": ["news"], "url": "https://x.org"}]}},
            "research source 1 has type ['news']; it must be text",
        ),
        ({"research": {"website_content": 12}}, "research website_content must be text"),
        ({"name": ["Org"]}, "name must be text"),
        ({"post_ledger": [{"title": 7}]}, "post 1 has title 7; it must be text"),
        ({"voice_examples": [{"content": {"x": 1}}]}, "voice example 1: its content and source"),
        ({"preferences": {"voice": {"tone": "warm"}}}, "preference 'voice' must be text"),
    ],
)
def test_profile_problem_names_each_mistake(profile, expected):
    """Every turn reads these sections, and the database constrains these post fields."""
    from agent.memory import profile_problem

    assert expected in profile_problem(profile)


@pytest.mark.parametrize(
    "profile",
    [
        {},
        {"research": None, "content_pillars": None},
        {"post_ledger": [{"title": "only a title", "performance": "lots"}]},
        {"post_ledger": [{"id": "post_007", "status": "posted", "created_at": "2026-10-01"}]},
        {"preferences": {"limit": 3, "banned_topics": ["politics", "religion"]}},
    ],
)
def test_profile_problem_accepts_a_valid_profile(profile):
    """The README tells users to set "research": null to research the org again."""
    from agent.memory import profile_problem

    assert profile_problem(profile) == ""


def test_read_profile_file_returns_a_valid_profile(tmp_path):
    from agent.memory import read_profile_file

    path = tmp_path / "profile.json"
    path.write_text('{"name": "Org", "research": null}')
    assert read_profile_file(path) == {"name": "Org", "research": None}


@pytest.mark.parametrize(
    "text, expected",
    [
        ('{\n  "name": "x",\n}', "formatting error at line 3"),
        ("[]", "must be a JSON object"),
        # The rest of the checks are profile_problem's: one case shows they are applied.
        ('{"post_ledger": ["post_001"]}', "post_ledger must be a list"),
        # Python's JSON reader accepts these; PostgreSQL's jsonb refuses them.
        ('{"research": {"score": NaN}}', "NaN is not a number PostgreSQL can store"),
        ('{"research": {"score": -Infinity}}', "-Infinity is not a number PostgreSQL can store"),
        ('{"research": {"score": 1e999}}', "1e999 is too large a number to store"),
        # An editor that saved it as Latin-1.
        ('{"name": "Caf\u00e9"}'.encode("latin-1"), "is not saved as UTF-8 text"),
    ],
)
def test_read_profile_file_explains_hand_editing_mistakes(tmp_path, text, expected):
    from agent.memory import ProfileError, read_profile_file

    path = tmp_path / "profile.json"
    if isinstance(text, bytes):
        path.write_bytes(text)
    else:
        path.write_text(text)
    with pytest.raises(ProfileError) as excinfo:
        read_profile_file(path)
    assert expected in str(excinfo.value)
    assert str(path) in str(excinfo.value)


async def test_save_research_stores_research_brief_and_when_it_ran(database):
    from agent.memory import load_profile, save_research

    await save_research("org", {"summary": "first"}, "Brief 1")
    await save_research("org", {"summary": "second"}, "Brief 2")
    profile = await load_profile("org")
    assert profile["research"] == {"summary": "second"}
    assert profile["org_brief"] == "Brief 2"
    stamped = datetime.fromisoformat(profile["research_updated_at"])
    assert stamped.tzinfo is not None
    assert abs(stamped - datetime.now().astimezone()) < timedelta(minutes=5)


async def test_update_research_saves_only_when_the_change_says_so(database):
    from agent.memory import load_profile, save_research, update_research

    await save_research("org", {"sources": ["a"]}, "Brief")

    def add_source(research):
        research["sources"].append("b")
        return True

    def change_without_saving(research):
        research["sources"].append("c")
        return False

    await update_research("org", add_source)
    await update_research("org", change_without_saving)
    assert (await load_profile("org"))["research"] == {"sources": ["a", "b"]}


async def test_update_research_does_nothing_for_an_org_without_research(database):
    from agent.memory import load_profile, update_research

    calls = []
    await update_research("org", lambda research: calls.append(research) or True)
    assert calls == []
    assert (await load_profile("org"))["research"] is None


async def test_add_post_numbers_posts_and_fills_in_status_and_time(database):
    from agent.memory import add_post, get_ledger

    ids = [
        await add_post("org", {"platform": "linkedin", "content": "Hello"}),
        await add_post("org", {"platform": "instagram", "content": "x", "status": "draft"}),
        await add_post("org", {"platform": "facebook", "content": "x"}),
    ]
    assert ids == ["post_001", "post_002", "post_003"]
    first, second, _ = await get_ledger("org")
    assert first["status"] == "suggested"
    assert datetime.fromisoformat(first["created_at"]).tzinfo
    assert second["status"] == "draft"


@pytest.mark.parametrize("bad, stored", [("\x00", ""), ("\ud800", "?")])
async def test_characters_postgresql_cant_store_are_replaced_on_save(database, bad, stored):
    """
    PostgreSQL text can't hold NUL or a lone surrogate (half an emoji); a model's
    arguments or a JSON file sometimes contain one, and the save must not fail.
    """
    from agent.memory import (
        add_post,
        forget_preference,
        load_profile,
        save_preference,
        save_profile,
    )

    await save_profile(
        "org",
        {
            "name": f"Food{bad} Bank",
            "preferences": {f"tone{bad}": [f"calm{bad}"]},
            "post_ledger": [{"content": f"x{bad}y"}],
        },
    )
    await add_post("org", {"platform": "linkedin", "content": f"a{bad}b", f"title{bad}": f"t{bad}"})
    await save_preference("org", "voice", f"warm{bad}", quote=f"be{bad} warm")
    for topic in ("a", "b"):
        await save_preference("org", f"topics{bad}", f"{topic}{bad}", append=True)
    # Forgetting must find what was stored, though the stored text differs from what was sent.
    assert await forget_preference("org", f"topics{bad}", f"a{bad}") is True
    assert await forget_preference("org", f"tone{bad}") is True
    profile = await load_profile("org")
    assert profile["name"] == f"Food{stored} Bank"
    assert f"tone{stored}" not in profile["preferences"]
    assert profile["preferences"][f"topics{stored}"] == [f"b{stored}"]
    first, second = profile["post_ledger"]
    assert first["content"] == f"x{stored}y"
    assert (second["content"], second[f"title{stored}"]) == (f"a{stored}b", f"t{stored}")
    assert profile["preferences"]["voice"] == f"warm{stored}"
    assert profile["preference_notes"]["voice"]["quote"] == f"be{stored} warm"


async def test_numbers_jsonb_cant_store_are_saved_as_null(database):
    """A model or hand edit can produce infinity or NaN; jsonb refuses both."""
    from agent.memory import load_profile, save_preference, save_profile

    await save_profile("org", {"preferences": {"limits": [1.5, float("inf")]}})
    await save_preference("org", "score", float("nan"))
    preferences = (await load_profile("org"))["preferences"]
    assert preferences == {"limits": [1.5, None], "score": None}


async def test_update_post_status_returns_the_old_status_or_none_for_no_post(database):
    from agent.memory import add_post, get_ledger, update_post_status

    post_id = await add_post("org", {"platform": "linkedin", "content": "Hello"})
    assert await update_post_status("org", post_id, "approved") == "suggested"
    assert (await get_ledger("org"))[0]["status"] == "approved"
    assert await update_post_status("org", post_id, "approved") == "approved"
    assert await update_post_status("org", "post_999", "approved") is None


async def test_update_post_status_refuses_an_unknown_status_and_keeps_the_post(database):
    """The database's CHECK on status is the last line behind the tools' validation."""
    from psycopg.errors import CheckViolation

    from agent.memory import add_post, get_post, update_post_status

    post_id = await add_post("org", {"platform": "linkedin", "content": "x", "status": "draft"})
    before = await get_post("org", post_id)
    with pytest.raises(CheckViolation):
        await update_post_status("org", post_id, "live")
    assert await get_post("org", post_id) == before


async def test_hand_edited_post_without_fields_does_not_break_ledger_operations(database):
    from agent.memory import add_post, get_post, save_profile, search_ledger, update_post_status

    await save_profile("org", {"post_ledger": [{"title": "Typed by hand", "performance": "lots"}]})
    post = await get_post("org", "post_001")
    assert (post["platform"], post["content"], post["status"]) == ("", "", "suggested")
    assert post["performance"] == "lots"
    assert await search_ledger("org", platform="linkedin", since="2026-01-01") == []
    assert await update_post_status("org", "post_001", "draft") == "suggested"
    assert (await get_post("org", "post_001"))["status"] == "draft"
    assert await add_post("org", {"platform": "linkedin", "content": "x"}) == "post_002"


async def test_get_post_returns_the_post_or_none(database):
    from agent.memory import add_post, get_post

    post_id = await add_post("org", {"platform": "linkedin", "content": "Hello", "title": "Hi"})
    post = await get_post("org", post_id)
    assert (post["id"], post["content"], post["title"]) == (post_id, "Hello", "Hi")
    assert await get_post("org", "post_404") is None


async def test_get_ledger_empty_for_new_org(database):
    from agent.memory import get_ledger

    assert await get_ledger("new-org") == []


async def test_save_preference_persists_with_quote_and_date(database):
    from agent.memory import load_profile, save_preference

    await save_preference("org", "voice", "warm and grassroots", quote="Too corporate")
    profile = await load_profile("org")
    assert profile["preferences"]["voice"] == "warm and grassroots"
    note = profile["preference_notes"]["voice"]
    assert note["quote"] == "Too corporate"
    assert note["learned_at"]


async def test_save_preference_overwrites_existing(database):
    from agent.memory import load_profile, save_preference

    await save_preference("org", "voice", "corporate")
    await save_preference("org", "audience", "donors")
    await save_preference("org", "voice", "grassroots")
    # Overwriting keeps the preference where it was, so the prompt doesn't reshuffle.
    assert (await load_profile("org"))["preferences"] == {
        "voice": "grassroots",
        "audience": "donors",
    }
    assert list((await load_profile("org"))["preferences"]) == ["voice", "audience"]


async def test_save_preference_append_builds_list_without_duplicates(database):
    from agent.memory import load_profile, save_preference

    await save_preference("org", "banned_topics", "politics")  # stored as a plain string first
    await save_preference("org", "banned_topics", "religion", append=True)
    await save_preference("org", "banned_topics", "religion", append=True)
    assert (await load_profile("org"))["preferences"]["banned_topics"] == ["politics", "religion"]


async def test_save_preference_refusal_sees_every_preference_and_saves_nothing(database):
    from agent.memory import load_profile, save_preference

    await save_preference("org", "voice", "warm", quote="be warm")
    seen = []

    def refuse(prefs, quotes):
        seen.append((prefs, quotes))
        return "too long"

    assert await save_preference("org", "tone", "plain", refuse=refuse) == "too long"
    assert seen == [({"voice": "warm"}, {"voice": "be warm"})]
    assert await save_preference("org", "tone", "plain", refuse=lambda *_: "") == ""
    assert (await load_profile("org"))["preferences"] == {"voice": "warm", "tone": "plain"}


@pytest.mark.parametrize(
    "current, value, expected",
    [([], "a", ["a"]), ("a", "b", ["a", "b"]), (["a", "b"], "a", ["a", "b"])],
)
def test_appended_adds_a_value_once_and_turns_a_plain_value_into_a_list(current, value, expected):
    from agent.memory import appended

    assert appended(current, value) == expected


async def test_forget_preference_removes_key_and_note(database):
    from agent.memory import forget_preference, load_profile, save_preference

    await save_preference("org", "voice", "formal", quote="be formal")
    assert await forget_preference("org", "voice") is True
    profile = await load_profile("org")
    assert "voice" not in profile["preferences"]
    assert "voice" not in profile["preference_notes"]


async def test_forget_preference_removes_single_list_item_ignoring_case(database):
    from agent.memory import forget_preference, load_profile, save_preference

    await save_preference("org", "banned_topics", "Politics", append=True)
    await save_preference("org", "banned_topics", "religion", append=True)
    assert await forget_preference("org", "banned_topics", "politics") is True
    assert (await load_profile("org"))["preferences"]["banned_topics"] == ["religion"]
    assert await forget_preference("org", "banned_topics", "religion") is True
    assert "banned_topics" not in (await load_profile("org"))["preferences"]


async def test_forget_preference_returns_false_when_nothing_matches(database):
    from agent.memory import forget_preference, save_preference

    assert await forget_preference("org", "voice") is False
    await save_preference("org", "banned_topics", "politics", append=True)
    assert await forget_preference("org", "banned_topics", "sports") is False


async def test_forget_preference_keeps_a_string_preference_when_the_value_differs(database):
    from agent.memory import forget_preference, load_profile, save_preference

    await save_preference("org", "voice", "warm and grassroots")
    assert await forget_preference("org", "voice", "corporate") is False
    assert (await load_profile("org"))["preferences"]["voice"] == "warm and grassroots"
    assert await forget_preference("org", "voice", "Warm and Grassroots") is True
    assert "voice" not in (await load_profile("org"))["preferences"]


async def test_add_voice_example_keeps_newest(database):
    from agent.memory import MAX_VOICE_EXAMPLES, add_voice_example, load_profile

    for i in range(MAX_VOICE_EXAMPLES + 2):
        await add_voice_example("org", f"post {i}", "pasted by user")
    examples = (await load_profile("org"))["voice_examples"]
    assert len(examples) == MAX_VOICE_EXAMPLES
    assert examples[0]["content"] == "post 2"
    assert examples[-1]["content"] == f"post {MAX_VOICE_EXAMPLES + 1}"
    assert examples[0]["source"] == "pasted by user"


async def test_add_voice_example_keeps_one_copy_of_a_text(database):
    """Regression: approved, reopened and approved again, a post pushed out another example."""
    from agent.memory import add_voice_example, load_profile

    for source in ("pasted by user", "approved post_001", "approved post_001"):
        await add_voice_example("org", "Our post", source)
    examples = (await load_profile("org"))["voice_examples"]
    assert [(e["content"], e["source"]) for e in examples] == [("Our post", "pasted by user")]


def _post(post_id: str, created_at: str, **fields) -> dict:
    """A ledger post for the search tests."""
    return {
        "id": post_id,
        "platform": "linkedin",
        "status": "draft",
        "content": "",
        "created_at": created_at,
        **fields,
    }


async def _ids(**filters) -> list[str]:
    """The ids search_ledger finds in org "org"."""
    from agent.memory import search_ledger

    return [p["id"] for p in await search_ledger("org", **filters)]


async def test_search_ledger_combines_filters(database):
    from agent.memory import save_profile

    ledger = [
        _post("post_001", "2026-09-28T10:00:00", platform="instagram", status="approved"),
        _post("post_002", "2026-10-02T10:00:00", platform="instagram", title="Meet Ana"),
        _post("post_003", "2026-10-05T10:00:00", content="Annual report"),
    ]
    await save_profile("org", {"post_ledger": ledger})

    assert await _ids(status="draft") == ["post_002", "post_003"]
    assert await _ids(platform="Instagram") == ["post_001", "post_002"]
    assert await _ids(text="meet ana") == ["post_002"]  # title is searched too
    assert await _ids(text="ANNUAL") == ["post_003"]
    assert await _ids(status="draft", platform="linkedin") == ["post_003"]
    assert await _ids() == ["post_001", "post_002", "post_003"]


async def test_search_ledger_since_and_until_include_their_whole_day(database):
    from agent.memory import save_profile

    ledger = [
        _post("post_001", "2026-09-30T23:59:59"),
        _post("post_002", "2026-10-01T00:00:00"),
        _post("post_003", "2026-10-02T23:59:59"),
        _post("post_004", "2026-10-03T00:00:00"),
    ]
    await save_profile("org", {"post_ledger": ledger})
    assert await _ids(since="2026-10-01", until="2026-10-02") == ["post_002", "post_003"]
    assert await _ids(since="2026-10-03") == ["post_004"]
    assert await _ids(until="2026-09-30") == ["post_001"]


async def test_search_ledger_matches_percent_and_underscore_literally(database):
    from agent.memory import save_profile

    ledger = [
        _post("post_001", "2026-10-01T10:00:00", content="50% off tickets"),
        _post("post_002", "2026-10-01T10:00:00", content="500 off tickets"),
        _post("post_003", "2026-10-01T10:00:00", content="the a_b test"),
        _post("post_004", "2026-10-01T10:00:00", content="the axb test"),
    ]
    await save_profile("org", {"post_ledger": ledger})
    assert await _ids(text="50%") == ["post_001"]
    assert await _ids(text="a_b") == ["post_003"]


async def test_search_ledger_rejects_a_date_that_is_not_yyyy_mm_dd(database):
    from agent.memory import search_ledger

    with pytest.raises(ValueError):
        await search_ledger("org", since="last week")


async def test_record_performance_stores_results_and_marks_posted_or_finds_no_post(database):
    from agent.memory import add_post, get_ledger, record_performance

    await add_post("org", {"platform": "linkedin", "content": "x", "status": "approved"})
    assert await record_performance(
        "org", "post_001", {"reach": 900, "reactions": 40}, "2 donations"
    )
    post = (await get_ledger("org"))[0]
    assert post["status"] == "posted"
    assert post["performance"]["reach"] == 900
    assert post["performance"]["note"] == "2 donations"
    assert "recorded_at" in post["performance"]
    assert await record_performance("org", "post_404", {"reactions": 1}) is False


async def test_record_performance_replaces_earlier_results(database):
    from agent.memory import add_post, get_ledger, record_performance

    await add_post("org", {"platform": "linkedin", "content": "x"})
    await record_performance("org", "post_001", {"reactions": 10}, "early")
    await record_performance("org", "post_001", {"reactions": 50})
    performance = (await get_ledger("org"))[0]["performance"]
    assert performance["reactions"] == 50
    assert "note" not in performance


def test_engagement_sums_engagement_metrics_but_not_reach():
    from agent.memory import engagement

    post = {"performance": {"reach": 1000, "reactions": 30, "comments": 5, "shares": 2}}
    assert engagement(post) == 37


@pytest.mark.parametrize(
    "performance",
    [None, "lots", {"reactions": "40"}, {"reactions": None}, {"reactions": True}],
)
def test_engagement_tolerates_hand_edited_results(performance):
    from agent.memory import engagement

    assert engagement({"performance": performance}) == 0


def test_top_performing_ranks_measured_posts_and_skips_the_rest():
    from agent.memory import top_performing

    ledger = [
        {"id": "post_001", "performance": {"reactions": 5}},
        {"id": "post_002"},
        {"id": "post_003", "performance": {"reactions": 50}},
        {"id": "post_004", "performance": {"shares": 20}},
    ]
    assert [p["id"] for p in top_performing(ledger, 2)] == ["post_003", "post_004"]
    assert top_performing([{"id": "post_002"}], 5) == []


def test_top_performing_keeps_ledger_order_for_equal_engagement():
    from agent.memory import top_performing

    ledger = [
        {"id": "post_001", "performance": {"reactions": 10}},
        {"id": "post_002", "performance": {"shares": 30}},
        {"id": "post_003", "performance": {"comments": 10}},
        {"id": "post_004", "performance": {"saves": 10}},
    ]
    ranked = [p["id"] for p in top_performing(ledger, 4)]
    assert ranked == ["post_002", "post_001", "post_003", "post_004"]


def test_agenda_groups_posts_by_next_step():
    """Planned posts are due 7 days ahead inclusive; results are asked for 3 days after posting."""
    from agent.memory import agenda

    today = date(2026, 10, 19)
    ledger = [
        {"id": "post_001", "status": "planned", "scheduled_for": "2026-10-26"},
        {"id": "post_002", "status": "planned", "scheduled_for": "2026-10-27"},
        {"id": "post_003", "status": "draft", "platform": "linkedin"},
        {"id": "post_004", "status": "approved", "platform": "instagram"},
        {"id": "post_005", "status": "posted", "scheduled_for": "2026-10-16"},
        {"id": "post_006", "status": "posted", "scheduled_for": "2026-10-17"},
        {"id": "post_007", "status": "posted", "scheduled_for": "2026-10-01", "performance": {}},
    ]
    groups = agenda(ledger, today)
    assert [p["id"] for p in groups["to_draft"]] == ["post_001"]
    assert [p["id"] for p in groups["awaiting_approval"]] == ["post_003"]
    assert [p["id"] for p in groups["to_publish"]] == ["post_004"]
    assert [p["id"] for p in groups["needs_results"]] == ["post_005"]


def test_agenda_skips_drafts_replaced_on_the_same_platform_only():
    """A rejected rewrite doesn't replace its original: that draft still awaits approval."""
    from agent.memory import agenda

    ledger = [
        {"id": "post_001", "status": "draft", "platform": "linkedin"},
        {"id": "post_002", "status": "draft", "platform": "linkedin", "parent_id": "post_001"},
        {"id": "post_003", "status": "draft", "platform": "instagram", "parent_id": "post_002"},
        {"id": "post_004", "status": "rejected", "platform": "instagram", "parent_id": "post_003"},
    ]
    waiting = agenda(ledger, date(2026, 10, 19))["awaiting_approval"]
    assert [p["id"] for p in waiting] == ["post_002", "post_003"]


def test_agenda_is_empty_for_an_empty_or_hand_edited_ledger():
    from agent.memory import agenda

    assert not any(agenda([], date(2026, 10, 19)).values())
    assert not any(agenda([{"id": "post_001"}], date(2026, 10, 19)).values())


async def test_fill_org_details_sets_a_new_orgs_name_and_website(database):
    from agent.memory import fill_org_details, load_profile

    await fill_org_details("food-bank", "Food Bank", "https://food.example")
    profile = await load_profile("food-bank")
    assert (profile["name"], profile["website"]) == ("Food Bank", "https://food.example")


async def test_fill_org_details_keeps_what_the_profile_already_has(database):
    from agent.memory import fill_org_details, load_profile, save_profile

    await save_profile("food-bank", {"name": "Food Bank of Ohio"})
    await fill_org_details("food-bank", "food bank", "https://food.example")
    await fill_org_details("food-bank", "Other", "https://other.example")
    profile = await load_profile("food-bank")
    assert (profile["name"], profile["website"]) == ("Food Bank of Ohio", "https://food.example")


async def test_append_trace_and_read_trace_keep_calls_in_order(database):
    from agent.memory import append_trace, read_trace

    entries = [
        {
            "ts": 1_790_000_000.25 + i,
            "tool": tool,
            "input": {"query": f"q{i}"},
            "result_preview": f"result {i}",
            "latency_ms": 10 * i,
        }
        for i, tool in enumerate(("search_web", "save_post"))
    ]
    for entry in entries:
        await append_trace("org", entry)
    assert await read_trace("org") == [{**e, "ts": pytest.approx(e["ts"])} for e in entries]


async def test_read_trace_is_empty_for_an_org_without_calls(database):
    from agent.memory import read_trace

    assert await read_trace("org") == []


@pytest.mark.parametrize(
    "value, expected",
    [
        ("2026-11-03", True),
        ("2026-02-30", False),  # no such day
        ("20261103", False),  # Python reads these, but they'd be stored and compared as written
        ("2026-W45", False),
        (20261103, False),
        (None, False),
    ],
)
def test_is_date_takes_only_a_yyyy_mm_dd_date(value, expected):
    from agent.memory import is_date

    assert is_date(value) is expected


@pytest.mark.parametrize(
    "value, expected", [(0, True), (40, True), (True, False), (4.0, False), ("4", False)]
)
def test_is_count_takes_only_a_whole_number(value, expected):
    from agent.memory import is_count

    assert is_count(value) is expected
