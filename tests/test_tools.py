"""
Unit tests for agent/tools.py.

Owns: each tool's handler; the save_post checks (platform and format rules,
visuals, parent ids, citing a source) and their re-check before a post goes
live; argument validation (statuses, dates, counts, required arguments, null
or empty optional ones); the preference size caps; research, its summary and
the org brief; web_search and the research sources; the content calendar
(weeks, dates, cadence, formats, parsing the model's reply); and that
handle_tool_call answers instead of raising and traces each call. Does not own
the fuzz of every tool, injection and the trace's fields (test_security.py),
parallel calendar weeks and linear-time checks (test_performance.py) or
concurrent writes (test_sessions.py).
"""

import json
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent.tools import MAX_PREFERENCE_KEY_CHARS, MAX_PREFERENCE_TEXT_CHARS
from tests.llm_fakes import completion

_EMPTY_RESEARCH = {
    "org_name": "Test Org",
    "website_content": "",
    "social_presence": [],
    "news": [],
    "similar_orgs": [],
    "sources": [],
}


def _calendar_reply(content: dict) -> MagicMock:
    """A completion response whose message is the given calendar week as JSON."""
    return completion(json.dumps(content))


async def test_save_post_returns_the_post_id_and_saves_a_suggestion_by_default(database):
    from agent.memory import get_ledger
    from agent.tools import handle_tool_call

    result = await handle_tool_call(
        "save_post", {"platform": "LinkedIn", "content": "Hello #a #b #c", "title": "T"}, "test-org"
    )
    assert "post_001" in result
    assert (await get_ledger("test-org"))[0]["status"] == "suggested"
    assert (await get_ledger("test-org"))[0]["platform"] == "linkedin"  # one spelling per platform


STORY = {
    "platform": "instagram",
    "content": "Frame 1: Hi",
    "title": "T",
    "format": "story",
    "status": "draft",
    "visual": "Club photo",
}
FACEBOOK_DRAFT = {
    "platform": "facebook",
    "content": "Come to the drive #food",
    "title": "Drive",
    "status": "draft",
}


@pytest.mark.parametrize(
    "args, expected, saved",
    [
        pytest.param(
            {"platform": "instagram", "title": "Long", "content": "x" * 2201, "status": "draft"},
            "Not saved: 2,201 characters exceeds Instagram's 2,200-character limit",
            False,
            id="over-the-length-limit",
        ),
        pytest.param(
            {"platform": "linkedin", "title": "t", "content": "Hello #one", "status": "draft"},
            "Not saved: 1 hashtag(s); LinkedIn posts need 3-5",
            False,
            id="too-few-hashtags",
        ),
        pytest.param(
            {
                "platform": "linkedin",
                "title": "t",
                "content": "Hi #one #two #three",
                "status": "draft",
            },
            "Saved to ledger as post_001",
            True,
            id="hashtags-fixed",
        ),
        pytest.param(
            {"platform": "instagram", "content": "x #a #b #c", "title": "T", "format": "hologram"},
            "Not saved: format 'hologram' is not one of",
            False,
            id="unknown-format",
        ),
        pytest.param(
            {"platform": "linkedin", "content": "x #a #b #c", "title": "T", "format": "story"},
            "Not saved: Story isn't a format on LinkedIn",
            False,
            id="format-not-on-the-platform",
        ),
        pytest.param(
            {"platform": "email", "content": "Subject: Hi", "title": "T", "format": "feed_post"},
            "Not saved: Feed Post isn't a format on Email",
            False,
            id="email-as-a-feed-post",
        ),
        # Instagram captions need 10-20 hashtags, but a story has no caption; a reel does.
        pytest.param(STORY, "Saved", True, id="story-needs-no-hashtags"),
        pytest.param(
            {**STORY, "format": "reel"}, "Not saved: 0 hashtag(s)", False, id="reel-needs-hashtags"
        ),
        # Ideas skip the rules, so the model spends no rounds polishing them; going live re-checks.
        *(
            pytest.param(
                {"platform": "instagram", "content": "Idea", "title": "T", "status": status},
                "Saved",
                True,
                id=f"{status}-idea-skips-the-rules",
            )
            for status in ("suggested", "planned")
        ),
        # A reply can show a Visual line; the draft itself must carry it.
        pytest.param(
            FACEBOOK_DRAFT,
            "Not saved: Facebook posts need a visual",
            False,
            id="facebook-draft-without-a-visual",
        ),
        pytest.param(
            {**FACEBOOK_DRAFT, "visual": "Volunteers loading boxes"},
            "Saved",
            True,
            id="facebook-draft-with-a-visual",
        ),
        pytest.param(
            {**FACEBOOK_DRAFT, "platform": "linkedin", "content": "Drive #a #b #c"},
            "Saved",
            True,
            id="linkedin-needs-no-visual",
        ),
        pytest.param(
            {"platform": "general", "title": "t", "content": "x" * 5000},
            "Saved to ledger as post_001 (status: suggested)",
            True,
            id="no-rules-for-an-unknown-platform",
        ),
    ],
)
async def test_save_post_applies_the_platform_and_format_rules(database, args, expected, saved):
    from agent.memory import get_ledger
    from agent.tools import handle_tool_call

    result = await handle_tool_call("save_post", args, "test-org")
    assert result.startswith(expected)
    ledger = await get_ledger("test-org")
    assert len(ledger) == saved
    if saved:
        assert ledger[0]["content"] == args["content"]
        assert ledger[0].get("visual") == args.get("visual")


@pytest.mark.parametrize(
    "platform, status, expected, kept",
    [
        (
            "instagram",
            "approved",
            "Not updated: post_001 can't be approved yet: 0 hashtag(s)",
            "suggested",
        ),
        (
            "instagram",
            "posted",
            "Not updated: post_001 can't be posted yet: 0 hashtag(s)",
            "suggested",
        ),
        ("general", "approved", "Updated post_001 → approved", "approved"),  # no skill file
    ],
)
async def test_going_live_re_checks_the_stored_post(database, platform, status, expected, kept):
    """Ideas skip the rules when saved, so approving or publishing re-checks the stored text."""
    from agent.memory import add_post, get_ledger
    from agent.tools import handle_tool_call

    await add_post(
        "test-org", {"platform": platform, "content": "Rough idea", "status": "suggested"}
    )
    result = await handle_tool_call(
        "update_post_status", {"post_id": "post_001", "status": status}, "test-org"
    )
    assert result.startswith(expected)
    assert ("parent_id post_001" in result) == (kept == "suggested")  # how to fix it
    assert (await get_ledger("test-org"))[0]["status"] == kept
    assert (
        await handle_tool_call(
            "update_post_status", {"post_id": "post_001", "status": "draft"}, "test-org"
        )
    ).startswith("Updated")


@pytest.mark.parametrize(
    "tool, args",
    [
        ("update_post_status", {"post_id": "post_999", "status": "approved"}),
        ("record_performance", {"post_id": "post_999", "reactions": 1}),
    ],
)
async def test_tools_report_an_unknown_post_id(database, tool, args):
    from agent.tools import handle_tool_call

    assert await handle_tool_call(tool, args, "test-org") == "Post post_999 not found in ledger"


@pytest.mark.parametrize(
    "results, expected",
    [
        (
            {"reactions": 120, "shares": 14, "note": "3 new volunteers"},
            "Recorded results for post_001 (120 reactions, 14 shares); status → posted",
        ),
        ({"note": "raised $400"}, "Recorded results for post_001 (raised $400); status → posted"),
    ],
    ids=["counts-and-note", "note-alone"],
)
async def test_record_performance_tool_saves_results(database, results, expected):
    from agent.memory import add_post, get_ledger
    from agent.tools import handle_tool_call

    await add_post("test-org", {"platform": "facebook", "content": "x", "status": "approved"})
    result = await handle_tool_call(
        "record_performance", {"post_id": "post_001", **results}, "test-org"
    )
    assert result == expected
    post = (await get_ledger("test-org"))[0]
    assert post["status"] == "posted"
    assert {name: post["performance"].get(name) for name in results} == results


@pytest.mark.parametrize(
    "unstorable, stored",
    [
        ({"content": "half an emoji \ud83d ARG-MARKER"}, {"content": "half an emoji ? ARG-MARKER"}),
        ({"content": "ARG-MARKER", "reach": float("inf")}, {"reach": None}),
    ],
)
async def test_a_tool_argument_postgresql_cant_store_is_saved_and_traced(
    database, unstorable, stored
):
    """Such a value is replaced (by "?" or null) so neither the save nor the trace fails."""
    from agent.memory import get_ledger, read_trace
    from agent.tools import handle_tool_call

    arguments = {"platform": "general", "title": "T", **unstorable}
    result = await handle_tool_call("save_post", arguments, "test-org")
    assert result.startswith("Saved to ledger as post_001")
    [post] = await get_ledger("test-org")
    assert "ARG-MARKER" in post["content"]
    [entry] = await read_trace("test-org")
    assert entry["tool"] == "save_post"
    assert {name: entry["input"].get(name) for name in stored} == stored


async def test_save_preference_tool_sets_a_value_then_appends_with_quotes(database):
    from agent.memory import load_profile
    from agent.tools import handle_tool_call

    await handle_tool_call(
        "save_preference", {"key": "voice", "value": "warm", "quote": "be warm"}, "test-org"
    )
    for topic in ("politics", "religion"):
        await handle_tool_call(
            "save_preference",
            {"key": "banned_topics", "value": topic, "quote": f"never {topic}", "append": True},
            "test-org",
        )
    profile = await load_profile("test-org")
    assert profile["preferences"]["voice"] == "warm"
    assert profile["preferences"]["banned_topics"] == ["politics", "religion"]
    assert profile["preference_notes"]["banned_topics"]["quote"] == "never religion"


async def test_research_org_tool_saves_to_profile_and_reminds_the_model_to_save(database):
    """Models tend to answer straight after research without saving; the result reminds them."""
    from agent.memory import load_profile
    from agent.tools import RESEARCH_NEXT_STEP, handle_tool_call

    mock_research = {
        "org_name": "Test NPO",
        "website_content": "We help kids.",
        "social_presence": [],
        "news": [],
        "similar_orgs": [],
        "sources": [{"type": "website", "url": "https://testnpo.org"}],
    }
    with patch("agent.tools.res.research_org", new=AsyncMock(return_value=mock_research)):
        result = await handle_tool_call(
            "research_org",
            {"org_name": "Test NPO", "org_url": "https://testnpo.org"},
            "test-org",
        )

    profile = await load_profile("test-org")
    assert profile["research"] is not None
    assert profile["name"] == "Test NPO"
    assert profile["website"] == "https://testnpo.org"
    assert result.endswith(RESEARCH_NEXT_STEP)
    assert "save_post" in RESEARCH_NEXT_STEP


async def test_research_org_tool_does_not_overwrite_existing_name_or_website(database):
    from agent.memory import load_profile, save_profile
    from agent.tools import handle_tool_call

    await save_profile(
        "test-org",
        {
            "org_id": "test-org",
            "name": "Original Name",
            "website": "https://original.com",
            "research": None,
            "preferences": {},
            "post_ledger": [],
        },
    )
    with patch("agent.tools.res.research_org", new=AsyncMock(return_value=_EMPTY_RESEARCH)):
        await handle_tool_call(
            "research_org",
            {"org_name": "Different Name", "org_url": "https://different.com"},
            "test-org",
        )

    profile = await load_profile("test-org")
    assert profile["name"] == "Original Name"
    assert profile["website"] == "https://original.com"
    assert profile["research"] is not None  # research itself IS updated


async def test_web_search_tool_formats_results(database):
    from agent.tools import handle_tool_call

    mock_results = [{"title": "NPO News", "url": "https://example.com", "snippet": "Great work"}]
    with patch("agent.tools.res.search_web", new=AsyncMock(return_value=mock_results)):
        result = await handle_tool_call("web_search", {"query": "nonprofit news"}, "test-org")
    assert "NPO News" in result
    assert "example.com" in result


async def test_refresh_research_replaces_the_research_and_keeps_everything_else(database):
    from agent.memory import add_post, load_profile, save_profile
    from agent.tools import RESEARCH_NEXT_STEP, handle_tool_call

    await save_profile(
        "test-org",
        {
            "org_id": "test-org",
            "name": "Test NPO",
            "website": "",
            "research": {"website_content": "old content"},
            "preferences": {"voice": "warm"},
            "post_ledger": [],
        },
    )
    await add_post("test-org", {"platform": "linkedin", "content": "Existing post"})
    new_research = {**_EMPTY_RESEARCH, "website_content": "new content"}
    with patch("agent.tools.res.research_org", new=AsyncMock(return_value=new_research)):
        result = await handle_tool_call("refresh_research", {"org_name": "Test NPO"}, "test-org")

    profile = await load_profile("test-org")
    assert profile["research"]["website_content"] == "new content"
    # When it ran, to the second; default_profile's empty string would mean it was never stamped.
    stamped = datetime.fromisoformat(profile["research_updated_at"])
    assert abs(stamped.astimezone().replace(tzinfo=None) - datetime.now()) < timedelta(minutes=5)
    assert profile["preferences"]["voice"] == "warm"
    assert len(profile["post_ledger"]) == 1
    # A refresh doesn't come before suggestions, so it gets no save reminder.
    assert RESEARCH_NEXT_STEP not in result


def test_format_research_summary_names_the_org_and_caps_the_website_text():
    from agent.tools import _format_research_summary

    result = {**_EMPTY_RESEARCH, "website_content": "A" * 3000 + "B" * 2000}
    summary = _format_research_summary("Test NPO", result)
    assert "Research complete" in summary and "Test NPO" in summary
    assert "A" * 3000 in summary
    assert "B" not in summary


def test_format_research_summary_limits_and_filters_results():
    from agent.tools import _format_research_summary

    social = [
        {"title": "Social A", "url": "https://a.com", "snippet": "About A"},
        {"title": "Social B", "url": "https://b.com", "snippet": "About B"},
        {"error": "fetch failed"},
        {"title": "Social D", "url": "https://d.com", "snippet": "About D"},
        {"title": "Social E", "url": "https://e.com", "snippet": "About E"},
    ]
    result = {
        "website_content": "",
        "social_presence": social,
        "news": [],
        "similar_orgs": [],
        "sources": [],
    }
    summary = _format_research_summary("Test NPO", result)
    assert "Social A" in summary
    assert "Social B" in summary
    assert "Social D" not in summary  # only the first 3 social results are shown
    assert "Social E" not in summary


async def test_generate_calendar_handles_partial_failure(database):
    from agent.memory import get_ledger, save_profile
    from agent.tools import handle_tool_call

    await save_profile(
        "test-org",
        {
            "org_id": "test-org",
            "name": "Test NPO",
            "website": "",
            "research": None,
            "preferences": {},
            "post_ledger": [],
        },
    )
    mock_ok = _calendar_reply(
        {
            "platform": "linkedin",
            "angle": "Community story",
            "pillar": "Community Impact",
            "rationale": "From research",
        }
    )
    with patch(
        "agent.tools.litellm.acompletion",
        new=AsyncMock(side_effect=[mock_ok, Exception("timeout"), mock_ok, mock_ok]),
    ):
        result = await handle_tool_call(
            "generate_calendar", {"month_year": "August 2026"}, "test-org"
        )

    assert "3 of 8 posts saved" in result
    assert "Error generating plan" in result
    assert len(await get_ledger("test-org")) == 3


async def test_handle_tool_call_returns_error_string_instead_of_raising(database):
    from agent.tools import handle_tool_call

    with patch("agent.tools.mem.update_post_status", side_effect=KeyError("status")):
        result = await handle_tool_call(
            "update_post_status", {"post_id": "post_001", "status": "draft"}, "test-org"
        )
    assert result.startswith("Error: update_post_status failed")
    assert "KeyError" in result


async def test_handle_tool_call_traces_failed_calls(database):
    from agent.memory import read_trace
    from agent.tools import handle_tool_call

    await handle_tool_call("save_post", {}, "test-org")
    [entry] = await read_trace("test-org")
    assert entry["tool"] == "save_post"
    assert entry["result_preview"].startswith("Error:")


def _profile_with_sources(*urls: str) -> dict:
    return {
        "org_id": "test-org",
        "name": "Test NPO",
        "website": "",
        "research": {**_EMPTY_RESEARCH, "sources": [{"type": "website", "url": u} for u in urls]},
        "preferences": {},
        "post_ledger": [],
    }


async def test_save_post_rejects_unknown_parent_id(database):
    from agent.memory import get_ledger
    from agent.tools import handle_tool_call

    result = await handle_tool_call(
        "save_post",
        {"platform": "general", "title": "t", "content": "a", "parent_id": "post_0489b2e1"},
        "test-org",
    )
    assert result.startswith("Not saved: parent_id post_0489b2e1 is not in the post ledger")
    assert await get_ledger("test-org") == []
    await handle_tool_call(
        "save_post", {"platform": "general", "title": "t", "content": "a"}, "test-org"
    )
    version = await handle_tool_call(
        "save_post",
        {"platform": "general", "title": "t2", "content": "b", "parent_id": "post_001"},
        "test-org",
    )
    assert version.startswith("Saved to ledger as post_002")


async def test_save_post_warns_when_source_url_is_not_a_research_source(database):
    from agent.memory import save_profile
    from agent.tools import handle_tool_call

    await save_profile("test-org", _profile_with_sources("https://www.test.org/programs/"))
    known = await handle_tool_call(
        "save_post",
        {
            "platform": "general",
            "title": "a",
            "content": "a",
            "source_url": "http://test.org/programs",
        },
        "test-org",
    )
    unknown = await handle_tool_call(
        "save_post",
        {
            "platform": "general",
            "title": "b",
            "content": "b",
            "source_url": "https://made-up.example/x",
        },
        "test-org",
    )
    assert "Warning" not in known
    assert "is not one of the research sources" in unknown


async def test_save_post_stores_optional_fields(database):
    from agent.memory import add_post, get_ledger
    from agent.tools import handle_tool_call

    await add_post("test-org", {"platform": "linkedin", "content": "First version"})
    result = await handle_tool_call(
        "save_post",
        {
            "platform": "instagram",
            "content": "Hi #a #b #c #d #e #f #g #h #i #j",
            "title": "Meet Ana",
            "pillar": "Community Impact",
            "content_type": "volunteer_call",
            "format": "carousel",
            "visual": "Ana at the pantry",
            "parent_id": "post_001",
            "scheduled_for": "2026-11-03",
            "source_url": "",
        },
        "test-org",
    )
    assert result.startswith("Saved")
    post = (await get_ledger("test-org"))[1]
    assert post["title"] == "Meet Ana"
    assert post["parent_id"] == "post_001"
    assert post["scheduled_for"] == "2026-11-03"
    assert post["content_type"] == "volunteer_call"
    assert post["format"] == "carousel"
    assert post["visual"] == "Ana at the pantry"
    assert "source_url" not in post  # empty optional fields are not stored


async def test_approving_a_post_saves_it_as_voice_example_once(database):
    from agent.memory import add_post, load_profile
    from agent.tools import handle_tool_call

    await add_post("test-org", {"platform": "linkedin", "content": "Our warm post #a #b #c"})
    for _ in range(2):  # approving it again must not add a second copy
        await handle_tool_call(
            "update_post_status", {"post_id": "post_001", "status": "approved"}, "test-org"
        )
    examples = (await load_profile("test-org"))["voice_examples"]
    assert examples == [
        {**examples[0], "content": "Our warm post #a #b #c", "source": "approved post_001"}
    ]


async def test_rejected_status_does_not_create_voice_example(database):
    from agent.memory import add_post, load_profile
    from agent.tools import handle_tool_call

    await add_post("test-org", {"platform": "instagram", "content": "Meh"})
    result = await handle_tool_call(
        "update_post_status", {"post_id": "post_001", "status": "rejected"}, "test-org"
    )
    assert "rejected" in result
    assert (await load_profile("test-org")).get("voice_examples", []) == []


@pytest.mark.parametrize(
    "field, limit",
    [
        ("key", MAX_PREFERENCE_KEY_CHARS),
        ("value", MAX_PREFERENCE_TEXT_CHARS),
        ("quote", MAX_PREFERENCE_TEXT_CHARS),
    ],
)
async def test_save_preference_refuses_an_overlong_key_value_or_quote(database, field, limit):
    """Preferences are in every prompt, so a pasted document or injected essay stays out."""
    from agent.memory import load_profile
    from agent.tools import handle_tool_call

    pref = {"key": "voice", "value": "warm", "quote": "be warm"}
    at_limit = await handle_tool_call("save_preference", {**pref, field: "x" * limit}, "test-org")
    assert at_limit.startswith("Preference saved")
    over = await handle_tool_call("save_preference", {**pref, field: "x" * (limit + 1)}, "test-org")
    assert over == (
        f"Not saved: the {field} is {limit + 1} characters; the limit is {limit}. "
        "Save the preference in a sentence or two."
    )
    assert len((await load_profile("test-org"))["preferences"]) == 1  # only the one at the limit


async def test_save_preference_refuses_a_list_that_appending_would_take_over_its_cap(database):
    """A text refused as too long can't be saved in parts under one key instead."""
    from agent.memory import load_profile, save_preference
    from agent.tools import MAX_PREFERENCE_LIST_CHARS, handle_tool_call

    await save_preference(
        "test-org", "style_guide", "a" * 400
    )  # a plain value counts as the first item
    part = {"key": "style_guide", "quote": "save it all", "append": True}
    for value in ("b" * 400, "c" * (MAX_PREFERENCE_LIST_CHARS - 800)):
        saved = await handle_tool_call("save_preference", {**part, "value": value}, "test-org")
        assert saved.startswith("Preference saved")
    over = await handle_tool_call("save_preference", {**part, "value": "d"}, "test-org")
    assert over == (
        f"Not saved: style_guide would be {MAX_PREFERENCE_LIST_CHARS + 1} characters in all; "
        f"the limit is {MAX_PREFERENCE_LIST_CHARS}. "
        "Tell the user, and offer to replace style_guide with a shorter version."
    )
    assert sum(map(len, (await load_profile("test-org"))["preferences"]["style_guide"])) == (
        MAX_PREFERENCE_LIST_CHARS
    )
    # An item already in the list adds nothing, and replacing the list is the way out.
    again = await handle_tool_call("save_preference", {**part, "value": "a" * 400}, "test-org")
    assert again.startswith("Preference saved")
    # A long value from an imported profile counts too.
    await save_preference("test-org", "voice", "v" * (MAX_PREFERENCE_LIST_CHARS - 100))
    added = await handle_tool_call(
        "save_preference", {**part, "key": "voice", "value": "w" * 101}, "test-org"
    )
    assert added.startswith("Not saved: voice would be")
    for item in ("e" * 500, "f" * 500, "g" * 500):  # a list typed past the cap stays usable
        await save_preference("test-org", "pasted", item, append=True)
    readded = await handle_tool_call(
        "save_preference", {**part, "key": "pasted", "value": "e" * 500}, "test-org"
    )
    assert readded.startswith("Preference saved")
    shorter = await handle_tool_call(
        "save_preference", {**part, "append": False, "value": "short"}, "test-org"
    )
    assert shorter.startswith("Preference saved")
    assert (await load_profile("test-org"))["preferences"]["style_guide"] == "short"


async def test_save_preference_refuses_what_would_take_all_preferences_over_their_cap(
    database,
):
    """A text refused as too long can't be split across keys, or moved into the quotes."""
    from agent.memory import load_profile, save_preference
    from agent.tools import MAX_PREFERENCES_TOTAL_CHARS, handle_tool_call

    quote = "q" * 10  # every preference below shows 400 characters with its quote
    for n in range(MAX_PREFERENCES_TOTAL_CHARS // 400):
        await save_preference("test-org", f"part_{n}", "x" * (389 if n == 0 else 390), quote=quote)
    pref = {"key": "part_0", "value": "x" * 390, "quote": quote}
    at_limit = await handle_tool_call("save_preference", pref, "test-org")
    assert at_limit.startswith("Preference saved")  # replaces part_0, so 4,000 in all
    over = await handle_tool_call(
        "save_preference", {**pref, "key": "part_extra", "value": "y"}, "test-org"
    )
    assert over == (
        f"Not saved: all preferences together, with their quotes, would be "
        f"{MAX_PREFERENCES_TOTAL_CHARS + 11} characters; the limit is {MAX_PREFERENCES_TOTAL_CHARS}. "
        "Tell the user, and offer to forget preferences that no longer apply or to shorten some."
    )
    longer_quote = await handle_tool_call(
        "save_preference", {**pref, "quote": quote + "q"}, "test-org"
    )
    assert longer_quote.startswith("Not saved: all preferences")
    # Replacing one with a value of the same length or shorter still works.
    same = await handle_tool_call("save_preference", {**pref, "value": "z" * 390}, "test-org")
    assert same.startswith("Preference saved")
    assert "part_extra" not in (await load_profile("test-org"))["preferences"]
    # In a profile imported past the cap, shortening is still allowed.
    await save_preference("test-org", "pasted", "p" * 500)
    shorter = await handle_tool_call(
        "save_preference", {**pref, "key": "pasted", "value": "brief"}, "test-org"
    )
    assert shorter.startswith("Preference saved")


async def test_forget_preference_tool(database):
    from agent.memory import save_preference
    from agent.tools import handle_tool_call

    await save_preference("test-org", "voice", "formal")
    assert await handle_tool_call("forget_preference", {"key": "voice"}, "test-org") == (
        "Forgot preference: voice"
    )
    assert "No saved preference" in await handle_tool_call(
        "forget_preference", {"key": "voice"}, "test-org"
    )


async def test_forget_preference_tool_removes_one_item_of_a_list(database):
    from agent.memory import load_profile, save_preference
    from agent.tools import handle_tool_call

    for topic in ("politics", "religion"):
        await save_preference("test-org", "banned_topics", topic, append=True)
    result = await handle_tool_call(
        "forget_preference", {"key": "banned_topics", "value": "politics"}, "test-org"
    )
    assert result == "Forgot preference: banned_topics (removed 'politics')"
    assert (await load_profile("test-org"))["preferences"]["banned_topics"] == ["religion"]


async def test_save_voice_example_tool(database):
    from agent.memory import load_profile
    from agent.tools import handle_tool_call

    await handle_tool_call(
        "save_voice_example", {"content": "Our own post", "note": "pasted by user"}, "test-org"
    )
    assert (await load_profile("test-org"))["voice_examples"][0]["content"] == "Our own post"


async def test_search_ledger_tool_formats_matches(database):
    from agent.memory import add_post
    from agent.tools import handle_tool_call

    await add_post(
        "test-org",
        {
            "platform": "instagram",
            "content": "x",
            "title": "Gala recap",
            "status": "draft",
            "scheduled_for": "2026-11-01",
        },
    )
    await add_post("test-org", {"platform": "linkedin", "content": "y", "status": "approved"})
    result = await handle_tool_call("search_ledger", {"status": "draft"}, "test-org")
    assert result.startswith("1 matching post(s):")
    assert "post_001" in result and "Gala recap" in result and "for 2026-11-01" in result
    assert "post_002" not in result
    assert (
        await handle_tool_call("search_ledger", {"platform": "tiktok"}, "test-org")
        == "No posts match."
    )


async def test_web_search_adds_results_to_research_sources(database):
    from agent.memory import load_profile, save_profile
    from agent.tools import handle_tool_call

    await save_profile("test-org", _profile_with_sources("https://test.org"))
    results = [
        {"title": "News", "url": "https://news.example/a", "snippet": "s"},
        {"title": "Home again", "url": "https://test.org/", "snippet": "s"},
    ]
    with patch("agent.tools.res.search_web", new=AsyncMock(return_value=results)):
        await handle_tool_call("web_search", {"query": "q"}, "test-org")
    urls = [s["url"] for s in (await load_profile("test-org"))["research"]["sources"]]
    assert urls == ["https://test.org", "https://news.example/a"]  # duplicate not re-added


async def test_web_search_with_only_failed_results_finds_nothing_and_adds_no_source(database):
    from agent.memory import load_profile, save_profile
    from agent.tools import handle_tool_call

    await save_profile("test-org", _profile_with_sources("https://test.org"))
    failed = AsyncMock(return_value=[{"error": "search failed"}])
    with patch("agent.tools.res.search_web", new=failed):
        result = await handle_tool_call("web_search", {"query": "q"}, "test-org")
    assert result == "No results found."
    sources = (await load_profile("test-org"))["research"]["sources"]
    assert [s["url"] for s in sources] == ["https://test.org"]


async def test_research_org_writes_org_brief(database):
    from agent.memory import load_profile
    from agent.tools import handle_tool_call

    research = {**_EMPTY_RESEARCH, "website_content": "We run a free tutoring program."}
    brief = completion("- Tutoring (https://test.org)")
    mock_completion = AsyncMock(return_value=brief)
    with (
        patch("agent.tools.res.research_org", new=AsyncMock(return_value=research)),
        patch("agent.tools.litellm.acompletion", new=mock_completion),
    ):
        await handle_tool_call("research_org", {"org_name": "Test Org"}, "test-org")

    assert (await load_profile("test-org"))["org_brief"] == "- Tutoring (https://test.org)"
    prompt = mock_completion.call_args.kwargs["messages"][0]["content"]
    assert "free tutoring program" in prompt
    assert "<external_content" in prompt


async def test_research_org_still_saves_research_when_brief_fails(database):
    from agent.memory import load_profile
    from agent.tools import handle_tool_call

    with (
        patch("agent.tools.res.research_org", new=AsyncMock(return_value=_EMPTY_RESEARCH)),
        patch("agent.tools.litellm.acompletion", new=AsyncMock(side_effect=Exception("down"))),
    ):
        result = await handle_tool_call("research_org", {"org_name": "Test Org"}, "test-org")
    profile = await load_profile("test-org")
    assert profile["research"] is not None
    assert profile["org_brief"] == ""
    assert result.startswith("Research complete")


async def test_research_org_tolerates_an_empty_brief_reply(database):
    # Reasoning models can spend the whole token budget thinking and return content=None.
    from agent.memory import load_profile
    from agent.tools import ORG_BRIEF_MAX_TOKENS, handle_tool_call

    empty = MagicMock(choices=[MagicMock(message=MagicMock(content=None))])
    mock_completion = AsyncMock(return_value=empty)
    with (
        patch("agent.tools.res.research_org", new=AsyncMock(return_value=_EMPTY_RESEARCH)),
        patch("agent.tools.litellm.acompletion", new=mock_completion),
    ):
        result = await handle_tool_call("research_org", {"org_name": "Test Org"}, "test-org")
    assert result.startswith("Research complete")
    assert (await load_profile("test-org"))["org_brief"] == ""
    assert mock_completion.call_args.kwargs["max_tokens"] == ORG_BRIEF_MAX_TOKENS >= 2000


async def test_org_brief_call_turns_down_reasoning_on_qwen(database):
    from agent.tools import handle_tool_call

    brief = completion("- Fact (https://test.org)")
    mock_completion = AsyncMock(return_value=brief)
    env = {
        "QWEN_API_KEY": "q",
        "QWEN_BASE_URL": "http://x/v1",
        "QWEN_MODEL": "",
        "LLM_PROVIDER": "qwen",
        "FAST_MODEL": "",
        "LLM_MODEL": "",
    }
    with (
        patch.dict("os.environ", env),
        patch("agent.tools.res.research_org", new=AsyncMock(return_value=_EMPTY_RESEARCH)),
        patch("agent.tools.litellm.acompletion", new=mock_completion),
    ):
        await handle_tool_call("research_org", {"org_name": "Test Org"}, "test-org")
    kwargs = mock_completion.call_args.kwargs
    assert kwargs["model"] == "hosted_vllm/qwen36-35b-a3b"
    assert kwargs["extra_body"] == {"chat_template_kwargs": {"enable_thinking": False}}


async def test_generate_calendar_uses_skill_platforms_and_org_pillars(database):
    from agent.memory import save_profile
    from agent.tools import handle_tool_call

    profile = _profile_with_sources()
    profile["content_pillars"] = [{"name": "Hope Stories", "description": "d"}]
    await save_profile("test-org", profile)
    week = _calendar_reply(
        {"platform": "instagram", "angle": "a", "pillar": "Hope Stories", "rationale": "r"}
    )
    mock_completion = AsyncMock(return_value=week)
    with patch("agent.tools.litellm.acompletion", new=mock_completion):
        await handle_tool_call("generate_calendar", {"month_year": "November 2026"}, "test-org")
    prompt = mock_completion.call_args.kwargs["messages"][0]["content"]
    assert '"facebook", "instagram", "linkedin"' in prompt
    assert '"Hope Stories"' in prompt
    assert "Community Impact" not in prompt


async def test_generate_calendar_names_an_org_without_a_name_generically(database):
    from agent.tools import handle_tool_call

    week = _calendar_reply({"platform": "instagram", "angle": "a", "pillar": "p", "rationale": "r"})
    mock_completion = AsyncMock(return_value=week)
    with patch("agent.tools.litellm.acompletion", new=mock_completion):
        await handle_tool_call("generate_calendar", {"month_year": "November 2026"}, "test-org")
    prompt = mock_completion.call_args.kwargs["messages"][0]["content"]
    assert "strategist for this organization." in prompt


@pytest.mark.parametrize(
    "tool, args",
    [
        (
            "save_post",
            {"platform": "linkedin", "content": "x", "title": "t", "status": "published"},
        ),
        ("update_post_status", {"post_id": "post_001", "status": "done"}),
        ("search_ledger", {"status": "live"}),
    ],
)
async def test_tools_reject_an_unknown_status(database, tool, args):
    from agent.memory import add_post, get_ledger
    from agent.tools import handle_tool_call

    await add_post("test-org", {"platform": "linkedin", "content": "x"})
    result = await handle_tool_call(tool, args, "test-org")
    assert result.startswith("Error: status")
    assert len(await get_ledger("test-org")) == 1
    assert (await get_ledger("test-org"))[0]["status"] == "suggested"


@pytest.mark.parametrize(
    "tool, args",
    [
        (
            "save_post",
            {"platform": "linkedin", "content": "x", "title": "t", "scheduled_for": "next Tuesday"},
        ),
        ("search_ledger", {"since": "last week"}),
        ("search_ledger", {"until": "2026-13-01"}),
    ],
)
async def test_tools_reject_a_date_not_in_iso_format(database, tool, args):
    from agent.memory import get_ledger
    from agent.tools import handle_tool_call

    result = await handle_tool_call(tool, args, "test-org")
    assert "is not a date in YYYY-MM-DD format" in result
    assert await get_ledger("test-org") == []


async def test_ledger_tools_tolerate_a_hand_edited_post_without_fields(database):
    from agent.memory import save_profile
    from agent.tools import handle_tool_call

    await save_profile("test-org", {"post_ledger": [{"id": "post_001"}]})
    assert "post_001 | " in await handle_tool_call("search_ledger", {}, "test-org")
    assert "Updated post_001" in await handle_tool_call(
        "update_post_status", {"post_id": "post_001", "status": "approved"}, "test-org"
    )


async def test_tools_tolerate_a_source_typed_into_profile_as_plain_text(database):
    """A source written by hand as a bare URL must not make save_post or web_search fail."""
    from agent.memory import load_profile, save_profile
    from agent.tools import handle_tool_call

    sources = ["https://typed-by-hand.example", {"type": "web", "url": "https://ok.example"}]
    await save_profile("test-org", {"research": {"sources": sources}})
    args = {"platform": "general", "content": "x", "title": "T", "source_url": "https://ok.example"}
    assert (await handle_tool_call("save_post", args, "test-org")).startswith("Saved")
    found = [{"title": "News", "url": "https://news.example", "snippet": "s"}]
    with patch("agent.tools.res.search_web", new=AsyncMock(return_value=found)):
        await handle_tool_call("web_search", {"query": "q"}, "test-org")
    assert (await load_profile("test-org"))["research"]["sources"][-1][
        "url"
    ] == "https://news.example"


async def test_optional_arguments_sent_as_null_or_empty_count_as_not_given(database):
    """Models fill every optional argument with null or ""; that must mean "not given"."""
    from agent.memory import add_post, get_ledger, load_profile
    from agent.tools import handle_tool_call

    await add_post("test-org", {"platform": "linkedin", "content": "x", "title": "Gala"})
    blank_filters = {"status": "", "platform": "", "since": "", "until": None, "text": ""}
    assert "1 matching post(s)" in await handle_tool_call(
        "search_ledger", blank_filters, "test-org"
    )

    post = {"platform": "general", "content": "x", "title": "T", "status": "", "scheduled_for": ""}
    assert (await handle_tool_call("save_post", post, "test-org")).startswith("Saved")
    assert (await get_ledger("test-org"))[-1]["status"] == "suggested"

    search = AsyncMock(return_value=[])
    with patch("agent.tools.res.search_web", new=search):
        await handle_tool_call("web_search", {"query": "q", "max_results": None}, "test-org")
    assert search.await_args.args == ("q", 5)

    plan = _calendar_reply({"angle": "a"})
    with patch("agent.tools.litellm.acompletion", new=AsyncMock(return_value=plan)):
        result = await handle_tool_call(
            "generate_calendar",
            {"month_year": "July 2026", "posts_per_week": None, "focus": None},
            "test-org",
        )
    assert "Focus" not in result and "posts saved to the ledger" in result

    with (
        patch("agent.tools.res.research_org", new=AsyncMock(return_value=_EMPTY_RESEARCH)),
        patch("agent.tools.litellm.acompletion", new=AsyncMock(side_effect=RuntimeError)),
    ):
        await handle_tool_call("research_org", {"org_name": "Org", "org_url": None}, "new-org")
    assert (await load_profile("new-org"))["website"] == ""


async def test_handle_tool_call_does_not_raise_when_the_trace_cannot_be_written(database):
    """The trace is a log: a failed write must not leave an unanswered tool call."""
    import psycopg

    from agent.tools import handle_tool_call

    failing = AsyncMock(side_effect=psycopg.OperationalError("server closed the connection"))
    with patch("agent.tools.mem.append_trace", new=failing):
        result = await handle_tool_call(
            "save_post", {"platform": "general", "content": "x", "title": "T"}, "test-org"
        )
    assert result.startswith("Saved to ledger as post_001")
    failing.assert_awaited_once()


@pytest.mark.parametrize("requested, sent", [(500, 10), (0, 1), (3, 3)])
async def test_web_search_caps_the_number_of_results(database, requested, sent):
    """Each result becomes a research source, so the model can't ask for an unbounded number."""
    from agent.tools import handle_tool_call

    search = AsyncMock(return_value=[])
    with patch("agent.tools.res.search_web", new=search):
        await handle_tool_call("web_search", {"query": "q", "max_results": requested}, "test-org")
    assert search.await_args.args == ("q", sent)


@pytest.mark.parametrize(
    "args, expected",
    [
        ({"post_id": "post_001"}, "give at least one count"),
        ({"post_id": "post_001", "reactions": -3}, "reactions -3 must be a whole number"),
        ({"post_id": "post_001", "reach": "1.2k"}, "reach '1.2k' must be a whole number"),
        ({"post_id": "post_001", "shares": True}, "shares True must be a whole number"),
    ],
)
async def test_record_performance_tool_rejects_bad_counts(database, args, expected):
    from agent.memory import add_post, get_ledger
    from agent.tools import handle_tool_call

    await add_post("test-org", {"platform": "linkedin", "content": "x"})
    result = await handle_tool_call("record_performance", args, "test-org")
    assert result.startswith("Error:") and expected in result
    assert "performance" not in (await get_ledger("test-org"))[0]


async def test_generate_calendar_dates_each_week_and_keeps_key_dates(database):
    from agent.memory import get_ledger
    from agent.tools import handle_tool_call

    week = {
        "date": "2026-12-01",
        "platform": "facebook",
        "angle": "Giving Tuesday thank-you",
        "pillar": "Events & Campaigns",
        "key_date": "Giving Tuesday",
        "rationale": "Biggest giving day",
    }
    with patch(
        "agent.tools.litellm.acompletion", new=AsyncMock(return_value=_calendar_reply(week))
    ):
        result = await handle_tool_call(
            "generate_calendar", {"month_year": "December 2026"}, "test-org"
        )

    ledger = await get_ledger("test-org")
    # Week 1 keeps the model's date; the other weeks don't contain Dec 1, so they get their start.
    assert [p["scheduled_for"] for p in ledger] == [
        "2026-12-01",
        "2026-12-08",
        "2026-12-15",
        "2026-12-22",
    ]
    assert ledger[0]["key_date"] == "Giving Tuesday"
    assert "| Giving Tuesday |" in result


async def test_generate_calendar_passes_focus_and_best_posts_to_each_week(database):
    from agent.memory import add_post, record_performance
    from agent.tools import handle_tool_call

    await add_post(
        "test-org", {"platform": "instagram", "content": "x", "title": "Volunteer spotlight"}
    )
    await record_performance("test-org", "post_001", {"reactions": 310})
    mock = AsyncMock(return_value=_calendar_reply({"angle": "a", "platform": "linkedin"}))
    with patch("agent.tools.litellm.acompletion", new=mock):
        result = await handle_tool_call(
            "generate_calendar",
            {"month_year": "Nov 2026", "focus": "year-end appeal"},
            "test-org",
        )

    prompt = mock.await_args.kwargs["messages"][0]["content"]
    assert "campaign focus: year-end appeal" in prompt
    assert "Volunteer spotlight (instagram): 310 engagements" in prompt
    assert "2026-11-22 to 2026-11-30" in prompt
    assert "Focus: year-end appeal" in result


async def test_generate_calendar_rejects_a_month_it_cannot_read(database):
    from agent.memory import get_ledger
    from agent.tools import handle_tool_call

    result = await handle_tool_call("generate_calendar", {"month_year": "next month"}, "test-org")
    assert result.startswith("Error: month_year 'next month'")
    assert await get_ledger("test-org") == []


@pytest.mark.parametrize(
    "proposed, expected",
    [("2026-07-10", "2026-07-10"), ("2026-07-20", "2026-07-08"), ("soon", "2026-07-08")],
)
def test_week_post_date_keeps_only_dates_inside_the_week(proposed, expected):
    from datetime import date

    from agent.tools import _week_post_date

    assert _week_post_date(proposed, date(2026, 7, 8), date(2026, 7, 14)).isoformat() == expected


def test_calendar_weeks_last_week_runs_to_month_end():
    from datetime import date

    from agent.tools import _calendar_weeks

    weeks = _calendar_weeks(date(2028, 2, 1))
    assert weeks[0] == (date(2028, 2, 1), date(2028, 2, 7))
    assert weeks[-1] == (date(2028, 2, 22), date(2028, 2, 29))


async def test_generate_calendar_plans_two_posts_a_week_by_default(database):
    from agent.memory import get_ledger
    from agent.tools import handle_tool_call

    reply = {
        "posts": [
            {"angle": "Reel tour", "platform": "instagram", "format": "reel"},
            {"angle": "Gala recap", "platform": "linkedin", "format": "made-up"},
            {"angle": "Extra idea beyond the cadence", "platform": "facebook"},
        ]
    }
    mock = AsyncMock(return_value=_calendar_reply(reply))
    with patch("agent.tools.litellm.acompletion", new=mock):
        result = await handle_tool_call(
            "generate_calendar", {"month_year": "July 2026"}, "test-org"
        )

    ledger = await get_ledger("test-org")
    assert len(ledger) == 8
    assert all(p["status"] == "planned" for p in ledger)
    assert "Content Calendar — July 2026" in result
    assert "Week 1" in result and "Week 4" in result
    assert ledger[0]["format"] == "reel"
    assert "format" not in ledger[1]  # an unknown format is dropped, the post is kept
    assert "Plan 2 post(s)" in mock.await_args.kwargs["messages"][0]["content"]
    assert "8 of 8 posts saved" in result


@pytest.mark.parametrize("requested, planned", [(1, 4), (9, 12), (0, 4)])
async def test_generate_calendar_caps_posts_per_week(database, requested, planned):
    from agent.memory import get_ledger
    from agent.tools import handle_tool_call

    reply = {"posts": [{"angle": f"Idea {n}"} for n in range(5)]}
    with patch(
        "agent.tools.litellm.acompletion", new=AsyncMock(return_value=_calendar_reply(reply))
    ):
        await handle_tool_call(
            "generate_calendar",
            {"month_year": "July 2026", "posts_per_week": requested},
            "test-org",
        )
    assert len(await get_ledger("test-org")) == planned


@pytest.mark.parametrize(
    "text, angles",
    [
        ('{"posts": [{"angle": "a"}, {"angle": "b"}]}', ["a", "b"]),
        ('{"angle": "solo"}', ["solo"]),
        ('[{"angle": "listed"}]', ["listed"]),
        ('Sure! ```json\n{"angle": "fenced"}\n```', ["fenced"]),
        ('{"posts": [{"angle": ""}, "junk", {"angle": "kept"}]}', ["kept"]),
        ('{"posts": "none"}', []),
        ("no json here", []),
        ("Plan: {angle: unquoted}", []),
    ],
)
def test_parse_week_plan_reads_the_shapes_models_reply_with(text, angles):
    from agent.tools import _parse_week_plan

    assert [p["angle"] for p in _parse_week_plan(text)] == angles


@pytest.mark.parametrize(
    "tool, args, missing",
    [
        ("save_post", {"platform": "linkedin", "title": "T"}, "content"),
        ("save_post", {"platform": "linkedin", "content": "", "title": ""}, "content, title"),
        ("record_performance", {"reactions": 3}, "post_id"),
        ("save_preference", {"key": "voice", "value": "warm"}, "quote"),
    ],
)
async def test_a_tool_called_without_required_arguments_says_which(database, tool, args, missing):
    """Models do leave out required arguments; the reply names them instead of a KeyError."""
    from agent.memory import load_profile
    from agent.tools import handle_tool_call

    result = await handle_tool_call(tool, args, "test-org")
    assert result == f"Error: {tool} needs {missing}; call it again with them."
    assert (await load_profile("test-org"))["post_ledger"] == []


async def test_a_suggestion_must_cite_a_source_once_there_is_research(database):
    """Once the org is researched, an idea must cite where it came from (source_url)."""
    from agent.memory import get_ledger, save_profile
    from agent.tools import handle_tool_call

    idea = {"platform": "general", "content": "Idea", "title": "T"}
    assert (await handle_tool_call("save_post", idea, "test-org")).startswith(
        "Saved"
    )  # nothing to cite yet

    await save_profile("cited", {"research": {"sources": [{"url": "https://org.example/about"}]}})
    refused = await handle_tool_call("save_post", idea, "cited")
    assert refused.startswith("Not saved: a suggestion must cite its source")
    cited = {**idea, "source_url": "https://org.example/about"}
    assert (await handle_tool_call("save_post", cited, "cited")).startswith("Saved")
    draft = {**idea, "status": "draft"}  # versions inherit the idea's source; only ideas must cite
    assert (await handle_tool_call("save_post", draft, "cited")).startswith("Saved")
    assert len(await get_ledger("cited")) == 2


@pytest.mark.parametrize(
    "post, expected",
    [
        ({"platform": "email", "format": "feed_post"}, ""),
        ({"platform": "Instagram", "format": "feed_post"}, "feed_post"),
        ({"platform": "linkedin", "format": "story"}, ""),
        ({"platform": "email", "format": "made-up"}, ""),
        ({"platform": "general"}, ""),
    ],
)
def test_calendar_keeps_a_format_only_where_the_platform_has_it(post, expected):
    """Models plan emails as feed posts; a format is kept only on a platform that has it."""
    from agent.tools import _calendar_format

    platforms = {"feed_post": ["linkedin", "instagram", "facebook"], "story": ["instagram"]}
    assert _calendar_format(post, platforms) == expected


@pytest.mark.parametrize(
    "planned, saved",
    [("Instagram", "instagram"), ("TikTok", "general"), ("", "general"), (None, "general")],
)
async def test_calendar_saves_a_platform_without_a_skill_file_as_general(database, planned, saved):
    """A planned post on a platform Mackie has no rules for is kept, as an idea for any platform."""
    from agent.memory import get_ledger, save_profile
    from agent.tools import handle_tool_call

    await save_profile("test-org", _profile_with_sources())
    plan = {"platform": planned, "format": "reel", "angle": "a", "pillar": "p", "rationale": "r"}
    week = _calendar_reply(plan)
    with patch("agent.tools.litellm.acompletion", new=AsyncMock(return_value=week)):
        table = await handle_tool_call(
            "generate_calendar", {"month_year": "November 2026", "posts_per_week": 1}, "test-org"
        )
    ledger = await get_ledger("test-org")
    assert {post["platform"] for post in ledger} == {saved}
    # A format is kept only on a platform that has it: reels are Instagram's.
    assert {post.get("format") for post in ledger} == {"reel" if saved == "instagram" else None}
    assert f"| {saved.capitalize()} |" in table


async def test_calendar_without_platform_skill_files_plans_general_ideas(database):
    """With every platform file removed or broken, the plan still saves, as ideas."""
    from agent.memory import get_ledger, save_profile
    from agent.skills import SkillSet
    from agent.tools import handle_tool_call

    await save_profile("test-org", _profile_with_sources())
    plan = {"platform": "instagram", "angle": "a", "pillar": "p", "rationale": "r"}
    week = _calendar_reply(plan)
    model = AsyncMock(return_value=week)
    with (
        patch("agent.tools.skills.load_skills", return_value=SkillSet()),
        patch("agent.tools.litellm.acompletion", new=model),
    ):
        await handle_tool_call(
            "generate_calendar", {"month_year": "November 2026", "posts_per_week": 1}, "test-org"
        )
    assert '"platform": one of "general"' in model.call_args.kwargs["messages"][0]["content"]
    assert [post["platform"] for post in await get_ledger("test-org")] == ["general"] * 4
