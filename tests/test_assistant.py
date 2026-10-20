"""
Unit tests for agent/assistant.py.

Owns: the system prompt (what each profile section shows, the stable and
per-turn blocks, tolerance of hand-edited entries), the agenda summary,
chat and achat_stream end to end (tool failures, the save check, resuming
from a stored summary), clear_history, recent_exchanges, which threads an
Assistant accepts, and the access check before every message.
Graph internals (window, summaries, fallback) are owned by test_graph.py.
"""

from unittest.mock import AsyncMock, patch

import pytest

from tests.llm_fakes import completion, text_stream, tool_call_stream


async def _full_prompt(org_id: str) -> str:
    """Both system prompt blocks as one string, as non-Anthropic providers receive them."""
    from agent.assistant import _build_system_prompt_blocks

    return "\n".join(await _build_system_prompt_blocks(org_id))


def _profile(**overrides: object) -> dict:
    """A fresh org's profile with the given fields replaced."""
    return {
        "org_id": "org",
        "name": "Test Org",
        "website": "",
        "research": None,
        "preferences": {},
        "post_ledger": [],
    } | overrides


SOURCES = [
    {"type": "website", "url": "https://example.org"},
    {"type": "news", "url": "https://news.org/article", "title": "Great article"},
]


@pytest.mark.parametrize(
    ("profile", "present", "absent"),
    [
        (
            _profile(),
            [
                "Not yet done",
                "LinkedIn",
                "Instagram",
                "Facebook",
                "Content Pillars",
                "Community Impact",
            ],
            ["## Learned Brand Preferences", "Research sources"],
        ),
        (
            _profile(
                research={"website_content": "some content"},
                research_updated_at="2026-06-12T10:00:00",
            ),
            ["Complete", "2026-06-12T10:00:00"],
            ["Not yet done"],
        ),
        (
            _profile(preferences={"voice": "warm and grassroots", "banned_topics": "politics"}),
            ["## Learned Brand Preferences", "warm and grassroots", "politics"],
            [],
        ),
        (
            _profile(
                post_ledger=[
                    {
                        "id": "post_001",
                        "platform": "linkedin",
                        "status": "draft",
                        "content": "Dignity matters for every patient.",
                        "created_at": "2026-06-12T00:00:00",
                    }
                ]
            ),
            ["post_001", "LINKEDIN", "draft"],
            [],
        ),
        (
            _profile(research={"website_content": "some content", "sources": SOURCES}),
            ["Research sources", "https://example.org", "Great article"],
            [],
        ),
        (
            _profile(research={"website_content": "some content", "sources": []}),
            [],
            ["Research sources"],
        ),
        (
            _profile(
                content_pillars=[{"name": "Custom Pillar", "description": "Our unique focus area"}]
            ),
            ["Custom Pillar"],
            ["Community Impact"],
        ),
    ],
    ids=[
        "fresh-org",
        "research-done",
        "preferences",
        "ledger",
        "sources",
        "no-sources",
        "own-pillars",
    ],
)
async def test_system_prompt_shows_what_the_profile_holds(database, profile, present, absent):
    from agent.memory import save_profile

    await save_profile("org", profile)
    prompt = await _full_prompt("org")
    for text in present:
        assert text in prompt
    for text in absent:
        assert text not in prompt


async def test_chat_returns_assistant_reply(database):
    from agent.assistant import Assistant

    with patch(
        "agent.graph.litellm.acompletion",
        new=AsyncMock(return_value=text_stream("Hello, I'm Mackie!")),
    ):
        assistant = Assistant("org")
        reply = await assistant.chat("Hello")
    assert "Mackie" in reply


async def test_chat_persists_conversation_to_checkpoint(database):
    from langchain_core.messages import AIMessage, HumanMessage

    from agent.assistant import Assistant
    from agent.graph import build_graph

    with patch(
        "agent.graph.litellm.acompletion", new=AsyncMock(return_value=text_stream("Reply text"))
    ):
        assistant = Assistant("org")
        reply = await assistant.chat("Hello")

    assert reply == "Reply text"
    config = {"configurable": {"thread_id": assistant.thread_id}}
    async with build_graph(assistant.model, "org") as graph:
        state = await graph.aget_state(config)
    msgs = state.values.get("messages", [])
    assert len(msgs) == 2
    assert isinstance(msgs[0], HumanMessage)
    assert isinstance(msgs[1], AIMessage)
    assert msgs[1].content == "Reply text"


async def test_clear_history_empties_only_its_own_thread_and_keeps_org_memory(database):
    from agent.assistant import Assistant
    from agent.graph import build_graph
    from agent.memory import load_profile, save_preference

    await save_preference("org", "voice", "warm")
    mine, other = Assistant("org"), Assistant("org", thread_id="org:other-session")
    with patch(
        "agent.graph.litellm.acompletion", new=AsyncMock(side_effect=lambda **_: text_stream("ok"))
    ):
        await mine.chat("Hello")
        await other.chat("Hi")

    await mine.clear_history()

    async with build_graph(mine.model, "org") as graph:
        mine_state = await graph.aget_state({"configurable": {"thread_id": "org"}})
        other_state = await graph.aget_state({"configurable": {"thread_id": "org:other-session"}})
    assert mine_state.values.get("messages", []) == []
    assert len(other_state.values["messages"]) == 2
    assert (await load_profile("org"))["preferences"] == {"voice": "warm"}


async def test_system_prompt_includes_todays_date(database):
    from datetime import date

    prompt = await _full_prompt("org")
    assert f"{date.today():%B %d, %Y}" in prompt


async def test_system_prompt_blocks_put_ledger_and_prefs_in_per_turn_block(database):
    from agent.assistant import _build_system_prompt_blocks
    from agent.memory import add_post, save_preference

    await save_preference("org", "voice", "playful and punchy")
    await add_post("org", {"platform": "instagram", "content": "Volunteer spotlight"})

    stable, per_turn = await _build_system_prompt_blocks("org")

    assert "Security Policy" in stable and "Platform Guidelines" in stable
    assert "playful and punchy" in per_turn and "Volunteer spotlight" in per_turn
    assert "playful and punchy" not in stable and "Volunteer spotlight" not in stable


async def test_stable_block_is_byte_identical_when_only_per_turn_state_changes(database):
    """The cached prefix survives a turn's writes and a new day; any byte change re-bills it."""
    from datetime import date, timedelta

    from agent import memory as mem
    from agent.assistant import _build_system_prompt_blocks

    await mem.save_research(
        "org", {"website_content": "We tutor kids", "sources": []}, "- Tutors kids"
    )
    stable_before, per_turn_before = await _build_system_prompt_blocks("org")

    post_id = await mem.add_post("org", {"platform": "linkedin", "content": "Tutor spotlight"})
    await mem.save_preference("org", "voice", "warm", quote="keep it warm")
    await mem.record_performance("org", post_id, {"reactions": 40})
    await mem.add_voice_example("org", "Our tutors rock", f"approved {post_id}")
    stable_after, per_turn_after = await _build_system_prompt_blocks("org")

    class Tomorrow(date):
        @classmethod
        def today(cls) -> date:
            return date.today() + timedelta(days=1)

    with patch("agent.assistant.date", Tomorrow):
        stable_tomorrow, per_turn_tomorrow = await _build_system_prompt_blocks("org")

    assert stable_before == stable_after == stable_tomorrow
    # The changes did reach the prompt, in the per-turn block.
    assert "Tutor spotlight" in per_turn_after and "Our tutors rock" in per_turn_after
    assert per_turn_after != per_turn_before
    assert f"{Tomorrow.today():%B %d, %Y}" in per_turn_tomorrow


async def test_chat_recovers_after_a_tool_fails(database):
    """A failing tool must not leave an unanswered tool call in the checkpoint."""
    from agent.assistant import Assistant

    second_turn = AsyncMock(return_value=text_stream("Back on track"))
    with patch(
        "agent.graph.litellm.acompletion",
        new=AsyncMock(
            side_effect=[
                # Missing 'status' makes update_post_status fail (an error result, not a reply).
                tool_call_stream("tc_bad", "update_post_status", '{"post_id": "post_001"}'),
                text_stream("Sorry, that failed."),
            ]
        ),
    ):
        assistant = Assistant("org")
        assert await assistant.chat("approve it") == "Sorry, that failed."

    with patch("agent.graph.litellm.acompletion", new=second_turn):
        assert await assistant.chat("hello again") == "Back on track"

    sent = second_turn.call_args.kwargs["messages"]
    tool_call_ids = {tc["id"] for m in sent if m.get("tool_calls") for tc in m["tool_calls"]}
    tool_result_ids = {m["tool_call_id"] for m in sent if m["role"] == "tool"}
    assert tool_call_ids == tool_result_ids == {"tc_bad"}


async def test_a_tool_call_with_broken_json_is_answered_as_broken_not_as_empty(database):
    """Regression: a live eval's cut-off save_post was reported as missing every argument."""
    from agent.assistant import Assistant
    from agent.memory import load_profile, read_trace
    from agent.tools import UNPARSED_ARGUMENTS_KEY

    cut_off = '{"platform": "instagram", "title": "Reel", "content": "Hook (0-3s'
    model = AsyncMock(
        side_effect=[tool_call_stream("tc1", "save_post", cut_off), text_stream("Retrying.")]
    )
    with patch("agent.graph.litellm.acompletion", new=model):
        assert await Assistant("org").chat("Make it a reel") == "Retrying."

    tool_result = model.call_args_list[1].kwargs["messages"][-1]
    assert tool_result["role"] == "tool"
    assert "were not valid JSON" in tool_result["content"]
    assert "needs" not in tool_result["content"]
    assert (await load_profile("org"))["post_ledger"] == []
    [trace] = await read_trace("org")
    assert trace["input"] == {UNPARSED_ARGUMENTS_KEY: cut_off}  # kept for diagnosis


async def _text_then_tool_call(text: str, tc_id: str, name: str, arguments: str):
    """A streamed reply that says text, then makes one tool call."""
    async for chunk in text_stream(text):
        if chunk.choices[0].delta.content:
            yield chunk
    async for chunk in tool_call_stream(tc_id, name, arguments):
        yield chunk


async def test_achat_stream_separates_pre_tool_text_from_reply(database):
    from agent.assistant import Assistant

    pre_tool = _text_then_tool_call(
        "Let me save that.", "tc_1", "save_post", '{"platform": "linkedin", "content": "x"}'
    )
    with patch(
        "agent.graph.litellm.acompletion",
        new=AsyncMock(side_effect=[pre_tool, text_stream("Saved!")]),
    ):
        events = [e async for e in Assistant("org").achat_stream("Save it")]

    text = "".join(e["content"] for e in events if e["type"] == "text")
    assert text == "Let me save that.\n\nSaved!"


async def test_a_restarted_assistant_carries_on_from_the_stored_summary(database):
    """
    A turn that overflows the window stores one summary in the checkpoint and keeps every
    message. A new Assistant on that thread (a restart, another process) sends the stored
    summary and the history from where it ends, and makes no second summary call.
    """
    from langchain_core.messages import AIMessage, HumanMessage

    from agent.assistant import Assistant
    from agent.graph import SUMMARY_BATCH_MESSAGES, _window_start, build_graph

    calls: list[dict] = []

    async def model(**kwargs):
        calls.append(kwargs)
        # The summarizer is the only non-streaming call.
        return text_stream("reply") if kwargs.get("stream") else completion("- zz-older-context")

    config = {"configurable": {"thread_id": "org"}}
    history = [m for n in range(25) for m in (HumanMessage(f"q{n}"), AIMessage(f"a{n}"))]
    async with build_graph("gpt-4o-mini", "org") as graph:
        await graph.aupdate_state(config, {"messages": history, "org_id": "org"})
    with patch("agent.graph.litellm.acompletion", new=AsyncMock(side_effect=model)):
        assert await Assistant("org").chat("latest question") == "reply"
        async with build_graph("gpt-4o-mini", "org") as graph:
            stored = (await graph.aget_state(config)).values
        summarized_count = stored["summarized_count"]
        assert stored["summary"] == "- zz-older-context"
        assert summarized_count == _window_start(stored["messages"][:-1]) >= SUMMARY_BATCH_MESSAGES
        assert len(stored["messages"]) == len(history) + 2  # nothing is deleted
        calls.clear()

        assert await Assistant("org").chat("after the restart") == "reply"

    [after_restart] = calls  # no summary call: fewer than a batch has left the window since
    system, *sent = after_restart["messages"]
    assert "- zz-older-context" in system["content"]
    assert [m["content"] for m in sent] == [
        m.content for m in stored["messages"][summarized_count:]
    ] + ["after the restart"]


async def test_system_prompt_includes_org_brief_in_stable_block(database):
    from agent.assistant import _build_system_prompt_blocks
    from agent.memory import load_profile, save_profile

    profile = await load_profile("org")
    profile.update(
        research={"sources": []}, org_brief="- Runs a free tutoring program (https://t.org)"
    )
    await save_profile("org", profile)
    stable, _ = await _build_system_prompt_blocks("org")
    assert "<org_brief>\n- Runs a free tutoring program (https://t.org)\n</org_brief>" in stable


async def test_system_prompt_includes_content_types(database):
    prompt = await _full_prompt("org")
    assert "## Content Types" in prompt
    assert "Donation Appeal (id: donation_appeal)" in prompt
    assert "Must include:" in prompt


async def test_system_prompt_shows_preference_quote_and_list_values(database):
    from agent.memory import save_preference

    await save_preference(
        "org", "voice", "warm and grassroots", quote="we're more warm and grassroots"
    )
    await save_preference("org", "banned_topics", "politics", append=True)
    await save_preference("org", "banned_topics", "religion", append=True)
    prompt = await _full_prompt("org")
    assert (
        '- **voice**: warm and grassroots (user said: "we\'re more warm and grassroots")' in prompt
    )
    assert "- **banned_topics**: politics; religion" in prompt


async def test_voice_examples_appear_in_per_turn_block(database):
    from agent.assistant import VOICE_EXAMPLES_IN_PROMPT, _build_system_prompt_blocks
    from agent.memory import add_voice_example

    for i in range(VOICE_EXAMPLES_IN_PROMPT + 1):
        await add_voice_example("org", f"zz-example-{i}", f"approved post_00{i}")
    stable, per_turn = await _build_system_prompt_blocks("org")
    assert "## Voice Examples" in per_turn
    assert "zz-example-0" not in per_turn  # only the newest few are shown
    assert f"zz-example-{VOICE_EXAMPLES_IN_PROMPT}" in per_turn
    assert "zz-example" not in stable


async def test_a_long_voice_example_is_cut_to_its_cap_in_the_prompt(database):
    from agent.assistant import VOICE_EXAMPLE_CHARS, _build_system_prompt_blocks
    from agent.memory import add_voice_example

    await add_voice_example("org", "q" * 2_000, "pasted by user")
    _, per_turn = await _build_system_prompt_blocks("org")
    assert "q" * VOICE_EXAMPLE_CHARS in per_turn
    assert "q" * (VOICE_EXAMPLE_CHARS + 1) not in per_turn


def test_ledger_section_shows_date_title_and_versions():
    from agent.assistant import _build_ledger_section

    section = _build_ledger_section(
        [
            {
                "id": "post_002",
                "platform": "instagram",
                "status": "draft",
                "content": "Long copy",
                "title": "Meet Ana",
                "parent_id": "post_001",
                "scheduled_for": "2026-11-03",
                "created_at": "2026-10-06T21:00:00",
            }
        ]
    )
    assert (
        "- **post_002** | 2026-10-06 | INSTAGRAM | draft | version of post_001 | for 2026-11-03 | Meet Ana"
        in section
    )


def test_ledger_section_points_to_search_ledger_for_older_posts():
    from agent.assistant import LEDGER_PROMPT_POSTS, _build_ledger_section

    ledger = [
        {"id": f"post_{i:03d}", "platform": "linkedin", "status": "draft", "content": "c"}
        for i in range(LEDGER_PROMPT_POSTS + 3)
    ]
    section = _build_ledger_section(ledger)
    assert f"## Post Ledger ({LEDGER_PROMPT_POSTS + 3} total)" in section
    assert "(+ 3 older posts — use search_ledger to look them up)" in section
    assert "**post_002**" not in section  # the oldest are left out
    assert "**post_003**" in section and f"**post_{LEDGER_PROMPT_POSTS + 2:03d}**" in section


async def test_chat_saves_posts_a_reply_showed_without_saving(database):
    """End to end: unsaved ids trigger one save_check round; the posts land in the ledger."""
    from agent.assistant import Assistant
    from agent.memory import load_profile

    completion = AsyncMock(
        side_effect=[
            text_stream("**1. Volunteer spotlight** · post_001"),
            tool_call_stream(
                "tc_1",
                "save_post",
                '{"platform": "linkedin", "content": "Meet Ana, our volunteer. #volunteer #community #nonprofit", "title": "Volunteer spotlight", "status": "suggested"}',
            ),
            text_stream("Saved: Volunteer spotlight · post_001"),
        ]
    )
    with patch("agent.graph.litellm.acompletion", new=completion):
        reply = await Assistant("org").chat("Suggest a post")

    assert reply == "**1. Volunteer spotlight** · post_001\n\nSaved: Volunteer spotlight · post_001"
    assert [p["title"] for p in (await load_profile("org"))["post_ledger"]] == [
        "Volunteer spotlight"
    ]
    check = completion.call_args_list[1].kwargs["messages"][-1]
    assert check["role"] == "user" and "Automatic check" in check["content"]


async def test_chat_runs_save_check_only_once_per_turn(database):
    """A model that still doesn't save must not loop: one follow-up, then the turn ends."""
    from agent.assistant import Assistant

    completion = AsyncMock(
        side_effect=[text_stream("Idea · post_005"), text_stream("Idea · post_005")]
    )
    with patch("agent.graph.litellm.acompletion", new=completion):
        reply = await Assistant("org").chat("Suggest a post")

    assert completion.await_count == 2
    assert reply == "Idea · post_005\n\nIdea · post_005"


async def test_system_prompt_tolerates_hand_edited_entries_inside_valid_sections(database):
    """
    Entries a person typed by hand in an imported profile: a post with only a
    title, a quote with no date, a voice example without content, odd sources.
    """
    from agent.assistant import _build_system_prompt_blocks
    from agent.memory import save_profile

    await save_profile(
        "org",
        {
            "research": {"sources": ["https://not-a-dict.example", {"url": "https://ok.example"}]},
            "preferences": {"voice": "warm", "banned_topics": ["politics", 3]},
            "preference_notes": {"voice": {"quote": "said in a meeting"}},
            "voice_examples": [{"source": "pasted by user"}],
            "post_ledger": [{"title": "Typed by hand"}],
        },
    )
    stable, per_turn = await _build_system_prompt_blocks("org")
    assert "https://ok.example" in stable
    assert "**voice**: warm" in per_turn and "politics; 3" in per_turn
    assert "said in a meeting" in per_turn
    assert "### pasted by user" in per_turn
    assert "Typed by hand" in per_turn


async def test_system_prompt_describes_the_job_and_the_approval_rule(database):
    from agent.assistant import _build_system_prompt_blocks

    stable = (await _build_system_prompt_blocks("org"))[0]
    assert "AI social media manager" in stable
    for part in (
        "**Plan:**",
        "**Create:**",
        "**Optimize:**",
        "record_performance",
        "Lead with what needs doing",
        "Nothing goes live without the user's approval",
        'call `update_post_status` with status "approved"',
    ):
        assert part in stable


async def test_performance_section_lists_best_posts_in_the_per_turn_block(database):
    from agent.assistant import _build_system_prompt_blocks
    from agent.memory import add_post, record_performance

    await add_post("org", {"platform": "linkedin", "content": "a", "title": "Gala recap"})
    await add_post("org", {"platform": "instagram", "content": "b", "title": "Meet Ana"})
    await record_performance("org", "post_001", {"reactions": 12})
    await record_performance("org", "post_002", {"reach": 2000, "reactions": 240}, "5 sign-ups")

    stable, per_turn = await _build_system_prompt_blocks("org")
    assert "## What Performed Best" not in stable
    section = per_turn[per_turn.index("## What Performed Best") :]
    assert section.index("Meet Ana") < section.index("Gala recap")
    assert "2000 reach, 240 reactions — 5 sign-ups" in section


async def test_no_performance_section_without_results(database):
    from agent.memory import add_post

    await add_post("org", {"platform": "linkedin", "content": "a"})
    assert "## What Performed Best" not in await _full_prompt("org")


async def test_performance_section_tolerates_hand_edited_results(database):
    from agent.memory import save_profile

    await save_profile(
        "org", {"post_ledger": [{"title": "t", "performance": {"reactions": "lots", "note": None}}]}
    )
    assert "## What Performed Best" in await _full_prompt("org")


async def test_system_prompt_lists_post_formats_with_their_roles(database):
    from agent.assistant import _build_system_prompt_blocks

    stable = (await _build_system_prompt_blocks("org"))[0]
    assert "## Post Formats" in stable
    assert "### Reel (id: reel; on instagram, facebook)" in stable
    assert "### Feed Post (id: feed_post; on linkedin, instagram, facebook)" in stable


async def test_agenda_section_and_summary_list_what_needs_attention(database):
    from agent.assistant import _build_system_prompt_blocks, agenda_summary
    from agent.memory import add_post

    await add_post(
        "org", {"platform": "linkedin", "content": "a", "status": "draft", "title": "Gala"}
    )
    await add_post("org", {"platform": "facebook", "content": "b", "status": "approved"})
    for status in ("posted", "planned"):
        await add_post(
            "org",
            {
                "platform": "linkedin",
                "content": "c",
                "status": status,
                "scheduled_for": "2020-01-01",
            },
        )

    per_turn = (await _build_system_prompt_blocks("org"))[1]
    section = per_turn[per_turn.index("## Needs Attention") :]
    assert "### Drafts waiting for your approval\n- **post_001** | LINKEDIN" in section
    # Most urgent first: approvals unblock publishing, so they lead (AGENDA_STEPS order).
    assert await agenda_summary("org") == (
        "Here's what needs you: 1 draft waiting for your approval; "
        "1 approved post ready to publish; 1 planned post to write up this week; "
        "1 published post to report results for."
    )


async def test_no_agenda_section_or_summary_when_nothing_needs_attention(database):
    from agent.assistant import agenda_summary
    from agent.memory import add_post

    await add_post("org", {"platform": "linkedin", "content": "a", "status": "rejected"})
    assert "## Needs Attention" not in await _full_prompt("org")
    assert await agenda_summary("org") == ""


async def test_system_prompt_leaves_out_empty_skill_sections(database):
    """An org with no content types, formats or pillars gets no empty headings."""
    from agent.assistant import _build_system_prompt_blocks
    from agent.skills import SkillSet

    bare_platform = {"id": "x", "name": "X", "max_length": 100}
    with (
        patch(
            "agent.assistant.skills.load_skills", return_value=SkillSet(platforms=[bare_platform])
        ),
        patch("agent.assistant.skills.load_content_pillars", return_value=[]),
    ):
        stable, _ = await _build_system_prompt_blocks("org")

    for heading in ("## Content Types", "## Post Formats", "## Content Pillars", "Format notes"):
        assert heading not in stable
    assert "### X (id: x)" in stable and "- Hashtags: none" in stable


@pytest.mark.parametrize(
    ("org_id", "thread_id"),
    [
        ("org-a", "org-b:x"),  # another org's session
        ("org-a", "org-ab"),  # another org whose id starts with this one's
        ("org-a", "org-ab:x"),
        ("org-a", "org-b"),
    ],
)
def test_assistant_refuses_a_thread_of_another_org(org_id, thread_id):
    from agent.assistant import Assistant

    with pytest.raises(ValueError):
        Assistant(org_id, thread_id=thread_id)


def test_assistant_accepts_its_own_org_threads():
    from agent.assistant import Assistant

    assert Assistant("org-a").thread_id == "org-a"
    assert Assistant("org-a", thread_id="org-a:web-1").thread_id == "org-a:web-1"


def test_count_noun_uses_the_right_form():
    from agent.assistant import count_noun

    assert count_noun(1, "draft", "drafts") == "1 draft"
    assert count_noun(0, "draft", "drafts") == "0 drafts"
    assert count_noun(3, "draft", "drafts") == "3 drafts"


@pytest.mark.parametrize("removal", ["revoke", "delete the account"])
async def test_removing_access_stops_an_open_conversation_at_its_next_message(database, removal):
    """The check runs before every turn and reset, so an open session can't keep using the org."""
    from agent import accounts
    from agent.assistant import Assistant
    from agent.graph import build_graph

    with patch("agent.accounts.SCRYPT_N", 2**4):
        user = await accounts.create_user("sam", "a long passphrase")
    await accounts.grant("sam", "org")
    assistant = Assistant("org", thread_id="org:tab", user_id=user.user_id)
    model = AsyncMock(side_effect=lambda **_: text_stream("Hello"))
    with patch("agent.graph.litellm.acompletion", new=model):
        assert await assistant.chat("first") == "Hello"
        if removal == "revoke":
            await accounts.revoke("sam", "org")
        else:
            await accounts.delete_user("sam")
        with pytest.raises(accounts.AccessRevoked, match="no longer have access to org"):
            await assistant.chat("second")
        with pytest.raises(accounts.AccessRevoked):
            async for _ in assistant.achat_stream("third"):
                pass
        with pytest.raises(accounts.AccessRevoked):
            await assistant.clear_history()
        with pytest.raises(accounts.AccessRevoked):
            await assistant.recent_exchanges(5)
    assert model.await_count == 1  # nothing after the removal reached the model
    async with build_graph(assistant.model, "org") as graph:
        state = await graph.aget_state({"configurable": {"thread_id": "org:tab"}})
    assert len(state.values["messages"]) == 2  # reset didn't delete the thread either


async def test_recent_exchanges_lists_only_what_the_user_saw(database):
    """
    No tool calls, tool results or save_check follow-up: the user's messages and every
    text Mackie streamed, including what it said before a tool call.
    """
    from agent.assistant import Assistant

    saved = '{"platform": "linkedin", "content": "Meet Ana", "title": "Spotlight"}'
    model = AsyncMock(
        side_effect=[
            text_stream("Spotlight · post_001"),
            _text_then_tool_call("Let me save that.", "tc_1", "save_post", saved),
            text_stream("Saved: Spotlight · post_001"),
            text_stream("You're welcome"),
        ]
    )
    assistant = Assistant("org")
    assert await assistant.recent_exchanges(10) == []  # a new conversation
    with patch("agent.graph.litellm.acompletion", new=model):
        await assistant.chat("Suggest a post")
        await assistant.chat("Thanks")

    assert await assistant.recent_exchanges(10) == [
        ("user", "Suggest a post"),
        ("assistant", "Spotlight · post_001"),
        ("assistant", "Let me save that."),  # text before a tool call was shown too
        ("assistant", "Saved: Spotlight · post_001"),
        ("user", "Thanks"),
        ("assistant", "You're welcome"),
    ]
    assert await assistant.recent_exchanges(2) == [
        ("user", "Thanks"),
        ("assistant", "You're welcome"),
    ]
    assert await assistant.recent_exchanges(0) == []
