"""
Performance tests: parallel I/O, bounded prompts, and linear-time parsing.

Owns: proof that the I/O CLAUDE.md requires to run in parallel (research,
subpages, calendar weeks, judge calls) really does, using ConcurrencyProbe from
conftest.py; responsiveness (replies stream token by token, tool steps show as
they finish, a slow session doesn't stall others, nothing waits forever); scale
(the prompt stays the same size however large an org's memory grows, a post
goes to its platforms in parallel, tool
results are capped however large the research, ledger operations stay fast at
20,000 posts, a long conversation's stored history grows linearly and what it
sends the model stays bounded, also in real tokens, every query reads its org's
rows through an index, a turn sends a fixed number of statements, a login flood
leaves the pool free and a password check leaves the event loop free, each skill
file is parsed once however often it is read); and proof that the parsers and
the blocked-term check given model or website text don't slow down sharply on
hostile input. Does not own functional behavior (the per-module test files) or the suite
time budget (conftest.py).
"""

import asyncio
import json
import threading
import time
from collections.abc import Iterator
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import psycopg
import pytest
from psycopg import sql

from tests.conftest import load_script
from tests.llm_fakes import completion, text_stream, tool_call_stream

# Wall-clock ceiling for the linear-time checks. The inputs take a fraction of a second
# when parsing is linear (up to 0.4 s under coverage) and minutes if backtracking were
# quadratic, so this generous bound doesn't flake on a slow shared runner.
LINEAR_TIME_CEILING_S: float = 2.0
HOSTILE_INPUT_CHARS: int = 200_000
# Upper bound for any wait in the responsiveness tests: a buffered stream or a blocked
# event loop deadlocks instead of passing slowly, and this turns that into a failure.
WAIT_S: float = 5.0
# Ledger size for the scale tests: years of posting for a busy org.
LARGE_LEDGER_POSTS: int = 20_000
# Posts 1..posts of the "large" org, in every status (n % 6), half with results (odd n);
# run as the superuser, past row-level security. One statement, so psycopg can bind params.
SEED_LARGE_LEDGER_SQL: str = """
WITH org AS (
    INSERT INTO orgs (org_id, name, last_post_number) VALUES ('large', 'Large Org', %(posts)s)
    RETURNING org_id
)
INSERT INTO posts (org_id, post_id, status, platform, content, created_at, details, performance)
    SELECT org_id, 'post_' || n,
        (ARRAY['planned', 'draft', 'approved', 'posted', 'suggested', 'rejected'])[n %% 6 + 1],
        (ARRAY['linkedin', 'instagram', 'facebook'])[n %% 3 + 1],
        repeat('Body text ', 20), '2026-01-01T00:00:00+00:00',
        jsonb_build_object('title', 'Post ' || n, 'scheduled_for', '2026-01-01'),
        CASE WHEN n %% 2 = 1 THEN jsonb_build_object('reactions', n) END
    FROM org, generate_series(1, %(posts)s) n ORDER BY n
"""
# Org sizes compared in the prompt-size test: a tenfold bigger memory, the same prompt.
SMALL_ORG_POSTS: int = 50
LARGE_ORG_POSTS: int = 500
# Turns per half of the long-conversation test; quadratic growth shows as ~4x, linear as ~2x.
LONG_CONVERSATION_TURNS: int = 8
# The bounded-model-input test: turns, each with one tool result of this size.
TOOL_HEAVY_TURNS: int = 16
LIVE_TOOL_HEAVY_TURNS: int = 4  # the turns run through the graph; the rest are stored directly
TOOL_RESULT_CHARS: int = 8_000
# Everything LangGraph stores for conversations, read as the superuser (past row-level security).
STORED_CHECKPOINT_BYTES_SQL: str = (
    "SELECT (SELECT coalesce(sum(pg_column_size(checkpoint)), 0) FROM checkpoints)"
    " + (SELECT coalesce(sum(octet_length(blob)), 0) FROM checkpoint_blobs)"
    " + (SELECT coalesce(sum(octet_length(blob)), 0) FROM checkpoint_writes)"
)

# Other orgs' rows for the query-plan test: enough that reading them would be real work.
SEEDED_ORGS: int = 200
ROWS_PER_SEEDED_ORG: int = 5
# Tables a turn, a login check or a ledger search reads; none may be read in full.
PER_ORG_TABLES: frozenset[str] = frozenset(
    {
        "orgs",
        "posts",
        "preferences",
        "voice_examples",
        "tool_calls",
        "users",
        "memberships",
        "checkpoints",
        "checkpoint_blobs",
        "checkpoint_writes",
        "platform_accounts",
        "publications",
    }
)
# ROWS_PER_SEEDED_ORG rows for each of SEEDED_ORGS orgs: o is the org, n the row.
_EACH_SEEDED_ROW: str = (
    f"FROM generate_series(1, {SEEDED_ORGS}) o, generate_series(1, {ROWS_PER_SEEDED_ORG}) n"
)
# Every per-org table gets other orgs' rows; run as the superuser, past row-level security.
SEED_OTHER_ORGS_SQL: str = f"""
INSERT INTO orgs (org_id) SELECT 'seed-' || o FROM generate_series(1, {SEEDED_ORGS}) o;
INSERT INTO posts (org_id, post_id, status) SELECT 'seed-' || o, 'post_' || n, 'draft'
    {_EACH_SEEDED_ROW};
INSERT INTO preferences (org_id, key, value) SELECT 'seed-' || o, 'key ' || n, '1'
    {_EACH_SEEDED_ROW};
INSERT INTO voice_examples (org_id, content) SELECT 'seed-' || o, 'Example' {_EACH_SEEDED_ROW};
INSERT INTO tool_calls (org_id, tool, input, result_preview, latency_ms)
    SELECT 'seed-' || o, 'save_post', '{{}}', 'ok', 1 {_EACH_SEEDED_ROW};
INSERT INTO users (username, password_hash)
    SELECT 'user' || o, 'not a hash' FROM generate_series(1, {SEEDED_ORGS}) o;
INSERT INTO memberships (user_id, org_id)
    SELECT user_id, 'seed-' || (user_id % {SEEDED_ORGS} + 1) FROM users;
INSERT INTO checkpoints (thread_id, checkpoint_id, checkpoint)
    SELECT 'seed-' || o || ':web', 'c' || n, '{{}}' {_EACH_SEEDED_ROW};
INSERT INTO checkpoint_blobs (thread_id, channel, version, type)
    SELECT 'seed-' || o || ':web', 'messages', n::text, 'empty' {_EACH_SEEDED_ROW};
INSERT INTO checkpoint_writes (thread_id, checkpoint_id, task_id, idx, channel, blob)
    SELECT 'seed-' || o || ':web', 'c' || n, 't', 0, 'messages', '' {_EACH_SEEDED_ROW};
INSERT INTO platform_accounts (org_id, platform, account_id)
    SELECT 'seed-' || o, 'p' || n, 'page' {_EACH_SEEDED_ROW};
INSERT INTO publications (org_id, post_id, platform, state, account_id)
    SELECT 'seed-' || o, 'post_' || n, 'linkedin', 'published', 'page' {_EACH_SEEDED_ROW};
ANALYZE {", ".join(sorted(PER_ORG_TABLES))};
"""
# Posts in the small and the large org of the statements-per-turn test.
FEW_POSTS: int = 50
MANY_POSTS: int = 2_000
SEED_LEDGER_SQL: str = """
WITH org AS (
    INSERT INTO orgs (org_id, last_post_number) VALUES (%(org)s, %(posts)s) RETURNING org_id
)
INSERT INTO posts (org_id, post_id, status, content)
    SELECT org_id, 'post_' || n, 'draft', 'Body' FROM org, generate_series(1, %(posts)s) n
"""
# Turns compared in the statements-per-turn test: a turn late in a conversation and the first.
LATER_TURN: int = 4
# The first turn has no earlier checkpoints to prune, so it may send this many fewer.
FIRST_TURN_STATEMENTS_SAVED: int = 1
# Wrong-password logins sent at once in the login-flood test, against a pool of two.
LOGIN_GUESSES: int = 20
LOGIN_FLOOD_POOL_SIZE: int = 2


# --- Parallel I/O -----------------------------------------------------------------


async def test_a_post_is_published_to_its_platforms_in_parallel(database, concurrency_probe):
    from agent import memory as mem
    from agent import publishing

    await mem.fill_org_details("org", "Org")
    accounts = [publishing.PlatformAccount(p, "page") for p in publishing.PUBLISHERS]
    probe = concurrency_probe(len(accounts), publishing.Receipt("id"))
    request = publishing.PublishRequest("post_001", "x")
    with patch.object(publishing.DummyPublisher, "publish", probe):
        outcomes = await publishing.publish("org", request, accounts)
    assert probe.peak == len(accounts) and all(outcome.published for outcome in outcomes)


async def test_calendar_plans_its_four_weeks_in_parallel(database, concurrency_probe):
    from agent.memory import get_ledger
    from agent.tools import handle_tool_call

    probe = concurrency_probe(4, completion(json.dumps({"angle": "Idea", "platform": "general"})))
    with patch("agent.tools.litellm.acompletion", new=probe):
        result = await handle_tool_call(
            "generate_calendar", {"month_year": "July 2026"}, "test-org"
        )

    assert probe.peak == 4
    assert "Error generating plan" not in result
    assert len(await get_ledger("test-org")) == 4


@pytest.mark.parametrize("url, parallel_calls", [("https://org.example", 4), ("", 3)])
async def test_research_runs_scrape_and_three_searches_in_parallel(
    concurrency_probe, url, parallel_calls
):
    """With a website: its scrape and the three searches at once; without one, the searches."""
    from agent import research

    def result(*args):
        # scrape_site(url) gets one argument; search_web(query, max_results) gets two.
        return ("Site text", ["https://org.example"]) if len(args) == 1 else []

    probe = concurrency_probe(parallel_calls, respond=result)
    with (
        patch.object(research, "scrape_site", new=probe),
        patch.object(research, "search_web", new=probe),
    ):
        found = await research.research_org("Test Org", url)

    assert probe.peak == parallel_calls
    assert found["website_content"] == ("Site text" if url else "")


async def test_scrape_site_fetches_subpages_in_parallel(concurrency_probe):
    from agent import research

    home = (
        "https://org.example/",
        '<a href="/about">About</a><a href="/programs">Programs</a><a href="/events">Events</a>',
    )
    subpage_probe = concurrency_probe(3, respond=lambda url: (url, "<p>Subpage</p>"))

    async def fetch(url: str) -> tuple[str, str]:
        # The homepage is read first, on its own; its links are then fetched together.
        return home if url == "https://org.example" else await subpage_probe(url)

    with patch.object(research, "_fetch_html", new=fetch):
        text, pages = await research.scrape_site("https://org.example")

    assert subpage_probe.peak == 3
    assert len(pages) == 4 and text.count("Subpage") == 3


async def test_eval_runs_all_judge_calls_in_parallel(concurrency_probe):
    module = load_script("eval")
    responses = [{"step": i, "user": "m", "reply": "r"} for i in range(1, 8)]
    calls = len(responses) * len(module.RUBRICS)
    probe = concurrency_probe(calls, completion(json.dumps({"score": 3, "reason": "ok"})))
    with patch.object(module.litellm, "acompletion", new=probe):
        scores = await module.score_all(responses)

    assert probe.peak == calls == 21
    assert all(s["score"] == 3 for step in scores for s in step)


# --- Bounded prompts --------------------------------------------------------------


def _big_profile(posts: int) -> dict:
    """A profile whose every growing section has `posts` entries."""
    statuses = ("planned", "draft", "approved", "posted", "suggested")
    return {
        "name": "Big Org",
        "research": {
            "sources": [
                {"type": "web", "url": f"https://org.example/{n}", "title": f"Page {n}"}
                for n in range(posts)
            ]
        },
        "org_brief": "- Mission: help.",
        "voice_examples": [{"content": "x" * 1500, "source": f"pasted {n}"} for n in range(posts)],
        "post_ledger": [
            {
                "id": f"post_{n:03d}",
                "platform": "linkedin",
                "status": statuses[n % len(statuses)],
                "title": f"Post number {n}",
                "content": "Body " * 50,
                "created_at": "2026-01-01T00:00:00",
                "scheduled_for": "2026-01-01",
                "performance": {"reactions": n},
            }
            for n in range(posts)
        ],
    }


async def test_system_prompt_size_does_not_grow_with_org_memory(database):
    """Ledger, agenda, results, voice examples and sources are all capped in the prompt."""
    from agent.assistant import _build_system_prompt_blocks
    from agent.memory import save_profile

    await save_profile("small", _big_profile(SMALL_ORG_POSTS))
    await save_profile("large", _big_profile(LARGE_ORG_POSTS))
    small_stable, small_turn = await _build_system_prompt_blocks("small")
    large_stable, large_turn = await _build_system_prompt_blocks("large")

    # Only counts ("+ N more", "N total") and different ids/titles may differ.
    assert len(large_stable) == len(small_stable)
    assert len(large_turn) < len(small_turn) * 1.1
    assert "use search_ledger to list them" in large_turn


async def test_long_titles_pillars_platforms_and_notes_are_cut_in_the_prompt(database):
    """
    Regression: titles, pillars and results notes were shown whole in the ledger, agenda and
    results sections, so twenty long ones (imported, or saved by a fooled model) filled
    every prompt, past the context window, and "start a new conversation" didn't help.
    """
    from agent.assistant import _build_system_prompt_blocks
    from agent.memory import save_profile

    profile = _big_profile(SMALL_ORG_POSTS)
    for post in profile["post_ledger"]:
        post |= {"title": "t" * 10_000, "pillar": "p" * 10_000, "platform": "x" * 10_000}
        post["performance"]["note"] = "n" * 10_000
    await save_profile("long", profile)
    await save_profile("short", _big_profile(SMALL_ORG_POSTS))
    long_turn = (await _build_system_prompt_blocks("long"))[1]
    short_turn = (await _build_system_prompt_blocks("short"))[1]
    assert len(long_turn) < len(short_turn) + 30_000
    for cut in ("t" * 100, "p" * 100, "X" * 100, "n" * 300):
        assert cut not in long_turn


def test_tool_results_are_capped_whatever_the_research_size():
    """A huge website or search result can't flood the model's context."""
    from agent.tools import _format_research_summary

    snippet = {"title": "T", "snippet": "s" * 50_000, "url": "https://x.example"}
    result = {
        "website_content": "w" * 1_000_000,
        "social_presence": [snippet] * 100,
        "news": [snippet] * 100,
        "similar_orgs": [snippet] * 100,
        "sources": [],
    }
    assert len(_format_research_summary("Org", result)) < 10_000


# --- Linear-time parsing of hostile input -----------------------------------------


@pytest.mark.parametrize(
    "hostile",
    [
        "{" * HOSTILE_INPUT_CHARS,
        "{" + "a" * HOSTILE_INPUT_CHARS,
        '{"posts": [' + '{"angle": "x"},' * (HOSTILE_INPUT_CHARS // 16) + "]}",
    ],
)
def test_calendar_reply_parsing_stays_linear(hostile):
    from agent.tools import _parse_week_plan

    started = time.perf_counter()
    _parse_week_plan(hostile)
    assert time.perf_counter() - started < LINEAR_TIME_CEILING_S


@pytest.mark.parametrize(
    "content",
    ["#" * HOSTILE_INPUT_CHARS, "#a" * (HOSTILE_INPUT_CHARS // 2), "a" * HOSTILE_INPUT_CHARS],
)
async def test_post_checks_stay_linear(database, content):
    from agent.tools import handle_tool_call

    started = time.perf_counter()
    result = await handle_tool_call(
        "save_post", {"platform": "facebook", "content": content, "title": "T"}, "test-org"
    )
    assert time.perf_counter() - started < LINEAR_TIME_CEILING_S
    assert isinstance(result, str)


@pytest.mark.parametrize(
    "text, term",
    [
        ("a" * HOSTILE_INPUT_CHARS, "a" * 50 + "*"),
        ("a " * (HOSTILE_INPUT_CHARS // 2), " ".join(["a"] * 50) + " b"),
        ("é" * HOSTILE_INPUT_CHARS, "e" * 50 + "*"),
    ],
)
def test_blocked_term_matching_stays_linear(text, term):
    """Every save checks every blocked term against the whole post."""
    from agent.skills import Guardrails

    guardrails = Guardrails(blocked_terms=tuple(f"word{n}" for n in range(20)) + (term,))
    started = time.perf_counter()
    guardrails.blocked_term(text)
    assert time.perf_counter() - started < LINEAR_TIME_CEILING_S


def test_page_text_extraction_stays_linear():
    from agent.research import _find_subpages, _page_text

    html = "<div>" * 5_000 + '<a href="/about">About</a>' * 2_000 + "</div>" * 5_000
    started = time.perf_counter()
    _page_text(html)
    _find_subpages(html, "https://org.example/")
    assert time.perf_counter() - started < LINEAR_TIME_CEILING_S


# --- Responsiveness ---------------------------------------------------------------


async def test_reply_tokens_reach_the_caller_before_the_model_finishes(database):
    """The second token is only produced after the caller has seen the first one."""
    from agent.assistant import Assistant
    from tests.llm_fakes import gated_stream

    first_token_seen = asyncio.Event()
    model = AsyncMock(return_value=gated_stream(["Hel", "lo"], {1: first_token_seen}))
    tokens = []
    with patch("agent.graph.litellm.acompletion", new=model):
        async for event in Assistant("org").achat_stream("hi"):
            if event["type"] == "text":
                tokens.append(event["content"])
                first_token_seen.set()
    assert tokens == ["Hel", "lo"]


async def test_a_tool_step_is_shown_before_the_next_model_call_answers(database):
    """The reply comes only once the caller has seen the step: what it did and how long it took."""
    from agent.assistant import Assistant
    from tests.llm_fakes import gated_stream

    step_seen = asyncio.Event()
    args = json.dumps({"platform": "general", "content": "x", "title": "T"})
    model = AsyncMock(
        side_effect=[
            tool_call_stream("tc1", "save_post", args),
            gated_stream(["Saved."], {0: step_seen}),
        ]
    )
    events = []
    with patch("agent.graph.litellm.acompletion", new=model):
        async for event in Assistant("org").achat_stream("save it"):
            events.append(event)
            if event["type"] == "tool_step":
                step_seen.set()
    assert [event["type"] for event in events] == ["tool_step", "text"]
    step, reply = events
    assert step["tool"] == "save_post" and step["result_preview"].startswith("Saved")
    assert "title" in step["input_preview"] and step["latency_ms"] >= 0
    assert reply["content"] == "Saved."


async def test_a_session_waiting_on_a_slow_website_does_not_stall_another(database):
    """Research I/O is awaited, never blocking: Bob's reply arrives while Alice's search hangs."""
    from agent.assistant import Assistant

    release_search = asyncio.Event()

    async def slow_search(*_):
        await asyncio.wait_for(release_search.wait(), WAIT_S)
        return []

    async def model(**kwargs):
        messages = kwargs["messages"]
        if messages[-1]["role"] == "tool":
            return text_stream("Nothing new.")
        if messages[-1]["content"] == "search the news":
            return tool_call_stream("tc1", "web_search", '{"query": "news"}')
        return text_stream("Hi Bob")

    with (
        patch("agent.tools.res.search_web", new=slow_search),
        patch("agent.graph.litellm.acompletion", new=AsyncMock(side_effect=model)),
    ):
        alice = asyncio.create_task(Assistant("org", thread_id="org:a").chat("search the news"))
        bob = await asyncio.wait_for(Assistant("org", thread_id="org:b").chat("hello"), WAIT_S)
        assert not alice.done()
        release_search.set()
        assert await asyncio.wait_for(alice, WAIT_S) == "Nothing new."
    assert bob == "Hi Bob"


async def test_every_llm_call_has_a_timeout(database):
    """Chat replies, summaries, the org brief, calendar plans and eval judging all time out."""
    from langchain_core.messages import HumanMessage

    from agent.assistant import Assistant
    from agent.config import LLM_TIMEOUT_S
    from agent.graph import _summarize
    from agent.tools import handle_tool_call

    calls: list[dict] = []

    async def model(**kwargs):
        calls.append(kwargs)
        if kwargs.get("stream"):
            return text_stream("ok")
        return completion(json.dumps({"angle": "a"}))

    research = {
        "website_content": "w",
        "social_presence": [],
        "news": [],
        "similar_orgs": [],
        "sources": [],
    }
    with (
        patch("agent.graph.litellm.acompletion", new=model),
        patch("agent.tools.litellm.acompletion", new=model),
        patch("agent.tools.res.research_org", new=AsyncMock(return_value=research)),
    ):
        await Assistant("org").chat("hi")
        await _summarize("claude-haiku-5-5", "", [HumanMessage(content="old")])
        await handle_tool_call("research_org", {"org_name": "Org"}, "org")
        await handle_tool_call("generate_calendar", {"month_year": "July 2026"}, "org")
    eval_script = load_script("eval")
    with patch.object(eval_script.litellm, "acompletion", new=model):
        await eval_script.score_all([{"step": 1, "user": "m", "reply": "r"}])

    assert len(calls) == 1 + 1 + 1 + 4 + len(eval_script.RUBRICS)
    assert all(call["timeout"] == LLM_TIMEOUT_S for call in calls)


async def test_every_website_fetch_has_a_timeout():
    from agent import research

    stream = MagicMock(side_effect=AssertionError("stop after the request"))
    with patch.object(research.httpx, "AsyncClient") as client_class:
        client_class.return_value.__aenter__.return_value = MagicMock(stream=stream)
        with pytest.raises(AssertionError):
            await research._fetch_html("https://org.example")
    assert stream.call_args.kwargs["timeout"] == research.FETCH_TIMEOUT_S


async def test_a_page_that_trickles_gives_up_after_the_fetch_timeout(monkeypatch):
    """httpx's timeout bounds each wait for data; a byte every few seconds must not last forever."""
    from agent import research

    monkeypatch.setattr(research, "FETCH_TIMEOUT_S", 0.05)

    async def trickle():
        while True:
            yield b"x"
            await asyncio.sleep(0.01)

    response = MagicMock(is_redirect=False, aiter_raw=trickle, encoding="utf-8", headers={})
    stream = MagicMock()
    stream.return_value.__aenter__.return_value = response
    with patch.object(research.httpx, "AsyncClient") as client_class:
        client_class.return_value.__aenter__.return_value = MagicMock(stream=stream)
        with pytest.raises(TimeoutError):
            await research._fetch_html("https://org.example")


# --- Scale ------------------------------------------------------------------------


@pytest.fixture
async def large_org(database) -> str:
    """An org with LARGE_LEDGER_POSTS posts in every status, half with results."""
    database["cleaner"].execute(SEED_LARGE_LEDGER_SQL, {"posts": LARGE_LEDGER_POSTS})
    return "large"


async def test_ledger_operations_stay_fast_with_a_very_large_ledger(large_org):
    """Every turn reads the whole profile and saves add to it, so both must scale."""
    from datetime import date

    from agent import memory as mem
    from agent.assistant import _build_system_prompt_blocks

    started = time.perf_counter()
    ledger = await mem.get_ledger(large_org)
    await mem.search_ledger(large_org, status="draft", text="post 1999")
    mem.agenda(ledger, date(2026, 1, 1))
    mem.top_performing(ledger, 5)
    await mem.add_post(large_org, {"platform": "general", "content": "x"})
    await _build_system_prompt_blocks(large_org)
    elapsed = time.perf_counter() - started

    assert await mem.get_post(large_org, f"post_{LARGE_LEDGER_POSTS + 1}") is not None
    assert elapsed < LINEAR_TIME_CEILING_S * 3


def test_each_skill_file_is_parsed_once_however_many_times_it_is_read():
    """The skill files are read on every prompt build and tool call; parsing each is the cost."""
    from agent import skills

    skills._parse_yaml.cache_clear()
    files = len(list(skills.SKILLS_DIR.glob("*.yaml")))
    with patch("agent.skills._load_yaml", wraps=skills._load_yaml) as parse:
        for _ in range(20):
            skills.load_skills()
            skills.load_guardrails()
    assert parse.call_count == files


async def test_a_long_conversation_stores_history_linearly_not_quadratically(database):
    """
    Regression: every checkpoint is a full snapshot, so keeping them all grows a thread
    quadratically; prune_history keeps one per thread. Doubling the turns must roughly
    double what is stored, not quadruple it.
    """
    from agent.assistant import Assistant

    def stored_bytes() -> int:
        # Checkpoint rows hold channel versions; the messages themselves are in the blobs.
        return database["cleaner"].execute(STORED_CHECKPOINT_BYTES_SQL).fetchone()[0]

    async def model(**_):
        return text_stream("reply " + "x" * 300)

    assistant = Assistant("org")
    sizes = []
    with patch("agent.graph.litellm.acompletion", new=AsyncMock(side_effect=model)):
        for turn in range(1, 2 * LONG_CONVERSATION_TURNS + 1):
            await assistant.chat("question " + "y" * 300)
            if turn % LONG_CONVERSATION_TURNS == 0:
                sizes.append(stored_bytes())

    assert sizes[1] < sizes[0] * 2.5


def _tool_heavy_history(turns: int) -> list:
    """Earlier turns, each a question, a tool call, its TOOL_RESULT_CHARS result and a reply."""
    from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

    history = []
    for turn in range(turns):
        call = {"id": f"old{turn}", "name": "search_ledger", "args": {}, "type": "tool_call"}
        history += [
            HumanMessage(content=f"question {turn}"),
            AIMessage(content="", tool_calls=[call]),
            ToolMessage(
                content="r" * TOOL_RESULT_CHARS, tool_call_id=call["id"], name=call["name"]
            ),
            AIMessage(content="Here is what I found."),
        ]
    return history


async def test_a_long_conversation_with_large_tool_results_sends_the_model_a_bounded_history(
    database,
):
    """
    Old tool results are shortened and old turns summarized, so the last turn sends
    no more than the window however long the conversation; and the summary runs once
    per batch of messages that left the window, not once per turn, leaving less than
    a batch unsummarized. The early turns are stored directly, to keep the test cheap.
    """
    from agent.assistant import Assistant
    from agent.graph import (
        CHARS_PER_TOKEN,
        SUMMARY_BATCH_MESSAGES,
        WINDOW_TOKENS,
        _message_chars,
        _window_start,
        build_graph,
    )

    calls: list[dict] = []

    async def model(**kwargs):
        calls.append(kwargs)
        if not kwargs.get("stream"):
            return completion("Open requests and dates:\n- none\nSummary:\n- older turns")
        if kwargs["messages"][-1]["role"] == "user":
            return tool_call_stream(f"tc{len(calls)}", "search_ledger", "{}")
        return text_stream("Here is what I found.")

    assistant = Assistant("org")
    config = {"configurable": {"thread_id": assistant.thread_id}}
    async with build_graph(assistant.model, "org") as graph:
        history = _tool_heavy_history(TOOL_HEAVY_TURNS - LIVE_TOOL_HEAVY_TURNS)
        await graph.aupdate_state(config, {"messages": history, "org_id": "org"})
    with (
        patch("agent.graph.handle_tool_call", new=AsyncMock(return_value="r" * TOOL_RESULT_CHARS)),
        patch("agent.graph.litellm.acompletion", new=AsyncMock(side_effect=model)),
    ):
        for turn in range(LIVE_TOOL_HEAVY_TURNS):
            await assistant.chat(f"question {turn}")
    async with build_graph(assistant.model, "org") as graph:
        state = (await graph.aget_state(config)).values

    def sent_chars(message: dict) -> int:
        arguments = [call["function"]["arguments"] for call in message.get("tool_calls", [])]
        return len(message["content"] or "") + sum(map(len, arguments))

    last_call = [c for c in calls if c.get("stream")][-1]["messages"]
    stored = state["messages"]
    assert len(stored) == 4 * TOOL_HEAVY_TURNS
    assert sum(map(_message_chars, stored)) > 2 * WINDOW_TOKENS * CHARS_PER_TOKEN
    assert sum(map(sent_chars, last_call[1:])) <= WINDOW_TOKENS * CHARS_PER_TOKEN

    summaries = sum(1 for c in calls if not c.get("stream"))
    assert 1 <= summaries <= state["summarized_count"] / SUMMARY_BATCH_MESSAGES
    assert _window_start(stored) - state["summarized_count"] < SUMMARY_BATCH_MESSAGES


# A long session of everyday turns, the last two researching.
EVERYDAY_TURNS: int = 28
RESEARCH_TURNS: int = 2
# How far real tokens may exceed WINDOW_TOKENS: a provider's context is several times larger.
TOKEN_ESTIMATE_TOLERANCE: float = 1.5
ASK: str = (
    "Could you draft a LinkedIn post about our autumn food drive? We collected 12,400 meals "
    "with 230 volunteers, and donations close on November 3rd."
)
REPLY: str = 4 * (
    "Here's a draft for LinkedIn. It leads with the impact figure, thanks the volunteers, "
    "and ends with a clear call to donate before November 3rd. Let me know if you'd like "
    "it warmer or shorter; I can also suggest an Instagram carousel version. "
)
DRAFT: dict = {
    "platform": "linkedin",
    "title": "Autumn food drive thank-you",
    "content": 20
    * "12,400 meals. 230 volunteers. One community. 🍎 Thank you all! #FoodSecurity #Volunteers\n",
    "content_type": "feed_post",
    "visual_idea": "Volunteers packing boxes at the warehouse",
}
RESEARCH_RESULT: str = json.dumps(
    {
        "website_content": 160 * "The Riverside Food Bank serves families across three counties. ",
        "social_presence": [{"platform": "instagram", "url": "https://instagram.com/riverside"}],
        "news": 5
        * [
            {
                "title": "Food bank expands pantry hours",
                "url": "https://news.example.org/2026/10/pantry-hours",
                "snippet": "Volunteers say demand rose 30% this year.",
            }
        ],
    }
)


def _everyday_turn(n: int, research: bool) -> list:
    """A user's ask, a save_post call (or research_org) with its result, and the reply."""
    from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

    name, args = (
        ("research_org", {"org_name": "Riverside Food Bank"}) if research else ("save_post", DRAFT)
    )
    result = RESEARCH_RESULT if research else json.dumps({"id": f"post_{n:03d}", "status": "draft"})
    return [
        HumanMessage(content=ASK),
        AIMessage(
            content="",
            tool_calls=[{"id": f"c{n}", "name": name, "args": args, "type": "tool_call"}],
        ),
        ToolMessage(content=result, tool_call_id=f"c{n}", name=name),
        AIMessage(content=REPLY),
    ]


def test_a_full_window_estimated_by_characters_stays_within_its_token_budget():
    """
    The window counts CHARS_PER_TOKEN characters as a token, as no tokenizer fits every
    provider. That estimate undercounts JSON (tool arguments and results), emoji and
    non-Latin text, so a window the estimate fills is measured with a real tokenizer
    (GPT-4o's, offline): realistic English turns, a research result and save_post drafts
    stay within TOKEN_ESTIMATE_TOLERANCE times WINDOW_TOKENS.
    """
    import litellm

    from agent.graph import (
        CHARS_PER_TOKEN,
        MAX_MESSAGES,
        WINDOW_TOKENS,
        _message_chars,
        _shorten_old_tool_results,
        _to_litellm,
        _window_start,
        is_user_turn,
    )

    turns = EVERYDAY_TURNS + RESEARCH_TURNS
    history = [m for n in range(turns) for m in _everyday_turn(n, n >= EVERYDAY_TURNS)]
    shortened = _shorten_old_tool_results(history)
    start = _window_start(history)
    previous_turn = max(i for i in range(start) if is_user_turn(history[i]))
    # The token budget, not the message cap, bounds this window: one more turn won't fit.
    assert len(history) - start < MAX_MESSAGES
    assert sum(map(_message_chars, shortened[previous_turn:])) > WINDOW_TOKENS * CHARS_PER_TOKEN

    tokens = litellm.token_counter(model="gpt-4o", messages=_to_litellm(shortened[start:]))
    assert tokens <= TOKEN_ESTIMATE_TOLERANCE * WINDOW_TOKENS


def _plan_nodes(plan: dict) -> Iterator[dict]:
    """Every node of an EXPLAIN (FORMAT JSON) plan, subplans included."""
    yield plan
    for child in plan.get("Plans", []):
        yield from _plan_nodes(child)


def _full_reads(plan: dict) -> list[str]:
    """The nodes of a plan that read a whole per-org table, or a whole index of one."""
    problems = []
    for node in _plan_nodes(plan):
        kind, table = node["Node Type"], node.get("Relation Name")
        if table not in PER_ORG_TABLES:
            continue
        if kind == "Seq Scan":
            problems.append(f"Seq Scan on {table}")
        elif kind in ("Index Scan", "Index Only Scan") and "Index Cond" not in node:
            problems.append(f"{kind} of all of {node['Index Name']}")
    return problems


async def test_every_query_reads_only_its_own_orgs_rows_through_an_index(database):
    """
    Every statement of a turn that saves a post, and of the ledger, trace and access
    lookups, is captured and planned as the app role, with other orgs' rows present and
    analyzed. Sequential scans are disabled for the planning: the planner then still picks
    one when no index fits the WHERE clause, so a Seq Scan, or an index scan with no Index
    Cond, means no index serves the query and it reads every org's rows. Left to its own
    costs, the planner rightly seq-scans small tables; tipping it would take tables too big
    for the time budget, and would prove the table size rather than the index.
    """
    from agent import accounts, publishing
    from agent.assistant import Assistant
    from agent.memory import get_post, read_trace, search_ledger

    cleaner = database["cleaner"]
    cleaner.execute(SEED_OTHER_ORGS_SQL)
    (user_id,) = cleaner.execute("SELECT min(user_id) FROM users").fetchone()
    statements: list[tuple] = []
    real_execute = psycopg.AsyncCursor.execute

    async def recording_execute(cursor, query, params=None, **kwargs):
        statements.append((query, params))
        return await real_execute(cursor, query, params, **kwargs)

    args = json.dumps({"platform": "general", "content": "x", "title": "T"})
    model = AsyncMock(side_effect=[tool_call_stream("tc1", "save_post", args), text_stream("ok")])
    with (
        patch.object(psycopg.AsyncCursor, "execute", recording_execute),
        patch("agent.graph.litellm.acompletion", new=model),
    ):
        await Assistant("probe", thread_id="probe:web").chat("save a post")
        await get_post("probe", "post_001")
        await search_ledger("probe", status="draft", platform="general", text="t")
        await read_trace("probe")
        await accounts.has_access(user_id, "probe")
        await accounts.user_orgs(user_id)
        account = publishing.PlatformAccount("linkedin", "page", "", ("w_organization_social",))
        await publishing.connect_account("probe", account)
        await publishing.publish("probe", publishing.PublishRequest("post_001", "x"), [account])
        await publishing.published_platforms("probe", "post_001")
        await publishing.disconnect_account("probe", "linkedin")

    explained = 0
    with psycopg.connect(database["app"]) as conn, conn.transaction():
        conn.execute("SELECT set_config('app.org_id', 'probe', true)")
        conn.execute("SET LOCAL enable_seqscan = off")
        for query, params in statements:
            text = query if isinstance(query, str) else query.as_string(conn)
            if text.lstrip().upper().startswith("SET "):
                continue
            statement = sql.SQL("EXPLAIN (FORMAT JSON) ") + (
                sql.SQL(query) if isinstance(query, str) else query
            )
            (plan,) = conn.execute(statement, params).fetchone()
            explained += 1
            assert _full_reads(plan[0]["Plan"]) == [], text
    assert explained > 10


class _StatementCounter:
    """Counts the statements sent through psycopg's async cursors while patched in."""

    def __init__(self) -> None:
        self.count = 0
        self._execute = psycopg.AsyncCursor.execute
        self._executemany = psycopg.AsyncCursor.executemany

    def patched(self) -> Any:
        counter = self

        async def execute(cursor, *args, **kwargs):
            counter.count += 1
            return await counter._execute(cursor, *args, **kwargs)

        async def executemany(cursor, query, params_seq, **kwargs):
            params_seq = list(params_seq)
            counter.count += len(params_seq)
            return await counter._executemany(cursor, query, params_seq, **kwargs)

        return patch.multiple(psycopg.AsyncCursor, execute=execute, executemany=executemany)

    async def during(self, work: Any) -> int:
        """The statements sent while awaiting work."""
        before = self.count
        await work
        return self.count - before


async def test_statements_per_turn_dont_grow_with_the_conversation_or_the_ledger(database):
    """
    A turn's round trips are a fixed number: a later turn of a conversation sends as
    many as the first, and an org with 2,000 posts as many as one with 50. A per-message
    or per-post query (one statement per row) would make the counts differ.
    """
    from agent import db
    from agent.assistant import Assistant

    for org, posts in (("few", FEW_POSTS), ("many", MANY_POSTS)):
        database["cleaner"].execute(SEED_LEDGER_SQL, {"org": org, "posts": posts})
    await db.get_pool()  # opening the pool runs its own checks, once
    counter = _StatementCounter()
    model = AsyncMock(side_effect=lambda **_: text_stream("ok"))
    with (
        counter.patched(),
        patch("agent.graph.litellm.acompletion", new=model),
    ):
        conversation = Assistant("few")
        per_turn = [await counter.during(conversation.chat("hi")) for _ in range(LATER_TURN)]
        few = await counter.during(Assistant("few", thread_id="few:new").chat("hi"))
        many = await counter.during(Assistant("many", thread_id="many:new").chat("hi"))

    first, *later = per_turn
    assert len(set(later)) == 1 and later[0] - first <= FIRST_TURN_STATEMENTS_SAVED, per_turn
    assert many == few == first


async def test_a_flood_of_wrong_passwords_does_not_hold_up_a_chat(fresh_pool, monkeypatch):
    """
    Twenty guesses at once on a pool of two connections: every guess is refused without
    an error, and while all of them are still checking their password a profile still
    loads, because a password check holds no connection.
    """
    from agent import accounts
    from agent.memory import load_profile

    monkeypatch.setenv("MACKIE_DATABASE_POOL_SIZE", str(LOGIN_FLOOD_POOL_SIZE))
    await accounts.create_user("alice", "a long passphrase")
    real_verify = accounts._verify
    checking, profile_loaded = 0, asyncio.Event()
    all_checking = asyncio.Event()

    async def slow_check(password: str, stored: str | None) -> bool:
        nonlocal checking
        checking += 1
        if checking == LOGIN_GUESSES:
            all_checking.set()
        await asyncio.wait_for(profile_loaded.wait(), WAIT_S)
        return await real_verify(password, stored)

    with patch.object(accounts, "_verify", slow_check):
        guesses = asyncio.gather(
            *(accounts.authenticate("alice", "wrong passphrase") for _ in range(LOGIN_GUESSES))
        )
        await asyncio.wait_for(all_checking.wait(), WAIT_S)
        profile = await asyncio.wait_for(load_profile("org"), WAIT_S)
        profile_loaded.set()
        results = await asyncio.wait_for(guesses, WAIT_S)

    assert profile["org_id"] == "org"
    assert results == [None] * LOGIN_GUESSES
    assert await accounts.authenticate("alice", "a long passphrase") is None  # locked out


async def test_a_login_checks_its_password_without_blocking_other_sessions(database, monkeypatch):
    """
    The password check is slow on purpose (scrypt), so it runs in a worker thread: while one
    is still checking, other coroutines run. Checked on the loop, it would freeze it, the
    release below would never come, and the check would fail.
    """
    from agent import accounts

    await accounts.create_user("alice", "a long passphrase")
    checking, released = threading.Event(), threading.Event()

    def blocking_check(password: str, stored: str) -> bool:
        checking.set()
        assert released.wait(WAIT_S), "the event loop was blocked during the password check"
        return True

    with patch.object(accounts, "verify_password", blocking_check):
        login = asyncio.create_task(accounts.authenticate("alice", "a long passphrase"))
        await asyncio.wait_for(asyncio.to_thread(checking.wait, WAIT_S), WAIT_S)
        await asyncio.wait_for(asyncio.sleep(0), WAIT_S)  # the loop still runs other work
        released.set()
        user = await asyncio.wait_for(login, WAIT_S)
    assert user is not None and user.username == "alice"
