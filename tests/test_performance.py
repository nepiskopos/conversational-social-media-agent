"""
Performance tests: parallel I/O, bounded prompts, and linear-time parsing.

Owns: proof that the I/O CLAUDE.md requires to run in parallel (research,
subpages, calendar weeks, judge calls) really does, using ConcurrencyProbe from
conftest.py; responsiveness (replies stream token by token, tool steps show
as they finish, a slow session doesn't stall others, nothing waits forever);
scale (the prompt stays the same size however large an org's memory grows,
ledger operations stay fast at 20,000 posts, a long conversation's stored
history grows linearly); and proof that the parsers given model or website
text don't slow down sharply on hostile input. Does not own functional
behavior (the per-module test files) or the suite time budget (conftest.py).
"""

import asyncio
import importlib.util
import json
import sys
import time
from pathlib import Path
from types import ModuleType
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Wall-clock ceiling for the linear-time checks. The inputs take milliseconds when
# parsing is linear and far longer if backtracking were quadratic, so this
# generous bound doesn't flake on a slow runner.
LINEAR_TIME_CEILING_S: float = 1.0
HOSTILE_INPUT_CHARS: int = 200_000
# Upper bound for any wait in the responsiveness tests: a buffered stream or a blocked
# event loop deadlocks instead of passing slowly, and this turns that into a failure.
WAIT_S: float = 5.0
# Ledger size for the scale tests: years of posting for a busy org.
LARGE_LEDGER_POSTS: int = 20_000
# Turns per half of the long-conversation test; quadratic growth shows as ~4x, linear as ~2x.
LONG_CONVERSATION_TURNS: int = 8


EVAL_SCRIPT: Path = Path(__file__).parent.parent / "scripts" / "eval.py"


def _load_eval_script() -> ModuleType:
    """Import scripts/eval.py without loading the real .env into the test process."""
    spec = importlib.util.spec_from_file_location("mackie_eval_perf", EVAL_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    with patch("dotenv.load_dotenv"):
        spec.loader.exec_module(module)
    sys.modules.pop("mackie_eval_perf", None)
    return module


def _completion(content: str) -> MagicMock:
    """A litellm completion response whose message is content."""
    return MagicMock(choices=[MagicMock(message=MagicMock(content=content))])


# --- Parallel I/O -----------------------------------------------------------------


async def test_calendar_plans_its_four_weeks_in_parallel(tmp_data_dir, concurrency_probe):
    from agent.memory import get_ledger
    from agent.tools import handle_tool_call

    probe = concurrency_probe(4, _completion(json.dumps({"angle": "Idea", "platform": "general"})))
    with patch("agent.tools.litellm.acompletion", new=probe):
        result = await handle_tool_call(
            "generate_calendar", {"month_year": "July 2026"}, "test-org"
        )

    assert probe.peak == 4
    assert "Error generating plan" not in result
    assert len(get_ledger("test-org")) == 4


async def test_research_runs_scrape_and_three_searches_in_parallel(concurrency_probe):
    from agent import research

    def result(*args):
        # scrape_site(url) gets one argument; search_web(query, max_results) gets two.
        return ("Site text", ["https://org.example"]) if len(args) == 1 else []

    probe = concurrency_probe(4, respond=result)
    with (
        patch.object(research, "scrape_site", new=probe),
        patch.object(research, "search_web", new=probe),
    ):
        found = await research.research_org("Test Org", "https://org.example")

    assert probe.peak == 4
    assert found["website_content"] == "Site text"


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
    module = _load_eval_script()
    responses = [{"step": i, "user": "m", "reply": "r"} for i in range(1, 8)]
    calls = len(responses) * len(module.RUBRICS)
    probe = concurrency_probe(calls, _completion(json.dumps({"score": 3, "reason": "ok"})))
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
        "voice_examples": [{"content": "x" * 5000, "source": f"pasted {n}"} for n in range(posts)],
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


@pytest.mark.parametrize("small, large", [(50, 2000)])
def test_system_prompt_size_does_not_grow_with_org_memory(tmp_data_dir, small, large):
    """Ledger, agenda, results, voice examples and sources are all capped in the prompt."""
    from agent.assistant import _build_system_prompt_blocks
    from agent.memory import save_profile

    save_profile("small", _big_profile(small))
    save_profile("large", _big_profile(large))
    small_stable, small_turn = _build_system_prompt_blocks("small")
    large_stable, large_turn = _build_system_prompt_blocks("large")

    # Only counts ("+ 1995 more", "2000 total") and different ids/titles may differ.
    assert len(large_stable) == len(small_stable)
    assert len(large_turn) < len(small_turn) * 1.1
    assert "use search_ledger to list them" in large_turn


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
async def test_post_checks_stay_linear(tmp_data_dir, content):
    from agent.tools import handle_tool_call

    started = time.perf_counter()
    result = await handle_tool_call(
        "save_post", {"platform": "facebook", "content": content, "title": "T"}, "test-org"
    )
    assert time.perf_counter() - started < LINEAR_TIME_CEILING_S
    assert isinstance(result, str)


def test_page_text_extraction_stays_linear():
    from agent.research import _find_subpages, _page_text

    html = "<div>" * 5_000 + '<a href="/about">About</a>' * 2_000 + "</div>" * 5_000
    started = time.perf_counter()
    _page_text(html)
    _find_subpages(html, "https://org.example/")
    assert time.perf_counter() - started < LINEAR_TIME_CEILING_S


# --- Responsiveness ---------------------------------------------------------------


async def test_reply_tokens_reach_the_caller_before_the_model_finishes(tmp_data_dir):
    """The second token is only produced after the caller has seen the first one."""
    from agent.assistant import Assistant
    from tests.test_sessions import _gated_stream

    first_token_seen = asyncio.Event()
    model = AsyncMock(return_value=_gated_stream(["Hel", "lo"], {1: first_token_seen}))
    tokens = []
    with patch("agent.graph.litellm.acompletion", new=model):
        async for event in Assistant("org").achat_stream("hi"):
            if event["type"] == "text":
                tokens.append(event["content"])
                first_token_seen.set()
    assert tokens == ["Hel", "lo"]


async def test_a_tool_step_is_shown_before_the_next_model_call_answers(tmp_data_dir):
    """The reply is only produced once the caller has seen the step, with what it did and how long."""
    from agent.assistant import Assistant
    from tests.test_assistant import _tool_call_stream
    from tests.test_sessions import _gated_stream

    step_seen = asyncio.Event()
    args = json.dumps({"platform": "general", "content": "x", "title": "T"})
    model = AsyncMock(
        side_effect=[
            _tool_call_stream("tc1", "save_post", args),
            _gated_stream(["Saved."], {0: step_seen}),
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


async def test_a_session_waiting_on_a_slow_website_does_not_stall_another(tmp_data_dir):
    """Research I/O is awaited, never blocking: Bob's reply arrives while Alice's search hangs."""
    from agent.assistant import Assistant
    from tests.test_assistant import _text_stream, _tool_call_stream

    release_search = asyncio.Event()

    async def slow_search(*_):
        await asyncio.wait_for(release_search.wait(), WAIT_S)
        return []

    async def model(**kwargs):
        messages = kwargs["messages"]
        if messages[-1]["role"] == "tool":
            return _text_stream("Nothing new.")
        if messages[-1]["content"] == "search the news":
            return _tool_call_stream("tc1", "web_search", '{"query": "news"}')
        return _text_stream("Hi Bob")

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


async def test_every_llm_call_has_a_timeout(tmp_data_dir):
    """Chat replies, summaries, the org brief and calendar plans: none can wait forever."""
    from langchain_core.messages import HumanMessage

    from agent.assistant import Assistant
    from agent.config import LLM_TIMEOUT_S
    from agent.graph import _summarize
    from agent.tools import handle_tool_call
    from tests.test_assistant import _text_stream

    calls: list[dict] = []

    async def model(**kwargs):
        calls.append(kwargs)
        if kwargs.get("stream"):
            return _text_stream("ok")
        return _completion(json.dumps({"angle": "a"}))

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

    assert len(calls) == 1 + 1 + 1 + 4
    assert all(call["timeout"] == LLM_TIMEOUT_S for call in calls)


async def test_every_website_fetch_has_a_timeout():
    from agent import research

    response = MagicMock(is_redirect=False, text="<p>Hi</p>")
    client = MagicMock(get=AsyncMock(return_value=response))
    with patch.object(research.httpx, "AsyncClient") as client_class:
        client_class.return_value.__aenter__.return_value = client
        await research._fetch_html("https://org.example")
    assert client.get.await_args.kwargs["timeout"] == research.FETCH_TIMEOUT_S


async def test_a_provider_that_times_out_hands_over_to_the_next(tmp_data_dir):
    import litellm
    from langchain_core.messages import HumanMessage

    from agent.graph import _build_call_model
    from tests.test_assistant import _text_stream

    async def model(**kwargs):
        if kwargs["model"].startswith("claude"):
            raise litellm.Timeout("no answer", model=kwargs["model"], llm_provider="anthropic")
        return _text_stream("from backup")

    with (
        patch("agent.graph.litellm.acompletion", new=model),
        patch("agent.graph.get_stream_writer", return_value=lambda _: None),
    ):
        call_model = _build_call_model("claude-haiku-5-5", ["gpt-4o-mini"])
        result = await call_model({"messages": [HumanMessage(content="hi")], "org_id": "org"})
    assert result["messages"][0].content == "from backup"


# --- Scale ------------------------------------------------------------------------


@pytest.fixture
def large_org(tmp_data_dir) -> str:
    """An org with LARGE_LEDGER_POSTS posts in every status, half with results."""
    from agent.memory import save_profile

    statuses = ("planned", "draft", "approved", "posted", "suggested", "rejected")
    save_profile(
        "large",
        {
            "name": "Large Org",
            "post_ledger": [
                {
                    "id": f"post_{n:05d}",
                    "platform": ("linkedin", "instagram", "facebook")[n % 3],
                    "status": statuses[n % len(statuses)],
                    "title": f"Post {n}",
                    "content": "Body text " * 20,
                    "created_at": "2026-01-01T00:00:00",
                    "scheduled_for": "2026-01-01",
                    **({"performance": {"reactions": n}} if n % 2 else {}),
                }
                for n in range(LARGE_LEDGER_POSTS)
            ],
        },
    )
    return "large"


def test_ledger_operations_stay_fast_with_a_very_large_ledger(large_org):
    """Every turn reads the whole profile and every save rewrites it, so both must scale."""
    from datetime import date

    from agent import memory as mem
    from agent.assistant import _build_system_prompt_blocks

    started = time.perf_counter()
    ledger = mem.get_ledger(large_org)
    mem.search_ledger(large_org, status="draft", text="post 1999")
    mem.agenda(ledger, date(2026, 1, 1))
    mem.top_performing(ledger, 5)
    mem.add_post(large_org, {"platform": "general", "content": "x"})
    _build_system_prompt_blocks(large_org)
    elapsed = time.perf_counter() - started

    assert mem.get_ledger(large_org)[-1]["id"] == f"post_{LARGE_LEDGER_POSTS:05d}"
    assert elapsed < LINEAR_TIME_CEILING_S * 3


async def test_a_long_conversation_stores_history_linearly_not_quadratically(tmp_data_dir):
    """
    Regression: every checkpoint is a full snapshot, and keeping them all made a
    thread grow quadratically (431 MB after 400 turns). Doubling the turns must
    roughly double what is stored, not quadruple it.
    """
    import aiosqlite

    from agent.assistant import Assistant
    from agent.graph import CHECKPOINTS_DB
    from tests.test_assistant import _text_stream

    async def stored_bytes() -> int:
        async with aiosqlite.connect(CHECKPOINTS_DB) as conn:
            cursor = await conn.execute("SELECT SUM(LENGTH(checkpoint)) FROM checkpoints")
            return (await cursor.fetchone())[0]

    async def model(**_):
        return _text_stream("reply " + "x" * 300)

    assistant = Assistant("org")
    sizes = []
    with patch("agent.graph.litellm.acompletion", new=AsyncMock(side_effect=model)):
        for turn in range(1, 2 * LONG_CONVERSATION_TURNS + 1):
            await assistant.chat("question " + "y" * 300)
            if turn % LONG_CONVERSATION_TURNS == 0:
                sizes.append(await stored_bytes())

    assert sizes[1] < sizes[0] * 2.5
