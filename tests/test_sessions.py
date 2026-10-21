"""
Isolation and concurrency between sessions, users and orgs.

Owns: what happens when several people use Mackie at once — many sessions
starting at once on one connection pool (also more sessions than connections),
two people on the same org (also the CLI and the web UI), two tabs of one user
taking turns on one conversation (also a new conversation started during a
turn), a profile read while another session writes it, every kind of org write
at once (edits of one post among them; also from separate processes, through the database, and research that
finishes after another session saved), and streams interleaving. The model is
mocked with scripts that answer each session by the content of its own messages,
so a reply or a tool write routed to the wrong session fails the test. Does not
own single-session behavior (test_assistant.py) or org isolation without
concurrency (test_security.py).
"""

import asyncio
import json
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from unittest.mock import AsyncMock, patch

from tests.llm_fakes import gated_stream, text_stream, tool_call_stream

# Enough sessions at once to overlap many database writes, small enough for the time budget.
CONCURRENT_SESSIONS: int = 4
# Fewer connections than sessions, so every connection serves several orgs in turn.
SMALL_POOL_SIZE: int = 2
# Upper bound for any single wait in these tests; a routing or locking bug hangs instead.
WAIT_S: float = 5.0
WRITER_PROCESSES: int = 2
# Separate processes start a Python interpreter each; a lost lock or a deadlock hangs instead.
PROCESS_WAIT_S: float = 30.0
POSTS_PER_PROCESS: int = 25
REPO_ROOT: Path = Path(__file__).parent.parent


def _last_user_message(messages: list[dict]) -> str:
    return next(m["content"] for m in reversed(messages) if m["role"] == "user")


# --- Many sessions at once ------------------------------------------------------------


async def test_concurrent_sessions_each_get_their_own_reply_writes_and_history(
    fresh_pool, monkeypatch
):
    """
    Many orgs, fewer connections, two rounds; every turn saves a post through a tool.
    Nothing may cross over: a connection still set to the org it served last would show
    as a reply, a post or a message in the wrong org, or as round 1 missing from round 2.
    """
    from agent import db
    from agent.assistant import Assistant
    from agent.memory import get_ledger

    monkeypatch.setenv("MACKIE_DATABASE_POOL_SIZE", str(SMALL_POOL_SIZE))
    history_seen: dict[tuple[str, str], list[str]] = {}

    async def model(**kwargs):
        messages = kwargs["messages"]
        org, turn = _last_user_message(messages).split(" round ")
        if messages[-1]["role"] == "tool":
            return text_stream(f"saved {org} round {turn}")
        history_seen[(org, turn)] = [m["content"] for m in messages if m["role"] == "user"]
        args = {"platform": "general", "content": "x", "title": f"{org} round {turn}"}
        return tool_call_stream(f"tc-{org}-{turn}", "save_post", json.dumps(args))

    orgs = [f"org-{n}" for n in range(CONCURRENT_SESSIONS)]
    with (
        patch("agent.graph.litellm.acompletion", new=AsyncMock(side_effect=model)),
    ):
        for turn in ("1", "2"):
            replies = await asyncio.wait_for(
                asyncio.gather(
                    *(
                        Assistant(org, thread_id=f"{org}:web").chat(f"{org} round {turn}")
                        for org in orgs
                    )
                ),
                WAIT_S,
            )
            assert replies == [f"saved {org} round {turn}" for org in orgs]

    assert (await db.get_pool()).max_size == SMALL_POOL_SIZE
    for org in orgs:
        assert history_seen[(org, "2")] == [f"{org} round 1", f"{org} round 2"]
        titles = [post["title"] for post in await get_ledger(org)]
        assert titles == [f"{org} round 1", f"{org} round 2"]


async def test_the_connection_pool_is_opened_once_however_many_sessions_start(fresh_pool):
    """Sessions starting together share one pool opening instead of each opening their own."""
    from agent import db
    from agent.assistant import Assistant

    openings = 0
    original_open = db._open_pool

    async def counting_open():
        nonlocal openings
        openings += 1
        return await original_open()

    with (
        patch.object(db, "_open_pool", counting_open),
        patch(
            "agent.graph.litellm.acompletion",
            new=AsyncMock(side_effect=lambda **_: text_stream("ok")),
        ),
    ):
        await asyncio.gather(
            *(Assistant("org", thread_id=f"org:{n}").chat("hi") for n in range(CONCURRENT_SESSIONS))
        )
        await Assistant("org", thread_id="org:later").chat("hi")

    assert openings == 1


# --- Two people, one org --------------------------------------------------------------


async def test_sessions_on_one_org_share_its_memory_but_not_their_conversations(database):
    """
    A web session and a CLI session on one org (thread ids as app.py and main.py name them):
    the post the web session saved is org memory and reaches the CLI's prompt; what either
    person typed reaches only their own session's later turns.
    """
    from agent.assistant import Assistant

    web = Assistant("food-bank", thread_id="food-bank:web-user-1")
    cli = Assistant("food-bank", thread_id="food-bank:cli-user-2")
    args = json.dumps({"platform": "general", "content": "x", "title": "Shared gala post"})
    first = AsyncMock(
        side_effect=[tool_call_stream("tc1", "save_post", args), text_stream("Saved")]
    )
    with patch("agent.graph.litellm.acompletion", new=first):
        await web.chat("A private draft idea typed in the browser")

    cli_model = AsyncMock(return_value=text_stream("Reply in the terminal"))
    with patch("agent.graph.litellm.acompletion", new=cli_model):
        await cli.chat("Typed in the terminal")
    system, *conversation = cli_model.call_args.kwargs["messages"]
    assert "Shared gala post" in json.dumps(system)  # the ledger is org memory
    assert [m["content"] for m in conversation] == ["Typed in the terminal"]

    web_model = AsyncMock(return_value=text_stream("ok"))
    with patch("agent.graph.litellm.acompletion", new=web_model):
        await web.chat("Back in the browser")
    conversation = web_model.call_args.kwargs["messages"][1:]
    assert [m["content"] for m in conversation if m["role"] == "user"] == [
        "A private draft idea typed in the browser",
        "Back in the browser",
    ]
    assert "terminal" not in json.dumps(conversation)


async def test_turns_on_one_thread_take_turns_while_other_threads_run(database):
    """
    Two tabs on one conversation: the second turn reaches the model only after the
    first has finished, and sees it, so neither saves a history missing the other.
    A turn on another thread is not held up meanwhile.
    """
    from agent.assistant import Assistant
    from agent.graph import build_graph

    release_first, first_streaming = asyncio.Event(), asyncio.Event()
    order: list[str] = []
    seen_by_second: list[str] = []

    async def model(**kwargs):
        message = kwargs["messages"][-1]["content"]
        order.append(f"model: {message}")
        if message == "one":
            first_streaming.set()
            return gated_stream(["reply one"], {0: release_first})
        if message == "two":
            seen_by_second.extend(m["content"] for m in kwargs["messages"][1:])
        return text_stream(f"reply {message}")

    async def first_turn() -> None:
        async for _event in Assistant("org", thread_id="org:tab").achat_stream("one"):
            pass
        order.append("first turn done")

    with patch("agent.graph.litellm.acompletion", new=AsyncMock(side_effect=model)):
        first = asyncio.create_task(first_turn())
        await asyncio.wait_for(first_streaming.wait(), WAIT_S)
        second = asyncio.create_task(Assistant("org", thread_id="org:tab").chat("two"))
        other = Assistant("org", thread_id="org:other").chat("three")
        assert await asyncio.wait_for(other, WAIT_S) == "reply three"
        assert not second.done()
        release_first.set()
        assert await asyncio.wait_for(second, WAIT_S) == "reply two"
        await first

    assert order == ["model: one", "model: three", "first turn done", "model: two"]
    assert seen_by_second == ["one", "reply one", "two"]
    async with build_graph("gpt-4o-mini", "org") as graph:
        state = await graph.aget_state({"configurable": {"thread_id": "org:tab"}})
    assert [m.content for m in state.values["messages"]] == ["one", "reply one", "two", "reply two"]


async def test_a_new_conversation_started_during_a_turn_waits_for_it_and_stays_new(database):
    """
    "Start a new conversation" in one tab while a turn runs in another: the clear waits
    for the turn to finish, so the finished turn can't save its messages back into the
    cleared thread.
    """
    from agent import assistant as assistant_module
    from agent.assistant import Assistant

    thread_id = "org:web-user-1"
    release, streaming, clear_waiting = asyncio.Event(), asyncio.Event(), asyncio.Event()
    order: list[str] = []
    real_thread_lock = assistant_module._thread_lock

    def thread_lock(thread: str) -> asyncio.Lock:
        lock = real_thread_lock(thread)
        if lock.locked():  # the clear arrives while the turn holds the thread
            clear_waiting.set()
        return lock

    async def model(**kwargs):
        streaming.set()
        return gated_stream(["reply"], {0: release})

    async def turn() -> None:
        assert await Assistant("org", thread_id=thread_id).chat("one") == "reply"
        order.append("turn done")

    async def new_conversation() -> None:
        await Assistant("org", thread_id=thread_id).clear_history()
        order.append("cleared")

    with (
        patch.object(assistant_module, "_thread_lock", thread_lock),
        patch("agent.graph.litellm.acompletion", new=AsyncMock(side_effect=model)),
    ):
        running = asyncio.create_task(turn())
        await asyncio.wait_for(streaming.wait(), WAIT_S)
        clearing = asyncio.create_task(new_conversation())
        await asyncio.wait_for(clear_waiting.wait(), WAIT_S)
        release.set()
        await asyncio.wait_for(asyncio.gather(running, clearing), WAIT_S)

    assert order == ["turn done", "cleared"]
    assert await Assistant("org", thread_id=thread_id).recent_exchanges(10) == []


async def test_interleaved_streams_reach_only_their_own_session(database):
    """Alice's reply pauses mid-stream while Bob's streams in full; neither gets the other's."""
    from agent.assistant import Assistant

    alice_started, bob_finished = asyncio.Event(), asyncio.Event()

    async def model(**kwargs):
        if _last_user_message(kwargs["messages"]) == "from Alice":
            stream = gated_stream(["A1", "A2"], {1: bob_finished})
        else:
            stream = gated_stream(["B1", "B2"], {0: alice_started})
        return stream

    async def collect(assistant: Assistant, message: str) -> list[str]:
        tokens = []
        async for event in assistant.achat_stream(message):
            if event["type"] == "text":
                tokens.append(event["content"])
                if event["content"] == "A1":
                    alice_started.set()
        return tokens

    with patch("agent.graph.litellm.acompletion", new=AsyncMock(side_effect=model)):
        alice = asyncio.create_task(collect(Assistant("org", thread_id="org:alice"), "from Alice"))
        bob = await asyncio.wait_for(
            collect(Assistant("org", thread_id="org:bob"), "from Bob"), WAIT_S
        )
        bob_finished.set()
        assert await asyncio.wait_for(alice, WAIT_S) == ["A1", "A2"]
    assert bob == ["B1", "B2"]


# --- Every kind of write at once, in one process and across processes -------------


RESEARCH = {"website_content": "", "social_presence": [], "news": [], "similar_orgs": []}


async def _seed_org_for_writers() -> None:
    from agent.memory import add_post, save_preference, save_research

    await add_post("org", {"platform": "linkedin", "content": "x", "title": "Existing"})
    await save_preference("org", "voice", "warm")
    await save_research("org", {"sources": []}, "")


async def test_every_kind_of_org_write_at_once_loses_no_update(database):
    """
    Regression: a writer that loads, changes and saves without locking the org's row
    overwrites the others. Every kind of writer runs at once; every write must land.
    """
    import agent.memory as mem
    from agent.tools import handle_tool_call

    await _seed_org_for_writers()
    await mem.add_post("org", {"platform": "general", "content": "Join us", "title": "Edited"})
    search = [{"title": "t", "snippet": "s", "url": "https://news.example/a"}]
    writers = [
        *(
            mem.save_preference("org", "banned_topics", f"topic {n}", append=True)
            for n in range(10)
        ),
        *(
            handle_tool_call(
                "save_post", {"platform": "general", "content": "x", "title": f"T{n}"}, "org"
            )
            for n in range(10)
        ),
        mem.update_post_status("org", "post_001", "draft"),
        mem.record_performance("org", "post_001", {"reach": 1}),
        mem.save_preference("org", "tone", "plain"),
        mem.forget_preference("org", "voice"),
        mem.add_voice_example("org", "x", "pasted"),
        mem.fill_org_details("org", "Org", "https://org.example"),
        handle_tool_call("research_org", {"org_name": "Org"}, "org"),
        handle_tool_call("web_search", {"query": "q"}, "org"),
        *(
            handle_tool_call(
                "edit_post", {"post_id": "post_002", "find": "Join", "replace": f"Meet {n}"}, "org"
            )
            for n in range(5)
        ),
    ]
    with (
        patch("agent.tools.res.research_org", new=AsyncMock(return_value=dict(RESEARCH))),
        patch("agent.tools.res.search_web", new=AsyncMock(return_value=search)),
        patch("agent.tools.litellm.acompletion", new=AsyncMock(side_effect=RuntimeError)),
    ):
        await asyncio.wait_for(asyncio.gather(*writers), WAIT_S)

    profile = await mem.load_profile("org")
    ledger = profile["post_ledger"]
    assert len(ledger) == 17 and len({post["id"] for post in ledger}) == 17
    assert "performance" in ledger[0] and ledger[1]["content"] == "Join us"
    versions = [post for post in ledger if post.get("parent_id") == "post_002"]
    assert sorted(post["content"] for post in versions) == [f"Meet {n} us" for n in range(5)]
    assert sorted(profile["preferences"]["banned_topics"]) == sorted(
        f"topic {n}" for n in range(10)
    )
    assert profile["preferences"]["tone"] == "plain" and "voice" not in profile["preferences"]
    assert len(profile["voice_examples"]) == 1
    assert profile["website"] == "https://org.example"
    assert profile["research"] is not None and "org_brief" in profile  # research saved too


async def test_one_post_approved_in_two_sessions_at_once_is_one_voice_example(database):
    """Regression: each session read the old status separately, and both added the post."""
    import agent.memory as mem
    from agent.tools import Turn, _format_post, handle_tool_call

    await mem.add_post("org", {"platform": "general", "content": "Our post", "status": "draft"})
    approve = {"post_id": "post_001", "status": "approved"}
    shown = _format_post(await mem.get_post("org", "post_001"))
    typed = Turn(user_text="approve it", shown_posts=(shown,))
    await asyncio.wait_for(
        asyncio.gather(
            *(handle_tool_call("update_post_status", approve, "org", typed) for _ in range(5))
        ),
        WAIT_S,
    )
    assert len((await mem.load_profile("org"))["voice_examples"]) == 1


async def test_appends_in_parallel_sessions_cannot_take_a_list_over_its_cap(database):
    """Regression: the cap was checked outside the transaction that saved, so all passed it."""
    import agent.memory as mem
    from agent.tools import MAX_PREFERENCE_LIST_CHARS, Turn, handle_tool_call

    item = "x" * (MAX_PREFERENCE_LIST_CHARS // 4)
    rules = [f"{n}{item}" for n in range(8)]
    typed = Turn(user_text=" ".join(["q", *rules]))
    saves = [
        handle_tool_call(
            "save_preference",
            {"key": "rules", "value": rule, "quote": "q", "append": True},
            "org",
            typed,
        )
        for rule in rules
    ]
    results = await asyncio.wait_for(asyncio.gather(*saves), WAIT_S)
    rules = (await mem.load_profile("org"))["preferences"]["rules"]
    assert sum(len(rule) for rule in rules) <= MAX_PREFERENCE_LIST_CHARS
    assert sum(result.startswith("Not saved") for result in results) == 8 - len(rules)


async def test_a_profile_read_is_one_snapshot_while_another_session_replaces_it(database):
    """
    load_profile reads four tables one after another. A whole-profile write that commits
    between those reads (here, after the first) must not show: the result is all the old
    profile, none of the new.
    """
    from agent import db
    from agent.memory import load_profile, save_profile

    def profile(tag: str) -> dict:
        return {
            "name": f"Org {tag}",
            "post_ledger": [
                {"id": f"post_00{n}", "platform": "general", "content": f"{tag} {n}"}
                for n in (1, 2)
            ],
            "preferences": {"tone": f"tone {tag}"},
            "voice_examples": [{"content": f"voice {tag}", "source": "pasted"}],
        }

    await save_profile("org", profile("A"))
    before = await load_profile("org")
    first_query_done, other_write_done = asyncio.Event(), asyncio.Event()
    real_transaction = db.org_transaction

    class PausingConnection:
        """Passes everything through, but holds the second table's read until the write commits."""

        def __init__(self, conn):
            self.conn = conn

        def __getattr__(self, name):
            return getattr(self.conn, name)

        async def execute(self, query, params=None):
            if "FROM preferences" in query:
                first_query_done.set()
                await asyncio.wait_for(other_write_done.wait(), WAIT_S)
            return await self.conn.execute(query, params)

    @asynccontextmanager
    async def pausing_reads(org_id: str, *, read_only: bool = False):
        async with real_transaction(org_id, read_only=read_only) as conn:
            yield PausingConnection(conn) if read_only else conn

    with patch.object(db, "org_transaction", pausing_reads):
        read = asyncio.create_task(load_profile("org"))
        await asyncio.wait_for(first_query_done.wait(), WAIT_S)
        await asyncio.wait_for(save_profile("org", profile("B")), WAIT_S)
        other_write_done.set()
        during = await asyncio.wait_for(read, WAIT_S)

    assert during == before
    after = await load_profile("org")
    assert after["name"] == "Org B" and after["preferences"] == {"tone": "tone B"}


async def test_separate_processes_writing_to_one_org_lose_nothing(database):
    """
    Regression: the web UI and the CLI on one org are separate processes, so only the
    database's lock keeps them from overwriting each other's posts. Each process saves
    posts and appends to one list preference.
    """
    from agent.memory import load_profile

    writer = (
        "import asyncio, sys; import agent.memory as mem; from agent import db\n"
        "async def write():\n"
        f"    for n in range({POSTS_PER_PROCESS}):\n"
        "        title = sys.argv[1] + str(n)\n"
        "        await mem.add_post('org', {'platform': 'general', 'content': 'x', 'title': title})\n"
        "        await mem.save_preference('org', 'topics', title, append=True)\n"
        "    await db.close_pool()\n"
        "asyncio.run(write())"
    )
    environment = {**os.environ, "MACKIE_DATABASE_URL": database["app"]}
    processes = [
        await asyncio.create_subprocess_exec(
            sys.executable,
            "-c",
            writer,
            f"P{n}-",
            cwd=REPO_ROOT,
            env=environment,
            stderr=asyncio.subprocess.PIPE,
        )
        for n in range(WRITER_PROCESSES)
    ]
    outputs = await asyncio.wait_for(
        asyncio.gather(*(process.communicate() for process in processes)), PROCESS_WAIT_S
    )
    errors = [stderr.decode() for _, stderr in outputs]

    assert [process.returncode for process in processes] == [0] * WRITER_PROCESSES, errors
    profile = await load_profile("org")
    expected = sorted(
        f"P{p}-{n}" for p in range(WRITER_PROCESSES) for n in range(POSTS_PER_PROCESS)
    )
    ledger = profile["post_ledger"]
    assert sorted(post["title"] for post in ledger) == expected
    assert len({post["id"] for post in ledger}) == len(ledger)
    assert sorted(profile["preferences"]["topics"]) == expected


async def test_research_finishing_late_keeps_posts_saved_while_it_ran(database):
    """Research takes seconds; it must not save a profile it loaded before another session wrote."""
    from agent.memory import get_ledger, load_profile
    from agent.tools import handle_tool_call

    research_started, other_session_saved = asyncio.Event(), asyncio.Event()

    async def slow_research(**_):
        research_started.set()
        await asyncio.wait_for(other_session_saved.wait(), WAIT_S)
        return dict(RESEARCH, sources=[])

    with (
        patch("agent.tools.res.research_org", new=slow_research),
        patch("agent.tools.litellm.acompletion", new=AsyncMock(side_effect=RuntimeError)),
    ):
        research = asyncio.create_task(handle_tool_call("research_org", {"org_name": "Org"}, "org"))
        await asyncio.wait_for(research_started.wait(), WAIT_S)
        args = {"platform": "general", "content": "x", "title": "Saved meanwhile"}
        await handle_tool_call("save_post", args, "org")
        other_session_saved.set()
        await asyncio.wait_for(research, WAIT_S)

    assert [post["title"] for post in await get_ledger("org")] == ["Saved meanwhile"]
    assert (await load_profile("org"))["research"] is not None
