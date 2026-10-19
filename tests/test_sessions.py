"""
Isolation and concurrency between sessions, users and orgs.

Owns: what happens when several people use Mackie at once — many sessions
sharing the SQLite checkpoint store, two people on the same org, the CLI and
the web UI on the same org (also as separate processes writing at once),
streams interleaving, and a model that names another org in its tool
arguments. The model is mocked with scripts that answer each session by the
content of its own messages, so a reply or a tool write routed to the wrong
session fails the test. Does not own
single-session behavior (test_assistant.py) or org-memory isolation without
concurrency (test_security.py).
"""

import asyncio
import fcntl
import json
import subprocess
import sys
import threading
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from tests.test_assistant import _text_stream, _tool_call_stream

# Enough sessions at once to overlap many SQLite writes, small enough for the time budget.
CONCURRENT_SESSIONS: int = 8
SIMULTANEOUS_WRITES: int = 30
# Upper bound for any single wait in these tests; a routing or locking bug hangs instead.
WAIT_S: float = 5.0
# How long a writer must stay blocked on another process's lock. A writer that skips the
# lock finishes in milliseconds and fails; a slow machine can only make it pass wrongly.
STILL_BLOCKED_S: float = 0.3
WRITER_PROCESSES: int = 3
POSTS_PER_PROCESS: int = 25
REPO_ROOT: Path = Path(__file__).parent.parent


def _last_user_message(messages: list[dict]) -> str:
    return next(m["content"] for m in reversed(messages) if m["role"] == "user")


def _text_chunk(text: str | None, finish_reason: str | None = None) -> MagicMock:
    """One stream chunk carrying text, or the closing chunk when finish_reason is set."""
    chunk = MagicMock()
    chunk.choices = [
        MagicMock(delta=MagicMock(content=text, tool_calls=None), finish_reason=finish_reason)
    ]
    return chunk


async def _gated_stream(tokens: list[str], gates: dict[int, asyncio.Event]):
    """Stream tokens; before token i, wait for gates[i] if there is one."""
    for i, token in enumerate(tokens):
        if i in gates:
            await asyncio.wait_for(gates[i].wait(), WAIT_S)
        yield _text_chunk(token)
    yield _text_chunk(None, "stop")


# --- Many sessions at once ------------------------------------------------------------


async def test_concurrent_sessions_each_get_their_own_reply_and_their_own_writes(tmp_data_dir):
    """Every session saves a post through a tool and replies; nothing crosses over."""
    from agent.assistant import Assistant
    from agent.memory import get_ledger

    async def model(**kwargs):
        messages = kwargs["messages"]
        session = _last_user_message(messages).removeprefix("session ")
        if messages[-1]["role"] == "tool":
            return _text_stream(f"saved for {session}")
        args = {"platform": "general", "content": f"post {session}", "title": f"Post {session}"}
        return _tool_call_stream(f"tc-{session}", "save_post", json.dumps(args))

    with patch("agent.graph.litellm.acompletion", new=AsyncMock(side_effect=model)):
        replies = await asyncio.gather(
            *(
                Assistant(f"org-{n}", thread_id=f"org-{n}:web").chat(f"session {n}")
                for n in range(CONCURRENT_SESSIONS)
            )
        )

    assert replies == [f"saved for {n}" for n in range(CONCURRENT_SESSIONS)]
    for n in range(CONCURRENT_SESSIONS):
        assert [p["title"] for p in get_ledger(f"org-{n}")] == [f"Post {n}"]


async def test_checkpoint_schema_is_set_up_once_however_many_sessions_start(tmp_data_dir):
    """
    Regression: every turn re-ran the schema setup on its own connection, and with
    ~100 sessions starting at once SQLite failed some with "database is locked".
    """
    from agent.assistant import Assistant

    setups = 0
    original_setup = AsyncSqliteSaver.setup

    async def counting_setup(self):
        nonlocal setups
        if not self.is_setup:
            setups += 1
        await original_setup(self)

    with (
        patch.object(AsyncSqliteSaver, "setup", counting_setup),
        patch(
            "agent.graph.litellm.acompletion",
            new=AsyncMock(side_effect=lambda **_: _text_stream("ok")),
        ),
    ):
        await asyncio.gather(
            *(Assistant("org", thread_id=f"org:{n}").chat("hi") for n in range(CONCURRENT_SESSIONS))
        )
        await Assistant("org", thread_id="org:later").chat("hi")

    assert setups == 1


async def test_a_failed_schema_setup_is_retried_by_the_next_turn(tmp_data_dir):
    import sqlite3

    from agent.assistant import Assistant

    original_setup = AsyncSqliteSaver.setup
    attempts = []

    async def flaky_setup(self):
        if not self.is_setup:
            attempts.append(1)
            if len(attempts) == 1:
                raise sqlite3.OperationalError("disk I/O error")
        await original_setup(self)

    with (
        patch.object(AsyncSqliteSaver, "setup", flaky_setup),
        patch(
            "agent.graph.litellm.acompletion",
            new=AsyncMock(side_effect=lambda **_: _text_stream("ok")),
        ),
    ):
        with pytest.raises(sqlite3.OperationalError):
            await Assistant("org").chat("first")
        assert await Assistant("org").chat("second") == "ok"


# --- Two people, one org --------------------------------------------------------------


async def test_two_browser_sessions_on_one_org_share_memory_but_not_conversation(tmp_data_dir):
    """Org memory (the ledger) is shared by design; what each person typed is not."""
    from agent.assistant import Assistant

    alice = Assistant("food-bank", thread_id="food-bank:alice")
    bob = Assistant("food-bank", thread_id="food-bank:bob")
    args = json.dumps({"platform": "general", "content": "x", "title": "Shared gala post"})
    first = AsyncMock(
        side_effect=[_tool_call_stream("tc1", "save_post", args), _text_stream("Saved")]
    )
    with patch("agent.graph.litellm.acompletion", new=first):
        await alice.chat("Alice's private draft idea about the board")

    second = AsyncMock(return_value=_text_stream("Hi Bob"))
    with patch("agent.graph.litellm.acompletion", new=second):
        await bob.chat("What's new?")

    sent = second.call_args.kwargs["messages"]
    system, conversation = sent[0], sent[1:]
    assert "Shared gala post" in json.dumps(system)  # the ledger is org memory
    assert [m["content"] for m in conversation] == ["What's new?"]  # nothing of Alice's


async def test_the_cli_and_web_sessions_on_one_org_keep_separate_histories(tmp_data_dir):
    """Neither direction leaks: the web model sees nothing typed in the terminal, and back."""
    from agent.assistant import Assistant

    cli, web = Assistant("org"), Assistant("org", thread_id="org:browser")
    with patch("agent.graph.litellm.acompletion", new=AsyncMock(return_value=_text_stream("ok"))):
        await cli.chat("typed in the terminal")

    web_model = AsyncMock(return_value=_text_stream("ok"))
    with patch("agent.graph.litellm.acompletion", new=web_model):
        await web.chat("typed in the browser")
    assert [m["content"] for m in web_model.call_args.kwargs["messages"][1:]] == [
        "typed in the browser"
    ]

    cli_model = AsyncMock(return_value=_text_stream("ok"))
    with patch("agent.graph.litellm.acompletion", new=cli_model):
        await cli.chat("back in the terminal")
    sent = [m["content"] for m in cli_model.call_args.kwargs["messages"][1:]]
    assert sent == ["typed in the terminal", "ok", "back in the terminal"]


async def test_simultaneous_writes_to_one_org_lose_nothing(tmp_data_dir):
    """Two people saving at the same moment in one process: every post kept, ids unique."""
    from agent.memory import get_ledger
    from agent.tools import handle_tool_call

    await asyncio.gather(
        *(
            handle_tool_call(
                "save_post", {"platform": "general", "content": "x", "title": f"T{n}"}, "food-bank"
            )
            for n in range(SIMULTANEOUS_WRITES)
        )
    )
    ledger = get_ledger("food-bank")
    assert len(ledger) == SIMULTANEOUS_WRITES
    assert len({p["id"] for p in ledger}) == SIMULTANEOUS_WRITES


async def test_interleaved_streams_reach_only_their_own_session(tmp_data_dir):
    """Alice's reply pauses mid-stream while Bob's streams in full; neither gets the other's tokens."""
    from agent.assistant import Assistant

    alice_started, bob_finished = asyncio.Event(), asyncio.Event()

    async def model(**kwargs):
        if _last_user_message(kwargs["messages"]) == "from Alice":
            stream = _gated_stream(["A1", "A2"], {1: bob_finished})
        else:
            stream = _gated_stream(["B1", "B2"], {0: alice_started})
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


# --- A session can only touch its own org ---------------------------------------------


async def test_a_model_naming_another_org_in_tool_arguments_still_writes_to_its_own(tmp_data_dir):
    """The org comes from the session, never from the model's arguments."""
    from agent.assistant import Assistant
    from agent.memory import get_ledger

    args = json.dumps(
        {"platform": "general", "content": "x", "title": "Injected", "org_id": "victim-org"}
    )
    model = AsyncMock(side_effect=[_tool_call_stream("tc1", "save_post", args), _text_stream("ok")])
    with patch("agent.graph.litellm.acompletion", new=model):
        await Assistant("attacker-org").chat("save it")

    assert [p["title"] for p in get_ledger("attacker-org")] == ["Injected"]
    assert not (tmp_data_dir / "victim-org").exists()


# --- Separate processes on one org (web UI, CLI, scripts) ------------------------------


def test_separate_processes_writing_to_one_org_lose_nothing(tmp_data_dir):
    """
    Regression: the web UI and the CLI on one org overwrote each other's posts
    (50 of 100 lost), and a shared temporary file name crashed one of them.
    """
    from agent.memory import get_ledger

    writer = (
        "import sys; from pathlib import Path; import agent.memory as mem\n"
        "mem.DATA_DIR = Path(sys.argv[1])\n"
        f"for n in range({POSTS_PER_PROCESS}):\n"
        "    mem.add_post('org', {'platform': 'general', 'content': 'x', 'title': sys.argv[2] + str(n)})"
    )
    processes = [
        subprocess.Popen(
            [sys.executable, "-c", writer, str(tmp_data_dir), f"P{n}-"],
            cwd=REPO_ROOT,
            stderr=subprocess.PIPE,
            text=True,
        )
        for n in range(WRITER_PROCESSES)
    ]
    errors = [process.communicate(timeout=30)[1] for process in processes]

    assert [process.returncode for process in processes] == [0] * WRITER_PROCESSES, errors
    ledger = get_ledger("org")
    assert len(ledger) == WRITER_PROCESSES * POSTS_PER_PROCESS
    assert len({post["id"] for post in ledger}) == len(ledger)


def _seed_org_for_writers() -> None:
    from agent.memory import add_post, load_profile, save_preference, save_profile

    add_post("org", {"platform": "linkedin", "content": "x", "title": "Existing"})
    save_preference("org", "voice", "warm")
    save_profile("org", load_profile("org") | {"research": {"sources": []}})


def _run_tool(tool_name: str, arguments: dict) -> None:
    from agent.tools import handle_tool_call

    asyncio.run(handle_tool_call(tool_name, arguments, "org"))


RESEARCH = {"website_content": "", "social_presence": [], "news": [], "similar_orgs": []}
WRITERS = {
    "add_post": lambda mem: mem.add_post("org", {"platform": "general", "content": "x"}),
    "update_post_status": lambda mem: mem.update_post_status("org", "post_001", "draft"),
    "record_performance": lambda mem: mem.record_performance("org", "post_001", {"reach": 1}),
    "save_preference": lambda mem: mem.save_preference("org", "tone", "plain"),
    "forget_preference": lambda mem: mem.forget_preference("org", "voice"),
    "add_voice_example": lambda mem: mem.add_voice_example("org", "x", "pasted"),
    "fill_org_details": lambda mem: mem.fill_org_details("org", "Org", "https://org.example"),
    "research_org tool": lambda _: _run_tool("research_org", {"org_name": "Org"}),
    "web_search tool": lambda _: _run_tool("web_search", {"query": "q"}),
}


def test_every_profile_writer_waits_for_another_processs_lock(tmp_data_dir):
    """
    Hold the org's lock as another process would; every writer must wait for it.

    All writers start together in their own threads, so one wait covers them
    all, and once released, every write must land: none may overwrite another.
    """
    import agent.memory as mem

    _seed_org_for_writers()
    profile_path = tmp_data_dir / "org" / "profile.json"
    search = [{"title": "t", "snippet": "s", "url": "https://news.example/a"}]
    brief = AsyncMock(side_effect=RuntimeError("no brief"))
    with (
        patch("agent.tools.res.research_org", new=AsyncMock(return_value=dict(RESEARCH))),
        patch("agent.tools.res.search_web", new=AsyncMock(return_value=search)),
        patch("agent.tools.litellm.acompletion", new=brief),
        open(tmp_data_dir / "org" / mem.LOCK_FILE_NAME, "w") as other_process,
    ):
        fcntl.flock(other_process, fcntl.LOCK_EX)  # a separate open file, as another process has
        before = profile_path.read_text()
        workers = {
            name: threading.Thread(target=write, args=(mem,), daemon=True)
            for name, write in WRITERS.items()
        }
        for worker in workers.values():
            worker.start()
        time.sleep(STILL_BLOCKED_S)
        assert [name for name, worker in workers.items() if not worker.is_alive()] == []
        assert profile_path.read_text() == before
        fcntl.flock(other_process, fcntl.LOCK_UN)
        for worker in workers.values():
            worker.join(WAIT_S)
    assert [name for name, worker in workers.items() if worker.is_alive()] == []

    profile = mem.load_profile("org")
    assert len(profile["post_ledger"]) == 2 and "performance" in profile["post_ledger"][0]
    assert profile["preferences"] == {"tone": "plain"}
    assert len(profile["voice_examples"]) == 1
    assert profile["website"] == "https://org.example"
    assert profile["research"] is not None and "org_brief" in profile  # research saved too


async def test_research_finishing_late_keeps_posts_saved_while_it_ran(tmp_data_dir):
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

    assert [post["title"] for post in get_ledger("org")] == ["Saved meanwhile"]
    assert load_profile("org")["research"] is not None
