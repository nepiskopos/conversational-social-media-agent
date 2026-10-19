"""
Tests for the two user interfaces: app.py (Chainlit web UI) and main.py (terminal).

Owns: onboarding (parsing, confirmation, timeouts), the session greeting
(including the agenda of what needs the user, most urgent first), broken
memory files, a thread per browser session, streaming as tokens arrive, what
a tool step shows, plain-language errors, and the welcome copy matching what
Mackie does. Chainlit's UI objects and the terminal prompt are replaced with
fakes; the Assistant is mocked, so no LLM is called. Does not own
conversation logic (test_assistant.py) or the real browser (tests/browser/).
"""

import asyncio
import importlib
import sys
from io import StringIO
from pathlib import Path
from types import ModuleType
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from rich.console import Console


def _import_entry_point(name: str) -> ModuleType:
    """Import app.py or main.py without loading the real .env into the test process."""
    sys.modules.pop(name, None)
    with patch("dotenv.load_dotenv"):
        return importlib.import_module(name)


@pytest.fixture(scope="module")
def app():
    return _import_entry_point("app")


@pytest.fixture(scope="module")
def main_module():
    return _import_entry_point("main")


class FakeChainlit:
    """Records what the web UI sends; stands in for chainlit's Message, Step and session."""

    def __init__(self) -> None:
        self.sent: list[str] = []
        self.streamed: list[str] = []
        self.session: dict = {}
        self.steps: list[str] = []
        self.step_objects: list[MagicMock] = []

    def message(self, content: str = "", **_):
        fake = MagicMock()
        fake.send = AsyncMock(side_effect=lambda: self.sent.append(content))
        fake.stream_token = AsyncMock(side_effect=self.streamed.append)
        fake.update = AsyncMock()
        return fake

    def step(self, name: str, **_):
        self.steps.append(name)
        step = MagicMock()
        step.__aenter__ = AsyncMock(return_value=step)
        step.__aexit__ = AsyncMock(return_value=False)
        self.step_objects.append(step)
        return step


@pytest.fixture
def fake_cl(app):
    fake = FakeChainlit()
    session = MagicMock()
    session.get.side_effect = fake.session.get
    session.set.side_effect = fake.session.__setitem__
    with (
        patch.object(app.cl, "Message", side_effect=fake.message),
        patch.object(app.cl, "Step", side_effect=fake.step),
        patch.object(app.cl, "user_session", session),
    ):
        yield fake


def _asks(*replies):
    """A fake chainlit Ask*Message class whose send() returns the replies in turn."""
    asker = MagicMock()
    asker.return_value.send = AsyncMock(side_effect=list(replies))
    return asker


# --- Web UI: onboarding ---------------------------------------------------------------


@pytest.mark.parametrize(
    "text, expected",
    [
        ("Food Bank, https://food.example", ("Food Bank", "https://food.example")),
        ("Food Bank\nhttps://food.example\n", ("Food Bank", "https://food.example")),
        ("  Food Bank  ", ("Food Bank", "")),
        ("Smith, Jones & Co\nhttps://sj.example", ("Smith, Jones & Co", "https://sj.example")),
        ("", ("", "")),
    ],
)
def test_parse_org_input_splits_name_and_url(app, text, expected):
    assert app._parse_org_input(text) == expected


async def test_ask_for_org_reasks_for_a_missing_name_and_after_a_no(app):
    ask = _asks({"output": "!!!"}, {"output": "Wrong Org"}, {"output": "Food Bank, food.example"})
    confirm = _asks({"payload": {"ok": False}}, {"payload": {"ok": True}})
    with (
        patch.object(app.cl, "AskUserMessage", ask),
        patch.object(app.cl, "AskActionMessage", confirm),
    ):
        assert await app._ask_for_org() == ("Food Bank", "food.example")

    prompts = [call.kwargs["content"] for call in ask.call_args_list]
    assert "couldn't find an organization name" in prompts[1]
    assert "No problem" in prompts[2]


@pytest.mark.parametrize("replies, confirms", [([None], []), ([{"output": "Org"}], [None])])
async def test_ask_for_org_returns_none_on_timeout(app, replies, confirms):
    with (
        patch.object(app.cl, "AskUserMessage", _asks(*replies)),
        patch.object(app.cl, "AskActionMessage", _asks(*confirms)),
    ):
        assert await app._ask_for_org() is None


# --- Web UI: session start ------------------------------------------------------------


async def test_start_greets_a_new_org_and_saves_its_name(app, fake_cl, tmp_data_dir):
    from agent.memory import load_profile

    with patch.object(app, "_ask_for_org", AsyncMock(return_value=("Food Bank", "food.example"))):
        await app.start()

    assert "I'm set up for **Food Bank**" in fake_cl.sent[-1]
    assert "paste two or three of your recent posts" in fake_cl.sent[-1]
    assert load_profile("food-bank")["website"] == "food.example"


async def test_start_leads_a_returning_org_with_what_needs_attention(app, fake_cl, tmp_data_dir):
    from agent.memory import add_post

    add_post("food-bank", {"platform": "linkedin", "content": "x", "status": "draft"})
    with patch.object(app, "_ask_for_org", AsyncMock(return_value=("Food Bank", ""))):
        await app.start()

    assert fake_cl.sent[-1].startswith("Welcome back!")
    assert "1 draft waiting for your approval" in fake_cl.sent[-1]
    assert "I remember 0 brand preferences and 1 post for **Food Bank**" in fake_cl.sent[-1]


async def test_start_gives_each_browser_session_its_own_thread(app, fake_cl, tmp_data_dir):
    threads = set()
    for _ in range(2):
        with patch.object(app, "_ask_for_org", AsyncMock(return_value=("Food Bank", ""))):
            await app.start()
        threads.add(fake_cl.session["assistant"].thread_id)

    assert len(threads) == 2
    assert all(t.startswith("food-bank:") and t != "food-bank" for t in threads)


async def test_start_explains_a_broken_memory_file_and_stops(app, fake_cl, tmp_data_dir):
    from agent.memory import _profile_path

    _profile_path("food-bank").write_text("{not json")
    with patch.object(app, "_ask_for_org", AsyncMock(return_value=("Food Bank", ""))):
        await app.start()

    assert "can't read this organization's memory file" in fake_cl.sent[-1]
    assert "line 1" in fake_cl.sent[-1]
    assert "assistant" not in fake_cl.session


async def test_start_warns_about_skipped_skill_files(app, fake_cl, tmp_data_dir):
    from agent.skills import SkillSet

    broken = SkillSet(problems=["threads.yaml: missing required field(s) max_length"])
    with (
        patch.object(app, "_ask_for_org", AsyncMock(return_value=("Food Bank", ""))),
        patch.object(app.skills, "load_skills", return_value=broken),
    ):
        await app.start()

    assert "**Heads-up:**" in fake_cl.sent[0] and "threads.yaml" in fake_cl.sent[0]
    assert "I'm set up for" in fake_cl.sent[-1]  # the session still starts


async def test_start_handles_an_onboarding_timeout(app, fake_cl, tmp_data_dir):
    with patch.object(app, "_ask_for_org", AsyncMock(return_value=None)):
        await app.start()

    assert "timed out" in fake_cl.sent[-1]
    assert list(tmp_data_dir.iterdir()) == []


# --- Web UI: messages -----------------------------------------------------------------


async def _events(*events):
    for event in events:
        yield event


async def test_on_message_without_a_session_asks_for_a_refresh(app, fake_cl):
    await app.on_message(MagicMock(content="hi"))
    assert "refresh the page" in fake_cl.sent[-1]


# --- Terminal UI ----------------------------------------------------------------------


async def _run_cli(main_module, answers, assistant=None) -> tuple[str, MagicMock]:
    """Run main() with scripted prompt answers; return (terminal output, Assistant mock)."""
    output = StringIO()
    assistant = assistant or MagicMock(
        chat=AsyncMock(return_value="**Hello** from Mackie"), clear_history=AsyncMock()
    )
    with (
        patch.object(main_module, "console", Console(file=output, width=120)),
        patch.object(main_module.Prompt, "ask", side_effect=answers),
        patch.object(main_module, "Assistant", return_value=assistant),
    ):
        await main_module.main()
    return output.getvalue(), assistant


async def test_cli_runs_a_conversation_and_resets_history(main_module, tmp_data_dir):
    from agent.memory import load_profile

    text, assistant = await _run_cli(
        main_module, ["Food Bank", "food.example", "", "reset", "hello", "quit"]
    )

    assistant.clear_history.assert_awaited_once()
    assistant.chat.assert_awaited_once_with("hello")
    assert "Hello from Mackie" in text
    assert "Org memory and ledger are preserved" in text
    assert load_profile("food-bank")["name"] == "Food Bank"


async def test_cli_reasks_a_blank_name(main_module, tmp_data_dir):
    text, _ = await _run_cli(main_module, ["  ", "!!", "Food Bank", "", "q"])
    assert "Working with Food Bank" in text


async def test_cli_shows_what_needs_attention_at_start(main_module, tmp_data_dir):
    from agent.memory import add_post

    add_post("food-bank", {"platform": "facebook", "content": "x", "status": "approved"})
    text, _ = await _run_cli(main_module, ["Food Bank", "", "exit"])
    assert "1 approved post ready to publish" in text


async def test_cli_reports_a_failed_turn_and_carries_on(main_module, tmp_data_dir):
    assistant = MagicMock(chat=AsyncMock(side_effect=[RuntimeError("rate limited"), "Back"]))
    text, _ = await _run_cli(main_module, ["Food Bank", "", "one", "two", "quit"], assistant)
    assert "Error: rate limited" in text and "Back" in text


async def test_cli_exits_cleanly_on_ctrl_c(main_module, tmp_data_dir):
    text, _ = await _run_cli(main_module, ["Food Bank", "", KeyboardInterrupt()])
    assert "Goodbye" in text


async def test_cli_explains_a_broken_memory_file(main_module, tmp_data_dir):
    from agent.memory import _profile_path

    _profile_path("food-bank").write_text("[]")
    text, assistant = await _run_cli(main_module, ["Food Bank", ""])
    assert "Can't read this organization's memory file" in text
    assistant.chat.assert_not_awaited()


# --- UX -------------------------------------------------------------------------------


async def test_web_ui_shows_the_first_token_before_the_reply_is_finished(app, fake_cl):
    """The second token only arrives once the first is on screen; a buffering UI would hang."""

    async def stream(_):
        yield {"type": "text", "content": "Hel"}
        for _ in range(100):
            if fake_cl.streamed:
                break
            await asyncio.sleep(0)
        yield {"type": "text", "content": "lo" if fake_cl.streamed == ["Hel"] else "BUFFERED"}

    fake_cl.session["assistant"] = MagicMock(achat_stream=stream)
    await app.on_message(MagicMock(content="hi"))
    assert fake_cl.streamed == ["Hel", "lo"]


async def test_tool_steps_show_what_was_done_with_the_result_and_time(app, fake_cl):
    async def stream(_):
        yield {
            "type": "tool_step",
            "tool": "record_performance",
            "input_preview": '{"post_id": "post_004", "reactions": 120}',
            "result_preview": "Recorded results for post_004",
            "latency_ms": 7,
        }

    fake_cl.session["assistant"] = MagicMock(achat_stream=stream)
    await app.on_message(MagicMock(content="it got 120 likes"))

    step = fake_cl.step_objects[0]
    assert fake_cl.steps == ["record_performance"]
    assert step.input == '{"post_id": "post_004", "reactions": 120}'
    assert step.output == "Recorded results for post_004  (7ms)"


async def test_tool_step_previews_stay_short_however_large_the_tool_input(tmp_data_dir):
    """A 50 KB email draft shouldn't fill the screen inside a collapsed step."""
    import json

    from agent.assistant import Assistant
    from tests.test_assistant import _text_stream, _tool_call_stream

    args = json.dumps({"platform": "general", "content": "x" * 50_000, "title": "Long"})
    model = AsyncMock(side_effect=[_tool_call_stream("tc1", "save_post", args), _text_stream("ok")])
    with patch("agent.graph.litellm.acompletion", new=model):
        steps = [e async for e in Assistant("org").achat_stream("save") if e["type"] == "tool_step"]
    assert len(steps[0]["input_preview"]) <= 300
    assert len(steps[0]["result_preview"]) <= 500


async def test_errors_are_shown_in_plain_words_with_a_way_forward(
    app, fake_cl, main_module, tmp_data_dir
):
    """No tracebacks or exception class names in front of the user, and a next step."""

    async def failing(_):
        raise TimeoutError("the AI provider took too long")
        yield  # pragma: no cover — makes this an async generator

    fake_cl.session["assistant"] = MagicMock(achat_stream=failing)
    await app.on_message(MagicMock(content="hi"))
    web = fake_cl.streamed[-1]

    assistant = MagicMock(chat=AsyncMock(side_effect=TimeoutError("the AI provider took too long")))
    cli, _ = await _run_cli(main_module, ["Food Bank", "", "hi", "quit"], assistant)

    for shown in (web, cli):
        assert "the AI provider took too long" in shown
        assert "Traceback" not in shown and "TimeoutError" not in shown
    assert "Please try again" in web


async def test_greeting_lists_the_most_urgent_work_first(app, fake_cl, tmp_data_dir):
    """Approvals unblock publishing, so they come before planned posts and results."""
    from agent.memory import add_post

    for status in ("posted", "planned", "approved", "draft"):
        add_post(
            "food-bank",
            {
                "platform": "linkedin",
                "content": "x",
                "status": status,
                "scheduled_for": "2020-01-01",
            },
        )
    with patch.object(app, "_ask_for_org", AsyncMock(return_value=("Food Bank", ""))):
        await app.start()

    greeting = fake_cl.sent[-1]
    order = ["waiting for your approval", "ready to publish", "to write up", "report results"]
    positions = [greeting.index(phrase) for phrase in order]
    assert positions == sorted(positions)


def test_welcome_copy_matches_what_mackie_does(app):
    """The first thing a user reads names Mackie and its plan, create, optimize job."""
    welcome = (Path(__file__).parent.parent / "chainlit.md").read_text()
    assert "Mackie" in app.ORG_PROMPT and "social media manager" in app.ORG_PROMPT
    for promise in ("**Plan**", "**Create**", "**Optimize**", "approval"):
        assert promise in welcome
