"""
Tests for the two user interfaces: app.py (Chainlit web UI) and main.py (terminal).

Owns: the login on both interfaces, onboarding (parsing, confirmation,
timeouts, which orgs a user may open), the session greeting (leading with
the agenda of what needs the user), one resumable conversation per user and
org (the recap shown on return, cut and noting an unanswered message; "Start
a new conversation", also offered when a conversation is too long), streaming
as tokens arrive, what a tool step shows, a warning for skipped skill files,
plain-language errors on screen (an unreachable database, a database blip that
fails one message), removed access ending an open session, and the welcome
copy matching what Mackie does. Chainlit's UI objects and the terminal prompt
are replaced with fakes; the Assistant or the model (tests/llm_fakes.py) is
mocked, so no LLM is called. Does
not own conversation logic or the agenda's order (test_assistant.py), tool
step previews (test_graph.py), which message each error maps to
(test_errors.py), account rules (test_accounts.py) or the real browser
(tests/browser/).
"""

import importlib
import sys
from io import StringIO
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import litellm
import psycopg
import psycopg_pool
import pytest
from rich.console import Console

from agent import accounts
from agent.db import DatabaseError
from agent.errors import (
    NEW_CONVERSATION_LABEL,
    UNAVAILABLE_MESSAGE,
    UNEXPECTED_MESSAGE,
    user_message,
)
from tests.llm_fakes import text_stream

USERNAME = "sam@food.example"
PASSWORD = "correct horse battery"
# Hashing at the production cost takes about 50 ms; the stored hash records its own cost.
# The driver's own message names the server; none of it may reach the screen.
DRIVER_DOWN = 'connection to server at "10.0.0.5", port 5432 failed: Connection refused'


def _import_entry_point(name: str) -> ModuleType:
    """Import app.py or main.py without loading the real .env into the test process."""
    sys.modules.pop(name, None)
    with patch("agent.config.load_dotenv"):
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
        # Each sent message as {"content", "author", "type", "actions"}.
        self.messages: list[dict] = []
        self.streamed: list[str] = []
        self.session: dict = {}
        self.steps: list[str] = []
        self.step_objects: list[MagicMock] = []

    def message(self, content: str = "", author: str = "Mackie", type: str = "", actions=()):
        fake = MagicMock()

        def send() -> None:
            self.sent.append(content)
            self.messages.append(
                {"content": content, "author": author, "type": type, "actions": list(actions)}
            )

        fake.send = AsyncMock(side_effect=send)
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


@pytest.fixture
async def user(database):
    """A Mackie account (USERNAME, PASSWORD), hashed at a low cost to keep the tests fast."""
    yield await accounts.create_user(USERNAME, PASSWORD)


@pytest.fixture
def signed_in(fake_cl, user):
    """The web session of a logged-in user, as Chainlit stores it after login()."""
    fake_cl.session["user"] = SimpleNamespace(
        metadata={"user_id": user.user_id, "session_key": user.session_key}
    )
    return user


async def _other_user_opens(org_id: str) -> None:
    """Another account opens org_id first, so it is that account's org."""
    other = await accounts.create_user("alex@food.example", PASSWORD)
    assert await accounts.open_org(other.user_id, org_id)


def _picks(app, org_name: str, org_url: str = ""):
    """Onboarding answers org_name and opens it for the user, as _ask_for_org does."""

    async def ask_for_org(user_id: int) -> tuple[str, str]:
        assert await accounts.open_org(user_id, app.mem.slugify(org_name))
        return org_name, org_url

    return patch.object(app, "_ask_for_org", ask_for_org)


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
        # Regression: only a website is taken as one; "Inc." became the org's website.
        ("Feed the Children, Inc.", ("Feed the Children, Inc.", "")),
        (
            "Feed the Children, Inc., feed.example",
            ("Feed the Children, Inc.", "https://feed.example"),
        ),
        ("Food Bank\nour site", ("Food Bank", "")),
        # Regression: these became one long org name, website included.
        ("BRCA Strong, brcastrong.org.", ("BRCA Strong", "https://brcastrong.org")),
        ("BRCA Strong (brcastrong.org)", ("BRCA Strong", "https://brcastrong.org")),
        ("BRCA Strong brcastrong.org", ("BRCA Strong", "https://brcastrong.org")),
        ("BRCA Strong — brcastrong.org", ("BRCA Strong", "https://brcastrong.org")),
        ("Code.org", ("Code.org", "")),  # a name that looks like a website stays the name
        ("", ("", "")),
    ],
)
def test_parse_org_input_splits_name_and_url(app, text, expected):
    assert app._parse_org_input(text) == expected


async def test_login_returns_the_user_only_for_the_right_password(app, user):
    logged_in = await app.login(USERNAME, PASSWORD)
    assert logged_in.identifier == USERNAME
    assert logged_in.metadata == {"user_id": user.user_id, "session_key": user.session_key}
    assert await app.login(USERNAME, "wrong password!") is None


async def test_ask_for_org_reasks_for_a_missing_name_and_after_a_no(app, user):
    ask = _asks({"output": "!!!"}, {"output": "Wrong Org"}, {"output": "Food Bank, food.example"})
    confirm = _asks({"payload": {"ok": False}}, {"payload": {"ok": True}})
    with (
        patch.object(app.cl, "AskUserMessage", ask),
        patch.object(app.cl, "AskActionMessage", confirm),
    ):
        assert await app._ask_for_org(user.user_id) == ("Food Bank", "https://food.example")

    prompts = [call.kwargs["content"] for call in ask.call_args_list]
    assert "couldn't find an organization name" in prompts[1]
    assert "No problem" in prompts[2]


@pytest.mark.parametrize("replies, confirms", [([None], []), ([{"output": "Org"}], [None])])
async def test_ask_for_org_returns_none_on_timeout(app, user, replies, confirms):
    with (
        patch.object(app.cl, "AskUserMessage", _asks(*replies)),
        patch.object(app.cl, "AskActionMessage", _asks(*confirms)),
    ):
        assert await app._ask_for_org(user.user_id) is None


async def test_ask_for_org_lists_the_users_orgs(app, user):
    for org_id in ("food-bank", "animal-shelter"):
        await accounts.open_org(user.user_id, org_id)
    ask = _asks(None)
    with patch.object(app.cl, "AskUserMessage", ask):
        await app._ask_for_org(user.user_id)
    assert (
        "Your organizations: **animal-shelter**, **food-bank**" in ask.call_args.kwargs["content"]
    )


async def test_ask_for_org_refuses_someone_elses_org_and_opens_a_new_one(app, user):
    await _other_user_opens("food-bank")
    ask = _asks({"output": "Food Bank"}, {"output": "Animal Shelter, shelter.example"})
    confirm = _asks({"payload": {"ok": True}}, {"payload": {"ok": True}})
    with (
        patch.object(app.cl, "AskUserMessage", ask),
        patch.object(app.cl, "AskActionMessage", confirm),
    ):
        assert await app._ask_for_org(user.user_id) == ("Animal Shelter", "https://shelter.example")

    assert ask.call_args.kwargs["content"] == app.NO_ACCESS_PROMPT.format(org_name="Food Bank")
    assert await accounts.user_orgs(user.user_id) == ["animal-shelter"]


# --- Web UI: session start ------------------------------------------------------------


async def test_start_greets_a_new_org_and_saves_its_name(app, fake_cl, signed_in):
    from agent.memory import load_profile

    with _picks(app, "Food Bank", "food.example"):
        await app.start()

    assert "I'm set up for **Food Bank**" in fake_cl.sent[-1]
    assert "paste two or three of your recent posts" in fake_cl.sent[-1]
    assert (await load_profile("food-bank"))["website"] == "food.example"


async def test_start_leads_a_returning_org_with_what_needs_attention(app, fake_cl, signed_in):
    from agent.memory import add_post

    assert await accounts.open_org(signed_in.user_id, "food-bank")
    await add_post("food-bank", {"platform": "linkedin", "content": "x", "status": "draft"})
    with _picks(app, "Food Bank"):
        await app.start()

    assert fake_cl.sent[-1].startswith("Welcome back!")
    assert "1 draft waiting for your approval" in fake_cl.sent[-1]
    assert "For **Food Bank** I remember 0 brand preferences and 1 post." in fake_cl.sent[-1]


async def test_start_asks_what_to_work_on_when_nothing_needs_attention(app, fake_cl, signed_in):
    from agent.memory import save_preference

    assert await accounts.open_org(signed_in.user_id, "food-bank")
    await save_preference("food-bank", "tone", "warm")
    with _picks(app, "Food Bank"):
        await app.start()

    assert fake_cl.sent[-1].startswith(
        "Welcome back! For **Food Bank** I remember 1 brand preference and 0 posts."
    )
    assert fake_cl.sent[-1].endswith("What would you like to work on?")


async def test_start_resumes_one_conversation_per_user_and_org(app, fake_cl, signed_in):
    """Two sessions of one user and org share a thread; another user of the org has their own."""
    threads = []
    for _ in range(2):
        with _picks(app, "Food Bank"):
            await app.start()
        threads.append(fake_cl.session["assistant"].thread_id)
    alex = await accounts.create_user("alex@food.example", PASSWORD)
    await accounts.grant("alex@food.example", "food-bank")
    fake_cl.session["user"] = SimpleNamespace(
        metadata={"user_id": alex.user_id, "session_key": alex.session_key}
    )
    with _picks(app, "Food Bank"):
        await app.start()
    threads.append(fake_cl.session["assistant"].thread_id)

    assert threads == [
        f"food-bank:web-user-{signed_in.user_id}",
        f"food-bank:web-user-{signed_in.user_id}",
        f"food-bank:web-user-{alex.user_id}",
    ]


RECAP = [("user", "Draft a post"), ("assistant", "Here it is"), ("user", "Shorter")]


@pytest.mark.parametrize(
    "recap", [RECAP, [], RuntimeError("recap query failed")], ids=["history", "new", "unreadable"]
)
async def test_start_shows_where_a_resumed_conversation_left_off(app, fake_cl, signed_in, recap):
    """An unreadable recap is a courtesy lost: the plain greeting, and the session works."""
    recent = AsyncMock(side_effect=[recap])
    with _picks(app, "Food Bank"), patch.object(app.Assistant, "recent_exchanges", recent):
        await app.start()

    recent.assert_awaited_once_with(app.RESUMED_MESSAGES_SHOWN)
    assert fake_cl.session["assistant"] is not None
    greeting, *rest = fake_cl.messages
    resumed = "We were in the middle of a conversation" in greeting["content"]
    assert resumed == (recap == RECAP)
    if recap != RECAP:
        assert greeting["content"].startswith("Got it! I'm set up for **Food Bank**")
        assert rest == []
        return
    *shown, last = rest
    assert [(m["author"], m["type"], m["content"]) for m in shown] == [
        ("You", "user_message", "Draft a post"),
        ("Mackie", "assistant_message", "Here it is"),
        ("You", "user_message", "Shorter"),
    ]
    assert all(not m["actions"] for m in [greeting, *shown])
    [action] = last["actions"]
    assert action.name == app.NEW_CONVERSATION_ACTION
    assert action.label == NEW_CONVERSATION_LABEL


@pytest.mark.parametrize("last_speaker", ["user", "assistant"])
async def test_a_recap_cuts_long_messages_and_notes_an_unanswered_one(
    app, fake_cl, signed_in, last_speaker
):
    """A pasted calendar mustn't fill the screen; a last message with no reply is pointed out."""
    limit = app.RECAP_MESSAGE_CHARS
    recap = [("user", "a" * limit), ("assistant", "b" * (limit + 1))]
    recap += [("user", "Shorter")] if last_speaker == "user" else []
    with (
        _picks(app, "Food Bank"),
        patch.object(app.Assistant, "recent_exchanges", AsyncMock(return_value=recap)),
    ):
        await app.start()

    _greeting, *shown, closing = fake_cl.sent
    assert shown[:2] == ["a" * limit, "b" * limit + " …"]
    assert closing.startswith(app.UNANSWERED_NOTE) == (last_speaker == "user")


async def test_start_ends_the_session_if_access_is_gone_when_the_recap_loads(
    app, fake_cl, signed_in
):
    # Onboarding answers, but the membership is gone by the time the conversation is read.
    with patch.object(app, "_ask_for_org", AsyncMock(return_value=("Food Bank", ""))):
        await app.start()

    assert fake_cl.sent[-1] == (
        "You no longer have access to food-bank, or your password was changed. "
        "Refresh the page to choose another organization."
    )
    assert fake_cl.session["assistant"] is None


async def test_start_warns_about_skipped_skill_files(app, fake_cl, signed_in):
    from agent.skills import SkillSet

    broken = SkillSet(problems=["threads.yaml: missing required field(s) max_length"])
    with (
        _picks(app, "Food Bank"),
        patch.object(app.skills, "load_skills", return_value=broken),
    ):
        await app.start()

    assert "**Heads-up:**" in fake_cl.sent[0] and "threads.yaml" in fake_cl.sent[0]
    assert "I'm set up for" in fake_cl.sent[-1]  # the session still starts


async def test_start_handles_an_onboarding_timeout(app, fake_cl, signed_in):
    with patch.object(app, "_ask_for_org", AsyncMock(return_value=None)):
        await app.start()

    assert "timed out" in fake_cl.sent[-1]
    assert "assistant" not in fake_cl.session


@pytest.mark.parametrize("ended_by", ["deleted-account", "password-changed", "no-login"])
async def test_start_refuses_a_login_that_has_ended(app, fake_cl, user, ended_by):
    """A changed password ends logins made with the old one, which may be an attacker's."""
    if ended_by != "no-login":
        fake_cl.session["user"] = SimpleNamespace(
            metadata={"user_id": user.user_id, "session_key": user.session_key}
        )
    if ended_by == "deleted-account":
        await accounts.delete_user(USERNAME)
    if ended_by == "password-changed":
        await accounts.set_password(USERNAME, "a brand new passphrase")
    ask_for_org = AsyncMock()
    with patch.object(app, "_ask_for_org", ask_for_org):
        await app.start()

    assert fake_cl.sent[-1] == app.SESSION_ENDED_MESSAGE
    ask_for_org.assert_not_awaited()
    assert "assistant" not in fake_cl.session


# --- Web UI: messages -----------------------------------------------------------------


async def test_removing_access_ends_an_open_web_session(app, fake_cl, signed_in):
    assert await accounts.open_org(signed_in.user_id, "food-bank")
    with _picks(app, "Food Bank"):
        await app.start()
    assert fake_cl.session["assistant"].user_id == signed_in.user_id

    await accounts.revoke(USERNAME, "food-bank")
    await app.on_message(MagicMock(content="draft a post"))
    assert "You no longer have access to food-bank" in "".join(fake_cl.streamed)
    assert "Something went wrong" not in "".join(fake_cl.streamed)
    assert fake_cl.session["assistant"] is None
    await app.on_message(MagicMock(content="hello?"))
    assert "refresh the page" in fake_cl.sent[-1]


@pytest.mark.parametrize(
    "error, shown, session_ends",
    [
        (None, "Started a new conversation. What would you like to work on?", False),
        (
            accounts.AccessRevoked("You no longer have access to food-bank."),
            "You no longer have access to food-bank. Refresh the page to choose another organization.",
            True,
        ),
        (
            psycopg.OperationalError(DRIVER_DOWN),
            f"**Something went wrong:** {UNAVAILABLE_MESSAGE}",
            False,
        ),
    ],
    ids=["cleared", "access-revoked", "database-down"],
)
async def test_a_new_conversation_is_started_or_its_failure_explained(
    app, fake_cl, error, shown, session_ends
):
    """The button goes once the history is cleared, and stays to try again if it wasn't."""
    assistant = MagicMock(org_id="food-bank", clear_history=AsyncMock(side_effect=error))
    fake_cl.session["assistant"] = assistant
    action = MagicMock(remove=AsyncMock())

    await app.start_new_conversation(action)

    assistant.clear_history.assert_awaited_once()
    assert fake_cl.sent == [shown]
    assert (fake_cl.session["assistant"] is None) == session_ends
    assert action.remove.await_count == (error is None)


async def test_a_new_conversation_without_a_session_asks_for_a_refresh(app, fake_cl):
    action = MagicMock(remove=AsyncMock())
    await app.start_new_conversation(action)
    assert "refresh the page" in fake_cl.sent[-1]
    action.remove.assert_not_awaited()


# --- Terminal UI ----------------------------------------------------------------------


LOGIN = [USERNAME, PASSWORD]


def _cli_assistant(**methods) -> MagicMock:
    """A mocked Assistant for the CLI: a new conversation unless recent_exchanges is given."""
    defaults = {"clear_history": AsyncMock(), "recent_exchanges": AsyncMock(return_value=[])}
    return MagicMock(**{**defaults, **methods})


async def _run_cli(main_module, answers, assistant=None) -> tuple[str, MagicMock]:
    """
    Run main() with scripted prompt answers; return (terminal output, Assistant class mock).

    The mock's return_value is the Assistant the conversation used; its calls
    are Assistant.for_user(org_id, user, "cli").
    """
    output = StringIO()
    assistant = assistant or _cli_assistant(chat=AsyncMock(return_value="**Hello** from Mackie"))
    with (
        patch.object(main_module, "console", Console(file=output, width=120)),
        patch.object(main_module.Prompt, "ask", side_effect=answers),
        patch.object(main_module.Assistant, "for_user", return_value=assistant) as for_user,
    ):
        await main_module.main()
    return output.getvalue(), for_user


async def test_cli_runs_a_conversation_and_resets_history(main_module, user):
    from agent.memory import load_profile

    text, assistant_class = await _run_cli(
        main_module, [*LOGIN, "Food Bank", "our site", "food.example", "", "reset", "hello", "quit"]
    )

    assistant = assistant_class.return_value
    assistant.clear_history.assert_awaited_once()
    assistant.chat.assert_awaited_once_with("hello")
    assert "Hello from Mackie" in text
    assert "Org memory and ledger are preserved" in text
    profile = await load_profile("food-bank")
    assert profile["name"] == "Food Bank"
    assert profile["website"] == "https://food.example"  # "our site" was asked again


async def test_cli_shows_an_org_name_with_brackets_as_text(main_module, user):
    """Regression: rich read "[/b]" in a name as markup and the session crashed."""
    text, _ = await _run_cli(main_module, [*LOGIN, "Acme [/b] Trust", "", "quit"])
    assert "Working with Acme [/b] Trust" in text


async def test_cli_prints_no_terminal_control_codes(main_module, user):
    """Regression: ESC codes in a reply could hide text (conceal) or move the cursor over it."""
    reply = "Join us!\x1b[8m wire money\x1b[0m See you.\x07"
    assistant = _cli_assistant(chat=AsyncMock(return_value=reply))
    text, _ = await _run_cli(main_module, [*LOGIN, "Food Bank", "", "hi", "quit"], assistant)
    assert "\x1b" not in text and "\x07" not in text
    assert "wire money" in text


async def test_cli_logs_in_after_a_wrong_password(main_module, user):
    text, _ = await _run_cli(
        main_module, [USERNAME, "wrong password!", *LOGIN, "Food Bank", "", "q"]
    )
    assert "Wrong username or password" in text
    assert "Working with Food Bank" in text


async def test_cli_gives_up_after_too_many_wrong_passwords(main_module, user):
    answers = [USERNAME, "wrong password!"] * main_module.LOGIN_ATTEMPTS
    text, assistant_class = await _run_cli(main_module, answers)
    assert text.count("Wrong username or password") == main_module.LOGIN_ATTEMPTS
    assistant_class.assert_not_called()


async def test_cli_refuses_an_org_the_user_cant_open(main_module, user):
    await _other_user_opens("food-bank")
    # No website answer: the URL is only asked for once access is granted.
    text, assistant_class = await _run_cli(main_module, [*LOGIN, "Food Bank"])
    assert "You don't have access to Food Bank" in text
    assistant_class.assert_not_called()


async def test_cli_gives_each_user_their_own_conversation(main_module, user):
    """Assistant.for_user names the thread (test_assistant.py); each user gets their own."""
    await _other_user_opens("food-bank")
    await accounts.grant(USERNAME, "food-bank")
    opened = {}
    for username in (USERNAME, "alex@food.example"):
        text, for_user = await _run_cli(main_module, [username, PASSWORD, "Food Bank", "", "q"])
        org_id, logged_in, interface = for_user.call_args.args
        opened[username] = (org_id, logged_in.user_id, interface)
        assert "Your organizations: food-bank" in text

    assert opened[USERNAME] == ("food-bank", user.user_id, "cli")
    assert len({user_id for _, user_id, _ in opened.values()}) == 2


@pytest.mark.parametrize(
    "recap",
    [[("user", "hi")], [], RuntimeError("recap query failed")],
    ids=["history", "new", "unreadable"],
)
async def test_cli_says_when_it_carries_on_from_the_last_conversation(main_module, user, recap):
    """An unreadable history only loses the hint: the session goes on."""
    assistant = _cli_assistant(
        chat=AsyncMock(return_value="Back"), recent_exchanges=AsyncMock(side_effect=[recap])
    )
    text, _ = await _run_cli(main_module, [*LOGIN, "Food Bank", "", "hello", "q"], assistant)
    carrying_on = "Carrying on from your last conversation; 'reset' starts a new one." in text
    assert carrying_on == (recap == [("user", "hi")])
    assistant.recent_exchanges.assert_awaited_once_with(1)
    assistant.chat.assert_awaited_once_with("hello")
    assert "Traceback" not in text and "Error" not in text


async def test_cli_reasks_a_blank_name(main_module, user):
    text, _ = await _run_cli(main_module, [*LOGIN, "  ", "!!", "Food Bank", "", "q"])
    assert "Working with Food Bank" in text


async def test_cli_shows_what_needs_attention_at_start(main_module, user):
    from agent.memory import add_post

    await accounts.open_org(user.user_id, "food-bank")
    await add_post("food-bank", {"platform": "facebook", "content": "x", "status": "approved"})
    text, _ = await _run_cli(main_module, [*LOGIN, "Food Bank", "", "exit"])
    assert "1 approved post ready to publish" in text


# What can end a turn, where the UIs differ: a bug (any exception), and a conversation too long
# for the model, which every later turn would hit too. What each error says, and that a
# provider's own text never shows, is test_errors.py's.
TURN_ERRORS = [
    pytest.param(RuntimeError("rate limited at 10.0.0.5"), id="unexpected"),
    pytest.param(
        litellm.ContextWindowExceededError(
            "prompt is too long: 250000 tokens > 200000 maximum",
            model="claude-haiku-5-5",
            llm_provider="anthropic",
        ),
        id="context-window",
    ),
]


def _assert_shown_in_plain_words(shown: str, error: Exception) -> None:
    """The user sees user_message(error), and none of the error's own text, class or key."""
    expected = UNEXPECTED_MESSAGE if isinstance(error, RuntimeError) else user_message(error)
    assert expected in " ".join(shown.split())  # the terminal may wrap the line
    own_text = str(error).split(": ", 1)[-1]
    for leak in (own_text, type(error).__name__, "litellm", "{", "Traceback"):
        assert leak not in shown


@pytest.mark.parametrize("error", TURN_ERRORS)
async def test_cli_reports_a_failed_turn_and_carries_on(main_module, user, error):
    assistant = _cli_assistant(chat=AsyncMock(side_effect=[error, "Back"]))
    text, _ = await _run_cli(
        main_module, [*LOGIN, "Food Bank", "", "one", "two", "quit"], assistant
    )
    _assert_shown_in_plain_words(text, error)
    assert "Error:" in text and "Back" in text


async def test_cli_reports_a_failed_reset_and_carries_on(main_module, user):
    assistant = _cli_assistant(
        chat=AsyncMock(return_value="Back"),
        clear_history=AsyncMock(side_effect=psycopg.OperationalError("server closed")),
    )
    text, _ = await _run_cli(
        main_module, [*LOGIN, "Food Bank", "", "reset", "hi", "quit"], assistant
    )
    assert f"Error: {UNAVAILABLE_MESSAGE}" in " ".join(text.split())
    assert "history cleared" not in text
    assistant.chat.assert_awaited_once_with("hi")


@pytest.mark.parametrize("stop", [KeyboardInterrupt(), EOFError()], ids=["ctrl-c", "eof"])
async def test_cli_exits_cleanly_on_ctrl_c_or_end_of_input(main_module, user, stop):
    text, _ = await _run_cli(main_module, [*LOGIN, "Food Bank", "", stop])
    assert "Goodbye" in text


# --- UX -------------------------------------------------------------------------------


async def test_web_ui_shows_the_first_token_before_the_reply_is_finished(app, fake_cl):
    """The second token only arrives once the first is on screen; a buffering UI would hang."""

    async def stream(_):
        yield {"type": "text", "content": "Hel"}
        yield {"type": "text", "content": "lo" if fake_cl.streamed == ["Hel"] else "BUFFERED"}

    fake_cl.session["assistant"] = MagicMock(achat_stream=stream)
    await app.on_message(MagicMock(content="hi"))
    assert fake_cl.streamed == ["Hel", "lo"]


async def test_tool_steps_show_what_was_done_with_the_result_and_time(app, fake_cl):
    """A step is on screen (closed) before the stream goes on to the next model call."""

    async def stream(_):
        yield {
            "type": "tool_step",
            "tool": "record_performance",
            "input_preview": '{"post_id": "post_004", "reactions": 120}',
            "result_preview": "Recorded results for post_004",
            "latency_ms": 7,
        }
        shown = fake_cl.step_objects and fake_cl.step_objects[0].__aexit__.await_count == 1
        yield {"type": "text", "content": "Done" if shown else "BUFFERED"}

    fake_cl.session["assistant"] = MagicMock(achat_stream=stream)
    await app.on_message(MagicMock(content="it got 120 likes"))

    step = fake_cl.step_objects[0]
    assert fake_cl.steps == ["record_performance"]
    assert step.input == '{"post_id": "post_004", "reactions": 120}'
    assert step.output == "Recorded results for post_004  (7ms)"
    assert fake_cl.streamed == ["Done"]


@pytest.mark.parametrize("error", TURN_ERRORS)
async def test_errors_are_shown_in_plain_words_with_a_way_forward(app, fake_cl, error):
    """No tracebacks, class names or provider text in front of the user, and a next step."""

    async def failing(_):
        raise error
        yield  # pragma: no cover — makes this an async generator

    fake_cl.session["assistant"] = MagicMock(achat_stream=failing)
    await app.on_message(MagicMock(content="hi"))
    web = fake_cl.streamed[-1]

    assert "**Something went wrong:**" in web
    _assert_shown_in_plain_words(web, error)
    # Every later turn of a too-long conversation would fail too: the way out comes with it.
    offered = [action for message in fake_cl.messages for action in message["actions"]]
    if isinstance(error, litellm.ContextWindowExceededError):
        [action] = offered
        assert action.name == app.NEW_CONVERSATION_ACTION
        assert fake_cl.messages[-1]["actions"] == [action]  # sent last, below the error
    else:
        assert offered == []


async def test_an_unexpected_failure_at_session_start_is_explained_not_raised(
    app, fake_cl, main_module
):
    """A broken skill file once raised TypeError here, reaching the user as a raw error."""
    from agent.errors import UNEXPECTED_MESSAGE

    boom = TypeError("unhashable type: 'list'")
    fake_cl.session["user"] = SimpleNamespace(metadata={"user_id": 1})
    with patch.object(accounts, "get_user", AsyncMock(side_effect=boom)):
        await app.start()
    with patch.object(accounts, "authenticate", AsyncMock(side_effect=boom)):
        cli, _ = await _run_cli(main_module, LOGIN)
    for shown in (fake_cl.sent[-1], cli):
        assert UNEXPECTED_MESSAGE.split(";")[0] in " ".join(shown.split())
        assert "unhashable" not in shown


@pytest.mark.parametrize(
    "down, expected",
    [
        (
            DatabaseError("MACKIE_DATABASE_URL is not set."),
            "Mackie can't reach its database. MACKIE_DATABASE_URL is not set.",
        ),
        (psycopg_pool.PoolTimeout("pool initialization incomplete"), UNAVAILABLE_MESSAGE),
        (psycopg.OperationalError(DRIVER_DOWN), UNAVAILABLE_MESSAGE),
    ],
    ids=["not-set-up", "pool-timeout", "server-down"],
)
async def test_a_database_that_cant_be_reached_is_explained_in_plain_words(
    app, fake_cl, main_module, down, expected
):
    fake_cl.session["user"] = SimpleNamespace(metadata={"user_id": 1})
    with patch.object(accounts, "get_user", AsyncMock(side_effect=down)):
        await app.start()
    with patch.object(accounts, "authenticate", AsyncMock(side_effect=down)):
        assert await app.login(USERNAME, PASSWORD) is None  # refused, not a crashed login page
        cli, assistant_class = await _run_cli(main_module, LOGIN)

    for shown in (fake_cl.sent[-1], cli):
        assert expected in " ".join(shown.split())  # the terminal may wrap the line
        assert type(down).__name__ not in shown
        assert "10.0.0.5" not in shown and "5432" not in shown
    assert "assistant" not in fake_cl.session
    assistant_class.assert_not_called()


def _fails_once(module_name: str, name: str):
    """Make the next call of module.name fail as if the database restarted; later calls work."""
    module = importlib.import_module(module_name)
    real = getattr(module, name)
    failures = [psycopg.OperationalError(DRIVER_DOWN)]

    async def flaky(*args, **kwargs):
        if failures:
            raise failures.pop()
        return await real(*args, **kwargs)

    return patch.object(module, name, flaky)


PROMPT_READ = ("agent.assistant", "_build_system_prompt_blocks")


@pytest.mark.parametrize(
    "failing", [("agent.accounts", "has_access"), PROMPT_READ], ids=["access-check", "prompt"]
)
async def test_a_database_error_fails_one_web_message_and_the_next_works(
    app, fake_cl, signed_in, failing
):
    """Only a removed membership ends a session; a database blip fails one message."""
    from agent.assistant import Assistant

    assert await accounts.open_org(signed_in.user_id, "food-bank")
    assistant = Assistant("food-bank", thread_id="food-bank:tab", user_id=signed_in.user_id)
    fake_cl.session["assistant"] = assistant
    model = AsyncMock(side_effect=lambda **_: text_stream("Hello again"))
    with _fails_once(*failing), patch("agent.graph.litellm.acompletion", new=model):
        await app.on_message(MagicMock(content="hi"))
        failed = "".join(fake_cl.streamed)
        fake_cl.streamed.clear()
        await app.on_message(MagicMock(content="hi again"))

    assert "Something went wrong" in failed and UNAVAILABLE_MESSAGE in failed
    assert "no longer have access" not in failed and "10.0.0.5" not in failed
    assert fake_cl.session["assistant"] is assistant
    assert "".join(fake_cl.streamed) == "Hello again"


async def test_a_database_error_fails_one_cli_message_and_the_next_works(main_module, user):
    from agent.assistant import Assistant

    assistant = Assistant("food-bank", thread_id=f"food-bank:cli-user-{user.user_id}")
    model = AsyncMock(side_effect=lambda **_: text_stream("Hello again"))
    with _fails_once(*PROMPT_READ), patch("agent.graph.litellm.acompletion", new=model):
        text, _ = await _run_cli(
            main_module, [*LOGIN, "Food Bank", "", "hi", "hi again", "quit"], assistant
        )

    assert f"Error: {UNAVAILABLE_MESSAGE}" in " ".join(text.split())
    assert "10.0.0.5" not in text
    assert "Hello again" in text


def test_welcome_copy_matches_what_mackie_does(app):
    """The first thing a user reads names Mackie and its plan, create, optimize job."""
    welcome = (Path(__file__).parent.parent / "chainlit.md").read_text()
    assert "Mackie" in app.ORG_PROMPT and "social media manager" in app.ORG_PROMPT
    for promise in ("**Plan**", "**Create**", "**Optimize**", "approval"):
        assert promise in welcome


@pytest.mark.parametrize("command", ["hi", "reset", None], ids=["hi", "reset", "at-start"])
async def test_removing_access_ends_an_open_cli_session(main_module, user, command):
    """None: access is gone by the time the start checks for a conversation to resume."""
    from agent.accounts import AccessRevoked

    assert await accounts.open_org(user.user_id, "food-bank")
    revoked = AsyncMock(side_effect=AccessRevoked("You no longer have access to food-bank."))
    assistant = _cli_assistant(chat=revoked, clear_history=revoked)
    if command is None:
        assistant.recent_exchanges = revoked
    # No "quit" answer: the session must end by itself after the refused command.
    answers = [*LOGIN, "Food Bank", "", *([command] if command else [])]
    text, assistant_class = await _run_cli(main_module, answers, assistant)
    assert assistant_class.call_args.args[1].user_id == user.user_id
    assert "You no longer have access to food-bank" in text and "session has ended" in text
    assert "Traceback" not in text
