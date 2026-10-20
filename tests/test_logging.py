"""
Cross-cutting tests of what Mackie logs and what it never logs, across modules.

Owns: that each event the logging policy names (logins, refused logins and
lockouts, refused or revoked org access, account changes, a blocked fetch,
and failures of a tool, a model, a page fetch, a search, the summary, the org
brief, a calendar week, the trace, the database, a login, a session start,
the recap at a session's start, a turn on the web or in the terminal, a
reset, a terminal session, pruning, and a skill file skipped) is logged once,
by the module that handles it, at the right level, naming the user, org or
tool involved; and that passwords, API keys,
connection-string passwords and what users write (messages, post content,
tool arguments, a username typed in the wrong field) never reach a log line,
nor can a value the model supplies forge one. Does not own the logging
helpers themselves (test_logs.py), what the user sees for an error
(test_errors.py) or the behavior around each failure (each module's own
tests).
"""

import json
import logging
import sys
from io import StringIO
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import psycopg
import pytest
from langchain_core.messages import AIMessage, HumanMessage
from rich.console import Console

from agent import accounts
from agent.logs import LEVEL_ENV, MASK, configure_logging
from tests.llm_fakes import text_stream, tool_call_stream
from tests.test_logs import root_log_stream
from tests.test_ui import FakeChainlit, _import_entry_point

PASSWORD = "correct horse battery"
# A failure's log line ends with where it happened (logs.stack()).
FRAME = 'File "'


@pytest.fixture
def logs(caplog):
    """caplog recording the "agent" loggers from INFO."""
    caplog.set_level(logging.INFO, logger="agent")
    return caplog


@pytest.fixture
async def sam(database):
    """The account "sam" (PASSWORD), hashed at a low cost to keep the tests fast."""
    yield await accounts.create_user("sam", PASSWORD)


@pytest.fixture(scope="module")
def app():
    """app.py, imported once: test_ui.py imports it too, and importing chainlit costs ~0.6 s."""
    return sys.modules.get("app") or _import_entry_point("app")


@pytest.fixture(scope="module")
def main_module():
    """main.py, imported once and shared with test_ui.py."""
    return sys.modules.get("main") or _import_entry_point("main")


def _records(caplog, logger: str) -> list[tuple[int, str]]:
    """(level, message) of each record from one logger."""
    return [(r.levelno, r.getMessage()) for r in caplog.records if r.name == logger]


def _errors(caplog) -> list[logging.LogRecord]:
    """Every record at ERROR or above, from any logger."""
    return [r for r in caplog.records if r.levelno >= logging.ERROR]


async def _web_turn(app, assistant, text: str) -> list[str]:
    """Send one message through app.on_message; return what was streamed to the user."""
    fake = FakeChainlit()
    session = MagicMock(get=MagicMock(return_value=assistant))
    with (
        patch.object(app.cl, "Message", side_effect=fake.message),
        patch.object(app.cl, "Step", side_effect=fake.step),
        patch.object(app.cl, "user_session", session),
    ):
        await app.on_message(MagicMock(content=text))
    return fake.streamed


async def _cli_session(
    main_module, answers: list[str], assistant, authenticate: AsyncMock | None = None
) -> None:
    """Run main.py's session with scripted answers, signed in as user 1 and opening food-bank."""
    user = accounts.User(1, "sam")
    with (
        patch.object(main_module, "console", Console(file=StringIO())),
        patch.object(main_module.Prompt, "ask", side_effect=answers),
        patch.object(accounts, "authenticate", new=authenticate or AsyncMock(return_value=user)),
        patch.object(accounts, "user_orgs", new=AsyncMock(return_value=[])),
        patch.object(accounts, "open_org", new=AsyncMock(return_value=True)),
        patch.object(main_module.Assistant, "for_user", return_value=assistant),
    ):
        await main_module.main()


async def _cli_turn(main_module, assistant, text: str) -> None:
    """Send one message through main.py's conversation loop, then quit."""
    await _cli_session(main_module, ["sam", PASSWORD, "Food Bank", "", text, "quit"], assistant)


def _failing_assistant(error: Exception) -> MagicMock:
    """An assistant whose turn fails with error, on the web (achat_stream) and in the CLI (chat)."""

    async def failing_stream(_message):
        raise error
        yield  # an async generator, like achat_stream

    return MagicMock(
        org_id="food-bank",
        achat_stream=failing_stream,
        chat=AsyncMock(side_effect=error),
        clear_history=AsyncMock(side_effect=error),
        recent_exchanges=AsyncMock(return_value=[]),
    )


# --- Accounts ---------------------------------------------------------------------------


async def test_a_login_is_logged(sam, logs):
    logs.clear()
    assert await accounts.authenticate("Sam", PASSWORD)
    assert _records(logs, "agent.accounts") == [(logging.INFO, "Login: 'sam'")]


LOCK_SAM = "UPDATE users SET locked_until = now() + interval '1 hour' WHERE username = 'sam'"


@pytest.mark.parametrize(
    "username, password, lock, reason",
    [
        ("nobody", PASSWORD, None, "'nobody': no such user"),
        ("sam", "wrong password!", None, "'sam': wrong password"),
        ("sam", PASSWORD, "before", "'sam': the account is locked"),
        ("sam", PASSWORD, "meanwhile", "'sam': the account was locked meanwhile"),
    ],
)
async def test_a_refused_login_is_logged_with_its_reason(
    sam, logs, database, username, password, lock, reason
):
    real_verify = accounts._verify

    async def guesses_lock_it_during_the_check(password: str, stored: str | None) -> bool:
        right = await real_verify(password, stored)
        database["cleaner"].execute(LOCK_SAM)
        return right

    if lock == "before":
        database["cleaner"].execute(LOCK_SAM)
    verify = guesses_lock_it_during_the_check if lock == "meanwhile" else real_verify
    logs.clear()
    with patch.object(accounts, "_verify", verify):
        assert await accounts.authenticate(username, password) is None
    assert _records(logs, "agent.accounts") == [(logging.WARNING, f"Login refused for {reason}")]


async def test_an_invalid_username_is_refused_without_logging_what_was_typed(database, logs):
    """What was typed may be a password typed into the username field."""
    typed = "Hunter2 Is My Password!"
    assert await accounts.authenticate(typed, PASSWORD) is None
    assert _records(logs, "agent.accounts") == [
        (logging.WARNING, "Login refused: the username typed is not a valid username")
    ]
    assert "Hunter2" not in logs.text and "hunter2" not in logs.text


async def test_a_lockout_is_logged_once_when_the_account_locks(sam, logs):
    logs.clear()
    for _ in range(accounts.MAX_FAILED_LOGINS):
        assert await accounts.authenticate("sam", "wrong password!") is None
    records = _records(logs, "agent.accounts")
    locks = [message for level, message in records if "locked for" in message]
    assert locks == [
        f"Account 'sam' locked for {accounts.LOCKOUT} after {accounts.MAX_FAILED_LOGINS} "
        "wrong passwords"
    ]
    assert {level for level, _ in records} == {logging.WARNING}


async def test_account_changes_are_logged(database, logs):
    await accounts.create_user("sam", PASSWORD)
    await accounts.ensure_user("alex", PASSWORD)
    await accounts.ensure_user("alex", PASSWORD)  # exists: nothing changes, nothing logged
    await accounts.set_password("sam", "another long password")
    await accounts.grant("sam", "food-bank")
    await accounts.revoke("sam", "food-bank")
    await accounts.revoke("sam", "food-bank")  # not a member any more: nothing logged
    await accounts.delete_user("sam")
    assert _records(logs, "agent.accounts") == [
        (logging.INFO, "Account created: 'sam'"),
        (logging.INFO, "Account created: 'alex'"),
        (logging.INFO, "Password changed (and any lockout lifted): 'sam'"),
        (logging.INFO, "Access granted: 'sam' to food-bank"),
        (logging.INFO, "Access revoked: 'sam' from food-bank"),
        (logging.INFO, "Account deleted: 'sam'"),
    ]


async def test_opening_an_org_is_logged_when_it_is_created_or_refused(sam, logs):
    alex = await accounts.create_user("alex", PASSWORD)
    logs.clear()
    assert await accounts.open_org(sam.user_id, "food-bank")
    assert await accounts.open_org(sam.user_id, "food-bank")  # a member: nothing logged
    assert not await accounts.open_org(alex.user_id, "food-bank")
    assert _records(logs, "agent.accounts") == [
        (logging.INFO, f"User {sam.user_id} created org food-bank"),
        (logging.WARNING, f"User {alex.user_id} refused org food-bank: not a member"),
    ]


async def test_a_message_after_revoked_access_is_logged(sam, logs):
    from agent.assistant import Assistant

    await accounts.grant("sam", "food-bank")
    assistant = Assistant("food-bank", thread_id="food-bank:tab", user_id=sam.user_id)
    await accounts.revoke("sam", "food-bank")
    with pytest.raises(accounts.AccessRevoked):
        await assistant.chat("hello")
    assert _records(logs, "agent.assistant") == [
        (
            logging.WARNING,
            f"Refused a message: user {sam.user_id} no longer has access to food-bank",
        )
    ]


# --- Tools ------------------------------------------------------------------------------


async def test_a_failing_tool_is_logged_once_with_where_but_not_its_arguments(database, logs):
    from agent.tools import handle_tool_call

    failing = AsyncMock(side_effect=RuntimeError("ledger unavailable"))
    with patch("agent.tools.mem.add_post", new=failing):
        result = await handle_tool_call(
            "save_post",
            {"platform": "general", "title": "TITLE-MARKER", "content": "CONTENT-MARKER"},
            "food-bank",
        )
    assert result.startswith("Error: save_post failed")
    [record] = _errors(logs)
    message = record.getMessage()
    assert record.name == "agent.tools"
    assert message.startswith(
        "Tool 'save_post' failed for food-bank: RuntimeError: 'ledger unavailable'\n"
    )
    assert FRAME in message and "in handle_tool_call" in message
    assert "MARKER" not in logs.text


async def test_a_database_error_in_a_tool_logs_only_its_primary_message(database, logs):
    """PostgreSQL's DETAIL line quotes the failing row, which holds what the user wrote."""
    from agent.tools import handle_tool_call

    def refuse(*_):
        database["cleaner"].execute(
            "DO $$ BEGIN RAISE EXCEPTION 'value too long' "
            "USING DETAIL = 'Failing row contains (CONTENT-MARKER).'; END $$"
        )

    with patch("agent.tools.mem.add_post", new=AsyncMock(side_effect=refuse)):
        await handle_tool_call(
            "save_post",
            {"platform": "general", "title": "T", "content": "CONTENT-MARKER"},
            "food-bank",
        )
    [record] = _errors(logs)
    assert "RaiseException: 'value too long'" in record.getMessage()
    assert "MARKER" not in logs.text and "DETAIL" not in logs.text


async def test_a_failed_trace_write_is_logged(database, logs):
    from agent.tools import handle_tool_call

    failing = AsyncMock(side_effect=psycopg.OperationalError("server closed the connection"))
    with patch("agent.tools.mem.append_trace", new=failing):
        await handle_tool_call(
            "save_post",
            {"platform": "general", "title": "T", "content": "CONTENT-MARKER"},
            "food-bank",
        )
    assert _records(logs, "agent.tools") == [
        (
            logging.WARNING,
            "Could not trace 'save_post' for food-bank: "
            "OperationalError: 'server closed the connection'",
        )
    ]
    assert "MARKER" not in logs.text


async def test_a_failed_org_brief_is_logged(logs):
    from agent.tools import _write_org_brief

    with patch("agent.tools.litellm.acompletion", new=AsyncMock(side_effect=TimeoutError("slow"))):
        assert await _write_org_brief("Food Bank", {}) == ""
    assert _records(logs, "agent.tools") == [
        (logging.WARNING, "Could not write the org brief for 'Food Bank': TimeoutError: 'slow'")
    ]


async def test_a_failed_calendar_week_is_logged(database, logs):
    from agent.memory import save_profile
    from agent.tools import handle_tool_call

    await save_profile(
        "food-bank",
        {"org_id": "food-bank", "name": "Food Bank", "website": "", "research": None},
    )
    week = json.dumps({"platform": "linkedin", "angle": "Story", "pillar": "Impact"})
    ok = MagicMock(choices=[MagicMock(message=MagicMock(content=week))])
    with patch(
        "agent.tools.litellm.acompletion",
        new=AsyncMock(side_effect=[ok, TimeoutError("timed out"), ok, ok]),
    ):
        await handle_tool_call("generate_calendar", {"month_year": "August 2026"}, "food-bank")
    assert _records(logs, "agent.tools") == [
        (
            logging.WARNING,
            "Could not plan week 2 of August 2026 for food-bank: TimeoutError: 'timed out'",
        )
    ]


# --- Graph ------------------------------------------------------------------------------


async def test_a_model_fallback_is_logged_and_the_last_failure_left_to_the_caller(logs):
    """The turn's failure is logged once, by app.py or main.py, not here as well."""
    from agent.graph import _open_stream

    models = ["anthropic/claude-sonnet-4-5", "gpt-4o-mini"]
    with patch(
        "agent.graph.litellm.acompletion", new=AsyncMock(side_effect=RuntimeError("overloaded"))
    ):
        with pytest.raises(RuntimeError):
            await _open_stream(models, "stable", "per turn", [])
    assert _records(logs, "agent.graph") == [
        (
            logging.WARNING,
            "Model anthropic/claude-sonnet-4-5 failed, falling back to gpt-4o-mini: "
            "RuntimeError: 'overloaded'",
        )
    ]


async def test_a_failed_summary_is_logged(logs):
    from agent.graph import _build_summarize

    history = [m for n in range(30) for m in (HumanMessage(f"q{n}"), AIMessage(f"a{n}"))]
    summarize = _build_summarize("gpt-4o-mini")
    with patch("agent.graph.litellm.acompletion", new=AsyncMock(side_effect=RuntimeError("429"))):
        assert await summarize({"messages": history, "org_id": "food-bank"}) == {}
    assert _records(logs, "agent.graph") == [
        (logging.WARNING, "Could not summarize older messages for food-bank: RuntimeError: '429'")
    ]


async def test_a_failed_prune_is_logged(logs):
    from agent.graph import prune_history

    timeout = psycopg.errors.QueryCanceled("statement timeout")
    graph = SimpleNamespace(checkpointer=SimpleNamespace(aprune=AsyncMock(side_effect=timeout)))
    await prune_history(graph, "food-bank:tab")
    assert _records(logs, "agent.graph") == [
        (
            logging.WARNING,
            "Pruning food-bank:tab failed, the next turn retries: QueryCanceled: 'statement timeout'",
        )
    ]


# --- Research and skills ----------------------------------------------------------------


async def test_a_blocked_fetch_is_logged(logs):
    from agent.research import scrape_site

    await scrape_site("http://169.254.169.254/latest/meta-data")
    [(level, message)] = _records(logs, "agent.research")
    assert level == logging.WARNING
    assert message.startswith(
        "Refused to fetch 'http://169.254.169.254/latest/meta-data': BlockedURLError: "
    )


async def test_a_failed_fetch_is_logged(logs):
    from agent.research import scrape_site

    failing = AsyncMock(side_effect=ConnectionError("connection refused"))
    with patch("agent.research._fetch_html", new=failing):
        await scrape_site("food.example")
    assert _records(logs, "agent.research") == [
        (
            logging.WARNING,
            "Could not fetch 'https://food.example': ConnectionError: 'connection refused'",
        )
    ]


async def test_a_failed_search_is_logged(logs):
    from agent.research import search_web

    with patch("agent.research.AsyncTavilyClient") as client_class:
        client_class.return_value.search = AsyncMock(side_effect=RuntimeError("quota exceeded"))
        await search_web("food bank volunteers")
    assert _records(logs, "agent.research") == [
        (logging.WARNING, "Web search failed: RuntimeError: 'quota exceeded'")
    ]


def test_a_skill_file_problem_is_logged_once_however_often_skills_load(tmp_path, logs, monkeypatch):
    """Skills load on every model call; the log must not repeat the problem each time."""
    from agent import skills

    (tmp_path / "broken.yaml").write_text("id: broken\nkind: platform\n", encoding="utf-8")
    monkeypatch.setattr(skills, "SKILLS_DIR", tmp_path)
    monkeypatch.setattr(skills, "_logged_problems", set())
    for _ in range(3):
        assert skills.load_skills().problems
    [(level, message)] = [
        record for record in _records(logs, "agent.skills") if "broken.yaml" in record[1]
    ]
    assert level == logging.WARNING and message.startswith("Skipped skill file — broken.yaml")


# --- Database and UI: each failure logged once, where it is handled ---------------------


@pytest.mark.parametrize("ui, logger", [("web", "agent.app"), ("cli", "agent.main")])
@pytest.mark.parametrize("failure", ["model", "database"])
async def test_a_failed_turn_is_logged_once_with_where(
    app, main_module, database, logs, ui, logger, failure
):
    from agent.assistant import Assistant

    def turn(assistant):
        if ui == "web":
            return _web_turn(app, assistant, "MESSAGE-MARKER")
        return _cli_turn(main_module, assistant, "MESSAGE-MARKER")

    if failure == "model":
        assistant = Assistant("food-bank", thread_id="food-bank:tab")
        failing = AsyncMock(side_effect=RuntimeError("model unavailable"))
        with patch("agent.graph.litellm.acompletion", new=failing):
            await turn(assistant)
        described = "RuntimeError: 'model unavailable'"
    else:
        error = psycopg.OperationalError("connection to db:5432 refused")
        await turn(_failing_assistant(error))
        described = "OperationalError: 'connection to db:5432 refused'"
    [record] = _errors(logs)
    assert record.name == logger
    assert record.getMessage().startswith(f"A turn failed for food-bank: {described}\n")
    assert FRAME in record.getMessage()
    assert "MARKER" not in logs.text


async def test_a_database_failure_at_login_or_session_start_is_logged_once(app, logs):
    error = psycopg.OperationalError("connection to db:5432 refused")
    with patch.object(app.accounts, "authenticate", new=AsyncMock(side_effect=error)):
        assert await app.login("sam", PASSWORD) is None
    with (
        patch.object(app, "_start", new=AsyncMock(side_effect=error)),
        patch.object(app.cl, "Message", side_effect=FakeChainlit().message),
    ):
        await app.start()
    described = "OperationalError: 'connection to db:5432 refused'"
    assert [(r.name, r.getMessage().split("\n")[0]) for r in _errors(logs)] == [
        ("agent.app", f"Login failed: the database can't be reached: {described}"),
        ("agent.app", f"A session couldn't start: {described}"),
    ]
    assert PASSWORD not in logs.text


async def test_a_failed_reset_or_cli_session_is_logged_once(app, main_module, database, logs):
    error = psycopg.OperationalError("connection to db:5432 refused")
    session = MagicMock(get=MagicMock(return_value=_failing_assistant(error)))
    with (
        patch.object(app.cl, "Message", side_effect=FakeChainlit().message),
        patch.object(app.cl, "user_session", session),
    ):
        await app.start_new_conversation(MagicMock(remove=AsyncMock()))
    await _cli_session(
        main_module, ["sam", PASSWORD, "Food Bank", "", "reset", "quit"], _failing_assistant(error)
    )
    await _cli_session(main_module, ["sam", PASSWORD], None, AsyncMock(side_effect=error))
    described = "OperationalError: 'connection to db:5432 refused'"
    assert [(r.name, r.getMessage().split("\n")[0]) for r in _errors(logs)] == [
        ("agent.app", f"Starting a new conversation failed for food-bank: {described}"),
        ("agent.main", f"Reset failed for food-bank: {described}"),
        ("agent.main", f"The terminal session stopped: {described}"),
    ]
    assert PASSWORD not in logs.text


async def test_a_recap_that_cant_load_is_logged_once(app, main_module, sam, logs):
    """Web start and CLI start both carry on without the recap, and say why in the log."""
    error = psycopg.OperationalError("connection to db:5432 refused")
    recent = AsyncMock(side_effect=error)
    fake = FakeChainlit()
    session = {
        "user": SimpleNamespace(metadata={"user_id": sam.user_id, "session_key": sam.session_key})
    }
    with (
        patch.object(app.cl, "Message", side_effect=fake.message),
        patch.object(app.cl, "user_session", MagicMock(get=session.get, set=session.__setitem__)),
        patch.object(app, "_ask_for_org", AsyncMock(return_value=("Food Bank", ""))),
        patch.object(app.Assistant, "recent_exchanges", recent),
    ):
        await app.start()
    assistant = _failing_assistant(error)
    assistant.recent_exchanges = recent
    await _cli_session(main_module, ["sam", PASSWORD, "Food Bank", "", "quit"], assistant)
    described = "OperationalError: 'connection to db:5432 refused'"
    assert [(r.name, r.getMessage()) for r in _errors(logs)] == [
        ("agent.app", f"Couldn't load the recap for food-bank: {described}"),
        ("agent.main", f"Couldn't check for a conversation to resume for food-bank: {described}"),
    ]


# --- Never logged -----------------------------------------------------------------------


async def test_secrets_and_what_users_write_are_never_logged(app, sam, logs, monkeypatch):
    """
    A whole web session's log, as the configured root handler writes it and as caplog records it.

    configure_logging() puts RedactSecrets on every root handler (Chainlit's
    under `chainlit run`, a StringIO here, and pytest's), so each sees the
    record masked.
    """
    from agent.assistant import Assistant
    from agent.tools import handle_tool_call

    api_key = "sk-ant-fake-0123456789abcdef"
    db_password = "pg-Pass-0123456789"
    monkeypatch.setenv("ANTHROPIC_API_KEY", api_key)
    monkeypatch.delenv(LEVEL_ENV, raising=False)
    post = {"platform": "linkedin", "title": "TITLE-MARKER", "content": "POST-MARKER"}
    replies = iter(
        [
            tool_call_stream("call_1", "save_post", json.dumps(post)),
            text_stream("Saved it."),
        ]
    )
    assistant = Assistant("food-bank", thread_id="food-bank:tab")
    with root_log_stream() as stream:
        configure_logging("INFO")
        assert await app.login("sam", PASSWORD)
        assert await app.login("sam", PASSWORD + "-typo") is None
        model = AsyncMock(side_effect=lambda **_: next(replies))
        with patch("agent.graph.litellm.acompletion", new=model):
            assert await _web_turn(app, assistant, "MESSAGE-MARKER: our pantry") == ["Saved it."]
        with patch("agent.tools.mem.add_post", new=AsyncMock(side_effect=RuntimeError("down"))):
            await handle_tool_call("save_post", {**post, "content": "ARG-MARKER"}, "food-bank")
        auth_error = RuntimeError(f"401 Unauthorized: invalid x-api-key {api_key}")
        with patch("agent.graph.litellm.acompletion", new=AsyncMock(side_effect=auth_error)):
            await _web_turn(app, assistant, "MESSAGE-MARKER again")
        db_error = psycopg.OperationalError(
            f'connection to "postgresql://mackie_app:{db_password}@db:5432/mackie" failed'
        )
        await _web_turn(app, _failing_assistant(db_error), "MESSAGE-MARKER once more")
    written = stream.getvalue()
    for text in (logs.text, written):
        assert "Login: 'sam'" in text and "wrong password" in text
        assert "RuntimeError: 'down'" in text and text.count("A turn failed for food-bank") == 2
        for never in (PASSWORD, "MARKER", api_key, db_password):
            assert never not in text, f"{never!r} was logged"
    assert f"mackie_app:{MASK}@db" in written and f"x-api-key {MASK}" in written


FORGED = "\n2026-10-20 INFO agent.accounts: Login: admin"


async def _brief_for_a_forged_org_name() -> None:
    """Log a failed org brief for an org name carrying a forged log line."""
    from agent.tools import _write_org_brief

    with patch("agent.tools.litellm.acompletion", new=AsyncMock(side_effect=TimeoutError("slow"))):
        await _write_org_brief(f"Food{FORGED}", {})


async def _fetch_of_a_forged_url() -> None:
    """Log a failed fetch of a URL carrying a forged log line."""
    from agent.research import scrape_site

    with patch("agent.research._fetch_html", new=AsyncMock(side_effect=ConnectionError("down"))):
        await scrape_site(f"http://x.example/{FORGED}")


@pytest.mark.parametrize("log_it", [_brief_for_a_forged_org_name, _fetch_of_a_forged_url])
async def test_a_model_supplied_value_cannot_forge_a_log_line(logs, log_it):
    """The org name and the URL come from the model; a line break in either stays escaped."""
    with root_log_stream() as stream:
        configure_logging("INFO")
        await log_it()
    [line] = stream.getvalue().splitlines()
    assert " WARNING agent." in line
    assert "\\n2026-10-20 INFO agent.accounts: Login: admin" in line
