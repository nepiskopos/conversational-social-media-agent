"""
Unit tests for agent/errors.py: what a user is told when a turn or a session fails.

Owns: every kind of error user_message() maps (Mackie's database not set up, a
database that can't be reached, each kind of AI-provider failure, anything
unexpected), that a too-long conversation names the "Start a new
conversation" button and the terminal's reset command, that no error's own
text reaches the screen (host, port, API key, raw JSON, class names), and that
user_message() leaves logging to its caller.
How the web UI and the terminal show the message is owned by test_ui.py.
"""

import litellm
import psycopg
import psycopg_pool
import pytest

FAKE_KEY = "sk-ant-api03-TESTKEY-not-real-1234567890"
PROVIDER = {"model": "claude-haiku-5-5", "llm_provider": "anthropic"}
RAW_REPLY = '{"type":"error","error":{"type":"authentication_error","message":"invalid x-api-key"}}'


def _provider_errors() -> list[tuple[Exception, str]]:
    """Real LiteLLM exceptions, each quoting a key and raw JSON, with the words the user should see."""
    detail = f"AnthropicException - {RAW_REPLY} key={FAKE_KEY}"
    return [
        (litellm.Timeout(detail, **PROVIDER), "took too long"),
        (litellm.RateLimitError(detail, **PROVIDER), "too many requests"),
        (litellm.AuthenticationError(detail, **PROVIDER), "refused Mackie's API key"),
        (
            litellm.PermissionDeniedError(detail, response=_response(403), **PROVIDER),
            "refused Mackie's API key",
        ),
        (litellm.NotFoundError(detail, **PROVIDER), "doesn't know the configured model"),
        (litellm.ContextWindowExceededError(detail, **PROVIDER), "too long for the AI model"),
        (litellm.BadRequestError(detail, **PROVIDER), "couldn't handle that request"),
        (litellm.APIConnectionError(detail, **PROVIDER), "can't reach the AI service"),
        (litellm.ServiceUnavailableError(detail, **PROVIDER), "having trouble"),
        (litellm.InternalServerError(detail, **PROVIDER), "having trouble"),
    ]


def test_a_too_long_conversation_names_the_button_that_starts_a_new_one():
    """
    The message names the web UI's button by its label constant, so the two can't drift
    apart, and the terminal's 'reset'.
    """
    from agent.errors import NEW_CONVERSATION_LABEL, user_message

    shown = user_message(litellm.ContextWindowExceededError("too long", **PROVIDER))
    assert f"'{NEW_CONVERSATION_LABEL}'" in shown and "'reset'" in shown


def _response(status: int):
    """An HTTP response with this status, which some LiteLLM errors require."""
    import httpx

    return httpx.Response(status, request=httpx.Request("POST", "https://api.example"))


@pytest.mark.parametrize(
    ("error", "expected"),
    _provider_errors(),
    ids=lambda value: type(value).__name__ if isinstance(value, Exception) else "",
)
def test_a_provider_failure_is_explained_without_its_own_text(error, expected):
    from agent.errors import user_message

    shown = user_message(error)
    assert expected in shown
    for leak in (FAKE_KEY, "{", "litellm", "Exception", type(error).__name__):
        assert leak not in shown


@pytest.mark.parametrize(
    "error",
    [
        psycopg.OperationalError('connection to server at "10.0.0.5", port 5432 failed'),
        psycopg_pool.PoolTimeout("pool initialization incomplete after 30.0 sec"),
        psycopg.errors.QueryCanceled("canceling statement due to statement timeout"),
        psycopg.errors.AdminShutdown("terminating connection due to administrator command"),
    ],
    ids=lambda error: type(error).__name__,
)
def test_a_database_failure_hides_the_server_details(error):
    from agent.errors import UNAVAILABLE_MESSAGE, user_message

    shown = user_message(error)
    assert shown == UNAVAILABLE_MESSAGE
    assert "10.0.0.5" not in shown and "5432" not in shown


def test_a_database_that_is_not_set_up_says_what_to_fix():
    from agent.db import DatabaseError
    from agent.errors import user_message

    shown = user_message(DatabaseError("MACKIE_DATABASE_URL is not set."))
    assert shown == "Mackie can't reach its database. MACKIE_DATABASE_URL is not set."


@pytest.mark.parametrize(
    "error",
    [RuntimeError(f"boom with {FAKE_KEY}"), KeyError("content"), TimeoutError("took too long")],
    ids=lambda error: type(error).__name__,
)
def test_anything_unexpected_gets_a_generic_message_with_a_next_step(error):
    from agent.errors import UNEXPECTED_MESSAGE, user_message

    assert user_message(error) == UNEXPECTED_MESSAGE
    assert "check the log" in UNEXPECTED_MESSAGE


def test_user_message_leaves_logging_to_its_caller(caplog):
    """The caller logs once where it handles the error; a second record here would duplicate it."""
    from agent.db import DatabaseError
    from agent.errors import user_message

    caplog.set_level("DEBUG")
    for error in (DatabaseError("not set"), psycopg.OperationalError("down"), RuntimeError("x")):
        user_message(error)
    assert caplog.records == []
