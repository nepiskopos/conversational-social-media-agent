"""
Tests for agent/logs.py: safe(), describe(), stack(), redact(), RedactSecrets
and configure_logging().

Owns: how a user-supplied value and an error are put in a log line, which
values are masked, that a record is rewritten (never dropped, never raising),
and where configure_logging() sends lines: one copy each, through the root
logger's handlers (Chainlit's when it has set one up), with chatty libraries
quietened. Does not own which events each module logs or that secrets and
user content stay out of the log (test_logging.py).
"""

import logging
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from io import StringIO
from unittest.mock import patch

import psycopg
import pytest

from agent.logs import (
    CHATTY_LOGGERS,
    ERROR_MESSAGE_CHARS,
    LEVEL_ENV,
    LOG_FORMAT,
    MASK,
    SAFE_VALUE_CHARS,
    RedactSecrets,
    configure_logging,
    describe,
    redact,
    safe,
    stack,
)

SECRET = "sk-test-0123456789abcdef"


@contextmanager
def isolated_logging() -> Iterator[None]:
    """
    Restore the logging setup configure_logging() changes when the block ends.

    The root logger's handlers (pytest's among them) and their filters, the
    "agent" level and the chatty libraries' levels all go back as they were,
    so other tests are unaffected.
    """
    root = logging.getLogger()
    handlers = list(root.handlers)
    filters = {handler: list(handler.filters) for handler in handlers}
    names = ("agent", *CHATTY_LOGGERS)
    levels = {name: logging.getLogger(name).level for name in names}
    try:
        yield
    finally:
        root.handlers = handlers
        for handler, kept in filters.items():
            handler.filters = kept
        for name, level in levels.items():
            logging.getLogger(name).setLevel(level)


@contextmanager
def root_log_stream() -> Iterator[StringIO]:
    """
    A StringIO the root logger writes to, as `logging.basicConfig(stream=...)` (Chainlit) sets up.

    It is added beside pytest's handlers, so caplog still records; configure_logging()
    inside the block adds RedactSecrets to every root handler.
    """
    stream = StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(logging.Formatter(LOG_FORMAT))
    with isolated_logging():
        logging.getLogger().addHandler(handler)
        yield stream


@contextmanager
def no_root_handler() -> Iterator[StringIO]:
    """A root logger with no handlers (the CLI, a script), and sys.stderr as a StringIO."""
    stream = StringIO()
    with isolated_logging(), patch.object(sys, "stderr", stream):
        logging.getLogger().handlers = []
        yield stream


def _record(msg: str, args: tuple = (), exc_info=None) -> logging.LogRecord:
    """An ERROR record from agent.test, as a logging call would make it."""
    return logging.LogRecord("agent.test", logging.ERROR, __file__, 1, msg, args, exc_info)


def _raise(error: Exception) -> Exception:
    """The error, raised and caught, so it has a traceback."""
    try:
        raise error
    except Exception as e:
        return e


# --- safe -------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "value, expected",
    [
        ("bob", "'bob'"),
        ("bob\n2026-10-20 INFO Login: admin", "'bob\\n2026-10-20 INFO Login: admin'"),
        ("a\rb\x1b[31m", "'a\\rb\\x1b[31m'"),
        (42, "'42'"),
        (None, "'None'"),
    ],
)
def test_safe_quotes_and_escapes_control_characters(value, expected):
    assert safe(value) == expected


@pytest.mark.parametrize("limit", [SAFE_VALUE_CHARS, 5])
def test_safe_cuts_a_long_value(limit):
    assert safe("x" * 100_000, limit) == repr("x" * limit)


# --- describe and stack -----------------------------------------------------------------


def test_describe_gives_the_type_and_a_short_escaped_message():
    error = RuntimeError("first line\nsecond line " + "x" * 10_000)
    described = describe(error)
    assert described == f"RuntimeError: {str(error)[:ERROR_MESSAGE_CHARS]!r}"
    assert described.startswith("RuntimeError: 'first line\\nsecond line x")
    assert "\n" not in described


def test_describe_gives_only_postgresqls_primary_message(database):
    """The DETAIL and CONTEXT lines quote the values that failed, which a user may have written."""
    with pytest.raises(psycopg.Error) as caught:
        database["cleaner"].execute(
            "DO $$ BEGIN RAISE EXCEPTION 'the post was refused' "
            "USING DETAIL = 'Failing row contains (DETAIL-MARKER).'; END $$"
        )
    error = caught.value
    assert error.diag.message_detail and error.diag.context  # the lines describe() leaves out
    assert describe(error) == "RaiseException: 'the post was refused'"


def test_describe_falls_back_to_the_message_of_a_driver_error_without_diagnostics():
    error = psycopg.OperationalError("connection to server at db, port 5432 failed")
    assert describe(error) == "OperationalError: 'connection to server at db, port 5432 failed'"


def test_stack_lists_the_frames_without_the_message():
    error = _raise(RuntimeError("STACK-MARKER"))
    frames = stack(error)
    assert 'File "' in frames and "in _raise" in frames
    assert "STACK-MARKER" not in frames and not frames.endswith("\n")


def test_stack_of_an_error_never_raised_is_empty():
    assert stack(RuntimeError("not raised")) == ""


# --- redact -----------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name", ["ANTHROPIC_API_KEY", "MACKIE_APP_PASSWORD", "SESSION_SECRET", "GITHUB_TOKEN"]
)
def test_redact_masks_the_value_of_a_secret_variable(monkeypatch, name):
    monkeypatch.setenv(name, SECRET)
    assert redact(f"auth failed for key {SECRET}.") == f"auth failed for key {MASK}."


@pytest.mark.parametrize(
    "name, value",
    [
        ("SHORT_TOKEN", "no"),  # too short: would mask ordinary words
        ("MACKIE_LOG_LEVEL", "WARNINGS"),  # not a secret's name
        ("KEYBOARD_LAYOUT", "colemak"),  # KEY at the start, not the end
    ],
)
def test_redact_leaves_short_values_and_other_variables(monkeypatch, name, value):
    monkeypatch.setenv(name, value)
    assert redact(f"value {value} here") == f"value {value} here"


def test_redact_masks_a_connection_string_password():
    text = "connection to postgresql://mackie_app:s3cr3t-pw@db:5432/mackie failed"
    masked = redact(text)
    assert masked == f"connection to postgresql://mackie_app:{MASK}@db:5432/mackie failed"


def test_redact_leaves_a_connection_string_without_a_password():
    assert redact("postgresql://mackie@db/mackie") == "postgresql://mackie@db/mackie"


# --- RedactSecrets ----------------------------------------------------------------------


def test_redact_secrets_masks_the_message_built_from_args(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", SECRET)
    record = _record("Model %s failed: bad key %s", ("claude", SECRET))
    assert RedactSecrets().filter(record) is True
    assert record.getMessage() == f"Model claude failed: bad key {MASK}"
    assert record.args is None


def test_redact_secrets_masks_the_traceback(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", SECRET)
    error = _raise(RuntimeError(f"401 for {SECRET} at postgresql://u:pw-123456@db/x"))
    record = _record("A turn failed", exc_info=(type(error), error, error.__traceback__))
    assert RedactSecrets().filter(record) is True
    assert "RuntimeError" in record.exc_text
    assert SECRET not in record.exc_text and "pw-123456" not in record.exc_text
    assert f"401 for {MASK} at postgresql://u:{MASK}@db/x" in record.exc_text


@pytest.mark.parametrize(
    "msg, args", [("%d posts for %s", ("many", SECRET)), ("%s and %s", (SECRET,))]
)
def test_redact_secrets_keeps_a_malformed_log_call_masked_without_raising(monkeypatch, msg, args):
    monkeypatch.setenv("ANTHROPIC_API_KEY", SECRET)
    record = _record(msg, args)
    assert RedactSecrets().filter(record) is True
    assert record.getMessage().startswith(msg)
    assert MASK in record.getMessage() and SECRET not in record.getMessage()


# --- configure_logging ------------------------------------------------------------------


def test_configure_logging_adds_a_stderr_handler_once_when_root_has_none(monkeypatch):
    monkeypatch.delenv(LEVEL_ENV, raising=False)
    with no_root_handler() as stream:
        configure_logging("INFO")
        configure_logging("WARNING")
        [handler] = logging.getLogger().handlers
        assert handler.stream is stream
        assert sum(isinstance(f, RedactSecrets) for f in handler.filters) == 1
        assert logging.getLogger("agent").handlers == []
        assert logging.getLogger("agent").level == logging.WARNING


def test_under_an_existing_root_handler_lines_appear_once_masked_and_libraries_quieter(
    monkeypatch,
):
    """As under `chainlit run`, whose basicConfig() set up a root handler before Mackie's import."""
    monkeypatch.delenv(LEVEL_ENV, raising=False)
    monkeypatch.setenv("TAVILY_API_KEY", SECRET)
    # LiteLLM's import puts its own redaction on httpx; take it off to see Mackie's at work.
    monkeypatch.setattr(logging.getLogger("httpx"), "filters", [])
    with root_log_stream() as stream:
        handlers = list(logging.getLogger().handlers)
        configure_logging("INFO")
        configure_logging("INFO")
        assert logging.getLogger().handlers == handlers  # nothing added beside Chainlit's
        logging.getLogger("agent.accounts").warning("Login refused for 'sam': wrong password")
        logging.getLogger("httpx").info("HTTP Request: POST https://api.tavily.com/search")
        logging.getLogger("httpx").warning("Retrying with key %s", SECRET)
    lines = stream.getvalue().splitlines()
    assert len(lines) == 2
    assert sum("Login refused for 'sam'" in line for line in lines) == 1
    assert lines[1].endswith(f" WARNING httpx: Retrying with key {MASK}")
    assert SECRET not in stream.getvalue()


@pytest.mark.parametrize(
    "requested, expected",
    [("DEBUG", logging.DEBUG), ("error", logging.ERROR), (" Info ", logging.INFO), ("", None)],
)
def test_configure_logging_takes_the_level_from_the_environment(monkeypatch, requested, expected):
    monkeypatch.setenv(LEVEL_ENV, requested)
    with no_root_handler() as stream:
        configure_logging("WARNING")
        assert logging.getLogger("agent").level == (expected or logging.WARNING)
    assert stream.getvalue() == ""


def test_configure_logging_falls_back_on_an_unknown_level_and_says_so_safely(monkeypatch):
    monkeypatch.setenv(LEVEL_ENV, "loud\n2026-10-20 INFO agent.accounts: Login: admin")
    with no_root_handler() as stream:
        configure_logging("INFO")
        assert logging.getLogger("agent").level == logging.INFO
    [line] = stream.getvalue().splitlines()
    assert " WARNING agent: " in line
    assert "'LOUD\\n2026-10-20 INFO AGENT.ACCOUNTS: LOGIN: ADMIN' is not a log level" in line
    assert line.endswith("using INFO")


def test_configured_output_is_formatted_redacted_and_filtered_by_level(monkeypatch):
    """A malformed log call (wrong argument types) is written masked, not raised."""
    monkeypatch.delenv(LEVEL_ENV, raising=False)
    monkeypatch.setenv("TAVILY_API_KEY", SECRET)
    with no_root_handler() as stream:
        configure_logging("INFO")
        logging.getLogger("agent.research").info("searching with %s", SECRET)
        logging.getLogger("agent.research").debug("not shown at INFO")
        logging.getLogger("agent.research").warning("%d results for %s", "some", SECRET)
    searching, malformed = stream.getvalue().splitlines()
    assert searching.endswith(f" INFO agent.research: searching with {MASK}")
    assert "WARNING agent.research: %d results for %s" in malformed and MASK in malformed
    assert "Logging error" not in stream.getvalue() and SECRET not in stream.getvalue()
