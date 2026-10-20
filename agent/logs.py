"""
Logging for Mackie: where log lines go, at what level, and what may appear in them.

Owns: configure_logging() (where lines go, their format, the level from
MACKIE_LOG_LEVEL, quieter chatty libraries), the RedactSecrets filter that masks
secret values in every line written, safe(), which makes a user-supplied value
safe to put in a log line, and describe() / stack(), which put an error in a
log line without the data its message may quote. Does not own what is logged:
each module logs its own events through logging.getLogger(__name__).

What Mackie logs: logins, refused logins and lockouts, refused or revoked org
access, account changes, and failures (a tool, a model, a page, the database).
What it never logs: passwords, API keys and other secrets (masked even if one
slips into an error message), and what users write: messages, posts and tool
arguments stay in the database (the tool_calls trace), not in log files.
"""

from __future__ import annotations

import logging
import os
import re
import sys
import traceback

LOG_FORMAT: str = "%(asctime)s %(levelname)s %(name)s: %(message)s"
LEVEL_ENV: str = "MACKIE_LOG_LEVEL"
# A user-supplied value in a log line is cut to this many characters before quoting.
SAFE_VALUE_CHARS: int = 80
# Environment variables whose values are secrets, by the end of their name.
SECRET_NAME_PATTERN: re.Pattern[str] = re.compile(r"(KEY|PASSWORD|SECRET|TOKEN)$")
# Shorter values ("", "1", "no") would mask ordinary words.
MIN_SECRET_CHARS: int = 6
# The password in a connection string: postgresql://user:PASSWORD@host
URL_PASSWORD_PATTERN: re.Pattern[str] = re.compile(r"(://[^:/@\s]+:)[^@\s]+@")
MASK: str = "***"
# An error's message in a log line is cut to this many characters.
ERROR_MESSAGE_CHARS: int = 300
# Libraries that log every request at INFO (URLs, model calls): only their warnings are kept.
CHATTY_LOGGERS: tuple[str, ...] = (
    "httpx",
    "httpcore",
    "LiteLLM",
    "LiteLLM Router",
    "LiteLLM Proxy",
)


def safe(value: object, limit: int = SAFE_VALUE_CHARS) -> str:
    """
    A user-supplied value quoted for a log line: cut short, control characters escaped.

    Without it a username like "bob\\n2026-10-20 INFO Login: admin" would forge
    a log line, and a 100 KB value would flood the log.
    """
    return repr(str(value)[:limit])


def describe(error: BaseException) -> str:
    """
    An error for a log line: its type and a short message.

    For a database error, only PostgreSQL's primary message: its detail and
    context lines quote the values that failed, which can be what a user wrote.
    """
    diag = getattr(error, "diag", None)
    primary = getattr(diag, "message_primary", None)
    return f"{type(error).__name__}: {safe(primary or error, ERROR_MESSAGE_CHARS)}"


def stack(error: BaseException) -> str:
    """Where an error happened, frame by frame, without its message (describe() gives that)."""
    return "".join(traceback.format_tb(error.__traceback__)).rstrip()


def redact(text: str) -> str:
    """The text with each environment secret and any connection-string password masked."""
    for name, value in os.environ.items():
        if SECRET_NAME_PATTERN.search(name) and len(value) >= MIN_SECRET_CHARS:
            text = text.replace(value, MASK)
    return URL_PASSWORD_PATTERN.sub(rf"\1{MASK}@", text)


class RedactSecrets(logging.Filter):
    """Mask secrets in a log record's message and traceback before a handler writes it."""

    def filter(self, record: logging.LogRecord) -> bool:
        """
        Rewrite the record with secrets masked; never drops a record, never raises.

        A log call whose arguments don't fit its format would otherwise raise
        here, at the call site, instead of being reported as a logging error.
        """
        try:
            record.msg, record.args = redact(record.getMessage()), None
        except Exception:
            record.msg, record.args = redact(f"{record.msg} {record.args!r}"), None
        if record.exc_info and not record.exc_text:
            record.exc_text = logging.Formatter().formatException(record.exc_info)
        if record.exc_text:
            record.exc_text = redact(record.exc_text)
        return True


def configure_logging(default_level: str) -> None:
    """
    Mask secrets in every log line and log Mackie's at MACKIE_LOG_LEVEL or default_level.

    Under `chainlit run` the root logger already writes to stdout; Mackie's
    lines go there too (one copy each), and its handlers get the RedactSecrets
    filter, so every library's lines are masked as well. Otherwise a stderr
    handler is added. The web UI logs from INFO, so logins and access changes
    are recorded, and so do scripts/users.py and scripts/migrate.py; the
    terminal UI and the other scripts from WARNING, so their screens stay
    readable. Safe to call more than once. An unknown level falls back to
    default_level, and says so.
    """
    root = logging.getLogger()
    if not root.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter(LOG_FORMAT))
        root.addHandler(handler)
    for handler in root.handlers:
        if not any(isinstance(f, RedactSecrets) for f in handler.filters):
            handler.addFilter(RedactSecrets())
    for name in CHATTY_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)
    logger = logging.getLogger("agent")
    requested = os.getenv(LEVEL_ENV, "").strip().upper()
    level = logging.getLevelName(requested) if requested else None
    if not isinstance(level, int):
        logger.setLevel(default_level)
        if requested:
            logger.warning(
                "%s=%s is not a log level; using %s", LEVEL_ENV, safe(requested), default_level
            )
        return
    logger.setLevel(level)
