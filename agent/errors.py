"""
What a user is told when something fails, in plain words with a next step.

Owns: user_message(), which maps the errors a turn can end with (the database,
the AI provider, anything unexpected) to a sentence for the screen. An error's
own text never reaches the screen: a driver error names the host and port, a
provider error can quote the API key, the raw JSON of the reply or a class
name. The full error goes to the log, which the caller writes (agent/logs.py).
Does not own logging, or errors a tool turns into a result for the model
(tools.py).
"""

from __future__ import annotations

import litellm
import psycopg

from .db import DatabaseError

# The web UI's button for starting afresh; the context-window message tells users to use it.
NEW_CONVERSATION_LABEL: str = "Start a new conversation"
UNAVAILABLE_MESSAGE: str = (
    "Mackie can't reach its database right now. Please try again in a moment."
)
UNEXPECTED_MESSAGE: str = (
    "Something unexpected went wrong. Please try again; if it keeps happening, "
    "ask whoever runs Mackie to check the log."
)
# The most specific class first: a context-window error is also a BadRequestError.
_PROVIDER_MESSAGES: tuple[tuple[type[Exception], str], ...] = (
    (
        litellm.ContextWindowExceededError,
        f"This conversation is now too long for the AI model. Start a new one with "
        f"'{NEW_CONVERSATION_LABEL}', or type 'reset' in the terminal. "
        "What Mackie learned is kept.",
    ),
    (
        litellm.Timeout,
        "The AI service took too long to answer. Please try again in a moment.",
    ),
    (
        litellm.RateLimitError,
        "The AI service is getting too many requests. Please wait a minute and try again.",
    ),
    (
        litellm.AuthenticationError,
        "The AI service refused Mackie's API key. Ask whoever runs Mackie to check the key in .env.",
    ),
    (
        litellm.PermissionDeniedError,
        "The AI service refused Mackie's API key. Ask whoever runs Mackie to check the key in .env.",
    ),
    (
        litellm.NotFoundError,
        "The AI service doesn't know the configured model. Ask whoever runs Mackie to check the "
        "model setting in .env.",
    ),
    (
        litellm.BadRequestError,
        "The AI service couldn't handle that request. Try rephrasing it, or start a new "
        "conversation.",
    ),
    (
        litellm.APIConnectionError,
        "Mackie can't reach the AI service right now. Please try again in a moment.",
    ),
    # Not subclasses of litellm.APIError, hence each listed.
    *(
        (kind, "The AI service is having trouble right now. Please try again in a few minutes.")
        for kind in (
            litellm.ServiceUnavailableError,
            litellm.InternalServerError,
            litellm.APIError,
        )
    ),
)


def user_message(error: BaseException) -> str:
    """
    What to tell the user about an error that ended a turn or a session.

    A DatabaseError (Mackie isn't set up) says what to fix; any other database
    error becomes UNAVAILABLE_MESSAGE; a provider error says what happened and
    what to do; anything else becomes UNEXPECTED_MESSAGE. Logging the error is
    the caller's job, once, where it is handled.
    """
    if isinstance(error, DatabaseError):
        return f"Mackie can't reach its database. {error}"
    if isinstance(error, psycopg.Error):
        return UNAVAILABLE_MESSAGE
    for kind, message in _PROVIDER_MESSAGES:
        if isinstance(error, kind):
            return message
    return UNEXPECTED_MESSAGE
