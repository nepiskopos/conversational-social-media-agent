"""
Chainlit web UI entry point for Mackie.

Owns: the login (password_auth_callback), onboarding (asking for and
confirming the org, then opening it only if the user may), the greeting with
the agenda of what needs the user, warnings about skipped skill files, one
resumable conversation per user and org (the last messages shown again on
return, and a "Start a new conversation" action), and streaming each reply
with its tool steps. Does not own conversation logic (assistant.py), accounts
(accounts.py) or persistence (memory.py).
"""

from __future__ import annotations

import logging
from pathlib import Path

import chainlit as cl
import litellm
import psycopg
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

from agent import accounts, skills  # after load_dotenv, which config reads
from agent import memory as mem
from agent.assistant import Assistant, agenda_summary, count_noun
from agent.db import DatabaseError
from agent.errors import NEW_CONVERSATION_LABEL, user_message
from agent.logs import configure_logging, describe, stack

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


def _new_conversation_action() -> cl.Action:
    """The button that clears this user's conversation with the org."""
    return cl.Action(name=NEW_CONVERSATION_ACTION, payload={}, label=NEW_CONVERSATION_LABEL)


def _recap_text(text: str) -> str:
    """A recapped message, cut to RECAP_MESSAGE_CHARS."""
    return text if len(text) <= RECAP_MESSAGE_CHARS else text[:RECAP_MESSAGE_CHARS].rstrip() + " …"


NO_ACCESS_PROMPT: str = (
    "You don't have access to **{org_name}**. Ask an admin to add you, "
    "or enter another organization."
)


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
    return cl.User(identifier=user.username, metadata={"user_id": user.user_id})


def _parse_org_input(text: str) -> tuple[str, str]:
    """Split "Name, url" or "Name\nurl" into (org_name, org_url); url may be empty."""
    text = text.strip()
    if "," in text and "\n" not in text:
        org_name, org_url = text.split(",", 1)
        return org_name.strip(), org_url.strip()
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return (lines[0] if lines else ""), (lines[1] if len(lines) > 1 else "")


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
            prompt = "I couldn't find an organization name in that. What's the name?"
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
    except (DatabaseError, psycopg.Error) as e:
        logger.error("A session couldn't start: %s", describe(e))
        await cl.Message(content=user_message(e)).send()


async def _start() -> None:
    """The session start, for start() to report a database that isn't set up."""
    login_user = cl.user_session.get("user")
    user = await accounts.get_user(login_user.metadata["user_id"]) if login_user else None
    if user is None:
        await cl.Message(content="This account no longer exists. Ask an admin for access.").send()
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
            + "\n\nEverything else works. Fix the file(s), or run `python scripts/validate.py` to check."
        ).send()

    org_id = mem.slugify(org_name)
    profile = await mem.load_profile(org_id)
    if not profile.get("name"):
        await mem.fill_org_details(org_id, org_name, org_url)
        profile["name"] = org_name

    # One conversation per user and org, resumed after a refresh or on another day; other
    # users of the org have their own. A ":" can't occur in an org id, so this thread id
    # can never equal another org's id (an Assistant's default thread).
    assistant = Assistant(
        org_id, thread_id=f"{org_id}:web-user-{user.user_id}", user_id=user.user_id
    )
    cl.user_session.set("assistant", assistant)

    remembered = len(profile.get("preferences", {})), len(profile.get("post_ledger", []))
    if any(remembered):
        # Lead with what needs doing, like a teammate would, instead of waiting to be asked.
        next_step = await agenda_summary(org_id) or "What would you like to work on?"
        greeting = (
            f"Welcome back! I remember "
            f"{count_noun(remembered[0], 'brand preference', 'brand preferences')} and "
            f"{count_noun(remembered[1], 'post', 'posts')} for **{profile['name']}**. {next_step}"
        )
    else:
        greeting = (
            f"Got it! I'm set up for **{org_name}**. What would you like to work on?\n\n"
            "Tip: paste two or three of your recent posts and I'll learn your voice from them."
        )
    try:
        recap = await assistant.recent_exchanges(RESUMED_MESSAGES_SHOWN)
    except accounts.AccessRevoked as e:  # removed in the moment since the org was opened
        cl.user_session.set("assistant", None)
        await cl.Message(content=f"{e} Refresh the page to choose another organization.").send()
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
        cl.user_session.set("assistant", None)
        await cl.Message(content=f"{e} Refresh the page to choose another organization.").send()
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
        # The session ends with the access: nothing more goes to this org from it.
        cl.user_session.set("assistant", None)
        await msg.stream_token(f"{e} Refresh the page to choose another organization.")
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
