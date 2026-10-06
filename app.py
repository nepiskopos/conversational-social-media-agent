"""
Chainlit web UI entry point for Mackie.

Owns: session initialization, org identification prompt, and message routing.
Does not own conversation logic (assistant.py) or persistence (memory.py).
"""

from __future__ import annotations

import uuid
from pathlib import Path

import chainlit as cl
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

from agent import memory as mem  # after load_dotenv, which config reads
from agent import skills
from agent.assistant import Assistant

ORG_PROMPT: str = (
    "Hi! I'm **Mackie**, your NPO social media assistant. "
    "What organization are you working with?\n\n"
    "Share the **name** and **website URL** (optional) — one per line or separated by a comma."
)
ONBOARDING_TIMEOUT_S: int = 120


def _parse_org_input(text: str) -> tuple[str, str]:
    """Split "Name, url" or "Name\nurl" into (org_name, org_url); url may be empty."""
    text = text.strip()
    if "," in text and "\n" not in text:
        org_name, org_url = text.split(",", 1)
        return org_name.strip(), org_url.strip()
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return (lines[0] if lines else ""), (lines[1] if len(lines) > 1 else "")


async def _ask_for_org() -> tuple[str, str] | None:
    """
    Ask for the org name and URL, then confirm before creating anything.

    Without the confirmation step a greeting like "hello" became an org with
    its own profile directory. Returns None if the user times out.
    """
    prompt = ORG_PROMPT
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
        if confirm["payload"].get("ok"):
            return org_name, org_url
        prompt = "No problem. What's the organization's name and website?"


@cl.on_chat_start
async def start() -> None:
    """Initialize a new Chainlit session: ask for and confirm the org, then create the Assistant."""
    org = await _ask_for_org()
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
    try:
        profile = mem.load_profile(org_id)
    except mem.ProfileError as e:
        await cl.Message(content=f"I can't read this organization's memory file. {e}").send()
        return
    if not profile.get("name"):
        profile["name"] = org_name
        profile["website"] = org_url
        mem.save_profile(org_id, profile)

    # Each browser session gets its own thread_id so concurrent users on the
    # same org don't share or corrupt each other's conversation history. The ":"
    # can't occur in an org id, so no org's default thread (its id) can match.
    thread_id = f"{org_id}:{uuid.uuid4().hex}"
    cl.user_session.set("assistant", Assistant(org_id, thread_id=thread_id))

    remembered = len(profile.get("preferences", {})), len(profile.get("post_ledger", []))
    if any(remembered):
        greeting = (
            f"Welcome back! I remember {remembered[0]} brand preference(s) and "
            f"{remembered[1]} post(s) for **{profile['name']}**. What would you like to work on?"
        )
    else:
        greeting = (
            f"Got it! I'm set up for **{org_name}**. What would you like to work on?\n\n"
            "Tip: paste two or three of your recent posts and I'll learn your voice from them."
        )
    await cl.Message(content=greeting).send()


@cl.on_message
async def on_message(message: cl.Message) -> None:
    """Forward each user message to the Assistant, streaming tool steps and text in real time."""
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
    except Exception as e:
        # Without this the user sees an empty reply and no hint that anything failed.
        await msg.stream_token(f"\n\n**Something went wrong:** {e}. Please try again.")

    await msg.update()
