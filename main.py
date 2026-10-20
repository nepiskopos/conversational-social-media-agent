"""
CLI entry point for Mackie.

Owns: the login, asking for the org and opening it only if the user may
(ending the session if that access is removed), the agenda of what needs the
user, the conversation loop with its commands (reset clears the conversation
and keeps org memory; quit exits), and rich terminal output. Each user has
one resumable conversation per org. Does not own LLM interaction
(assistant.py), accounts (accounts.py) or persistence (memory.py).
"""

from __future__ import annotations

import asyncio
import logging
import unicodedata
from pathlib import Path

from rich.console import Console
from rich.markdown import Markdown
from rich.markup import escape
from rich.prompt import Prompt

from agent.config import load_settings

load_settings(Path(__file__).parent / ".env")

from agent import accounts, db  # after load_settings, as config reads the settings
from agent import memory as mem
from agent.assistant import Assistant, agenda_summary
from agent.errors import user_message
from agent.logs import configure_logging, describe, stack
from agent.research import website_url

# Warnings and errors only, so log lines don't interrupt the conversation on screen.
configure_logging("WARNING")
logger = logging.getLogger("agent.main")

console = Console()
LOGIN_ATTEMPTS: int = 3


def _without_controls(text: str) -> str:
    """
    Text without control characters, but for line breaks and tabs.

    rich passes ESC through, so a reply could carry terminal codes that hide
    text (conceal), move the cursor over what was shown, or set the clipboard.
    """
    return "".join(char for char in text if char in "\n\t" or unicodedata.category(char) != "Cc")


async def _login() -> accounts.User | None:
    """Ask for a username and password, up to LOGIN_ATTEMPTS times; None if every try fails."""
    for _ in range(LOGIN_ATTEMPTS):
        username = Prompt.ask("[bold]Username[/bold]")
        password = Prompt.ask("[bold]Password[/bold]", password=True)
        if user := await accounts.authenticate(username, password):
            return user
        console.print("[red]Wrong username or password, or the account is locked.[/red]")
    return None


async def main() -> None:
    """Run the interactive CLI conversation loop, closing the database pool at the end."""
    try:
        await _converse()
    except Exception as e:
        logger.error("The terminal session stopped: %s\n%s", describe(e), stack(e))
        console.print(f"[red]{user_message(e)}[/red]")
    finally:
        await db.close_pool()


async def _converse() -> None:
    """Log in, open an org and hold the conversation."""
    console.print("\n[bold cyan]Mackie[/bold cyan] — AI Social Media Manager for Nonprofits\n")
    user = await _login()
    if user is None:
        return
    if orgs := await accounts.user_orgs(user.user_id):
        console.print(f"[dim]Your organizations: {', '.join(orgs)}[/dim]")

    org_name = Prompt.ask("[bold]Organization name[/bold]")
    while not mem.slugify(org_name):
        org_name = Prompt.ask(
            f"[bold]Organization name[/bold] [dim](required, up to {mem.MAX_ORG_NAME_CHARS} characters)[/dim]"
        )
    org_id = mem.slugify(org_name)
    if not await accounts.open_org(user.user_id, org_id):
        console.print(
            f"[red]You don't have access to {escape(org_name)}.[/red] Ask an admin to add you."
        )
        return
    typed = Prompt.ask("Organization website [dim](press Enter to skip)[/dim]", default="")
    while typed.strip() and not website_url(typed):
        typed = Prompt.ask(
            "That isn't a website address; try again, or press Enter to skip", default=""
        )
    org_url = website_url(typed)

    # Kept where the org already has them, so a typed website still fills a missing one.
    await mem.fill_org_details(org_id, org_name, org_url)
    assistant = Assistant.for_user(org_id, user, "cli")

    # escape: a name like "Acme [/b] Trust" is text to show, not markup that crashes rich.
    console.print(f"\n[dim]Working with [bold]{escape(org_name)}[/bold] (id: {org_id})[/dim]")
    console.print(
        "[dim]Commands: 'quit' to exit · 'reset' to clear conversation (memory kept)[/dim]\n"
    )
    if summary := await agenda_summary(org_id):
        console.print(f"{summary}\n")
    try:
        resumed = bool(await assistant.recent_exchanges(1))
    except accounts.AccessRevoked as e:  # removed in the moment since the org was opened
        console.print(f"[red]{e}[/red] The session has ended.")
        return
    except Exception as e:
        logger.error("Couldn't check for a conversation to resume for %s: %s", org_id, describe(e))
        resumed = False  # the hint is a courtesy; the session works without it
    if resumed:
        console.print(
            "[dim]Carrying on from your last conversation; 'reset' starts a new one.[/dim]\n"
        )

    while True:
        try:
            user_input = Prompt.ask("[bold green]You[/bold green]")
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]Goodbye![/dim]")
            break

        if not user_input.strip():
            continue

        cmd = user_input.strip().lower()
        if cmd in ("quit", "exit", "q"):
            console.print("[dim]Goodbye![/dim]")
            break
        if cmd == "reset":
            try:
                await assistant.clear_history()
            except accounts.AccessRevoked as e:
                console.print(f"[red]{e}[/red] The session has ended.")
                break
            except Exception as e:
                logger.error("Reset failed for %s: %s", org_id, describe(e))
                console.print(f"[red]Error:[/red] {user_message(e)}")
                continue
            console.print(
                "[dim]Conversation history cleared. Org memory and ledger are preserved.[/dim]\n"
            )
            continue

        with console.status("[dim]Mackie is thinking…[/dim]", spinner="dots"):
            try:
                response = await assistant.chat(user_input)
            except accounts.AccessRevoked as e:
                console.print(f"[red]{e}[/red] The session has ended.")
                break
            except Exception as e:
                logger.error("A turn failed for %s: %s\n%s", org_id, describe(e), stack(e))
                console.print(f"[red]Error:[/red] {user_message(e)}")
                continue

        console.print()
        console.print("[bold cyan]Mackie[/bold cyan]")
        console.print(Markdown(_without_controls(response)))
        console.print()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except (KeyboardInterrupt, EOFError):
        console.print("\n[dim]Goodbye![/dim]")
