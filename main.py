"""
CLI entry point for Mackie.

Owns: org identification prompt, conversation loop, and rich terminal output.
Does not own LLM interaction (assistant.py) or persistence (memory.py).
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from dotenv import load_dotenv
from rich.console import Console
from rich.markdown import Markdown
from rich.prompt import Prompt

load_dotenv(Path(__file__).parent / ".env")

from agent import memory as mem  # after load_dotenv, which config reads
from agent.assistant import Assistant

console = Console()


async def main() -> None:
    """Run the interactive CLI conversation loop."""
    console.print("\n[bold cyan]Mackie[/bold cyan] — NPO Social Media Assistant\n")

    org_name = Prompt.ask("[bold]Organization name[/bold]")
    while not mem.slugify(org_name):
        org_name = Prompt.ask("[bold]Organization name[/bold] [dim](required)[/dim]")
    org_url = Prompt.ask("Organization website [dim](press Enter to skip)[/dim]", default="")

    org_id = mem.slugify(org_name)
    try:
        profile = mem.load_profile(org_id)
    except mem.ProfileError as e:
        console.print(f"[red]Can't read this organization's memory file.[/red] {e}")
        return
    if not profile.get("name"):
        profile["name"] = org_name
        profile["website"] = org_url
        mem.save_profile(org_id, profile)

    assistant = Assistant(org_id)

    console.print(f"\n[dim]Working with [bold]{org_name}[/bold] (id: {org_id})[/dim]")
    console.print(
        "[dim]Commands: 'quit' to exit · 'reset' to clear conversation (memory kept)[/dim]\n"
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
            await assistant.clear_history()
            console.print(
                "[dim]Conversation history cleared. Org memory and ledger are preserved.[/dim]\n"
            )
            continue

        with console.status("[dim]Mackie is thinking…[/dim]", spinner="dots"):
            try:
                response = await assistant.chat(user_input)
            except Exception as e:
                console.print(f"[red]Error:[/red] {e}")
                continue

        console.print()
        console.print("[bold cyan]Mackie[/bold cyan]")
        console.print(Markdown(response))
        console.print()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except (KeyboardInterrupt, EOFError):
        console.print("\n[dim]Goodbye![/dim]")
