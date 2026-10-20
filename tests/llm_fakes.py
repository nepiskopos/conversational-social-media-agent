"""
Stand-ins for LiteLLM's responses, shared by the test files.

Owns: streamed chunks (text, a tool call, the closing chunk), a gated stream
that waits on events between tokens, and a non-streamed completion, shaped
like what litellm.acompletion returns. Patch
agent.graph.litellm.acompletion (or another module's) with a function that
returns these.
"""

import asyncio
from unittest.mock import MagicMock

# How long a gated stream waits for its gate: a hang fails the test instead of the run.
GATE_WAIT_S: float = 5.0


def text_chunk(text: str | None, finish_reason: str | None = None) -> MagicMock:
    """One stream chunk carrying text, or the closing chunk when finish_reason is set."""
    chunk = MagicMock()
    chunk.choices = [
        MagicMock(delta=MagicMock(content=text, tool_calls=None), finish_reason=finish_reason)
    ]
    return chunk


async def text_stream(text: str):
    """A streamed reply: one text chunk, then the closing chunk."""
    yield text_chunk(text)
    yield text_chunk(None, "stop")


async def tool_call_stream(tc_id: str, name: str, arguments: str):
    """A streamed reply that is one tool call (arguments as a JSON string), then the closing chunk."""
    tc_delta = MagicMock()
    tc_delta.index = 0
    tc_delta.id = tc_id
    tc_delta.function.name = name
    tc_delta.function.arguments = arguments
    chunk = MagicMock()
    chunk.choices = [
        MagicMock(delta=MagicMock(content=None, tool_calls=[tc_delta]), finish_reason=None)
    ]
    yield chunk
    yield text_chunk(None, "tool_calls")


def completion(content: str) -> MagicMock:
    """A non-streamed completion whose message is content (summaries, the brief, the calendar, the judge)."""
    return MagicMock(choices=[MagicMock(message=MagicMock(content=content))])


async def gated_stream(tokens: list[str], gates: dict[int, asyncio.Event]):
    """Stream tokens; before token i, wait for gates[i] if there is one (at most GATE_WAIT_S)."""
    for i, token in enumerate(tokens):
        if i in gates:
            await asyncio.wait_for(gates[i].wait(), GATE_WAIT_S)
        yield text_chunk(token)
    yield text_chunk(None, "stop")
