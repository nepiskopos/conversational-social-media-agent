"""
Unit tests for agent/graph.py: message conversion (including answering tool
calls whose result was lost), routing (tools, save_check) and turn_reply, the
sliding window (capped by messages and tokens), old tool results shortened,
the running summary batched once SUMMARY_BATCH_MESSAGES have left the window
(also mid-turn) and what its prompt asks for, prompt caching, reading streamed
replies and tool-call fragments, tool-step previews, provider fallback and
empty-reply retries, checkpoint pruning in PostgreSQL, and threads that keep
working after an interrupted turn or a lone surrogate. The checkpointer's
row-level security is tested in test_security.py.
"""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import litellm
import psycopg
import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.graph import END
from psycopg import errors

from tests.llm_fakes import completion, text_chunk, text_stream, tool_call_stream


def test_text_said_before_a_tool_call_stays_in_the_history_sent_to_the_model():
    """Sent as None, the model would forget what it had told the user."""
    from agent.graph import _to_litellm

    call = {"id": "tc_1", "name": "save_post", "args": {}, "type": "tool_call"}
    sent = _to_litellm(
        [
            AIMessage(content="Let me save that.", tool_calls=[call]),
            ToolMessage(content="Saved", tool_call_id="tc_1", name="save_post"),
        ]
    )
    assert sent[0]["content"] == "Let me save that." and sent[0]["tool_calls"][0]["id"] == "tc_1"
    assert _to_litellm([AIMessage(content="", tool_calls=[call])])[0]["content"] is None


def test_to_litellm_converts_a_mixed_message_sequence():
    from agent.graph import _to_litellm

    msgs = [
        HumanMessage(content="Save a post"),
        AIMessage(
            content="",
            tool_calls=[
                {
                    "id": "tc_1",
                    "name": "save_post",
                    "args": {"platform": "linkedin", "content": "Hi"},
                    "type": "tool_call",
                }
            ],
        ),
        ToolMessage(content="Saved as post_001", tool_call_id="tc_1", name="save_post"),
        AIMessage(content="Done!"),
    ]
    assert _to_litellm(msgs) == [
        {"role": "user", "content": "Save a post"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "tc_1",
                    "type": "function",
                    "function": {
                        "name": "save_post",
                        "arguments": json.dumps({"platform": "linkedin", "content": "Hi"}),
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "tc_1",
            "name": "save_post",
            "content": "Saved as post_001",
        },
        {"role": "assistant", "content": "Done!"},
    ]


def _calls(*ids: str) -> AIMessage:
    """An AI message calling save_post once per id."""
    return AIMessage(
        content="",
        tool_calls=[{"id": i, "name": "save_post", "args": {}, "type": "tool_call"} for i in ids],
    )


def _result(call_id: str) -> ToolMessage:
    """The save_post result answering call_id."""
    return ToolMessage(content=f"done {call_id}", tool_call_id=call_id, name="save_post")


LOST = "LOST"  # stands for LOST_TOOL_RESULT in the expected (role, call id, content) rows


@pytest.mark.parametrize(
    "messages, expected",
    [
        pytest.param(
            [HumanMessage(content="q"), _calls("a")],
            [("user", None, "q"), ("assistant", None, None), ("tool", "a", LOST)],
            id="call-at-the-end-of-history",
        ),
        pytest.param(
            [HumanMessage(content="q"), _calls("a"), HumanMessage(content="again")],
            [
                ("user", None, "q"),
                ("assistant", None, None),
                ("tool", "a", LOST),
                ("user", None, "again"),
            ],
            id="call-followed-by-a-new-user-message",
        ),
        pytest.param(
            [_calls("a", "b"), _result("a"), AIMessage(content="ok")],
            [
                ("assistant", None, None),
                ("tool", "a", "done a"),
                ("tool", "b", LOST),
                ("assistant", None, "ok"),
            ],
            id="one-of-two-calls-answered",
        ),
        pytest.param(
            [_calls("a", "b"), _result("b"), _result("a"), AIMessage(content="ok")],
            [
                ("assistant", None, None),
                ("tool", "b", "done b"),
                ("tool", "a", "done a"),
                ("assistant", None, "ok"),
            ],
            id="complete-history-unchanged",
        ),
    ],
)
def test_to_litellm_answers_tool_calls_whose_result_was_lost(messages, expected):
    """Providers reject a tool call with no result; a lost one is answered right after its turn."""
    from agent.graph import LOST_TOOL_RESULT, _to_litellm

    result = _to_litellm(messages)
    rows = [(m["role"], m.get("tool_call_id"), m["content"]) for m in result]
    assert rows == [
        (role, call_id, LOST_TOOL_RESULT if content == LOST else content)
        for role, call_id, content in expected
    ]


@pytest.mark.parametrize(
    "message, expected",
    [
        pytest.param(_calls("tc_1"), "tools", id="tool-calls"),
        pytest.param(AIMessage(content="Here is my response."), END, id="text-reply"),
        pytest.param(AIMessage(content="ok", tool_calls=[]), END, id="empty-tool-calls"),
    ],
)
async def test_should_continue_routes_on_the_last_message(message, expected):
    """No post ids in the reply, so routing needs no ledger (and no database)."""
    from agent.graph import should_continue

    assert await should_continue({"messages": [message], "org_id": "test-org"}) == expected


def _tool_turn(n: int, tool_results: int) -> list:
    """One user turn: question, AI tool calls, their results, final reply."""
    calls = [
        {"id": f"t{n}_{k}", "name": "save_post", "args": {}, "type": "tool_call"}
        for k in range(tool_results)
    ]
    return [
        HumanMessage(content=f"q{n}"),
        AIMessage(content="", tool_calls=calls),
        *[ToolMessage(content="ok", tool_call_id=c["id"], name="save_post") for c in calls],
        AIMessage(content=f"a{n}"),
    ]


def _long_history(turns: int = 30, padding: str = "") -> list:
    """Plain question/answer turns ("q0", "a0", ...), each message followed by padding."""
    messages = []
    for n in range(turns):
        messages += [HumanMessage(content=f"q{n}{padding}"), AIMessage(content=f"a{n}{padding}")]
    return messages


def _big_result_turn(n: int, result_chars: int) -> list:
    """One user turn whose single tool result is result_chars long."""
    call = {"id": f"t{n}", "name": "research_org", "args": {}, "type": "tool_call"}
    return [
        HumanMessage(content=f"q{n}"),
        AIMessage(content="", tool_calls=[call]),
        ToolMessage(content="r" * result_chars, tool_call_id=f"t{n}", name="research_org"),
        AIMessage(content=f"a{n}"),
    ]


def _a_slice_would_start_on_a_tool_result(max_messages: int) -> list:
    """Turns of nine messages, so a plain slice of the newest max_messages starts mid-turn."""
    messages = [m for n in range(10) for m in _tool_turn(n, tool_results=6)]
    assert isinstance(messages[-max_messages], ToolMessage)
    return messages


def _a_long_turn_with_a_save_check(max_messages: int) -> list:
    """A turn longer than the window, with the save_check follow-up inside the window."""
    from agent.graph import SAVE_CHECK_NAME

    return [
        HumanMessage(content="q0"),
        AIMessage(content="a0"),
        HumanMessage(content="Suggest posts"),
        *[AIMessage(content=f"step {n}") for n in range(max_messages)],
        HumanMessage(content="Automatic check", name=SAVE_CHECK_NAME),
        AIMessage(content="done"),
    ]


@pytest.mark.parametrize(
    "history, first",
    [
        pytest.param(lambda m: _tool_turn(0, tool_results=2), "q0", id="short-history-unchanged"),
        pytest.param(_a_slice_would_start_on_a_tool_result, "q6", id="not-on-an-orphan-result"),
        pytest.param(_a_long_turn_with_a_save_check, "Suggest posts", id="not-on-the-save-check"),
        pytest.param(
            lambda m: _tool_turn(0, tool_results=2) + _tool_turn(1, tool_results=m),
            "q1",
            id="whole-turn-of-more-messages-than-the-window",
        ),
        pytest.param(
            lambda m: _tool_turn(0, tool_results=2) + _big_result_turn(1, 200_000),
            "q1",
            id="whole-turn-of-more-tokens-than-the-window",
        ),
    ],
)
def test_window_starts_at_a_user_turn(history, first):
    """
    Providers reject a tool result whose call was cut off, so the window starts at a
    user message: never at a tool result, nor at the save_check follow-up, which is
    part of the user's turn. A current turn bigger than the window is kept whole, so
    its tool calls stay paired with their results.
    """
    from agent.graph import MAX_MESSAGES, _is_user_turn, _window_start

    messages = history(MAX_MESSAGES)
    start = _window_start(messages)
    assert _is_user_turn(messages[start]) and messages[start].content == first


LONG_MESSAGE_CHARS: int = 3_000


def _long_tool_call_arguments() -> list:
    """Turns whose size is all in the tool call's arguments, as with a long save_post draft."""
    messages = []
    for n in range(20):
        args = {"content": "x" * 2 * LONG_MESSAGE_CHARS}
        call = {"id": f"t{n}", "name": "save_post", "args": args, "type": "tool_call"}
        messages += [
            HumanMessage(content=f"q{n}"),
            AIMessage(content="", tool_calls=[call]),
            ToolMessage(content="ok", tool_call_id=f"t{n}", name="save_post"),
            AIMessage(content=f"a{n}"),
        ]
    return messages


def _a_turn_that_fills_the_budget_exactly() -> list:
    """Three turns; the newest two together are exactly the window's budget."""
    from agent.graph import CHARS_PER_TOKEN, WINDOW_TOKENS

    filler = WINDOW_TOKENS * CHARS_PER_TOKEN - len("q1") - len("q2") - len("a2")
    return [
        HumanMessage(content="q0"),
        AIMessage(content="a0"),
        HumanMessage(content="q1"),
        AIMessage(content="x" * filler),
        HumanMessage(content="q2"),
        AIMessage(content="a2"),
    ]


@pytest.mark.parametrize(
    "history",
    [
        pytest.param(lambda: _long_history(30), id="many-short-messages"),
        pytest.param(lambda: _long_history(15, "x" * LONG_MESSAGE_CHARS), id="few-long-messages"),
        pytest.param(_long_tool_call_arguments, id="long-tool-call-arguments"),
        pytest.param(_a_turn_that_fills_the_budget_exactly, id="exactly-the-budget-fits"),
    ],
)
def test_window_is_capped_by_messages_and_by_tokens(history):
    """
    The window is as long as fits both caps: one more turn would break one.
    Short messages hit MAX_MESSAGES; long ones hit WINDOW_TOKENS first, and tool-call
    arguments count as much as text. A window of exactly the budget still fits.
    """
    from agent.graph import (
        CHARS_PER_TOKEN,
        MAX_MESSAGES,
        WINDOW_TOKENS,
        _is_user_turn,
        _message_chars,
        _window_start,
    )

    messages = history()
    budget = WINDOW_TOKENS * CHARS_PER_TOKEN
    start = _window_start(messages)
    one_more_turn = messages[max(i for i in range(start) if _is_user_turn(messages[i])) :]

    assert _is_user_turn(messages[start])
    assert sum(map(_message_chars, messages[start:])) <= budget
    assert len(messages) - start <= MAX_MESSAGES
    assert sum(map(_message_chars, one_more_turn)) > budget or len(one_more_turn) > MAX_MESSAGES


def test_window_counts_old_tool_results_at_their_shortened_size():
    """Ten 18k-char results would be far over budget in full, but only the last two are."""
    from agent.graph import CHARS_PER_TOKEN, WINDOW_TOKENS, _window_start

    messages = [m for n in range(8) for m in _big_result_turn(n, 18_000)]
    assert sum(len(m.content) for m in messages) > 2 * WINDOW_TOKENS * CHARS_PER_TOKEN
    assert _window_start(messages) == 0


def test_old_tool_results_are_shortened_on_copies():
    """Results before the last two user turns are cut; recent, short and non-tool ones are not."""
    from agent.graph import (
        OLD_TOOL_RESULT_CHARS,
        OLD_TOOL_RESULT_NOTE,
        SAVE_CHECK_NAME,
        _shorten_old_tool_results,
    )

    long, short = "L" * (OLD_TOOL_RESULT_CHARS + 400), "s" * OLD_TOOL_RESULT_CHARS
    old = [
        HumanMessage(content="q0"),
        _calls("a", "b"),
        ToolMessage(content=long, tool_call_id="a", name="save_post"),
        ToolMessage(content=short, tool_call_id="b", name="save_post"),
        AIMessage(content=long),
    ]
    # The save_check follow-up is not a user turn, so it doesn't move the boundary.
    recent = [
        HumanMessage(content="q1"),
        _calls("c"),
        ToolMessage(content=long, tool_call_id="c", name="save_post"),
        AIMessage(content="a1"),
        HumanMessage(content="q2"),
        AIMessage(content="post_001"),
        HumanMessage(content="Automatic check", name=SAVE_CHECK_NAME),
        _calls("d"),
        ToolMessage(content=long, tool_call_id="d", name="save_post"),
    ]
    messages = old + recent

    shortened = _shorten_old_tool_results(messages)

    assert shortened[2].content == long[:OLD_TOOL_RESULT_CHARS] + OLD_TOOL_RESULT_NOTE
    assert shortened[2].tool_call_id == "a"
    assert shortened[:2] + shortened[3:] == messages[:2] + messages[3:]
    assert messages[2].content == long  # the stored history keeps the full result
    assert _shorten_old_tool_results(old) == old  # one user turn: everything is recent


def _out_of_window(count: int) -> list:
    """Plain turns that leave exactly count messages (an even number) before the window."""
    from agent.graph import MAX_MESSAGES, _window_start

    messages = _long_history((MAX_MESSAGES + count) // 2)
    assert _window_start(messages) == count
    return messages


@pytest.mark.parametrize(
    "out",
    [
        pytest.param(0, id="history-fits-the-window"),
        pytest.param(8, id="fewer-than-a-batch-out-of-the-window"),
    ],
)
async def test_summarize_waits_for_a_batch_of_messages_to_leave_the_window(out):
    """No summary call per turn: none until a whole batch has left the window."""
    from agent.graph import SUMMARY_BATCH_MESSAGES, _build_summarize

    assert out < SUMMARY_BATCH_MESSAGES
    messages = _out_of_window(out)
    mock_completion = AsyncMock()
    with patch("agent.graph.litellm.acompletion", mock_completion):
        update = await _build_summarize("gpt-4o-mini")({"messages": messages, "org_id": "org"})
    assert update == {}
    mock_completion.assert_not_called()


async def test_summarize_folds_a_batch_that_left_the_window_into_the_summary():
    """One call covers exactly the messages before the window."""
    from agent.graph import SUMMARY_BATCH_MESSAGES, _build_summarize

    messages = _out_of_window(SUMMARY_BATCH_MESSAGES)
    mock_completion = AsyncMock(return_value=completion("- user prefers warm voice"))
    events: list[dict] = []
    with (
        patch("agent.graph.litellm.acompletion", mock_completion),
        patch("agent.graph.get_stream_writer", return_value=events.append),
    ):
        update = await _build_summarize("gpt-4o-mini")({"messages": messages, "org_id": "org"})

    assert update == {
        "summary": "- user prefers warm voice",
        "summarized_count": SUMMARY_BATCH_MESSAGES,
    }
    mock_completion.assert_awaited_once()
    transcript = mock_completion.call_args.kwargs["messages"][0]["content"].split("<new_messages>")[
        1
    ]
    assert transcript.count("USER: ") == transcript.count("MACKIE: ") == SUMMARY_BATCH_MESSAGES // 2
    assert f"USER: q{SUMMARY_BATCH_MESSAGES // 2 - 1}\n" in transcript
    assert f"USER: q{SUMMARY_BATCH_MESSAGES // 2}\n" not in transcript  # inside the window
    assert events[0]["tool"] == "summarize_history"


@pytest.mark.parametrize(
    "out, summarized_count, expected",
    [
        pytest.param(0, 0, 0, id="nothing-out-of-the-window"),
        pytest.param(8, 0, 0, id="unsummarized-messages-stay-visible"),
        pytest.param(10, 0, 10, id="a-batch-out-starts-at-the-window"),
        pytest.param(14, 10, 10, id="starts-where-the-summary-ends-until-the-next-batch"),
        pytest.param(20, 10, 20, id="the-next-batch"),
        pytest.param(8, 10, 10, id="window-moved-back-before-the-summary-end"),
    ],
)
def test_view_start_leaves_no_gap_between_the_summary_and_the_window(
    out, summarized_count, expected
):
    from agent.graph import SUMMARY_BATCH_MESSAGES, _view_start

    assert SUMMARY_BATCH_MESSAGES == 10  # the cases are written for a batch of ten
    assert _view_start(_out_of_window(out), summarized_count) == expected


async def _sent_history(state: dict) -> list[dict]:
    """The messages after the system prompt that call_model sends for state."""
    from agent.graph import _build_call_model

    mock_completion = AsyncMock(return_value=text_stream("ok"))
    with (
        patch("agent.graph.litellm.acompletion", mock_completion),
        patch("agent.graph.get_stream_writer", return_value=lambda x: None),
    ):
        await _build_call_model("gpt-4o-mini")(state)
    return mock_completion.call_args.kwargs["messages"][1:]


def _messages_left_the_window_unsummarized() -> tuple[list, int, int]:
    """Fewer than a batch out of the window: everything is sent."""
    return _out_of_window(8), 0, 0


def _a_batch_left_the_window() -> tuple[list, int, int]:
    """Ninety plain messages: the window alone is sent."""
    return _out_of_window(50), 0, 50


def _the_window_moved_back_over_the_summary() -> tuple[list, int, int]:
    """
    History summarized up to its window, then one more turn: the big result two turns
    back is now shortened, which frees budget, so the window moves back over messages
    the summary already covers.
    """
    from agent.graph import _window_start

    before = _long_history(30, "x" * 1_000) + _big_result_turn(30, 40_000)
    after = [*before, HumanMessage(content="q31"), AIMessage(content="a31")]
    after.append(HumanMessage(content="q32"))
    summarized_count = _window_start([*before, HumanMessage(content="q31")])
    assert _window_start(after) < summarized_count
    return after, summarized_count, summarized_count


@pytest.mark.parametrize(
    "case",
    [
        pytest.param(_messages_left_the_window_unsummarized, id="unsummarized-messages-sent"),
        pytest.param(_a_batch_left_the_window, id="a-summarized-batch-is-not-sent"),
        pytest.param(_the_window_moved_back_over_the_summary, id="summarized-messages-sent-once"),
    ],
)
async def test_call_model_sends_history_from_the_view_start(database, case):
    """
    The model gets the window (at most MAX_MESSAGES) once the messages before it are
    summarized, and everything since the summary's end until then, never a summarized
    message twice.
    """
    from agent.graph import MAX_MESSAGES

    messages, summarized_count, start = case()
    state = {"messages": messages, "org_id": "org", "summarized_count": summarized_count}
    sent = await _sent_history(state)

    assert len(sent) == len(messages) - start
    assert sent[0] == {"role": "user", "content": messages[start].content}
    assert len(sent) <= MAX_MESSAGES or start == summarized_count


async def test_call_model_sends_old_tool_results_shortened(database):
    from agent.graph import OLD_TOOL_RESULT_CHARS, OLD_TOOL_RESULT_NOTE

    messages = [m for n in range(3) for m in _big_result_turn(n, 5_000)]
    sent = await _sent_history({"messages": messages, "org_id": "org"})

    assert [m["content"] for m in sent if m["role"] == "tool"] == [
        "r" * OLD_TOOL_RESULT_CHARS + OLD_TOOL_RESULT_NOTE,
        "r" * 5_000,
        "r" * 5_000,
    ]
    assert messages[2].content == "r" * 5_000  # the state itself is not changed


async def test_summarize_is_incremental_and_carries_open_requests_forward():
    """
    Only messages not yet covered are sent, with the previous memory, so an open
    request or date in it can be carried forward; the instructions ask for that.
    """
    from agent.graph import OPEN_ITEMS_HEADING, _build_summarize

    messages = _long_history(30)
    previous = f"{OPEN_ITEMS_HEADING}\n- zz-gala-post due 2026-11-03\nSummary:\n- earlier facts"
    mock_completion = AsyncMock(return_value=completion("updated"))
    state = {
        "messages": messages,
        "org_id": "org",
        "summary": previous,
        "summarized_count": 10,
    }
    with (
        patch("agent.graph.litellm.acompletion", mock_completion),
        patch("agent.graph.get_stream_writer", return_value=lambda x: None),
    ):
        update = await _build_summarize("gpt-4o-mini")(state)

    assert update["summarized_count"] == 20
    prompt = mock_completion.call_args.kwargs["messages"][0]["content"]
    instructions, rest = prompt.split("<previous_summary>")
    assert f"\n{previous}\n</previous_summary>" in rest
    assert "USER: q4" not in prompt  # message 8, already summarized
    assert "USER: q5" in prompt  # message 10, first uncovered
    assert instructions.index(f"{OPEN_ITEMS_HEADING}\n") < instructions.index("Summary:\n")
    for asked in (
        "date, deadline or event",
        "Carry each open request and date forward",
        "'- none'",
        "300 words",
    ):
        assert asked in instructions


async def test_summarize_empty_reply_keeps_previous_state():
    """A reasoning model that used its whole budget thinking returns content=None."""
    from agent.graph import _build_summarize

    empty = MagicMock(choices=[MagicMock(message=MagicMock(content=None))])
    with patch("agent.graph.litellm.acompletion", AsyncMock(return_value=empty)):
        update = await _build_summarize("hosted_vllm/qwen")(
            {"messages": _long_history(30), "org_id": "org"}
        )
    assert update == {}  # summarized_count not advanced, so the same messages are retried


def test_render_transcript_includes_tool_calls_and_caps_long_results():
    from agent.graph import SUMMARY_MESSAGE_CHARS, _render_transcript

    messages = [
        HumanMessage(content="research us"),
        AIMessage(
            content="",
            tool_calls=[
                {"id": "t1", "name": "research_org", "args": {"org_name": "X"}, "type": "tool_call"}
            ],
        ),
        ToolMessage(
            content="y" * (SUMMARY_MESSAGE_CHARS + 500), tool_call_id="t1", name="research_org"
        ),
    ]
    text = _render_transcript(messages)
    assert "USER: research us" in text
    assert 'MACKIE called research_org({"org_name": "X"})' in text
    assert "y" * SUMMARY_MESSAGE_CHARS in text
    assert "y" * (SUMMARY_MESSAGE_CHARS + 1) not in text


async def test_call_model_adds_summary_to_uncached_block(database):
    from agent.graph import _build_call_model

    mock_completion = AsyncMock(return_value=text_stream("ok"))
    state = {
        "messages": [HumanMessage(content="hi")],
        "org_id": "org",
        "summary": "- zz-summary-fact",
    }
    with (
        patch("agent.graph.litellm.acompletion", mock_completion),
        patch("agent.graph.get_stream_writer", return_value=lambda x: None),
    ):
        await _build_call_model("claude-haiku-5-5")(state)

    cached, per_turn = mock_completion.call_args.kwargs["messages"][0]["content"]
    assert "zz-summary-fact" in per_turn["text"]
    assert "<conversation_summary>" in per_turn["text"]
    assert "zz-summary-fact" not in cached["text"]


def test_parse_tool_args_handles_empty_and_malformed_json():
    from agent.graph import _parse_tool_args
    from agent.tools import UNPARSED_ARGUMENTS_KEY

    assert _parse_tool_args('{"a": 1}') == {"a": 1}
    assert _parse_tool_args("") == {}
    assert _parse_tool_args('{"content": "line one\nline two"}') == {
        "content": "line one\nline two"
    }  # a raw line break inside a string, as models often write it
    for bad in ('{"content": "truncated', "[1, 2]"):
        assert _parse_tool_args(bad) == {UNPARSED_ARGUMENTS_KEY: bad}


@pytest.mark.parametrize(
    "model, cached",
    [("claude-haiku-5-5", True), ("anthropic/claude-haiku-5-5", True), ("gpt-4o-mini", False)],
)
def test_system_content_marks_only_the_stable_block_for_anthropic_caching(model, cached):
    """The per-turn block changes almost every turn, so it sits after the cache breakpoint."""
    from agent.graph import _system_content

    content = _system_content(model, "STABLE", "PER TURN")
    if cached:
        assert content == [
            {"type": "text", "text": "STABLE", "cache_control": {"type": "ephemeral"}},
            {"type": "text", "text": "PER TURN"},
        ]
    else:
        assert content == "STABLE\nPER TURN"  # other providers get no Anthropic-only fields


@pytest.mark.parametrize(
    "failure",
    [
        Exception("overloaded"),
        litellm.Timeout("no answer", model="claude-haiku-5-5", llm_provider="anthropic"),
    ],
    ids=["error", "timeout"],
)
async def test_call_model_falls_back_to_next_provider_when_primary_fails(database, failure):
    from agent.graph import _build_call_model

    calls: list[dict] = []

    async def completion(**kwargs):
        calls.append(kwargs)
        if kwargs["model"].startswith("claude"):
            raise failure
        return text_stream("from backup")

    with (
        patch("agent.graph.litellm.acompletion", new=completion),
        patch("agent.graph.get_stream_writer", return_value=lambda x: None),
    ):
        call_model = _build_call_model("claude-haiku-5-5", ["gpt-4o-mini"])
        result = await call_model({"messages": [HumanMessage(content="hi")], "org_id": "org"})

    assert result["messages"][0].content == "from backup"
    assert [c["model"] for c in calls] == ["claude-haiku-5-5", "gpt-4o-mini"]
    assert [c["num_retries"] for c in calls] == [1, 3]  # hand over fast; the last keeps retrying
    assert isinstance(calls[0]["messages"][0]["content"], list)  # Anthropic: cache blocks
    assert isinstance(calls[1]["messages"][0]["content"], str)  # OpenAI: plain string


async def test_call_model_raises_when_every_model_fails(database):
    from agent.graph import _build_call_model

    down = AsyncMock(side_effect=Exception("all down"))
    with (
        patch("agent.graph.litellm.acompletion", new=down),
        patch("agent.graph.get_stream_writer", return_value=lambda x: None),
    ):
        call_model = _build_call_model("gpt-4o-mini", ["xai/grok-3-mini"])
        with pytest.raises(Exception, match="all down"):
            await call_model({"messages": [HumanMessage(content="hi")], "org_id": "org"})
        down.reset_mock()
        with pytest.raises(Exception, match="all down"):  # one model: nothing to hand over to
            await _build_call_model("gpt-4o-mini")(
                {"messages": [HumanMessage(content="hi")], "org_id": "org"}
            )
    assert [c.kwargs["num_retries"] for c in down.call_args_list] == [3]


async def test_call_model_sends_endpoint_settings_for_an_openai_compatible_provider(database):
    from agent.graph import _build_call_model

    calls: list[dict] = []

    async def completion(**kwargs):
        calls.append(kwargs)
        return text_stream("ok")

    env = {
        "OPENCODE_API_KEY": "zen-key",
        "OPENCODE_BASE_URL": "",
        "OPENCODE_MODEL": "",
        "LLM_PROVIDER": "",
    }
    with (
        patch.dict("os.environ", env),
        patch("agent.graph.litellm.acompletion", new=completion),
        patch("agent.graph.get_stream_writer", return_value=lambda x: None),
    ):
        call_model = _build_call_model("openai/glm-5.3-flash")
        await call_model({"messages": [HumanMessage(content="hi")], "org_id": "org"})

    assert calls[0]["api_base"] == "https://opencode.ai/zen/v1"
    assert calls[0]["api_key"] == "zen-key"


async def test_call_model_retries_an_empty_reply_with_reasoning_turned_down(database):
    from agent.graph import _build_call_model

    calls: list[dict] = []

    async def completion(**kwargs):
        calls.append(kwargs)
        return text_stream("" if len(calls) == 1 else "warm rewrite")

    env = {
        "QWEN_API_KEY": "q",
        "QWEN_BASE_URL": "http://x/v1",
        "QWEN_MODEL": "",
        "LLM_PROVIDER": "",
        "LLM_MODEL": "",
    }
    with (
        patch.dict("os.environ", env),
        patch("agent.graph.litellm.acompletion", new=completion),
        patch("agent.graph.get_stream_writer", return_value=lambda x: None),
    ):
        result = await _build_call_model("hosted_vllm/qwen36-35b-a3b")(
            {"messages": [HumanMessage(content="Redo it.")], "org_id": "org"}
        )

    assert result["messages"][0].content == "warm rewrite"
    assert "extra_body" not in calls[0]  # first try keeps full reasoning
    assert calls[1]["extra_body"] == {"chat_template_kwargs": {"enable_thinking": False}}


async def test_call_model_shows_a_message_instead_of_a_blank_reply(database):
    from agent.graph import EMPTY_REPLY_MESSAGE, _build_call_model

    events: list[dict] = []

    async def completion(**kwargs):
        return text_stream("")

    with (
        patch("agent.graph.litellm.acompletion", new=completion),
        patch("agent.graph.get_stream_writer", return_value=events.append),
    ):
        result = await _build_call_model("gpt-4o-mini")(
            {"messages": [HumanMessage(content="hi")], "org_id": "org"}
        )

    assert result["messages"][0].content == EMPTY_REPLY_MESSAGE
    assert {"type": "text", "content": EMPTY_REPLY_MESSAGE} in events


# ── save_check: replies that show posts which were never saved ────────────────


def _reply_state(reply: str, *earlier) -> dict:
    """A finished turn: the user's message, any earlier messages, then the final reply."""
    return {
        "messages": [
            HumanMessage(content="Suggest some posts"),
            *earlier,
            AIMessage(content=reply),
        ],
        "org_id": "org",
    }


async def test_should_continue_routes_to_save_check_for_unsaved_post_ids(database):
    from agent.graph import should_continue

    state = _reply_state("**1. Volunteer spotlight** · post_001\n**2. Gala recap** · post_002")
    assert await should_continue(state) == "save_check"


async def test_unsaved_post_ids_skips_ids_in_the_ledger_and_sorts_numerically(database):
    from agent.graph import _unsaved_post_ids
    from agent.memory import add_post

    await add_post("org", {"platform": "linkedin", "content": "Saved idea"})  # post_001

    state = _reply_state("post_010, post_001 and post_002")
    assert await _unsaved_post_ids(state) == ["post_002", "post_010"]


async def test_should_continue_ends_when_every_shown_id_is_in_the_ledger(database):
    from agent.graph import should_continue
    from agent.memory import add_post

    await add_post("org", {"platform": "linkedin", "content": "Saved idea"})
    assert await should_continue(_reply_state("Here it is · post_001")) == END


async def test_should_continue_ends_for_ids_the_user_or_a_tool_mentioned(database):
    """Explaining "post_009 doesn't exist" is not showing an unsaved post."""
    from agent.graph import should_continue

    state = {
        "messages": [
            HumanMessage(content="Approve post_009"),
            AIMessage(
                content="", tool_calls=[{"id": "tc", "name": "update_post_status", "args": {}}]
            ),
            ToolMessage(
                content="Post post_007 not found in ledger",
                tool_call_id="tc",
                name="update_post_status",
            ),
            AIMessage(content="I can't find post_009 or post_007 in the ledger."),
        ],
        "org_id": "org",
    }
    assert await should_continue(state) == END


async def test_should_continue_runs_save_check_at_most_once_per_turn(database):
    from agent.graph import SAVE_CHECK_NAME, should_continue

    state = _reply_state(
        "Still post_001",
        AIMessage(content="**1. Idea** · post_001"),
        HumanMessage(content="Automatic check", name=SAVE_CHECK_NAME),
    )
    assert await should_continue(state) == END


async def test_save_check_node_asks_to_save_the_unsaved_posts(database):
    from agent.graph import SAVE_CHECK_NAME, _build_save_check

    events = []
    with patch("agent.graph.get_stream_writer", return_value=events.append):
        update = await _build_save_check()(_reply_state("post_002 and post_001"))

    [message] = update["messages"]
    assert isinstance(message, HumanMessage) and message.name == SAVE_CHECK_NAME
    assert "post_001, post_002" in message.content and "save_post" in message.content
    assert events[0]["type"] == "tool_step" and events[0]["tool"] == "save_check"
    assert events[1] == {"type": "text", "content": "\n\n"}


def test_turn_reply_joins_the_reply_and_the_save_check_answer():
    from agent.graph import SAVE_CHECK_NAME, turn_reply

    messages = [
        HumanMessage(content="old question"),
        AIMessage(content="old answer"),
        HumanMessage(content="Suggest posts"),
        AIMessage(
            content="Let me look.", tool_calls=[{"id": "t", "name": "research_org", "args": {}}]
        ),
        ToolMessage(content="research", tool_call_id="t", name="research_org"),
        AIMessage(content="1. Idea · post_001"),
        HumanMessage(content="Automatic check", name=SAVE_CHECK_NAME),
        AIMessage(content="Saved: Idea · post_004"),
    ]
    assert turn_reply(messages) == "1. Idea · post_001\n\nSaved: Idea · post_004"


@pytest.mark.parametrize(
    "messages, expected",
    [
        pytest.param(
            [AIMessage(content="first"), AIMessage(content="second")],
            "first\n\nsecond",
            id="no-user-turn-takes-every-reply",
        ),
        pytest.param(
            [HumanMessage(content="hi"), AIMessage(content="")], "", id="empty-final-reply"
        ),
        pytest.param(
            [HumanMessage(content="hi"), AIMessage(content="shown"), AIMessage(content="")],
            "shown",
            id="empty-reply-adds-no-blank-paragraph",
        ),
    ],
)
def test_turn_reply_edge_cases(messages, expected):
    from agent.graph import turn_reply

    assert turn_reply(messages) == expected


def test_summary_transcript_labels_the_save_check_message():
    from agent.graph import SAVE_CHECK_NAME, _render_transcript

    transcript = _render_transcript(
        [
            HumanMessage(content="hi"),
            HumanMessage(content="save them", name=SAVE_CHECK_NAME),
        ]
    )
    assert transcript == "USER: hi\nAUTOMATIC CHECK: save them"


CHECKPOINT_TABLES: tuple[str, ...] = ("checkpoints", "checkpoint_writes", "checkpoint_blobs")


def _checkpoint_rows(database, thread_id: str) -> dict[str, int]:
    """How many rows each checkpoint table holds for a thread (read past row-level security)."""
    return {
        table: database["cleaner"]
        .execute(f"SELECT count(*) FROM {table} WHERE thread_id = %s", (thread_id,))
        .fetchone()[0]
        for table in CHECKPOINT_TABLES
    }


def _rows_of_older_checkpoints(database, thread_id: str) -> int:
    """
    Writes and blobs of a thread that its newest checkpoint doesn't use.

    A blob is used when the newest checkpoint's channel_versions name its version.
    """
    return (
        database["cleaner"]
        .execute(
            """
            WITH newest AS (
                SELECT checkpoint_id, checkpoint FROM checkpoints
                WHERE thread_id = %(t)s AND checkpoint_ns = ''
                ORDER BY checkpoint_id DESC LIMIT 1
            )
            SELECT
                (SELECT count(*) FROM checkpoint_writes w WHERE w.thread_id = %(t)s
                 AND w.checkpoint_id NOT IN (SELECT checkpoint_id FROM newest))
              + (SELECT count(*) FROM checkpoint_blobs b WHERE b.thread_id = %(t)s
                 AND b.version IS DISTINCT FROM
                     (SELECT checkpoint -> 'channel_versions' ->> b.channel FROM newest))
            """,
            {"t": thread_id},
        )
        .fetchone()[0]
    )


async def _chat(assistant, message: str, reply: str) -> AsyncMock:
    """One turn of the assistant with a scripted model reply; returns the model mock."""
    model = AsyncMock(return_value=text_stream(reply))
    with patch("agent.graph.litellm.acompletion", new=model):
        await assistant.chat(message)
    return model


async def test_prune_history_keeps_one_checkpoint_with_the_whole_conversation(database):
    from agent.assistant import Assistant
    from agent.graph import build_graph

    assistant = Assistant("org")
    for n in range(3):
        await _chat(assistant, f"q{n}", f"a{n}")

    assert _checkpoint_rows(database, "org")["checkpoints"] == 1
    assert _rows_of_older_checkpoints(database, "org") == 0
    async with build_graph(assistant.model, "org") as graph:
        state = await graph.aget_state({"configurable": {"thread_id": "org"}})
    assert [m.content for m in state.values["messages"]] == ["q0", "a0", "q1", "a1", "q2", "a2"]

    model = await _chat(assistant, "q3", "a3")
    sent = [m["content"] for m in model.call_args.kwargs["messages"][1:]]
    assert sent == ["q0", "a0", "q1", "a1", "q2", "a2", "q3"]


async def test_prune_history_leaves_other_threads_alone(database):
    from agent.assistant import Assistant
    from agent.graph import build_graph, prune_history

    with patch("agent.assistant.prune_history", new=AsyncMock()):  # org:b keeps every checkpoint
        await _chat(Assistant("org", thread_id="org:b"), "hi", "x")
    unpruned = _checkpoint_rows(database, "org:b")
    assert unpruned["checkpoints"] > 1
    assert _rows_of_older_checkpoints(database, "org:b") > 0  # so the 0 above means pruned

    await _chat(Assistant("org", thread_id="org:a"), "hi", "x")
    async with build_graph("claude-haiku-4-5-20251001", "org") as graph:
        await prune_history(graph, "org:missing")  # an unknown thread is a no-op

    assert _checkpoint_rows(database, "org:a")["checkpoints"] == 1
    assert _checkpoint_rows(database, "org:b") == unpruned
    assert _checkpoint_rows(database, "org:missing") == dict.fromkeys(CHECKPOINT_TABLES, 0)


@pytest.mark.parametrize(
    "error",
    [
        errors.SerializationFailure,  # two prunes of one thread collided
        errors.DeadlockDetected,
        errors.QueryCanceled,  # statement timeout
        psycopg.OperationalError,  # the connection dropped
    ],
)
async def test_a_failed_prune_does_not_fail_the_turn(database, error):
    """The turn has finished when pruning runs; the reply is returned and the next turn prunes."""
    from agent.assistant import Assistant
    from agent.graph import OrgCheckpointer

    aprune = AsyncMock(side_effect=error("prune failed"))
    with patch.object(OrgCheckpointer, "aprune", new=aprune):
        model = AsyncMock(return_value=text_stream("hello"))
        with patch("agent.graph.litellm.acompletion", new=model):
            assert await Assistant("org").chat("hi") == "hello"
    aprune.assert_awaited_once_with(["org"])


async def test_a_turn_interrupted_after_its_tool_ran_leaves_a_working_thread(database):
    """
    The database went away after save_post ran but before its result was
    checkpointed: turn 1 fails, and turn 2 answers the lost call instead of
    every later request being rejected for an unanswered tool call.
    """
    from agent.assistant import Assistant
    from agent.graph import LOST_TOOL_RESULT, OrgCheckpointer
    from agent.graph import handle_tool_call as real_handle_tool_call
    from agent.memory import get_ledger

    database_gone = False

    def tool_call_is_stored() -> bool:
        # LangGraph saves the step that made the tool call in the background, so the outage
        # waits for that save; otherwise there is no unanswered call to answer (a race the
        # slower coverage run lost).
        row = (
            database["cleaner"]
            .execute(
                "SELECT count(*) FROM checkpoint_writes WHERE position('tc_lost'::bytea IN blob) > 0"
            )
            .fetchone()
        )
        return row[0] > 0

    async def tool_then_outage(name: str, args: dict, org_id: str) -> str:
        nonlocal database_gone
        result = await real_handle_tool_call(name, args, org_id)
        for _ in range(500):
            if tool_call_is_stored():
                break
            await asyncio.sleep(0.01)
        assert tool_call_is_stored(), "the step with the tool call was never saved"
        database_gone = True
        return result

    def failing_after_outage(write):
        async def wrapper(self, *args, **kwargs):
            if database_gone:
                raise psycopg.OperationalError("server closed the connection unexpectedly")
            return await write(self, *args, **kwargs)

        return wrapper

    save = json.dumps({"platform": "linkedin", "title": "Spotlight", "content": "Meet Ana"})
    with (
        patch("agent.graph.handle_tool_call", new=tool_then_outage),
        patch.object(OrgCheckpointer, "aput", failing_after_outage(OrgCheckpointer.aput)),
        patch.object(
            OrgCheckpointer, "aput_writes", failing_after_outage(OrgCheckpointer.aput_writes)
        ),
        patch(
            "agent.graph.litellm.acompletion",
            new=AsyncMock(return_value=tool_call_stream("tc_lost", "save_post", save)),
        ),
        pytest.raises(psycopg.OperationalError),
    ):
        await Assistant("org").chat("Write a spotlight post")

    second_turn = AsyncMock(return_value=text_stream("Back on track"))
    with patch("agent.graph.litellm.acompletion", new=second_turn):
        assert await Assistant("org").chat("Did that save?") == "Back on track"

    sent = second_turn.call_args.kwargs["messages"]
    calls = [tc["id"] for m in sent if m.get("tool_calls") for tc in m["tool_calls"]]
    answers = {m["tool_call_id"]: m["content"] for m in sent if m["role"] == "tool"}
    assert calls == ["tc_lost"]
    assert answers == {"tc_lost": LOST_TOOL_RESULT}
    assert sent[-1] == {"role": "user", "content": "Did that save?"}
    assert [post.get("title") for post in await get_ledger("org")] == ["Spotlight"]


async def test_a_message_with_a_lone_surrogate_is_answered_and_the_thread_keeps_working(database):
    """Half an emoji (a lone surrogate) can't be encoded as UTF-8; it must not break the checkpoint."""
    from agent.assistant import Assistant

    assistant = Assistant("org")
    replies = AsyncMock(side_effect=[text_stream("hello"), text_stream("still here")])
    with patch("agent.graph.litellm.acompletion", new=replies):
        assert await assistant.chat("hi \ud800") == "hello"
        assert await assistant.chat("and now?") == "still here"
    sent = [m["content"] for m in replies.call_args.kwargs["messages"][1:]]
    assert len(sent) == 3 and sent[0].startswith("hi ") and sent[1:] == ["hello", "and now?"]


async def test_streamed_turns_are_pruned_too(database):
    from agent.assistant import Assistant

    assistant = Assistant("org")
    for _ in range(2):
        with patch("agent.graph.litellm.acompletion", new=AsyncMock(return_value=text_stream("x"))):
            async for _event in assistant.achat_stream("hi"):
                pass
    assert _checkpoint_rows(database, "org")["checkpoints"] == 1


async def test_read_stream_skips_usage_only_chunks_and_tool_fragments_without_a_function():
    """Regression: a chunk with no choices, or a tool delta with function None, crashed the turn."""
    from agent.graph import _read_stream

    def chunk(content=None, tool_calls=None) -> MagicMock:
        c = MagicMock()
        c.choices = [MagicMock(delta=MagicMock(content=content, tool_calls=tool_calls))]
        return c

    named = MagicMock(index=0, id="tc1")
    named.function.name, named.function.arguments = "web_search", '{"query": "q"}'
    bare = MagicMock(index=0, id=None, function=None)
    usage_only = MagicMock(choices=[])

    async def stream():
        for c in (chunk("Hi"), chunk(tool_calls=[named]), chunk(tool_calls=[bare]), usage_only):
            yield c

    text, tool_calls = await _read_stream(stream(), lambda event: None)
    assert text == "Hi"
    assert tool_calls[0] == {
        "id": "tc1",
        "function": {"name": "web_search", "arguments": '{"query": "q"}'},
    }


def _tool_delta_chunk(index: int, tc_id: str | None, name: str | None, arguments: str) -> MagicMock:
    """One stream chunk carrying a fragment of the tool call at index (id and name come first)."""
    tc_delta = MagicMock(index=index, id=tc_id)
    tc_delta.function.name, tc_delta.function.arguments = name, arguments
    chunk = MagicMock()
    chunk.choices = [
        MagicMock(delta=MagicMock(content=None, tool_calls=[tc_delta]), finish_reason=None)
    ]
    return chunk


async def test_streamed_tool_call_fragments_are_joined_and_returned_in_index_order(database):
    """Providers split arguments over chunks and may interleave calls; index 0 still comes first."""
    from agent.graph import _build_call_model

    async def stream():
        yield _tool_delta_chunk(1, "tc_b", "search_ledger", '{"que')
        yield _tool_delta_chunk(0, "tc_a", "save_post", '{"platform"')
        yield _tool_delta_chunk(1, None, None, 'ry": "gala"}')
        yield _tool_delta_chunk(0, None, None, ': "linkedin"}')
        yield text_chunk(None, "tool_calls")

    with (
        patch("agent.graph.litellm.acompletion", new=AsyncMock(return_value=stream())),
        patch("agent.graph.get_stream_writer", return_value=lambda x: None),
    ):
        result = await _build_call_model("gpt-4o-mini")(
            {"messages": [HumanMessage(content="hi")], "org_id": "org"}
        )

    assert result["messages"][0].tool_calls == [
        {"id": "tc_a", "name": "save_post", "args": {"platform": "linkedin"}, "type": "tool_call"},
        {"id": "tc_b", "name": "search_ledger", "args": {"query": "gala"}, "type": "tool_call"},
    ]


async def test_a_tool_loop_that_overflows_the_window_is_summarized_before_the_next_model_call(
    database,
):
    """The graph runs tools → summarize → agent, so older turns are folded in mid-turn."""
    from agent.assistant import Assistant
    from agent.graph import MAX_MESSAGES, SUMMARY_BATCH_MESSAGES, _window_start, build_graph

    assistant = Assistant("org")
    config = {"configurable": {"thread_id": assistant.thread_id}}
    # With this turn's message, two short of a batch leave the window; its tool call and
    # result push out two more, so the batch is complete before the second model call.
    history = _long_history((MAX_MESSAGES + SUMMARY_BATCH_MESSAGES - 4) // 2)
    assert _window_start([*history, HumanMessage(content="q")]) == SUMMARY_BATCH_MESSAGES - 2
    async with build_graph(assistant.model, "org") as graph:
        await graph.aupdate_state(config, {"messages": history, "org_id": "org"})

    replies = [tool_call_stream("tc1", "search_ledger", "{}"), text_stream("done")]

    async def model(**kwargs):
        return replies.pop(0) if kwargs.get("stream") else completion("- zz-folded-in")

    mock = AsyncMock(side_effect=model)
    with patch("agent.graph.litellm.acompletion", new=mock):
        assert await assistant.chat("what's in the ledger?") == "done"

    calls = mock.call_args_list
    assert ["agent" if c.kwargs.get("stream") else "summary" for c in calls] == [
        "agent",
        "summary",
        "agent",
    ]
    assert "zz-folded-in" in json.dumps(calls[2].kwargs["messages"][0]["content"])


async def test_tool_step_previews_stay_short_however_large_the_tool_input(database):
    """A 50 KB email draft shouldn't fill the screen inside a collapsed step."""
    from agent.assistant import Assistant

    args = json.dumps({"platform": "general", "content": "x" * 50_000, "title": "Long"})
    model = AsyncMock(side_effect=[tool_call_stream("tc1", "save_post", args), text_stream("ok")])
    with patch("agent.graph.litellm.acompletion", new=model):
        steps = [e async for e in Assistant("org").achat_stream("save") if e["type"] == "tool_step"]
    assert len(steps[0]["input_preview"]) <= 300
    assert len(steps[0]["result_preview"]) <= 500
