"""
Unit tests for scripts/eval.py: running the scenario (a fresh thread, a memory
snapshot per step), the saved-state checks, the voice invariant, judge scoring
(which replies the judge sees, retries, failed and unparseable calls), the
memory check (a date pushed out of the window, kept in the summary's open
items and recalled) and the report.
The judge and the assistant are mocked, so no LLM or network is used.
"""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tests.conftest import load_script


@pytest.fixture(scope="module")
def eval_module():
    return load_script("eval")


def _post(post_id: str, title: str, source_url: str = "https://org.example/about", **extra) -> dict:
    """A ledger post with the fields the saved-state checks read."""
    return {"id": post_id, "title": title, "source_url": source_url, **extra}


def _snapshots(step1: list[dict], step7: list[dict], prefs_after_step3: dict) -> list[dict]:
    """Before + 7 steps: step 1 adds step1 posts, step 3 saves prefs, step 7 adds step7 posts."""
    snaps = []
    for step in range(8):
        ledger = (step1 if step >= 1 else []) + (step7 if step == 7 else [])
        prefs = prefs_after_step3 if step >= 3 else {}
        snaps.append({"ledger": ledger, "preferences": prefs})
    return snaps


def test_check_saved_state_passes_a_good_run(eval_module):
    snaps = _snapshots(
        [_post("post_001", "Volunteer spotlight"), _post("post_002", "Walk recap")],
        [_post("post_005", "Survivor story month")],
        {"brand_voice": "warm and grassroots"},
    )
    checks = eval_module.check_saved_state(snaps)
    assert all(passed for passed, _ in checks.values())
    assert "brand_voice" in checks["preference_saved_by_step3"][1]


@pytest.mark.parametrize(
    "before, after_step3",
    [({}, {}), ({"tone": "warm"}, {"tone": "warm"})],
    ids=["nothing saved", "saved before the run"],
)
def test_check_saved_state_fails_when_nothing_was_learned(eval_module, before, after_step3):
    snaps = _snapshots([_post("post_001", "A")], [_post("post_002", "B")], after_step3)
    snaps[0]["preferences"] = before
    passed, detail = eval_module.check_saved_state(snaps)["preference_saved_by_step3"]
    assert not passed and detail == "no preference saved"


def test_check_saved_state_flags_step7_repeats_by_title_or_parent(eval_module):
    snaps = _snapshots(
        [_post("post_001", "Volunteer Spotlight"), _post("post_002", "Walk recap")],
        [
            _post("post_006", " volunteer spotlight "),
            _post("post_007", "New angle", parent_id="post_002"),
        ],
        {"tone": "warm"},
    )
    passed, detail = eval_module.check_saved_state(snaps)["step7_posts_new"]
    assert not passed
    assert "post_006" in detail and "post_007" in detail


def test_check_saved_state_does_not_treat_untitled_posts_as_repeats(eval_module):
    snaps = _snapshots([_post("post_001", "")], [_post("post_006", "")], {"tone": "warm"})
    assert eval_module.check_saved_state(snaps)["step7_posts_new"][0]


def test_check_saved_state_fails_when_step7_saved_nothing(eval_module):
    snaps = _snapshots([_post("post_001", "A")], [], {"tone": "warm"})
    assert eval_module.check_saved_state(snaps)["step7_posts_new"] == (
        False,
        "no post saved in step 7",
    )


@pytest.mark.parametrize(
    "step1, detail",
    [
        ([_post("post_001", "A", source_url="")], "no source_url: post_001"),
        ([], "no post saved in step 1"),
    ],
    ids=["unsourced", "step 1 saved nothing"],
)
def test_check_saved_state_flags_suggestions_without_source(eval_module, step1, detail):
    snaps = _snapshots(step1, [_post("post_002", "B")], {"tone": "warm"})
    assert eval_module.check_saved_state(snaps)["suggestions_have_sources"] == (False, detail)


def _scores(eval_module, score: int, judge_errors: int = 0) -> list[list[dict]]:
    """Every rubric of every step scored score, except the first judge_errors calls failed (-1)."""
    per_step = len(eval_module.RUBRICS)
    flat = [-1] * judge_errors + [score] * (7 * per_step - judge_errors)
    return [
        [
            {"rubric": name, "score": flat[step * per_step + i], "reason": "ok"}
            for i, name in enumerate(eval_module.RUBRICS)
        ]
        for step in range(7)
    ]


@pytest.mark.parametrize(
    "score, judge_errors, overall, step1",
    [
        (3, 0, "63/63 (100%) — PASS", "**Score: 9/9**"),
        (2, 0, "42/63 (67%) — NEEDS WORK", "**Score: 6/9**"),
        # A failed judge call scores -1: left out of the sum rather than subtracted from it.
        (3, 1, "60/63 (95%) — PASS", "**Score: 6/9**"),
    ],
)
def test_print_report_scores_names_models_and_shows_saved_state(
    eval_module, capsys, score, judge_errors, overall, step1
):
    from agent.config import resolve_model

    assert len(eval_module.RUBRICS) == 3  # the expected totals below assume three rubrics
    responses = [{"step": i, "user": "u", "reply": "r"} for i in range(1, 8)]
    checks = {
        "preference_saved_by_step3": (False, "no preference saved"),
        "step7_posts_new": (True, "1 new post(s)"),
        "suggestions_have_sources": (True, "all sourced"),
    }
    with patch.object(eval_module, "JUDGE_MODEL", "openai/deepseek-v4-pro"):
        eval_module.print_report(responses, _scores(eval_module, score, judge_errors), checks)
    out = capsys.readouterr().out
    assert f"Assistant: `{resolve_model()}` · Judge: `openai/deepseek-v4-pro`" in out
    assert f"## Overall: {overall}" in out
    assert out.split("## Step 1: u\n", 1)[1].startswith(step1)
    assert "## Saved-State Checks" in out
    assert (
        "**FAIL** — The voice correction was saved as a preference by step 3 (no preference saved)"
        in out
    )


async def test_score_all_shows_the_judge_step1_only_for_the_step7_repeat_rubric(eval_module):
    responses = [{"step": i, "user": f"msg {i}", "reply": f"reply-{i}"} for i in range(1, 8)]
    verdict = MagicMock(
        choices=[MagicMock(message=MagicMock(content=json.dumps({"score": 3, "reason": "ok"})))]
    )
    judge = AsyncMock(return_value=verdict)
    with patch.object(eval_module.litellm, "acompletion", judge):
        scores = await eval_module.score_all(responses)

    assert len(scores) == 7 and all(len(step) == len(eval_module.RUBRICS) for step in scores)
    prompts = [call.kwargs["messages"][0]["content"] for call in judge.call_args_list]
    with_step1 = [p for p in prompts if "STEP 1 — Assistant replied" in p]
    assert len(with_step1) == 1
    assert (
        "ledger_integrity" in with_step1[0]
        and "reply-1" in with_step1[0]
        and "reply-7" in with_step1[0]
    )


async def test_score_all_scores_a_failed_judge_call_as_minus_one(eval_module):
    responses = [{"step": i, "user": "m", "reply": "r"} for i in range(1, 8)]
    with patch.object(
        eval_module.litellm, "acompletion", AsyncMock(side_effect=RuntimeError("down"))
    ):
        scores = await eval_module.score_all(responses)
    assert all(s["score"] == -1 and "judge error" in s["reason"] for step in scores for s in step)


def test_voice_invariant_checks_the_posts_in_steps_4_and_7_only(eval_module):
    def scores(voice_by_step: dict[int, int]) -> list[list[dict]]:
        return [
            [{"score": 3}, {"score": voice_by_step.get(step, 3)}, {"score": 3}]
            for step in range(1, 8)
        ]

    assert eval_module._check_invariants(scores({5: 0, 6: 0}))["voice_consistency_steps4_7"]
    assert not eval_module._check_invariants(scores({4: 1}))["voice_consistency_steps4_7"]
    assert not eval_module._check_invariants(scores({7: 1}))["voice_consistency_steps4_7"]


@pytest.mark.parametrize(
    "reply, score",
    [
        ('{"score": 2, "reason": "fine"}', 2),
        ('```json\n{"score": 3, "reason": "fenced"}\n```', 3),
        ('Here you go: {"score": 1, "reason": "prose around it"} — hope that helps', 1),
        ('{"score": "high", "reason": "not a number"}', -1),
        ("I can't score this.", -1),
        ("", -1),
    ],
)
async def test_score_one_reads_the_judge_reply_shapes(eval_module, reply, score):
    verdict = MagicMock(choices=[MagicMock(message=MagicMock(content=reply))])
    with patch.object(eval_module.litellm, "acompletion", AsyncMock(return_value=verdict)):
        result = await eval_module._score_one(1, "msg", "reply", "voice", "desc")
    assert result["rubric"] == "voice" and result["score"] == score


async def test_run_scenario_snapshots_memory_after_every_step(eval_module, database):
    """run_scenario runs its own event loop, so it runs in a thread here; it closes its pool."""
    from agent import db
    from agent.memory import add_post, load_profile

    assistant = MagicMock()
    turns = iter(range(1, 8))

    async def chat(message: str) -> str:
        step = next(turns)
        await add_post(
            "eval-org", {"platform": "general", "content": message, "title": f"Step {step}"}
        )
        return f"reply {step}"

    assistant.chat.side_effect = chat
    close_pool = AsyncMock(wraps=db.close_pool)
    with (
        patch("agent.assistant.Assistant", return_value=assistant) as assistant_class,
        patch.object(db, "close_pool", close_pool),
    ):
        responses, snapshots = await asyncio.to_thread(
            eval_module.run_scenario, "Eval Org", "https://eval.example"
        )

    assert [r["reply"] for r in responses] == [f"reply {n}" for n in range(1, 8)]
    assert [len(s["ledger"]) for s in snapshots] == list(range(8))
    assert ":eval-" in assistant_class.call_args.kwargs["thread_id"]  # fresh history per run
    assert (await load_profile("eval-org"))["website"] == "https://eval.example"
    close_pool.assert_awaited_once()


async def test_an_empty_judge_reply_is_asked_again(eval_module):
    """A reasoning judge can spend its tokens thinking and answer nothing; ask once more."""
    empty = MagicMock(choices=[MagicMock(message=MagicMock(content=""))])
    verdict = MagicMock(
        choices=[MagicMock(message=MagicMock(content='{"score": 3, "reason": "fine"}'))]
    )
    judge = AsyncMock(side_effect=[empty, verdict])
    with patch.object(eval_module.litellm, "acompletion", judge):
        result = await eval_module._score_one(1, "msg", "reply", "voice", "desc")
    assert result["score"] == 3 and judge.await_count == 2


async def test_a_judge_that_stays_empty_is_scored_as_a_parse_error(eval_module):
    empty = MagicMock(choices=[MagicMock(message=MagicMock(content=None))])
    judge = AsyncMock(return_value=empty)
    with patch.object(eval_module.litellm, "acompletion", judge):
        result = await eval_module._score_one(1, "msg", "reply", "voice", "desc")
    assert result["score"] == -1 and result["reason"].startswith("parse error")
    assert judge.await_count == eval_module.JUDGE_ATTEMPTS


def test_the_memory_history_pushes_the_date_out_of_the_window_by_a_summary_batch(eval_module):
    """Otherwise the live check would never make the summarizer handle the date."""
    from langchain_core.messages import HumanMessage

    from agent.graph import SUMMARY_BATCH_MESSAGES, _window_start

    history = [*eval_module.memory_history(), HumanMessage(content=eval_module.MEMORY_QUESTION)]
    assert eval_module.MEMORY_DATE in history[0].content
    assert _window_start(history) >= SUMMARY_BATCH_MESSAGES


async def test_run_memory_check_summarizes_the_old_turns_and_asks_about_the_date(
    eval_module, database
):
    """The filler is stored, not chatted: one summary call and one answer are the only model calls."""
    from tests.llm_fakes import completion, text_stream

    summary = (
        "Open requests and dates:\n- Spring Gala on May 3: remind a week before\nSummary:\n- x"
    )
    calls: list[dict] = []

    async def model(**kwargs):
        calls.append(kwargs)
        return (
            text_stream("Your Spring Gala is on May 3.")
            if kwargs.get("stream")
            else completion(summary)
        )

    from agent.assistant import Assistant

    with (
        patch("agent.graph.litellm.acompletion", new=AsyncMock(side_effect=model)),
        patch("agent.assistant.Assistant", wraps=Assistant) as assistant_class,
    ):
        result = await asyncio.to_thread(eval_module.run_memory_check, "Eval Org")

    # A fresh thread, so no conversation a user has with the org is read or changed.
    assert assistant_class.call_args.kwargs["thread_id"].startswith("eval-org:eval-memory-")

    summary_calls = [call for call in calls if not call.get("stream")]
    assert len(summary_calls) == 1 and len(calls) == 2
    assert eval_module.MEMORY_SEED in summary_calls[0]["messages"][0]["content"]
    assert result == {"summary": summary, "reply": "Your Spring Gala is on May 3."}


def test_the_memory_check_reads_the_section_the_summarizer_is_told_to_write(eval_module):
    """A renamed section would otherwise fail the live check for every summary."""
    from agent.graph import SUMMARY_INSTRUCTIONS

    assert eval_module.OPEN_ITEMS_HEADING in SUMMARY_INSTRUCTIONS
    assert "Summary:" in SUMMARY_INSTRUCTIONS.split(eval_module.OPEN_ITEMS_HEADING, 1)[1]


MISSING = "missing from the open requests and dates"


@pytest.mark.parametrize(
    ("summary", "reply", "kept", "recalled"),
    [
        (
            "Open requests and dates:\n- Gala May 3\nSummary:\n- x",
            "It's on May 3.",
            (True, "in the open requests and dates"),
            True,
        ),
        (
            "Open requests and dates:\n- none\nSummary:\n- gala on May 3",
            "On May 3.",
            (False, MISSING),
            True,
        ),
        ("", "I don't know.", (False, "no summary was written"), False),
        ("Summary only, no sections: May 3", "No idea.", (False, MISSING), False),
    ],
    ids=["kept-and-recalled", "only-in-the-summary-part", "no-summary", "no-sections"],
)
def test_check_memory_needs_the_date_among_the_open_items_and_in_the_reply(
    eval_module, summary, reply, kept, recalled
):
    checks = eval_module.check_memory({"summary": summary, "reply": reply})
    assert checks["date_kept_as_open_item"] == kept
    assert checks["date_recalled"] == (recalled, f"reply: {reply[:120]!r}")


def test_the_report_shows_the_memory_checks_only_when_they_ran(eval_module, capsys):
    responses = [{"step": n, "user": "u", "reply": "r"} for n in range(1, 8)]
    scores = [
        [{"rubric": rubric, "score": 3, "reason": "ok"} for rubric in eval_module.RUBRICS]
        for _ in responses
    ]
    saved = {
        key: (True, "ok")
        for key in ("preference_saved_by_step3", "step7_posts_new", "suggestions_have_sources")
    }
    eval_module.print_report(responses, scores, saved)
    assert "Memory Checks" not in capsys.readouterr().out
    memory = {"date_kept_as_open_item": (True, "kept"), "date_recalled": (False, "reply: 'no'")}
    eval_module.print_report(responses, scores, saved, memory)
    out = capsys.readouterr().out
    assert "**PASS** — The summary keeps May 3 as an open date (kept)" in out
    assert "**FAIL** — Asked much later, Mackie recalls May 3 (reply: 'no')" in out
