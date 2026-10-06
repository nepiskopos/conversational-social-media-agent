"""Unit tests for scripts/eval.py — the saved-state checks (no LLM, no network)."""

import importlib.util
import json
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

EVAL_PATH = Path(__file__).parent.parent / "scripts" / "eval.py"


@pytest.fixture(scope="module")
def eval_module():
    """Import scripts/eval.py without loading the real .env into the test process."""
    spec = importlib.util.spec_from_file_location("mackie_eval", EVAL_PATH)
    module = importlib.util.module_from_spec(spec)
    with patch("dotenv.load_dotenv"):
        spec.loader.exec_module(module)
    sys.modules.pop("mackie_eval", None)
    return module


def _post(post_id: str, title: str, source_url: str = "https://org.example/about", **extra) -> dict:
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


def test_check_saved_state_fails_when_nothing_was_learned(eval_module):
    snaps = _snapshots([_post("post_001", "A")], [_post("post_002", "B")], {})
    passed, detail = eval_module.check_saved_state(snaps)["preference_saved_by_step3"]
    assert not passed and detail == "no preference saved"


def test_check_saved_state_ignores_preferences_that_existed_before(eval_module):
    snaps = _snapshots([_post("post_001", "A")], [_post("post_002", "B")], {"tone": "warm"})
    snaps[0]["preferences"] = {"tone": "warm"}
    assert not eval_module.check_saved_state(snaps)["preference_saved_by_step3"][0]


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


def test_check_saved_state_flags_suggestions_without_source(eval_module):
    snaps = _snapshots(
        [_post("post_001", "A", source_url="")], [_post("post_002", "B")], {"tone": "warm"}
    )
    passed, detail = eval_module.check_saved_state(snaps)["suggestions_have_sources"]
    assert not passed and "post_001" in detail


def test_print_report_includes_saved_state_section(eval_module, capsys):
    responses = [{"step": i, "user": "u", "reply": "r"} for i in range(1, 8)]
    scores = [
        [{"rubric": name, "score": 3, "reason": "ok"} for name in eval_module.RUBRICS]
        for _ in range(7)
    ]
    checks = {
        "preference_saved_by_step3": (False, "no preference saved"),
        "step7_posts_new": (True, "1 new post(s)"),
        "suggestions_have_sources": (True, "all sourced"),
    }
    eval_module.print_report(responses, scores, checks)
    out = capsys.readouterr().out
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
