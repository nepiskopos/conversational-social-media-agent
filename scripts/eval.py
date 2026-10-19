"""
LLM-as-judge evaluator for the end-to-end 7-step acceptance scenario.

Owns: driving the 7-step scenario against a real org (it saves to that org's
profile under data/, like a normal chat), scoring each response on three
rubrics in parallel (21 total judge calls), checking what the run saved to
profile.json (no LLM involved), and printing a markdown report. Does not own
the assistant logic — it imports and drives it.

Usage (run from project root):
    python scripts/eval.py
    python scripts/eval.py --org "Myra's Kids" --url https://myraskids.org

Requires at least one configured provider (see agent/config.py and
.env.example) and TAVILY_API_KEY for the research step.
The judge model is JUDGE_MODEL (and JUDGE_PROVIDER) if set (recommended: a
different provider, or at least another model family, than the assistant, to
avoid self-grading bias), otherwise the assistant's own model.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
import time
from pathlib import Path

import litellm
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent.parent))
load_dotenv(Path(__file__).parent.parent / ".env")

from agent.config import (
    background_kwargs,
    resolve_judge_model,
    resolve_model,
)  # needs the path and .env above

# A judge from a different model family avoids the assistant grading its own work.
JUDGE_MODEL: str = resolve_judge_model()

SCENARIO: list[str] = [
    "Suggest some posts for us.",
    "Let's go with the second one, write it for LinkedIn.",
    "Too corporate, we're more warm and grassroots. Redo it.",
    "Now give me an Instagram version.",
    "What programs does our org actually run?",
    "Which posts have we worked on so far, and what's the status of each?",
    "Suggest another post for next month.",
]

RUBRICS: dict[str, str] = {
    "source_attribution": (
        "Does this response cite at least one specific source that informed its content? "
        "MANDATORY EXEMPTIONS — output score=3 immediately with no evaluation for: "
        "step 2 (writing a platform draft of an already-sourced suggestion), "
        "step 3 (voice rewrite of an existing draft), "
        "step 4 (platform adaptation of an existing draft), "
        "step 6 (ledger status query — answered from conversation history, not external research). "
        "For steps 1, 5, and 7 only: "
        "0 = no source mentioned; "
        "1 = vague reference only ('my research', 'your website') with no URL; "
        "2 = source named but URL missing, or URL present but no descriptive label; "
        "3 = at least one complete citation with BOTH a descriptive label AND a URL."
    ),
    "voice_consistency": (
        "In a 7-step scenario, the user corrected the tone to 'warm and grassroots' in step 3. "
        "MANDATORY RULE: if this is step 1 or step 2, you MUST output score=3 with no evaluation — "
        "the tone correction has not happened yet. If this is step 5 or step 6, you MUST output score=3 "
        "with no evaluation — those are factual answers, not post copy. "
        "For steps 3, 4 and 7 only, judge the post copy: "
        "0 = formal or corporate; 1 = neutral; 2 = somewhat warm; 3 = clearly warm and grassroots."
    ),
    "ledger_integrity": (
        "For step 7 only: does this response avoid repeating the specific post ideas in the STEP 1 reply shown? "
        "0 = same ideas recycled; 1 = mostly same; 2 = mostly different; 3 = all new ideas. "
        "For every step except step 7: score 3 (not applicable)."
    ),
}

# A ceiling, not a target: a reasoning-model judge thinks before it answers. 120 tokens left
# Qwen with no answer at all. With DeepSeek V4 Pro, 60 calls on real replies used up to 1,271
# tokens (median 143); at a 1,000 cap 3 of them came back empty and 1 was cut off.
JUDGE_MAX_TOKENS: int = 4000
# An empty judge reply is asked once more before it is scored as a parse error.
JUDGE_ATTEMPTS: int = 2

# A rubric score of MIN_PASS_SCORE or above counts as passing for invariant checks.
MIN_PASS_SCORE: int = 2
# An overall score of PASS_PERCENT or above is reported as PASS.
PASS_PERCENT: int = 70


async def _ask_judge(prompt: str) -> str:
    """The judge's reply text; an empty reply is asked again, up to JUDGE_ATTEMPTS times."""
    text = ""
    for _ in range(JUDGE_ATTEMPTS):
        response = await litellm.acompletion(
            model=JUDGE_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=JUDGE_MAX_TOKENS,
            temperature=0,
            **background_kwargs(JUDGE_MODEL),
        )
        text = (response.choices[0].message.content or "").strip()
        if text:
            break
    return text


async def _score_one(
    step: int,
    user_msg: str,
    reply: str,
    rubric_name: str,
    rubric_desc: str,
    earlier_reply: str = "",
) -> dict:
    """
    Call the judge LLM for one (step, rubric) pair and return {rubric, score, reason}.

    earlier_reply is step 1's reply, given only where the rubric compares
    against it; otherwise the judge would be guessing what step 1 suggested.
    """
    earlier = f'STEP 1 — Assistant replied: "{earlier_reply[:1500]}"\n\n' if earlier_reply else ""
    prompt = (
        f"You are a strict evaluator. Score the assistant response on the rubric below.\n\n"
        f"RUBRIC ({rubric_name}): {rubric_desc}\n\n"
        f"{earlier}"
        f'STEP {step} — User said: "{user_msg}"\n'
        f'Assistant replied: "{reply[:1500]}"\n\n'
        "Reply with ONLY valid JSON — no markdown fences, no extra text: "
        '{"score": <integer 0-3>, "reason": "<one sentence>"}'
    )
    try:
        text = await _ask_judge(prompt)
    except Exception as exc:  # one failed judge call must not lose the other 20 scores
        return {
            "rubric": rubric_name,
            "score": -1,
            "reason": f"judge error: {type(exc).__name__}: {exc}"[:120],
        }
    # Strip markdown fences if the model wrapped its JSON (e.g. ```json...```)
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text.strip())
    try:
        data = json.loads(text)
        return {"rubric": rubric_name, "score": int(data["score"]), "reason": data["reason"]}
    except Exception:
        # Last-resort: extract the first {...} block
        match = re.search(r"\{[^{}]+\}", text, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group(0))
                return {
                    "rubric": rubric_name,
                    "score": int(data["score"]),
                    "reason": data["reason"],
                }
            except Exception:
                pass
        return {"rubric": rubric_name, "score": -1, "reason": f"parse error: {text[:80]}"}


async def score_all(responses: list[dict]) -> list[list[dict]]:
    """
    Run all 21 judge calls (7 steps × 3 rubrics) concurrently via asyncio.gather.

    Args:
        responses: List of dicts with keys step, user, reply.

    Returns:
        List of 7 items, each a list of 3 rubric-score dicts.
    """
    step1_reply = responses[0]["reply"]
    tasks = [
        _score_one(
            item["step"],
            item["user"],
            item["reply"],
            name,
            desc,
            earlier_reply=step1_reply if name == "ledger_integrity" and item["step"] == 7 else "",
        )
        for item in responses
        for name, desc in RUBRICS.items()
    ]
    flat = await asyncio.gather(*tasks)
    n = len(RUBRICS)
    return [list(flat[i * n : (i + 1) * n]) for i in range(len(responses))]


def run_scenario(org_name: str, org_url: str) -> tuple[list[dict], list[dict]]:
    """
    Drive the Assistant through the 7-step acceptance scenario.

    Records the org's name and website before the first turn, as onboarding
    does in app.py, so the system prompt shows the right org and the
    assistant can call research_org as soon as it is asked for suggestions.

    Args:
        org_name: Test org name (used as org_id after slugification).
        org_url:  Optional website URL. Stored in the profile so the
                  assistant can scrape it during the research step.

    Returns:
        (responses, snapshots). responses: 7 dicts with keys step, user, reply.
        snapshots: 8 dicts with keys ledger and preferences — index 0 is the
        profile before the scenario, index i the profile after step i.
    """
    from agent.assistant import Assistant
    from agent.memory import fill_org_details, load_profile, slugify

    org_id = slugify(org_name)
    fill_org_details(org_id, org_name, org_url)

    def _snapshot() -> dict:
        saved = load_profile(org_id)
        return {"ledger": saved["post_ledger"], "preferences": saved["preferences"]}

    async def _run() -> tuple[list[dict], list[dict]]:
        # Use a unique thread_id so each eval run starts with a clean
        # conversation history — no contamination from prior runs stored
        # in data/checkpoints.db under the same org_id thread.
        thread_id = f"{org_id}:eval-{int(time.time())}"
        assistant = Assistant(org_id, thread_id=thread_id)
        responses: list[dict] = []
        snapshots = [_snapshot()]
        for i, msg in enumerate(SCENARIO, start=1):
            print(f"  [{i}/7] {msg}")
            reply = await assistant.chat(msg)
            print(f"        → {reply[:100].replace(chr(10), ' ')}...\n")
            responses.append({"step": i, "user": msg, "reply": reply})
            snapshots.append(_snapshot())
        return responses, snapshots

    return asyncio.run(_run())


def _check_invariants(scores: list[list[dict]]) -> dict[str, bool]:
    """
    Derive the three scenario invariant results from the rubric scores.

    Returns a dict mapping invariant name → pass/fail bool.
    """
    # 0-based indices 3 and 6 = steps 4 and 7, the posts written after the step-3 voice correction
    return {
        "source_attribution_step1": scores[0][0]["score"] >= MIN_PASS_SCORE,
        "voice_consistency_steps4_7": all(scores[i][1]["score"] >= MIN_PASS_SCORE for i in (3, 6)),
        "ledger_integrity_step7": scores[6][2]["score"] >= MIN_PASS_SCORE,
    }


def _posts_added(snapshots: list[dict], step: int) -> list[dict]:
    """Ledger entries that appeared during the given step (1-based)."""
    before = {post.get("id") for post in snapshots[step - 1]["ledger"]}
    return [post for post in snapshots[step]["ledger"] if post.get("id") not in before]


def check_saved_state(snapshots: list[dict]) -> dict[str, tuple[bool, str]]:
    """
    Check the scenario invariants against what was saved, not what was said.

    The judge reads replies only, and in a live Qwen run it contradicted
    itself on step 7 while the voice correction was never saved. These checks
    read the profile snapshots from run_scenario, so their verdicts are exact.

    Args:
        snapshots: Profile snapshots from run_scenario (before, then after each step).

    Returns:
        Check name → (passed, detail for the report).
    """
    step1, step7 = _posts_added(snapshots, 1), _posts_added(snapshots, 7)

    before, after = snapshots[0]["preferences"], snapshots[3]["preferences"]
    learned = sorted(key for key, value in after.items() if before.get(key) != value)

    def title(post: dict) -> str:
        return str(post.get("title", "")).strip().lower()

    step1_ids = {post["id"] for post in step1}
    # Untitled posts are not repeats of each other.
    step1_titles = {title(post) for post in step1} - {""}
    repeats = [
        post["id"]
        for post in step7
        if post.get("parent_id") in step1_ids or title(post) in step1_titles
    ]

    unsourced = [post["id"] for post in step1 + step7 if not post.get("source_url")]

    return {
        "preference_saved_by_step3": (
            bool(learned),
            f"learned: {', '.join(learned)}" if learned else "no preference saved",
        ),
        "step7_posts_new": (
            bool(step7) and not repeats,
            "no post saved in step 7"
            if not step7
            else f"repeats step 1: {', '.join(repeats)}"
            if repeats
            else f"{len(step7)} new post(s), none repeat step 1's {len(step1)}",
        ),
        "suggestions_have_sources": (
            bool(step1) and not unsourced,
            "no post saved in step 1"
            if not step1
            else f"no source_url: {', '.join(unsourced)}"
            if unsourced
            else f"all {len(step1) + len(step7)} suggestions from steps 1 and 7 have a source_url",
        ),
    }


def print_report(
    responses: list[dict],
    scores: list[list[dict]],
    saved_checks: dict[str, tuple[bool, str]],
) -> None:
    """Print the full evaluation report as markdown to stdout."""
    lines: list[str] = [
        "# Eval Report\n",
        f"Assistant: `{resolve_model()}` · Judge: `{JUDGE_MODEL}`\n",
    ]

    total, max_total = 0, 0
    for item, step_scores in zip(responses, scores, strict=True):
        step_sum = sum(s["score"] for s in step_scores if s["score"] >= 0)
        total += step_sum
        max_total += len(step_scores) * 3
        lines.append(f"## Step {item['step']}: {item['user']}")
        lines.append(f"**Score: {step_sum}/{len(step_scores) * 3}**\n")
        for s in step_scores:
            mark = "pass" if s["score"] >= MIN_PASS_SCORE else "FAIL"
            lines.append(f"- [{mark}] **{s['rubric']}**: {s['score']}/3 — {s['reason']}")
        lines.append("")

    pct = round(total / max_total * 100) if max_total else 0
    verdict = "PASS" if pct >= PASS_PERCENT else "NEEDS WORK"
    lines.append(f"## Overall: {total}/{max_total} ({pct}%) — {verdict}\n")

    lines.append("## Invariant Checks\n")
    invariants = _check_invariants(scores)
    labels = {
        "source_attribution_step1": "Source attribution present in step 1",
        "voice_consistency_steps4_7": "Warm/grassroots voice carried into the posts in steps 4 and 7",
        "ledger_integrity_step7": "Step 7 introduces new content (not step 1 repeats)",
    }
    for key, label in labels.items():
        result = "PASS" if invariants[key] else "FAIL"
        lines.append(f"- **{result}** — {label}")

    lines.append("\n## Saved-State Checks (profile.json, no judge)\n")
    saved_labels = {
        "preference_saved_by_step3": "The voice correction was saved as a preference by step 3",
        "step7_posts_new": "Step 7 saved new posts, not versions or copies of step 1's",
        "suggestions_have_sources": "Every suggestion saved in steps 1 and 7 has a source_url",
    }
    for key, label in saved_labels.items():
        passed, detail = saved_checks[key]
        lines.append(f"- **{'PASS' if passed else 'FAIL'}** — {label} ({detail})")

    print("\n".join(lines))


def main() -> None:
    """Parse CLI args, run the scenario, score, and print the report."""
    parser = argparse.ArgumentParser(
        description="Evaluate Mackie on the 7-step acceptance scenario."
    )
    parser.add_argument("--org", default="BRCA Strong", help="Test org name (default: BRCA Strong)")
    parser.add_argument("--url", default="", help="Org website URL (optional)")
    args = parser.parse_args()

    print(f"Running 7-step scenario for: {args.org}\n")
    responses, snapshots = run_scenario(args.org, args.url)

    print(f"Scoring {len(responses) * len(RUBRICS)} rubric calls in parallel...")
    scores = asyncio.run(score_all(responses))

    print_report(responses, scores, check_saved_state(snapshots))


if __name__ == "__main__":
    main()
