# Mackie — AI Social Media Manager for Nonprofits

Mackie is a conversational social media manager for nonprofit marketing teams. It keeps your social media active and on-brand so your staff can spend their time on the mission:

- **Plan:** monthly content calendars built around your campaigns, events and the awareness days that matter to your cause.
- **Create:** posts for LinkedIn, Instagram and Facebook, plus supporter email, in your own voice, each with an idea for the visual.
- **Optimize:** tell Mackie how posts performed and it suggests what to post next based on what worked.

Like a teammate, Mackie doesn't wait to be asked: each session opens with what needs you (drafts to approve, posts to publish, planned posts due this week, results to report). Nothing goes live without your approval. It researches your organization first, learns your voice from feedback, and keeps a ledger of every post and its status. Your team still publishes the posts and makes the final graphics.

## Features

- **Research before writing.** Reads your homepage and up to three About/Programs-style pages, runs three web searches in parallel, and keeps a short cited org brief for every session. Every suggestion names its source.
- **Platform-aware drafts.** Each platform's tone, length and hashtag rules live in a YAML file. Length, hashtag and visual rules are enforced in code on finished drafts, and checked again before a post is approved or marked posted; rough ideas are exempt. Email (subject, preview text, one call to action) is a platform too. Instagram and Facebook drafts must describe the photo or graphic to pair with them.
- **Content types, formats and pillars.** Donation appeals, volunteer calls and event announcements follow a defined structure. Each post format plays its own role, and Mackie picks the one that fits: feed posts are the lasting record, carousels teach step by step, stories keep followers close, and reels reach new people. Story frames, carousel slides and reel shot lists are written out in full. Five editorial pillars frame suggestions. All three are editable YAML.
- **Learns from feedback.** Corrections are saved in your own words. Approved and pasted posts become voice examples. Ask "what have you learned about us?" or "forget that rule" at any time. All of it carries over to future sessions.
- **Post ledger.** Every suggestion and draft is tracked with a title, status (`suggested`, `draft`, `approved`, `posted`, `planned`, `rejected`), versions and dates, so "what did we work on this week?" has an answer. If a reply shows posts the model forgot to save, Mackie saves them and lists their ledger ids underneath.
- **Content calendar.** A dated four-week plan for any month, two posts a week by default (one to three), mixing platforms and formats, generated in parallel and saved to the ledger. Each week is tied to an awareness day, giving day or org event when one falls in it, serves the campaign you name ("plan November around our year-end appeal"), and builds on your best-performing posts.
- **Learns what works.** Report results in plain words ("the volunteer post got 240 likes and 30 shares", "that email raised $400") and Mackie records them on the post. Your best performers are shown to Mackie on every turn, so "what should we post next?" is answered from evidence.
- **Any LLM provider.** Ten built-in providers plus anything LiteLLM supports. Extra configured providers act as automatic fallbacks.
- **No-code changes.** Add a platform, content type or format by dropping in a YAML file, and edit what Mackie learned in a JSON file. Changes apply on the next message. A broken file is skipped and reported, never fatal.

## Quick Start

You need a [Tavily](https://tavily.com) API key (free tier available) and a key for at least one LLM provider.

### Docker

```bash
cp .env.example .env        # set TAVILY_API_KEY and one provider key
docker compose up --build   # http://localhost:8000
```

### Local (Python 3.12; Linux, macOS, or WSL on Windows)

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt   # runtime + test dependencies
cp .env.example .env                  # set TAVILY_API_KEY and one provider key

chainlit run app.py   # web UI at http://localhost:8000
python main.py        # or the terminal version
```

### First run

Mackie asks for your organization's name and website, then asks you to confirm. The first time you ask for content, it researches the org, which takes about 5–20 seconds. The research is saved in `data/{org-id}/profile.json`, so later sessions skip this step. New orgs are invited to paste 2–3 recent posts so Mackie can learn their voice.

Each browser tab is its own conversation, and the terminal version resumes one conversation per org. In the terminal, `reset` clears the conversation (what Mackie learned is kept) and `quit` exits.

A typical conversation:

```
Suggest some posts for us.
Let's go with the second one, write it for LinkedIn.
Too corporate, we're more warm and grassroots. Redo it.
Now give me an Instagram version.
What programs does our org actually run?
Which posts have we worked on so far, and what's the status of each?
Suggest another post for next month.
```

The voice correction ("warm and grassroots") applies to every later post, in this session and in future ones. The last suggestion avoids ideas that are already in the ledger.

## Configuration

All settings live in `.env`; `.env.example` lists every variable with its default.

| Variable | Purpose |
|---|---|
| `TAVILY_API_KEY` | **Required.** Web search. |
| `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `XAI_API_KEY`, `GEMINI_API_KEY`, `MISTRAL_API_KEY`, `GROQ_API_KEY`, `DEEPSEEK_API_KEY`, `OPENROUTER_API_KEY`, `OPENCODE_API_KEY` | A provider is used when its key is set. |
| `QWEN_BASE_URL` + `QWEN_API_KEY` | Qwen on a self-hosted vLLM server. Both are required. From Docker, a server on the host is `http://host.docker.internal:PORT/v1`. |
| `LLM_MODEL` (+ `LLM_API_BASE`, `LLM_API_KEY`) | Any other model LiteLLM supports, as a full model string, e.g. `ollama_chat/llama3.1` with `http://localhost:11434`. |
| `{PROVIDER}_MODEL` | Optional model override for a provider, e.g. `ANTHROPIC_MODEL=claude-sonnet-5-5`. Unset, each provider uses the default in `.env.example`; Anthropic's is `claude-haiku-5-5`. |
| `LLM_PROVIDER` | Optional. Which configured provider to use first: `custom`, `anthropic`, `openai`, `xai`, `google`, `mistral`, `groq`, `deepseek`, `openrouter`, `opencode` or `qwen`. |
| `FAST_MODEL` | Optional. A cheaper model for background work: conversation summaries, the org brief and calendar plans. |
| `JUDGE_MODEL` | Optional. The judge for `scripts/eval.py`. Pick a different provider, or at least a different model family, from the assistant's, so it doesn't grade its own model's work. With only an OpenCode Zen key, use a Chat Completions model such as `openai/deepseek-v4-pro`; Zen's Claude, GPT and Gemini models don't work through this path. With an OpenAI key set too, `openai/…` models go to OpenAI unless `JUDGE_PROVIDER=opencode` (otherwise Mackie prints a warning). |
| `FAST_PROVIDER`, `JUDGE_PROVIDER` | Optional. The provider id for `FAST_MODEL` or `JUDGE_MODEL`, needed when two configured providers accept the same model string (OpenAI and OpenCode Zen both take `openai/…`). Set without a model, that provider's own model is used. |

With several providers configured, the first one in this order is the main model, unless `LLM_PROVIDER` says otherwise: custom (`LLM_MODEL`), Anthropic, OpenAI, xAI, Google, Mistral, Groq, DeepSeek, OpenRouter, OpenCode Zen, Qwen. The others become fallbacks: if a request to the main model fails to start (outage, rate limit, bad key), the next one is tried.

OpenCode Zen's free models (the `-free` ones and `big-pickle`) currently answer only requests from the OpenCode app itself. Through Mackie they return "OpenCode's free tier can only be used from within OpenCode", so Mackie's default is `glm-5.3-flash`, the cheapest paid model on Zen's Chat Completions endpoint ($0.15 in / $0.50 out per million tokens). Your OpenCode account needs credit; without it every request fails with "Insufficient account funds". Any other Chat Completions model works through `OPENCODE_MODEL`.

## Customizing Without Code

### Add a platform

Create a file in `skills/`, e.g. `skills/threads.yaml`:

```yaml
name: Threads
id: threads
tone: casual, conversational, text-forward
max_length: 500
hashtag_count: "0-3"
hashtag_style: minimal, only if natural
format_hints: |
  - Short and punchy
  - Put the call to action in the text; there are no link previews
```

`id`, `name` and a whole-number `max_length` are required. Write a multi-line field with `|`, as `format_hints` does above. Mackie uses the new platform from the next message on.

### Add a content type

Copy `skills/donation_appeal.yaml` (its comments explain each field) to a new file, keep `kind: content_type`, and give it a unique `id`, a `name`, a `purpose` and a `structure`.

### Add a format

Copy `skills/carousel.yaml` (`skills/feed_post.yaml` explains each field) to a new file, keep `kind: format`, and give it a unique `id`, a `name`, a `purpose` (the role it plays) and a `structure`. List the platforms it exists on under `platforms`, or leave that out for all. Add `has_caption: false` for a format without a caption, such as a story, so the platform's hashtag rule doesn't apply.

### Edit content pillars

Edit `skills/content_pillars.yaml`. To override the pillars for one org, add `"content_pillars": [{"name": "...", "description": "..."}]` to that org's `profile.json`.

### Edit what Mackie learned

Open `data/{org-id}/profile.json` in any text editor. It holds:

- the post ledger, including any results you reported (`performance`);
- preferences, with when and in what words each was learned (`preference_notes`);
- voice examples;
- the org brief and the research.

Set `"research": null` to make Mackie research the org again next session. Every save keeps the previous version as `profile.json.bak`, and a formatting mistake is reported with its line and column. You can also ask in chat: "what have you learned about us?" or "forget the LinkedIn voice rule".

### Check your edits

```bash
python scripts/validate.py                              # local
docker compose exec mackie python scripts/validate.py   # Docker
```

It lists the platforms, content types and formats it found, plus every problem in `skills/` (the content pillars file included) and `data/*/profile.json`. It changes nothing and exits with code 1 if anything is wrong. The web UI also reports problems: a broken skill file gets a **Heads-up** at the start of each chat, and a broken profile is named, with what's wrong, when you open that org.

## Tests and Evaluation

```bash
pytest                                  # offline: no API calls, no network, under 10 seconds
ruff check . && ruff format --check .   # lint and formatting (settings in ruff.toml)
```

Besides a test file per module, four files cover concerns that cut across modules:

- `tests/test_security.py`: SSRF bypasses, malformed tool arguments, path traversal, org isolation, prompt injection end to end.
- `tests/test_performance.py`: parallel research, calendar and judge calls; responsiveness (streaming, timeouts, a slow session not stalling others); scale (prompt size that stays flat as an org's memory grows, 20,000-post ledgers); linear-time parsing.
- `tests/test_ui.py`: the web and terminal UIs, and UX details such as streaming and error wording.
- `tests/test_sessions.py`: many people at once: concurrent sessions, two people on one org, interleaved replies, the web UI and CLI writing to one org from separate processes.

`tests/test_deploy.py` checks deployment settings and makes one real offline LiteLLM tool call that catches a broken LiteLLM install. A run that takes longer than 10 seconds fails.

```bash
pytest --cov=agent --cov=scripts --cov=app --cov=main --cov-branch   # branch coverage
```

Browser tests (`tests/browser/`) drive the real web UI in headless Chromium with a scripted model (no API calls): onboarding, streaming, tool steps, a phone-width layout and two browsers on one org. They take about 15 seconds, so a plain `pytest` skips them:

```bash
playwright install chromium   # once
pytest -m browser
```

CI runs lint, the tests, a 95% branch-coverage floor and the browser tests on every push and pull request.

`scripts/eval.py` runs the conversation above against a real org with the configured models (this costs API calls). It reports two kinds of checks:

- **LLM judge.** Each reply is scored 0–3 on source attribution, voice consistency and not repeating earlier ideas: 21 judge calls, run in parallel, for a total out of 63 (70% or more is a pass).
- **Saved data.** Checks what the run saved to `profile.json`, with no LLM: a preference learned by step 3, new posts in step 7, and a source on every suggestion.

```bash
python scripts/eval.py --org "BRCA Strong" --url https://brcastrong.org
```

The eval saves to the org's real profile under `data/`, the same as a normal chat.

## Project Structure

```
app.py              Web UI (Chainlit)
main.py             Terminal UI
agent/
  config.py         Provider registry and model selection
  assistant.py      System prompt and chat entry point
  graph.py          LangGraph loop: summarize → agent ⇄ tools, SQLite checkpoints
  tools.py          Tool definitions and dispatch, post checks, org brief, calendar
  memory.py         profile.json storage: ledger, preferences, voice examples
  research.py       Website scraping (with SSRF protection) and Tavily search
  skills.py         Loads and validates skills/*.yaml
  untrusted.py      Tags that mark third-party text as data (prompt-injection defense)
skills/             Platforms, content types, formats, content pillars (YAML)
scripts/
  validate.py       Checks skills/ and every profile.json
  eval.py           End-to-end evaluation
tests/              Tests (offline); tests/browser/ drives the real web UI
data/               Created at runtime; not committed
  checkpoints.db    Conversation history, one thread per chat session
  {org-id}/profile.json, profile.json.bak, trace.jsonl, .lock
```

`data/{org-id}/trace.jsonl` logs every tool call with its input, a result preview and its latency. `.lock` makes the web UI, the terminal and scripts take turns when they save to the same org.

See [design_doc.md](design_doc.md) for the architecture and the reasons behind each design choice.

## Known Limitations

- **One machine.** Several people can work on one org at once, from the web UI and the terminal, because every save to `profile.json` holds a per-org lock. That lock (`flock`) works within one machine and doesn't exist on native Windows, so run Mackie on Linux, macOS, WSL or Docker.
- **No authentication.** Anyone who can reach the web UI can open any org. Run it on a trusted network or put it behind an authenticating proxy.
- **Single process.** The SQLite checkpoint store does not support several replicas. One process handles hundreds of simultaneous chats.
- **No publishing.** Mackie drafts posts but does not publish them. `posted` is only a status in the ledger.
- **Results are entered by hand.** Mackie doesn't connect to platform analytics; you tell it the numbers in chat. It doesn't run paid ads.
- **Static websites only.** Sites rendered by JavaScript yield little text.
- **No image generation.** Mackie describes the visual for each post; your team makes it.

The design doc's Roadmap describes how each would be addressed.
