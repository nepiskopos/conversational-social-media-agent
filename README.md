# Mackie — Social Media Assistant for Nonprofits

Mackie is a conversational assistant for nonprofit marketing teams. It researches your organization, suggests and drafts on-brand posts for LinkedIn, Instagram and Facebook, learns your voice from feedback, and keeps a ledger of every post and its status.

## Features

- **Research before writing.** Reads your homepage and up to three About/Programs-style pages, runs three web searches in parallel, and keeps a short cited org brief for every session. Every suggestion names its source.
- **Platform-aware drafts.** Each platform's tone, length and hashtag rules live in a YAML file. Length and hashtag limits are enforced in code.
- **Content types and pillars.** Donation appeals, volunteer calls and event announcements follow a defined structure. Five editorial pillars frame suggestions. Both are editable YAML.
- **Learns from feedback.** Corrections are saved in your own words. Approved and pasted posts become voice examples. Ask "what have you learned about us?" or "forget that rule" at any time. All of it carries over to future sessions.
- **Post ledger.** Every suggestion and draft is tracked with a title, status (`suggested`, `draft`, `approved`, `posted`, `planned`, `rejected`), versions and dates, so "what did we work on this week?" has an answer. If a reply shows posts the model forgot to save, Mackie saves them and lists their ledger ids underneath.
- **Content calendar.** A four-week plan, generated in parallel and saved to the ledger.
- **Any LLM provider.** Ten built-in providers plus anything LiteLLM supports. Extra configured providers act as automatic fallbacks.
- **No-code changes.** Add a platform or content type by dropping in a YAML file, and edit what Mackie learned in a JSON file. Changes apply on the next message. A broken file is skipped and reported, never fatal.

## Quick Start

You need a [Tavily](https://tavily.com) API key (free tier available) and a key for at least one LLM provider.

### Docker

```bash
cp .env.example .env        # set TAVILY_API_KEY and one provider key
docker compose up --build   # http://localhost:8000
```

### Local (Python 3.12)

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
| `{PROVIDER}_MODEL` | Optional model override for a provider, e.g. `ANTHROPIC_MODEL=claude-sonnet-4-6`. |
| `LLM_PROVIDER` | Optional. Which configured provider to use first: `custom`, `anthropic`, `openai`, `xai`, `google`, `mistral`, `groq`, `deepseek`, `openrouter`, `opencode` or `qwen`. |
| `FAST_MODEL` | Optional. A cheaper model for background work: conversation summaries, the org brief and calendar plans. |
| `JUDGE_MODEL` | Optional. The judge for `scripts/eval.py`. Pick a different provider from the assistant's, so it doesn't grade its own model's work. |

With several providers configured, the first one in this order is the main model, unless `LLM_PROVIDER` says otherwise: custom (`LLM_MODEL`), Anthropic, OpenAI, xAI, Google, Mistral, Groq, DeepSeek, OpenRouter, OpenCode Zen, Qwen. The others become fallbacks: if a request to the main model fails to start (outage, rate limit, bad key), the next one is tried.

OpenCode Zen's free models may use your data to train the provider's models, so don't send anything you can't share.

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

### Edit content pillars

Edit `skills/content_pillars.yaml`. To override the pillars for one org, add `"content_pillars": [{"name": "...", "description": "..."}]` to that org's `profile.json`.

### Edit what Mackie learned

Open `data/{org-id}/profile.json` in any text editor. It holds:

- the post ledger;
- preferences, with when and in what words each was learned (`preference_notes`);
- voice examples;
- the org brief and the research.

Set `"research": null` to make Mackie research the org again next session. Every save keeps the previous version as `profile.json.bak`, and a formatting mistake is reported with its line and column. You can also ask in chat: "what have you learned about us?" or "forget the LinkedIn voice rule".

### Check your edits

```bash
python scripts/validate.py                              # local
docker compose exec mackie python scripts/validate.py   # Docker
```

It lists the platforms and content types it found, plus every problem in `skills/` and `data/*/profile.json`. It changes nothing and exits with code 1 if anything is wrong. The web UI also reports problems: a broken skill file gets a **Heads-up** at the start of each chat, and a broken profile is named, with what's wrong, when you open that org.

## Tests and Evaluation

```bash
pytest                                  # offline: no API calls, no network, under 10 seconds
ruff check . && ruff format --check .   # lint and formatting (settings in ruff.toml)
```

CI runs both on every push and pull request.

`scripts/eval.py` runs the conversation above against a real org with the configured models (this costs API calls). It reports two kinds of checks:

- **LLM judge.** Each reply is scored on source attribution, voice consistency and not repeating earlier ideas: 21 judge calls, run in parallel.
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
skills/             Platforms, content types, content pillars (YAML)
scripts/
  validate.py       Checks skills/ and every profile.json
  eval.py           End-to-end evaluation
tests/              Unit tests (offline)
data/               Created at runtime; not committed
  checkpoints.db    Conversation history, one thread per chat session
  {org-id}/profile.json, profile.json.bak, trace.jsonl
```

`data/{org-id}/trace.jsonl` logs every tool call with its input, a result preview and its latency.

See [design_doc.md](design_doc.md) for the architecture and the reasons behind each design choice.

## Known Limitations

- **Single user per org at a time.** Writes to `profile.json` are not locked; two people editing the same org at once can overwrite each other.
- **No authentication.** Anyone who can reach the web UI can open any org. Run it on a trusted network or put it behind an authenticating proxy.
- **Single process.** The SQLite checkpoint store does not support several replicas.
- **No publishing.** Mackie drafts posts but does not publish them. `posted` is only a status in the ledger.
- **Static websites only.** Sites rendered by JavaScript yield little text.
- **No image generation.**

The design doc's Roadmap describes how each would be addressed.
