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
- **Learns from feedback.** Corrections are saved in your own words, a sentence or two each. Approved and pasted posts become voice examples. Ask "what have you learned about us?" or "forget that rule" at any time. All of it carries over to future sessions.
- **Post ledger.** Every suggestion and draft is tracked with a title, status (`suggested`, `draft`, `approved`, `posted`, `planned`, `rejected`), versions and dates, so "what did we work on this week?" has an answer. If a reply shows posts the model forgot to save, Mackie saves them and lists their ledger ids underneath.
- **Content calendar.** A dated four-week plan for any month, two posts a week by default (one to three), mixing platforms and formats, generated in parallel and saved to the ledger. Each week is tied to an awareness day, giving day or org event when one falls in it, serves the campaign you name ("plan November around our year-end appeal"), and builds on your best-performing posts.
- **Learns what works.** Report results in plain words ("the volunteer post got 240 likes and 30 shares", "that email raised $400") and Mackie records them on the post. Your best performers are shown to Mackie on every turn, so "what should we post next?" is answered from evidence.
- **Any LLM provider.** Ten built-in providers plus anything LiteLLM supports. Extra configured providers act as automatic fallbacks.
- **Accounts.** Each person logs in, and sees only the organizations they belong to. Each org's data is kept apart by the database itself.
- **No-code changes.** Add a platform, content type or format by dropping in a YAML file; it applies from the next message. Edit what Mackie learned in an exported JSON file and import it back. A broken skill file is skipped and reported, and an edited export with a mistake is refused; neither stops Mackie.

## Quick Start

You need a [Tavily](https://tavily.com) API key (free tier available), a key for at least one LLM provider, and PostgreSQL 15 or later (Docker brings its own).

### Docker

```bash
cp .env.example .env   # set TAVILY_API_KEY, one provider key, the three database passwords,
                       # MACKIE_DEFAULT_PASSWORD and CHAINLIT_AUTH_SECRET (see the file's comments)
docker compose up --build -d   # http://localhost:8000: log in as admin with MACKIE_DEFAULT_PASSWORD
```

Compose starts PostgreSQL (reachable only by Mackie's containers), creates or updates the schema, makes sure the default account exists, then starts Mackie.

### Local (Python 3.12; Linux, macOS, or WSL on Windows)

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt   # runtime + test dependencies
cp .env.example .env                  # set the keys, the two database URLs and CHAINLIT_AUTH_SECRET
python scripts/migrate.py             # create or update the schema, and the default account

chainlit run app.py   # web UI at http://localhost:8000
python main.py        # or the terminal version
```

`scripts/migrate.py` makes sure a default account exists, so there is always someone who can log in: `MACKIE_DEFAULT_USERNAME` (`admin` unless set) with the password `MACKIE_DEFAULT_PASSWORD`. Without a password set, it makes a random one and prints it once. An existing account is never changed, so a password changed with `scripts/users.py passwd` stays changed.

The database needs two roles, neither a superuser: an owner for the schema (`MACKIE_MIGRATION_URL`) and the role Mackie connects as (`MACKIE_DATABASE_URL`). `docker/postgres-init.sh` shows how to create them. Mackie refuses to start as a superuser, since that would bypass the per-org protection.

### Users and access

```bash
python scripts/users.py add alice@example.org                # asks for a password (12+ characters)
python scripts/users.py grant alice@example.org "BRCA Strong"
python scripts/users.py revoke alice@example.org "BRCA Strong"
python scripts/users.py passwd alice@example.org             # also unlocks the account
python scripts/users.py remove alice@example.org
python scripts/users.py list
```

In Docker, prefix each command with `docker compose run --rm mackie`. A user who starts a new organization becomes its member; an existing one needs a `grant`. Five wrong passwords in a row lock an account for 15 minutes. A `revoke` or `remove` also ends chats that are already open: their next message is refused.

### First run

Log in, then tell Mackie your organization's name and website and confirm. The first time you ask for content, it researches the org, which takes about 5–20 seconds. The research is saved, so later sessions skip this step. New orgs are invited to paste 2–3 recent posts so Mackie can learn their voice.

Your conversation with an organization carries on where it left off, after a refresh or on another day: the web UI shows the last few messages again and offers **Start a new conversation**; the terminal says it is carrying on, and `reset` starts afresh. Either way, what Mackie learned is kept. Each person has their own conversation; the web UI and the terminal have one each. In the terminal, `quit` exits.

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

### Moving from the data/ folder

Earlier versions kept each org in `data/{org-id}/profile.json`. Import them once:

```bash
python scripts/import_profiles.py data/*/profile.json
docker compose run --rm -v ./data:/app/data:ro mackie sh -c 'python scripts/import_profiles.py data/*/profile.json'
```

The old files have no time zone, so their times are read in the importing machine's: in Docker, add `-e TZ=Europe/Athens` (your zone) after `run --rm`. Posts, preferences, voice examples and research move over. Conversation history (`data/checkpoints.db`) does not: chats start fresh. Then `grant` each user the orgs they work on.

## Configuration

All settings live in `.env`; `.env.example` lists every variable with its default.

| Variable | Purpose |
|---|---|
| `MACKIE_DATABASE_URL` | **Required.** The PostgreSQL role Mackie connects as. It must not own the schema or be a superuser. Add `?sslmode=verify-full` for a database on another machine. |
| `MACKIE_MIGRATION_URL` | The schema owner's role, used only by `scripts/migrate.py`. |
| `MACKIE_DATABASE_POOL_SIZE` | Optional: database connections per Mackie process (10 unless set). A turn holds one only while a query runs, so 10 serves many simultaneous chats. |
| `CHAINLIT_AUTH_SECRET` | **Required** for the web UI: signs login sessions. Generate one with `chainlit create-secret`. |
| `POSTGRES_PASSWORD`, `MACKIE_OWNER_PASSWORD`, `MACKIE_APP_PASSWORD` | Docker only: passwords for the database container's superuser and Mackie's two roles. Use letters and digits. |
| `MACKIE_DEFAULT_USERNAME`, `MACKIE_DEFAULT_PASSWORD` | The account `scripts/migrate.py` makes sure exists (`admin` unless set). The password is required with Docker. |
| `MACKIE_LOG_LEVEL` | Optional: `DEBUG`, `INFO`, `WARNING` or `ERROR`. Unset, the web UI logs from `INFO` (logins, refused access, failures) and the terminal from `WARNING`. Passwords, keys and what users write are never logged. |
| `TZ` | Your time zone, e.g. `Europe/Athens`: what "today" means for the agenda. Unset, the server's (UTC in Docker). |
| `TAVILY_API_KEY` | **Required.** Web search. |
| `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `XAI_API_KEY`, `GEMINI_API_KEY`, `MISTRAL_API_KEY`, `GROQ_API_KEY`, `DEEPSEEK_API_KEY`, `OPENROUTER_API_KEY`, `OPENCODE_API_KEY` | A provider is used when its key is set. |
| `QWEN_BASE_URL` + `QWEN_API_KEY` | Qwen on a self-hosted vLLM server. Both are required. From Docker, a server on the host is `http://host.docker.internal:PORT/v1`. |
| `LLM_MODEL` (+ `LLM_API_BASE`, `LLM_API_KEY`) | Any other model LiteLLM supports, as a full model string, e.g. `ollama_chat/llama3.1` with `http://localhost:11434`. |
| `{PROVIDER}_MODEL` | Optional model override for a provider, e.g. `ANTHROPIC_MODEL=claude-sonnet-5-5`. Unset, each provider uses the default in `.env.example`; Anthropic's is `claude-haiku-5-5`. |
| `LLM_PROVIDER` | Optional. Which configured provider to use first: `custom`, `anthropic`, `openai`, `xai`, `google`, `mistral`, `groq`, `deepseek`, `openrouter`, `opencode` or `qwen`. |
| `FAST_MODEL` | Optional. A cheaper model for background work: conversation summaries, the org brief and calendar plans. |
| `JUDGE_MODEL` | Optional. The judge for `scripts/eval.py`. Pick a different provider, or at least a different model family, from the assistant's, so it doesn't grade its own model's work. With only an OpenCode Zen key, use a Chat Completions model such as `openai/deepseek-v4-pro`; Zen's Claude, GPT and Gemini models don't work through this path. With an OpenAI key set too, `openai/…` models go to OpenAI unless `JUDGE_PROVIDER=opencode` (otherwise Mackie prints a warning). |
| `FAST_PROVIDER`, `JUDGE_PROVIDER` | Optional. The provider id for `FAST_MODEL` or `JUDGE_MODEL`, needed when two configured providers accept the same model string (OpenAI and OpenCode Zen both take `openai/…`). Set without a model, that provider's own model is used. |

With several providers configured, the first one in this order is the main model, unless `LLM_PROVIDER` says otherwise: custom (`LLM_MODEL`), Anthropic, OpenAI, xAI, Google, Mistral, Groq, DeepSeek, OpenRouter, OpenCode Zen, Qwen. The others become fallbacks: if a request to the main model fails to start (outage, rate limit, bad key) and one retry fails too, the next one is tried.

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

Edit `skills/content_pillars.yaml`. To override the pillars for one org, set `"content_pillars": [{"name": "...", "description": "..."}]` in that org's exported profile and import it (below).

### Edit what Mackie learned

Export the org to a JSON file, edit it in any text editor, then import it:

```bash
python scripts/export_profile.py "BRCA Strong" -o brca-strong.json
python scripts/import_profiles.py brca-strong.json   # replaces what is stored for the org
```

The file holds:

- the post ledger, including any results you reported (`performance`);
- preferences, with when and in what words each was learned (`preference_notes`);
- voice examples;
- the org brief and the research.

Set `"research": null` to make Mackie research the org again next session. A post you add without an `id` gets the next free one. A file with a mistake (a formatting error, with its line and column, or an unknown status) is refused and nothing changes. Import with `--check` to only check it. You can also ask in chat: "what have you learned about us?" or "forget the LinkedIn voice rule".

### Check your edits

```bash
python scripts/validate.py                               # skills/
python scripts/validate.py brca-strong.json              # skills/ and an edited export
docker compose exec mackie python scripts/validate.py    # Docker
```

It lists the platforms, content types and formats it found, plus every problem in `skills/` (the content pillars file included) and in any profile file you name. It changes nothing and exits with code 1 if anything is wrong. The web UI also reports a broken skill file with a **Heads-up** at the start of each chat.

## Tests and Evaluation

The tests need a disposable PostgreSQL server; each run creates its own database and roles there and drops the database afterwards:

```bash
docker run -d --name mackie-test-db -p 127.0.0.1:55432:5432 -e POSTGRES_PASSWORD=mackie \
  --tmpfs /var/lib/postgresql/data postgres:17-alpine -c fsync=off   # once; or set TEST_DATABASE_URL
pytest                                  # no API calls, no outside network, under 30 seconds
ruff check . && ruff format --check .   # lint and formatting (settings in ruff.toml)
```

Besides a test file per module, these files cover concerns that cut across modules:

- `tests/test_security.py`: SSRF bypasses, malformed tool arguments, hostile org ids, org isolation (row-level security on every table, an app role that can't change the schema or switch the policies off, SQL that is stored and never run), prompt injection end to end.
- `tests/test_performance.py`: parallel research, calendar and judge calls; responsiveness (streaming, timeouts, a slow session not stalling others); scale (prompt size that stays flat as an org's memory grows, 20,000-post ledgers); linear-time parsing.
- `tests/test_ui.py`: the web and terminal UIs, and UX details such as streaming and error wording.
- `tests/test_logging.py`: what is logged (logins, refused access, every absorbed failure) and what never is (secrets, what users write), and forged log lines.
- `tests/test_errors.py`: what users are told when the database or the AI service fails, with no raw error text on screen.
- `tests/test_sessions.py`: many people at once: concurrent sessions, two people on one org, interleaved replies, separate processes writing to one org.

`tests/test_deploy.py` checks deployment settings and makes one real offline LiteLLM tool call that catches a broken LiteLLM install. A run that takes longer than 30 seconds fails (60 on CI, set with `MACKIE_TEST_BUDGET_S`).

```bash
pytest --cov=agent --cov=scripts --cov=app --cov=main --cov-branch   # branch coverage
```

Browser tests (`tests/browser/`) drive the real web UI in headless Chromium with a scripted model (no API calls): the login, onboarding, an org the user may not open, streaming, tool steps, a phone-width layout and two users on one org. They take about 30 seconds, so a plain `pytest` skips them:

```bash
playwright install chromium   # once
pytest -m browser
```

CI runs lint, the tests, a 95% branch-coverage floor and the browser tests on every push and pull request, and deploys the Docker Compose stack: the migration must run cleanly and create the default account, the app must not run as root or reach the database as a privileged role, the database must not be published, and the default account must log in and onboard an organization in a browser (`tests/deployed/`, `pytest -m deployed` against a running stack).

`scripts/eval.py` runs the conversation above against a real org with the configured models (this costs API calls). It reports three kinds of checks:

- **LLM judge.** Each reply is scored 0–3 on source attribution, voice consistency and not repeating earlier ideas: 21 judge calls, run in parallel, for a total out of 63 (70% or more is a pass).
- **Saved data.** Checks what the run saved, with no LLM: a preference learned by step 3, new posts in step 7, and a source on every suggestion.
- **Memory.** A date mentioned at the start of a long conversation must survive into the summary's open requests and dates, and Mackie must recall it when asked much later. The long conversation is stored, not chatted, so this costs one summary call and one answer.

```bash
python scripts/eval.py --org "BRCA Strong" --url https://brcastrong.org
```

The eval saves to the org's real memory in the database, the same as a normal chat.

## Project Structure

```
app.py              Web UI (Chainlit), with the login
main.py             Terminal UI, with the login
agent/
  config.py         Provider registry and model selection
  assistant.py      System prompt and chat entry point
  graph.py          LangGraph loop: summarize → agent ⇄ tools, PostgreSQL checkpoints
  tools.py          Tool definitions and dispatch, post checks, org brief, calendar
  db.py             PostgreSQL pool, per-org transactions, migrations
  memory.py         Org memory: ledger, preferences, voice examples, research, tool trace
  accounts.py       Users, passwords, logins and org access
  errors.py         What users are told when something fails
  logs.py           Log setup, with secrets redacted
  research.py       Website scraping (with SSRF protection) and Tavily search
  skills.py         Loads and validates skills/*.yaml
  untrusted.py      Tags that mark third-party text as data (prompt-injection defense)
skills/             Platforms, content types, formats, content pillars (YAML)
migrations/         Database schema, applied in order by scripts/migrate.py
docker/             The database container's first-start script (roles, database)
scripts/
  migrate.py        Creates or updates the schema
  users.py          Accounts and their org access
  export_profile.py / import_profiles.py   An org's memory as an editable JSON file
  validate.py       Checks skills/ and profile files
  eval.py           End-to-end evaluation
tests/              Tests; tests/browser/ drives the real web UI, tests/deployed/ a running Compose stack
```

The `tool_calls` table logs every tool call with its input, a result preview and its latency.

See [design_doc.md](design_doc.md) for the architecture and the reasons behind each design choice.

## Known Limitations

- **Password logins only.** No single sign-on (OAuth) and no self-service password reset; an admin runs `scripts/users.py`. Removing someone's access stops their open chats at the next message; a reply already being written finishes.
- **One conversation per person, organization and screen.** Two browser tabs of the same person on one organization share their conversation and take turns; two terminals (separate processes), or two servers, don't, and replies sent at the same moment can overwrite each other's history.
- **Serve it over HTTPS.** Put Mackie behind a reverse proxy with TLS, or passwords and session cookies cross the network in the clear.
- **No publishing.** Mackie drafts posts but does not publish them. `posted` is only a status in the ledger.
- **Results are entered by hand.** Mackie doesn't connect to platform analytics; you tell it the numbers in chat. It doesn't run paid ads.
- **Static websites only.** Sites rendered by JavaScript yield little text.
- **No image generation.** Mackie describes the visual for each post; your team makes it.

The design doc's Roadmap describes how each would be addressed.
