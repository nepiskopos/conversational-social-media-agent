# Mackie — AI Social Media Manager for Nonprofits

Mackie is a conversational social media manager for nonprofit marketing teams. It keeps your social media active and on-brand so your staff can spend their time on the mission:

- **Plan:** monthly content calendars built around your campaigns, events and the awareness days that matter to your cause.
- **Create:** posts for LinkedIn, Instagram and Facebook, plus supporter email, in your own voice, each with an idea for the visual.
- **Optimize:** tell Mackie how posts performed and it suggests what to post next based on what worked.

Like a teammate, Mackie doesn't wait to be asked: each session opens with what needs you (drafts to approve, posts to publish, planned posts due this week, results to report). Nothing is approved without you, and nothing is published unless you ask: Mackie publishes approved posts to your connected LinkedIn, Facebook and Instagram accounts when you say so (simulated for now), or your team posts them. Your team makes the final graphics.

## Features

- **Research before writing.** Reads your homepage and up to three About/Programs-style pages, runs three web searches, and keeps a short cited org brief for every session. Every suggestion names its source.
- **Platform-aware drafts.** Each platform's tone, length and hashtag rules live in a YAML file and are enforced on finished drafts, and again before a post is approved; rough ideas are exempt. Instagram and Facebook drafts must describe the photo or graphic to pair with them. Email (subject, preview text, one call to action) is a platform too.
- **Content types, formats and pillars.** Donation appeals, volunteer calls and event announcements follow a defined structure. Mackie picks the format that fits (feed posts are the lasting record, carousels teach step by step, stories keep followers close, reels reach new people) and writes out story frames, carousel slides and reel shot lists in full. Five editorial pillars frame suggestions.
- **Content guardrails.** Every post is honest, kind and inclusive, takes no political side, and is never dismissive, aggressive or crude. The rules, and a list of words no post may ever contain (checked in code, so a post with one is never saved or approved), live in one file you can edit.
- **Learns from feedback.** Corrections are saved in your own words, a sentence or two each, and approved or pasted posts become voice examples. Only what you type or approve is learned, so a web page Mackie reads can't teach it anything. Ask "what have you learned about us?" or "forget that rule" at any time. All of it carries over to future sessions.
- **Post ledger.** Every suggestion and draft is tracked with a title, status (`suggested`, `draft`, `approved`, `posted`, `planned`, `rejected`), versions and dates, so "what did we work on this week?" has an answer.
- **Content calendar.** A dated four-week plan for any month, two posts a week by default (one to three), mixing platforms and formats. Each week is tied to an awareness day, giving day or org event when one falls in it, serves the campaign you name ("plan November around our year-end appeal"), and builds on your best-performing posts.
- **Publishing (simulated for now).** Say "publish post_004 to LinkedIn and Facebook" and Mackie publishes the approved post, exactly as you approved it, to each account you named, never twice to the same one. Until live publishing is switched on, nothing reaches the platforms: the publication is recorded and Mackie says it was simulated.
- **Learns what works.** Report results in plain words ("the volunteer post got 240 likes and 30 shares", "that email raised $400") and Mackie records them, so "what should we post next?" is answered from evidence.
- **Any LLM provider.** Ten built-in providers plus anything LiteLLM supports; extra configured providers act as automatic fallbacks.
- **Accounts.** Each person logs in and sees only the organizations they belong to; the database itself keeps each org's data apart.
- **No-code changes.** Add a platform, content type or format by dropping in a YAML file; it applies from the next message. Edit what Mackie learned in an exported JSON file and import it back. A broken file is reported, never fatal.

## Quick Start

You need a [Tavily](https://tavily.com) API key (free tier available), a key for at least one LLM provider, and PostgreSQL 15 or later (Docker brings its own).

### Docker

1. `cp .env.example .env`, then set in `.env`:
   - `TAVILY_API_KEY` and one provider key, such as `ANTHROPIC_API_KEY`;
   - `POSTGRES_PASSWORD`, `MACKIE_OWNER_PASSWORD` and `MACKIE_APP_PASSWORD`: letters and digits, e.g. the output of `openssl rand -hex 24`, one each;
   - `MACKIE_DEFAULT_PASSWORD`: the password of the `admin` account, 12 characters or more;
   - `CHAINLIT_AUTH_SECRET`: e.g. the output of `openssl rand -hex 32`.

   Leave `MACKIE_DATABASE_URL` and `MACKIE_MIGRATION_URL` as they are: Compose builds both from the passwords.
2. Start it:

   ```bash
   docker compose up --build -d
   ```

3. Open http://localhost:8000 and log in as `admin` with `MACKIE_DEFAULT_PASSWORD`.

Compose starts PostgreSQL (reachable only by Mackie's containers), creates or updates the schema, makes sure the default account exists, then starts Mackie. Compose refuses to start while any of the passwords or the secret is missing. The web UI answers only on this machine; see [Running it for your team](#running-it-for-your-team).

### Local (Python 3.12; Linux, macOS, or WSL on Windows)

1. Create the database and Mackie's two roles once, as a PostgreSQL superuser (e.g. `psql -U postgres`). Mackie connects as `mackie_app`, which can only read and write rows; `mackie_owner` owns the schema and is used only for migrations. Neither may be a superuser, since that would bypass the per-org protection. Use passwords of letters and digits (e.g. `openssl rand -hex 24`), as they go into connection URLs:

   ```sql
   CREATE ROLE mackie_owner LOGIN PASSWORD 'owner-password';
   CREATE ROLE mackie_app LOGIN PASSWORD 'app-password';
   CREATE DATABASE mackie OWNER mackie_owner;
   REVOKE ALL ON DATABASE mackie FROM PUBLIC;
   GRANT CONNECT ON DATABASE mackie TO mackie_app;
   \c mackie
   REVOKE CREATE ON SCHEMA public FROM PUBLIC;
   ALTER SCHEMA public OWNER TO mackie_owner;
   ```

2. Install Mackie and configure it:

   ```bash
   python3.12 -m venv .venv              # any Python 3.12
   source .venv/bin/activate             # the commands below assume this; or call .venv/bin/python
   pip install -r requirements.txt       # requirements-dev.txt instead to run the tests
   cp .env.example .env
   ```

   In `.env`, put the two passwords into `MACKIE_DATABASE_URL` and `MACKIE_MIGRATION_URL` (in place of `CHANGE_ME`), and set `TAVILY_API_KEY`, one provider key, `CHAINLIT_AUTH_SECRET` (e.g. the output of `openssl rand -hex 32`) and `MACKIE_DEFAULT_PASSWORD` (12 characters or more).

3. Create the schema and the default account, then start Mackie:

   ```bash
   python scripts/migrate.py   # safe to run again after every update
   chainlit run app.py         # web UI at http://localhost:8000
   python main.py              # or the terminal version
   ```

   Log in as `admin` with `MACKIE_DEFAULT_PASSWORD`. Without it, `migrate.py` makes a random password and prints it this once.

## Using Mackie

Log in, then tell Mackie your organization's name, and its website if it has one (the web UI asks at every visit, waits two minutes for an answer and asks you to confirm). Mackie reads only the website given here, never one a web page or the conversation suggests; a website typed for an org that has none yet is added, and changing one is done in an [exported profile](#edit-what-mackie-learned). The first time you ask for content, it researches the org, which takes about 5–20 seconds. The research is saved, so later sessions skip this step; ask Mackie to update its research when your site or news changes. In the web UI, new orgs are invited to paste 2–3 recent posts so Mackie can learn their voice.

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

The voice correction ("warm and grassroots") applies to every later post, in this session and in future ones, and the last suggestion avoids ideas that are already in the ledger.

Everything happens in chat, approvals included: say "approve post_003", "drop the second idea" or "we posted it on Friday", and Mackie updates the ledger. Mackie approves only a post it has shown you, so a draft from an earlier conversation is shown in full first. To change a draft, say what to change ("make it warmer") or name the exact words ("replace 'our helpers' with 'our volunteers'", "cut the last sentence"): an exact change swaps just those words, and Mackie says so if they aren't in the post, or appear more than once. Either way you get a new version, shown in full for approval; earlier versions stay in the ledger. A draft that breaks its platform's rules (too long, wrong hashtag count, no visual where one is needed) can't be approved until it is fixed.

To publish an approved post, ask in one message and name each platform: "publish it to LinkedIn and Facebook". A plain "yes" isn't enough, so a web page Mackie reads can never make it publish. Mackie checks everything first and publishes nothing if one platform can't take the post:

- the post must be approved, and meet each platform's rules (a LinkedIn version goes only to LinkedIn; a general post goes wherever it fits);
- the organization needs an account connected for that platform, with the permissions publishing needs (see [Connect social media accounts](#connect-social-media-accounts)).

A published post is marked `posted`, and asking again for the same platform is refused, so nothing goes out twice. Publishing is simulated for now: nothing reaches the platforms yet, and Mackie says so.

Your conversation with an organization carries on where it left off, after a refresh or on another day: the web UI shows the last few messages again and offers **Start a new conversation**; the terminal says it is carrying on, and `reset` starts afresh. Either way, what Mackie learned is kept. Each person has their own conversation, one in the web UI and one in the terminal. In the terminal, `quit` exits. A message can be up to 20,000 characters; sent messages can't be edited, so send a new one instead.

## Users and access

`scripts/migrate.py` makes sure a default account exists, so there is always someone who can log in: `MACKIE_DEFAULT_USERNAME` (`admin` unless set) with the password `MACKIE_DEFAULT_PASSWORD`. An existing account is never changed, so a password changed with `passwd` stays changed. `admin` is an ordinary account: accounts and access are managed only with these commands, by whoever runs the server.

```bash
python scripts/users.py add alice@example.org                # asks for a password (12+ characters)
python scripts/users.py grant alice@example.org "BRCA Strong"
python scripts/users.py revoke alice@example.org "BRCA Strong"
python scripts/users.py passwd alice@example.org             # also unlocks the account
python scripts/users.py remove alice@example.org
python scripts/users.py list
```

In Docker, prefix each command with `docker compose run --rm mackie` (`-T` after `run` when you pipe a password in). `add` and `passwd` also take `--password-stdin`, which reads the password from one line of input, for scripts.

- A user who starts a new organization becomes its member; anyone else needs a `grant`.
- Five wrong passwords in a row lock an account for 15 minutes.
- `revoke`, `remove` and `passwd` also end chats that are already open: their next message is refused.

## Connect social media accounts

Mackie publishes only to accounts an admin has connected for the organization, one per platform:

```bash
python scripts/platform_accounts.py connect "BRCA Strong" linkedin urn:li:organization:123 --name "BRCA Strong"
python scripts/platform_accounts.py connect "BRCA Strong" facebook 1234567890
python scripts/platform_accounts.py connect "BRCA Strong" instagram 17841400000 --scopes instagram_basic
python scripts/platform_accounts.py list "BRCA Strong"
python scripts/platform_accounts.py disconnect "BRCA Strong" instagram
```

The account id is the platform's id of the page or profile. `--scopes` lists the permissions the organization actually granted; without it, the ones publishing needs are assumed. `list` shows what an account is missing:

| Platform | Permissions needed |
|---|---|
| LinkedIn | `w_organization_social` |
| Facebook | `pages_manage_posts` |
| Instagram | `instagram_basic`, `instagram_content_publish` |

No access token is stored in Mackie's database. Publishing is simulated for now; when live publishing arrives, each platform's token will come from a secret store (see the design doc's Publishing section). In Docker, prefix each command with `docker compose run --rm mackie`.

## Running it for your team

- **Serve it over HTTPS.** Compose publishes the web UI on `127.0.0.1:8000` only. Put a reverse proxy with TLS (Caddy, nginx, Traefik) in front of it for other machines, have it mark cookies `Secure`, limit request bodies to 1 MB (Mackie refuses larger ones itself), and rate-limit `/login` there: Mackie's lockout is per account, so it doesn't slow down one password tried against many accounts.
- **Rename the default account** with `MACKIE_DEFAULT_USERNAME` before the first start, so `admin` isn't a known target.
- **Several organizations on one server?** Set `MACKIE_USERS_CREATE_ORGS=false`, so nobody can claim an organization's name, and set its website, before its team arrives. `grant` then creates each one, and the first member to open it types its website.
- **Back up the database**, which holds everything (org memory, accounts, conversations): `docker compose exec -T postgres pg_dump -U postgres mackie > mackie-backup.sql`. To restore it into a fresh stack, start only the database (`docker compose up -d --wait postgres`), run `docker compose exec -T postgres psql -U postgres mackie < mackie-backup.sql`, then `docker compose up -d`.
- **Update** with `git pull`, then `docker compose up --build -d`, which also updates the schema. Locally: `pip install -r requirements.txt`, `python scripts/migrate.py`, and restart Mackie.
- **Watch the log** (`docker compose logs mackie`) for refused logins, lockouts and failures; passwords, keys and what users write are never logged.

## Configuration

All runtime settings live in `.env`; `.env.example` lists each one with notes on its use.

| Variable | Purpose |
|---|---|
| `MACKIE_DATABASE_URL` | **Required.** The PostgreSQL role Mackie connects as. It must not own the schema or be a superuser. Add `?sslmode=verify-full` for a database on another machine. |
| `MACKIE_MIGRATION_URL` | **Required** for `scripts/migrate.py`: the schema owner's role, used for nothing else. Docker sets both URLs itself. |
| `MACKIE_DATABASE_POOL_SIZE` | Optional: database connections per Mackie process (10 unless set); 10 serves many simultaneous chats. |
| `CHAINLIT_AUTH_SECRET` | **Required** for the web UI: signs login sessions. |
| `POSTGRES_PASSWORD`, `MACKIE_OWNER_PASSWORD`, `MACKIE_APP_PASSWORD` | Docker only: passwords for the database container's superuser and Mackie's two roles. |
| `MACKIE_USERS_CREATE_ORGS` | Optional: `false` (or `0`, `no`, `off`) lets only `scripts/users.py grant` create organizations (see [Running it for your team](#running-it-for-your-team)). Unset, any logged-in user can start one. |
| `MACKIE_DEFAULT_USERNAME`, `MACKIE_DEFAULT_PASSWORD` | The default account (see [Users and access](#users-and-access)). The password is required with Docker. |
| `MACKIE_LOG_LEVEL` | Optional: `DEBUG`, `INFO`, `WARNING` or `ERROR`. Unset, the web UI, `users.py` and `migrate.py` log from `INFO`, the terminal and the export, import and eval scripts from `WARNING`; `validate.py` prints only its report. |
| `TZ` | Your time zone, e.g. `Europe/Athens`: what "today" means for the agenda and date searches. Unset, the server's (UTC in Docker). |
| `TAVILY_API_KEY` | **Required.** Web search. |
| `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `XAI_API_KEY`, `GEMINI_API_KEY`, `MISTRAL_API_KEY`, `GROQ_API_KEY`, `DEEPSEEK_API_KEY`, `OPENROUTER_API_KEY`, `OPENCODE_API_KEY` | A provider is used when its key is set. Leave the others blank: any value, even a placeholder, counts as set. |
| `OPENCODE_BASE_URL` | Optional: OpenCode Zen's endpoint, `https://opencode.ai/zen/v1` unless set. |
| `QWEN_BASE_URL` + `QWEN_API_KEY` | Qwen on a self-hosted vLLM server; both are required. From Docker, a server on the host is `http://host.docker.internal:PORT/v1`. |
| `LLM_MODEL` (+ `LLM_API_BASE`, `LLM_API_KEY`) | Any other model LiteLLM supports, as a full model string, e.g. `ollama_chat/llama3.1` with `http://localhost:11434`. |
| `ANTHROPIC_MODEL`, `OPENAI_MODEL`, `GEMINI_MODEL`, … | Optional model for a provider, named after its key (`GEMINI_MODEL` for Google). `.env.example` shows each default; Anthropic's is `claude-haiku-5-5`. |
| `LLM_PROVIDER` | Optional: which configured provider to use first, by id (listed in `.env.example`). |
| `FAST_MODEL` | Optional: a cheaper model for background work (conversation summaries, the org brief, calendar plans). |
| `JUDGE_MODEL` | Optional: the judge for `scripts/eval.py`; pick another provider, or at least another model family, than the assistant's. |
| `FAST_PROVIDER`, `JUDGE_PROVIDER` | Optional: only needed when two configured providers accept the same model name. OpenAI and OpenCode Zen both take `openai/…`, which goes to OpenAI unless this says `opencode`. Set alone, it uses that provider's own model. |

With several providers configured, the first one in this order is the main model, unless `LLM_PROVIDER` says otherwise: custom (`LLM_MODEL`), Anthropic, OpenAI, xAI, Google, Mistral, Groq, DeepSeek, OpenRouter, OpenCode Zen, Qwen. The others are fallbacks: if a request to the main model fails to start (outage, rate limit, bad key) and one retry fails too, the next one is tried.

**OpenCode Zen** needs an account with credit. Its free models answer only the OpenCode app, so Mackie defaults to the paid `glm-5.3-flash`; the errors "can only be used from within OpenCode" and "Insufficient account funds" mean what they say.

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
needs_visual: false
```

`id` (lowercase letters, digits and `_`), `name` and a whole-number `max_length` are required. Write a multi-line field with `|`, as `format_hints` does above. With `needs_visual: true`, a finished post must describe its photo or graphic, as on Instagram and Facebook. Mackie uses the new platform from the next message on.

### Add a content type

Copy `skills/donation_appeal.yaml` (its comments explain each field) to a new file, keep `kind: content_type`, and give it a unique `id`, a `name`, a `purpose` and a `structure`.

### Add a format

Copy `skills/feed_post.yaml` (its comments explain each field) to a new file, keep `kind: format`, and give it a unique `id`, a `name`, a `purpose` (the role it plays) and a `structure`. List the platforms it exists on under `platforms`, or leave that out for all. Add `has_caption: false` for a format without a caption, such as a story, so the platform's hashtag rule doesn't apply.

### Edit content pillars

Edit `skills/content_pillars.yaml`. To override the pillars for one org, set `"content_pillars": [{"name": "...", "description": "..."}]` in that org's exported profile and import it (below).

### Edit the content guardrails

Edit `skills/guardrails.yaml`; it applies to every organization from the next message. It has two lists, and its comments explain both:

- `rules`: plain-English rules Mackie follows whenever it writes (ideas, drafts, rewrites, calendar plans), e.g. "Keep politics out."
- `blocked_terms`: words and phrases no post may contain. These are checked in code: a post, edit or calendar idea with one is never saved, and an older post with one can't be approved. They match whole words, ignoring case, spacing and accents ("damn" doesn't block "Amsterdam"); end a term with `*` to block every word that starts with it (`swear*`).

Delete a line to drop a guardrail, or empty a list to turn that part off. For rules that apply to one organization only, tell Mackie in chat ("never mention our old name"): it saves them as that org's preferences.

### Edit what Mackie learned

Export the org to a JSON file, edit it in any text editor (saved as UTF-8), then import it:

```bash
python scripts/export_profile.py "BRCA Strong" -o brca-strong.json
python scripts/import_profiles.py brca-strong.json   # replaces what is stored for the org

# Docker: the file is written to, and read from, this machine
docker compose run --rm -T mackie python scripts/export_profile.py "BRCA Strong" > brca-strong.json
docker compose run --rm -v ./brca-strong.json:/app/brca-strong.json:ro mackie \
  python scripts/import_profiles.py brca-strong.json
```

The file holds the post ledger with any results you reported (`performance`), preferences with when and in what words each was learned (`preference_notes`), voice examples, the website, the org brief and the research.

- Set `"research": null` to make Mackie research the org again next session, and change `"website"` to have it read another site: research reads only the org's own website.
- A post you add without an `id` gets the next free one.
- A file with a mistake (a formatting error, with its line and column, an unknown status, a `scheduled_for` date that isn't `YYYY-MM-DD`) is refused and nothing changes. [Check it](#check-your-edits) first: the check names each file's first mistake without importing anything.
- Importing an org that doesn't exist yet creates it with no members: `grant` the people who work on it.

### Check your edits

```bash
python scripts/validate.py                    # skills/
python scripts/validate.py brca-strong.json   # skills/ and an edited export
docker compose run --rm mackie python scripts/validate.py   # Docker (skills/ only)
```

It lists the platforms, content types and formats it found and how many guardrail rules and blocked terms it read, plus every problem in `skills/` (the content pillars and guardrails files included) and in any profile file you name. It changes nothing and exits with code 1 if anything is wrong. The web UI also reports a broken skill file with a **Heads-up** at the start of each chat.

## Upgrading from the data/ folder

Earlier versions kept each org in `data/{org-id}/profile.json`. Import them once:

```bash
python scripts/import_profiles.py data/*/profile.json
docker compose run --rm -v ./data:/app/data:ro mackie sh -c 'python scripts/import_profiles.py data/*/profile.json'
```

The old files have no time zone, so their times are read in the importing machine's (`TZ` in `.env`, or the server's). Posts, preferences, voice examples and research move over; conversation history (`data/checkpoints.db`) and the old tool traces (`trace.jsonl`) don't, so chats start fresh. The imported organizations have no members yet: `grant` each user the ones they work on.

## Tests and Evaluation

The tests need a disposable PostgreSQL server; each run creates its own databases there (one per worker: the suite runs in four processes) and drops them afterwards. Nothing calls an API or the outside network.

```bash
docker run -d --name mackie-test-db -p 127.0.0.1:55432:5432 -e POSTGRES_PASSWORD=mackie \
  --tmpfs /var/lib/postgresql/data postgres:17-alpine -c fsync=off   # once; or set TEST_DATABASE_URL
pytest                                  # must finish within 30 seconds
pytest -n 0 tests/test_tools.py -k edit # one process: quicker for a few tests, and for a debugger
ruff check . && ruff format --check .   # lint and formatting (settings in ruff.toml)
pytest --cov=agent --cov=scripts --cov=app --cov=main --cov-branch --cov-fail-under=95   # as CI

playwright install --with-deps chromium   # once, for the two opt-in suites below
pytest -m browser             # the real web UI in headless Chromium, with a scripted model
MACKIE_DEPLOYED_PASSWORD=... pytest -m deployed   # a running Compose stack (MACKIE_DEPLOYED_URL, _USERNAME)
```

CI runs all of these on every push and pull request to `main`, the last against a stack it builds from `.env.example`. CLAUDE.md lists what each test file covers.

`scripts/eval.py` runs the conversation above against an org with the configured models, and saves to that org's memory like a normal chat, so use a test org; it costs API calls. It reports three kinds of checks:

- **LLM judge.** Each reply is scored 0–3 on source attribution, voice consistency and not repeating earlier ideas: 21 judge calls, for a total out of 63 (70% or more is a pass).
- **Saved data.** What the run saved, with no LLM: a preference learned by step 3, new posts in step 7, and a source on every suggestion.
- **Memory.** A date mentioned at the start of a long conversation must survive into the summary and be recalled much later.

```bash
python scripts/eval.py --org "BRCA Strong eval" --url https://brcastrong.org
```

## Project Structure

```
app.py              Web UI (Chainlit), with the login
main.py             Terminal UI, with the login
chainlit.md         The web UI's welcome screen
agent/
  config.py         Provider registry and model selection
  assistant.py      System prompt and chat entry point
  graph.py          LangGraph loop: summarize → agent ⇄ tools, PostgreSQL checkpoints
  tools.py          Tool definitions and dispatch, post checks, org brief, calendar
  db.py             PostgreSQL pool, per-org transactions, migrations
  memory.py         Org memory: ledger, preferences, voice examples, research, tool trace
  publishing.py     Connected accounts, publishers (dummies for now), what went out where
  accounts.py       Users, passwords, logins and org access
  errors.py         What users are told when something fails
  logs.py           Log setup, with secrets redacted
  research.py       Website scraping (with SSRF protection) and Tavily search
  skills.py         Loads and validates skills/*.yaml
  untrusted.py      Tags that mark third-party text as data (prompt-injection defense)
skills/             Platforms, content types, formats, content pillars, guardrails (YAML)
migrations/         Database schema, applied in order by scripts/migrate.py
docker/             The database container's first-start script (roles, database)
scripts/
  migrate.py        Creates or updates the schema and the default account
  users.py          Accounts and their org access
  platform_accounts.py   An org's social media accounts, for publishing
  export_profile.py / import_profiles.py   An org's memory as an editable JSON file
  validate.py       Checks skills/ and profile files
  eval.py           End-to-end evaluation
tests/              Tests; tests/browser/ drives the real web UI, tests/deployed/ a running Compose stack
```

See [design_doc.md](design_doc.md) for the architecture and the reasons behind each design choice.

## Known Limitations

- **Password logins only.** No single sign-on (OAuth) and no self-service password reset; an admin runs `scripts/users.py`. A reply already being written when someone's access is removed still finishes.
- **A misspelled organization name starts a new organization.** Check `scripts/users.py list` before granting access to one.
- **One conversation per person, organization and interface (web or terminal).** Two browser tabs of the same person share their conversation and take turns; two terminals, or two servers, don't, and replies sent at the same moment can overwrite each other's history.
- **Publishing is simulated.** `publish_post` records a publication but sends nothing: each platform's publisher is a dummy until a live one replaces it. Publishing happens when you ask, not at a scheduled time, and live Instagram publishing will also need the image or video itself, not just its description.
- **Results are entered by hand.** Mackie doesn't connect to platform analytics and doesn't run paid ads.
- **Static websites only.** Sites rendered by JavaScript yield little text.
- **No image generation.** Mackie describes the visual for each post; your team makes it.
- **Guardrail rules guide the model; only blocked terms are enforced.** A model can still miss a rule such as "keep politics out", so read every draft before approving it. Blocked terms are matched ignoring case, spacing, accents and look-alike forms such as fullwidth letters, but a misspelling, symbols in place of letters ("d@mn") or a look-alike letter from another alphabet (a Cyrillic "а") gets through.

The design doc's Roadmap describes how each would be addressed.
