# Mackie — guide for AI coding assistants

Mackie is a conversational AI social media manager for nonprofit marketing staff, modelled on Vee's Maggie: it **plans** (dated, campaign-aware content calendars, two posts a week), **creates** (feed posts, carousels, stories and reels for LinkedIn, Instagram and Facebook, supporter email, a visual idea per post) and **optimizes** (records reported results and suggests what to post next from what worked). It researches the org first, learns its voice from feedback across sessions, and tracks every post in a ledger. It is proactive (each session leads with the agenda of what needs the user) and nothing goes live without the user's approval. Publishing, graphics and paid ads stay with the org's staff. User docs: `README.md`. Architecture and the reasons behind each choice: `design_doc.md`.

**Naming.** The assistant is called **Mackie** everywhere: code, UI and docs.

## Commands

The project uses Python 3.12 with a virtualenv in `.venv`. There is no bare `python` on this machine; use `.venv/bin/python`. All data lives in PostgreSQL (15+). The tests need a disposable server at `TEST_DATABASE_URL` (default `127.0.0.1:55432`, user `postgres`, password `mackie`); each run creates and drops its own database there.

```bash
docker run -d --name mackie-test-db -p 127.0.0.1:55432:5432 -e POSTGRES_PASSWORD=mackie \
  --tmpfs /var/lib/postgresql/data postgres:17-alpine -c fsync=off   # test database, once
.venv/bin/python -m pytest                 # full suite, no outside network
.venv/bin/python -m pytest --cov=agent --cov=scripts --cov=app --cov=main --cov-branch --cov-fail-under=95   # coverage, as CI
.venv/bin/python -m pytest -m browser      # real web UI in headless Chromium, scripted model, ~30s
MACKIE_DEPLOYED_PASSWORD=... .venv/bin/python -m pytest -m deployed   # a running `docker compose` stack
.venv/bin/ruff check . && .venv/bin/ruff format .   # lint, then format (settings: ruff.toml)
.venv/bin/chainlit run app.py              # web UI on :8000
.venv/bin/python main.py                   # terminal UI
.venv/bin/python scripts/migrate.py        # create/update the schema (MACKIE_MIGRATION_URL)
.venv/bin/python scripts/users.py add NAME # accounts: add, passwd, remove, grant, revoke, list
.venv/bin/python scripts/validate.py       # check skills/ (and profile JSON files given as arguments)
.venv/bin/python scripts/eval.py --org "BRCA Strong eval" --url https://brcastrong.org   # live, costs API calls
docker compose up --build
```

`requirements.txt` holds the runtime dependencies and `requirements-dev.txt` adds pytest, pytest-asyncio, pytest-cov, ruff (pinned exactly, so CI formats as you do) and playwright (`playwright install --with-deps chromium` once). CI (`.github/workflows/ci.yml`, read-only token, a timeout per job) runs ruff, the tests (within the time budget), a branch-coverage floor of 95%, the browser tests, and a deployment job (builds and starts the Compose stack, checks its settings, logs in with the default account) on every push and pull request to `main`.

## Code Map

| File | Owns |
|---|---|
| `app.py` / `main.py` | Web UI (Chainlit) and terminal UI: login, org onboarding (only orgs the user may open), streaming replies, tool steps; `app.restrict_media` (the Content-Security-Policy for images, video and audio), `app.refuse_large_requests` (no uploads, request bodies up to 1 MB) |
| `agent/config.py` | Provider registry `PROVIDERS` and model selection: `resolve_model`, `resolve_fallbacks`, `resolve_fast_model`, `resolve_judge_model` (`FAST_PROVIDER`, `JUDGE_PROVIDER`), `connection_kwargs`, `background_kwargs`; `LLM_TIMEOUT_S`, `BACKGROUND_RETRIES` |
| `agent/assistant.py` | System prompt as (stable, per-turn) blocks; `Assistant.for_user` (the per-user thread) / `chat` / `achat_stream` / `recent_exchanges` / `clear_history`, `MAX_MESSAGE_CHARS`; `agenda_summary` and `count_noun` for the greetings |
| `agent/graph.py` | LangGraph `summarize → agent ⇄ tools` loop plus `save_check` (unsaved post ids), at most `MAX_TOOL_ROUNDS` per turn, sliding window, running summary, streaming, provider fallback, `OrgCheckpointer` (PostgreSQL checkpoints limited to one org's threads; `prune_history` keeps one checkpoint per thread) |
| `agent/tools.py` | `TOOL_DEFINITIONS` (OpenAI format, `show_post` among them), `handle_tool_call` / `_dispatch`, `save_post` checks, argument validation (incl. `record_performance` counts), research only of the org's stored website (`_research_url`), `Turn` (what the user typed and was shown) for preferences, voice examples and approvals, org brief, calendar, the tool-call trace |
| `agent/db.py` | Connection pool per event loop (timeouts; refuses a superuser role or an old schema), `org_transaction` (row-level security scope), `migrate`, `check_org_id` |
| `agent/memory.py` | Org memory in PostgreSQL (database calls `async`): the profile dict (`load_profile`, `save_profile`), `profile_problem` / `read_profile_file` (`ProfileError`) for edited exports, `slugify`, `is_date` / `is_count`, ledger, post results (`record_performance`, `top_performing`), preferences, voice examples, research, trace |
| `agent/accounts.py` | Users (scrypt passwords, lockout, `User.session_key`, which a password change renews), memberships, `open_org` (`MACKIE_USERS_CREATE_ORGS`), `has_access`, `ensure_user` |
| `agent/errors.py` | `user_message`: what users are told when a turn or session fails (database, each kind of provider failure, no provider set up, a reply cut off mid-stream, too many tool rounds, anything else); an error's own text never reaches the screen, except Mackie's own `DatabaseError` setup messages and `MessageTooLong` |
| `agent/logs.py` | `configure_logging` (`MACKIE_LOG_LEVEL`), `RedactSecrets`, `safe` for user-supplied values in log lines |
| `migrations/*.sql` | Schema, applied in order by `db.migrate`; row-level security on every org table |
| `agent/research.py` | `website_url` (what counts as a typed website), `scrape_site` with the SSRF guard (`_is_safe_url`, `_resolves_to_internal`, manual redirects) and a time and size limit per page, Tavily `search_web`, `research_org` |
| `agent/skills.py` | Loads and validates `skills/*.yaml` (platforms, content types, formats) and the content pillars file; never raises on a bad file |
| `agent/untrusted.py` | `wrap_untrusted`: trust-boundary tags around third-party text |
| `scripts/validate.py` | Plain-English check of the files non-engineers edit |
| `scripts/migrate.py`, `users.py`, `export_profile.py`, `import_profiles.py` | Schema, accounts and access, an org's memory as editable JSON (also moves old `data/`) |
| `scripts/eval.py` | Live 7-step run, 21 parallel judge calls, saved-state checks, the long-conversation memory check |
| `tests/browser/serve.py` | The real web UI with a scripted model, test accounts and the test database, for the opt-in browser tests |
| `tests/llm_fakes.py` | Fake model responses for the tests: `text_stream`, `tool_call_stream`, `completion`, `gated_stream` |
| `Dockerfile`, `docker-compose.yml`, `docker/postgres-init.sh` | The image (non-root), the stack (PostgreSQL unpublished, `migrate`, `mackie`) and the database's first-start roles |
| `.chainlit/config.toml`, `chainlit.md` | Web UI settings (file uploads off, as nothing reads them; message edits off, as the stored conversation can't be rewound; checked by `tests/test_deploy.py`) and the welcome screen |

## Invariants: don't break these

- **LiteLLM only.** Every LLM call goes through `litellm.acompletion` with `**connection_kwargs(model)`, or `**background_kwargs(model)` for summaries, the brief, the calendar and the judge. Never import a provider SDK. A new provider is one `PROVIDERS` entry.
- **Isolation.** Every org table, the checkpoint tables included, has forced row-level security on `app.org_id`; the one exception is `memberships`, which accounts read across orgs and the app reaches only through `accounts.py`. Reach the database only through `db.org_transaction(org_id)` (org data) or `db.transaction()` (accounts), never `pool.connection()` directly; a new table with org rows gets a policy in its migration (`test_security.py` fails otherwise). Mackie connects as a role that can't change the schema. Conversation history is keyed by `thread_id`, named by `Assistant.for_user` (only `scripts/eval.py` names its own, `{org_id}:eval-…`): `{org_id}:web-user-{user_id}` or `{org_id}:cli-user-{user_id}` (one resumable conversation per user, org and UI; turns of one thread take turns through `assistant._thread_lock`, within one process); `Assistant` refuses another org's thread. Org ids are `[a-z0-9-]` only (`slugify`, `db.check_org_id`), so a `:` can never collide with an org id. A user opens only orgs they belong to (`accounts.open_org`), re-checked with their session key before every message, so removed access or a changed password ends open sessions.
- **Untrusted text is wrapped.** Scraped pages, search snippets and titles (the research sources in every prompt included), the org brief and the summary go through `wrap_untrusted()`; never build these tags with an f-string.
- **The model can't send data out or plant instructions.** Assume a fooled model: research reads only the org's stored website, which only the user sets, judged as it will be fetched (`tools._research_url`, `research.website_url`); web pages load images, video and audio only from Mackie (`app.restrict_media`). Anything that outlasts the turn traces back to the user through the `tools.Turn` the graph passes each call: a preference (quote, value, key) and a voice example use the user's typed words, forgetting one needs them to name it, results are recorded only for posts they named or saw, and only a post Mackie itself displayed to them (`show_post`, shown escaped by `tools.post_display`; never judge what the model's Markdown shows), or that they wrote, can go live. Text the model saves that the prompt shows (titles, pillars, platforms, notes) is capped. A new tool that writes standing memory takes the same checks. A new tool that fetches a URL or a new way to render model output needs the same care.
- **No fetch skips the SSRF check.** Every fetch goes through `_fetch_html`, which checks every redirect hop with `_is_safe_url` and the DNS lookup.
- **Nothing waits forever.** `connection_kwargs` gives every LLM call `LLM_TIMEOUT_S`; a whole page fetch (redirects included) and a web search each get `FETCH_TIMEOUT_S`, and a page is read, and unpacked by Mackie, up to `MAX_PAGE_BYTES`; a turn stops after `graph.MAX_TOOL_ROUNDS` tool rounds; database connections have connect, statement, lock and idle-transaction timeouts, and migrations connect and lock timeouts (`db.py`). Test streaming and isolation with gated streams that deadlock if output is buffered, not with sleeps.
- **Tools never raise.** `handle_tool_call` turns exceptions into an `"Error: …"` result. A raised exception leaves an unanswered tool call in the checkpoint and breaks the thread.
- **Hand-edited files never crash the app.** Skill files with a missing or wrongly typed field are skipped and reported. An edited profile export with a mistake, including a section of the wrong type, raises `ProfileError` with a plain message and is not imported. Read fields of ledger posts, voice examples and sources with `.get()`: imported posts can lack fields.
- **Logging.** Log through `logging.getLogger(__name__)`; `agent/logs.py` configures output. Log security events and every failure you absorb; never log secrets or what users write (messages, posts, tool arguments: the trace has those). Pass any user-supplied value through `logs.safe()`.
- **Prompt blocks.** Content that changes per turn goes in the per-turn block, never the stable (cached) block.
- **Writes lock the org's row.** Every read-change-write runs in one `org_transaction` that locks the org's row first (`memory._lock_org`), so any number of processes and servers can write to one org; a change to one post's status locks just that post's row. A check that must hold against parallel writes (a size cap) runs inside that transaction (`save_preference(refuse=...)`). Never wait on the model inside a transaction.
- **Tests never touch your own database.** A test that uses the database requests the `database` fixture (a fresh, empty test database reached as the app role); without it, database access fails. Parameterize every query; never build SQL from values with an f-string.

## Extending

- **Platform, content type or format:** add a YAML file to `skills/` (see the README). No code change.
- **Tool:**
  1. Add a schema to `TOOL_DEFINITIONS`.
  2. Add a branch to `_dispatch()` in `agent/tools.py`; validate arguments the model might get wrong.
  3. Add tests to `tests/test_tools.py`.
  4. Mention it in the working rules in `_STABLE_PROMPT_TEMPLATE` if the model needs guidance.
- **Provider:** add a `_Provider` to `PROVIDERS`, a line to `.env.example`, and a row to the README's configuration table. Its place in `PROVIDERS` is its priority: update the documented order in the README, `.env.example` and `test_providers_are_chosen_in_registry_order` (which also checks its default model), and add a test for anything special, such as an endpoint.
- **Memory field:** add a migration (`migrations/NNN_name.sql`, the next number; never edit an applied one), read and write it in `memory.py`, give `default_profile` its default, and check its type in `profile_problem` if the prompt reads it.

## Standards

- **Formatting:** `ruff format` and `ruff check` with `ruff.toml`; CI fails on either.
- **Simplicity:** the simplest code that solves the problem; one job per function; descriptive names; named constants instead of magic values. No dead code, commented-out code or unused imports. Prefer the core done well over extra features.
- **Types and docs:**
  - type hints on every signature;
  - a module docstring saying what the module owns and what it doesn't;
  - a docstring on every public function;
  - inline comments only for a non-obvious *why*.
- **Parallelism:** run independent I/O with `asyncio.gather`. Research, subpages, judge calls and calendar weeks already do.
- **Dependencies:** must be actively maintained and widely used; pin a minimum version (`pkg>=x.y`) that has what the code uses and no known vulnerability. Exclude a known-broken release with `!=` and say why in `design_doc.md` (LiteLLM 1.106.0 breaks tool calls). ruff is the one exact pin.
- **Tests:**
  - every change ships with tests;
  - every public function in `agent/` gets a happy-path test and an error or edge-case test;
  - security fixes get a regression test;
  - each behavior has one owner: details live with their module's tests, cross-cutting checks in their own file (below). Before adding a test, search for one that already covers it; add a case to an existing parametrized test rather than a near-copy, and test at another layer only when that layer can break on its own (e.g. `test_ui.py` with a fake Chainlit and `tests/browser/` with the real one);
  - cross-cutting checks have their own files: `test_security.py` (SSRF bypasses, every tool fuzzed with malformed arguments, hostile org ids through every entry point, org isolation incl. row-level security and the app role's limits, SQL injection, injection end to end, trace contents), `test_performance.py` (parallel I/O, responsiveness, scale: prompt size that doesn't grow with memory, linear-time parsing), `test_ui.py` (`app.py` and `main.py`, UX), `test_logging.py` (what every module logs and never logs; `test_logs.py` unit-tests `agent/logs.py`), `test_errors.py` (what users are told when something fails), `test_sessions.py` (concurrent sessions, users on one org, interleaved streams, concurrent writers, separate processes), `tests/browser/` (opt-in `-m browser`: the real web UI in Chromium, scripted by `serve.py`; login, phone layout, streaming, two users), `tests/deployed/` (opt-in `-m deployed`: a running Compose stack) and `test_deploy.py` (deployment and CI settings, and one real offline LiteLLM tool call). A new tool is fuzzed automatically; a new `asyncio.gather` gets a `concurrency_probe` test.
- **Offline tests.** Autouse fixtures in `tests/conftest.py` fail any unmocked LLM call or HTTP request and stub DNS. Mock `agent.graph.litellm.acompletion` with an async generator of stream chunks (`text_stream` / `tool_call_stream` / `completion` in `tests/llm_fakes.py`). Mock `agent.research.httpx.AsyncClient` as an async context manager whose `stream()` serves the page (`_fake_client` in `test_research.py`). Every setting `.env` defines is blanked for the run, and passwords are hashed at a low cost (`cheap_hashing`). Import a script with `tests.conftest.load_script`. The suite must stay under 30 seconds (it runs every test against a real PostgreSQL): `conftest.py` fails a slower run, 60 s on CI through `MACKIE_TEST_BUDGET_S`, since shared runners vary; runs under coverage or with the opt-in browser or deployed tests are exempt. Prove parallelism with the `concurrency_probe` fixture, never with wall-clock timing.
- **Docs:** when behavior changes, update `README.md` (user-facing) and `design_doc.md` (decisions: what, why, trade-offs) in the same change. Keep both concise.
