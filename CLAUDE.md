# Mackie — guide for AI coding assistants

Mackie is a conversational social media assistant for nonprofit marketing staff. It researches an org, suggests and drafts posts for LinkedIn, Instagram and Facebook, learns the org's voice from feedback across sessions, and tracks every post in a ledger. User docs: `README.md`. Architecture and the reasons behind each choice: `design_doc.md`.

**Naming.** The assistant is called **Mackie** everywhere: code, UI and docs.

## Commands

The project uses Python 3.12 with a virtualenv in `.venv`. There is no bare `python` on this machine; use `.venv/bin/python`.

```bash
.venv/bin/python -m pytest                 # full suite, offline, a few seconds
.venv/bin/ruff check . && .venv/bin/ruff format .   # lint, then format (settings: ruff.toml)
.venv/bin/chainlit run app.py              # web UI on :8000
.venv/bin/python main.py                   # terminal UI
.venv/bin/python scripts/validate.py       # check skills/ and data/*/profile.json
.venv/bin/python scripts/eval.py --org "BRCA Strong" --url https://brcastrong.org   # live, costs API calls
docker compose up --build
```

`requirements.txt` holds the runtime dependencies and `requirements-dev.txt` adds pytest and ruff. CI (`.github/workflows/ci.yml`) runs ruff and the tests on every push and pull request.

## Code Map

| File | Owns |
|---|---|
| `app.py` / `main.py` | Web UI (Chainlit) and terminal UI: org onboarding, streaming replies, tool steps |
| `agent/config.py` | Provider registry `PROVIDERS` and model selection: `resolve_model`, `resolve_fallbacks`, `resolve_fast_model`, `connection_kwargs`, `background_kwargs` |
| `agent/assistant.py` | System prompt as (stable, per-turn) blocks; `Assistant.chat` / `achat_stream` / `clear_history` |
| `agent/graph.py` | LangGraph `summarize → agent ⇄ tools` loop plus `save_check` (unsaved post ids), sliding window, running summary, streaming, provider fallback, SQLite checkpoints |
| `agent/tools.py` | `TOOL_DEFINITIONS` (OpenAI format), `handle_tool_call` / `_dispatch`, `save_post` checks, argument validation, org brief, calendar, `trace.jsonl` |
| `agent/memory.py` | `profile.json` I/O (atomic write + `.bak`, `ProfileError` with section type checks), `slugify` and the org-id guard, ledger, preferences, voice examples |
| `agent/research.py` | `scrape_site` with the SSRF guard (`_is_safe_url`, `_resolves_to_internal`, manual redirects), Tavily `search_web`, `research_org` |
| `agent/skills.py` | Loads and validates `skills/*.yaml` and content pillars; never raises on a bad file |
| `agent/untrusted.py` | `wrap_untrusted`: trust-boundary tags around third-party text |
| `scripts/validate.py` | Plain-English check of the files non-engineers edit |
| `scripts/eval.py` | Live 7-step run, 21 parallel judge calls, saved-state checks on `profile.json` |
| `.chainlit/config.toml` | Web UI settings; file uploads are off because nothing reads them (`tests/test_deploy.py`) |

## Invariants: don't break these

- **LiteLLM only.** Every LLM call goes through `litellm.acompletion` with `**connection_kwargs(model)`, or `**background_kwargs(model)` for summaries, the brief, the calendar and the judge. Never import a provider SDK. A new provider is one `PROVIDERS` entry.
- **Isolation.** Conversation history is keyed by `thread_id`. The web UI uses `{org_id}:{uuid}` and the CLI uses `{org_id}`. Org ids are `[a-z0-9-]` only (`slugify`), so a `:` can never collide with an org id. `_org_dir` refuses any other id, so nothing can reach outside `data/`.
- **Untrusted text is wrapped.** Scraped pages, search snippets, the org brief and the summary go through `wrap_untrusted()`; never build these tags with an f-string.
- **No fetch skips the SSRF check.** Every fetch goes through `_fetch_html`, which checks every redirect hop with `_is_safe_url` and the DNS lookup.
- **Tools never raise.** `handle_tool_call` turns exceptions into an `"Error: …"` result. A raised exception leaves an unanswered tool call in the checkpoint and breaks the thread.
- **Hand-edited files never crash the app.** Skill files with a missing or wrongly typed field are skipped and reported. `profile.json` problems, including a section of the wrong type, raise `ProfileError` with a plain message. Read fields of ledger posts, voice examples and sources with `.get()`.
- **Prompt blocks.** Content that changes per turn goes in the per-turn block, never the stable (cached) block.
- **`data/` holds real user data.** Tests must use the `tmp_data_dir` fixture; never write to the real `data/` from a test.

## Extending

- **Platform or content type:** add a YAML file to `skills/` (see the README). No code change.
- **Tool:**
  1. Add a schema to `TOOL_DEFINITIONS`.
  2. Add a branch to `_dispatch()` in `agent/tools.py`; validate arguments the model might get wrong.
  3. Add tests to `tests/test_tools.py`.
  4. Mention it in the working rules in `_STABLE_PROMPT_TEMPLATE` if the model needs guidance.
- **Provider:** add a `_Provider` to `PROVIDERS`, a line to `.env.example`, a test to `tests/test_config.py`, and a row to the README's configuration table.
- **Memory field:** add its default to `load_profile`, so older profiles get it, and its type to `_section_problem` if the prompt reads it.

## Standards

- **Formatting:** `ruff format` and `ruff check` with `ruff.toml`; CI fails on either.
- **Simplicity:** the simplest code that solves the problem; one job per function; descriptive names; named constants instead of magic values. No dead code, commented-out code or unused imports. Prefer the core done well over extra features.
- **Types and docs:**
  - type hints on every signature;
  - a module docstring saying what the module owns and what it doesn't;
  - a docstring on every public function;
  - inline comments only for a non-obvious *why*.
- **Parallelism:** run independent I/O with `asyncio.gather`. Research, subpages, judge calls and calendar weeks already do.
- **Dependencies:** must be actively maintained and widely used; pin a minimum version (`pkg>=x.y`).
- **Tests:**
  - every change ships with tests;
  - every public function in `agent/` gets a happy-path test and an error or edge-case test;
  - security fixes get a regression test.
- **Offline tests.** Autouse fixtures in `tests/conftest.py` fail any unmocked LLM call or HTTP request and stub DNS. Mock `agent.graph.litellm.acompletion` with an async generator of stream chunks (see `_text_stream` in the tests). Mock `agent.research.httpx.AsyncClient` as an async context manager. The suite must stay under 10 seconds.
- **Docs:** when behavior changes, update `README.md` (user-facing) and `design_doc.md` (decisions: what, why, trade-offs) in the same change. Keep both concise.
