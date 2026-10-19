# Mackie — guide for AI coding assistants

Mackie is a conversational AI social media manager for nonprofit marketing staff, modelled on Vee's Maggie: it **plans** (dated, campaign-aware content calendars, two posts a week), **creates** (feed posts, carousels, stories and reels for LinkedIn, Instagram and Facebook, supporter email, a visual idea per post) and **optimizes** (records reported results and suggests what to post next from what worked). It researches the org first, learns its voice from feedback across sessions, and tracks every post in a ledger. It is proactive (each session leads with the agenda of what needs the user) and nothing goes live without the user's approval. Publishing, graphics and paid ads stay with the org's staff. User docs: `README.md`. Architecture and the reasons behind each choice: `design_doc.md`.

**Naming.** The assistant is called **Mackie** everywhere: code, UI and docs.

## Commands

The project uses Python 3.12 with a virtualenv in `.venv`. There is no bare `python` on this machine; use `.venv/bin/python`. It runs on Linux, macOS, WSL or Docker: the org lock uses `fcntl.flock`, which native Windows lacks.

```bash
.venv/bin/python -m pytest                 # full suite, offline, a few seconds
.venv/bin/python -m pytest --cov=agent --cov=scripts --cov=app --cov=main --cov-branch   # coverage (CI floor: 95%)
.venv/bin/python -m pytest -m browser      # real web UI in headless Chromium, scripted model, ~15s
.venv/bin/ruff check . && .venv/bin/ruff format .   # lint, then format (settings: ruff.toml)
.venv/bin/chainlit run app.py              # web UI on :8000
.venv/bin/python main.py                   # terminal UI
.venv/bin/python scripts/validate.py       # check skills/ and data/*/profile.json
.venv/bin/python scripts/eval.py --org "BRCA Strong" --url https://brcastrong.org   # live, costs API calls
docker compose up --build
```

`requirements.txt` holds the runtime dependencies and `requirements-dev.txt` adds pytest, pytest-cov, ruff and playwright (`playwright install chromium` once). CI (`.github/workflows/ci.yml`) runs ruff, the tests (within the time budget), a branch-coverage floor of 95% and the browser tests on every push and pull request.

## Code Map

| File | Owns |
|---|---|
| `app.py` / `main.py` | Web UI (Chainlit) and terminal UI: org onboarding, streaming replies, tool steps |
| `agent/config.py` | Provider registry `PROVIDERS` and model selection: `resolve_model`, `resolve_fallbacks`, `resolve_fast_model`, `resolve_judge_model` (`FAST_PROVIDER`, `JUDGE_PROVIDER`), `connection_kwargs`, `background_kwargs` |
| `agent/assistant.py` | System prompt as (stable, per-turn) blocks; `Assistant.chat` / `achat_stream` / `clear_history` |
| `agent/graph.py` | LangGraph `summarize → agent ⇄ tools` loop plus `save_check` (unsaved post ids), sliding window, running summary, streaming, provider fallback, SQLite checkpoints (WAL, schema set up once, `prune_history` keeps one checkpoint per thread) |
| `agent/tools.py` | `TOOL_DEFINITIONS` (OpenAI format), `handle_tool_call` / `_dispatch`, `save_post` checks, argument validation (incl. `record_performance` counts), org brief, calendar, `trace.jsonl` |
| `agent/memory.py` | `profile.json` I/O (atomic write + `.bak`, `ProfileError` with section type checks, the per-org `locked()` lock file), `slugify` and the org-id guard, ledger, post results (`record_performance`, `top_performing`), preferences, voice examples |
| `agent/research.py` | `scrape_site` with the SSRF guard (`_is_safe_url`, `_resolves_to_internal`, manual redirects), Tavily `search_web`, `research_org` |
| `agent/skills.py` | Loads and validates `skills/*.yaml` (platforms, content types, formats) and the content pillars file; never raises on a bad file |
| `agent/untrusted.py` | `wrap_untrusted`: trust-boundary tags around third-party text |
| `scripts/validate.py` | Plain-English check of the files non-engineers edit |
| `scripts/eval.py` | Live 7-step run, 21 parallel judge calls, saved-state checks on `profile.json` |
| `tests/browser/serve.py` | The real web UI with a scripted model and scratch data, for the opt-in browser tests |
| `.chainlit/config.toml` | Web UI settings; file uploads are off because nothing reads them (`tests/test_deploy.py`, which also makes one real offline LiteLLM tool call) |

## Invariants: don't break these

- **LiteLLM only.** Every LLM call goes through `litellm.acompletion` with `**connection_kwargs(model)`, or `**background_kwargs(model)` for summaries, the brief, the calendar and the judge. Never import a provider SDK. A new provider is one `PROVIDERS` entry.
- **Isolation.** Conversation history is keyed by `thread_id`. The web UI uses `{org_id}:{uuid}` and the CLI uses `{org_id}`. Org ids are `[a-z0-9-]` only (`slugify`), so a `:` can never collide with an org id. `_org_dir` refuses any other id, so nothing can reach outside `data/`.
- **Untrusted text is wrapped.** Scraped pages, search snippets, the org brief and the summary go through `wrap_untrusted()`; never build these tags with an f-string.
- **No fetch skips the SSRF check.** Every fetch goes through `_fetch_html`, which checks every redirect hop with `_is_safe_url` and the DNS lookup.
- **Nothing waits forever.** `connection_kwargs` gives every LLM call `LLM_TIMEOUT_S`; page fetches use `FETCH_TIMEOUT_S`. Test streaming and isolation with gated streams that deadlock if output is buffered, not with sleeps.
- **Tools never raise.** `handle_tool_call` turns exceptions into an `"Error: …"` result. A raised exception leaves an unanswered tool call in the checkpoint and breaks the thread.
- **Hand-edited files never crash the app.** Skill files with a missing or wrongly typed field are skipped and reported. `profile.json` problems, including a section of the wrong type, raise `ProfileError` with a plain message. Read fields of ledger posts, voice examples and sources with `.get()`.
- **Prompt blocks.** Content that changes per turn goes in the per-turn block, never the stable (cached) block.
- **Writes hold the org lock.** Every load-change-save of `profile.json` runs inside `mem.locked(org_id)` (the mutating `memory` functions take it themselves), with no `await` inside, so the web UI and CLI can write to one org from separate processes.
- **`data/` holds real user data.** Tests must use the `tmp_data_dir` fixture; never write to the real `data/` from a test.

## Extending

- **Platform, content type or format:** add a YAML file to `skills/` (see the README). No code change.
- **Tool:**
  1. Add a schema to `TOOL_DEFINITIONS`.
  2. Add a branch to `_dispatch()` in `agent/tools.py`; validate arguments the model might get wrong.
  3. Add tests to `tests/test_tools.py`.
  4. Mention it in the working rules in `_STABLE_PROMPT_TEMPLATE` if the model needs guidance.
- **Provider:** add a `_Provider` to `PROVIDERS`, a line to `.env.example`, and a row to the README's configuration table. Its place in `PROVIDERS` is its priority: update the documented order in the README, `.env.example` and `test_providers_are_chosen_in_registry_order` (which also checks its default model), and add a test for anything special, such as an endpoint.
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
- **Dependencies:** must be actively maintained and widely used; pin a minimum version (`pkg>=x.y`). Exclude a known-broken release with `!=` and say why in `design_doc.md` (LiteLLM 1.106.0 breaks tool calls).
- **Tests:**
  - every change ships with tests;
  - every public function in `agent/` gets a happy-path test and an error or edge-case test;
  - security fixes get a regression test;
  - each behavior has one owner: details live with their module's tests, cross-cutting checks in their own file (below). Before adding a test, search for one that already covers it; add a case to an existing parametrized test rather than a near-copy, and test at another layer only when that layer can break on its own (e.g. `test_ui.py` with a fake Chainlit and `tests/browser/` with the real one);
  - cross-cutting checks have their own files: `test_security.py` (SSRF bypasses, every tool fuzzed with malformed arguments, org-id traversal through every entry point, org isolation, injection end to end, trace contents), `test_performance.py` (parallel I/O, prompt size that doesn't grow with memory, linear-time parsing) `test_ui.py` (`app.py` and `main.py`, UX), `test_sessions.py` (concurrent sessions, users on one org, interleaved streams, separate processes) and `tests/browser/` (opt-in `-m browser`: the real web UI in Chromium, scripted by `serve.py`; phone layout, streaming, two browsers). Responsiveness and scale live in `test_performance.py`. A new tool is fuzzed automatically; a new `asyncio.gather` gets a `concurrency_probe` test.
- **Offline tests.** Autouse fixtures in `tests/conftest.py` fail any unmocked LLM call or HTTP request and stub DNS. Mock `agent.graph.litellm.acompletion` with an async generator of stream chunks (see `_text_stream` in the tests). Mock `agent.research.httpx.AsyncClient` as an async context manager. The suite must stay under 10 seconds; `conftest.py` fails a slower run (except under coverage, and for the opt-in browser tests). Prove parallelism with the `concurrency_probe` fixture, never with wall-clock timing.
- **Docs:** when behavior changes, update `README.md` (user-facing) and `design_doc.md` (decisions: what, why, trade-offs) in the same change. Keep both concise.
