# Mackie — Design Doc

## What It Is

Mackie is a conversational assistant for nonprofit marketing staff. It researches an organization, then suggests and drafts on-brand posts for LinkedIn, Instagram and Facebook. It learns from feedback (corrections in the user's own words, approved posts, pasted posts) and remembers it across sessions. Every suggestion and draft goes into a post ledger with a status.

**Goals**

- Content is grounded in research, and every suggestion names its source.
- Corrections stick within the session and across sessions.
- A non-engineer can add a platform or content type, and view or edit what Mackie learned, without a code change or a redeploy.
- Switching LLM providers is a configuration change.
- Each conversation's history is isolated.

**First run.** The user gives the org's name and website and confirms them. The first content request triggers `research_org`, which:

1. reads the homepage and up to three About/Programs-style pages;
2. runs three web searches in parallel;
3. writes a short cited **org brief** that stays in the prompt in every future session.

Corrections ("too corporate, we're more grassroots") are saved with `save_preference` and appear in the prompt from the next turn on.

---

## Architecture

```
User ──► app.py (Chainlit web UI)  or  main.py (terminal)
              │
              ▼
         Assistant (agent/assistant.py)
              │  system prompt rebuilt every turn from profile.json and skills/:
              │    stable block   — identity, org brief, sources, pillars, platforms, content types, rules
              │    per-turn block — date, preferences, voice examples, recent ledger, conversation summary
              ▼
         LangGraph (agent/graph.py) — AgentState checkpointed in SQLite per thread_id
              │  summarize → agent ⇄ tools   (tools → summarize → agent)
              │  agent → save_check → summarize → agent   (at most once per turn)
              │  model sees the last 40 messages, starting at a user turn;
              │  older messages are folded into a running summary
              ▼
         LiteLLM ──► main provider, then fallbacks (agent/config.py)
              │
              └─ 10 tools (agent/tools.py):
                   research_org, refresh_research, web_search      → agent/research.py
                   save_post, update_post_status, search_ledger,
                   save_preference, forget_preference,
                   save_voice_example                              → agent/memory.py
                   generate_calendar                               → 4 parallel LLM calls

Storage:  data/{org-id}/profile.json  (+ .bak)  org memory: research, brief, preferences, voice, ledger
          data/{org-id}/trace.jsonl            one line per tool call
          data/checkpoints.db                  conversation history, keyed by thread_id
Config:   skills/*.yaml                         platforms, content types, content pillars
```

Each user message runs the loop until the model replies without calling a tool. If that reply shows post ids that were never saved, `save_check` sends one follow-up (see Checks enforced in code). Text streams token by token; each tool call appears as a step in the web UI when it finishes.

---

## Decisions

### LLM access: LiteLLM behind one provider registry

**What.** Every LLM call goes through `litellm.acompletion()`. `agent/config.py` holds a registry (`PROVIDERS`) of ten named providers, plus a `custom` entry that accepts any LiteLLM model string. The ten are Anthropic, OpenAI, xAI, Google, Mistral, Groq, DeepSeek, OpenRouter, OpenCode Zen and self-hosted Qwen on vLLM. A provider is configured when its required variables are set. The first configured provider in registry order is the main model, or the one named by `LLM_PROVIDER`. `connection_kwargs(model)` supplies `api_base` and `api_key` for OpenAI-compatible endpoints.

**Why.** Swapping models must not need code changes, and nonprofits may need free or self-hosted models. The alternative, provider SDKs called directly in each module, spreads provider logic everywhere. Adding a provider is one registry entry.

**Reasoning models.** Background calls (summary, org brief, calendar, eval judge) use `background_kwargs(model)`, which turns reasoning down for providers that need it:

- OpenCode Zen gets `reasoning_effort="low"`. Its brief needed 2,112 thinking tokens at the default effort and 59 at low.
- Qwen 3 has thinking switched off. Without that, it never finished the brief.

Token limits on those calls are ceilings, not targets. User-facing replies keep full reasoning. If a reply has neither text nor a tool call, it is retried once with reasoning turned down, and then a short message is shown instead of a blank reply.

### Provider fallback and cost routing

**What.** `resolve_fallbacks()` lists the models of every other configured provider. `_open_stream()` tries them in order when a request fails to *start* (outage, rate limit, bad key). An error after tokens have streamed is not retried, because part of the reply is already on screen. `FAST_MODEL` routes background work to a cheaper model. `JUDGE_MODEL` picks the eval judge.

**Why.** A second API key is enough to survive one provider's outage. LiteLLM's `Router` was the alternative, but it would send Anthropic-format system blocks to other providers. `_system_content()` builds the system message per model instead.

### Agent loop: LangGraph with SQLite checkpoints

**What.** A four-node `StateGraph` (`summarize`, `agent`, `tools`, `save_check`), checkpointed by `AsyncSqliteSaver` in `data/checkpoints.db`.

**Why.** History survives restarts, and typed state with an explicit router (`should_continue`) stays easy to debug. LangChain's AgentExecutor was rejected because it hides its retries and callbacks.

**Session isolation.**

- **Web UI:** each browser session gets the thread `{org_id}:{uuid}`. A `:` can't occur in an org id, so a session thread can never equal another org's default thread.
- **CLI:** resumes the thread `{org_id}`. Its `reset` command deletes that thread's checkpoints (`Assistant.clear_history()`); org memory is kept.

**Limitation.** SQLite is single-process, so several replicas would need `langgraph-checkpoint-postgres`.

### Context management

**What.**

- **Ledger:** the prompt lists the 20 most recent posts. Older ones are reachable with `search_ledger`.
- **Messages:** the model sees the last 40, starting at the first user message in that slice (`_window_start()`). A plain slice can start on a tool result whose tool call was cut off, and providers reject that. If one turn is longer than 40 messages, the whole turn is kept.
- **Running summary:** messages that leave the window are folded into a summary of 300 words or less, one incremental LLM call each time the window moves (`summarize` node). It goes into the per-turn prompt block. If the call fails, the turn continues and the same messages are retried next time.
- **Full history:** stays in the checkpoint; nothing is deleted.

**Why.** Without the summary, anything that lived only in old messages was lost, such as the scraped research, which exists only in tool results. Durable facts don't depend on the window at all: the org brief, preferences and ledger are in `profile.json` and rebuilt into every prompt.

**Trade-offs.** The summary is lossy and costs an extra call per turn once a conversation is long. A vector store for retrieving old turns is the next step if that matters.

### Prompt caching (Anthropic only)

**What.** The system prompt is two blocks:

- **Stable:** identity, org brief, sources, pillars, skills and rules. It changes only when research or skill files change.
- **Per-turn:** date, preferences, voice examples, ledger and summary. It changes almost every turn.

For Anthropic models the stable block carries `cache_control`. Other providers get one plain string.

**Why.** The ledger changes on almost every turn, so caching the whole prompt would invalidate the cache constantly.

**Caveat.** Claude Haiku 4.5 caches only prefixes of 4,096 tokens or more. Tools plus the stable block are just above that once an org brief exists. The savings have not been measured; check `usage.cache_read_input_tokens`.

### Org brief

**What.** After research, one `FAST_MODEL` call condenses the results into a brief of 350 words or less (mission, programs, news, audiences, facts), with the source URL after each fact. It is stored as `org_brief` and shown in the stable prompt block.

**Why.** Raw research reaches the model once, as a tool result. Without the brief, a new session could answer "what programs do we run?" only from source titles.

If the brief call fails, research is still saved with an empty brief.

### Research: Tavily and httpx with BeautifulSoup

**What.**

- `research_org()` runs `scrape_site()` and three Tavily searches in parallel (social presence, recent news, similar orgs).
- `scrape_site()` reads the homepage, then fetches in parallel up to three same-site pages whose path suggests About, Programs, Events and similar. Each page is labelled with its URL, so it can be cited.
- Mid-conversation `web_search` results (at most 10 per call) are added to the research sources. Only pages that were actually read become sources; a homepage that failed or was blocked does not.

**Why.**

- Tavily returns structured results and has an async client, so the searches run in parallel; the unofficial DuckDuckGo API has no async client and rate-limits. Tavily's free tier is 1,000 searches a month.
- httpx is async; BeautifulSoup drops scripts, navigation and footers.

**Limitation.** Pages rendered by JavaScript yield little text; supporting them would mean a headless browser such as Playwright.

### SSRF protection

**What.** `_fetch_html()` never lets httpx follow redirects. It follows up to five hops itself and checks every hop twice:

1. **`_is_safe_url()`** checks the URL as written. It blocks:
   - non-HTTP(S) schemes;
   - metadata hostnames, `localhost` and `*.localhost`;
   - any IP literal that isn't a public unicast address (`not is_global`), which covers loopback, private, link-local, unspecified and carrier-grade NAT (100.64.0.0/10, including Alibaba Cloud's metadata endpoint);
   - IPv4-mapped IPv6;
   - hex, decimal, octal and short IP forms.
2. **`_resolves_to_internal()`** resolves the hostname and blocks it if *any* address is internal, so a public-looking domain pointing at `127.0.0.1` is refused.

**Remaining gap.** httpx resolves the name again when it connects, so a DNS server that answers differently the second time (DNS rebinding) is not caught. The fix is to pin the connection to the checked address.

### Prompt-injection defense

**What.** Text from third parties is wrapped by `wrap_untrusted()` (`agent/untrusted.py`):

- website text and search snippets in `<external_content>`;
- the org brief in `<org_brief>`;
- the conversation summary in `<conversation_summary>`.

The wrapper escapes any copy of its own tag inside the text, so a page containing `</external_content>` can't close the wrapper and pass the rest of its text off as instructions. It also HTML-escapes attribute values. The stable prompt ends with a Security Policy telling the model to treat tagged content as data. The writers of the brief and the summary are told the same.

**Limitation.** It still depends on the model obeying its system prompt. `save_preference` accepts any key, so an injected preference that the model obeys would persist. Each preference records the user's quote, which helps spot one. A key allowlist and a check on the model's output are the next steps.

### Memory: JSON on disk

**What.** One `data/{org-id}/profile.json` per org:

```json
{
  "name": "Org Name", "website": "https://…",
  "research": { "sources": [ … ], … }, "org_brief": "- Mission: … (https://…)",
  "preferences": { "voice": "warm and grassroots", "banned_topics": ["politics"] },
  "preference_notes": { "voice": { "learned_at": "…", "quote": "Too corporate, we're more warm and grassroots" } },
  "voice_examples": [ { "content": "…", "source": "approved post_004", "added_at": "…" } ],
  "post_ledger": [ { "id": "post_002", "title": "Volunteer spotlight", "platform": "linkedin",
                     "status": "draft", "pillar": "Community Impact", "parent_id": "post_001",
                     "source_url": "https://…", "created_at": "…" } ]
}
```

**Why.** A non-engineer must be able to read and fix what Mackie learned. A JSON file in a text editor does that with no setup and no migrations. SQLite and Redis were the alternatives.

**Hand-edit safety.**

- **Saving:** writes go to a temp file that is swapped in atomically, and the previous version is kept as `profile.json.bak`.
- **Loading:** `ProfileError` describes the mistake in plain words: invalid JSON (with line and column), a top level that isn't an object, or a section of the wrong type (e.g. `"preferences": null`, or a `post_ledger` that isn't a list of objects). Keys missing from an older or hand-edited file get their defaults.
- **Hand-written posts:** a post with missing fields doesn't break the prompt or the ledger tools.
- **Post IDs:** `_next_post_id()` assigns one above the highest existing number, so deleting a post never causes an ID to be reused.

**Org ids.** `slugify()` keeps only `[a-z0-9-]`, and `_org_dir()` refuses any id that isn't in that form, so no id can reach outside `data/`. A name with no Latin letters or digits (Greek, Japanese) gets a stable `org-<hash>` id instead of an empty one.

**Limitation.** Load, change and save is not locked: two people editing the same org at once can overwrite each other. A multi-user deployment needs a database with row-level locking.

### Learning from feedback

**What.**

- `save_preference(key, value, quote, append)` keeps `preferences[key]` a plain value, or a list with `append` (e.g. `banned_topics`). It stores the date and the user's own words in `preference_notes`. Keys can be scoped to a platform, e.g. `linkedin_voice`.
- `forget_preference(key, value?)` removes a key, one item of a list, or a string preference only when the value matches.
- Approved posts and posts the user pastes become voice examples. The newest 5 are kept, and 3 appear in the prompt as "match their voice, not their topics".
- Ideas the user turns down get the status `rejected`.
- Asked "what have you learned about us?", Mackie answers from these sections.

**Why.** Corrections say what not to do; approved and pasted posts show what the org sounds like. Keeping the user's words beside each preference makes it explainable, while `preferences` stays easy to edit by hand.

### Post ledger

**What.** `save_post` requires a title. It optionally takes a pillar, content type, `parent_id` (the post this is a version of), `scheduled_for` and `source_url`. The statuses are `suggested`, `draft`, `approved`, `posted`, `planned` and `rejected` (`memory.POST_STATUSES`). Suggestions are shown with their ledger id (`**2. Volunteer spotlight** · post_003`), so "the second one" still resolves after a summary or in a new session. `search_ledger` filters by status, platform, date range and text.

**Why.** Users ask "what did we work on this week?", which needs dates and a query beyond the 20 posts in the prompt. Versions link a LinkedIn draft, its rewrite and its Instagram version.

### Checks enforced in code

**What.** `save_post` checks a post before saving it:

| Problem | Result |
|---|---|
| Longer than the platform's `max_length` | Not saved; the model is told to fix it and save again. |
| Hashtag count outside `hashtag_count` | Not saved, same as above. |
| `parent_id` that isn't in the ledger | Not saved. |
| `source_url` that isn't a known research source | Saved, with a warning. |

Tools that take a status or a date (`save_post`, `update_post_status`, `search_ledger`) reject a status outside `POST_STATUSES` and a date that isn't `YYYY-MM-DD`, with a message the model can act on. Platforms not defined in `skills/` (e.g. `general`) skip the platform checks.

**Why.** Models miscount characters and don't always follow the tool schema:

- In a live run, a model that was only *warned* about hashtags saved a corrected copy each time, which duplicated six ideas in the ledger.
- A Qwen run invented a `parent_id`.
- A date like "last week" was compared as text and silently matched the wrong posts.

Sources only get a warning, because whether a source is good enough is a judgment call.

**Unsaved suggestions.** Two checks make sure a post shown to the user is in the ledger:

- `research_org`'s result ends with a reminder to save every idea with `save_post` and label it with the returned id. Research comes right before the first suggestions.
- After the final reply, `should_continue` looks for `post_N` ids in it that aren't in the ledger and that the user or a tool didn't mention this turn. If there are any, the `save_check` node adds one follow-up message (named `save_check`, not a user turn) asking the model to save those posts and reply with their real ids. It runs at most once per turn. The first reply is already on screen, so the answer is appended to it; `chat()` returns both.

**Why.** In live evals, both Claude Haiku 4.5 and an OpenCode Zen model went straight from the research result to the suggestions without calling `save_post`, and wrote ledger ids themselves. "The second one" then pointed at a post that didn't exist, and step 7 repeated step 1's ideas because the ledger didn't contain them. The rule was already in the system prompt; a reminder in the tool result is where the model looks next. The id check is a backstop for any model, at the cost of one extra LLM call only when it fires. A reply with numbered ideas and no ids at all isn't caught; detecting that from free text would be guesswork.

### Skills: platforms, content types and pillars as YAML

**What.** Each `skills/*.yaml` file is one of two kinds:

- **`kind: platform`** (the default): `id`, `name` and `max_length` are required; `tone`, `hashtag_count`, `hashtag_style` and `format_hints` are optional.
- **`kind: content_type`**: `id`, `name`, `purpose` and `structure` are required; `must_include` and `avoid` are optional.

`skills/content_pillars.yaml` holds the default pillars, and a `content_pillars` list in `profile.json` overrides them for one org. Everything is loaded on every turn by `agent/skills.py`, so no restart is needed.

**Why.** Platforms and editorial policy are data, not code. One directory and one file format mean there is only one thing to learn.

**Broken files.** Invalid YAML, missing or wrongly typed fields, unknown kinds and duplicate ids are skipped and reported in plain English:

- in the log;
- as a Heads-up at the start of a web chat;
- by `scripts/validate.py`, which exits with code 1.

A broken pillars file means no pillars, not a crash.

### Content calendar

**What.** `generate_calendar` plans one post per week for four weeks, with the four LLM calls run in parallel. Each call gets the platforms from `skills/`, the org's pillars, its brief and preferences, and its recent posts. The results are saved as `planned` ledger entries and returned as a table. A failed week becomes an error row; the other weeks are still saved.

**Why.** Running the weeks in parallel takes about as long as one call.

**Limitation.** The calendar plans angles; it doesn't write the posts.

### Tool failures become tool results

**What.** `handle_tool_call()` never raises. An exception becomes the tool result `Error: {tool} failed — …` and is traced. Malformed tool-call JSON is parsed as `{}`.

**Why.** The tool-call message is checkpointed before the tool runs. A raised exception left that call unanswered, and providers reject every later request in that thread. Returning the error keeps the thread valid and lets the model recover.

### Observability

**What.** Every tool call appends `ts`, `tool`, `input`, `result_preview` and `latency_ms` to `data/{org-id}/trace.jsonl`. The web UI shows each tool call as a step.

**Why.** It needs no extra service. Langfuse or Helicone would add cost tracking and dashboards.

**Note.** The trace records tool inputs, including quotes and pasted posts, but never API keys.

### Evaluation (`scripts/eval.py`)

**What.** The script runs a 7-step acceptance conversation against a real org:

1. Suggest some posts.
2. Write the second one for LinkedIn.
3. Correct the voice to "warm and grassroots".
4. Make an Instagram version.
5. Ask what programs the org runs.
6. Ask which posts we've worked on and their status.
7. Suggest a post for next month.

It then reports two kinds of checks:

- **LLM judge.** Each reply is scored 0–3 on source attribution, voice consistency, and step 7 not repeating step 1: 21 judge calls in parallel. For the repeat check the judge is also shown step 1's reply; without it, the judge was guessing. A failed judge call scores -1 and the others still count.
- **Saved data, no LLM.** From snapshots of the profile taken after each step: a preference saved by step 3, step 7 saving posts that aren't copies or versions of step 1's, and a `source_url` on every suggestion.

**Why both.** The judge reads only the replies. In one run it contradicted itself, and nothing showed that the voice correction was never saved. The saved-data checks are exact about what the system learned, but they can't catch a paraphrased repeat; that is the judge's job.

**Judge bias.** Set `JUDGE_MODEL` to a model from another provider, so the judge doesn't grade its own model's work.

---

## Roadmap

| Gap | Today | Next step |
|---|---|---|
| Concurrent edits | `profile.json` writes are not locked | PostgreSQL for org memory; `langgraph-checkpoint-postgres` for history (also enables several replicas) |
| Authentication | Anyone who can reach the UI can open any org | Chainlit OAuth or a reverse proxy; per-org access rules |
| Rate limiting | None | Per-org, per-tool counters in Redis or Postgres |
| DNS rebinding | Names are checked, then resolved again by httpx | Connect to the checked address |
| Injected preferences | Any key is accepted; quotes help spot injections | Key allowlist and value length cap; flag preferences that have no user quote |
| Runtime quality check | The judge runs only in `scripts/eval.py` | A cheap judge call after `save_post` for voice, and a check for near-duplicate ideas |
| Calendar output | Week-level angles, saved as `planned` | Draft each entry with the platform rules; specific dates |
| Publishing and export | `posted` is only a status label | CSV export; Buffer or Hootsuite APIs; then native platform APIs gated on `approved` |
| Research freshness | Refreshed only when asked | Expiry date and a prompt to refresh at session start |
| Org switching | One org per web session | An org switcher that starts a new thread |
| Approval in the UI | Status changes through chat only | Approve and reject buttons on each post |
| Images | Not supported | `litellm.image_generation` for Instagram and Facebook posts, with moderation |
| JavaScript-rendered sites | Little text is extracted | A headless browser as a fallback |
