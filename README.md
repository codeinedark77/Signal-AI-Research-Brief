# Signal — a daily research + arc-status brief

n8n does the plumbing (schedule, calendar, delivery). This package does the
one part worth being real code: dedupe → score → rank → summarize → compose,
as a small LangGraph graph behind a FastAPI endpoint.

Every day it tells you two things: which of the last day or two's
arXiv/GitHub activity actually intersects RLVR / GRPO / reward hacking /
agentic coding, and where you left off across the portfolio arc (so
"Assay GPU run: not yet executed" is something the tool remembers, not
something you hold in your head).

## Architecture

```
n8n:    Cron (7am) -> Google Calendar -> reshape -> ─┐
                                                       ▼
signal_brief:                              POST /brief/auto
                                                       │
                              ┌────────────────────────┤
                              │  ArxivClient.search()   │  (inside the
                              │  GitHubClient.search()  │   FastAPI call,
                              │        │                │   not n8n --
                              │        ▼                │   see "Why
                              │  dedupe (SeenStore)      │   ingestion
                              │        │                │   isn't in n8n"
                              │        ▼                │   below)
                              │  score (LLMClient)       │
                              │        │                │
                              │        ▼                │
                              │  rank + select top-K     │
                              │        │                │
                              │        ▼                │
                              │  summarize (LLMClient)   │
                              │        │                │
                              │        ▼                │
                              │  compose brief_text      │
                              │        │                │
                              │        ▼                │
                              │  persist_seen (SeenStore)│
                              └────────────┬─────────────┘
                                           ▼
n8n:                              Send to Telegram


-- evening, separate n8n workflow, event-driven not scheduled --

n8n:    Telegram message -> reshape (text, chat_id) -> ─┐
                                                          ▼
signal_brief:                            POST /arc-status/extract
                                                          │
                              ┌───────────────────────────┤
                              │  known_projects lookup      │  (stored
                              │  (ArcStatusStore)            │   projects,
                              │        │                     │   falling
                              │        ▼                     │   back to
                              │  extract_arc_updates(LLM)     │   Yash's
                              │        │                     │   5 actual
                              │        ▼                     │   projects)
                              │  upsert each (ArcStatusStore) │
                              └───────────────┬───────────────┘
                                              ▼
n8n:                          Send confirmation_text to Telegram
```

Two separate n8n workflows because they trigger differently: the brief is
scheduled, the capture loop is event-driven on incoming Telegram messages.
Both are thin by design -- every actual decision (relevance scoring,
status extraction) happens in tested Python, not in n8n expressions.

## Quick start (no API keys, no n8n, just see it work)

```bash
python3 -m venv venv && ./venv/bin/pip install -r requirements.txt
SIGNAL_USE_FAKE_LLM=1 ./venv/bin/python -m signal_brief.cli --skip-arxiv --skip-github
```

That prints a brief with no candidates (since both sources are skipped) --
confirms the graph, stores, and CLI wiring all work before you touch a
credential. Drop `--skip-github` to pull real live GitHub results (no token
needed for a few requests); arXiv will depend on your network (see below).

To see the whole thing against real, currently-indexed papers and repos in
your actual research lane:

```bash
export ANTHROPIC_API_KEY=sk-ant-...   # or leave SIGNAL_USE_FAKE_LLM=1
./venv/bin/python -m signal_brief.cli
```

To try the evening capture loop's extraction without touching Telegram or
n8n at all:

```bash
SIGNAL_DATA_DIR=/tmp/signal-try SIGNAL_USE_FAKE_LLM=1 ./venv/bin/uvicorn signal_brief.api:app &
curl -sX POST localhost:8000/arc-status/extract \
  -H 'Content-Type: application/json' \
  -d '{"text": "Kicked off the Assay GPU run tonight. Next: check the loss curve tomorrow."}' | python3 -m json.tool
```

## Real setup (n8n + always-on hosting)

Your RTX 3050 laptop isn't an always-on box, and this doesn't need a GPU
anyway (the reasoning step is a handful of short API calls/day) -- run this
on a cheap VPS (Hetzner/DigitalOcean, a few dollars a month) or a spare Pi.

```bash
cp .env.example .env      # fill in ANTHROPIC_API_KEY, TELEGRAM_CHAT_ID, etc.
docker compose up -d
```

This starts `signal-api` (this package, port 8000 internal-only) and `n8n`
(port 5678, open http://your-host:5678 to finish setup):

1. **Credentials > New > Google Calendar OAuth2 API** — standard n8n OAuth
   flow, follow the prompts.
2. **Credentials > New > Telegram API** — paste the bot token you got from
   [@BotFather](https://t.me/BotFather).
3. Import both `n8n/signal_workflow.json` and
   `n8n/signal_evening_capture.json` (Workflows > Import from File).
4. In "Signal - Daily Brief": open "Get Today's Calendar Events" and attach
   your Google Calendar credential; open "Send to Telegram" and attach
   your Telegram credential.
5. In "Signal - Evening Capture": attach the same Telegram credential to
   both "On Telegram Message" and "Send Confirmation".
6. Activate both workflows.

The evening capture workflow needs a public webhook (n8n's Telegram
Trigger registers one automatically on activation) -- this is why "a cheap
VPS with a public IP" from the section above, not a laptop behind NAT,
matters for this half specifically.

### Import notes (read this before assuming it Just Works)

This sandbox couldn't install n8n to live-test either import — a
transitive dependency of `@n8n/n8n-nodes-langchain` fetches from
`cdn.sheetjs.com`, which is outside this environment's network allowlist.
So:

- **Structurally verified**: `scripts/validate_n8n_workflow.py` checks
  both files' JSON shape for real (no dangling connections, no duplicate
  ids/names, no orphaned nodes) — `tests/test_validate_n8n_workflow.py`
  proves the validator actually catches these by feeding it deliberately
  broken workflows, not just that it passes on these two files.
- **Not live-verified**: the exact parameter names for `googleCalendar`,
  `telegram`, and `telegramTrigger`. I'm confident in the shapes (all
  common, stable nodes) but haven't clicked "import" in a real n8n editor
  against these exact files. If a field's greyed out or renamed when you
  open a node, that's why -- it's a quick fix in the UI, not a sign
  anything deeper is wrong. The evening-capture workflow was deliberately
  kept linear (no IF/Switch node for the voice-vs-text branch — that
  decision was pushed into `/arc-status/extract` instead, which IS fully
  tested) specifically to avoid stacking a third kind of unverified node
  schema on top of these two.
- **Everything else** — the Schedule Trigger, the Code nodes, and above
  all the actual `POST /brief/auto` and `POST /arc-status/extract` calls
  and everything behind them — is ordinary, well-understood n8n plus a
  FastAPI service with 116 passing tests, 3 of which hit real external
  APIs live.

## What's real vs. mocked in the test suite

Yash's usual bar here, applied precisely:

| Component | Verified how |
|---|---|
| GitHub client | **Live**, against the real API (`pytest -m live`). One run surfaced `zacharyspeck/research-study` — GRPO+LoRA fine-tuning of Qwen2.5-1.5B, the same base model as Assay. |
| arXiv parser | Real captured data — two genuine, currently-indexed papers (SpecBench, arXiv:2605.21384; MO-GRPO, arXiv:2509.22047), not invented fixtures. |
| arXiv live fetch | **Cannot be verified from this sandbox** — `export.arxiv.org` returns `403` here specifically (confirmed directly with `ArxivClient.fetch_raw`, not assumed). Almost certainly an IP-range block on Anthropic's cloud infra, not a code issue — arXiv's API needs no auth and works fine from normal home/VPS networks. Run `./venv/bin/python -m signal_brief.cli --skip-github` from your own machine to confirm; if it 403s there too, tell me and I'll dig further. |
| LangGraph pipeline | Full unit + integration coverage with a deterministic `FakeLLMClient` — dedupe, ranking, top-K-only summarization, taxonomy override, and the persist-seen ordering (see "Bugs found" below). |
| Arc-status extraction | `FakeLLMClient` (deterministic) + `AnthropicLLMClient` tested against a mocked SDK response covering well-formed JSON, markdown-fenced JSON, malformed JSON, wrong JSON type, missing keys, non-dict array items — the actual model call itself isn't run for real (no API key here), but every parsing/defensive path it needs is. |
| FastAPI service | `TestClient`, all endpoints, malformed-input rejection, dedupe-across-calls, null/empty/whitespace text handling. |
| n8n workflows (both) | Structural only, per above. |
| Anthropic-backed scoring/summarizing/extraction | **Not run for real** — no API key in this sandbox. `AnthropicLLMClient` fails loudly (`RuntimeError`, not a silent no-op) if `ANTHROPIC_API_KEY` is unset; verified that failure mode directly. |
| Voice transcription | **Not built.** Would need either an OpenAI API key (new dependency) or downloading Whisper model weights from Hugging Face -- `huggingface.co` isn't in this sandbox's network allowlist either, so I couldn't verify that path even structurally. Text-based evening capture gets you the full feature; see "Extending it" for where voice plugs in. |

Run the suite: `./venv/bin/python -m pytest tests/` (116 tests). Skip the
live ones (e.g. offline, or GitHub rate-limited): `pytest -m "not live"`.

## Bugs found and fixed while building this

1. **Dedupe persistence was an external, forgettable step.** The first
   version had `SeenStore.mark_seen()` as something the *caller*
   (API/CLI) had to remember to call after `graph.invoke()`. A test
   script that forgot it silently produced wrong behavior with no error
   — exactly the kind of quiet bug that's worse than a crash. Fixed by
   moving it into the graph itself as a `persist_seen` node that runs
   last, so a mid-pipeline failure (e.g. the LLM call in `score_node`
   throwing) means nothing gets marked seen and today's candidates are
   still eligible next attempt. `test_graph.py::TestDedupe` covers both
   the happy path and the failure-doesn't-persist case.
2. **A vacuous test assertion** (`... or True`) in the arXiv query-builder
   tests that would have passed regardless of what the code did.
   Replaced with real URL-decoding assertions against the actual
   `search_query` parameter.
3. **An off-by-one in a test, not the source** — asserted the fake
   summarizer's truncation produced 31 words when it actually and
   correctly produces 30 (the `...` suffix attaches to the last word
   rather than being appended as a separate token).
4. **`ArcExtractRequest.text` was typed `str` (required), not
   `Optional[str]`.** Telegram's own payload has no `.text` field at all
   for a voice message, so the n8n reshape node passes `null` through --
   which a required `str` field would 422 on. Caught by tracing the real
   Telegram payload shape before it shipped, not by a failing test;
   widened to `Optional[str] = None` and added
   `test_missing_text_field_degrades_gracefully_rather_than_422` /
   `test_explicit_null_text_degrades_gracefully` /
   `test_whitespace_only_text_degrades_gracefully` so the three ways
   "nothing to work with" can arrive are all covered the same way.

## Extending it

- **New source**: add a client in `signal_brief/clients/`, add it to
  `create_brief_auto()` in `api.py` (same try/except-per-source pattern
  so one source failing doesn't take down the others), done — the graph
  doesn't care where candidates came from.
- **Change what counts as relevant**: edit `DEFAULT_TAXONOMY` in
  `graph.py`, or pass `taxonomy` in the request body per-call.
- **Cheaper scoring**: `AnthropicLLMClient(model="claude-haiku-4-5-20251001")`
  instead of the `claude-sonnet-5` default — coarser judgments, a fraction
  of the cost, plausibly fine for a first filter pass.
- **Voice transcription for the evening capture loop**: the only piece
  deliberately left out (see the table above for why it couldn't be
  verified from here). Cheapest path: an OpenAI-compatible Whisper API
  call from an added HTTP Request node between "On Telegram Message" and
  "Extract Text And Chat Id" in `signal_evening_capture.json` (Telegram's
  `message.voice.file_id` -> `getFile` -> download -> transcribe), feeding
  the transcript into the existing `text` field. Everything downstream
  (`/arc-status/extract`) already handles arbitrary text identically
  whether it came from typing or transcription -- no Python changes
  needed, only a new n8n node.
