# LumenX Auto-Reply Agent — Build Plan

> Updated: 2026-05-27  
> Rule: Ask user permission before starting each new phase.

---

## Overall Progress

| Phase | Name | Status |
|-------|------|--------|
| 1 | Project Scaffolding & LLM Wiki | ✅ COMPLETE |
| 2 | Intent Router | ✅ COMPLETE |
| 3 | Context Builder | ✅ COMPLETE |
| 4 | Draft Agent | ✅ COMPLETE |
| 5 | Human Review UI + Feedback Capture | ✅ COMPLETE |
| 6 | Feedback Log | ⏳ PENDING |
| 7 | Confidence Net (MLP) | ⏳ PENDING |
| 8 | Auto-Reply Router | ⏳ PENDING |
| 9 | Cost Dashboard | ⏳ PENDING |
| 10 | Hardening & Deployment | ⏳ PENDING |

---

## Phase 1 — Project Scaffolding & LLM Wiki ✅ COMPLETE

**Goal**: Repo structure, dependency install, LLM Wiki built from product API.

| Task | Status |
|------|--------|
| `requirements.txt` created | ✅ |
| `.env` + `.env.example` created | ✅ |
| `ssl_utils.py` (corporate proxy SSL patch) | ✅ |
| `db/models.py` — SQLAlchemy: FeedbackEntry, CostLog, ReviewQueue, MLPTrainingRow | ✅ |
| `db/session.py` — `get_db()` context manager | ✅ |
| `wiki/build_wiki.py` — fetch 20 products → chunk → embed → FAISS | ✅ |
| `wiki/retriever.py` — query FAISS, return top-k chunks | ✅ |
| `wiki/index.faiss` + `wiki/chunks.json` built | ✅ 127 chunks, 20 products |
| **BONUS**: `wiki_server.py` + `static/index.html` — D3.js knowledge graph | ✅ |

**Notes**:
- Corporate TLS proxy requires `ssl_utils.patch_ssl()` at top of every external-calling module
- `use_hf_cache_only()` prevents HuggingFace from phoning home after first download
- Knowledge graph website at http://localhost:8000 with RAG query chat panel

---

## Phase 2 — Intent Router ✅ COMPLETE

**Goal**: Classify every incoming message (greeting/pricing/technical/refund/other) before the context builder.

| Task | Status |
|------|--------|
| `agent/intent_router.py` created | ✅ |
| Haiku model (`claude-haiku-4-5-20251001`) for classification | ✅ |
| System prompt — JSON-only output `{"intent": "..."}` | ✅ |
| 5 intents with explicit boundary rules | ✅ |
| Greeting fast-path — skip LLM, reply directly from `GREETING_REPLIES` dict | ✅ |
| `CostLog` DB insert via `_log_cost()` | ✅ |
| Fix `datetime.utcnow()` → `datetime.now(timezone.utc)` | ✅ |
| System prompt tuned: technical=errors only; capability questions→other | ✅ |
| **Re-run 10-message accuracy test (target ≥ 90%)** | ✅ 10/10 (100%) |
| Git commit Phase 2 | ✅ |

**Final test result**: 10/10 (100%) — total cost $0.002568 for 10 messages  
**Fix applied**: System prompt updated — capability questions explicitly → `other`; greeting fast-path skips Haiku entirely

---

## Phase 3 — Context Builder ✅ COMPLETE

**Goal**: Assemble a rich, token-budgeted context window (≤ 4,000 tokens) for every non-greeting reply.

| Task | Status |
|------|--------|
| `agent/context_builder.py` created | ✅ |
| `build_conversation_summary()` — Haiku summary of all threads, cached 24h | ✅ |
| `get_feedback_log_entries(query, k=5)` — Phase 6 stub (returns []) | ✅ |
| `get_current_thread(thread_id)` — fetch last 10 messages | ✅ |
| `assemble(thread_id, message, intent)` → `{system_prompt, context_str, exact_tokens, sections}` | ✅ |
| Token counting — exact via `client.messages.count_tokens()` | ✅ |
| Budget trimming — thread first, then wiki, if over 4,000 tokens | ✅ |
| Fixed double-header bug in wiki chunk formatting | ✅ |

**Final test result**: 810 tokens exact (20.2% of 4,000 budget) ✅  
**Summary cost**: $0.001906 Haiku (cached after first call)  
**`AGENT_SYSTEM_PROMPT`** defined here as single source of truth → imported by Phase 4

**Context window layout**:
```
[SYSTEM PROMPT]         ~400 tokens  (cached)
[PRODUCT WIKI CHUNKS]   ~600 tokens  (top-k retrieved, cached if same product)
[CONVERSATION SUMMARY]  ~400 tokens  (daily summary of all threads)
[FEEDBACK LOG ENTRIES]  ~600 tokens  (top-5 similar past approved replies)
[CURRENT THREAD]        ~800 tokens  (last 10 messages)
[USER MESSAGE]          ~200 tokens
```

---

## Phase 4 — Draft Agent ✅ COMPLETE

**Goal**: Generate a high-quality, grounded reply using Claude Sonnet.

| Task | Status |
|------|--------|
| `agent/draft_agent.py` created | ✅ |
| System prompt with strict no-hallucination rules (imported from `context_builder.AGENT_SYSTEM_PROMPT`) | ✅ |
| Sonnet (`claude-sonnet-4-6`) with `max_tokens=500` | ✅ |
| Prompt caching: `cache_control: {"type":"ephemeral"}` on system + wiki/summary blocks | ✅ |
| Full usage tracking (input/output/cache_read/cache_creation tokens) | ✅ |
| USD cost calculation + `CostLog` insert | ✅ |
| `DraftResult` dataclass: text, model, tokens, cost_usd, cache_hit, context_json | ✅ |
| Self-test: 4 intents × real LumenX thread, sign-off verified | ✅ |

**Final test result**: 4/4 replies generated — sign-off "— LumenX Support" present ✅  
**Cost per reply**: $0.004–$0.007 (Sonnet, no cache yet)  
**DB logging**: 4 CostLog rows confirmed in `data/agent.db`

**Cache note**: Test context was ~550 tokens (below Anthropic's 1024-token minimum for prompt caching). Cache misses are expected on small test contexts. In production, system prompt (~400 tok) + full wiki chunks (~600 tok) + summary (~400 tok) easily exceed 1024 tokens and the cache will activate on repeat requests for the same product.

**`DraftResult` fields**:
```
text                        — generated reply
model                       — "claude-sonnet-4-6"
input_tokens                — total input tokens billed
output_tokens               — reply tokens
cache_read_input_tokens     — tokens loaded from cache (0.1× price)
cache_creation_input_tokens — tokens written to cache (1.25× price)
cost_usd                    — total USD cost
cache_hit                   — True if cache_read_input_tokens > 0
context_json                — serialised system/cacheable/dynamic split (for Phase 9 dashboard)
exact_context_tokens        — token count from Phase 3 context builder
```

---

## Phase 5 — Human Review UI + Feedback Capture ✅ COMPLETE

**Goal**: Review panel in admin UI for approving/editing/rejecting agent drafts.

| Task | Status |
|------|--------|
| `GET /agent/queue` — list pending drafts (+ status filter, limit) | ✅ |
| `GET /agent/queue/{id}` — full detail with context_json | ✅ |
| `POST /agent/queue/{id}/approve` — send to LumenX + FeedbackEntry(approved_as_is=True) | ✅ |
| `POST /agent/queue/{id}/edit` — send edited text + FeedbackEntry(approved_as_is=False) | ✅ |
| `POST /agent/queue/{id}/reject` — mark rejected, no LumenX send | ✅ |
| `POST /agent/queue/{id}/feedback` — thumbs up/down → FeedbackEntry.thumbs | ✅ |
| `agent/main.py` — FastAPI app: startup events, CORS, static files, health check | ✅ |
| `agent/routers/queue.py` — all queue endpoints in dedicated router | ✅ |
| `static/review.html` — standalone React review panel (dark theme, no build step) | ✅ |
| DB migration: `customer_msg` + `cost_usd` added to `review_queue` table | ✅ |

**Final test result**: 10/10 checks passed  
**FeedbackEntry creation**: approve → `approved_as_is=True`; edit → `approved_as_is=False` ✅  
**Routing**: resolved items immediately removed from pending queue ✅

**Review UI features** (`/review`):
- Dark themed React SPA served from FastAPI static files
- Left sidebar: queue items with intent colour, confidence %, cost chip
- Status filter tabs: pending / approved / edited / rejected / auto_sent
- ReplyCard: customer message, confidence bar, draft (editable textarea)
- Approve / Edit & Send / Reject buttons with loading spinner
- Thumbs up/down feedback buttons
- Expandable "Show Context Window" accordion (shows system/cacheable/dynamic sections)
- Toast notifications for all actions
- 5-second auto-poll while on pending tab

**Design note**: approve/edit are non-fatal on LumenX send failure — the FeedbackEntry and status update still commit, `sent: false` is returned. This prevents losing approvals when the LumenX API is temporarily unreachable.

---

## Phase 6 — Feedback Log ⏳ PENDING PERMISSION

**Goal**: Every approved/edited reply becomes a few-shot example for future context.

| Task | Status |
|------|--------|
| `db/feedback_log.py` — insert `FeedbackEntry` on approve/edit | ⏳ |
| `wiki/feedback_index.faiss` — embed customer messages for similarity search | ⏳ |
| Rebuild feedback index nightly (or every 10th new entry) | ⏳ |
| Wire `get_feedback_log_entries()` in context builder | ⏳ |

---

## Phase 7 — Confidence Net (MLP) ⏳ PENDING PERMISSION

**Goal**: Predict P(reply approved as-is) for every new draft.

| Task | Status |
|------|--------|
| `training/featurize.py` — 6 float features extraction | ⏳ |
| `scripts/bootstrap_labels.py` — heuristic labels from 100 demo conversations | ⏳ |
| `training/train.py` — MLPClassifier (64×64, ReLU, Adam), StratifiedKFold eval | ⏳ |
| `agent/confidence_net.py` — inference wrapper, returns 0.5 if < 50 real labels | ⏳ |
| `scripts/nightly_retrain.py` — retrain cron, deploy if PR-AUC improves | ⏳ |

**MLP Features**: `len_ratio`, `intent_encoded`, `retrieval_hits`, `edit_dist_norm`, `has_price_mention`, `draft_len_tokens`  
**Gate**: Only route via MLP when ≥ 50 real labelled examples exist

---

## Phase 8 — Auto-Reply Router ⏳ PENDING PERMISSION

**Goal**: Gate between auto-send and human review using confidence score.

| Task | Status |
|------|--------|
| `agent/auto_router.py` | ⏳ |
| Read `CONFIDENCE_THRESHOLD` from env (default 0.90) | ⏳ |
| Auto-send path: `POST /api/admin/threads/{id}/reply` | ⏳ |
| Queue path: insert into `ReviewQueue` | ⏳ |
| `GET /agent/config` — view/update threshold without redeploy | ⏳ |

---

## Phase 9 — Cost Dashboard ⏳ PENDING PERMISSION

**Goal**: Full visibility into per-reply cost, token usage, and context windows.

| Task | Status |
|------|--------|
| `GET /agent/stats?period=day\|week\|month` | ⏳ |
| `GET /agent/replies?page=1&limit=50` — reply log table | ⏳ |
| `GET /agent/replies/{id}/context` — full prompt viewer | ⏳ |
| Dashboard: total cost, auto-sent vs reviewed split, avg confidence | ⏳ |
| Reply log: timestamp, intent, model, tokens, cost, confidence, routed_to | ⏳ |

---

## Phase 10 — Hardening & Deployment ⏳ PENDING PERMISSION

**Goal**: Production-ready service on Railway.

| Task | Status |
|------|--------|
| Rate limiting: max 1 reply per thread per 5s | ⏳ |
| Retry logic: exponential backoff on Anthropic calls (max 3 retries) | ⏳ |
| Error fallback: route to human review if LLM call fails | ⏳ |
| `GET /health` — status, model_loaded, wiki_loaded, db_connected | ⏳ |
| `Dockerfile` | ⏳ |
| `railway.toml` | ⏳ |
| `docs/env-vars.md` | ⏳ |
| Load test: 50 concurrent messages | ⏳ |

---

## Key Files

| File | Phase | Purpose |
|------|-------|---------|
| `ssl_utils.py` | 1 | Corporate proxy SSL patch — import in every module |
| `db/models.py` | 1 | SQLAlchemy ORM: FeedbackEntry, CostLog, ReviewQueue, MLPTrainingRow |
| `wiki/build_wiki.py` | 1 | Fetch products → embed → FAISS |
| `wiki/retriever.py` | 1 | Query FAISS, return top-k chunks |
| `wiki_server.py` | 1 | FastAPI + D3.js knowledge graph at :8000 |
| `agent/intent_router.py` | 2 | Haiku intent classification (5 intents) |
| `agent/context_builder.py` | 3 | Assemble ≤4000-token context window |
| `agent/draft_agent.py` | 4 | Sonnet reply generation + cost tracking |
| `agent/confidence_net.py` | 7 | MLP inference wrapper |
| `agent/auto_router.py` | 8 | Threshold gate → send or queue |

---

## Environment Variables

```bash
ANTHROPIC_API_KEY=sk-ant-...
LUMENX_ADMIN_TOKEN=lmx_GQlch0Q5NOwVuVSADXRuFNJvxIpzVGwI
LUMENX_BASE_URL=https://lumenx-demo.up.railway.app
CONFIDENCE_THRESHOLD=0.90
MIN_REAL_LABELS_FOR_ROUTING=50
CONTEXT_BUDGET_TOKENS=4000
REPLY_MAX_TOKENS=500
DATABASE_URL=sqlite:///./data/agent.db
```
