# Environment Variables — LumenX Auto-Reply Agent

All variables can be set in a `.env` file (local development) or as Railway secret
environment variables (production). Variables marked **Required** will cause startup
to fail if unset.

---

## Required

| Variable | Example | Description |
|---|---|---|
| `ANTHROPIC_API_KEY` | `sk-ant-api03-…` | Anthropic API key. Used for all LLM calls (Haiku + Sonnet). Never commit this value. |
| `LUMENX_ADMIN_TOKEN` | `lmx_GQlch0Q5…` | LumenX admin bearer token sent as `X-Admin-Token` header. Never commit this value. |
| `LUMENX_BASE_URL` | `https://lumenx-demo.up.railway.app` | Base URL of the LumenX API service. |

---

## Agent Behaviour

| Variable | Default | Description |
|---|---|---|
| `CONFIDENCE_THRESHOLD` | `0.90` | Minimum MLP confidence score required to auto-send a reply. Replies below this threshold are always routed to the human review queue. Range: `[0.0, 1.0]`. Can be updated live via `PUT /agent/config`. |
| `MIN_REAL_LABELS_FOR_ROUTING` | `50` | The Confidence Net is only used for routing decisions once this many real human-labelled examples have been collected. Until then, every reply goes to human review regardless of predicted confidence. Can be updated live via `PUT /agent/config`. |
| `POLLER_ENABLED` | `true` | Set to `false` to disable the background inbox polling thread at startup. Useful for one-off scripts or debugging. |
| `POLL_INTERVAL_SECONDS` | `5` | How often the inbox poller checks for new unanswered customer messages (in seconds). |

---

## Context Window

| Variable | Default | Description |
|---|---|---|
| `CONTEXT_BUDGET_TOKENS` | `4000` | Maximum total tokens allowed in the assembled context window (system + wiki + summary + feedback + thread). If the assembled context exceeds this budget, sections are trimmed (thread first, then wiki). |
| `REPLY_MAX_TOKENS` | `500` | Maximum tokens Claude Sonnet may generate for a reply draft. |

---

## Database

| Variable | Default | Description |
|---|---|---|
| `DATABASE_URL` | `sqlite:///./data/agent.db` | SQLAlchemy database URL. For Railway, use `sqlite:////data/agent.db` (absolute path to the mounted volume). PostgreSQL is also supported: `postgresql://user:pass@host:5432/dbname`. |

---

## Logging & Monitoring

| Variable | Default | Description |
|---|---|---|
| `LOG_LEVEL` | `INFO` | Python logging level: `DEBUG`, `INFO`, `WARNING`, `ERROR`. Set to `DEBUG` to log full context assembly and token budgets. |
| `DAILY_COST_ALERT_USD` | *(unset)* | If set, the agent will log a WARNING when daily LLM cost exceeds this USD amount. Example: `5.00`. |

---

## Local Development Setup

```bash
cp .env.example .env
# Edit .env and fill in ANTHROPIC_API_KEY, LUMENX_ADMIN_TOKEN, LUMENX_BASE_URL

# Build the wiki index (one-time)
python wiki/build_wiki.py

# Seed feedback examples (one-time)
python scripts/seed_feedback.py

# Bootstrap the Confidence Net (one-time)
python scripts/bootstrap_labels.py

# Start the agent
uvicorn agent.main:app --host 0.0.0.0 --port 8001 --reload
```

Open http://localhost:8001/review — human review panel  
Open http://localhost:8001/dashboard — cost dashboard  
Open http://localhost:8001/docs — FastAPI auto-docs

---

## Railway Production Deployment

1. Create a new Railway service from this repo.
2. Set the following **secret** environment variables in Railway project settings:
   - `ANTHROPIC_API_KEY`
   - `LUMENX_ADMIN_TOKEN`
   - `LUMENX_BASE_URL`
3. Add the remaining variables as public config (see defaults above — all have safe defaults).
4. Add a Railway **Volume** mounted at `/data` for SQLite persistence.
5. Railway will build from `Dockerfile` and deploy automatically on push.

The `/health` endpoint is used by Railway as the health-check URL.

---

## Full `.env.example`

```bash
# ── Required ──────────────────────────────────────────────────────────────────
ANTHROPIC_API_KEY=sk-ant-api03-REPLACE_ME
LUMENX_ADMIN_TOKEN=lmx_REPLACE_ME
LUMENX_BASE_URL=https://lumenx-demo.up.railway.app

# ── Agent behaviour ───────────────────────────────────────────────────────────
CONFIDENCE_THRESHOLD=0.90
MIN_REAL_LABELS_FOR_ROUTING=50
POLLER_ENABLED=true
POLL_INTERVAL_SECONDS=5

# ── Context window ────────────────────────────────────────────────────────────
CONTEXT_BUDGET_TOKENS=4000
REPLY_MAX_TOKENS=500

# ── Database ──────────────────────────────────────────────────────────────────
DATABASE_URL=sqlite:///./data/agent.db

# ── Logging ───────────────────────────────────────────────────────────────────
LOG_LEVEL=INFO
# DAILY_COST_ALERT_USD=5.00
```
