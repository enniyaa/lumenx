"""
LumenX Auto-Reply Agent — FastAPI entrypoint
Phase 5+: serves the review queue API and the human review UI.

Run locally:
    uvicorn agent.main:app --host 0.0.0.0 --port 8001 --reload

Endpoints:
  GET  /              → redirects to /review
  GET  /health        → service health check
  GET  /review        → HTML review panel (Phase 5 UI)
  GET  /agent/queue   → list pending review items
  POST /agent/queue/{id}/approve
  POST /agent/queue/{id}/edit
  POST /agent/queue/{id}/reject
  POST /agent/queue/{id}/feedback
  GET  /agent/config  → view threshold config (Phase 8)
  GET  /agent/stats   → cost stats (Phase 9)
"""

import os
import sys
import time

# Fix corporate proxy TLS before any imports that call the network
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ssl_utils import patch_ssl
patch_ssl()

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI
from fastapi.responses import RedirectResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

# ── App init ──────────────────────────────────────────────────────────────────

app = FastAPI(
    title="LumenX Auto-Reply Agent",
    description="AI-powered customer support reply agent with human review queue.",
    version="0.5.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files (review.html, knowledge graph index.html, etc.)
_static_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "static")
if os.path.isdir(_static_dir):
    app.mount("/static", StaticFiles(directory=_static_dir), name="static")

# ── Startup ───────────────────────────────────────────────────────────────────

_start_state: dict = {
    "wiki_loaded":   False,
    "model_loaded":  False,
    "db_connected":  False,
    "started_at":    None,
    "errors":        [],
}


@app.on_event("startup")
async def startup():
    _start_state["started_at"] = time.time()

    # 1. Init DB
    try:
        from db.models import init_db
        init_db()
        _start_state["db_connected"] = True
    except Exception as e:
        _start_state["errors"].append(f"db: {e}")

    # 2. Warm up FAISS wiki (pre-load into memory)
    try:
        from wiki.retriever import retrieve
        retrieve("pricing plans", k=1)
        _start_state["wiki_loaded"] = True
    except Exception as e:
        _start_state["errors"].append(f"wiki: {e}")

    # 3. Load ConfidenceNet (will be 0.5 stub if model not trained yet)
    try:
        from agent.confidence_net import get_confidence_net
        net = get_confidence_net()
        _start_state["model_loaded"] = net.is_loaded
    except Exception as e:
        _start_state["errors"].append(f"confidence_net: {e}")


# ── Core routes ───────────────────────────────────────────────────────────────

@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/review")


@app.get("/health")
def health():
    """Service health check — used by Railway and load tests."""
    uptime = time.time() - (_start_state["started_at"] or time.time())
    return {
        "status":        "ok" if _start_state["db_connected"] else "degraded",
        "db_connected":  _start_state["db_connected"],
        "wiki_loaded":   _start_state["wiki_loaded"],
        "model_loaded":  _start_state["model_loaded"],
        "uptime_seconds": round(uptime, 1),
        "errors":        _start_state["errors"],
    }


@app.get("/review", response_class=HTMLResponse, include_in_schema=False)
def review_ui():
    """Serve the human review panel."""
    html_path = os.path.join(_static_dir, "review.html")
    if os.path.exists(html_path):
        with open(html_path, encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse(content="<h2>review.html not found</h2>", status_code=404)


# ── Agent config (Phase 8 preview) ───────────────────────────────────────────

@app.get("/agent/config")
def get_config():
    """View current agent routing configuration."""
    from db.feedback_log import real_label_count
    real_labels = 0
    try:
        real_labels = real_label_count()
    except Exception:
        pass

    threshold  = float(os.getenv("CONFIDENCE_THRESHOLD", "0.90"))
    min_labels = int(os.getenv("MIN_REAL_LABELS_FOR_ROUTING", "50"))

    return {
        "confidence_threshold":       threshold,
        "min_real_labels_for_routing": min_labels,
        "real_label_count":           real_labels,
        "routing_active":             real_labels >= min_labels,
        "reply_max_tokens":           int(os.getenv("REPLY_MAX_TOKENS", "500")),
        "context_budget_tokens":      int(os.getenv("CONTEXT_BUDGET_TOKENS", "4000")),
        "poll_interval_seconds":      int(os.getenv("POLL_INTERVAL_SECONDS", "5")),
    }


# ── Include routers ───────────────────────────────────────────────────────────

from agent.routers.queue import router as queue_router
app.include_router(queue_router)
