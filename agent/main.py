"""
LumenX Auto-Reply Agent — FastAPI entrypoint
Phase 8: full agent with polling loop, live config, and review queue UI.

Run locally:
    uvicorn agent.main:app --host 0.0.0.0 --port 8001 --reload

Endpoints:
  GET  /              → redirect to /review
  GET  /health        → service health + poller status
  GET  /review        → React review panel
  GET  /agent/queue   → list queue items (status filter)
  POST /agent/queue/{id}/approve|edit|reject|feedback
  GET  /agent/config  → view routing config
  PUT  /agent/config  → update threshold live (no restart needed)
  POST /agent/poll    → trigger a single poll cycle (for testing / webhooks)
  GET  /agent/stats   → cost stats (Phase 9)
"""

import logging
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ssl_utils import patch_ssl
patch_ssl()

from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(
    level=getattr(logging, os.getenv("LOG_LEVEL", "INFO")),
    format="%(asctime)s  %(levelname)-7s  %(name)s  %(message)s",
)
logger = logging.getLogger("lumenx.main")

from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="LumenX Auto-Reply Agent",
    description="AI-powered customer support reply agent with human review queue.",
    version="0.9.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_static_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "static")
if os.path.isdir(_static_dir):
    app.mount("/static", StaticFiles(directory=_static_dir), name="static")

# ── Startup state ─────────────────────────────────────────────────────────────

_state: dict = {
    "db_connected":  False,
    "wiki_loaded":   False,
    "model_loaded":  False,
    "poller_active": False,
    "started_at":    None,
    "errors":        [],
}

# In-process mutable config (updated by PUT /agent/config)
_live_config: dict = {
    "confidence_threshold": float(os.getenv("CONFIDENCE_THRESHOLD", "0.90")),
    "poller_enabled":       os.getenv("POLLER_ENABLED", "true").lower() != "false",
}


@app.on_event("startup")
async def startup():
    import anthropic
    _state["started_at"] = time.time()

    # 1. Init DB
    try:
        from db.models import init_db
        init_db()
        _state["db_connected"] = True
        logger.info("DB initialised")
    except Exception as e:
        _state["errors"].append(f"db: {e}")
        logger.error("DB init failed: %s", e)

    # 2. Warm up FAISS wiki
    try:
        from wiki.retriever import retrieve
        retrieve("pricing plans", k=1)
        _state["wiki_loaded"] = True
        logger.info("FAISS wiki loaded")
    except Exception as e:
        _state["errors"].append(f"wiki: {e}")
        logger.warning("Wiki warm-up failed: %s", e)

    # 3. Load ConfidenceNet
    try:
        from agent.confidence_net import get_confidence_net
        net = get_confidence_net()
        _state["model_loaded"] = net.is_loaded
        logger.info("ConfidenceNet loaded=%s", net.is_loaded)
    except Exception as e:
        _state["errors"].append(f"confidence_net: {e}")
        logger.warning("ConfidenceNet load failed: %s", e)

    # 4. Start inbox poller (background daemon thread)
    if _live_config.get("poller_enabled", True):
        try:
            from agent.poller import start_poller
            client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
            start_poller(client=client)
            _state["poller_active"] = True
            logger.info("Inbox poller started")
        except Exception as e:
            _state["errors"].append(f"poller: {e}")
            logger.error("Poller start failed: %s", e)


@app.on_event("shutdown")
async def shutdown():
    try:
        from agent.poller import stop_poller
        stop_poller()
    except Exception:
        pass


# ── Core routes ───────────────────────────────────────────────────────────────

@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/review")


@app.get("/health")
def health():
    """Service health check — used by Railway and load tests."""
    uptime = time.time() - (_state["started_at"] or time.time())
    from db.feedback_log import real_label_count
    real_labels = 0
    try:
        real_labels = real_label_count()
    except Exception:
        pass

    from db.models import masked_db_url
    return {
        "status":          "ok" if _state["db_connected"] else "degraded",
        "db_connected":    _state["db_connected"],
        "wiki_loaded":     _state["wiki_loaded"],
        "model_loaded":    _state["model_loaded"],
        "poller_active":   _state["poller_active"],
        "real_labels":     real_labels,
        "uptime_seconds":  round(uptime, 1),
        "db_url":          masked_db_url(),
        "errors":          _state["errors"],
    }


@app.get("/review", response_class=HTMLResponse, include_in_schema=False)
def review_ui():
    html_path = os.path.join(_static_dir, "review.html")
    if os.path.exists(html_path):
        with open(html_path, encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse(content="<h2>review.html not found</h2>", status_code=404)


@app.get("/dashboard", response_class=HTMLResponse, include_in_schema=False)
def dashboard_ui():
    """Cost & performance dashboard."""
    html_path = os.path.join(_static_dir, "dashboard.html")
    if os.path.exists(html_path):
        with open(html_path, encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse(content="<h2>dashboard.html not found</h2>", status_code=404)


# ── Agent config ──────────────────────────────────────────────────────────────

class ConfigUpdate(BaseModel):
    confidence_threshold:        float | None = None
    poller_enabled:              bool  | None = None
    min_real_labels_for_routing: int   | None = None


@app.get("/agent/config")
def get_config():
    """View current agent routing configuration."""
    from db.feedback_log import real_label_count
    real_labels = 0
    try:
        real_labels = real_label_count()
    except Exception:
        pass

    min_labels = int(os.getenv("MIN_REAL_LABELS_FOR_ROUTING", "50"))
    threshold  = _live_config["confidence_threshold"]

    return {
        "confidence_threshold":        threshold,
        "min_real_labels_for_routing": min_labels,
        "real_label_count":            real_labels,
        "routing_active":              real_labels >= min_labels,
        "poller_enabled":              _live_config.get("poller_enabled", True),
        "poller_active":               _state["poller_active"],
        "reply_max_tokens":            int(os.getenv("REPLY_MAX_TOKENS", "500")),
        "context_budget_tokens":       int(os.getenv("CONTEXT_BUDGET_TOKENS", "4000")),
        "poll_interval_seconds":       int(os.getenv("POLL_INTERVAL_SECONDS", "5")),
    }


@app.put("/agent/config")
def update_config(body: ConfigUpdate):
    """
    Update routing config live — no restart needed.
    Changes are in-process only; they reset if the service restarts.
    To persist, update the env var in Railway.
    """
    changed = {}

    if body.confidence_threshold is not None:
        if not (0.0 <= body.confidence_threshold <= 1.0):
            raise HTTPException(400, "confidence_threshold must be in [0, 1]")
        _live_config["confidence_threshold"] = body.confidence_threshold
        # Also update the env var so auto_router.route() picks it up
        os.environ["CONFIDENCE_THRESHOLD"] = str(body.confidence_threshold)
        changed["confidence_threshold"] = body.confidence_threshold

    if body.poller_enabled is not None:
        _live_config["poller_enabled"] = body.poller_enabled
        changed["poller_enabled"] = body.poller_enabled

    if body.min_real_labels_for_routing is not None:
        os.environ["MIN_REAL_LABELS_FOR_ROUTING"] = str(body.min_real_labels_for_routing)
        changed["min_real_labels_for_routing"] = body.min_real_labels_for_routing

    logger.info("Config updated: %s", changed)
    return {"ok": True, "changed": changed}


# ── Manual poll trigger ───────────────────────────────────────────────────────

@app.post("/agent/poll")
def poll_now():
    """
    Trigger a single inbox poll cycle immediately.
    Useful for testing and webhook-driven polling.
    """
    from agent.poller import get_poller, InboxPoller
    import anthropic

    poller = get_poller()
    if poller is None:
        # Poller not started as a background thread — create a one-shot instance
        client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        poller = InboxPoller(client=client)

    results = poller.poll_once()
    return {
        "ok":       True,
        "messages": len(results),
        "results":  results,
    }


# ── Include routers ───────────────────────────────────────────────────────────

from agent.routers.queue  import router as queue_router
from agent.routers.stats  import router as stats_router
app.include_router(queue_router)
app.include_router(stats_router)
