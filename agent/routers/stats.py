"""
Stats Router — Phase 9
Cost dashboard API endpoints.

Endpoints:
  GET /agent/stats?period=day|week|month  — aggregate KPIs + breakdowns
  GET /agent/replies?page&limit&intent&status&period
                                          — paginated reply log
  GET /agent/replies/{id}/context         — full context + features for one item
"""

import json
import os
from datetime import datetime, timezone, timedelta
from typing import Optional

from fastapi import APIRouter, HTTPException, Query
from sqlalchemy import func

router = APIRouter(tags=["stats"])


# ── Helpers ───────────────────────────────────────────────────────────────────

def _period_start(period: str) -> datetime:
    """Return the UTC start of the requested period."""
    now = datetime.now(timezone.utc)
    if period == "day":
        return now.replace(hour=0, minute=0, second=0, microsecond=0)
    if period == "week":
        return (now - timedelta(days=now.weekday())).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
    if period == "month":
        return now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    raise ValueError(f"Unknown period: {period!r}")


def _naive(dt: datetime) -> datetime:
    """Strip timezone info for SQLite comparisons (stored as naive UTC)."""
    if dt.tzinfo is not None:
        return dt.replace(tzinfo=None)
    return dt


# ── GET /agent/stats ──────────────────────────────────────────────────────────

@router.get("/agent/stats")
def get_stats(period: str = Query("day", regex="^(day|week|month)$")):
    """
    Aggregate KPIs for the requested period.

    Returns:
        period, from/to timestamps, total_replies, total_cost_usd,
        avg_cost_per_reply, auto_sent, queued, auto_sent_pct,
        avg_confidence, by_intent, cost_by_model, hourly_costs (for chart)
    """
    from db.session import get_db
    from db.models import CostLog, ReviewQueue

    since     = _period_start(period)
    since_str = since.isoformat()

    with get_db() as db:
        # ── ReviewQueue stats (per-message) ──────────────────────────────────
        rq_rows = (
            db.query(ReviewQueue)
            .filter(ReviewQueue.created_at >= _naive(since))
            .all()
        )

        total_replies  = len(rq_rows)
        auto_sent      = sum(1 for r in rq_rows if r.status == "auto_sent")
        queued         = sum(1 for r in rq_rows if r.status != "auto_sent")
        confidences    = [r.confidence for r in rq_rows if r.confidence is not None]
        avg_confidence = round(sum(confidences) / len(confidences), 4) if confidences else 0.0

        # Per-intent breakdown
        intent_counts: dict = {}
        for r in rq_rows:
            key = r.intent or "other"
            if key not in intent_counts:
                intent_counts[key] = {"count": 0, "cost": 0.0}
            intent_counts[key]["count"] += 1
            intent_counts[key]["cost"]  += r.cost_usd or 0.0

        # Round per-intent costs
        for v in intent_counts.values():
            v["cost"] = round(v["cost"], 6)

        # ── CostLog stats (per-LLM-call) ──────────────────────────────────────
        cl_rows = (
            db.query(CostLog)
            .filter(CostLog.created_at >= _naive(since))
            .all()
        )

        total_cost_usd = round(sum(r.cost_usd or 0.0 for r in cl_rows), 6)

        # Cost by model
        model_stats: dict = {}
        for r in cl_rows:
            m = r.model or "unknown"
            if m not in model_stats:
                model_stats[m] = {"calls": 0, "cost": 0.0,
                                  "input_tokens": 0, "output_tokens": 0,
                                  "cache_read_tokens": 0}
            model_stats[m]["calls"]             += 1
            model_stats[m]["cost"]              += r.cost_usd or 0.0
            model_stats[m]["input_tokens"]      += r.input_tokens or 0
            model_stats[m]["output_tokens"]     += r.output_tokens or 0
            model_stats[m]["cache_read_tokens"] += r.cache_read_input_tokens or 0

        for v in model_stats.values():
            v["cost"] = round(v["cost"], 6)

        # Hourly cost buckets for chart (last 24 h for day, daily for week/month)
        hourly: dict = {}
        bucket_fmt = "%Y-%m-%dT%H:00" if period == "day" else "%Y-%m-%d"
        for r in cl_rows:
            bucket = r.created_at.strftime(bucket_fmt)
            hourly[bucket] = round(hourly.get(bucket, 0.0) + (r.cost_usd or 0.0), 6)

        cost_timeline = [
            {"bucket": k, "cost": v}
            for k, v in sorted(hourly.items())
        ]

    avg_cost_per_reply = (
        round(total_cost_usd / total_replies, 6) if total_replies else 0.0
    )

    return {
        "period":             period,
        "from":               since_str,
        "to":                 datetime.now(timezone.utc).isoformat(),
        "total_replies":      total_replies,
        "total_cost_usd":     total_cost_usd,
        "avg_cost_per_reply": avg_cost_per_reply,
        "auto_sent":          auto_sent,
        "queued":             queued,
        "auto_sent_pct":      round(auto_sent / total_replies * 100, 1) if total_replies else 0.0,
        "avg_confidence":     avg_confidence,
        "by_intent":          intent_counts,
        "cost_by_model":      model_stats,
        "cost_timeline":      cost_timeline,
    }


# ── GET /agent/replies ─────────────────────────────────────────────────────────

@router.get("/agent/replies")
def list_replies(
    page:   int = Query(1,    ge=1),
    limit:  int = Query(50,   ge=1, le=200),
    intent: Optional[str] = Query(None),
    status: Optional[str] = Query(None),
    period: Optional[str] = Query(None, regex="^(day|week|month)$"),
):
    """
    Paginated reply log (most recent first).

    Returns total count + list of items with key fields.
    """
    from db.session import get_db
    from db.models import ReviewQueue

    with get_db() as db:
        q = db.query(ReviewQueue)

        if period:
            since = _naive(_period_start(period))
            q = q.filter(ReviewQueue.created_at >= since)
        if intent:
            q = q.filter(ReviewQueue.intent == intent)
        if status:
            q = q.filter(ReviewQueue.status == status)

        total = q.count()
        rows  = (
            q.order_by(ReviewQueue.created_at.desc())
            .offset((page - 1) * limit)
            .limit(limit)
            .all()
        )

        items = [
            {
                "id":           r.id,
                "thread_id":    r.thread_id,
                "created_at":   r.created_at.isoformat() if r.created_at else None,
                "resolved_at":  r.resolved_at.isoformat() if r.resolved_at else None,
                "intent":       r.intent,
                "confidence":   r.confidence,
                "status":       r.status,
                "cost_usd":     r.cost_usd,
                "customer_msg": (r.customer_msg or "")[:200],   # truncated for list view
                "draft_text":   (r.draft_text or "")[:300],
            }
            for r in rows
        ]

    return {
        "total": total,
        "page":  page,
        "limit": limit,
        "items": items,
    }


# ── GET /agent/replies/{id}/context ───────────────────────────────────────────

@router.get("/agent/replies/{item_id}/context")
def get_reply_context(item_id: int):
    """
    Full detail for one ReviewQueue item, including:
    - customer_msg, draft_text (complete)
    - parsed context_json (system / cacheable / dynamic sections)
    - parsed features_json
    - associated CostLog rows for this thread
    """
    from db.session import get_db
    from db.models import ReviewQueue, CostLog

    with get_db() as db:
        item = db.query(ReviewQueue).filter(ReviewQueue.id == item_id).first()
        if not item:
            raise HTTPException(404, f"Reply {item_id} not found")

        # Parse JSON blobs
        try:
            context  = json.loads(item.context_json)  if item.context_json  else {}
        except Exception:
            context  = {"raw": item.context_json}

        try:
            features = json.loads(item.features_json) if item.features_json else {}
        except Exception:
            features = {}

        # Associated cost log rows for this thread_id
        cost_rows = (
            db.query(CostLog)
            .filter(CostLog.thread_id == item.thread_id)
            .order_by(CostLog.created_at)
            .all()
        )
        cost_detail = [
            {
                "task_type":                   r.task_type,
                "model":                       r.model,
                "input_tokens":                r.input_tokens,
                "output_tokens":               r.output_tokens,
                "cache_read_input_tokens":     r.cache_read_input_tokens,
                "cache_creation_input_tokens": r.cache_creation_input_tokens,
                "cost_usd":                    r.cost_usd,
                "created_at":                  r.created_at.isoformat() if r.created_at else None,
            }
            for r in cost_rows
        ]

        return {
            "id":           item.id,
            "thread_id":    item.thread_id,
            "customer_msg": item.customer_msg,
            "draft_text":   item.draft_text,
            "intent":       item.intent,
            "confidence":   item.confidence,
            "status":       item.status,
            "cost_usd":     item.cost_usd,
            "created_at":   item.created_at.isoformat() if item.created_at else None,
            "resolved_at":  item.resolved_at.isoformat() if item.resolved_at else None,
            "features":     features,
            "context":      context,
            "cost_detail":  cost_detail,
        }
