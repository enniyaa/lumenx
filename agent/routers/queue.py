"""
Review Queue Router — Phase 5
All endpoints for the human review workflow.

Endpoints:
  GET  /agent/queue                    — list pending queue items
  POST /agent/queue/{id}/approve       — approve draft as-is → send to LumenX
  POST /agent/queue/{id}/edit          — send edited text → save training example
  POST /agent/queue/{id}/reject        — discard draft (human writes from scratch)
  POST /agent/queue/{id}/feedback      — thumbs up/down on a resolved item

Every approve / edit inserts a FeedbackEntry row (the Phase 6 training signal).
"""

import json
import os
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

# ── Router ────────────────────────────────────────────────────────────────────

router = APIRouter(prefix="/agent/queue", tags=["review-queue"])


# ── Request / response models ─────────────────────────────────────────────────

class EditBody(BaseModel):
    edited_text: str

class FeedbackBody(BaseModel):
    thumbs: str  # "up" | "down"

class QueueItem(BaseModel):
    id:            int
    thread_id:     str
    customer_msg:  Optional[str]
    draft_text:    str
    confidence:    Optional[float]
    intent:        Optional[str]
    cost_usd:      Optional[float]
    status:        str
    created_at:    str

    class Config:
        from_attributes = True


# ── DB dependency ─────────────────────────────────────────────────────────────

def get_db_session():
    from db.session import get_db
    with get_db() as db:
        yield db


# ── Helpers ───────────────────────────────────────────────────────────────────

def _get_item_or_404(item_id: int, db: Session):
    from db.models import ReviewQueue
    item = db.query(ReviewQueue).filter(ReviewQueue.id == item_id).first()
    if not item:
        raise HTTPException(status_code=404, detail=f"Queue item {item_id} not found")
    return item


def _resolve(item, status: str, db: Session):
    """Mark a queue item as resolved (approved / edited / rejected)."""
    item.status      = status
    item.resolved_at = datetime.now(timezone.utc)


def _insert_feedback(
    item,
    final_text: str,
    approved_as_is: bool,
    db: Session,
) -> int:
    """Insert a FeedbackEntry row for the resolved item. Returns entry ID."""
    from db.models import FeedbackEntry

    try:
        import Levenshtein
        max_len        = max(len(item.draft_text), len(final_text), 1)
        edit_dist_norm = Levenshtein.distance(item.draft_text, final_text) / max_len
    except ImportError:
        edit_dist_norm = 0.0 if item.draft_text == final_text else 0.5

    entry = FeedbackEntry(
        thread_id=item.thread_id,
        customer_msg=item.customer_msg or "",
        draft_text=item.draft_text,
        final_text=final_text,
        intent=item.intent or "other",
        thumbs=None,
        edit_dist_norm=edit_dist_norm,
        approved_as_is=approved_as_is,
        is_bootstrap=False,
        created_at=datetime.now(timezone.utc),
    )
    db.add(entry)
    db.flush()   # populates entry.id before commit
    return entry.id


def _send_to_lumenx(thread_id: str, text: str, confidence: float) -> bool:
    """POST the reply to the LumenX platform API."""
    import requests
    from ssl_utils import patch_ssl
    patch_ssl()

    base  = os.getenv("LUMENX_BASE_URL", "https://lumenx-demo.up.railway.app")
    token = os.getenv("LUMENX_ADMIN_TOKEN", "")
    try:
        resp = requests.post(
            f"{base}/api/admin/threads/{thread_id}/reply",
            headers={"X-Admin-Token": token, "Content-Type": "application/json"},
            json={"text": text, "draft_source": "agent", "confidence": confidence or 0.5},
            verify=False,
            timeout=10,
        )
        return resp.status_code in (200, 201)
    except Exception:
        return False


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.get("", response_model=list[QueueItem])
def list_queue(status: str = "pending", limit: int = 50):
    """
    Return queue items filtered by status (default: pending).
    The review UI polls this to refresh its card list.
    """
    from db.session import get_db
    from db.models import ReviewQueue
    with get_db() as db:
        rows = (
            db.query(ReviewQueue)
            .filter(ReviewQueue.status == status)
            .order_by(ReviewQueue.created_at.asc())
            .limit(limit)
            .all()
        )
        return [
            QueueItem(
                id=r.id,
                thread_id=r.thread_id,
                customer_msg=r.customer_msg,
                draft_text=r.draft_text,
                confidence=r.confidence,
                intent=r.intent,
                cost_usd=r.cost_usd,
                status=r.status,
                created_at=r.created_at.isoformat() if r.created_at else "",
            )
            for r in rows
        ]


@router.get("/{item_id}")
def get_queue_item(item_id: int):
    """Return a single queue item including its full context_json."""
    from db.session import get_db
    from db.models import ReviewQueue
    with get_db() as db:
        item = db.query(ReviewQueue).filter(ReviewQueue.id == item_id).first()
        if not item:
            raise HTTPException(404, f"Queue item {item_id} not found")
        ctx = {}
        if item.context_json:
            try:
                ctx = json.loads(item.context_json)
            except Exception:
                ctx = {"raw": item.context_json}
        return {
            "id":           item.id,
            "thread_id":    item.thread_id,
            "customer_msg": item.customer_msg,
            "draft_text":   item.draft_text,
            "confidence":   item.confidence,
            "intent":       item.intent,
            "cost_usd":     item.cost_usd,
            "features":     json.loads(item.features_json) if item.features_json else {},
            "context":      ctx,
            "status":       item.status,
            "created_at":   item.created_at.isoformat() if item.created_at else "",
            "resolved_at":  item.resolved_at.isoformat() if item.resolved_at else None,
        }


@router.post("/{item_id}/approve")
def approve(item_id: int):
    """
    Approve the draft as-is.
      1. Send draft_text to LumenX API  (non-fatal — records sent=false on failure)
      2. Insert FeedbackEntry(approved_as_is=True)
      3. Mark queue item 'approved'
    """
    from db.session import get_db
    with get_db() as db:
        item = _get_item_or_404(item_id, db)

        if item.status != "pending":
            raise HTTPException(400, f"Item {item_id} is already {item.status}")

        # Non-fatal: record the approval regardless of LumenX reachability
        sent = _send_to_lumenx(item.thread_id, item.draft_text, item.confidence or 0.5)
        feedback_id = _insert_feedback(item, item.draft_text, approved_as_is=True, db=db)
        _resolve(item, "approved", db)

    # Trigger feedback index rebuild every 10 entries (background-safe)
    try:
        from db.feedback_log import rebuild_feedback_index
        from db.session import get_db as _gdb
        from db.models import FeedbackEntry as _FE
        with _gdb() as _db:
            count = _db.query(_FE).count()
        if count % 10 == 0:
            rebuild_feedback_index()
    except Exception:
        pass

    return {"ok": True, "action": "approved", "feedback_id": feedback_id, "sent": sent}


@router.post("/{item_id}/edit")
def edit_and_send(item_id: int, body: EditBody):
    """
    Send an edited version of the draft.
      1. Send edited_text to LumenX API  (non-fatal — records sent=false on failure)
      2. Insert FeedbackEntry(approved_as_is=False, final_text=edited_text)
      3. Mark queue item 'edited'
    """
    edited = body.edited_text.strip()
    if not edited:
        raise HTTPException(400, "edited_text must not be empty")

    from db.session import get_db
    with get_db() as db:
        item = _get_item_or_404(item_id, db)

        if item.status != "pending":
            raise HTTPException(400, f"Item {item_id} is already {item.status}")

        sent = _send_to_lumenx(item.thread_id, edited, item.confidence or 0.5)
        feedback_id = _insert_feedback(item, edited, approved_as_is=False, db=db)
        _resolve(item, "edited", db)

    return {"ok": True, "action": "edited", "feedback_id": feedback_id, "sent": sent}


@router.post("/{item_id}/reject")
def reject(item_id: int):
    """
    Discard the agent draft — human will write from scratch.
    No LumenX send. No FeedbackEntry (there's no final text to record).
    """
    from db.session import get_db
    with get_db() as db:
        item = _get_item_or_404(item_id, db)

        if item.status != "pending":
            raise HTTPException(400, f"Item {item_id} is already {item.status}")

        _resolve(item, "rejected", db)

    return {"ok": True, "action": "rejected"}


@router.post("/{item_id}/feedback")
def feedback(item_id: int, body: FeedbackBody):
    """
    Record a thumbs up/down on a resolved queue item.
    Updates the most recent FeedbackEntry for this thread.
    """
    if body.thumbs not in ("up", "down"):
        raise HTTPException(400, "thumbs must be 'up' or 'down'")

    from db.session import get_db
    from db.models import FeedbackEntry, ReviewQueue
    with get_db() as db:
        item = _get_item_or_404(item_id, db)

        # Find the most recent FeedbackEntry for this thread
        entry = (
            db.query(FeedbackEntry)
            .filter(FeedbackEntry.thread_id == item.thread_id)
            .order_by(FeedbackEntry.id.desc())
            .first()
        )
        if entry:
            entry.thumbs = body.thumbs
        else:
            # No FeedbackEntry yet (e.g., thumbs on a pending item preview)
            # Record against the queue item for now
            pass

    return {"ok": True, "action": "feedback", "thumbs": body.thumbs}
