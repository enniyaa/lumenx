"""
Auto-Reply Router — Phase 8
Gate between auto-send and human review using the confidence score.

Logic:
  IF model_deployed AND real_labels >= MIN_REAL_LABELS AND confidence >= CONFIDENCE_THRESHOLD
    → POST reply to LumenX API, record as auto_sent
  ELSE
    → INSERT into review_queue for human review
"""

import json
import os
import sys
import requests
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ssl_utils import patch_ssl
patch_ssl()

from dotenv import load_dotenv
load_dotenv()

LUMENX_BASE_URL    = os.getenv("LUMENX_BASE_URL", "https://lumenx-demo.up.railway.app")
LUMENX_ADMIN_TOKEN = os.getenv("LUMENX_ADMIN_TOKEN", "")


def _headers():
    return {"X-Admin-Token": LUMENX_ADMIN_TOKEN, "Content-Type": "application/json"}


# ── Public: send reply ────────────────────────────────────────────────────────

def send_reply(
    thread_id: str,
    text: str,
    draft_source: str = "agent",
    confidence: float = 0.5,
) -> bool:
    """POST a reply to the LumenX API. Returns True on HTTP 200/201."""
    try:
        resp = requests.post(
            f"{LUMENX_BASE_URL}/api/admin/threads/{thread_id}/reply",
            headers=_headers(),
            json={"text": text, "draft_source": draft_source, "confidence": confidence},
            verify=False,
            timeout=10,
        )
        return resp.status_code in (200, 201)
    except Exception:
        return False


# ── Public: route ─────────────────────────────────────────────────────────────

def route(
    thread_id: str,
    customer_msg: str,
    draft_text: str,
    confidence: float,
    intent: str,
    features: dict,
    context_json: str,
) -> dict:
    """
    Decide whether to auto-send or queue the draft.

    Returns:
        {"action": "auto_sent" | "queued",  "queue_id": int | None}
    """
    from agent.confidence_net import get_confidence_net
    from db.feedback_log import real_label_count

    net         = get_confidence_net()
    real_labels = real_label_count()
    min_labels  = int(os.getenv("MIN_REAL_LABELS_FOR_ROUTING", "50"))
    threshold   = float(os.getenv("CONFIDENCE_THRESHOLD", "0.90"))
    model_ready = net.is_loaded and real_labels >= min_labels

    if model_ready and confidence >= threshold:
        success = send_reply(thread_id, draft_text, draft_source="agent", confidence=confidence)
        if success:
            _record_auto_sent(thread_id, customer_msg, draft_text, confidence, intent, features, context_json)
            return {"action": "auto_sent", "queue_id": None}
        # Fall through to queue on send failure

    queue_id = _enqueue(thread_id, customer_msg, draft_text, confidence, intent, features, context_json)
    return {"action": "queued", "queue_id": queue_id}


# ── Private helpers ───────────────────────────────────────────────────────────

def _enqueue(thread_id, customer_msg, draft_text, confidence, intent, features, context_json) -> int:
    from db.session import get_db
    from db.models import ReviewQueue
    with get_db() as db:
        item = ReviewQueue(
            thread_id=thread_id,
            customer_msg=customer_msg,
            draft_text=draft_text,
            confidence=confidence,
            intent=intent,
            features_json=json.dumps(features),
            context_json=context_json,
            status="pending",
            created_at=datetime.now(timezone.utc),
        )
        db.add(item)
        db.flush()
        return item.id


def _record_auto_sent(thread_id, customer_msg, draft_text, confidence, intent, features, context_json):
    """Record an auto-sent reply in ReviewQueue for audit / dashboard."""
    from db.session import get_db
    from db.models import ReviewQueue
    now = datetime.now(timezone.utc)
    with get_db() as db:
        db.add(ReviewQueue(
            thread_id=thread_id,
            customer_msg=customer_msg,
            draft_text=draft_text,
            confidence=confidence,
            intent=intent,
            features_json=json.dumps(features),
            context_json=context_json,
            status="auto_sent",
            created_at=now,
            resolved_at=now,
        ))
