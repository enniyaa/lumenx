"""
Feedback Log — Phase 6
Manages FeedbackEntry inserts and the FAISS similarity index over customer messages.
Every approved or edited reply is stored here and becomes a few-shot example
for future context windows.
"""

import json
import os
import sys
import numpy as np
from datetime import datetime, timezone
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ssl_utils import patch_ssl, use_hf_cache_only
patch_ssl()
use_hf_cache_only()

FEEDBACK_INDEX_PATH  = os.path.join(os.path.dirname(os.path.dirname(__file__)), "wiki", "feedback_index.faiss")
FEEDBACK_CHUNKS_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "wiki", "feedback_chunks.json")

# ── Insert feedback entry ────────────────────────────────────────────────────

def insert_feedback(
    thread_id: str,
    customer_msg: str,
    draft_text: str,
    final_text: str,
    intent: str,
    approved_as_is: bool,
    thumbs: Optional[str] = None,
    is_bootstrap: bool = False,
) -> int:
    """
    Insert a FeedbackEntry row and return its ID.
    Calculates edit_dist_norm automatically.
    """
    try:
        import Levenshtein
        max_len = max(len(draft_text), len(final_text), 1)
        edit_dist_norm = Levenshtein.distance(draft_text, final_text) / max_len
    except ImportError:
        edit_dist_norm = 0.0 if draft_text == final_text else 0.5

    from db.session import get_db
    from db.models import FeedbackEntry
    with get_db() as db:
        entry = FeedbackEntry(
            thread_id=thread_id,
            customer_msg=customer_msg,
            draft_text=draft_text,
            final_text=final_text,
            intent=intent,
            thumbs=thumbs,
            edit_dist_norm=edit_dist_norm,
            approved_as_is=approved_as_is,
            is_bootstrap=is_bootstrap,
            created_at=datetime.now(timezone.utc),
        )
        db.add(entry)
        db.flush()
        entry_id = entry.id

    # Rebuild index every 10 new entries
    try:
        from db.session import get_db as _get_db
        from db.models import FeedbackEntry as _FE
        with _get_db() as db:
            count = db.query(_FE).count()
        if count % 10 == 0:
            rebuild_feedback_index()
    except Exception:
        pass

    return entry_id


# ── Rebuild FAISS feedback index ─────────────────────────────────────────────

def rebuild_feedback_index() -> int:
    """
    Embed all customer_msg values from FeedbackEntry and build a FAISS index.
    Returns number of entries indexed.
    """
    try:
        import faiss
        from sentence_transformers import SentenceTransformer
        from db.session import get_db
        from db.models import FeedbackEntry
    except ImportError as e:
        raise RuntimeError(f"Missing dependency: {e}")

    with get_db() as db:
        entries = db.query(FeedbackEntry).all()

    if not entries:
        return 0

    model = SentenceTransformer("all-MiniLM-L6-v2")

    texts = [e.customer_msg for e in entries]
    embeddings = model.encode(texts, show_progress_bar=False)
    embeddings = np.array(embeddings, dtype="float32")
    faiss.normalize_L2(embeddings)

    dim   = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    faiss.write_index(index, FEEDBACK_INDEX_PATH)

    chunks = [
        {
            "id":           e.id,
            "thread_id":    e.thread_id,
            "customer_msg": e.customer_msg,
            "final_text":   e.final_text,
            "intent":       e.intent,
            "edit_dist_norm": e.edit_dist_norm or 0.0,
        }
        for e in entries
    ]
    with open(FEEDBACK_CHUNKS_PATH, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

    return len(entries)


# ── Search feedback index ────────────────────────────────────────────────────

def search_feedback(query: str, k: int = 5) -> list[dict]:
    """
    Find top-k similar past approved replies for `query`.
    Returns [] if index doesn't exist yet.
    """
    if not os.path.exists(FEEDBACK_INDEX_PATH):
        return []

    try:
        import faiss
        from sentence_transformers import SentenceTransformer
    except ImportError:
        return []

    model      = SentenceTransformer("all-MiniLM-L6-v2")
    embedding  = model.encode([query], show_progress_bar=False)
    embedding  = np.array(embedding, dtype="float32")
    faiss.normalize_L2(embedding)

    index = faiss.read_index(FEEDBACK_INDEX_PATH)
    with open(FEEDBACK_CHUNKS_PATH, encoding="utf-8") as f:
        chunks = json.load(f)

    k_actual         = min(k, index.ntotal)
    scores, indices  = index.search(embedding, k_actual)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0:
            continue
        item = dict(chunks[idx])
        item["score"] = float(score)
        results.append(item)

    return results


# ── Real label count ─────────────────────────────────────────────────────────

def real_label_count() -> int:
    """Return number of non-bootstrap FeedbackEntry rows."""
    try:
        from db.session import get_db
        from db.models import FeedbackEntry
        with get_db() as db:
            return db.query(FeedbackEntry).filter(FeedbackEntry.is_bootstrap == False).count()
    except Exception:
        return 0
