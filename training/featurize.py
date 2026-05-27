"""
Featurize — Phase 7a
Extract MLP features from a draft reply for the Confidence Net.

Feature vector (6 floats, always in this order):
  [0] len_ratio          : len(draft_chars) / avg_len_approved_chars   (~0.5–2.0)
  [1] intent_encoded     : INTENT_TO_INT[intent]                       (0–4)
  [2] retrieval_hits     : number of wiki chunks retrieved              (0–5)
  [3] edit_dist_norm     : Levenshtein(draft, final) / max_len         (0–1, 0 at inference)
  [4] has_price_mention  : 1 if "$" or "price" in draft                (0 or 1)
  [5] draft_len_tokens   : len(draft.split())                          (word count)
"""

import os
import sys
from typing import Optional

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ── Constants ────────────────────────────────────────────────────────────────

FEATURE_NAMES = [
    "len_ratio",
    "intent_encoded",
    "retrieval_hits",
    "edit_dist_norm",
    "has_price_mention",
    "draft_len_tokens",
]

INTENT_TO_INT = {
    "greeting":  0,
    "pricing":   1,
    "technical": 2,
    "refund":    3,
    "other":     4,
}

_AVG_APPROVED_LEN_FALLBACK = 300.0   # chars — used when DB is empty


# ── Helpers ───────────────────────────────────────────────────────────────────

def _avg_approved_len() -> float:
    """
    Average character length of all approved-as-is replies in FeedbackEntry.
    Falls back to a constant when the DB has no approved rows yet.
    """
    try:
        from sqlalchemy import func
        from db.session import get_db
        from db.models import FeedbackEntry
        with get_db() as db:
            result = db.query(
                func.avg(func.length(FeedbackEntry.final_text))
            ).filter(
                FeedbackEntry.approved_as_is == True
            ).scalar()
            return float(result) if result else _AVG_APPROVED_LEN_FALLBACK
    except Exception:
        return _AVG_APPROVED_LEN_FALLBACK


# ── Inference features (no edit_dist available yet) ───────────────────────────

def compute_inference_features(
    draft_text: str,
    intent: str,
    wiki_chunks: list,
) -> dict:
    """
    Compute the 6 MLP features for a freshly generated draft.
    edit_dist_norm is 0.0 because we don't know the final text yet.
    """
    avg_len = _avg_approved_len()
    return {
        "len_ratio":         len(draft_text) / max(avg_len, 1.0),
        "intent_encoded":    float(INTENT_TO_INT.get(intent, 4)),
        "retrieval_hits":    float(len(wiki_chunks)),
        "edit_dist_norm":    0.0,   # unknown at inference time
        "has_price_mention": float("$" in draft_text or "price" in draft_text.lower()),
        "draft_len_tokens":  float(len(draft_text.split())),
    }


# ── Feature dict → numpy array ────────────────────────────────────────────────

def features_to_array(features: dict) -> np.ndarray:
    """
    Convert a feature dict to a float32 numpy array in the canonical order
    defined by FEATURE_NAMES.  Missing keys default to 0.0.
    """
    return np.array(
        [features.get(name, 0.0) for name in FEATURE_NAMES],
        dtype=np.float32,
    )


# ── Training features (from a FeedbackEntry ORM object) ──────────────────────

def featurize_entry(entry, avg_len: Optional[float] = None) -> dict:
    """
    Extract the feature dict from a FeedbackEntry ORM object.
    Includes edit_dist_norm from the stored value.
    retrieval_hits defaults to 0 (not stored on FeedbackEntry).
    """
    if avg_len is None:
        avg_len = _avg_approved_len()
    return {
        "len_ratio":         len(entry.draft_text) / max(avg_len, 1.0),
        "intent_encoded":    float(INTENT_TO_INT.get(entry.intent, 4)),
        "retrieval_hits":    0.0,
        "edit_dist_norm":    float(entry.edit_dist_norm) if entry.edit_dist_norm is not None else 0.0,
        "has_price_mention": float("$" in entry.draft_text or "price" in entry.draft_text.lower()),
        "draft_len_tokens":  float(len(entry.draft_text.split())),
    }


# ── Bulk extraction for training ──────────────────────────────────────────────

def featurize_all() -> tuple:
    """
    Build (X, y) training matrices from the database.

    Priority:
      1. MLPTrainingRow rows (already featurized, include real labels)
      2. FeedbackEntry rows (fallback, re-featurize on the fly)

    Returns:
        X : np.ndarray of shape (n_samples, 6), dtype float32
        y : np.ndarray of shape (n_samples,),   dtype int32  (0 or 1)
    """
    from db.session import get_db
    from db.models import MLPTrainingRow, FeedbackEntry

    with get_db() as db:
        # Eagerly materialise everything inside the session (avoid DetachedInstanceError)
        mlp_rows = db.query(MLPTrainingRow).all()

        if mlp_rows:
            X = np.array(
                [[r.len_ratio, r.intent_encoded, r.retrieval_hits,
                  r.edit_dist_norm, r.has_price_mention, r.draft_len_tokens]
                 for r in mlp_rows],
                dtype=np.float32,
            )
            y = np.array([r.label for r in mlp_rows], dtype=np.int32)
            return X, y

        # Fall back: featurize FeedbackEntry rows inside the session
        raw_entries = db.query(FeedbackEntry).all()
        avg_len     = _avg_approved_len()   # calls get_db() internally — fine, different session

        # Convert to feature dicts while session is still open
        entry_feats = [featurize_entry(e, avg_len) for e in raw_entries]
        entry_labels = [int(bool(e.approved_as_is)) for e in raw_entries]

    if not entry_feats:
        return np.zeros((0, 6), dtype=np.float32), np.zeros(0, dtype=np.int32)

    X = np.array(
        [list(f.values()) for f in entry_feats],
        dtype=np.float32,
    )
    y = np.array(entry_labels, dtype=np.int32)
    return X, y
