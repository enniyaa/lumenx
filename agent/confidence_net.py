"""
Confidence Net — Phase 7d
MLP inference wrapper: predicts P(reply_will_be_approved_as_is).

Returns 0.5 (always route to human review) when:
  - Model file (models/confidence_net.pkl) doesn't exist yet
  - Fewer than MIN_REAL_LABELS_FOR_ROUTING real labels are in the DB

This ensures the MLP is never used for routing during the cold-start period
(Layers 0–1 of the training data strategy).
"""

import os
import sys
import pickle
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "models", "confidence_net.pkl",
)


class ConfidenceNet:
    """Thin wrapper around the scikit-learn MLP Pipeline."""

    def __init__(self):
        self._model = None
        self._loaded = False
        self._load()

    # ── Private ───────────────────────────────────────────────────────────────

    def _load(self):
        if os.path.exists(MODEL_PATH):
            try:
                with open(MODEL_PATH, "rb") as f:
                    self._model = pickle.load(f)
                self._loaded = True
            except Exception:
                self._loaded = False
        else:
            self._loaded = False

    # ── Public API ────────────────────────────────────────────────────────────

    def predict(self, features: dict) -> float:
        """
        Returns P(approved_as_is) ∈ [0, 1].

        Returns 0.5 (human review) if:
          - model not loaded
          - real_label_count < MIN_REAL_LABELS_FOR_ROUTING
          - any inference error
        """
        if not self._loaded:
            return 0.5

        min_labels = int(os.getenv("MIN_REAL_LABELS_FOR_ROUTING", "50"))
        try:
            from db.feedback_log import real_label_count
            if real_label_count() < min_labels:
                return 0.5
        except Exception:
            return 0.5

        try:
            import numpy as np
            from training.featurize import features_to_array
            X = features_to_array(features).reshape(1, -1)
            # Pipeline.predict_proba: column 1 = P(class=1 = approved_as_is)
            proba = self._model.predict_proba(X)[0][1]
            return float(proba)
        except Exception:
            return 0.5

    def reload(self):
        """Reload model from disk after a nightly retrain."""
        self._load()

    @property
    def is_loaded(self) -> bool:
        return self._loaded


# ── Global singleton ──────────────────────────────────────────────────────────

_net: Optional[ConfidenceNet] = None


def get_confidence_net() -> ConfidenceNet:
    """Return the process-wide ConfidenceNet singleton."""
    global _net
    if _net is None:
        _net = ConfidenceNet()
    return _net
