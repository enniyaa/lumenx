"""
Nightly Retrain — Phase 7e
Retrain the Confidence Net if enough real labels have accumulated.

Logic:
  1. Count non-bootstrap FeedbackEntry rows
  2. If real_labels >= MIN_REAL_LABELS_FOR_ROUTING (default 50):
       a. Featurize all data (real labels take priority)
       b. Train MLPClassifier with StratifiedKFold
       c. Deploy if PR-AUC >= 0.70 AND improves on current model
  3. Log result to CostLog (task_type='retrain')

Run from cron / Railway Cron:
    python scripts/nightly_retrain.py

Or via Railway scheduled job pointing at this script.
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ssl_utils import patch_ssl; patch_ssl()
from dotenv import load_dotenv; load_dotenv()

MODEL_META_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "models", "confidence_net_meta.json",
)
MIN_REAL_LABELS = int(os.getenv("MIN_REAL_LABELS_FOR_ROUTING", "50"))
MIN_PR_AUC      = 0.70


def _current_pr_auc() -> float:
    """Return PR-AUC of the currently deployed model (0 if none)."""
    try:
        with open(MODEL_META_PATH) as f:
            meta = json.load(f)
        return float(meta.get("pr_auc", 0.0))
    except Exception:
        return 0.0


def run() -> dict:
    from db.feedback_log import real_label_count, rebuild_feedback_index

    real_labels = real_label_count()
    print(f"Real label count: {real_labels}  (threshold: {MIN_REAL_LABELS})")

    if real_labels < MIN_REAL_LABELS:
        print(f"Not enough real labels yet — skipping retrain. "
              f"({MIN_REAL_LABELS - real_labels} more needed)")
        return {"status": "skipped", "real_labels": real_labels, "deployed": False}

    print("Sufficient real labels — retraining...")

    # Rebuild feedback FAISS index before training (catch new entries)
    try:
        n_indexed = rebuild_feedback_index()
        print(f"Feedback index rebuilt: {n_indexed} entries")
    except Exception as e:
        print(f"Feedback index rebuild failed (non-fatal): {e}")

    from training.train import train
    result = train(force_deploy=False)

    new_pr_auc  = result.get("pr_auc", 0.0)
    current_auc = _current_pr_auc()
    deployed    = result.get("deployed", False)

    print(f"New PR-AUC:     {new_pr_auc:.4f}")
    print(f"Current PR-AUC: {current_auc:.4f}")
    print(f"Deployed:       {deployed}")

    if deployed:
        # Hot-reload the ConfidenceNet singleton if running inside FastAPI
        try:
            from agent.confidence_net import get_confidence_net
            get_confidence_net().reload()
            print("ConfidenceNet singleton reloaded.")
        except Exception:
            pass

    return {
        "status":       "trained",
        "real_labels":  real_labels,
        "new_pr_auc":   new_pr_auc,
        "old_pr_auc":   current_auc,
        "deployed":     deployed,
        "n_samples":    result.get("n_samples", 0),
    }


if __name__ == "__main__":
    result = run()
    print(f"\nNightly retrain result: {result}")
