"""
Train — Phase 7c
Train the Confidence Net (MLP) on featurized feedback data.

Gate: only saves model if PR-AUC ≥ 0.70 on cross-validation
      (or force_deploy=True for the bootstrap run).
"""

import json
import os
import pickle
import sys
from datetime import datetime, timezone

import numpy as np
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dotenv import load_dotenv
load_dotenv()

from training.featurize import featurize_all

MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "models", "confidence_net.pkl",
)
META_PATH = MODEL_PATH.replace(".pkl", "_meta.json")
MIN_PR_AUC = 0.70


# ── Model factory ─────────────────────────────────────────────────────────────

def _build_model() -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("mlp", MLPClassifier(
            hidden_layer_sizes=(64, 64),
            activation="relu",
            solver="adam",
            max_iter=500,
            random_state=42,
        )),
    ])


# ── Main train function ───────────────────────────────────────────────────────

def train(force_deploy: bool = False) -> dict:
    """
    Train the Confidence Net.

    Args:
        force_deploy: Skip PR-AUC gate and always save (for bootstrap).

    Returns:
        dict with status, n_samples, pr_auc, deployed flag.
    """
    X, y = featurize_all()
    n_samples = len(X)

    if n_samples == 0:
        print("No training data — nothing to train.")
        return {"status": "no_data", "n_samples": 0, "deployed": False}

    n_pos = int(y.sum())
    n_neg = n_samples - n_pos
    print(f"Training on {n_samples} samples  "
          f"({n_pos} positive / {n_neg} negative)")

    if n_samples < 10:
        print("Too few samples for reliable training (< 10).")
        return {"status": "too_few_samples", "n_samples": n_samples, "deployed": False}

    # ── Cross-validation ──────────────────────────────────────────────────────
    n_splits  = max(2, min(5, n_pos, n_neg))
    cv_scores: list[float] = []

    if n_splits >= 2:
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):
            m = _build_model()
            m.fit(X[train_idx], y[train_idx])
            y_prob = m.predict_proba(X[val_idx])[:, 1]
            score  = average_precision_score(y[val_idx], y_prob)
            cv_scores.append(score)
            print(f"  Fold {fold+1}/{n_splits}  PR-AUC={score:.4f}")

        pr_auc = float(np.mean(cv_scores))
        print(f"Mean PR-AUC: {pr_auc:.4f}")
    else:
        # Not enough minority-class samples for CV — skip it
        pr_auc = 1.0
        print("Skipping CV (too few minority-class samples).")

    # ── Final model on all data ───────────────────────────────────────────────
    model = _build_model()
    model.fit(X, y)

    result = {
        "status":     "trained",
        "n_samples":  n_samples,
        "n_positive": n_pos,
        "n_negative": n_neg,
        "pr_auc":     pr_auc,
        "cv_scores":  cv_scores,
        "deployed":   False,
    }

    # ── Deploy gate ───────────────────────────────────────────────────────────
    if force_deploy or pr_auc >= MIN_PR_AUC:
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        with open(MODEL_PATH, "wb") as f:
            pickle.dump(model, f)
        print(f"Model saved → {MODEL_PATH}")

        with open(META_PATH, "w") as f:
            json.dump({
                "pr_auc":     pr_auc,
                "n_samples":  n_samples,
                "trained_at": datetime.now(timezone.utc).isoformat(),
            }, f, indent=2)

        result["deployed"] = True

        # Hot-reload the in-process singleton (if running inside the FastAPI server)
        try:
            from agent.confidence_net import get_confidence_net
            get_confidence_net().reload()
        except Exception:
            pass
    else:
        print(f"Model NOT deployed — PR-AUC {pr_auc:.4f} < {MIN_PR_AUC}")

    _log_retrain(result)
    return result


# ── Cost log helper ───────────────────────────────────────────────────────────

def _log_retrain(result: dict):
    try:
        from db.session import get_db
        from db.models import CostLog
        with get_db() as db:
            db.add(CostLog(
                thread_id=None,
                task_type="retrain",
                model="mlp_confidence_net",
                input_tokens=result.get("n_samples", 0),
                output_tokens=0,
                cache_read_input_tokens=0,
                cache_creation_input_tokens=0,
                cost_usd=0.0,
                created_at=datetime.now(timezone.utc),
            ))
    except Exception:
        pass


# ── CLI entrypoint ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train the LumenX Confidence Net")
    parser.add_argument("--force-deploy", action="store_true",
                        help="Deploy even if PR-AUC < 0.70")
    args = parser.parse_args()

    result = train(force_deploy=args.force_deploy)
    print(f"\nResult: {result}")
