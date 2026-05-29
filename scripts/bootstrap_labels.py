"""
Bootstrap Labels â€” Phase 7b
Generates the initial MLP training dataset from the LumenX demo export.

Strategy (from CLAUDE.md):
  Layer 0 (Day 0):
  1. GET /api/admin/export â€” fetch all demo conversations
  2. Conversations where last_admin_at is set â†’ label as approved_as_is=1
     (Proxy: seeded admin replies were "accepted by the demo")
  3. Create artificial negatives by perturbing approved replies:
       - Random word deletion (30%)
       - Random word insertion of filler words (30%)
       - Truncation to first 60% of chars (40%)
  4. Mark all as is_bootstrap=True
  5. Train the initial model with force_deploy=True
     (model is scoring-only until â‰¥50 real labels accumulate)

Run once at launch:
    python scripts/bootstrap_labels.py

Safe to re-run â€” clears existing bootstrap rows before inserting.
"""

import os
import sys
import random
import string

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ssl_utils import patch_ssl; patch_ssl()
from dotenv import load_dotenv; load_dotenv()

import requests
from datetime import datetime, timezone

LUMENX_BASE_URL    = os.getenv("LUMENX_BASE_URL", "https://lumenx-demo.up.railway.app").strip()
LUMENX_ADMIN_TOKEN = os.getenv("LUMENX_ADMIN_TOKEN", "")

FILLER_WORDS = [
    "actually", "basically", "literally", "just", "really", "very",
    "quite", "perhaps", "maybe", "honestly", "frankly", "well",
]

random.seed(42)


# â”€â”€ Perturbation helpers â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def _perturb(text: str) -> str:
    """Generate a plausibly bad version of `text` (label=0)."""
    strategy = random.choice(["delete", "insert", "truncate"])
    words = text.split()

    if strategy == "delete" and len(words) > 8:
        # Remove ~25% of words randomly
        n_delete = max(1, len(words) // 4)
        indices  = sorted(random.sample(range(len(words)), n_delete), reverse=True)
        for i in indices:
            words.pop(i)
        return " ".join(words)

    if strategy == "insert":
        # Insert filler words at random positions
        n_insert = max(1, len(words) // 6)
        for _ in range(n_insert):
            pos  = random.randint(0, len(words))
            word = random.choice(FILLER_WORDS)
            words.insert(pos, word)
        return " ".join(words)

    # Truncate to first 55â€“70% of chars
    keep_pct = random.uniform(0.55, 0.70)
    cutoff   = int(len(text) * keep_pct)
    return text[:cutoff].rsplit(" ", 1)[0] + " [incomplete]"


# â”€â”€ Fetch export â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def fetch_export() -> list[dict]:
    """Fetch all conversations from /api/admin/export."""
    try:
        resp = requests.get(
            f"{LUMENX_BASE_URL}/api/admin/export",
            headers={"X-Admin-Token": LUMENX_ADMIN_TOKEN},
            verify=False,
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        # Endpoint returns {"conversations": [...]} or list directly
        if isinstance(data, list):
            return data
        return data.get("conversations", data.get("threads", []))
    except Exception as e:
        print(f"  WARNING: export fetch failed ({e}) â€” using empty list")
        return []


# â”€â”€ Main â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def main(force_train: bool = True) -> dict:
    from db.session import get_db
    from db.models import FeedbackEntry, MLPTrainingRow
    from db.feedback_log import rebuild_feedback_index

    print("Fetching LumenX demo export...")
    conversations = fetch_export()
    print(f"  Got {len(conversations)} conversations")

    # â”€â”€ Clear existing bootstrap rows â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    with get_db() as db:
        deleted = db.query(FeedbackEntry).filter(FeedbackEntry.is_bootstrap == True).delete()
        print(f"  Cleared {deleted} previous bootstrap FeedbackEntry rows")

    # â”€â”€ Insert new bootstrap rows â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    positives = []
    negatives = []
    now       = datetime.now(timezone.utc)

    for conv in conversations:
        # Normalise: export might return threads or conversations
        messages = conv.get("messages", [])
        if not messages:
            continue

        # Find the last admin reply
        admin_msgs = [m for m in messages if m.get("role") == "admin" and m.get("text", "").strip()]
        if not admin_msgs:
            continue

        # Find the customer message that preceded it
        customer_msgs = [m for m in messages if m.get("role") == "customer" and m.get("text", "").strip()]
        if not customer_msgs:
            continue

        admin_text    = admin_msgs[-1].get("text", "").strip()
        customer_text = customer_msgs[-1].get("text", "").strip()
        thread_id     = conv.get("id", f"bootstrap-{len(positives):04d}")
        intent        = conv.get("intent") or "other"   # guard against None

        if len(admin_text) < 20 or len(customer_text) < 5:
            continue

        positives.append(dict(
            thread_id=thread_id,
            customer_msg=customer_text,
            draft_text=admin_text,
            final_text=admin_text,
            intent=intent,
            approved_as_is=True,
            is_bootstrap=True,
            created_at=now,
        ))

    # Generate negatives (perturbed versions of positives)
    for pos in positives:
        perturbed = _perturb(pos["draft_text"])
        negatives.append(dict(
            thread_id=pos["thread_id"] + "-neg",
            customer_msg=pos["customer_msg"],
            draft_text=perturbed,
            final_text=pos["final_text"],   # what the human would have sent
            intent=pos["intent"],
            approved_as_is=False,
            is_bootstrap=True,
            created_at=now,
        ))

    all_rows = positives + negatives
    print(f"  Inserting {len(positives)} positive + {len(negatives)} negative bootstrap rows...")

    with get_db() as db:
        for row in all_rows:
            try:
                import Levenshtein
                ml  = max(len(row["draft_text"]), len(row["final_text"]), 1)
                edn = Levenshtein.distance(row["draft_text"], row["final_text"]) / ml
            except ImportError:
                edn = 0.0 if row["draft_text"] == row["final_text"] else 0.5
            db.add(FeedbackEntry(
                thread_id=row["thread_id"],
                customer_msg=row["customer_msg"],
                draft_text=row["draft_text"],
                final_text=row["final_text"],
                intent=row["intent"],
                thumbs=None,
                edit_dist_norm=edn,
                approved_as_is=row["approved_as_is"],
                is_bootstrap=row["is_bootstrap"],
                created_at=row["created_at"],
            ))

    total_inserted = len(all_rows)
    print(f"  Inserted {total_inserted} bootstrap FeedbackEntry rows")

    # â”€â”€ Rebuild feedback FAISS index â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    print("Rebuilding feedback FAISS index...")
    n_indexed = rebuild_feedback_index()
    print(f"  Index built with {n_indexed} total entries (bootstrap + real)")

    result = {
        "positives":      len(positives),
        "negatives":      len(negatives),
        "total_inserted": total_inserted,
        "n_indexed":      n_indexed,
        "trained":        False,
    }

    # â”€â”€ Optionally train the initial model â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    if force_train and total_inserted > 0:
        print("\nTraining initial Confidence Net on bootstrap data...")
        from training.train import train
        train_result = train(force_deploy=True)   # bypass PR-AUC gate for bootstrap
        result["trained"]  = train_result.get("deployed", False)
        result["pr_auc"]   = train_result.get("pr_auc", 0.0)
        result["n_samples"]= train_result.get("n_samples", 0)
        print(f"  PR-AUC: {train_result.get('pr_auc', 0):.4f}  deployed={train_result.get('deployed')}")

    return result


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--no-train", action="store_true", help="Skip training after seeding")
    args = p.parse_args()

    result = main(force_train=not args.no_train)
    print(f"\nBootstrap complete: {result}")
