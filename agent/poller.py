"""
Inbox Poller â€” Phase 8
The agent's heartbeat: polls LumenX for unanswered customer messages and runs
the full pipeline (intent â†’ context â†’ draft â†’ confidence â†’ route) for each one.

Pipeline per message:
  1. Classify intent (Haiku)
  2. Greeting fast-path â†’ send direct reply, skip context + Sonnet
  3. Assemble context (wiki + summary + feedback + thread)
  4. Generate draft reply (Sonnet, with prompt caching)
  5. Extract MLP features, predict confidence score
  6. Route: auto-send (if gate open + score â‰¥ threshold) or enqueue for review
  7. Log cost to CostLog

Safety guarantees:
  â€¢ Rate limit: max 1 reply per thread per 5 seconds (in-process dict)
  â€¢ Already-answered check: inbox API returns awaiting_admin=false for answered threads
  â€¢ Error fallback: any LLM exception â†’ enqueue for human review (never silent failure)
  â€¢ Duplicate guard: tracks processed message IDs across the process lifetime

Run via FastAPI startup (background thread) or directly:
  python -m agent.poller --once    # single poll
  python -m agent.poller --loop    # continuous loop (POLL_INTERVAL_SECONDS)
"""

import json
import logging
import os
import sys
import time
import threading
from datetime import datetime, timezone, timedelta
from typing import Optional

import anthropic
import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ssl_utils import patch_ssl
patch_ssl()

from dotenv import load_dotenv
load_dotenv()

logger = logging.getLogger("lumenx.poller")

# â”€â”€ Configuration â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

LUMENX_BASE_URL    = os.getenv("LUMENX_BASE_URL", "https://lumenx-demo.up.railway.app").strip()
LUMENX_ADMIN_TOKEN = os.getenv("LUMENX_ADMIN_TOKEN", "").strip()
POLL_INTERVAL_SEC  = int(os.getenv("POLL_INTERVAL_SECONDS", "5"))
RATE_LIMIT_SEC     = 5       # min seconds between replies on the same thread
LOOKBACK_SEC       = 3600    # inbox lookback window (catch missed msgs after restart)


# â”€â”€ InboxPoller â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

class InboxPoller:
    """
    Polls the LumenX inbox and processes new customer messages end-to-end.
    Designed to run as a single persistent background thread.
    """

    def __init__(self, client: Optional[anthropic.Anthropic] = None):
        self._client        = client or anthropic.Anthropic(
            api_key=os.getenv("ANTHROPIC_API_KEY")
        )
        self._processed_ids: set   = set()   # message IDs handled this session
        self._last_reply_ts: dict  = {}       # thread_id -> epoch seconds of last reply
        self._running              = False
        self._lock                 = threading.Lock()

    # â”€â”€ Public API â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def poll_once(self) -> list[dict]:
        """
        Single poll cycle. Returns list of per-message result dicts.
        Safe to call from any thread.
        """
        entries = self._fetch_inbox()
        results = []
        for entry in entries:
            if not entry.get("awaiting_admin"):
                continue
            thread   = entry.get("thread", {})
            last_msg = entry.get("last_customer_message", {})
            thread_id = thread.get("id", "")
            msg_id    = last_msg.get("id", "")
            msg_text  = (last_msg.get("text") or "").strip()

            if not thread_id or not msg_text:
                continue

            # Duplicate guard
            with self._lock:
                if msg_id and msg_id in self._processed_ids:
                    continue

            # Rate limit
            if self._is_rate_limited(thread_id):
                logger.debug("Rate-limited thread %s â€” skipping", thread_id)
                continue

            result = self._process_message(thread_id, msg_id, msg_text)
            results.append(result)

            # Mark as processed
            with self._lock:
                if msg_id:
                    self._processed_ids.add(msg_id)
                self._last_reply_ts[thread_id] = time.time()

        return results

    def run_loop(self):
        """Blocking poll loop â€” call from a daemon thread."""
        self._running = True
        logger.info("Poller started (interval=%ds)", POLL_INTERVAL_SEC)
        while self._running:
            try:
                results = self.poll_once()
                if results:
                    actions = [r.get("action", "?") for r in results]
                    logger.info("Processed %d message(s): %s", len(results), actions)
            except Exception as exc:
                logger.error("Poll cycle error: %s", exc, exc_info=True)
            time.sleep(POLL_INTERVAL_SEC)

    def stop(self):
        self._running = False

    # â”€â”€ Internal: LumenX I/O â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def _headers(self) -> dict:
        return {"X-Admin-Token": LUMENX_ADMIN_TOKEN, "Content-Type": "application/json"}

    def _fetch_inbox(self) -> list[dict]:
        since = (datetime.now(timezone.utc) - timedelta(seconds=LOOKBACK_SEC)).isoformat()
        try:
            r = requests.get(
                f"{LUMENX_BASE_URL}/api/admin/inbox",
                params={"since": since},
                headers=self._headers(),
                verify=False,
                timeout=10,
            )
            r.raise_for_status()
            data = r.json()
            return data.get("entries", [])
        except Exception as exc:
            logger.warning("Inbox fetch failed: %s", exc)
            return []

    def _send_greeting(self, thread_id: str, reply_text: str) -> bool:
        try:
            r = requests.post(
                f"{LUMENX_BASE_URL}/api/admin/threads/{thread_id}/reply",
                headers=self._headers(),
                json={"text": reply_text, "draft_source": "agent", "confidence": 1.0},
                verify=False,
                timeout=10,
            )
            return r.status_code in (200, 201)
        except Exception:
            return False

    # â”€â”€ Internal: rate limiting â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def _is_rate_limited(self, thread_id: str) -> bool:
        last = self._last_reply_ts.get(thread_id, 0)
        return (time.time() - last) < RATE_LIMIT_SEC

    # â”€â”€ Internal: full pipeline â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def _process_message(self, thread_id: str, msg_id: str, msg_text: str) -> dict:
        """
        Run the complete pipeline for one customer message.
        Any exception â†’ enqueue for human review (never raises).
        """
        logger.info("Processing thread=%s  msg_id=%s  text=%r", thread_id, msg_id, msg_text[:60])

        # â”€â”€ Step 1: Intent classification â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        try:
            from agent.intent_router import classify
            intent_result = classify(msg_text, thread_id=thread_id, client=self._client)
            intent        = intent_result["intent"]
        except Exception as exc:
            logger.error("Intent classification failed: %s", exc)
            return self._fallback_enqueue(thread_id, msg_text, "intent_error", str(exc))

        # â”€â”€ Step 2: Greeting fast-path â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        if intent == "greeting":
            reply = intent_result.get("greeting_reply", "Hello! How can LumenX Support help you today?")
            sent  = self._send_greeting(thread_id, reply)
            logger.info("Greeting fast-path thread=%s  sent=%s", thread_id, sent)
            return {
                "thread_id": thread_id,
                "intent":    "greeting",
                "action":    "greeting_sent" if sent else "greeting_failed",
                "cost_usd":  intent_result.get("cost_usd", 0.0),
            }

        # â”€â”€ Step 3: Context assembly â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        try:
            from agent.context_builder import assemble
            ctx = assemble(thread_id, msg_text, intent, client=self._client)
        except Exception as exc:
            logger.error("Context assembly failed: %s", exc)
            return self._fallback_enqueue(thread_id, msg_text, "context_error", str(exc))

        # â”€â”€ Step 4: Draft generation â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        try:
            from agent.draft_agent import generate_draft
            draft = generate_draft(
                thread_id=thread_id,
                user_message=msg_text,
                intent=intent,
                client=self._client,
                context=ctx,
            )
        except Exception as exc:
            logger.error("Draft generation failed: %s", exc)
            return self._fallback_enqueue(thread_id, msg_text, "draft_error", str(exc))

        # â”€â”€ Step 5: Confidence features + score â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        try:
            from training.featurize import compute_inference_features
            from agent.confidence_net import get_confidence_net

            features   = compute_inference_features(
                draft_text  = draft.text,
                intent      = intent,
                wiki_chunks = ctx.get("wiki_chunks", []),
            )
            net        = get_confidence_net()
            confidence = net.predict(features)
        except Exception as exc:
            logger.warning("Confidence scoring failed (defaulting to 0.5): %s", exc)
            features   = {}
            confidence = 0.5

        # â”€â”€ Step 6: Route â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        try:
            from agent.auto_router import route
            routing = route(
                thread_id   = thread_id,
                customer_msg= msg_text,
                draft_text  = draft.text,
                confidence  = confidence,
                intent      = intent,
                features    = features,
                context_json= draft.context_json,
                cost_usd    = draft.cost_usd,
            )
        except Exception as exc:
            logger.error("Routing failed: %s", exc)
            return self._fallback_enqueue(thread_id, msg_text, "routing_error", str(exc))

        logger.info(
            "Routed thread=%s  intent=%s  confidence=%.3f  action=%s  cost=$%.5f",
            thread_id, intent, confidence, routing["action"], draft.cost_usd,
        )
        return {
            "thread_id":  thread_id,
            "intent":     intent,
            "confidence": confidence,
            "action":     routing["action"],
            "queue_id":   routing.get("queue_id"),
            "cost_usd":   draft.cost_usd,
        }

    def _fallback_enqueue(self, thread_id: str, msg_text: str,
                          error_type: str, error_msg: str) -> dict:
        """
        Error fallback: enqueue for human review so no message is silently dropped.
        """
        logger.warning("Fallback enqueue thread=%s  error=%s: %s",
                       thread_id, error_type, error_msg)
        try:
            from db.session import get_db
            from db.models import ReviewQueue
            with get_db() as db:
                db.add(ReviewQueue(
                    thread_id=thread_id,
                    customer_msg=msg_text,
                    draft_text=f"[Agent error: {error_type}] {error_msg}\n\nPlease handle manually.",
                    confidence=0.0,
                    intent="other",
                    features_json="{}",
                    context_json=json.dumps({"error": error_msg}),
                    cost_usd=0.0,
                    status="pending",
                    created_at=datetime.now(timezone.utc),
                ))
        except Exception as db_exc:
            logger.error("Fallback enqueue DB write failed: %s", db_exc)

        return {
            "thread_id":  thread_id,
            "action":     "fallback_queued",
            "error_type": error_type,
            "error":      error_msg,
        }


# â”€â”€ Global singleton â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

_poller: Optional[InboxPoller] = None
_poller_thread: Optional[threading.Thread] = None


def get_poller() -> Optional[InboxPoller]:
    return _poller


def start_poller(client: Optional[anthropic.Anthropic] = None) -> InboxPoller:
    """Start the background polling thread. Safe to call multiple times."""
    global _poller, _poller_thread
    if _poller_thread and _poller_thread.is_alive():
        return _poller

    _poller = InboxPoller(client=client)
    _poller_thread = threading.Thread(
        target=_poller.run_loop,
        name="inbox-poller",
        daemon=True,   # dies when main process exits
    )
    _poller_thread.start()
    logger.info("Poller thread started (tid=%d)", _poller_thread.ident)
    return _poller


def stop_poller():
    global _poller
    if _poller:
        _poller.stop()


# â”€â”€ CLI entry point â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

if __name__ == "__main__":
    import argparse
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-7s  %(name)s  %(message)s",
    )

    ap = argparse.ArgumentParser(description="LumenX Inbox Poller")
    ap.add_argument("--once",   action="store_true", help="Single poll then exit")
    ap.add_argument("--loop",   action="store_true", help="Continuous polling loop")
    args = ap.parse_args()

    client  = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    poller  = InboxPoller(client=client)

    if args.once:
        results = poller.poll_once()
        print(f"\nProcessed {len(results)} message(s):")
        for r in results:
            print(f"  {r}")
    elif args.loop:
        try:
            poller.run_loop()
        except KeyboardInterrupt:
            print("\nPoller stopped.")
    else:
        ap.print_help()
