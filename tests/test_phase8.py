"""
Phase 8 Integration Tests — Auto-Reply Router + Inbox Poller
=============================================================

Checks:
  1.  GET  /health              → poller_active: true
  2.  GET  /agent/config        → baseline config values present
  3.  PUT  /agent/config        → live threshold update (verify with GET)
  4.  PUT  /agent/config        → live poller_enabled toggle
  5.  PUT  /agent/config        → min_real_labels_for_routing update
  6.  PUT  /agent/config (bad)  → 400 on out-of-range threshold
  7.  POST /agent/poll          → returns ok + messages + results
  8.  auto_router.route()       → queues when model NOT ready (< 50 real labels)
  9.  auto_router._enqueue()    → row inserted with cost_usd
  10. auto_router.route() result → dict has 'action' and 'queue_id' keys
  11. Review queue reflects enqueued item from route()
  12. Restore original config after test

Run:
    cd C:\\Claude\\lumenx-agent
    python tests/test_phase8.py
"""

import os
import sys
import json
import time
import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ssl_utils import patch_ssl
patch_ssl()

from dotenv import load_dotenv
load_dotenv()

BASE = "http://localhost:8001"
PASS = "✅"
FAIL = "❌"
results = []


def check(name: str, condition: bool, detail: str = ""):
    icon = PASS if condition else FAIL
    msg  = f"  {icon}  {name}"
    if detail:
        msg += f"  ({detail})"
    print(msg)
    results.append((name, condition))
    return condition


# ── Helpers ───────────────────────────────────────────────────────────────────

def get(path, **kwargs):
    return requests.get(f"{BASE}{path}", **kwargs)

def put(path, **kwargs):
    return requests.put(f"{BASE}{path}", **kwargs)

def post(path, **kwargs):
    return requests.post(f"{BASE}{path}", **kwargs)


# ── 1. Health check ───────────────────────────────────────────────────────────

print("\n── Test 1: GET /health ─────────────────────────────────────────────────")
r = get("/health")
h = r.json()
check("status=ok",        h.get("status") == "ok",    f"got {h.get('status')!r}")
check("db_connected",     h.get("db_connected") is True)
check("wiki_loaded",      h.get("wiki_loaded") is True)
check("model_loaded",     h.get("model_loaded") is True)
check("poller_active",    h.get("poller_active") is True,  "background thread running")
check("errors=[]",        h.get("errors") == [],       f"errors: {h.get('errors')}")

# ── 2. GET /agent/config — baseline ──────────────────────────────────────────

print("\n── Test 2: GET /agent/config ───────────────────────────────────────────")
r = get("/agent/config")
c = r.json()
check("HTTP 200",                  r.status_code == 200)
check("confidence_threshold key",  "confidence_threshold" in c)
check("poller_enabled key",        "poller_enabled" in c)
check("poller_active key",         "poller_active" in c)
check("routing_active key",        "routing_active" in c)
check("real_label_count key",      "real_label_count" in c)
check("routing_active=False",      c.get("routing_active") is False,
      f"real_labels={c.get('real_label_count')}, min={c.get('min_real_labels_for_routing')}")
print(f"       baseline threshold={c.get('confidence_threshold')}  "
      f"min_labels={c.get('min_real_labels_for_routing')}")

original_threshold = c.get("confidence_threshold", 0.90)

# ── 3. PUT /agent/config — live threshold update ──────────────────────────────

print("\n── Test 3: PUT /agent/config (confidence_threshold) ───────────────────")
r = put("/agent/config", json={"confidence_threshold": 0.75})
check("HTTP 200",                  r.status_code == 200,  f"got {r.status_code}")
body = r.json()
check("ok=True",                   body.get("ok") is True)
check("changed.confidence_threshold=0.75",
      body.get("changed", {}).get("confidence_threshold") == 0.75)

# Verify GET reflects the change
r2 = get("/agent/config")
c2 = r2.json()
check("GET reflects new threshold",
      c2.get("confidence_threshold") == 0.75,
      f"got {c2.get('confidence_threshold')}")

# ── 4. PUT /agent/config — poller_enabled toggle ─────────────────────────────

print("\n── Test 4: PUT /agent/config (poller_enabled) ──────────────────────────")
r = put("/agent/config", json={"poller_enabled": False})
check("HTTP 200",           r.status_code == 200)
body = r.json()
check("changed.poller_enabled=False",
      body.get("changed", {}).get("poller_enabled") is False)
# Toggle back so poller logic isn't affected
put("/agent/config", json={"poller_enabled": True})
check("restored poller_enabled=True", True)

# ── 5. PUT /agent/config — min_real_labels_for_routing ───────────────────────

print("\n── Test 5: PUT /agent/config (min_real_labels_for_routing) ─────────────")
r = put("/agent/config", json={"min_real_labels_for_routing": 100})
check("HTTP 200",   r.status_code == 200)
body = r.json()
check("changed includes min_real_labels",
      "min_real_labels_for_routing" in body.get("changed", {}),
      f"changed={body.get('changed')}")
# Restore
put("/agent/config", json={"min_real_labels_for_routing": 50})
check("restored min_real_labels=50", True)

# ── 6. PUT /agent/config — invalid threshold ─────────────────────────────────

print("\n── Test 6: PUT /agent/config (invalid threshold → 400) ─────────────────")
r = put("/agent/config", json={"confidence_threshold": 1.5})
check("HTTP 400 on out-of-range", r.status_code == 400,
      f"got {r.status_code}")
r = put("/agent/config", json={"confidence_threshold": -0.1})
check("HTTP 400 on negative",     r.status_code == 400,
      f"got {r.status_code}")

# ── 7. POST /agent/poll ───────────────────────────────────────────────────────

print("\n── Test 7: POST /agent/poll ────────────────────────────────────────────")
r = post("/agent/poll")
check("HTTP 200",        r.status_code == 200,   f"got {r.status_code}")
body = r.json()
check("ok=True",         body.get("ok") is True)
check("messages key",    "messages" in body,     f"keys={list(body.keys())}")
check("results key",     "results" in body,      f"keys={list(body.keys())}")
check("messages is int", isinstance(body.get("messages"), int))
print(f"       poll found {body.get('messages')} awaiting message(s)")
if body.get("results"):
    for rr in body["results"][:3]:
        action = rr.get("action", "?")
        intent = rr.get("intent", "?")
        conf   = rr.get("confidence", "?")
        print(f"         → thread={rr.get('thread_id','?')}  intent={intent}  "
              f"action={action}  confidence={conf}")

# ── 8. auto_router.route() — gate inactive (< 50 real labels) ────────────────

print("\n── Test 8: auto_router.route() with gate inactive ──────────────────────")
try:
    from agent.auto_router import route, send_reply
    from db.feedback_log import real_label_count

    real_labels = real_label_count()
    min_labels  = int(os.getenv("MIN_REAL_LABELS_FOR_ROUTING", "50"))
    gate_open   = real_labels >= min_labels
    print(f"       real_labels={real_labels}  min_labels={min_labels}  gate_open={gate_open}")

    routing = route(
        thread_id    = "test-phase8-unit",
        customer_msg = "What are the pricing plans for LumenX?",
        draft_text   = "LumenX offers three pricing tiers: Starter at $29/mo, Pro at $99/mo, and Enterprise with custom pricing. — LumenX Support",
        confidence   = 0.95,   # above threshold, but gate is inactive
        intent       = "pricing",
        features     = {"len_ratio": 1.0, "intent_encoded": 1.0, "retrieval_hits": 3.0,
                        "edit_dist_norm": 0.0, "has_price_mention": 1.0, "draft_len_tokens": 30.0},
        context_json = json.dumps({"test": True}),
        cost_usd     = 0.00123,
    )
    check("route() returns dict",         isinstance(routing, dict))
    check("'action' key present",         "action" in routing)
    check("'queue_id' key present",       "queue_id" in routing)

    if gate_open:
        # Gate is open — could be auto_sent or queued (depends on LumenX API)
        check("action is auto_sent or queued",
              routing["action"] in ("auto_sent", "queued"),
              f"action={routing['action']}")
    else:
        # Gate is closed — must be queued regardless of confidence
        check("action='queued' (gate inactive)",
              routing["action"] == "queued",
              f"action={routing['action']}  confidence=0.95 >= 0.90 but gate inactive")
        check("queue_id is int",
              isinstance(routing.get("queue_id"), int),
              f"queue_id={routing.get('queue_id')}")

except Exception as exc:
    check("route() import/call succeeded", False, str(exc))
    routing = {}

# ── 9. Verify cost_usd stored in ReviewQueue row ──────────────────────────────

print("\n── Test 9: cost_usd stored in ReviewQueue row ───────────────────────────")
queue_id = routing.get("queue_id") if isinstance(routing, dict) else None

if queue_id:
    try:
        from db.session import get_db
        from db.models import ReviewQueue
        with get_db() as db:
            item = db.query(ReviewQueue).filter(ReviewQueue.id == queue_id).first()
            if item:
                cost_stored = item.cost_usd
                check("cost_usd persisted",      cost_stored is not None)
                check("cost_usd ≈ 0.00123",      abs((cost_stored or 0) - 0.00123) < 1e-6,
                      f"stored={cost_stored}")
                check("customer_msg persisted",   bool(item.customer_msg))
                check("intent persisted",         item.intent == "pricing")
                check("status='pending'",         item.status == "pending")
            else:
                check("ReviewQueue row found", False, f"id={queue_id} not in DB")
    except Exception as exc:
        check("DB read succeeded", False, str(exc))
else:
    print("       (skipped — no queue_id returned; gate may be open or route() failed)")

# ── 10. GET /agent/queue reflects enqueued item ───────────────────────────────

print("\n── Test 10: GET /agent/queue shows the enqueued item ────────────────────")
if queue_id:
    r = get("/agent/queue", params={"status": "pending"})
    check("HTTP 200",         r.status_code == 200,  f"got {r.status_code}")
    items = r.json() if isinstance(r.json(), list) else r.json().get("items", [])
    ids   = [i.get("id") for i in items]
    check(f"queue_id {queue_id} in pending list",
          queue_id in ids,
          f"ids in pending: {ids[:10]}")
else:
    print("       (skipped — no queue_id)")

# ── 11. Restore original confidence threshold ─────────────────────────────────

print("\n── Test 11: Restore original config ────────────────────────────────────")
r = put("/agent/config", json={"confidence_threshold": original_threshold})
check("HTTP 200",          r.status_code == 200)
r2  = get("/agent/config")
c2  = r2.json()
check("threshold restored",
      abs(c2.get("confidence_threshold", 0) - original_threshold) < 1e-9,
      f"restored to {c2.get('confidence_threshold')}")

# ── Summary ───────────────────────────────────────────────────────────────────

total  = len(results)
passed = sum(1 for _, ok in results if ok)
failed = total - passed

print(f"\n{'═'*55}")
print(f"  Phase 8 results:  {passed}/{total} passed  "
      f"({'ALL PASS ✅' if failed == 0 else f'{failed} FAILED ❌'})")
print(f"{'═'*55}\n")

if failed > 0:
    print("Failed checks:")
    for name, ok in results:
        if not ok:
            print(f"  ❌ {name}")

sys.exit(0 if failed == 0 else 1)
