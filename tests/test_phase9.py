"""
Phase 9 Integration Tests — Cost Dashboard
===========================================

Checks:
  1.  GET /agent/stats?period=day     → required fields present
  2.  GET /agent/stats?period=week    → responds 200
  3.  GET /agent/stats?period=month   → responds 200
  4.  GET /agent/stats?period=bad     → 422
  5.  Stats data types correct         → floats, ints, lists
  6.  GET /agent/replies               → total, page, limit, items
  7.  GET /agent/replies?page=1&limit=2
  8.  GET /agent/replies?intent=refund → filters applied
  9.  GET /agent/replies?status=pending
  10. GET /agent/replies/{id}/context  → full detail, context, features
  11. GET /agent/replies/99999/context → 404
  12. GET /dashboard                   → 200 HTML
  13. stats by_intent sums match total_replies
  14. cost_timeline is sorted ascending
  15. cost_by_model keys look like model names

Run:
    cd C:\\Claude\\lumenx-agent
    $env:PYTHONIOENCODING="utf-8"; python tests/test_phase9.py
"""

import os, sys, requests
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ssl_utils import patch_ssl; patch_ssl()
from dotenv import load_dotenv; load_dotenv()

BASE    = "http://localhost:8001"
results = []
PASS, FAIL = "OK", "FAIL"

def check(name, cond, detail=""):
    icon = PASS if cond else FAIL
    print(f"  [{icon}] {name}" + (f"  ({detail})" if detail else ""))
    results.append((name, cond))
    return cond

def get(path, **kw): return requests.get(f"{BASE}{path}", **kw)


# ── 1-3: stats periods ────────────────────────────────────────────────────────
print("\n-- Test 1-3: GET /agent/stats periods --")
for period in ("day","week","month"):
    r = get(f"/agent/stats?period={period}")
    check(f"period={period} HTTP 200", r.status_code == 200, f"got {r.status_code}")

# ── 4: invalid period ─────────────────────────────────────────────────────────
print("\n-- Test 4: invalid period -> 422 --")
r = get("/agent/stats?period=year")
check("period=year -> 422", r.status_code == 422, f"got {r.status_code}")

# ── 5: data types ─────────────────────────────────────────────────────────────
print("\n-- Test 5: stats data types --")
r = get("/agent/stats?period=day")
s = r.json()
check("total_replies is int",       isinstance(s.get("total_replies"), int))
check("total_cost_usd is float",    isinstance(s.get("total_cost_usd"), float))
check("avg_cost_per_reply float",   isinstance(s.get("avg_cost_per_reply"), float))
check("auto_sent is int",           isinstance(s.get("auto_sent"), int))
check("queued is int",              isinstance(s.get("queued"), int))
check("auto_sent_pct is float",     isinstance(s.get("auto_sent_pct"), float))
check("avg_confidence is float",    isinstance(s.get("avg_confidence"), float))
check("by_intent is dict",          isinstance(s.get("by_intent"), dict))
check("cost_by_model is dict",      isinstance(s.get("cost_by_model"), dict))
check("cost_timeline is list",      isinstance(s.get("cost_timeline"), list))
check("from/to present",            "from" in s and "to" in s)
print(f"     total_replies={s['total_replies']}  "
      f"total_cost=${s['total_cost_usd']:.5f}  "
      f"auto_sent={s['auto_sent']}  queued={s['queued']}")

# ── 6: replies list ───────────────────────────────────────────────────────────
print("\n-- Test 6: GET /agent/replies --")
r = get("/agent/replies")
check("HTTP 200", r.status_code == 200)
d = r.json()
check("total key present",     "total" in d)
check("page key present",      "page" in d)
check("limit key present",     "limit" in d)
check("items is list",         isinstance(d.get("items"), list))
if d.get("items"):
    item = d["items"][0]
    for key in ("id","thread_id","created_at","intent","confidence","status","cost_usd","customer_msg"):
        check(f"item has '{key}'", key in item, f"keys={list(item.keys())}")
print(f"     total={d.get('total')}  items_returned={len(d.get('items',[]))}")

# ── 7: pagination ─────────────────────────────────────────────────────────────
print("\n-- Test 7: pagination limit=2 --")
r = get("/agent/replies?page=1&limit=2")
check("HTTP 200", r.status_code == 200)
d2 = r.json()
check("limit respected", len(d2.get("items",[])) <= 2,
      f"got {len(d2.get('items',[]))} items")
check("page=1", d2.get("page") == 1)
check("limit=2 reflected", d2.get("limit") == 2)

# ── 8: intent filter ─────────────────────────────────────────────────────────
print("\n-- Test 8: intent filter --")
r = get("/agent/replies?intent=refund&limit=50")
check("HTTP 200", r.status_code == 200)
items = r.json().get("items", [])
if items:
    wrong = [i for i in items if i.get("intent") != "refund"]
    check("all items have intent=refund", len(wrong) == 0, f"{len(wrong)} wrong intents")
else:
    check("intent filter returns list (may be empty)", True)

# ── 9: status filter ──────────────────────────────────────────────────────────
print("\n-- Test 9: status filter --")
r = get("/agent/replies?status=pending&limit=50")
check("HTTP 200", r.status_code == 200)
items = r.json().get("items", [])
if items:
    wrong = [i for i in items if i.get("status") != "pending"]
    check("all items have status=pending", len(wrong) == 0, f"{len(wrong)} wrong status")
else:
    check("status filter returns list", True)

# ── 10: reply context ─────────────────────────────────────────────────────────
print("\n-- Test 10: GET /agent/replies/{id}/context --")
# Find a real ID first
r = get("/agent/replies?limit=1")
items = r.json().get("items", [])
if items:
    rid = items[0]["id"]
    r2 = get(f"/agent/replies/{rid}/context")
    check(f"HTTP 200 for id={rid}", r2.status_code == 200, f"got {r2.status_code}")
    ctx = r2.json()
    for key in ("id","thread_id","customer_msg","draft_text","intent","confidence",
                "status","cost_usd","created_at","features","context","cost_detail"):
        check(f"context has '{key}'", key in ctx, f"keys={list(ctx.keys())}")
    check("features is dict",     isinstance(ctx.get("features"), dict))
    check("context is dict",      isinstance(ctx.get("context"), dict))
    check("cost_detail is list",  isinstance(ctx.get("cost_detail"), list))
    print(f"     id={ctx['id']}  intent={ctx['intent']}  "
          f"features={list(ctx['features'].keys())}")
else:
    print("     (skipped — no replies in DB)")

# ── 11: 404 on missing id ─────────────────────────────────────────────────────
print("\n-- Test 11: 404 on missing id --")
r = get("/agent/replies/99999/context")
check("HTTP 404", r.status_code == 404, f"got {r.status_code}")

# ── 12: dashboard HTML ────────────────────────────────────────────────────────
print("\n-- Test 12: GET /dashboard HTML --")
import urllib.request
resp = urllib.request.urlopen("http://localhost:8001/dashboard")
body = resp.read().decode("utf-8")
check("HTTP 200",              resp.status == 200)
check("is HTML",               "<html" in body.lower())
check("contains React script", "react" in body.lower())
check("contains Chart.js",     "chart.js" in body.lower())
check("contains /agent/stats", "/agent/stats" in body)
check("contains /agent/replies","/agent/replies" in body)
print(f"     dashboard.html size={len(body)} bytes")

# ── 13: by_intent sums ────────────────────────────────────────────────────────
print("\n-- Test 13: by_intent count sums --")
r  = get("/agent/stats?period=month")   # month to catch everything
s  = r.json()
intent_total = sum(v["count"] for v in s.get("by_intent", {}).values())
total        = s.get("total_replies", 0)
# Intent breakdown comes from ReviewQueue; total_replies also from ReviewQueue
# They should match
check("sum(by_intent counts) == total_replies",
      intent_total == total,
      f"intent_sum={intent_total}  total={total}")

# ── 14: cost_timeline is sorted ───────────────────────────────────────────────
print("\n-- Test 14: cost_timeline sorted ascending --")
tl = s.get("cost_timeline", [])
if len(tl) > 1:
    buckets = [b["bucket"] for b in tl]
    check("cost_timeline sorted", buckets == sorted(buckets), f"buckets={buckets[:5]}")
else:
    check("cost_timeline sorted (single or empty — trivially ok)", True)

# ── 15: cost_by_model keys ────────────────────────────────────────────────────
print("\n-- Test 15: cost_by_model keys look like model names --")
models = list(s.get("cost_by_model", {}).keys())
check("at least one model present", len(models) > 0, f"models={models}")
for m in models:
    check(f"model key non-empty: {m}", len(m) > 0)
print(f"     models: {models}")

# ── Summary ───────────────────────────────────────────────────────────────────
total_c  = len(results)
passed   = sum(1 for _,ok in results if ok)
failed   = total_c - passed
print(f"\n{'='*55}")
print(f"  Phase 9 results:  {passed}/{total_c} passed  "
      f"({'ALL PASS' if not failed else f'{failed} FAILED'})")
print(f"{'='*55}")
if failed:
    for n,ok in results:
        if not ok: print(f"  FAIL: {n}")
sys.exit(0 if not failed else 1)
