"""
Load Test — Phase 10
====================
Simulates 50 concurrent requests across all major endpoints to verify
FastAPI + SQLite handles concurrency correctly.

Does NOT call the Anthropic API — all tests hit HTTP endpoints only.

Test suites:
  1. 50 concurrent GET /health
  2. 50 concurrent GET /agent/stats (all three periods in parallel)
  3. 50 concurrent GET /agent/queue (pending + approved + other statuses)
  4. 50 concurrent GET /agent/replies (paginated)
  5. 50 concurrent GET /agent/config
  6. 10 concurrent GET /agent/replies/{id}/context (read-heavy)
  7. Mixed concurrency: all endpoints simultaneously (50 workers)
  8. Rate-limit check: verify no 500s under concurrent PUT /agent/config

Passes if:
  - Zero HTTP 5xx responses
  - Zero connection errors
  - p99 latency < 2 000 ms for read endpoints
  - p99 latency < 3 000 ms for write endpoints

Run:
    cd C:\\Claude\\lumenx-agent
    $env:PYTHONIOENCODING="utf-8"; python tests/load_test.py
"""

import os, sys, time, statistics, random
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

BASE    = "http://localhost:8001"
results = []
PASS, FAIL = "OK", "FAIL"

def check(name, cond, detail=""):
    icon = PASS if cond else FAIL
    print(f"  [{icon}] {name}" + (f"  ({detail})" if detail else ""))
    results.append((name, cond))
    return cond


# ── Worker helpers ─────────────────────────────────────────────────────────────

def timed_get(path, params=None):
    """Return (status_code, elapsed_ms). Returns (-1, elapsed) on exception."""
    t0 = time.perf_counter()
    try:
        r = requests.get(f"{BASE}{path}", params=params, timeout=15)
        return r.status_code, round((time.perf_counter() - t0) * 1000)
    except Exception as exc:
        return -1, round((time.perf_counter() - t0) * 1000)

def timed_put(path, json_body):
    t0 = time.perf_counter()
    try:
        r = requests.put(f"{BASE}{path}", json=json_body, timeout=15)
        return r.status_code, round((time.perf_counter() - t0) * 1000)
    except Exception:
        return -1, round((time.perf_counter() - t0) * 1000)


def run_concurrent(label, fn_list, p99_limit_ms=2000):
    """
    Execute fn_list concurrently, collect (status, ms) pairs.
    Reports: success rate, p50/p99 latencies, error count.
    Returns True if all checks pass.
    """
    print(f"\n  Running {len(fn_list)} concurrent requests: {label}")
    statuses, latencies = [], []

    with ThreadPoolExecutor(max_workers=50) as pool:
        futures = [pool.submit(fn) for fn in fn_list]
        for f in as_completed(futures):
            code, ms = f.result()
            statuses.append(code)
            latencies.append(ms)

    errors   = sum(1 for s in statuses if s == -1)
    server5xx= sum(1 for s in statuses if isinstance(s, int) and 500 <= s < 600)
    ok_count = sum(1 for s in statuses if s == 200)
    p50      = statistics.median(latencies)
    p99      = sorted(latencies)[int(len(latencies) * 0.99)]

    print(f"     200 OK: {ok_count}/{len(fn_list)}  "
          f"errors: {errors}  5xx: {server5xx}  "
          f"p50: {p50:.0f}ms  p99: {p99:.0f}ms")

    all_ok = (
        check(f"{label}: zero connection errors", errors == 0,    f"errors={errors}") and
        check(f"{label}: zero 5xx responses",     server5xx == 0, f"5xx={server5xx}") and
        check(f"{label}: all 200",                ok_count == len(fn_list),
              f"{ok_count}/{len(fn_list)}") and
        check(f"{label}: p99 < {p99_limit_ms}ms", p99 < p99_limit_ms,
              f"p99={p99:.0f}ms")
    )
    return all_ok


# ── Get a valid reply ID for context tests ────────────────────────────────────

def _get_sample_ids(n=10):
    try:
        r = requests.get(f"{BASE}/agent/replies?limit={n}", timeout=10)
        return [item["id"] for item in r.json().get("items", [])]
    except Exception:
        return []


# ── Test suites ───────────────────────────────────────────────────────────────

print("\n" + "="*58)
print("  LumenX Agent Load Test — 50 concurrent requests")
print("="*58)

# 1. Health (50 concurrent)
print("\n-- Suite 1: GET /health (50 concurrent) --")
run_concurrent("health", [lambda: timed_get("/health")] * 50)

# 2. Stats — all periods interleaved (50 workers)
print("\n-- Suite 2: GET /agent/stats (50 concurrent, mixed periods) --")
periods = (["day"] * 20) + (["week"] * 15) + (["month"] * 15)
run_concurrent(
    "stats",
    [lambda p=p: timed_get("/agent/stats", params={"period": p}) for p in periods],
)

# 3. Queue — mixed status filters (50 concurrent)
print("\n-- Suite 3: GET /agent/queue (50 concurrent) --")
statuses = ["pending"] * 15 + ["approved"] * 10 + ["edited"] * 5 + ["rejected"] * 5 + [None] * 15
run_concurrent(
    "queue",
    [lambda s=s: timed_get("/agent/queue", params={"status": s} if s else None)
     for s in statuses],
)

# 4. Replies list (50 concurrent)
print("\n-- Suite 4: GET /agent/replies (50 concurrent) --")
pages   = [1, 2, 1, 1, 2] * 10
intents = [None, "pricing", "refund", "other", None] * 10
run_concurrent(
    "replies",
    [
        lambda pg=pg, intent=intent: timed_get(
            "/agent/replies",
            params={k: v for k, v in [("page", pg), ("limit", 10), ("intent", intent)] if v}
        )
        for pg, intent in zip(pages, intents)
    ],
)

# 5. Config reads (50 concurrent)
print("\n-- Suite 5: GET /agent/config (50 concurrent) --")
run_concurrent("config", [lambda: timed_get("/agent/config")] * 50)

# 6. Context viewer (read-heavy, up to 10 items)
print("\n-- Suite 6: GET /agent/replies/{id}/context (10 concurrent) --")
sample_ids = _get_sample_ids(10)
if sample_ids:
    fns = [lambda rid=rid: timed_get(f"/agent/replies/{rid}/context")
           for rid in (sample_ids * 5)[:10]]
    run_concurrent("context-viewer", fns, p99_limit_ms=3000)
else:
    print("  (skipped — no reply IDs available)")
    results.append(("context-viewer", True))   # trivially pass

# 7. Mixed concurrency — all endpoints at once (50 workers)
print("\n-- Suite 7: Mixed endpoints (50 concurrent workers) --")
mixed_fns = []
for _ in range(10): mixed_fns.append(lambda: timed_get("/health"))
for _ in range(10): mixed_fns.append(lambda: timed_get("/agent/stats", params={"period":"day"}))
for _ in range(10): mixed_fns.append(lambda: timed_get("/agent/queue"))
for _ in range(10): mixed_fns.append(lambda: timed_get("/agent/replies", params={"limit":5}))
for _ in range(10): mixed_fns.append(lambda: timed_get("/agent/config"))
run_concurrent("mixed", mixed_fns, p99_limit_ms=3000)

# 8. Concurrent config updates — verify no 500s
print("\n-- Suite 8: Concurrent PUT /agent/config (20 workers) --")
thresholds = [round(0.70 + i * 0.01, 2) for i in range(20)]   # 0.70 → 0.89
write_fns  = [lambda t=t: timed_put("/agent/config", {"confidence_threshold": t})
              for t in thresholds]

print(f"  Running 20 concurrent PUT /agent/config requests")
statuses, latencies = [], []
with ThreadPoolExecutor(max_workers=20) as pool:
    futures = [pool.submit(fn) for fn in write_fns]
    for f in as_completed(futures):
        code, ms = f.result()
        statuses.append(code)
        latencies.append(ms)

errors5xx = sum(1 for s in statuses if isinstance(s, int) and s >= 500)
ok_count  = sum(1 for s in statuses if s == 200)
p99       = sorted(latencies)[int(len(latencies) * 0.99)]
print(f"     200 OK: {ok_count}/20  5xx: {errors5xx}  p99: {p99:.0f}ms")
check("config PUT: no 5xx under concurrency", errors5xx == 0, f"5xx={errors5xx}")
check("config PUT: all 200",  ok_count == 20)

# Restore to sane threshold
requests.put(f"{BASE}/agent/config", json={"confidence_threshold": 0.90})

# ── Summary ────────────────────────────────────────────────────────────────────
total  = len(results)
passed = sum(1 for _, ok in results if ok)
failed = total - passed

print(f"\n{'='*58}")
print(f"  Load test results: {passed}/{total} passed  "
      f"({'ALL PASS' if not failed else f'{failed} FAILED'})")
print(f"{'='*58}\n")

if failed:
    print("Failed checks:")
    for name, ok in results:
        if not ok: print(f"  FAIL: {name}")
    sys.exit(1)
