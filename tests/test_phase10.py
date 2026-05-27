"""
Phase 10 Integration Tests — Hardening & Deployment
=====================================================

Checks:
  1.  agent/retry.py exists and imports cleanly
  2.  call_with_retry succeeds on first attempt (no retries needed)
  3.  call_with_retry retries on APIConnectionError then succeeds
  4.  call_with_retry retries on RateLimitError then succeeds
  5.  call_with_retry raises after exhausting max_attempts
  6.  call_with_retry does NOT retry on 4xx APIStatusError
  7.  call_with_retry retries on 503 APIStatusError then succeeds
  8.  @with_retry decorator works identically
  9.  intent_router imports call_with_retry
  10. draft_agent imports call_with_retry
  11. context_builder imports call_with_retry
  12. Dockerfile exists and contains key directives
  13. railway.toml exists and contains key fields
  14. docs/env-vars.md exists and documents all required vars
  15. GET /health has all required fields
  16. Rate limiting: poller._is_rate_limited() works correctly
  17. Fallback enqueue: poller._fallback_enqueue() writes to DB

Run:
    cd C:\\Claude\\lumenx-agent
    $env:PYTHONIOENCODING="utf-8"; python tests/test_phase10.py
"""

import os, sys, time, unittest.mock as mock
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ssl_utils import patch_ssl; patch_ssl()
from dotenv import load_dotenv; load_dotenv()

import requests

BASE    = "http://localhost:8001"
results = []
PASS, FAIL = "OK", "FAIL"

def check(name, cond, detail=""):
    icon = PASS if cond else FAIL
    print(f"  [{icon}] {name}" + (f"  ({detail})" if detail else ""))
    results.append((name, cond))
    return cond


# ── 1. retry module exists ────────────────────────────────────────────────────
print("\n-- Test 1: retry module import --")
try:
    from agent.retry import call_with_retry, with_retry
    check("agent.retry imports cleanly",      True)
    check("call_with_retry is callable",      callable(call_with_retry))
    check("with_retry is callable",           callable(with_retry))
except ImportError as e:
    check("agent.retry imports cleanly",      False, str(e))
    check("call_with_retry is callable",      False)
    check("with_retry is callable",           False)

# ── 2. Succeeds on first attempt ──────────────────────────────────────────────
print("\n-- Test 2: call_with_retry — success on first attempt --")
calls = []
def _ok_fn():
    calls.append(1)
    return "result"

result = call_with_retry(_ok_fn)
check("returns correct value",     result == "result")
check("called exactly once",       len(calls) == 1, f"calls={len(calls)}")

# ── 3. Retries on APIConnectionError then succeeds ────────────────────────────
print("\n-- Test 3: retry on APIConnectionError --")
import anthropic

attempt_log = []
def _conn_error_then_ok():
    attempt_log.append(len(attempt_log) + 1)
    if len(attempt_log) < 2:
        raise anthropic.APIConnectionError(request=mock.MagicMock())
    return "ok_after_retry"

with mock.patch("time.sleep"):   # skip actual sleep
    res = call_with_retry(_conn_error_then_ok, base_delay=0.001)
check("returned after retry",     res == "ok_after_retry")
check("took exactly 2 attempts",  len(attempt_log) == 2, f"attempts={len(attempt_log)}")

# ── 4. Retries on RateLimitError then succeeds ────────────────────────────────
print("\n-- Test 4: retry on RateLimitError --")
rate_attempts = []
def _rate_limit_then_ok():
    rate_attempts.append(1)
    if len(rate_attempts) < 3:
        raise anthropic.RateLimitError(
            message="rate limit",
            response=mock.MagicMock(status_code=429, headers={}),
            body={}
        )
    return "ok"

with mock.patch("time.sleep"):
    res = call_with_retry(_rate_limit_then_ok, base_delay=0.001)
check("returned after 2 retries",  res == "ok")
check("took exactly 3 attempts",   len(rate_attempts) == 3, f"attempts={len(rate_attempts)}")

# ── 5. Raises after exhausting max_attempts ───────────────────────────────────
print("\n-- Test 5: raises after max_attempts exhausted --")
always_fails_count = []
def _always_fails():
    always_fails_count.append(1)
    raise anthropic.APIConnectionError(request=mock.MagicMock())

raised = False
with mock.patch("time.sleep"):
    try:
        call_with_retry(_always_fails, max_attempts=3, base_delay=0.001)
    except anthropic.APIConnectionError:
        raised = True
check("raised after exhausting attempts", raised)
check("attempted exactly 3 times",        len(always_fails_count) == 3,
      f"attempts={len(always_fails_count)}")

# ── 6. Does NOT retry on 4xx errors ──────────────────────────────────────────
print("\n-- Test 6: no retry on 4xx APIStatusError --")
client_err_count = []
def _client_error():
    client_err_count.append(1)
    raise anthropic.BadRequestError(
        message="bad request",
        response=mock.MagicMock(status_code=400, headers={}),
        body={}
    )

raised_4xx = False
with mock.patch("time.sleep"):
    try:
        call_with_retry(_client_error, max_attempts=3, base_delay=0.001)
    except anthropic.BadRequestError:
        raised_4xx = True
check("4xx raised immediately",    raised_4xx)
check("called exactly once (no retry)", len(client_err_count) == 1,
      f"calls={len(client_err_count)}")

# ── 7. Retries on 5xx APIStatusError ─────────────────────────────────────────
print("\n-- Test 7: retry on 503 server error --")
server_attempts = []
def _server_error_then_ok():
    server_attempts.append(1)
    if len(server_attempts) < 2:
        raise anthropic.APIStatusError(
            message="service unavailable",
            response=mock.MagicMock(status_code=503, headers={}),
            body={}
        )
    return "recovered"

with mock.patch("time.sleep"):
    res = call_with_retry(_server_error_then_ok, base_delay=0.001)
check("recovered after 503",       res == "recovered")
check("attempted exactly 2 times", len(server_attempts) == 2,
      f"attempts={len(server_attempts)}")

# ── 8. @with_retry decorator ──────────────────────────────────────────────────
print("\n-- Test 8: @with_retry decorator --")
deco_calls = []

@with_retry(max_attempts=3, base_delay=0.001)
def decorated_fn():
    deco_calls.append(1)
    if len(deco_calls) < 2:
        raise anthropic.APIConnectionError(request=mock.MagicMock())
    return "decorated_ok"

with mock.patch("time.sleep"):
    res = decorated_fn()
check("decorator returns correct value", res == "decorated_ok")
check("decorator retried correctly",     len(deco_calls) == 2, f"calls={len(deco_calls)}")

# ── 9-11. Retry wired into LLM call sites ─────────────────────────────────────
print("\n-- Tests 9-11: retry wired into LLM modules --")
import ast, pathlib

for name, path, expected_pattern in [
    ("intent_router",   "agent/intent_router.py",   "call_with_retry"),
    ("draft_agent",     "agent/draft_agent.py",     "call_with_retry"),
    ("context_builder", "agent/context_builder.py", "call_with_retry"),
]:
    src = pathlib.Path(f"C:/Claude/lumenx-agent/{path}").read_text(encoding="utf-8")
    check(f"{name} uses call_with_retry", expected_pattern in src,
          f"pattern not found in {path}")

# ── 12. Dockerfile ────────────────────────────────────────────────────────────
print("\n-- Test 12: Dockerfile --")
dockerfile = pathlib.Path("C:/Claude/lumenx-agent/Dockerfile")
check("Dockerfile exists",           dockerfile.exists())
if dockerfile.exists():
    content = dockerfile.read_text(encoding="utf-8")
    check("FROM python:3.12",        "python:3.12" in content)
    check("EXPOSE 8001",             "EXPOSE 8001" in content)
    check("HEALTHCHECK present",     "HEALTHCHECK" in content)
    check("uvicorn start command",   "uvicorn agent.main:app" in content)
    check("non-root USER",           "USER agent" in content)
    check("wiki build on startup",   "build_wiki.py" in content)

# ── 13. railway.toml ──────────────────────────────────────────────────────────
print("\n-- Test 13: railway.toml --")
railway = pathlib.Path("C:/Claude/lumenx-agent/railway.toml")
check("railway.toml exists",         railway.exists())
if railway.exists():
    content = railway.read_text(encoding="utf-8")
    check("healthcheckPath=/health", "healthcheckPath" in content)
    check("DOCKERFILE builder",      "DOCKERFILE" in content)
    check("PORT variable",           "PORT" in content)
    check("DATABASE_URL variable",   "DATABASE_URL" in content)

# ── 14. docs/env-vars.md ──────────────────────────────────────────────────────
print("\n-- Test 14: docs/env-vars.md --")
envdoc = pathlib.Path("C:/Claude/lumenx-agent/docs/env-vars.md")
check("docs/env-vars.md exists",      envdoc.exists())
if envdoc.exists():
    content = envdoc.read_text(encoding="utf-8")
    for var in ["ANTHROPIC_API_KEY", "LUMENX_ADMIN_TOKEN", "LUMENX_BASE_URL",
                "CONFIDENCE_THRESHOLD", "MIN_REAL_LABELS_FOR_ROUTING",
                "CONTEXT_BUDGET_TOKENS", "REPLY_MAX_TOKENS", "DATABASE_URL",
                "LOG_LEVEL", "POLL_INTERVAL_SECONDS"]:
        check(f"documents {var}", var in content)

# ── 15. GET /health fields ────────────────────────────────────────────────────
print("\n-- Test 15: GET /health fields --")
r = requests.get(f"{BASE}/health", timeout=5)
check("HTTP 200",                     r.status_code == 200)
h = r.json()
for field in ["status","db_connected","wiki_loaded","model_loaded",
              "poller_active","real_labels","uptime_seconds","errors"]:
    check(f"health has '{field}'",    field in h)
check("status='ok'",                  h.get("status") == "ok")
check("errors=[]",                    h.get("errors") == [])

# ── 16. Rate-limit logic in InboxPoller ──────────────────────────────────────
print("\n-- Test 16: InboxPoller rate-limit logic --")
try:
    from agent.poller import InboxPoller
    poller = InboxPoller.__new__(InboxPoller)
    poller._last_reply_ts = {}
    poller._processed_ids = set()

    # No entry → not rate-limited
    check("not rate-limited on first call",
          not poller._is_rate_limited("thread-A"))

    # Set timestamp to now → rate-limited
    poller._last_reply_ts["thread-A"] = time.time()
    check("rate-limited within window",
          poller._is_rate_limited("thread-A"),
          "window=5s")

    # Set timestamp to 10 seconds ago → not rate-limited
    poller._last_reply_ts["thread-A"] = time.time() - 10
    check("not rate-limited after 10s",
          not poller._is_rate_limited("thread-A"))
except Exception as exc:
    check("rate-limit logic works", False, str(exc))

# ── 17. Fallback enqueue writes to DB ─────────────────────────────────────────
print("\n-- Test 17: _fallback_enqueue writes ReviewQueue row --")
try:
    from agent.poller import InboxPoller
    from db.session import get_db
    from db.models import ReviewQueue

    poller = InboxPoller.__new__(InboxPoller)

    result = poller._fallback_enqueue(
        thread_id  = "test-ph10-fallback",
        msg_text   = "Test fallback message",
        error_type = "test_error",
        error_msg  = "This is a unit test fallback",
    )
    check("_fallback_enqueue returns dict",  isinstance(result, dict))
    check("action='fallback_queued'",        result.get("action") == "fallback_queued")
    check("error_type in result",            result.get("error_type") == "test_error")

    # Verify DB row
    with get_db() as db:
        row = db.query(ReviewQueue).filter(
            ReviewQueue.thread_id == "test-ph10-fallback"
        ).order_by(ReviewQueue.id.desc()).first()
        check("DB row created",              row is not None)
        if row:
            check("status='pending'",        row.status == "pending")
            check("confidence=0.0",          row.confidence == 0.0)
            check("draft contains error",    "test_error" in (row.draft_text or ""))
except Exception as exc:
    check("_fallback_enqueue works", False, str(exc))


# ── Summary ────────────────────────────────────────────────────────────────────
total  = len(results)
passed = sum(1 for _, ok in results if ok)
failed = total - passed

print(f"\n{'='*58}")
print(f"  Phase 10 results: {passed}/{total} passed  "
      f"({'ALL PASS' if not failed else f'{failed} FAILED'})")
print(f"{'='*58}")
if failed:
    print("Failed checks:")
    for name, ok in results:
        if not ok: print(f"  FAIL: {name}")
sys.exit(0 if not failed else 1)
