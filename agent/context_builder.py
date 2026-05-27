"""
Context Builder — Phase 3
Assembles a rich, token-budgeted context window (≤ 4,000 tokens) for every
non-greeting reply, in this order:

  ┌─────────────────────────────┬──────────────────────┐
  │ Section                     │ Target tokens        │
  ├─────────────────────────────┼──────────────────────┤
  │ [SYSTEM PROMPT]             │ ~400  (prompt-cached)│
  │ [PRODUCT WIKI CHUNKS]       │ ~600  (top-k FAISS)  │
  │ [CONVERSATION SUMMARY]      │ ~400  (Haiku, 24 h)  │
  │ [FEEDBACK LOG ENTRIES]      │ ~600  (Phase 6 stub) │
  │ [CONVERSATION HISTORY]      │ ~800  (last 10 msgs) │
  │ [CURRENT CUSTOMER MESSAGE]  │ ~200                 │
  ├─────────────────────────────┼──────────────────────┤
  │ TOTAL BUDGET                │ ≤ 4,000 tokens       │
  └─────────────────────────────┴──────────────────────┘

Public API:
  assemble(thread_id, user_message, intent, client?) -> AssembledContext
  build_conversation_summary(client?) -> str
  get_current_thread(thread_id) -> list[dict]
"""

import os
import sys
import time
import json
import requests
from datetime import datetime, timezone
from typing import Optional

import anthropic
from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ssl_utils import patch_ssl
patch_ssl()

load_dotenv()

# ── Constants ────────────────────────────────────────────────────────────────

SUMMARY_MODEL = "claude-haiku-4-5-20251001"

HAIKU_INPUT_PRICE  = 0.80 / 1_000_000
HAIKU_OUTPUT_PRICE = 4.00 / 1_000_000

CONTEXT_BUDGET     = int(os.getenv("CONTEXT_BUDGET_TOKENS", "4000"))
SUMMARY_CACHE_TTL  = 86_400   # 24 hours in seconds
THREAD_MAX_MSGS    = 10       # keep last N messages from current thread

LUMENX_BASE_URL    = os.getenv("LUMENX_BASE_URL", "https://lumenx-demo.up.railway.app")
LUMENX_ADMIN_TOKEN = os.getenv("LUMENX_ADMIN_TOKEN", "")

# Per-section soft targets (used for trimming priority)
SECTION_BUDGETS = {
    "system":   400,
    "wiki":     600,
    "summary":  400,
    "feedback": 600,
    "thread":   800,
    "message":  200,
}

# ── Agent System Prompt ───────────────────────────────────────────────────────
# Single source of truth — imported by draft_agent.py in Phase 4.

AGENT_SYSTEM_PROMPT = """\
You are a professional, empathetic customer support agent for LumenX, a B2B SaaS company.

RULES — follow these without exception:
1. NEVER invent or guess pricing, trial periods, refund windows, or discount details.
   If the exact figure is not in the provided product context, say:
   "I don't have that specific detail on hand — our team will follow up shortly."
2. Be warm, concise, and professional. No filler phrases like "Great question!".
3. Answer only what the customer asked. Do not upsell unless directly relevant.
4. If the customer is frustrated, acknowledge it before solving the problem.
5. Always sign off with: "— LumenX Support"
"""

# ── In-memory summary cache (24 h TTL) ───────────────────────────────────────

_summary_cache: dict = {"text": None, "expires_at": 0.0, "cost_usd": 0.0}

# ── Utility helpers ───────────────────────────────────────────────────────────

def _lumenx_headers() -> dict:
    return {"X-Admin-Token": LUMENX_ADMIN_TOKEN}


def _approx_tokens(text: str) -> int:
    """Fast approximation: ~4 characters per token (good enough for budget checks)."""
    return max(1, len(text) // 4)


def _count_tokens_exact(
    client: anthropic.Anthropic,
    system: str,
    messages: list[dict],
) -> int:
    """
    Call Anthropic's token-counting endpoint once for the full assembled context.
    Falls back to approximation on any error.
    """
    try:
        result = client.messages.count_tokens(
            model=SUMMARY_MODEL,
            system=system,
            messages=messages,
        )
        return result.input_tokens
    except Exception:
        total_text = system + " ".join(m.get("content", "") for m in messages)
        return _approx_tokens(total_text)


def _trim_text(text: str, max_approx_tokens: int) -> str:
    """
    Hard-trim `text` to stay within `max_approx_tokens` (approximate).
    Appends a truncation notice so the LLM knows.
    """
    if _approx_tokens(text) <= max_approx_tokens:
        return text
    max_chars = max_approx_tokens * 4
    return text[:max_chars].rsplit(" ", 1)[0] + "\n[... truncated to fit token budget]"


def _log_summary_cost(
    thread_id: str | None,
    input_tokens: int,
    output_tokens: int,
    cost_usd: float,
) -> None:
    """Silently write summary cost to CostLog — never raises."""
    try:
        from db.session import get_db
        from db.models import CostLog
        with get_db() as db:
            db.add(CostLog(
                thread_id=thread_id,
                task_type="summary",
                model=SUMMARY_MODEL,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cache_read_input_tokens=0,
                cache_creation_input_tokens=0,
                cost_usd=cost_usd,
                created_at=datetime.now(timezone.utc),
            ))
    except Exception:
        pass


# ── Section: Conversation Summary ────────────────────────────────────────────

def build_conversation_summary(
    client: anthropic.Anthropic | None = None,
    thread_id: str | None = None,
    force_refresh: bool = False,
) -> str:
    """
    Fetch all LumenX threads, summarise with Haiku, cache result for 24 h.

    Returns a short paragraph (≤ 200 tokens) summarising recent customer
    support patterns — common topics, recurring issues, overall sentiment.
    The summary is injected into every non-greeting context window.
    """
    now = time.time()
    if not force_refresh and _summary_cache["text"] and now < _summary_cache["expires_at"]:
        return _summary_cache["text"]

    if client is None:
        client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    # 1. Fetch threads list
    try:
        resp = requests.get(
            f"{LUMENX_BASE_URL}/api/admin/threads",
            headers=_lumenx_headers(),
            verify=False,
            timeout=10,
        )
        resp.raise_for_status()
        payload  = resp.json()                                    # {"count": N, "threads": [...]}
        threads  = payload.get("threads", []) if isinstance(payload, dict) else payload
    except Exception as exc:
        return f"[Conversation summary unavailable: {exc}]"

    if not threads:
        return "[No previous threads found.]"

    # 2. Build a compact thread list for Haiku to summarise (cap at 40 threads)
    lines = []
    for t in threads[:40]:
        last_msg = t.get("last_message") or {}
        preview  = (last_msg.get("text") or "").strip()[:100]
        customer = t.get("customer_display_name") or t.get("customer_username") or "?"
        intent   = t.get("intent") or "unknown intent"
        status   = "open" if t.get("unread_admin") else "resolved"
        lines.append(f"- [{status}|{intent}] {customer}: {preview}")

    thread_list = "\n".join(lines)
    prompt = (
        "Summarise the following recent customer support threads in 3–4 sentences. "
        "Focus on: common question topics, recurring pain points, and overall sentiment. "
        "Be concise — this summary will be injected into future reply prompts.\n\n"
        f"Threads ({len(threads)} total, showing {len(lines)}):\n{thread_list}"
    )

    # 3. Call Haiku (with retry)
    try:
        from agent.retry import call_with_retry
        response = call_with_retry(
            client.messages.create,
            model=SUMMARY_MODEL,
            max_tokens=200,
            messages=[{"role": "user", "content": prompt}],
        )
        summary       = response.content[0].text.strip()
        input_tokens  = response.usage.input_tokens
        output_tokens = response.usage.output_tokens
        cost_usd      = input_tokens * HAIKU_INPUT_PRICE + output_tokens * HAIKU_OUTPUT_PRICE
    except Exception as exc:
        return f"[Summary generation failed: {exc}]"

    # 4. Cache + log cost
    _summary_cache["text"]       = summary
    _summary_cache["expires_at"] = now + SUMMARY_CACHE_TTL
    _summary_cache["cost_usd"]   = cost_usd
    _log_summary_cost(thread_id, input_tokens, output_tokens, cost_usd)

    return summary


# ── Section: Feedback Log Entries ─────────────────────────────────────────────

def get_feedback_log_entries(query: str, k: int = 5) -> list[dict]:
    """
    Return top-k similar past approved (customer_msg, final_reply) pairs.

    Phase 6: calls db.feedback_log.search_feedback() which queries the FAISS
    feedback index (wiki/feedback_index.faiss).  Returns [] gracefully when
    the index doesn't exist yet (first run, or fewer than 1 entry).

    Each returned dict: {thread_id, customer_msg, final_text, intent,
                         edit_dist_norm, score}
    """
    try:
        from db.feedback_log import search_feedback
        return search_feedback(query, k=k)
    except Exception:
        return []


# ── Section: Current Thread ───────────────────────────────────────────────────

def get_current_thread(
    thread_id: str,
    max_messages: int = THREAD_MAX_MSGS,
) -> list[dict]:
    """
    Fetch a LumenX thread and return its last `max_messages` messages.

    Each returned dict: {role, text, ts}
    Role is "customer" or "admin".
    """
    try:
        resp = requests.get(
            f"{LUMENX_BASE_URL}/api/admin/threads/{thread_id}",
            headers=_lumenx_headers(),
            verify=False,
            timeout=10,
        )
        resp.raise_for_status()
        data     = resp.json()                           # {"thread": {..., "messages": [...]}}
        thread   = data.get("thread", data)
        messages = thread.get("messages", [])
    except Exception:
        return []

    # Keep only the last N, normalise field names
    tail = messages[-max_messages:]
    return [
        {
            "role": m.get("role", "unknown"),
            "text": (m.get("text") or "").strip(),
            "ts":   m.get("ts", "")[:10],         # date only
        }
        for m in tail
    ]


# ── Formatters ────────────────────────────────────────────────────────────────

def _format_thread(messages: list[dict]) -> str:
    if not messages:
        return "(no previous messages in this thread)"
    lines = []
    for m in messages:
        role = m["role"].upper()
        ts   = m["ts"]
        text = m["text"]
        lines.append(f"[{ts}] {role}: {text}")
    return "\n".join(lines)


def _format_wiki_chunks(chunks: list[dict]) -> str:
    if not chunks:
        return "(no relevant product documentation found)"
    # chunk["text"] already includes its own "[Product — section]" header
    # (written by build_wiki.py), so we just join the texts directly.
    parts = [c.get("text", "").strip() for c in chunks if c.get("text")]
    return "\n\n".join(parts)


def _format_feedback_entries(entries: list[dict]) -> str:
    if not entries:
        return ""
    parts = []
    for e in entries:
        parts.append(
            f"Customer asked: {e.get('customer_msg', '').strip()}\n"
            f"Approved reply: {e.get('final_text', '').strip()}"
        )
    return "\n\n---\n\n".join(parts)


def _wrap_section(tag: str, content: str) -> str:
    return f"=== {tag} ===\n{content.strip()}\n"


# ── Main assembler ────────────────────────────────────────────────────────────

def assemble(
    thread_id: str,
    user_message: str,
    intent: str,
    client: anthropic.Anthropic | None = None,
) -> dict:
    """
    Assemble the full context window for the draft agent.

    Args:
        thread_id:    LumenX thread ID for the active conversation.
        user_message: The raw customer message to reply to.
        intent:       Classified intent (from Phase 2 intent router).
        client:       Optional pre-built Anthropic client (reused for efficiency).

    Returns:
        {
            "system_prompt":     str,   # agent persona → pass as system= in API call
            "context_str":       str,   # assembled context → pass as user message
            "estimated_tokens":  int,   # total tokens in context_str (approximate)
            "exact_tokens":      int,   # accurate token count from Anthropic API
            "sections":          dict,  # per-section token counts for debugging
            "wiki_chunks":       list,  # raw retrieved chunks (for dashboard viewer)
            "summary_cost_usd":  float, # cost of summary generation (0 if cached)
        }
    """
    if client is None:
        client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    # ── 1. Wiki chunks ────────────────────────────────────────────────────────
    from wiki.retriever import retrieve, K_BY_INTENT
    k           = K_BY_INTENT.get(intent, 3)
    wiki_chunks = retrieve(user_message, intent=intent, k=k) if k > 0 else []
    wiki_text   = _format_wiki_chunks(wiki_chunks)

    # ── 2. Conversation summary (cached 24 h) ─────────────────────────────────
    summary_text = build_conversation_summary(client, thread_id=thread_id)

    # ── 3. Feedback log entries (Phase 6 stub → []) ───────────────────────────
    feedback_entries = get_feedback_log_entries(user_message, k=5)
    feedback_text    = _format_feedback_entries(feedback_entries)

    # ── 4. Current thread (last 10 messages) ─────────────────────────────────
    thread_messages = get_current_thread(thread_id)
    thread_text     = _format_thread(thread_messages)

    # ── 5. Build sections ─────────────────────────────────────────────────────
    wiki_section     = _wrap_section("PRODUCT CONTEXT",            wiki_text)
    summary_section  = _wrap_section("RECENT CUSTOMER PATTERNS",   summary_text)
    feedback_section = _wrap_section("SIMILAR PAST REPLIES",       feedback_text) if feedback_text else ""
    thread_section   = _wrap_section("CONVERSATION HISTORY",       thread_text)
    msg_section      = _wrap_section("CURRENT CUSTOMER MESSAGE",   user_message)

    # ── 6. Approximate token counts per section ───────────────────────────────
    sections = {
        "system":   _approx_tokens(AGENT_SYSTEM_PROMPT),
        "wiki":     _approx_tokens(wiki_section),
        "summary":  _approx_tokens(summary_section),
        "feedback": _approx_tokens(feedback_section) if feedback_section else 0,
        "thread":   _approx_tokens(thread_section),
        "message":  _approx_tokens(msg_section),
    }
    total_approx = sum(sections.values())

    # ── 7. Trim if over budget (priority: thread → feedback → wiki) ───────────
    if total_approx > CONTEXT_BUDGET:
        overage = total_approx - CONTEXT_BUDGET

        # Try trimming thread first (least critical to trim)
        if sections["thread"] > SECTION_BUDGETS["thread"]:
            available_trim = sections["thread"] - SECTION_BUDGETS["thread"]
            trim_amount    = min(overage, available_trim)
            new_thread_budget = sections["thread"] - trim_amount
            thread_text   = _trim_text(thread_text, new_thread_budget)
            thread_section = _wrap_section("CONVERSATION HISTORY", thread_text)
            sections["thread"] = _approx_tokens(thread_section)
            overage = max(0, sum(sections.values()) - CONTEXT_BUDGET)

        # Then trim wiki if still over
        if overage > 0 and sections["wiki"] > SECTION_BUDGETS["wiki"]:
            available_trim = sections["wiki"] - SECTION_BUDGETS["wiki"]
            trim_amount    = min(overage, available_trim)
            new_wiki_budget = sections["wiki"] - trim_amount
            wiki_text    = _trim_text(wiki_text, new_wiki_budget)
            wiki_section = _wrap_section("PRODUCT CONTEXT", wiki_text)
            sections["wiki"] = _approx_tokens(wiki_section)

    # ── 8. Assemble final context string ─────────────────────────────────────
    # Split into cacheable (stable product context) vs dynamic (per-request)
    # so draft_agent can apply cache_control correctly.
    cacheable_str = wiki_section + "\n" + summary_section
    dynamic_parts = []
    if feedback_section:
        dynamic_parts.append(feedback_section)
    dynamic_parts += [thread_section, msg_section]
    dynamic_str   = "\n".join(dynamic_parts)
    context_str   = cacheable_str + "\n" + dynamic_str
    total_approx  = sum(sections.values())

    # ── 9. One exact token count from Anthropic API ───────────────────────────
    exact_tokens = _count_tokens_exact(
        client,
        system=AGENT_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": context_str}],
    )

    return {
        "system_prompt":    AGENT_SYSTEM_PROMPT,
        "context_str":      context_str,        # full combined (for logging)
        "cacheable_str":    cacheable_str,       # wiki + summary  → cache_control
        "dynamic_str":      dynamic_str,         # feedback + thread + message
        "estimated_tokens": total_approx,
        "exact_tokens":     exact_tokens,
        "sections":         sections,
        "wiki_chunks":      wiki_chunks,
        "summary_cost_usd": _summary_cache.get("cost_usd", 0.0),
    }


# ── Self-test ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    # Fetch first real thread from LumenX API
    try:
        r = requests.get(
            f"{LUMENX_BASE_URL}/api/admin/threads",
            headers=_lumenx_headers(),
            verify=False,
            timeout=10,
        )
        r.raise_for_status()
        payload    = r.json()
        threads    = payload.get("threads", []) if isinstance(payload, dict) else payload
        thread_id  = threads[0]["id"] if threads else "test-thread-001"
        # Use the last customer message as the user message
        last_msg   = threads[0].get("last_message", {})
        user_msg   = last_msg.get("text", "What are your pricing plans?")
        intent     = "pricing"
    except Exception as e:
        print(f"Warning: could not fetch live thread ({e}), using fallback test data")
        thread_id = "test-thread-001"
        user_msg  = "What are your pricing plans for the Pro tier?"
        intent    = "pricing"

    print(f"\nThread  : {thread_id}")
    print(f"Message : {user_msg[:80]}")
    print(f"Intent  : {intent}")
    print("\nAssembling context...\n")

    ctx = assemble(thread_id, user_msg, intent, client=client)

    # ── Print section breakdown ───────────────────────────────────────────────
    print("┌─────────────────────────────────┬──────────────┐")
    print("│ Section                         │ Tokens (est) │")
    print("├─────────────────────────────────┼──────────────┤")
    for name, toks in ctx["sections"].items():
        print(f"│ {name:<31} │ {toks:>12} │")
    print("├─────────────────────────────────┼──────────────┤")
    print(f"│ {'TOTAL (approximate)':<31} │ {ctx['estimated_tokens']:>12} │")
    print(f"│ {'TOTAL (Anthropic exact)':<31} │ {ctx['exact_tokens']:>12} │")
    print(f"│ {'BUDGET':<31} │ {CONTEXT_BUDGET:>12} │")
    print("└─────────────────────────────────┴──────────────┘")

    budget_pct = ctx["exact_tokens"] / CONTEXT_BUDGET * 100
    status = "✅ WITHIN BUDGET" if ctx["exact_tokens"] <= CONTEXT_BUDGET else "❌ OVER BUDGET"
    print(f"\nBudget used: {budget_pct:.1f}%  {status}")

    if ctx["summary_cost_usd"] > 0:
        print(f"Summary cost: ${ctx['summary_cost_usd']:.6f} (Haiku)")
    else:
        print("Summary cost: $0.000000 (cache hit)")

    print(f"\nWiki chunks retrieved: {len(ctx['wiki_chunks'])}")
    for c in ctx["wiki_chunks"]:
        print(f"  • {c.get('product_name','?')} — {c.get('section','?')}  (score: {c.get('score',0):.4f})")

    print("\n" + "─" * 60)
    print("ASSEMBLED CONTEXT PREVIEW (first 800 chars):")
    print("─" * 60)
    print(ctx["context_str"][:800])
    print("...")

    if ctx["exact_tokens"] <= CONTEXT_BUDGET:
        print("\nPhase 3 PASSED — context builder ready.")
    else:
        print(f"\nWARNING: Context {ctx['exact_tokens']} tokens > budget {CONTEXT_BUDGET}. Review trimming logic.")
