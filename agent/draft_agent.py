"""
Draft Agent â€” Phase 4
Generates a grounded customer support reply using Claude Sonnet.

Prompt-caching strategy:
  system block          â†’ cache_control: ephemeral  (agent persona, ~400 tok)
  user block 1 (stable) â†’ cache_control: ephemeral  (wiki + summary, ~1 000 tok)
  user block 2 (dynamic)â†’ no cache                  (thread + message, ~1 000 tok)

Every call logs full token breakdown + USD cost to CostLog.
"""

import os
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

import anthropic
from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ssl_utils import patch_ssl
patch_ssl()

load_dotenv()

# â”€â”€ Constants â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

DRAFT_MODEL = "claude-sonnet-4-6"

SONNET_INPUT_PRICE       = 3.00  / 1_000_000
SONNET_OUTPUT_PRICE      = 15.00 / 1_000_000
SONNET_CACHE_WRITE_PRICE = 3.75  / 1_000_000
SONNET_CACHE_READ_PRICE  = 0.30  / 1_000_000

REPLY_MAX_TOKENS = int(os.getenv("REPLY_MAX_TOKENS", "500"))


# â”€â”€ Result dataclass â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

@dataclass
class DraftResult:
    text:                       str
    model:                      str
    thread_id:                  Optional[str]
    intent:                     Optional[str]
    input_tokens:               int
    output_tokens:              int
    cache_read_input_tokens:    int
    cache_creation_input_tokens: int
    cost_usd:                   float
    cache_hit:                  bool          # True if any cache_read tokens > 0
    context_json:               str = ""      # serialised context for dashboard viewer
    exact_context_tokens:       int = 0


# â”€â”€ Cost log helper â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def _log_cost(result: DraftResult) -> None:
    """Silently insert a CostLog row â€” never raises."""
    try:
        from db.session import get_db
        from db.models import CostLog
        with get_db() as db:
            db.add(CostLog(
                thread_id=result.thread_id,
                task_type="draft",
                model=result.model,
                input_tokens=result.input_tokens,
                output_tokens=result.output_tokens,
                cache_read_input_tokens=result.cache_read_input_tokens,
                cache_creation_input_tokens=result.cache_creation_input_tokens,
                cost_usd=result.cost_usd,
                created_at=datetime.now(timezone.utc),
            ))
    except Exception:
        pass


# â”€â”€ Main draft function â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def generate_draft(
    thread_id: str,
    user_message: str,
    intent: str,
    client: anthropic.Anthropic | None = None,
    context: dict | None = None,
) -> DraftResult:
    """
    Generate a support reply draft for `user_message`.

    Args:
        thread_id:    LumenX thread ID.
        user_message: The raw customer message.
        intent:       Classified intent (from Phase 2 intent router).
        client:       Optional pre-built Anthropic client.
        context:      Optional pre-assembled context dict (from context_builder.assemble).
                      If None, assemble() is called internally.

    Returns:
        DraftResult with reply text, token counts, and USD cost.
    """
    if client is None:
        client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    # â”€â”€ 1. Assemble context (or use provided) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    if context is None:
        from agent.context_builder import assemble
        context = assemble(thread_id, user_message, intent, client=client)

    system_prompt  = context["system_prompt"]
    cacheable_str  = context["cacheable_str"]   # wiki + summary (cache-eligible)
    dynamic_str    = context["dynamic_str"]     # thread + message (per-request)

    import json
    context_json = json.dumps({
        "system":    system_prompt,
        "cacheable": cacheable_str,
        "dynamic":   dynamic_str,
        "sections":  context.get("sections", {}),
    }, ensure_ascii=False)

    # â”€â”€ 2. Build prompt-cached API request â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    # System block: agent persona (cache_control â†’ ephemeral)
    system_blocks = [
        {
            "type": "text",
            "text": system_prompt,
            "cache_control": {"type": "ephemeral"},
        }
    ]

    # User content: stable part cached, dynamic part not cached
    user_content = [
        {
            "type": "text",
            "text": cacheable_str,
            "cache_control": {"type": "ephemeral"},   # wiki + summary â†’ cache
        },
        {
            "type": "text",
            "text": dynamic_str,                       # thread + message â†’ no cache
        },
    ]

    # â”€â”€ 3. Call Sonnet â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    from agent.retry import call_with_retry
    response = call_with_retry(
        client.messages.create,
        model=DRAFT_MODEL,
        max_tokens=REPLY_MAX_TOKENS,
        system=system_blocks,
        messages=[{"role": "user", "content": user_content}],
    )

    reply_text = response.content[0].text.strip()
    usage      = response.usage

    input_tokens                = usage.input_tokens
    output_tokens               = usage.output_tokens
    cache_read_input_tokens     = getattr(usage, "cache_read_input_tokens", 0) or 0
    cache_creation_input_tokens = getattr(usage, "cache_creation_input_tokens", 0) or 0

    cost_usd = (
        input_tokens                * SONNET_INPUT_PRICE       +
        output_tokens               * SONNET_OUTPUT_PRICE      +
        cache_creation_input_tokens * SONNET_CACHE_WRITE_PRICE +
        cache_read_input_tokens     * SONNET_CACHE_READ_PRICE
    )

    result = DraftResult(
        text=reply_text,
        model=DRAFT_MODEL,
        thread_id=thread_id,
        intent=intent,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_read_input_tokens=cache_read_input_tokens,
        cache_creation_input_tokens=cache_creation_input_tokens,
        cost_usd=cost_usd,
        cache_hit=(cache_read_input_tokens > 0),
        context_json=context_json,
        exact_context_tokens=context.get("exact_tokens", 0),
    )

    # â”€â”€ 4. Log cost â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    _log_cost(result)

    return result


# â”€â”€ Self-test â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

if __name__ == "__main__":
    import requests, os as _os

    BASE_URL    = _os.getenv("LUMENX_BASE_URL", "https://lumenx-demo.up.railway.app").strip()
    ADMIN_TOKEN = _os.getenv("LUMENX_ADMIN_TOKEN", "")

    from ssl_utils import patch_ssl
    patch_ssl()

    client = anthropic.Anthropic(api_key=_os.getenv("ANTHROPIC_API_KEY"))

    TEST_CASES = [
        ("pricing",   "What is the monthly price for the Pro plan of TaskGrid?"),
        ("technical", "I'm getting a 403 error when calling the TaskGrid API."),
        ("refund",    "I was charged twice this month. I need a refund."),
        ("other",     "Can multiple team members share one NoteHub account?"),
    ]

    # Try to get a real thread id
    try:
        r = requests.get(f"{BASE_URL}/api/admin/threads",
                         headers={"X-Admin-Token": ADMIN_TOKEN}, verify=False, timeout=8)
        threads   = r.json().get("threads", [])
        thread_id = threads[0]["id"] if threads else "test-001"
    except Exception:
        thread_id = "test-001"

    print(f"\nUsing thread: {thread_id}\n")
    print(f"{'Intent':<12} {'Cache':<6} {'In':>6} {'Out':>6} {'CacheR':>7} {'CacheW':>7} {'Cost($)':>10}")
    print("â”€" * 70)

    total_cost = 0.0
    for intent, message in TEST_CASES:
        result = generate_draft(thread_id, message, intent, client=client)
        flag   = "HIT" if result.cache_hit else "MISS"
        print(
            f"{intent:<12} {flag:<6} {result.input_tokens:>6} {result.output_tokens:>6} "
            f"{result.cache_read_input_tokens:>7} {result.cache_creation_input_tokens:>7} "
            f"${result.cost_usd:>9.6f}"
        )
        total_cost += result.cost_usd

    print("â”€" * 70)
    print(f"{'TOTAL':<12} {'':6} {'':>6} {'':>6} {'':>7} {'':>7} ${total_cost:>9.6f}")
    print(f"\nLast reply preview:\n{result.text[:300]}...")
    print("\nPhase 4 PASSED â€” draft agent ready.")
