"""
Intent Router — Phase 2
Classifies incoming customer messages into one of five intents using
Claude Haiku (cheapest model). Greeting queries skip the context builder
and are handled with a lightweight direct reply.

Every classification call is logged to the CostLog table.
"""

import json
import os
import sys
from datetime import datetime, timezone
import anthropic
from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ssl_utils import patch_ssl
patch_ssl()

load_dotenv()

# ── Constants ────────────────────────────────────────────────────────────────

INTENT_MODEL = "claude-haiku-4-5-20251001"

VALID_INTENTS = {"greeting", "pricing", "technical", "refund", "other"}

# Haiku pricing (per token)
HAIKU_INPUT_PRICE  = 0.80 / 1_000_000
HAIKU_OUTPUT_PRICE = 4.00 / 1_000_000

SYSTEM_PROMPT = """\
You are an intent classifier for a B2B SaaS customer support system.

Classify the customer message into EXACTLY ONE of these intents:
- greeting    : casual openers, thanks, closings (hi, hello, how are you, bye, good morning)
- pricing     : questions about cost, plans, tiers, upgrades, billing, discounts, trials
- technical   : bugs, errors, broken features, API errors, integration setup, data not syncing
- refund      : refunds, cancellations, money back, dispute charges, downgrade requests
- other       : general company questions, sales inquiries, compliments, feature requests,
                capability questions ("can it do X?"), account/team/access questions,
                anything that is not clearly pricing, technical, refund, or greeting

When in doubt between technical and other: if the user is NOT reporting a problem or error,
classify as other.

Reply with valid JSON ONLY — no markdown, no explanation:
{"intent": "<one of the five intents above>"}
"""

# ── Greeting responses ───────────────────────────────────────────────────────

GREETING_REPLIES = {
    "hi":           "Hi there! Welcome to LumenX Support. How can I help you today?",
    "hello":        "Hello! Welcome to LumenX Support. What can I assist you with?",
    "hey":          "Hey! Great to hear from you. How can LumenX Support help today?",
    "good morning": "Good morning! Hope you're having a great day. How can we help?",
    "good afternoon":"Good afternoon! How can LumenX Support assist you today?",
    "good evening": "Good evening! How can we help you today?",
    "thanks":       "You're very welcome! Is there anything else I can help you with?",
    "thank you":    "You're very welcome! Feel free to reach out anytime.",
    "bye":          "Thanks for reaching out! Have a great day. — LumenX Support",
    "goodbye":      "Goodbye! Don't hesitate to contact us if you need anything. — LumenX Support",
    "default":      "Hello! Thanks for getting in touch with LumenX Support. How can I assist you today?",
}


def _greeting_reply(message: str) -> str:
    msg = message.lower()
    for keyword, reply in GREETING_REPLIES.items():
        if keyword in msg:
            return reply
    return GREETING_REPLIES["default"]


# ── CostLog DB insert ─────────────────────────────────────────────────────────

def _log_cost(thread_id: str | None, input_tokens: int, output_tokens: int, cost_usd: float):
    """Silently insert a CostLog row — never raise, never block the caller."""
    try:
        from db.session import get_db
        from db.models import CostLog
        with get_db() as db:
            db.add(CostLog(
                thread_id=thread_id,
                task_type="intent",
                model=INTENT_MODEL,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cache_read_input_tokens=0,
                cache_creation_input_tokens=0,
                cost_usd=cost_usd,
                created_at=datetime.now(timezone.utc),
            ))
    except Exception:
        pass  # Never let DB errors break classification


# ── Main classifier ───────────────────────────────────────────────────────────

def classify(
    message: str,
    thread_id: str | None = None,
    client: anthropic.Anthropic | None = None,
) -> dict:
    """
    Classify a customer message into one of five intents.

    Args:
        message:   The raw customer message text.
        thread_id: Optional LumenX thread ID for cost log association.
        client:    Optional pre-built Anthropic client (reused for efficiency).

    Returns:
        {
            "intent":         str,         # one of VALID_INTENTS
            "input_tokens":   int,
            "output_tokens":  int,
            "cost_usd":       float,
            "greeting_reply": str | None,  # ready-made reply if greeting, else None
        }
    """
    if client is None:
        client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    response = client.messages.create(
        model=INTENT_MODEL,
        max_tokens=20,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": message}],
    )

    raw = response.content[0].text.strip()

    # Parse JSON defensively
    intent = "other"
    try:
        parsed = json.loads(raw)
        candidate = parsed.get("intent", "other").lower().strip()
        intent = candidate if candidate in VALID_INTENTS else "other"
    except (json.JSONDecodeError, AttributeError):
        # Fallback: scan raw text for intent keywords
        for candidate in VALID_INTENTS:
            if candidate in raw.lower():
                intent = candidate
                break

    input_tokens  = response.usage.input_tokens
    output_tokens = response.usage.output_tokens
    cost_usd = (
        input_tokens  * HAIKU_INPUT_PRICE +
        output_tokens * HAIKU_OUTPUT_PRICE
    )

    # Log to DB (fire-and-forget, never raises)
    _log_cost(thread_id, input_tokens, output_tokens, cost_usd)

    return {
        "intent":         intent,
        "input_tokens":   input_tokens,
        "output_tokens":  output_tokens,
        "cost_usd":       cost_usd,
        "greeting_reply": _greeting_reply(message) if intent == "greeting" else None,
    }


# ── Self-test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # 10 messages covering all 5 intents (2 per intent)
    TEST_CASES = [
        # (message, expected_intent)
        ("Hi there, how are you doing today?",                    "greeting"),
        ("Thanks so much for the quick response!",                "greeting"),
        ("What is the monthly price for the Pro plan?",           "pricing"),
        ("Do you offer any annual discount or free trial?",       "pricing"),
        ("I'm getting a 403 Forbidden error calling your API.",   "technical"),
        ("How do I integrate TaskGrid with our Slack workspace?", "technical"),
        ("I'd like a refund — I was charged twice this month.",   "refund"),
        ("How do I cancel my subscription to InvoiceFlow?",       "refund"),
        ("What timezone does CalendarSync use by default?",       "other"),
        ("Can multiple team members share one NoteHub account?",  "other"),
    ]

    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    correct = 0
    total_cost = 0.0

    print(f"\n{'#':<3} {'Message':<50} {'Expected':<12} {'Got':<12} {'Match':<6} {'Cost ($)'}")
    print("-" * 100)

    for i, (msg, expected) in enumerate(TEST_CASES, 1):
        result = classify(msg, thread_id="test", client=client)
        got     = result["intent"]
        match   = "OK" if got == expected else "FAIL"
        if got == expected:
            correct += 1
        total_cost += result["cost_usd"]
        print(f"{i:<3} {msg[:49]:<50} {expected:<12} {got:<12} {match:<6} ${result['cost_usd']:.6f}")
        if result["greeting_reply"]:
            print(f"    -> Auto-reply: {result['greeting_reply']}")

    print("-" * 100)
    accuracy = correct / len(TEST_CASES) * 100
    print(f"Accuracy: {correct}/{len(TEST_CASES)} ({accuracy:.0f}%)    Total cost: ${total_cost:.6f}")
    print()
    if accuracy < 90:
        print("WARNING: Accuracy below 90% — review system prompt.")
    else:
        print("Phase 2 PASSED — intent router ready.")
