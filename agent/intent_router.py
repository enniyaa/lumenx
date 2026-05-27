"""
Intent Router — Phase 2
Classifies incoming customer messages into one of five intents using
Claude Haiku (cheapest model). Greeting queries skip the context builder
and are handled with a lightweight direct reply.
"""

import json
import os
import sys
import re
import anthropic
from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ssl_utils import patch_ssl
patch_ssl()

load_dotenv()

# ── Constants ────────────────────────────────────────────────────────────────

INTENT_MODEL = "claude-haiku-4-5-20251001"

VALID_INTENTS = {"greeting", "pricing", "technical", "refund", "other"}

SYSTEM_PROMPT = """\
You are an intent classifier for a B2B SaaS customer support system.

Classify the customer message into EXACTLY ONE of these intents:
- greeting    : casual openers, thanks, closings (hi, hello, thanks, bye, how are you)
- pricing     : questions about cost, plans, tiers, upgrades, billing, discounts, trials
- technical   : bugs, errors, how-to questions, API/integration help, feature questions
- refund      : refunds, cancellations, money back, dispute charges
- other       : anything not covered above

Reply with valid JSON ONLY — no markdown, no explanation:
{"intent": "<one of the five intents above>"}
"""


# ── Greeting responses ───────────────────────────────────────────────────────

GREETING_REPLIES = {
    "hi":      "Hi there! 👋 Welcome to LumenX Support. How can I help you today?",
    "hello":   "Hello! Welcome to LumenX Support. What can I assist you with?",
    "thanks":  "You're very welcome! Is there anything else I can help you with today?",
    "bye":     "Thanks for reaching out! Feel free to contact us anytime. Have a great day! — LumenX Support",
    "default": "Hello! Thanks for getting in touch with LumenX Support. How can I assist you today?",
}


def _greeting_reply(message: str) -> str:
    msg = message.lower()
    for keyword, reply in GREETING_REPLIES.items():
        if keyword in msg:
            return reply
    return GREETING_REPLIES["default"]


# ── Main classifier ───────────────────────────────────────────────────────────

def classify(message: str, client: anthropic.Anthropic | None = None) -> dict:
    """
    Classify a customer message.

    Returns:
        {
            "intent":       str,     # one of VALID_INTENTS
            "input_tokens": int,
            "output_tokens": int,
            "cost_usd":     float,
            "greeting_reply": str | None,  # pre-built reply if intent == "greeting"
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

    # Parse JSON — be defensive
    intent = "other"
    try:
        parsed = json.loads(raw)
        intent = parsed.get("intent", "other").lower().strip()
        if intent not in VALID_INTENTS:
            intent = "other"
    except (json.JSONDecodeError, AttributeError):
        # Try extracting intent from raw text if JSON parse fails
        for candidate in VALID_INTENTS:
            if candidate in raw.lower():
                intent = candidate
                break

    # Cost calculation (Haiku pricing)
    input_tokens  = response.usage.input_tokens
    output_tokens = response.usage.output_tokens
    cost_usd = (
        input_tokens  * (0.80 / 1_000_000) +
        output_tokens * (4.00 / 1_000_000)
    )

    result = {
        "intent":         intent,
        "input_tokens":   input_tokens,
        "output_tokens":  output_tokens,
        "cost_usd":       cost_usd,
        "greeting_reply": _greeting_reply(message) if intent == "greeting" else None,
    }

    return result


# ── Quick test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    test_messages = [
        "Hi, how are you?",
        "What is the price of the Pro plan for EmailPilot?",
        "I'm getting a 403 error when calling the API with my token.",
        "I want a refund — this software doesn't work as advertised.",
        "Can I integrate TaskGrid with Slack?",
        "Thanks for your help!",
        "What's the difference between FormCraft and DocuMerge?",
    ]

    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    total_cost = 0.0
    print(f"\n{'Message':<55} {'Intent':<12} {'Cost ($)'}")
    print("-" * 80)
    for msg in test_messages:
        result = classify(msg, client)
        total_cost += result["cost_usd"]
        greeting = f"  -> {result['greeting_reply']}" if result["greeting_reply"] else ""
        print(f"{msg[:54]:<55} {result['intent']:<12} ${result['cost_usd']:.6f}{greeting}")

    print("-" * 80)
    print(f"{'Total cost:':<68} ${total_cost:.6f}")
