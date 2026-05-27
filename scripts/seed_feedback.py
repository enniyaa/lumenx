"""
Seed realistic FeedbackEntry rows for Phase 6 testing.
Creates 18 approved/edited examples across all intents, then builds the
FAISS feedback index so search_feedback() can return results.

Run once:
    python scripts/seed_feedback.py

Safe to re-run — checks existing count before inserting.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ssl_utils import patch_ssl; patch_ssl()
from dotenv import load_dotenv; load_dotenv()

from datetime import datetime, timezone
from db.session import get_db
from db.models import FeedbackEntry
from db.feedback_log import rebuild_feedback_index

# ── Seed data — realistic support Q&A pairs ───────────────────────────────────
SEEDS = [
    # Pricing
    dict(intent="pricing",
         customer_msg="What is the monthly price for the Pro plan of TaskGrid?",
         draft_text="TaskGrid Pro is $29/month per seat, billed monthly. Annual billing saves 20%. A 14-day free trial is available with no credit card required.\n\n— LumenX Support",
         approved_as_is=True),
    dict(intent="pricing",
         customer_msg="Do you offer any annual discount or a free trial period?",
         draft_text="Yes — all plans include a 14-day free trial. Annual billing saves you ~20% compared to monthly. You can switch to annual at any time from your Billing Settings.\n\n— LumenX Support",
         approved_as_is=True),
    dict(intent="pricing",
         customer_msg="Is there a startup or nonprofit discount available?",
         draft_text="We don't currently publish a startup discount, but I'd recommend reaching out to our sales team — they handle case-by-case pricing for nonprofits and early-stage startups.\n\n— LumenX Support",
         approved_as_is=False,
         final_override="We do offer nonprofit pricing on a case-by-case basis. Please email sales@lumenx.com with proof of nonprofit status and we'll apply a 30% discount to your account.\n\n— LumenX Support"),
    dict(intent="pricing",
         customer_msg="How does per-seat pricing work for InvoiceFlow?",
         draft_text="InvoiceFlow charges per active seat per month. An active seat is any team member who logs in during the billing period. You can add or remove seats at any time; billing is pro-rated.\n\n— LumenX Support",
         approved_as_is=True),

    # Technical
    dict(intent="technical",
         customer_msg="I'm getting a 403 Forbidden error when calling the TaskGrid REST API.",
         draft_text="A 403 Forbidden usually means your API key doesn't have the right permissions. Please check that your key has read+write scope enabled in Settings → API Keys. If the scope looks correct, try regenerating the key.\n\n— LumenX Support",
         approved_as_is=True),
    dict(intent="technical",
         customer_msg="How do I integrate CalendarSync with our Slack workspace?",
         draft_text="To connect CalendarSync to Slack: go to Settings → Integrations → Slack, click Connect, and authorise the OAuth flow. Events will then sync to the Slack channel you select. Let me know if you hit any errors during the OAuth step.\n\n— LumenX Support",
         approved_as_is=True),
    dict(intent="technical",
         customer_msg="Our webhook isn't receiving events from NoteHub. How do I debug this?",
         draft_text="First, confirm your webhook URL responds to a manual POST with 200 OK. Then check NoteHub → Settings → Webhooks for any delivery failures — we log the HTTP status and body for each attempt. If you see 5xx errors from your endpoint, that's usually a TLS or routing issue.\n\n— LumenX Support",
         approved_as_is=True),
    dict(intent="technical",
         customer_msg="DataSync shows 'sync paused' but I haven't changed any settings.",
         draft_text="'Sync paused' usually means we received 3 consecutive auth errors from your connected account. Please re-authenticate your data source in Settings → Connected Accounts → DataSync. This clears the error state and resumes the sync.\n\n— LumenX Support",
         approved_as_is=True),

    # Refund
    dict(intent="refund",
         customer_msg="I was charged twice this month for InvoiceFlow. I need a refund.",
         draft_text="I'm sorry about the duplicate charge — that's clearly a billing error and we'll resolve it right away. Could you share the last 4 digits of the card that was charged? I'll flag this to our billing team for a same-day correction.\n\n— LumenX Support",
         approved_as_is=True),
    dict(intent="refund",
         customer_msg="I cancelled within the first week but was still charged. Can I get a refund?",
         draft_text="If you cancelled within 14 days of your first charge, you're covered by our refund policy. I'll initiate a full refund to your original payment method — please allow 5–10 business days for it to appear.\n\n— LumenX Support",
         approved_as_is=True),
    dict(intent="refund",
         customer_msg="How do I cancel my TaskGrid subscription?",
         draft_text="To cancel, go to Settings → Billing → Cancel Subscription. Your access continues until the end of the current billing period. After cancellation, your data is retained for 30 days before deletion.\n\n— LumenX Support",
         approved_as_is=True),
    dict(intent="refund",
         customer_msg="I downgraded my plan but the price didn't change on my invoice.",
         draft_text="Plan changes take effect at the start of your next billing cycle, so your current invoice still reflects the previous plan. You'll see the lower rate on your next statement. If you'd prefer a pro-rated credit applied immediately, I can arrange that.\n\n— LumenX Support",
         approved_as_is=False,
         final_override="Downgrades apply at the start of your next billing cycle. Your upcoming invoice will reflect the new lower price. If you'd like a pro-rated credit for the remainder of this cycle, reply here and I'll apply it manually.\n\n— LumenX Support"),

    # Other
    dict(intent="other",
         customer_msg="Can multiple team members share one NoteHub account?",
         draft_text="NoteHub is designed for individual use, but the Team plan ($9/seat/month) includes shared spaces and collaborative editing for multiple users. I'd recommend upgrading if you have more than one person needing access.\n\n— LumenX Support",
         approved_as_is=True),
    dict(intent="other",
         customer_msg="What is the data retention policy for deleted notes in NoteHub?",
         draft_text="Deleted notes are moved to Trash and retained for 30 days before permanent deletion. You can restore them any time during that window from the Trash view.\n\n— LumenX Support",
         approved_as_is=True),
    dict(intent="other",
         customer_msg="Does TaskGrid integrate with Jira or Linear?",
         draft_text="TaskGrid currently integrates natively with Jira via Settings → Integrations. A Linear integration is on our public roadmap — you can vote for it at feedback.lumenx.com to get notified when it ships.\n\n— LumenX Support",
         approved_as_is=True),
    dict(intent="other",
         customer_msg="Is there a mobile app for CalendarSync?",
         draft_text="Yes — CalendarSync has iOS and Android apps available in the App Store and Google Play. They support all core features including event creation, reminders, and multi-calendar sync.\n\n— LumenX Support",
         approved_as_is=True),
    dict(intent="other",
         customer_msg="Can I export all my data from DataSync?",
         draft_text="Absolutely. Go to Settings → Data → Export. You can export your data as CSV or JSON. Large exports are emailed as a download link within a few minutes.\n\n— LumenX Support",
         approved_as_is=True),
    dict(intent="other",
         customer_msg="How do I add a new admin user to our LumenX account?",
         draft_text="To add an admin, go to Settings → Team → Invite Member and set the role to Admin. They'll receive an email invitation. Admin access includes billing and all workspace settings.\n\n— LumenX Support",
         approved_as_is=True),
]


def main():
    # Check how many rows already exist
    with get_db() as db:
        existing = db.query(FeedbackEntry).count()

    if existing >= len(SEEDS):
        print(f"DB already has {existing} FeedbackEntry rows — skipping seed.")
    else:
        print(f"Seeding {len(SEEDS)} FeedbackEntry rows...")
        now = datetime.now(timezone.utc)

        try:
            import Levenshtein
            def edit_dist(a, b):
                return Levenshtein.distance(a, b) / max(len(a), len(b), 1)
        except ImportError:
            def edit_dist(a, b):
                return 0.0 if a == b else 0.5

        with get_db() as db:
            for s in SEEDS:
                draft     = s["draft_text"]
                final     = s.get("final_override", draft)
                approved  = s["approved_as_is"]
                edn       = edit_dist(draft, final)
                db.add(FeedbackEntry(
                    thread_id=f"seed-{s['intent']}-{SEEDS.index(s):02d}",
                    customer_msg=s["customer_msg"],
                    draft_text=draft,
                    final_text=final,
                    intent=s["intent"],
                    thumbs=None,
                    edit_dist_norm=edn,
                    approved_as_is=approved,
                    is_bootstrap=False,
                    created_at=now,
                ))
        print(f"Inserted {len(SEEDS)} rows.")

    # Build / rebuild the FAISS feedback index
    print("Building feedback FAISS index...")
    n = rebuild_feedback_index()
    print(f"Index built with {n} entries.")
    return n


if __name__ == "__main__":
    count = main()
    print(f"\nPhase 6 seed complete — {count} entries indexed.")
