"""
LLM Wiki Builder — Phase 1
Fetches all 20 LumenX products from the admin API, chunks each product
into focused passages, embeds them with sentence-transformers, and
stores a FAISS index for fast retrieval at query time.

Usage:
    python wiki/build_wiki.py
"""

import json
import os
import ssl
import sys
import re
import urllib3
import requests
import numpy as np

# ── Corporate TLS proxy fix ────────────────────────────────────────────────
# Many corporate networks intercept HTTPS with their own CA.  We disable
# certificate verification for all outbound HTTPS calls so that requests to
# HuggingFace and the LumenX API succeed.

ssl._create_default_https_context = ssl._create_unverified_context
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Patch the huggingface_hub HTTP session (configure_http_backend added in 0.22)
try:
    from huggingface_hub import configure_http_backend

    def _no_verify_backend() -> requests.Session:
        session = requests.Session()
        session.verify = False
        return session

    configure_http_backend(backend_factory=_no_verify_backend)
except (ImportError, AttributeError):
    pass  # Older version — fall back to env vars

os.environ["CURL_CA_BUNDLE"] = ""
os.environ["REQUESTS_CA_BUNDLE"] = ""

# Allow running from repo root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

BASE_URL   = os.getenv("LUMENX_BASE_URL", "https://lumenx-demo.up.railway.app")
ADMIN_TOKEN = os.getenv("LUMENX_ADMIN_TOKEN", "")
WIKI_DIR   = os.path.join(os.path.dirname(__file__))
INDEX_PATH = os.path.join(WIKI_DIR, "index.faiss")
CHUNKS_PATH = os.path.join(WIKI_DIR, "chunks.json")

HEADERS = {"X-Admin-Token": ADMIN_TOKEN}


# ── Fetch ────────────────────────────────────────────────────────────────────

def fetch_products() -> list[dict]:
    print("Fetching products from LumenX API …")
    resp = requests.get(f"{BASE_URL}/api/admin/products", headers=HEADERS, timeout=30, verify=False)
    resp.raise_for_status()
    data = resp.json()
    # API may return {products: [...]} or [...]
    if isinstance(data, list):
        products = data
    else:
        products = data.get("products", data.get("data", []))
    print(f"  → {len(products)} products fetched")
    return products


# ── Chunking ─────────────────────────────────────────────────────────────────

def _safe(val):
    """Return a non-None string."""
    if val is None:
        return ""
    if isinstance(val, (dict, list)):
        return json.dumps(val)
    return str(val)


def chunk_product(product: dict) -> list[dict]:
    """
    Split one product dict into focused passages.
    Each chunk: {product_id, product_name, section, text}
    Sections: overview, features, pricing, refund, integrations, support, policies
    """
    pid  = product.get("id", product.get("slug", "unknown"))
    name = product.get("name", pid)
    chunks = []

    def add(section: str, text: str):
        text = text.strip()
        if text:
            chunks.append({
                "product_id":   pid,
                "product_name": name,
                "section":      section,
                "text":         f"[{name} — {section}]\n{text}",
            })

    # Overview / description
    overview_parts = []
    for key in ("description", "overview", "summary", "tagline"):
        if product.get(key):
            overview_parts.append(_safe(product[key]))
    if overview_parts:
        add("overview", " ".join(overview_parts))

    # Target audience
    if product.get("target_audience"):
        add("target_audience", f"Designed for: {_safe(product['target_audience'])}")

    # Features
    features = product.get("features", product.get("feature_list", []))
    if features:
        if isinstance(features, list):
            feat_text = "\n".join(f"• {f}" for f in features)
        else:
            feat_text = _safe(features)
        add("features", feat_text)

    # Pricing — critical, never hallucinate
    pricing = product.get("pricing", product.get("pricing_tiers", product.get("tiers")))
    if pricing:
        if isinstance(pricing, list):
            lines = []
            for tier in pricing:
                if isinstance(tier, dict):
                    tier_name  = tier.get("name", tier.get("tier", ""))
                    tier_price = tier.get("price", tier.get("cost", ""))
                    tier_desc  = tier.get("description", tier.get("features", ""))
                    lines.append(f"• {tier_name}: {tier_price} — {_safe(tier_desc)}")
                else:
                    lines.append(f"• {tier}")
            add("pricing", "\n".join(lines))
        else:
            add("pricing", _safe(pricing))

    # Refund & cancellation policy — critical
    refund_parts = []
    for key in ("refund_policy", "refund", "cancellation_policy", "cancellation", "money_back"):
        if product.get(key):
            refund_parts.append(f"{key.replace('_', ' ').title()}: {_safe(product[key])}")
    if refund_parts:
        add("refund_policy", "\n".join(refund_parts))

    # Integrations
    integrations = product.get("integrations", product.get("integration_list", []))
    if integrations:
        if isinstance(integrations, list):
            add("integrations", "Integrates with: " + ", ".join(str(i) for i in integrations))
        else:
            add("integrations", _safe(integrations))

    # Support SLA
    support_parts = []
    for key in ("support_sla", "support", "sla", "support_level"):
        if product.get(key):
            support_parts.append(f"{_safe(product[key])}")
    if support_parts:
        add("support_sla", "Support: " + " | ".join(support_parts))

    # Free trial
    trial_parts = []
    for key in ("free_trial", "trial", "trial_period", "trial_days"):
        if product.get(key) is not None:
            trial_parts.append(f"{key.replace('_', ' ')}: {_safe(product[key])}")
    if trial_parts:
        add("free_trial", "\n".join(trial_parts))

    # Discounts
    for key in ("discounts", "discount", "promotions"):
        if product.get(key):
            add("discounts", _safe(product[key]))
            break

    # Catch any remaining top-level string fields not yet covered
    covered = {
        "id", "slug", "name", "description", "overview", "summary", "tagline",
        "features", "feature_list", "pricing", "pricing_tiers", "tiers",
        "refund_policy", "refund", "cancellation_policy", "cancellation", "money_back",
        "integrations", "integration_list", "support_sla", "support", "sla", "support_level",
        "free_trial", "trial", "trial_period", "trial_days", "discounts", "discount",
        "promotions", "target_audience",
    }
    extras = []
    for k, v in product.items():
        if k not in covered and isinstance(v, str) and len(v) > 10:
            extras.append(f"{k.replace('_', ' ').title()}: {v}")
    if extras:
        add("additional_info", "\n".join(extras))

    return chunks


# ── Embed & Index ─────────────────────────────────────────────────────────────

def build_index(chunks: list[dict]):
    try:
        from sentence_transformers import SentenceTransformer
        import faiss
    except ImportError:
        print("ERROR: sentence-transformers or faiss-cpu not installed.")
        print("Run: pip install sentence-transformers faiss-cpu")
        sys.exit(1)

    print(f"Embedding {len(chunks)} chunks with all-MiniLM-L6-v2 …")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    texts = [c["text"] for c in chunks]
    embeddings = model.encode(texts, show_progress_bar=True, batch_size=32)
    embeddings = np.array(embeddings, dtype="float32")

    # Normalise for cosine similarity (inner product on unit vectors)
    faiss.normalize_L2(embeddings)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # Inner Product (cosine after normalisation)
    index.add(embeddings)

    faiss.write_index(index, INDEX_PATH)
    print(f"  → FAISS index saved: {INDEX_PATH}  ({index.ntotal} vectors, dim={dim})")
    return index


def save_chunks(chunks: list[dict]):
    with open(CHUNKS_PATH, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)
    print(f"  → Chunks saved: {CHUNKS_PATH}  ({len(chunks)} entries)")


# ── Main ─────────────────────────────────────────────────────────────────────

def build_wiki():
    if not ADMIN_TOKEN:
        print("ERROR: LUMENX_ADMIN_TOKEN not set in environment")
        sys.exit(1)

    products = fetch_products()

    all_chunks = []
    for p in products:
        c = chunk_product(p)
        all_chunks.extend(c)
        print(f"  {p.get('name', p.get('id', '?')):30s}  → {len(c)} chunks")

    print(f"\nTotal chunks: {len(all_chunks)}")
    save_chunks(all_chunks)
    build_index(all_chunks)
    print("\n✅ LLM Wiki built successfully.")


if __name__ == "__main__":
    build_wiki()
