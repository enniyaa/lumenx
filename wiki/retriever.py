"""
LLM Wiki Retriever
Loads the FAISS index + chunks.json and retrieves the top-k most relevant
passages for a given query string.
"""

import json
import os
import sys
import numpy as np
from functools import lru_cache
from typing import Optional

# Apply SSL patch and use offline HF cache (model already downloaded by build_wiki.py)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ssl_utils import patch_ssl, use_hf_cache_only
patch_ssl()
use_hf_cache_only()

WIKI_DIR    = os.path.dirname(os.path.abspath(__file__))
INDEX_PATH  = os.path.join(WIKI_DIR, "index.faiss")
CHUNKS_PATH = os.path.join(WIKI_DIR, "chunks.json")

# Number of chunks to return for each intent type
K_BY_INTENT = {
    "technical": 4,
    "pricing":   3,
    "refund":    3,
    "other":     2,
    "greeting":  0,  # skip wiki for greetings
}
DEFAULT_K = 3


@lru_cache(maxsize=1)
def _load_resources():
    """Load FAISS index and chunks once, cache in memory."""
    try:
        import faiss
        from sentence_transformers import SentenceTransformer
    except ImportError:
        raise RuntimeError("Install: pip install faiss-cpu sentence-transformers")

    if not os.path.exists(INDEX_PATH):
        raise FileNotFoundError(
            f"FAISS index not found at {INDEX_PATH}. Run: python wiki/build_wiki.py"
        )

    index  = faiss.read_index(INDEX_PATH)
    model  = SentenceTransformer("all-MiniLM-L6-v2")

    with open(CHUNKS_PATH, encoding="utf-8") as f:
        chunks = json.load(f)

    return index, model, chunks


def retrieve(query: str, intent: str = "other", k: Optional[int] = None) -> list[dict]:
    """
    Return top-k chunks most relevant to `query`.

    Args:
        query:  The user's message (or a combined query string)
        intent: One of greeting|pricing|technical|refund|other
        k:      Override the intent-default k

    Returns:
        List of chunk dicts: {product_id, product_name, section, text, score}
    """
    import faiss

    if k is None:
        k = K_BY_INTENT.get(intent, DEFAULT_K)

    if k == 0:
        return []

    index, model, chunks = _load_resources()

    # Embed and normalise query
    embedding = model.encode([query], show_progress_bar=False)
    embedding = np.array(embedding, dtype="float32")
    faiss.normalize_L2(embedding)

    scores, indices = index.search(embedding, k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0:
            continue
        chunk = dict(chunks[idx])
        chunk["score"] = float(score)
        results.append(chunk)

    return results


def retrieve_text(query: str, intent: str = "other", k: Optional[int] = None) -> str:
    """
    Convenience wrapper — returns chunks as a single formatted string
    ready to inject into a prompt.
    """
    chunks = retrieve(query, intent, k)
    if not chunks:
        return ""

    parts = []
    for c in chunks:
        parts.append(c["text"])

    return "\n\n---\n\n".join(parts)


def wiki_is_ready() -> bool:
    """Return True if index and chunks files exist."""
    return os.path.exists(INDEX_PATH) and os.path.exists(CHUNKS_PATH)


# ── Quick test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    query  = " ".join(sys.argv[1:]) or "What is the refund policy for EmailPilot?"
    intent = "pricing"
    print(f"Query : {query}")
    print(f"Intent: {intent}\n")
    result = retrieve_text(query, intent=intent, k=3)
    print(result or "(no results)")
