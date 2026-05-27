"""
LumenX Knowledge Graph Wiki Server
FastAPI server with D3 graph data and RAG-powered query endpoint.
"""

import sys
import os

# Apply SSL patch at the very top
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ssl_utils import patch_ssl, use_hf_cache_only
patch_ssl()
use_hf_cache_only()

import json
import re
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from dotenv import load_dotenv
import anthropic

load_dotenv()

# ── Constants ─────────────────────────────────────────────────────────────────

WIKI_DIR = Path(__file__).parent / "wiki"
CHUNKS_PATH = WIKI_DIR / "chunks.json"
STATIC_DIR = Path(__file__).parent / "static"

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")

PRODUCT_CATEGORIES = {
    "emailpilot": "communication",
    "inboxclean": "communication",
    "chatrelay": "communication",
    "invoiceflow": "finance",
    "billsplit": "finance",
    "receiptvault": "finance",
    "taskgrid": "productivity",
    "timemark": "productivity",
    "kanbanlite": "productivity",
    "meetminutes": "productivity",
    "documerge": "documents",
    "signpath": "documents",
    "formcraft": "documents",
    "audittrail": "analytics",
    "teampulse": "analytics",
    "calendarsync": "utilities",
    "notehub": "utilities",
    "pollwise": "utilities",
    "pixeldeck": "utilities",
    "linkvault": "utilities",
}

# Haiku pricing (per million tokens)
HAIKU_INPUT_PRICE = 1.00 / 1_000_000
HAIKU_OUTPUT_PRICE = 5.00 / 1_000_000

# ── Load chunks ───────────────────────────────────────────────────────────────

def load_chunks():
    with open(CHUNKS_PATH, encoding="utf-8") as f:
        return json.load(f)

CHUNKS = load_chunks()

# ── Build graph data ──────────────────────────────────────────────────────────

def _tool_id(tool_name: str) -> str:
    """Convert tool name to integration node ID."""
    return "int_" + re.sub(r"[^a-z0-9]+", "_", tool_name.lower()).strip("_")


def build_graph_data():
    # Collect product info and sections
    products: dict[str, dict] = {}
    for chunk in CHUNKS:
        pid = chunk["product_id"]
        if pid not in products:
            products[pid] = {
                "id": pid,
                "name": chunk["product_name"],
                "type": "product",
                "category": PRODUCT_CATEGORIES.get(pid, "utilities"),
                "sections": {}
            }
        section_key = chunk["section"]
        products[pid]["sections"][section_key] = chunk["text"]

    # Parse integrations: map tool_name -> set of product_ids
    tool_products: dict[str, set] = {}
    for chunk in CHUNKS:
        if chunk["section"] != "integrations":
            continue
        text = chunk["text"]
        # Extract after "Integrates with: "
        m = re.search(r"Integrates with:\s*(.+)", text, re.IGNORECASE)
        if not m:
            continue
        tools_str = m.group(1)
        tools = [t.strip() for t in tools_str.split(",") if t.strip()]
        for tool in tools:
            tid = _tool_id(tool)
            if tid not in tool_products:
                tool_products[tid] = {"name": tool, "products": set()}
            tool_products[tid]["products"].add(chunk["product_id"])

    # Only keep integrations used by 2+ products (true cross-references)
    cross_ref_tools = {
        tid: info
        for tid, info in tool_products.items()
        if len(info["products"]) >= 2
    }

    # Build nodes list
    nodes = list(products.values())
    for tid, info in cross_ref_tools.items():
        nodes.append({
            "id": tid,
            "name": info["name"],
            "type": "integration",
            "category": "integration",
            "products": sorted(info["products"])
        })

    # Build links
    links = []
    for tid, info in cross_ref_tools.items():
        for pid in info["products"]:
            links.append({
                "source": pid,
                "target": tid,
                "type": "integrates"
            })

    return {
        "nodes": nodes,
        "links": links,
        "stats": {
            "products": len(products),
            "chunks": len(CHUNKS),
            "integrations": len(cross_ref_tools)
        }
    }


# Build once at startup
GRAPH_DATA = build_graph_data()

# ── FastAPI app ───────────────────────────────────────────────────────────────

app = FastAPI(title="LumenX Knowledge Graph")

# Serve static files
STATIC_DIR.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
async def serve_index():
    index_path = STATIC_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="index.html not found")
    return FileResponse(str(index_path))


@app.get("/api/graph")
async def get_graph():
    return JSONResponse(content=GRAPH_DATA)


@app.get("/api/stats")
async def get_stats():
    return {
        "products": 20,
        "chunks": len(CHUNKS),
        "integrations": len([n for n in GRAPH_DATA["nodes"] if n["type"] == "integration"])
    }


class QueryRequest(BaseModel):
    question: str


@app.post("/api/query")
async def query_wiki(req: QueryRequest):
    from wiki import retriever

    question = req.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    # Retrieve top-4 chunks
    chunks = retriever.retrieve(question, intent="other", k=4)
    if not chunks:
        return {
            "answer": "I couldn't find relevant information for your question.",
            "sources": [],
            "meta": {"model": "claude-haiku-4-5", "input_tokens": 0, "output_tokens": 0, "cost_usd": 0}
        }

    # Build context
    context_parts = []
    for c in chunks:
        context_parts.append(c["text"])
    context = "\n\n---\n\n".join(context_parts)

    # Build source list for response
    sources = [
        {
            "product_id": c["product_id"],
            "product_name": c["product_name"],
            "section": c["section"],
            "score": round(c.get("score", 0.0), 4),
            "snippet": c["text"][:200] + ("..." if len(c["text"]) > 200 else "")
        }
        for c in chunks
    ]

    system_prompt = (
        "You are a helpful assistant for the LumenX product wiki. "
        "Answer questions based strictly on the provided context. "
        "Cite every fact with **[ProductName — section]** format inline. "
        "Be concise and accurate. If the context does not contain the answer, say so clearly."
    )

    user_message = f"""Context from the LumenX wiki:

{context}

Question: {question}

Answer the question using only the context above. Cite every fact with **[ProductName — section]** format."""

    # Call Claude Haiku
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    response = client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=1024,
        system=system_prompt,
        messages=[{"role": "user", "content": user_message}]
    )

    answer_text = ""
    for block in response.content:
        if block.type == "text":
            answer_text += block.text

    input_tokens = response.usage.input_tokens
    output_tokens = response.usage.output_tokens
    cost_usd = (input_tokens * HAIKU_INPUT_PRICE) + (output_tokens * HAIKU_OUTPUT_PRICE)

    return {
        "answer": answer_text,
        "sources": sources,
        "meta": {
            "model": response.model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cost_usd": round(cost_usd, 6)
        }
    }
