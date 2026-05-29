# ── LumenX Auto-Reply Agent — Dockerfile ────────────────────────────────────
# Multi-stage build: builder installs deps, runtime bakes in the ML model.
#
# Build:  docker build -t lumenx-agent .
# Run:    docker run -p 8001:8001 --env-file .env lumenx-agent

# ── Stage 1: builder ─────────────────────────────────────────────────────────
FROM python:3.12-slim AS builder

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc git curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip \
 && pip install --no-cache-dir --prefix=/install -r requirements.txt

# ── Stage 2: runtime ──────────────────────────────────────────────────────────
FROM python:3.12-slim AS runtime

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy application source
COPY . .

# ── Pre-download sentence-transformers model ───────────────────────────────────
# Must happen here (as root, before USER switch) so the model is baked into
# the image and never needs to be downloaded at runtime.
ENV HF_HUB_CACHE=/app/.cache/huggingface
RUN python -c "\
from sentence_transformers import SentenceTransformer; \
print('Downloading all-MiniLM-L6-v2 ...'); \
SentenceTransformer('all-MiniLM-L6-v2'); \
print('Model ready.')"

# Lock HuggingFace to offline mode — no network calls at runtime
ENV HF_HUB_OFFLINE=1
ENV TRANSFORMERS_OFFLINE=1

# ── Directories ───────────────────────────────────────────────────────────────
# /data  → Railway volume mount point for SQLite persistence
# /app/data → fallback if no volume is mounted
RUN mkdir -p /data /app/data /app/wiki /app/models \
 && chmod 777 /data /app/data

# Default DATABASE_URL points at the Railway volume path
ENV DATABASE_URL=sqlite:////data/agent.db

# ── Non-root user ─────────────────────────────────────────────────────────────
RUN useradd -m -u 1000 agent \
 && chown -R agent:agent /app
USER agent

EXPOSE 8001

# Health check for local Docker runs
HEALTHCHECK --interval=30s --timeout=10s --start-period=90s --retries=3 \
  CMD curl -f http://localhost:8001/health || exit 1

# Default CMD (Railway overrides via startCommand in railway.toml)
RUN chmod +x start.sh
CMD ["sh", "start.sh"]
