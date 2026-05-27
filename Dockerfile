# ── LumenX Auto-Reply Agent — Dockerfile ────────────────────────────────────
# Multi-stage build: builder installs deps, runtime is lean.
#
# Build:
#   docker build -t lumenx-agent .
#
# Run locally:
#   docker run -p 8001:8001 --env-file .env lumenx-agent
#
# Railway: set all env vars in the Railway project settings.
# The /data directory is mounted as a Railway volume for SQLite persistence.

# ── Stage 1: builder ─────────────────────────────────────────────────────────
FROM python:3.12-slim AS builder

WORKDIR /build

# System deps needed to compile wheels (faiss-cpu, sentence-transformers, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy only requirements first — layer-cache until deps change
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

# Create directories Railway volume will mount over
RUN mkdir -p /data /app/wiki /app/models \
 && chmod 777 /data

# Expose FastAPI port
EXPOSE 8001

# Non-root user for security
RUN useradd -m -u 1000 agent \
 && chown -R agent:agent /app
USER agent

# Health check (Railway uses this to decide when the container is ready)
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
  CMD curl -f http://localhost:8001/health || exit 1

# Startup: run wiki build if index is missing, then start the server
CMD ["sh", "-c", "\
  if [ ! -f /app/wiki/index.faiss ]; then \
    echo 'Building wiki index...' && python wiki/build_wiki.py; \
  fi && \
  uvicorn agent.main:app --host 0.0.0.0 --port 8001 --workers 1 \
"]
