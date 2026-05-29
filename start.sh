#!/bin/sh
# Railway injects PORT as an env var; default to 8001 for local runs.
PORT=${PORT:-8001}
exec uvicorn agent.main:app --host 0.0.0.0 --port "$PORT" --workers 1
