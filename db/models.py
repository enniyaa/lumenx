"""SQLAlchemy ORM models for the LumenX Auto-Reply Agent."""

from datetime import datetime
from sqlalchemy import (
    Column, Integer, Float, String, Text, Boolean,
    DateTime, ForeignKey, create_engine
)
from sqlalchemy.orm import DeclarativeBase, relationship
import os


class Base(DeclarativeBase):
    pass


class FeedbackEntry(Base):
    """Every approved or edited reply — the core training signal."""
    __tablename__ = "feedback_entries"

    id               = Column(Integer, primary_key=True, autoincrement=True)
    thread_id        = Column(String(128), nullable=False, index=True)
    customer_msg     = Column(Text, nullable=False)
    draft_text       = Column(Text, nullable=False)
    final_text       = Column(Text, nullable=False)
    intent           = Column(String(32), nullable=False)
    thumbs           = Column(String(8), nullable=True)   # 'up' | 'down' | None
    edit_dist_norm   = Column(Float, nullable=True)       # 0 = no edits, 1 = full rewrite
    approved_as_is   = Column(Boolean, nullable=False)    # True = sent unchanged
    is_bootstrap     = Column(Boolean, default=False)     # True = heuristic label
    created_at       = Column(DateTime, default=datetime.utcnow, nullable=False)

    mlp_row = relationship("MLPTrainingRow", back_populates="feedback_entry", uselist=False)


class CostLog(Base):
    """Every LLM API call — model, tokens, USD cost."""
    __tablename__ = "cost_log"

    id                          = Column(Integer, primary_key=True, autoincrement=True)
    thread_id                   = Column(String(128), nullable=True, index=True)
    task_type                   = Column(String(32), nullable=False)  # 'intent'|'draft'|'summary'|'retrain'
    model                       = Column(String(64), nullable=False)
    input_tokens                = Column(Integer, default=0)
    output_tokens               = Column(Integer, default=0)
    cache_read_input_tokens     = Column(Integer, default=0)
    cache_creation_input_tokens = Column(Integer, default=0)
    cost_usd                    = Column(Float, default=0.0)
    created_at                  = Column(DateTime, default=datetime.utcnow, nullable=False)


class ReviewQueue(Base):
    """Pending replies waiting for human review."""
    __tablename__ = "review_queue"

    id            = Column(Integer, primary_key=True, autoincrement=True)
    thread_id     = Column(String(128), nullable=False, index=True)
    customer_msg  = Column(Text, nullable=True)   # The incoming customer message
    draft_text    = Column(Text, nullable=False)
    confidence    = Column(Float, nullable=True)
    intent        = Column(String(32), nullable=True)
    features_json = Column(Text, nullable=True)   # JSON dict of MLP features
    context_json  = Column(Text, nullable=True)   # Full prompt for dashboard viewer
    cost_usd      = Column(Float, nullable=True)  # Draft generation cost (Phase 9)
    status        = Column(String(16), default="pending")  # pending|approved|edited|rejected|auto_sent
    created_at    = Column(DateTime, default=datetime.utcnow, nullable=False)
    resolved_at   = Column(DateTime, nullable=True)


class MLPTrainingRow(Base):
    """Featurized training rows for the Confidence Net."""
    __tablename__ = "mlp_training"

    id                  = Column(Integer, primary_key=True, autoincrement=True)
    feedback_entry_id   = Column(Integer, ForeignKey("feedback_entries.id"), nullable=True)
    len_ratio           = Column(Float, nullable=False)
    intent_encoded      = Column(Float, nullable=False)
    retrieval_hits      = Column(Float, nullable=False)
    edit_dist_norm      = Column(Float, nullable=False)
    has_price_mention   = Column(Float, nullable=False)
    draft_len_tokens    = Column(Float, nullable=False)
    label               = Column(Integer, nullable=False)   # 1 | 0
    is_bootstrap        = Column(Boolean, default=False)
    created_at          = Column(DateTime, default=datetime.utcnow, nullable=False)

    feedback_entry = relationship("FeedbackEntry", back_populates="mlp_row")


def _build_postgres_url_from_parts() -> str | None:
    """
    Construct a PostgreSQL URL from individual PG* env vars that Railway
    always sets alongside DATABASE_URL.  Falls back to None if any key is missing.
    This sidesteps special-character password encoding issues in the raw URL string.
    """
    from urllib.parse import quote_plus
    host     = os.getenv("PGHOST")
    port     = os.getenv("PGPORT", "5432")
    user     = os.getenv("PGUSER")
    password = os.getenv("PGPASSWORD")
    dbname   = os.getenv("PGDATABASE")
    if all([host, user, password, dbname]):
        return f"postgresql://{quote_plus(user)}:{quote_plus(password)}@{host}:{port}/{dbname}"
    return None


def get_engine(database_url: str | None = None):
    import logging
    logger = logging.getLogger("lumenx.db")

    raw = database_url or os.getenv("DATABASE_URL", "sqlite:///./data/agent.db")

    # Railway (and Heroku) emit postgres:// — SQLAlchemy 2.x requires postgresql://
    if raw.startswith("postgres://"):
        raw = raw.replace("postgres://", "postgresql://", 1)

    if "sqlite" in raw:
        # SQLite: disable thread check so background poller can write
        return create_engine(raw, connect_args={"check_same_thread": False})

    # PostgreSQL path — try to create engine; if the raw URL has unencoded special
    # characters in the password (common on Railway), fall back to PG* vars.
    from sqlalchemy.exc import ArgumentError
    try:
        engine = create_engine(raw, pool_pre_ping=True, pool_size=5, max_overflow=10)
        # Probe the connection immediately so we fail fast here, not at query time
        with engine.connect():
            pass
        return engine
    except (ArgumentError, Exception) as exc:
        logger.warning("DATABASE_URL parse/connect failed (%s); trying PG* vars…", exc)

    fallback = _build_postgres_url_from_parts()
    if fallback:
        logger.info("Connecting via PG* env vars")
        return create_engine(fallback, pool_pre_ping=True, pool_size=5, max_overflow=10)

    # Last resort: local SQLite so the app at least starts
    sqlite_url = "sqlite:///./data/agent.db"
    logger.error("No valid DB URL found — falling back to %s", sqlite_url)
    return create_engine(sqlite_url, connect_args={"check_same_thread": False})


def masked_db_url() -> str:
    """Return DATABASE_URL with the password replaced by ***  (for /health display)."""
    import re
    url = os.getenv("DATABASE_URL", "sqlite:///./data/agent.db").strip()
    # Replace password in scheme://user:PASSWORD@host  — fixed-width lookbehind safe
    return re.sub(r"(://[^:/?#]+:)[^@]+(@)", r"\1***\2", url)


def init_db(database_url: str | None = None):
    engine = get_engine(database_url)
    Base.metadata.create_all(engine)
    return engine
