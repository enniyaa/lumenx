"""Database session factory."""

from contextlib import contextmanager
from sqlalchemy.orm import sessionmaker, Session
from .models import get_engine

_engine = None
_SessionLocal = None


def _get_session_factory():
    global _engine, _SessionLocal
    if _SessionLocal is None:
        _engine = get_engine()
        _SessionLocal = sessionmaker(bind=_engine, autoflush=False, autocommit=False)
    return _SessionLocal


@contextmanager
def get_db() -> Session:
    factory = _get_session_factory()
    db = factory()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()
