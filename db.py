from __future__ import annotations

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from config import load_config
from models.base import Base

_engine = None
_SessionLocal = None


def get_engine():
    global _engine
    if _engine is None:
        cfg = load_config()
        _engine = create_engine(cfg.database_url, future=True)
    return _engine


def get_session():
    global _SessionLocal
    if _SessionLocal is None:
        _SessionLocal = sessionmaker(bind=get_engine(), expire_on_commit=False)
    return _SessionLocal()


def init_db() -> None:
    import models  # noqa: F401 — ensure all models are registered
    Base.metadata.create_all(get_engine())
