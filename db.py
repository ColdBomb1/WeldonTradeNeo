from __future__ import annotations

from sqlalchemy import create_engine, inspect, text
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
    engine = get_engine()
    Base.metadata.create_all(engine)
    _run_lightweight_migrations(engine)


def _run_lightweight_migrations(engine) -> None:
    """Patch known nullable columns in older databases."""
    if engine.dialect.name != "postgresql":
        return

    inspector = inspect(engine)
    table_names = set(inspector.get_table_names())
    if "live_trades" not in table_names:
        return

    live_trade_columns = {col["name"] for col in inspector.get_columns("live_trades")}
    statements = []
    if "raw" not in live_trade_columns:
        statements.append("ALTER TABLE live_trades ADD COLUMN raw JSON")
    if "trade_analysis" not in live_trade_columns:
        statements.append("ALTER TABLE live_trades ADD COLUMN trade_analysis JSON")

    if not statements:
        return

    with engine.begin() as conn:
        for statement in statements:
            conn.execute(text(statement))
