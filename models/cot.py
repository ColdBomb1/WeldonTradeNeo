from __future__ import annotations

from datetime import datetime

from sqlalchemy import DateTime, Float, Index, Integer, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from .base import Base


class CotPosition(Base):
    __tablename__ = "cot_positions"
    __table_args__ = (
        UniqueConstraint("currency", "report_date", "source", name="uq_cot_currency_report_source"),
        Index("ix_cot_currency_available", "currency", "available_at"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    currency: Mapped[str] = mapped_column(String(8), index=True)
    market_name: Mapped[str] = mapped_column(String(128))
    report_date: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    available_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    open_interest: Mapped[float] = mapped_column(Float)
    noncommercial_long: Mapped[float] = mapped_column(Float)
    noncommercial_short: Mapped[float] = mapped_column(Float)
    commercial_long: Mapped[float] = mapped_column(Float)
    commercial_short: Mapped[float] = mapped_column(Float)
    pct_noncommercial_long: Mapped[float] = mapped_column(Float)
    pct_noncommercial_short: Mapped[float] = mapped_column(Float)
    noncommercial_net: Mapped[float] = mapped_column(Float)
    noncommercial_net_pct: Mapped[float] = mapped_column(Float)
    source: Mapped[str] = mapped_column(String(32), default="cftc_legacy_futures")
    fetched_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
