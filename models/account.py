from __future__ import annotations

from datetime import datetime

from sqlalchemy import DateTime, Float, Integer, JSON, String
from sqlalchemy.orm import Mapped, mapped_column

from .base import Base


class AccountSnapshot(Base):
    __tablename__ = "account_snapshots"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    server_id: Mapped[str] = mapped_column(String(64), index=True)
    account_name: Mapped[str] = mapped_column(String(64), index=True)
    ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    balance: Mapped[float] = mapped_column(Float)
    equity: Mapped[float] = mapped_column(Float)
    raw: Mapped[dict | None] = mapped_column(JSON, nullable=True)


class AccountDeal(Base):
    __tablename__ = "account_deals"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    account_name: Mapped[str] = mapped_column(String(64), index=True)
    platform: Mapped[str] = mapped_column(String(16), index=True)
    ticket: Mapped[str] = mapped_column(String(64), index=True)
    symbol: Mapped[str] = mapped_column(String(32))
    side: Mapped[str] = mapped_column(String(8))
    volume: Mapped[float] = mapped_column(Float)
    price: Mapped[float] = mapped_column(Float)
    profit: Mapped[float] = mapped_column(Float)
    closed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    raw: Mapped[dict | None] = mapped_column(JSON, nullable=True)
