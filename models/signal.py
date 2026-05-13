from __future__ import annotations

from datetime import datetime

from sqlalchemy import DateTime, Float, Integer, JSON, String, Text, ForeignKey
from sqlalchemy.orm import Mapped, mapped_column

from .base import Base


class Signal(Base):
    __tablename__ = "signals"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    symbol: Mapped[str] = mapped_column(String(32), index=True)
    timeframe: Mapped[str] = mapped_column(String(8))
    strategy_type: Mapped[str] = mapped_column(String(64))
    side: Mapped[str] = mapped_column(String(8))  # "buy" or "sell"
    price: Mapped[float] = mapped_column(Float)
    stop_loss: Mapped[float | None] = mapped_column(Float, nullable=True)
    take_profit: Mapped[float | None] = mapped_column(Float, nullable=True)
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    # Unbounded: Claude's multi-TF reasoning can run 2-3KB; VARCHAR(512) was
    # truncating inserts and rolling back the whole signal. create_all() won't
    # migrate existing columns, so an ALTER was applied manually.
    reason: Mapped[str] = mapped_column(Text, default="")
    claude_analysis: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    status: Mapped[str] = mapped_column(String(32), default="pending", index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    resolved_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class LiveTrade(Base):
    __tablename__ = "live_trades"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    signal_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("signals.id"), nullable=True, index=True
    )
    symbol: Mapped[str] = mapped_column(String(32), index=True)
    side: Mapped[str] = mapped_column(String(8))
    entry_price: Mapped[float] = mapped_column(Float)
    current_price: Mapped[float | None] = mapped_column(Float, nullable=True)
    stop_loss: Mapped[float] = mapped_column(Float)
    take_profit: Mapped[float] = mapped_column(Float)
    volume: Mapped[float] = mapped_column(Float)
    platform_ticket: Mapped[int | None] = mapped_column(Integer, nullable=True, index=True)
    platform: Mapped[str] = mapped_column(String(16), default="mt5")
    status: Mapped[str] = mapped_column(String(32), default="pending", index=True)
    pnl: Mapped[float | None] = mapped_column(Float, nullable=True)
    exit_price: Mapped[float | None] = mapped_column(Float, nullable=True)
    exit_type: Mapped[str | None] = mapped_column(String(16), nullable=True)
    opened_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    closed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    error_message: Mapped[str | None] = mapped_column(String(512), nullable=True)
    raw: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    trade_analysis: Mapped[dict | None] = mapped_column(JSON, nullable=True)
