from __future__ import annotations

from datetime import datetime

from sqlalchemy import DateTime, Float, Integer, JSON, String
from sqlalchemy.orm import Mapped, mapped_column

from .base import Base


class NewsEvent(Base):
    __tablename__ = "news_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    title: Mapped[str] = mapped_column(String(256))
    currency: Mapped[str] = mapped_column(String(8), index=True)
    impact: Mapped[str] = mapped_column(String(16), index=True)  # low, medium, high
    event_time: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    actual: Mapped[str | None] = mapped_column(String(32), nullable=True)
    forecast: Mapped[str | None] = mapped_column(String(32), nullable=True)
    previous: Mapped[str | None] = mapped_column(String(32), nullable=True)
    source: Mapped[str] = mapped_column(String(32), default="forex_factory")
    fetched_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))


class SentimentScore(Base):
    __tablename__ = "sentiment_scores"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    symbol: Mapped[str] = mapped_column(String(32), index=True)
    score: Mapped[float] = mapped_column(Float)  # -1.0 to 1.0
    source: Mapped[str] = mapped_column(String(32), default="claude_analysis")
    details: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
