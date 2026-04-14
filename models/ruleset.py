from __future__ import annotations

from datetime import datetime

from sqlalchemy import DateTime, Float, Integer, JSON, String, Text, ForeignKey
from sqlalchemy.orm import Mapped, mapped_column

from .base import Base


class RuleSet(Base):
    __tablename__ = "rule_sets"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(128))
    description: Mapped[str] = mapped_column(String(512), default="")
    status: Mapped[str] = mapped_column(String(32), default="inactive", index=True)
    rules_text: Mapped[str] = mapped_column(Text, default="")
    parameters: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    symbols: Mapped[list] = mapped_column(JSON, default=list)
    timeframes: Mapped[list] = mapped_column(JSON, default=list)
    version: Mapped[int] = mapped_column(Integer, default=1)
    parent_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("rule_sets.id"), nullable=True
    )
    performance_metrics: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
