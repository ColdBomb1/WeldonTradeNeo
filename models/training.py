from __future__ import annotations

from datetime import datetime

from sqlalchemy import DateTime, Float, Integer, JSON, String, ForeignKey
from sqlalchemy.orm import Mapped, mapped_column

from .base import Base


class TrainingRun(Base):
    __tablename__ = "training_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(128))
    status: Mapped[str] = mapped_column(String(32), default="pending", index=True)
    symbols: Mapped[list] = mapped_column(JSON, default=list)
    timeframes: Mapped[list] = mapped_column(JSON, default=list)
    strategies: Mapped[list] = mapped_column(JSON, default=list)
    start_date: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    end_date: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    iterations_completed: Mapped[int] = mapped_column(Integer, default=0)
    config_snapshot: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    final_report: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class TrainingIteration(Base):
    __tablename__ = "training_iterations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    training_run_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("training_runs.id"), index=True
    )
    iteration_number: Mapped[int] = mapped_column(Integer)
    symbol: Mapped[str] = mapped_column(String(32))
    timeframe: Mapped[str] = mapped_column(String(8))
    strategy_type: Mapped[str] = mapped_column(String(64))
    parameters: Mapped[dict] = mapped_column(JSON, default=dict)
    final_balance: Mapped[float | None] = mapped_column(Float, nullable=True)
    total_trades: Mapped[int | None] = mapped_column(Integer, nullable=True)
    win_rate: Mapped[float | None] = mapped_column(Float, nullable=True)
    max_drawdown: Mapped[float | None] = mapped_column(Float, nullable=True)
    profit_factor: Mapped[float | None] = mapped_column(Float, nullable=True)
    sharpe_ratio: Mapped[float | None] = mapped_column(Float, nullable=True)
    avg_monthly_return: Mapped[float | None] = mapped_column(Float, nullable=True)
    equity_curve: Mapped[list | None] = mapped_column(JSON, nullable=True)
    trades: Mapped[list | None] = mapped_column(JSON, nullable=True)
    claude_analysis: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    suggested_params: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    improvement_pct: Mapped[float | None] = mapped_column(Float, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
