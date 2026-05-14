"""Performance analytics: live vs backtest comparison, strategy scorecards, P&L tracking."""

from __future__ import annotations

import logging
from collections import defaultdict
from datetime import datetime, timedelta, timezone

from db import get_session
from models.signal import Signal, LiveTrade
from models.trade_plan import BacktestRun
from services.performance_archive import filter_query_after_cutoff

logger = logging.getLogger(__name__)


def get_daily_pnl(days: int = 30) -> list[dict]:
    """Daily realized P&L breakdown from closed trades."""
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    session = get_session()
    try:
        query = session.query(LiveTrade).filter(
            LiveTrade.status == "closed",
            LiveTrade.pnl.isnot(None),
        )
        query = filter_query_after_cutoff(query, LiveTrade.closed_at)
        trades = query.filter(LiveTrade.closed_at >= cutoff).order_by(LiveTrade.closed_at.asc()).all()

        daily = defaultdict(lambda: {"pnl": 0.0, "trades": 0, "wins": 0})
        for t in trades:
            day_key = t.closed_at.strftime("%Y-%m-%d")
            daily[day_key]["pnl"] += t.pnl
            daily[day_key]["trades"] += 1
            if t.pnl > 0:
                daily[day_key]["wins"] += 1

        return [
            {
                "date": k,
                "pnl": round(v["pnl"], 2),
                "trades": v["trades"],
                "wins": v["wins"],
                "win_rate": round(v["wins"] / v["trades"], 4) if v["trades"] else 0,
            }
            for k, v in sorted(daily.items())
        ]
    finally:
        session.close()


def get_strategy_scorecard() -> list[dict]:
    """Rank strategies by live trading performance."""
    session = get_session()
    try:
        # Get all closed trades joined with their signals for strategy info
        query = (
            session.query(LiveTrade, Signal)
            .join(Signal, LiveTrade.signal_id == Signal.id)
            .filter(LiveTrade.status == "closed", LiveTrade.pnl.isnot(None))
        )
        query = filter_query_after_cutoff(query, LiveTrade.closed_at)
        trades = query.all()

        by_strategy = defaultdict(lambda: {
            "trades": 0, "wins": 0, "losses": 0,
            "total_pnl": 0.0, "gross_profit": 0.0, "gross_loss": 0.0,
            "confidences": [], "max_drawdown": 0.0, "peak_pnl": 0.0,
            "running_pnl": 0.0,
        })

        for trade, signal in trades:
            s = by_strategy[signal.strategy_type]
            s["trades"] += 1
            s["total_pnl"] += trade.pnl
            s["running_pnl"] += trade.pnl

            if trade.pnl > 0:
                s["wins"] += 1
                s["gross_profit"] += trade.pnl
            else:
                s["losses"] += 1
                s["gross_loss"] += abs(trade.pnl)

            if signal.confidence is not None:
                s["confidences"].append((signal.confidence, trade.pnl > 0))

            # Track drawdown
            if s["running_pnl"] > s["peak_pnl"]:
                s["peak_pnl"] = s["running_pnl"]
            dd = s["peak_pnl"] - s["running_pnl"]
            if dd > s["max_drawdown"]:
                s["max_drawdown"] = dd

        result = []
        for name, s in by_strategy.items():
            win_rate = s["wins"] / s["trades"] if s["trades"] else 0
            profit_factor = s["gross_profit"] / s["gross_loss"] if s["gross_loss"] > 0 else float("inf") if s["gross_profit"] > 0 else 0

            # Confidence accuracy: did high-confidence signals perform better?
            high_conf = [(c, w) for c, w in s["confidences"] if c >= 0.7]
            low_conf = [(c, w) for c, w in s["confidences"] if c < 0.7]
            high_conf_accuracy = sum(1 for _, w in high_conf if w) / len(high_conf) if high_conf else None
            low_conf_accuracy = sum(1 for _, w in low_conf if w) / len(low_conf) if low_conf else None

            result.append({
                "strategy": name,
                "trades": s["trades"],
                "wins": s["wins"],
                "losses": s["losses"],
                "win_rate": round(win_rate, 4),
                "total_pnl": round(s["total_pnl"], 2),
                "profit_factor": round(profit_factor, 2) if profit_factor != float("inf") else 999.0,
                "max_drawdown": round(s["max_drawdown"], 2),
                "avg_pnl": round(s["total_pnl"] / s["trades"], 2) if s["trades"] else 0,
                "high_conf_accuracy": round(high_conf_accuracy, 4) if high_conf_accuracy is not None else None,
                "low_conf_accuracy": round(low_conf_accuracy, 4) if low_conf_accuracy is not None else None,
            })

        result.sort(key=lambda x: x["total_pnl"], reverse=True)
        return result
    finally:
        session.close()


def get_equity_curve() -> list[dict]:
    """Build an equity curve from closed trades in chronological order."""
    session = get_session()
    try:
        query = session.query(LiveTrade).filter(
            LiveTrade.status == "closed",
            LiveTrade.pnl.isnot(None),
        )
        query = filter_query_after_cutoff(query, LiveTrade.closed_at)
        trades = query.order_by(LiveTrade.closed_at.asc()).all()

        curve = []
        running = 0.0
        peak = 0.0
        for t in trades:
            running += t.pnl
            if running > peak:
                peak = running
            dd = (peak - running) / peak if peak > 0 else 0
            curve.append({
                "time": t.closed_at.isoformat(),
                "pnl": round(running, 2),
                "drawdown": round(dd, 4),
                "symbol": t.symbol,
                "side": t.side,
                "trade_pnl": round(t.pnl, 2),
            })

        return curve
    finally:
        session.close()


def get_live_vs_backtest(strategy_type: str, symbol: str) -> dict:
    """Compare live performance to most recent backtest for a strategy/symbol."""
    session = get_session()
    try:
        # Live stats
        query = (
            session.query(LiveTrade, Signal)
            .join(Signal, LiveTrade.signal_id == Signal.id)
            .filter(
                LiveTrade.status == "closed",
                LiveTrade.pnl.isnot(None),
                Signal.strategy_type == strategy_type,
                Signal.symbol == symbol,
            )
        )
        query = filter_query_after_cutoff(query, LiveTrade.closed_at)
        live_trades = query.all()

        live_total = len(live_trades)
        live_wins = sum(1 for t, _ in live_trades if t.pnl > 0)
        live_pnl = sum(t.pnl for t, _ in live_trades)

        # Latest backtest for same strategy/symbol
        bt = (
            session.query(BacktestRun)
            .join(BacktestRun.__table__.c.trade_plan_id == None)  # noqa
            .filter(BacktestRun.status == "completed")
            .order_by(BacktestRun.created_at.desc())
            .first()
        )

        # Simpler: just find any backtest run for this strategy
        from models.trade_plan import TradePlan
        bt = (
            session.query(BacktestRun)
            .join(TradePlan, BacktestRun.trade_plan_id == TradePlan.id)
            .filter(
                TradePlan.strategy_type == strategy_type,
                TradePlan.symbol == symbol,
                BacktestRun.status == "completed",
            )
            .order_by(BacktestRun.created_at.desc())
            .first()
        )

        backtest = None
        if bt:
            backtest = {
                "total_trades": bt.total_trades,
                "win_rate": bt.win_rate,
                "profit_factor": bt.profit_factor,
                "max_drawdown": bt.max_drawdown,
                "sharpe_ratio": bt.sharpe_ratio,
                "final_balance": bt.final_balance,
                "initial_balance": bt.initial_balance,
            }

        return {
            "strategy": strategy_type,
            "symbol": symbol,
            "live": {
                "total_trades": live_total,
                "wins": live_wins,
                "win_rate": round(live_wins / live_total, 4) if live_total else 0,
                "total_pnl": round(live_pnl, 2),
            },
            "backtest": backtest,
        }
    finally:
        session.close()


def get_symbol_breakdown() -> list[dict]:
    """P&L breakdown by symbol."""
    session = get_session()
    try:
        query = session.query(LiveTrade).filter(
            LiveTrade.status == "closed",
            LiveTrade.pnl.isnot(None),
        )
        query = filter_query_after_cutoff(query, LiveTrade.closed_at)
        trades = query.all()

        by_symbol = defaultdict(lambda: {"trades": 0, "wins": 0, "pnl": 0.0})
        for t in trades:
            s = by_symbol[t.symbol]
            s["trades"] += 1
            s["pnl"] += t.pnl
            if t.pnl > 0:
                s["wins"] += 1

        return [
            {
                "symbol": sym,
                "trades": d["trades"],
                "wins": d["wins"],
                "win_rate": round(d["wins"] / d["trades"], 4) if d["trades"] else 0,
                "pnl": round(d["pnl"], 2),
            }
            for sym, d in sorted(by_symbol.items(), key=lambda x: x[1]["pnl"], reverse=True)
        ]
    finally:
        session.close()


def get_signal_accuracy() -> dict:
    """Analyze signal confirmation quality — did Claude's confidence predict outcomes?"""
    session = get_session()
    try:
        query = (
            session.query(Signal, LiveTrade)
            .join(LiveTrade, LiveTrade.signal_id == Signal.id)
            .filter(
                LiveTrade.status == "closed",
                LiveTrade.pnl.isnot(None),
                Signal.confidence.isnot(None),
            )
        )
        query = filter_query_after_cutoff(query, LiveTrade.closed_at)
        results = query.all()

        if not results:
            return {"total": 0}

        buckets = {"high": [], "medium": [], "low": []}
        for sig, trade in results:
            profitable = trade.pnl > 0
            if sig.confidence >= 0.7:
                buckets["high"].append(profitable)
            elif sig.confidence >= 0.4:
                buckets["medium"].append(profitable)
            else:
                buckets["low"].append(profitable)

        return {
            "total": len(results),
            "high_confidence": {
                "count": len(buckets["high"]),
                "accuracy": round(sum(buckets["high"]) / len(buckets["high"]), 4) if buckets["high"] else None,
            },
            "medium_confidence": {
                "count": len(buckets["medium"]),
                "accuracy": round(sum(buckets["medium"]) / len(buckets["medium"]), 4) if buckets["medium"] else None,
            },
            "low_confidence": {
                "count": len(buckets["low"]),
                "accuracy": round(sum(buckets["low"]) / len(buckets["low"]), 4) if buckets["low"] else None,
            },
        }
    finally:
        session.close()
