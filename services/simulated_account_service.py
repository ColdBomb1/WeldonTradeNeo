"""Derived simulated account metrics from paper trades."""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone

from config import load_config
from db import get_session
from models.account import AccountSnapshot
from models.signal import LiveTrade
from services.performance_archive import cutoff_iso, filter_query_after_cutoff


def get_simulated_account() -> dict:
    """Return account-like metrics built only from paper trades."""
    cfg = load_config()
    session = get_session()
    try:
        starting_balance = _starting_balance(session)
        query = session.query(LiveTrade).filter(LiveTrade.platform == "paper")
        query = filter_query_after_cutoff(query, LiveTrade.opened_at)
        trades = query.order_by(LiveTrade.opened_at.asc()).all()

        closed = [
            trade for trade in trades
            if trade.status == "closed" and trade.pnl is not None and trade.closed_at is not None
        ]
        open_trades = [trade for trade in trades if trade.status == "open"]
        closed.sort(key=lambda trade: trade.closed_at)

        realized_pnl = round(sum(float(trade.pnl or 0.0) for trade in closed), 2)
        unrealized_pnl = round(sum(float(trade.pnl or 0.0) for trade in open_trades), 2)
        balance = round(starting_balance + realized_pnl, 2)
        equity = round(balance + unrealized_pnl, 2)
        gross_profit = sum(float(trade.pnl or 0.0) for trade in closed if float(trade.pnl or 0.0) > 0.0)
        gross_loss = abs(sum(float(trade.pnl or 0.0) for trade in closed if float(trade.pnl or 0.0) < 0.0))
        wins = sum(1 for trade in closed if float(trade.pnl or 0.0) > 0.0)
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else (999.0 if gross_profit > 0 else 0.0)
        curve, max_drawdown_pct = _equity_curve(starting_balance, closed, unrealized_pnl)

        summary = {
            "name": "Simulated Account",
            "platform": "paper",
            "cutoff": cutoff_iso(),
            "starting_balance": round(starting_balance, 2),
            "balance": balance,
            "equity": equity,
            "realized_pnl": realized_pnl,
            "unrealized_pnl": unrealized_pnl,
            "return_pct": round((equity - starting_balance) / starting_balance * 100.0, 3)
            if starting_balance else 0.0,
            "open_positions": len(open_trades),
            "closed_trades": len(closed),
            "total_trades": len(trades),
            "winning_trades": wins,
            "losing_trades": len(closed) - wins,
            "win_rate": round(wins / len(closed), 4) if closed else 0.0,
            "profit_factor": round(min(profit_factor, 999.0), 4),
            "max_drawdown_pct": round(max_drawdown_pct, 3),
            "first_trade_at": trades[0].opened_at.isoformat() if trades else None,
            "last_trade_at": _last_trade_time(trades),
        }
        return {
            "summary": summary,
            "equity_curve": curve,
            "daily_pnl": _daily_pnl(closed),
            "symbols": _symbol_breakdown(closed),
            "open_trades": [_trade_row(trade) for trade in sorted(open_trades, key=lambda item: item.opened_at, reverse=True)],
            "recent_trades": [_trade_row(trade) for trade in sorted(trades, key=_trade_sort_time, reverse=True)[:100]],
        }
    finally:
        session.close()


def _starting_balance(session) -> float:
    cfg = load_config()
    if float(cfg.execution.account_start_balance or 0.0) > 0:
        return float(cfg.execution.account_start_balance)

    latest_snapshot = (
        session.query(AccountSnapshot)
        .order_by(AccountSnapshot.ts.desc())
        .first()
    )
    if latest_snapshot and latest_snapshot.balance:
        return float(latest_snapshot.balance)
    return float(cfg.training.initial_balance or 10000.0)


def _equity_curve(starting_balance: float, closed: list[LiveTrade], unrealized_pnl: float) -> tuple[list[dict], float]:
    curve = []
    running = starting_balance
    peak = starting_balance
    max_drawdown_pct = 0.0
    if closed:
        curve.append({
            "time": closed[0].opened_at.isoformat() if closed[0].opened_at else closed[0].closed_at.isoformat(),
            "equity": round(starting_balance, 2),
            "pnl": 0.0,
            "drawdown_pct": 0.0,
            "event": "start",
        })
    for trade in closed:
        running += float(trade.pnl or 0.0)
        peak = max(peak, running)
        drawdown_pct = (peak - running) / peak * 100.0 if peak > 0 else 0.0
        max_drawdown_pct = max(max_drawdown_pct, drawdown_pct)
        curve.append({
            "time": trade.closed_at.isoformat(),
            "equity": round(running, 2),
            "pnl": round(running - starting_balance, 2),
            "drawdown_pct": round(drawdown_pct, 3),
            "symbol": trade.symbol,
            "side": trade.side,
            "trade_pnl": round(float(trade.pnl or 0.0), 2),
            "event": "close",
        })
    if unrealized_pnl:
        current_equity = running + unrealized_pnl
        peak = max(peak, current_equity)
        drawdown_pct = (peak - current_equity) / peak * 100.0 if peak > 0 else 0.0
        max_drawdown_pct = max(max_drawdown_pct, drawdown_pct)
        curve.append({
            "time": datetime.now(timezone.utc).isoformat(),
            "equity": round(current_equity, 2),
            "pnl": round(current_equity - starting_balance, 2),
            "drawdown_pct": round(drawdown_pct, 3),
            "event": "open_unrealized",
        })
    return curve, max_drawdown_pct


def _daily_pnl(closed: list[LiveTrade]) -> list[dict]:
    daily = defaultdict(lambda: {"pnl": 0.0, "trades": 0, "wins": 0})
    for trade in closed:
        key = trade.closed_at.strftime("%Y-%m-%d")
        daily[key]["pnl"] += float(trade.pnl or 0.0)
        daily[key]["trades"] += 1
        if float(trade.pnl or 0.0) > 0.0:
            daily[key]["wins"] += 1
    return [
        {
            "date": key,
            "pnl": round(value["pnl"], 2),
            "trades": value["trades"],
            "wins": value["wins"],
            "win_rate": round(value["wins"] / value["trades"], 4) if value["trades"] else 0.0,
        }
        for key, value in sorted(daily.items())
    ]


def _symbol_breakdown(closed: list[LiveTrade]) -> list[dict]:
    symbols = defaultdict(lambda: {"pnl": 0.0, "trades": 0, "wins": 0})
    for trade in closed:
        row = symbols[str(trade.symbol).upper()]
        row["pnl"] += float(trade.pnl or 0.0)
        row["trades"] += 1
        if float(trade.pnl or 0.0) > 0.0:
            row["wins"] += 1
    return [
        {
            "symbol": symbol,
            "pnl": round(value["pnl"], 2),
            "trades": value["trades"],
            "wins": value["wins"],
            "win_rate": round(value["wins"] / value["trades"], 4) if value["trades"] else 0.0,
        }
        for symbol, value in sorted(symbols.items(), key=lambda item: item[1]["pnl"], reverse=True)
    ]


def _trade_row(trade: LiveTrade) -> dict:
    return {
        "id": trade.id,
        "signal_id": trade.signal_id,
        "symbol": trade.symbol,
        "side": trade.side,
        "entry_price": trade.entry_price,
        "current_price": trade.current_price,
        "stop_loss": trade.stop_loss,
        "take_profit": trade.take_profit,
        "volume": trade.volume,
        "status": trade.status,
        "pnl": round(float(trade.pnl or 0.0), 2),
        "exit_price": trade.exit_price,
        "exit_type": trade.exit_type,
        "opened_at": trade.opened_at.isoformat() if trade.opened_at else None,
        "closed_at": trade.closed_at.isoformat() if trade.closed_at else None,
    }


def _trade_sort_time(trade: LiveTrade) -> datetime:
    return trade.closed_at or trade.opened_at or datetime.min.replace(tzinfo=timezone.utc)


def _last_trade_time(trades: list[LiveTrade]) -> str | None:
    if not trades:
        return None
    last = max((_trade_sort_time(trade) for trade in trades), default=None)
    return last.isoformat() if last else None
