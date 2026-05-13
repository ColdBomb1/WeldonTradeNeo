"""Global risk management: exposure limits, correlation checks, drawdown circuit breakers.

Provides pre-trade validation beyond the basic checks in trade_manager.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

from config import load_config
from db import get_session
from models.signal import LiveTrade

logger = logging.getLogger(__name__)

# Static correlation matrix for major forex pairs.
# Positive = move together, Negative = move opposite.
# Approximate values based on typical market conditions.
_CORRELATION = {
    ("EURUSD", "GBPUSD"): 0.85,
    ("EURUSD", "AUDUSD"): 0.70,
    ("EURUSD", "NZDUSD"): 0.65,
    ("EURUSD", "USDCHF"): -0.90,
    ("EURUSD", "USDJPY"): -0.30,
    ("EURUSD", "USDCAD"): -0.55,
    ("GBPUSD", "AUDUSD"): 0.60,
    ("GBPUSD", "NZDUSD"): 0.55,
    ("GBPUSD", "USDCHF"): -0.80,
    ("GBPUSD", "USDJPY"): -0.20,
    ("GBPUSD", "USDCAD"): -0.45,
    ("AUDUSD", "NZDUSD"): 0.90,
    ("AUDUSD", "USDCAD"): -0.65,
    ("AUDUSD", "USDJPY"): 0.40,
    ("USDJPY", "USDCHF"): 0.60,
    ("USDJPY", "USDCAD"): 0.50,
    ("USDCHF", "AUDUSD"): -0.50,
    ("USDCHF", "NZDUSD"): -0.45,
    ("USDCHF", "USDCAD"): 0.35,
    ("NZDUSD", "USDJPY"): 0.35,
    ("NZDUSD", "USDCAD"): -0.60,
}


def _get_correlation(pair_a: str, pair_b: str) -> float | None:
    """Get correlation between two pairs (order-independent)."""
    a, b = pair_a.upper(), pair_b.upper()
    if a == b:
        return 1.0
    return _CORRELATION.get((a, b)) or _CORRELATION.get((b, a))


def check_correlation(symbol: str, side: str) -> tuple[bool, str]:
    """Check if opening a position creates excessive correlated risk.

    Returns (allowed, warning_message). Blocks if >2 highly correlated
    positions would exist in the same direction.
    """
    session = get_session()
    try:
        open_trades = session.query(LiveTrade).filter(
            LiveTrade.status == "open"
        ).all()

        # Deduplicate by symbol+side — the same trade mirrored across
        # multiple accounts counts as ONE position, not multiple.
        unique_positions = {}
        for t in open_trades:
            key = (t.symbol, t.side)
            if key not in unique_positions:
                unique_positions[key] = t

        correlated_same_dir = 0
        warnings = []

        for (sym, s), t in unique_positions.items():
            corr = _get_correlation(symbol, sym)
            if corr is None:
                continue

            same_direction = (s == side)
            if (same_direction and corr > 0.6) or (not same_direction and corr < -0.6):
                correlated_same_dir += 1
                warnings.append(f"{sym} {s} (corr={corr:+.2f})")

        if correlated_same_dir >= 2:
            return False, f"Too many correlated positions: {', '.join(warnings)}"

        if warnings:
            return True, f"Correlated with: {', '.join(warnings)}"

        return True, ""
    finally:
        session.close()


def check_global_exposure() -> dict:
    """Calculate total exposure across all open positions."""
    session = get_session()
    try:
        open_trades = session.query(LiveTrade).filter(
            LiveTrade.status == "open"
        ).all()

        total_volume = sum(t.volume for t in open_trades)
        total_unrealized = sum(t.pnl or 0 for t in open_trades)
        by_symbol = {}
        for t in open_trades:
            if t.symbol not in by_symbol:
                by_symbol[t.symbol] = {"volume": 0, "pnl": 0, "count": 0}
            by_symbol[t.symbol]["volume"] += t.volume
            by_symbol[t.symbol]["pnl"] += t.pnl or 0
            by_symbol[t.symbol]["count"] += 1

        return {
            "total_positions": len(open_trades),
            "total_volume": round(total_volume, 2),
            "total_unrealized_pnl": round(total_unrealized, 2),
            "by_symbol": {
                sym: {
                    "volume": round(d["volume"], 2),
                    "pnl": round(d["pnl"], 2),
                    "count": d["count"],
                }
                for sym, d in by_symbol.items()
            },
        }
    finally:
        session.close()


def check_drawdown_circuit_breaker() -> tuple[bool, str]:
    """Check weekly and monthly drawdown against configurable thresholds.

    Returns (allowed, reason). If False, no new trades should be opened.
    """
    cfg = load_config()
    max_daily = cfg.execution.max_daily_loss_pct
    session = get_session()

    try:
        from models.account import AccountDeal

        now = datetime.now(timezone.utc)

        # Use broker deal history (AccountDeal) as the source of truth for PnL
        today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        daily_deals = session.query(AccountDeal).filter(
            AccountDeal.closed_at >= today_start,
        ).all()
        daily_pnl = sum(d.profit or 0 for d in daily_deals)

        week_start = now - timedelta(days=now.weekday())
        week_start = week_start.replace(hour=0, minute=0, second=0, microsecond=0)
        weekly_deals = session.query(AccountDeal).filter(
            AccountDeal.closed_at >= week_start,
        ).all()
        weekly_pnl = sum(d.profit or 0 for d in weekly_deals)

        month_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        monthly_deals = session.query(AccountDeal).filter(
            AccountDeal.closed_at >= month_start,
        ).all()
        monthly_pnl = sum(d.profit or 0 for d in monthly_deals)

        weekly_limit = max_daily * 2
        monthly_limit = max_daily * 4

        # Use latest account snapshots from DB for balance reference
        from models.account import AccountSnapshot
        ref_balance = 0
        for acc in cfg.accounts:
            if not acc.enabled:
                continue
            latest_snap = session.query(AccountSnapshot).filter(
                AccountSnapshot.account_name == acc.name,
            ).order_by(AccountSnapshot.ts.desc()).first()
            if latest_snap and latest_snap.balance > 0:
                ref_balance += latest_snap.balance
        if ref_balance <= 0:
            ref_balance = 100000.0  # fallback

        daily_loss_pct = abs(min(0, daily_pnl)) / ref_balance * 100
        weekly_loss_pct = abs(min(0, weekly_pnl)) / ref_balance * 100
        monthly_loss_pct = abs(min(0, monthly_pnl)) / ref_balance * 100

        if daily_loss_pct >= max_daily:
            return False, f"Daily loss circuit breaker: {daily_loss_pct:.1f}% (limit {max_daily}%)"

        if weekly_loss_pct >= weekly_limit:
            return False, f"Weekly loss circuit breaker: {weekly_loss_pct:.1f}% (limit {weekly_limit}%)"

        if monthly_loss_pct >= monthly_limit:
            return False, f"Monthly loss circuit breaker: {monthly_loss_pct:.1f}% (limit {monthly_limit}%)"

        return True, ""
    finally:
        session.close()


def get_risk_summary() -> dict:
    """Full risk dashboard data."""
    exposure = check_global_exposure()
    cb_ok, cb_reason = check_drawdown_circuit_breaker()

    # Check correlation for each open position
    session = get_session()
    try:
        open_trades = session.query(LiveTrade).filter(
            LiveTrade.status == "open"
        ).all()
    finally:
        session.close()

    correlation_warnings = []
    for t in open_trades:
        _, warn = check_correlation(t.symbol, t.side)
        if warn:
            correlation_warnings.append(f"{t.symbol} {t.side}: {warn}")

    return {
        "exposure": exposure,
        "circuit_breaker": {"ok": cb_ok, "reason": cb_reason},
        "correlation_warnings": correlation_warnings,
    }
