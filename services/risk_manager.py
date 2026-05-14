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

_LOT_UNITS = {"micro": 1000, "mini": 10000, "standard": 100000}
_BROKER_LOT_UNITS = 100000

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


def estimate_trade_risk_amount(trade: LiveTrade) -> float:
    """Estimate broker-account dollars at risk if the stop loss is hit."""
    try:
        if not trade.entry_price or not trade.stop_loss or not trade.volume:
            return 0.0
        distance = abs(float(trade.entry_price) - float(trade.stop_loss))
        if distance <= 0:
            return 0.0
        lot_units = _BROKER_LOT_UNITS
        if getattr(trade, "platform", "") == "paper":
            cfg = load_config()
            lot_units = _LOT_UNITS.get(cfg.execution.default_lot_type, _BROKER_LOT_UNITS)
        risk = distance * lot_units * float(trade.volume)
        if "JPY" in (trade.symbol or "").upper() and trade.entry_price:
            risk = risk / float(trade.entry_price)
        return max(0.0, risk)
    except Exception:
        return 0.0


def _currency_legs(symbol: str, side: str) -> tuple[tuple[str, int], tuple[str, int]]:
    cleaned = (symbol or "").upper().replace("/", "").replace("=X", "")
    base, quote = cleaned[:3], cleaned[3:6]
    if side == "buy":
        return (base, 1), (quote, -1)
    return (base, -1), (quote, 1)


def check_currency_exposure(symbol: str, side: str, candidate_risk_pct: float = 0.0) -> tuple[bool, str]:
    """Limit stacked directional exposure to the same currency."""
    cfg = load_config()
    ex = cfg.execution
    risk_limit = float(getattr(ex, "max_currency_open_risk_pct", 0.0) or 0.0)
    position_limit = int(getattr(ex, "max_same_currency_positions", 0) or 0)
    if risk_limit <= 0 and position_limit <= 0:
        return True, ""

    session = get_session()
    try:
        state = _account_risk_state(session=session, cfg=cfg)
        reference_balance = float(state.get("reference_balance") or 100000.0)
        open_trades = session.query(LiveTrade).filter(LiveTrade.status == "open").all()

        risk_by_currency: dict[str, float] = {}
        positions_by_currency: dict[tuple[str, int], int] = {}
        for trade in open_trades:
            risk_pct = estimate_trade_risk_amount(trade) / reference_balance * 100.0 if reference_balance > 0 else 0.0
            for currency, direction in _currency_legs(trade.symbol, trade.side):
                if not currency:
                    continue
                risk_by_currency[currency] = risk_by_currency.get(currency, 0.0) + direction * risk_pct
                positions_by_currency[(currency, direction)] = positions_by_currency.get((currency, direction), 0) + 1

        for currency, direction in _currency_legs(symbol, side):
            if not currency:
                continue
            projected_risk = risk_by_currency.get(currency, 0.0) + direction * max(0.0, float(candidate_risk_pct or 0.0))
            projected_positions = positions_by_currency.get((currency, direction), 0) + 1
            if risk_limit > 0 and abs(projected_risk) > risk_limit:
                return False, (
                    f"{currency} directional risk {abs(projected_risk):.2f}% exceeds "
                    f"currency cap {risk_limit:.2f}%"
                )
            if position_limit > 0 and projected_positions > position_limit:
                label = "long" if direction > 0 else "short"
                return False, (
                    f"{currency} {label} exposure would have {projected_positions} positions "
                    f"(limit {position_limit})"
                )

        return True, ""
    finally:
        session.close()


def _latest_snapshot_for_account(session, account_name: str):
    from models.account import AccountSnapshot

    return (
        session.query(AccountSnapshot)
        .filter(AccountSnapshot.account_name == account_name)
        .order_by(AccountSnapshot.ts.desc())
        .first()
    )


def _account_risk_state(session=None, cfg=None) -> dict:
    """Return current account drawdown and loss-limit state."""
    owns_session = session is None
    session = session or get_session()
    cfg = cfg or load_config()
    ex = cfg.execution
    try:
        from models.account import AccountDeal, AccountSnapshot

        enabled_accounts = [a for a in cfg.accounts if a.enabled]
        names = [a.name for a in enabled_accounts]

        latest_snaps = []
        for name in names:
            snap = _latest_snapshot_for_account(session, name)
            if snap:
                latest_snaps.append(snap)

        current_balance = sum(float(s.balance or 0) for s in latest_snaps)
        current_equity = sum(float(s.equity or s.balance or 0) for s in latest_snaps)

        high_water = 0.0
        if names:
            for name in names:
                rows = (
                    session.query(AccountSnapshot)
                    .filter(AccountSnapshot.account_name == name)
                    .all()
                )
                if rows:
                    high_water += max(float(r.equity or r.balance or 0) for r in rows)

        configured_start = float(ex.account_start_balance or 0)
        reference_balance = configured_start or high_water or current_balance or 100000.0
        if current_equity <= 0:
            current_equity = current_balance or reference_balance
        if current_balance <= 0:
            current_balance = current_equity
        if high_water <= 0:
            high_water = max(reference_balance, current_equity, current_balance)
        else:
            high_water = max(high_water, reference_balance)

        now = datetime.now(timezone.utc)
        today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        week_start = (now - timedelta(days=now.weekday())).replace(hour=0, minute=0, second=0, microsecond=0)
        month_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)

        def _deals_since(start):
            query = session.query(AccountDeal).filter(AccountDeal.closed_at >= start)
            if names:
                query = query.filter(AccountDeal.account_name.in_(names))
            return query.all()

        daily_pnl = sum(d.profit or 0 for d in _deals_since(today_start))
        weekly_pnl = sum(d.profit or 0 for d in _deals_since(week_start))
        monthly_pnl = sum(d.profit or 0 for d in _deals_since(month_start))

        open_trades = session.query(LiveTrade).filter(LiveTrade.status == "open").all()
        unrealized_pnl = sum(t.pnl or 0 for t in open_trades)
        open_risk_amount = sum(estimate_trade_risk_amount(t) for t in open_trades)

        daily_loss = abs(min(0.0, daily_pnl + min(0.0, unrealized_pnl)))
        weekly_loss = abs(min(0.0, weekly_pnl + min(0.0, unrealized_pnl)))
        monthly_loss = abs(min(0.0, monthly_pnl + min(0.0, unrealized_pnl)))

        relative_drawdown_pct = (
            max(0.0, reference_balance - current_equity) / reference_balance * 100
            if reference_balance > 0 else 0.0
        )
        trailing_drawdown_pct = (
            max(0.0, high_water - current_equity) / high_water * 100
            if high_water > 0 else 0.0
        )

        return {
            "account_names": names,
            "current_balance": round(current_balance, 2),
            "current_equity": round(current_equity, 2),
            "reference_balance": round(reference_balance, 2),
            "high_water_equity": round(high_water, 2),
            "relative_drawdown_pct": round(relative_drawdown_pct, 3),
            "trailing_drawdown_pct": round(trailing_drawdown_pct, 3),
            "daily_pnl": round(daily_pnl, 2),
            "weekly_pnl": round(weekly_pnl, 2),
            "monthly_pnl": round(monthly_pnl, 2),
            "unrealized_pnl": round(unrealized_pnl, 2),
            "daily_loss_pct": round(daily_loss / reference_balance * 100, 3) if reference_balance > 0 else 0.0,
            "weekly_loss_pct": round(weekly_loss / reference_balance * 100, 3) if reference_balance > 0 else 0.0,
            "monthly_loss_pct": round(monthly_loss / reference_balance * 100, 3) if reference_balance > 0 else 0.0,
            "open_risk_amount": round(open_risk_amount, 2),
            "open_risk_pct": round(open_risk_amount / reference_balance * 100, 3) if reference_balance > 0 else 0.0,
            "open_positions": len(open_trades),
        }
    finally:
        if owns_session:
            session.close()


def get_effective_risk_per_trade_pct(cfg=None) -> float:
    """Scale per-trade risk down as drawdown approaches the hard stop."""
    cfg = cfg or load_config()
    ex = cfg.execution
    base = max(0.0, float(ex.risk_per_trade_pct or 0))
    minimum = max(0.0, min(base, float(ex.min_risk_per_trade_pct or 0)))
    try:
        state = _account_risk_state(cfg=cfg)
    except Exception as exc:
        logger.warning("Could not compute drawdown-adjusted risk: %s", exc)
        return base

    dd = max(state.get("relative_drawdown_pct", 0.0), state.get("trailing_drawdown_pct", 0.0))
    reduce_at = max(0.0, float(ex.risk_reduction_drawdown_pct or 0))
    hard_stop = max(0.0, float(ex.max_relative_drawdown_pct or ex.max_total_loss_pct or 0) -
                    float(ex.drawdown_hard_stop_buffer_pct or 0))
    if hard_stop <= 0 or dd < reduce_at:
        return base
    if dd >= hard_stop:
        if bool(getattr(ex, "allow_drawdown_override", False)):
            return minimum
        return 0.0
    span = max(hard_stop - reduce_at, 0.01)
    scale = max(0.0, min(1.0, (hard_stop - dd) / span))
    return round(max(minimum, base * scale), 4)


def check_aggregate_open_risk(candidate_risk_pct: float = 0.0) -> tuple[bool, str]:
    """Block new entries when open stop-loss risk plus candidate risk is too high."""
    cfg = load_config()
    limit = float(cfg.execution.max_aggregate_open_risk_pct or 0)
    if limit <= 0:
        return True, ""
    state = _account_risk_state(cfg=cfg)
    total = float(state.get("open_risk_pct", 0.0)) + max(0.0, float(candidate_risk_pct or 0))
    if total > limit:
        return False, (
            f"Aggregate open risk {total:.2f}% exceeds limit {limit:.2f}% "
            f"(open {state.get('open_risk_pct', 0):.2f}% + candidate {candidate_risk_pct:.2f}%)"
        )
    return True, ""


def check_drawdown_circuit_breaker() -> tuple[bool, str]:
    """Check weekly and monthly drawdown against configurable thresholds.

    Returns (allowed, reason). If False, no new trades should be opened.
    """
    cfg = load_config()
    ex = cfg.execution
    try:
        state = _account_risk_state(cfg=cfg)
        hard_stop = max(
            0.0,
            float(ex.max_relative_drawdown_pct or ex.max_total_loss_pct or 0)
            - float(ex.drawdown_hard_stop_buffer_pct or 0),
        )
        allow_override = bool(getattr(ex, "allow_drawdown_override", False))
        if hard_stop > 0 and state["relative_drawdown_pct"] >= hard_stop and not allow_override:
            return False, (
                f"Relative drawdown circuit breaker: {state['relative_drawdown_pct']:.2f}% "
                f"(hard stop {hard_stop:.2f}%)"
            )
        if hard_stop > 0 and state["trailing_drawdown_pct"] >= hard_stop and not allow_override:
            return False, (
                f"Trailing drawdown circuit breaker: {state['trailing_drawdown_pct']:.2f}% "
                f"(hard stop {hard_stop:.2f}%)"
            )
        if state["daily_loss_pct"] >= ex.max_daily_loss_pct:
            return False, (
                f"Daily loss circuit breaker: {state['daily_loss_pct']:.2f}% "
                f"(limit {ex.max_daily_loss_pct}%)"
            )
        if state["monthly_loss_pct"] >= ex.max_total_loss_pct:
            return False, (
                f"Monthly loss circuit breaker: {state['monthly_loss_pct']:.2f}% "
                f"(limit {ex.max_total_loss_pct}%)"
            )
    except Exception as exc:
        logger.warning("Enhanced drawdown check failed, falling back to deal history: %s", exc)

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
    cfg = load_config()
    exposure = check_global_exposure()
    cb_ok, cb_reason = check_drawdown_circuit_breaker()
    account_state = _account_risk_state(cfg=cfg)
    effective_risk_pct = get_effective_risk_per_trade_pct(cfg)

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
        "account": account_state,
        "effective_risk_per_trade_pct": effective_risk_pct,
        "limits": {
            "base_risk_per_trade_pct": cfg.execution.risk_per_trade_pct,
            "min_risk_per_trade_pct": cfg.execution.min_risk_per_trade_pct,
            "max_daily_loss_pct": cfg.execution.max_daily_loss_pct,
            "max_total_loss_pct": cfg.execution.max_total_loss_pct,
            "max_relative_drawdown_pct": cfg.execution.max_relative_drawdown_pct,
            "drawdown_hard_stop_buffer_pct": cfg.execution.drawdown_hard_stop_buffer_pct,
            "allow_drawdown_override": cfg.execution.allow_drawdown_override,
            "max_aggregate_open_risk_pct": cfg.execution.max_aggregate_open_risk_pct,
            "max_currency_open_risk_pct": cfg.execution.max_currency_open_risk_pct,
            "max_same_currency_positions": cfg.execution.max_same_currency_positions,
        },
        "circuit_breaker": {"ok": cb_ok, "reason": cb_reason},
        "correlation_warnings": correlation_warnings,
    }
