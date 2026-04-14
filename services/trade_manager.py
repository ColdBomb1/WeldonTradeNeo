"""Trade lifecycle manager: signal execution, position monitoring, risk checks.

Orchestrates the flow from confirmed signal to executed trade and monitors
open positions for SL/TP exits.
"""

from __future__ import annotations

import asyncio
import logging
import random
import threading
from datetime import datetime, timedelta, timezone

from config import load_config, AccountConfig
from db import get_session
from models.signal import Signal, LiveTrade
from services import mt5_trade_service, dxtrade_service, indicator_service, risk_manager

logger = logging.getLogger(__name__)

LOT_UNITS = {"micro": 1000, "mini": 10000, "standard": 100000}

_paper_ticket_counter = 100000


def _next_paper_ticket() -> int:
    global _paper_ticket_counter
    _paper_ticket_counter += 1
    return _paper_ticket_counter


def _get_trading_account() -> AccountConfig | None:
    """Find the first enabled trading account (any type)."""
    cfg = load_config()
    for acc in cfg.accounts:
        if acc.enabled:
            return acc
    return None


def _get_account_by_type(account_type: str) -> AccountConfig | None:
    """Find first enabled account of a specific type."""
    cfg = load_config()
    for acc in cfg.accounts:
        if acc.account_type == account_type and acc.enabled:
            return acc
    return None


def _get_trade_service(account_type: str):
    """Get the service module for an account type."""
    if account_type == "mt5":
        return mt5_trade_service
    elif account_type == "dxtrade":
        return dxtrade_service
    return None


def _pip_value(symbol: str) -> float:
    """Return pip value for a symbol (0.01 for JPY pairs, 0.0001 otherwise)."""
    return 0.01 if "JPY" in symbol.upper() else 0.0001


def compute_position_size(
    balance: float,
    risk_pct: float,
    sl_distance: float,
    symbol: str,
    lot_type: str = "mini",
) -> float:
    """Compute lot volume using ATR-based position sizing.

    Same formula as backtest_engine.py:
    volume = risk_amount / (sl_pips * pip_value_per_lot)
    """
    pip_val = _pip_value(symbol)
    lot_units = LOT_UNITS.get(lot_type, 10000)

    sl_pips = sl_distance / pip_val
    if sl_pips <= 0:
        return 0.01
    # Enforce minimum 10 pips SL to prevent oversized positions
    sl_pips = max(sl_pips, 10)

    risk_amount = balance * (risk_pct / 100.0)
    pip_value_per_lot = pip_val * lot_units
    volume = risk_amount / (sl_pips * pip_value_per_lot)
    # JPY pairs: pip_value_per_lot is ~$1000 (vs $10 for EUR/GBP) because
    # pip=0.01 vs 0.0001. Scale up by the JPY rate (~159) to normalize risk.
    # This makes a 25-pip SL on USDJPY risk ~$450 at 2.86 lots, matching EUR/GBP.
    if "JPY" in symbol.upper():
        jpy_rate = 159.0  # approximate — doesn't need to be exact
        volume *= jpy_rate / 2
    # Per-symbol lot caps — normalized to ~$450 max risk per trade
    max_lots = {"EURUSD": 3.0, "GBPUSD": 2.25, "USDJPY": 2.86}
    cap = max_lots.get(symbol.upper(), 3.0)
    volume = min(volume, cap)
    return max(round(volume, 2), 0.01)


def _check_blackouts() -> tuple[bool, str]:
    """Check if current time falls within a trading blackout window."""
    cfg = load_config()
    now = datetime.now(timezone.utc)

    for blackout in cfg.execution.blackouts:
        try:
            if blackout.recurring:
                # HH:MM format — check daily
                start_parts = blackout.start.split(":")
                end_parts = blackout.end.split(":")
                start_h, start_m = int(start_parts[0]), int(start_parts[1])
                end_h, end_m = int(end_parts[0]), int(end_parts[1])
                current_minutes = now.hour * 60 + now.minute
                start_minutes = start_h * 60 + start_m
                end_minutes = end_h * 60 + end_m
                if start_minutes <= current_minutes < end_minutes:
                    return False, f"Trading blackout: {blackout.reason or f'{blackout.start}-{blackout.end} UTC'}"
            else:
                # ISO datetime range
                start_dt = datetime.fromisoformat(blackout.start)
                end_dt = datetime.fromisoformat(blackout.end)
                if start_dt.tzinfo is None:
                    start_dt = start_dt.replace(tzinfo=timezone.utc)
                if end_dt.tzinfo is None:
                    end_dt = end_dt.replace(tzinfo=timezone.utc)
                if start_dt <= now <= end_dt:
                    return False, f"Trading blackout: {blackout.reason or f'{blackout.start} to {blackout.end}'}"
        except (ValueError, IndexError):
            continue

    return True, ""


def check_risk_limits(symbol: str, side: str) -> tuple[bool, str]:
    """Check whether a new trade is allowed given current risk limits.

    Returns (allowed, reason).
    """
    cfg = load_config()
    ex = cfg.execution

    # Check paused
    if ex.paused:
        return False, "Trading is paused"

    # Check blackouts
    bo_ok, bo_reason = _check_blackouts()
    if not bo_ok:
        return False, bo_reason

    session = get_session()
    try:
        # Check max open positions
        open_count = session.query(LiveTrade).filter(
            LiveTrade.status == "open"
        ).count()
        if open_count >= ex.max_open_positions:
            return False, f"Max open positions ({ex.max_open_positions}) reached"

        # Check max 1 position per currency pair (prop firm rule)
        existing_on_pair = session.query(LiveTrade).filter(
            LiveTrade.symbol == symbol,
            LiveTrade.status == "open",
        ).first()
        if existing_on_pair:
            return False, f"Already have an open position on {symbol} (1 per pair limit)"

        # Get account balance for loss calculations
        account = _get_trading_account()
        balance = 10000.0
        if account and mt5_trade_service.is_available():
            if mt5_trade_service.initialize(account):
                snap = mt5_trade_service.get_account_snapshot(account)
                mt5_trade_service.shutdown()
                if snap:
                    balance = snap["balance"]

        # Check daily loss limit (prop firm: 4%)
        today_start = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
        closed_today = session.query(LiveTrade).filter(
            LiveTrade.status == "closed",
            LiveTrade.closed_at >= today_start,
        ).all()
        daily_loss = sum(t.pnl for t in closed_today if t.pnl and t.pnl < 0)
        # Include unrealized losses from open positions
        open_trades = session.query(LiveTrade).filter(LiveTrade.status == "open").all()
        unrealized_loss = sum(t.pnl for t in open_trades if t.pnl and t.pnl < 0)
        total_daily_exposure = abs(daily_loss) + abs(unrealized_loss)
        daily_loss_pct = total_daily_exposure / balance * 100 if balance > 0 else 0
        if daily_loss_pct >= ex.max_daily_loss_pct:
            return False, f"Daily loss limit ({ex.max_daily_loss_pct}%) reached ({daily_loss_pct:.1f}% including unrealized)"

        # Check total account drawdown (prop firm: 8%)
        all_closed = session.query(LiveTrade).filter(
            LiveTrade.status == "closed",
            LiveTrade.pnl.isnot(None),
        ).all()
        total_pnl = sum(t.pnl for t in all_closed) + sum(t.pnl or 0 for t in open_trades)
        if total_pnl < 0:
            total_loss_pct = abs(total_pnl) / balance * 100
            if total_loss_pct >= ex.max_total_loss_pct:
                return False, f"Total loss limit ({ex.max_total_loss_pct}%) reached ({total_loss_pct:.1f}%) — STOP TRADING"

        # Circuit breaker (weekly/monthly)
        cb_ok, cb_reason = risk_manager.check_drawdown_circuit_breaker()
        if not cb_ok:
            return False, cb_reason

        # Correlation check
        corr_ok, corr_reason = risk_manager.check_correlation(symbol, side)
        if not corr_ok:
            return False, corr_reason

        return True, "ok"
    finally:
        session.close()


def _get_account_balance(account: AccountConfig) -> float:
    """Fetch live balance for an account. Returns fallback on failure."""
    service = _get_trade_service(account.account_type)
    if not service:
        return 10000.0
    try:
        if not service.initialize(account):
            return 10000.0
        # DxTrade WS needs time to deliver data — retry briefly
        snap = service.get_account_snapshot(account)
        if snap is None and account.account_type == "dxtrade":
            import time as _t
            for _ in range(6):
                _t.sleep(1)
                snap = service.get_account_snapshot(account)
                if snap and snap.get("balance", 0) > 0:
                    break
        if snap and snap.get("balance", 0) > 0:
            return snap["balance"]
    except Exception:
        pass
    return 10000.0


# Normal spread ranges per pair (in pips). Reject if spread exceeds 2x normal.
_NORMAL_SPREAD = {"EURUSD": 1.5, "GBPUSD": 2.0, "USDJPY": 2.0}
_MAX_SPREAD_MULTIPLIER = 2.0


def _check_spread(symbol: str, account: AccountConfig) -> tuple[bool, float]:
    """Check if current spread is acceptable. Returns (ok, spread_pips)."""
    pip_val = 0.01 if "JPY" in symbol.upper() else 0.0001
    normal = _NORMAL_SPREAD.get(symbol, 2.0)
    max_spread = normal * _MAX_SPREAD_MULTIPLIER

    if account.account_type == "mt5":
        try:
            import MetaTrader5 as mt5
            # Use the suffix from config
            from config import load_config
            cfg = load_config()
            resolved = f"{symbol}{cfg.mt5.symbol_suffix}" if cfg.mt5.symbol_suffix else symbol
            tick = mt5.symbol_info_tick(resolved)
            if tick and tick.ask > 0 and tick.bid > 0:
                spread_pips = (tick.ask - tick.bid) / pip_val
                return spread_pips <= max_spread, spread_pips
        except Exception:
            pass
    # DxTrade: no easy way to get live spread, assume ok
    return True, 0


def _execute_on_account(
    account: AccountConfig, sig: Signal, ex, sl: float, tp: float,
    sl_distance: float, session,
) -> LiveTrade:
    """Place an order on a single broker account. Returns a LiveTrade record."""
    now = datetime.now(timezone.utc)

    # Spread check (MT5 only — DxTrade doesn't expose live spread)
    spread_ok, spread_pips = _check_spread(sig.symbol, account)
    if not spread_ok:
        trade = LiveTrade(
            signal_id=sig.id, symbol=sig.symbol, side=sig.side,
            entry_price=sig.price, stop_loss=sl, take_profit=tp, volume=0,
            platform=account.account_type, status="error",
            error_message=f"Spread too wide: {spread_pips:.1f} pips", opened_at=now,
        )
        session.add(trade)
        logger.warning("Spread too wide for %s on %s: %.1f pips", sig.symbol, account.name, spread_pips)
        return trade

    balance = _get_account_balance(account)
    volume = compute_position_size(
        balance, ex.risk_per_trade_pct, sl_distance, sig.symbol, ex.default_lot_type
    )
    service = _get_trade_service(account.account_type)
    if not service:
        trade = LiveTrade(
            signal_id=sig.id, symbol=sig.symbol, side=sig.side,
            entry_price=sig.price, stop_loss=sl, take_profit=tp, volume=volume,
            platform=account.account_type, status="error",
            error_message=f"No service for {account.account_type}", opened_at=now,
        )
        session.add(trade)
        return trade

    result = service.place_market_order(account, sig.symbol, sig.side, volume, sl, tp)

    if "error" in result:
        trade = LiveTrade(
            signal_id=sig.id, symbol=sig.symbol, side=sig.side,
            entry_price=sig.price, stop_loss=sl, take_profit=tp, volume=volume,
            platform=account.account_type, status="error",
            error_message=result["error"], opened_at=now, raw=result,
        )
    else:
        fill_price = result.get("price") or sig.price  # MT5 returns price=0 for market orders
        # Recalculate SL/TP from actual fill price (signal price may differ)
        if fill_price and fill_price != sig.price:
            sl_dist = abs(sig.price - sl)
            tp_dist = abs(sig.price - tp)
            if sig.side == "buy":
                sl = fill_price - sl_dist
                tp = fill_price + tp_dist
            else:
                sl = fill_price + sl_dist
                tp = fill_price - tp_dist
        trade = LiveTrade(
            signal_id=sig.id, symbol=sig.symbol, side=sig.side,
            entry_price=fill_price, stop_loss=sl, take_profit=tp,
            volume=result.get("volume", volume),
            platform_ticket=result.get("ticket"), platform=account.account_type,
            status="open", opened_at=now, raw=result,
        )
    session.add(trade)
    return trade


def execute_signal(signal_id: int) -> dict:
    """Execute a confirmed signal on ALL enabled accounts.

    In paper mode, simulates at current price. In live mode, copies the trade
    to every enabled broker account with per-account position sizing.
    """
    cfg = load_config()
    ex = cfg.execution
    session = get_session()

    try:
        sig = session.query(Signal).filter(Signal.id == signal_id).first()
        if sig is None:
            return {"error": f"Signal {signal_id} not found"}
        if sig.status != "confirmed":
            return {"error": f"Signal {signal_id} is {sig.status}, not confirmed"}

        # Risk checks
        allowed, reason = check_risk_limits(sig.symbol, sig.side)
        if not allowed:
            sig.status = "rejected"
            sig.claude_analysis = {**(sig.claude_analysis or {}), "reject_reason": reason}
            sig.resolved_at = datetime.now(timezone.utc)
            session.commit()
            return {"error": f"Risk check failed: {reason}"}

        # Compute SL/TP (shared across all accounts)
        sl_distance = abs(sig.price - sig.stop_loss) if sig.stop_loss else sig.price * 0.005
        sl = sig.stop_loss or (sig.price - sl_distance if sig.side == "buy" else sig.price + sl_distance)
        tp_distance = sl_distance * (ex.tp_atr_multiplier / ex.sl_atr_multiplier) if sig.stop_loss else sig.price * 0.01
        tp = sig.take_profit or (sig.price + tp_distance if sig.side == "buy" else sig.price - tp_distance)

        # Cap SL/TP distances to realistic 15min intraday ranges
        pip_val = 0.01 if "JPY" in sig.symbol.upper() else 0.0001
        max_sl_pips = {"EURUSD": 15, "GBPUSD": 20, "USDJPY": 25}
        max_tp_pips = {"EURUSD": 25, "GBPUSD": 35, "USDJPY": 40}
        sl_cap = max_sl_pips.get(sig.symbol, 20) * pip_val
        tp_cap = max_tp_pips.get(sig.symbol, 30) * pip_val

        if abs(sig.price - sl) > sl_cap:
            sl = sig.price - sl_cap if sig.side == "buy" else sig.price + sl_cap
            logger.info("SL capped to %d pips for %s", sl_cap / pip_val, sig.symbol)
        if abs(sig.price - tp) > tp_cap:
            tp = sig.price + tp_cap if sig.side == "buy" else sig.price - tp_cap
            logger.info("TP capped to %d pips for %s", tp_cap / pip_val, sig.symbol)

        now = datetime.now(timezone.utc)
        results = []

        if ex.mode == "live":
            accounts = [a for a in cfg.accounts if a.enabled]
            if not accounts:
                return {"error": "No enabled accounts"}

            for account in accounts:
                try:
                    trade = _execute_on_account(account, sig, ex, sl, tp, sl_distance, session)
                    session.flush()  # assign trade.id before logging
                    results.append(trade)
                    logger.info(
                        "Executed signal %d → trade %d on %s (%s %s %.2f lots @ %.5f) [%s]",
                        sig.id, trade.id, account.name, sig.side, sig.symbol,
                        trade.volume, trade.entry_price, trade.status,
                    )
                except Exception as exc:
                    logger.error("Signal %d failed on %s: %s", sig.id, account.name, exc)
                    try:
                        error_trade = LiveTrade(
                            signal_id=sig.id, symbol=sig.symbol, side=sig.side,
                            entry_price=sig.price, stop_loss=sl, take_profit=tp,
                            volume=0, platform=account.account_type, status="error",
                            error_message=str(exc)[:200], opened_at=now,
                        )
                        session.add(error_trade)
                        results.append(error_trade)
                    except Exception:
                        pass
        else:
            # Paper trading — single simulated trade
            balance = 10000.0
            account = _get_trading_account()
            if account:
                balance = _get_account_balance(account)
            volume = compute_position_size(
                balance, ex.risk_per_trade_pct, sl_distance, sig.symbol, ex.default_lot_type
            )
            trade = LiveTrade(
                signal_id=sig.id, symbol=sig.symbol, side=sig.side,
                entry_price=sig.price, current_price=sig.price,
                stop_loss=sl, take_profit=tp, volume=volume,
                platform_ticket=_next_paper_ticket(), platform="paper",
                status="open", opened_at=now,
            )
            session.add(trade)
            results.append(trade)
            logger.info(
                "Executed signal %d → trade %d (paper %s %s %.2f lots @ %.5f)",
                sig.id, trade.id, sig.side, sig.symbol, volume, trade.entry_price,
            )

        sig.status = "executed"
        sig.resolved_at = now
        session.commit()

        trade_ids = [t.id for t in results]
        errors = [t.error_message for t in results if t.status == "error"]
        return {"trade_ids": trade_ids, "errors": errors}
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def close_trade(trade_id: int, reason: str = "manual") -> dict:
    """Programmatically close an open trade."""
    cfg = load_config()
    session = get_session()
    try:
        trade = session.query(LiveTrade).filter(LiveTrade.id == trade_id).first()
        if trade is None:
            return {"error": f"Trade {trade_id} not found"}
        if trade.status != "open":
            return {"error": f"Trade {trade_id} is {trade.status}"}

        now = datetime.now(timezone.utc)
        close_price = trade.current_price or trade.entry_price

        if trade.platform in ("mt5", "dxtrade") and trade.platform_ticket:
            account = _get_account_by_type(trade.platform)
            service = _get_trade_service(trade.platform)
            if account and service:
                result = service.close_position(account, trade.platform_ticket)
                if "error" in result:
                    return {"error": result["error"]}
                close_price = result.get("close_price", close_price)

        # Compute P&L
        pip_val = _pip_value(trade.symbol)
        lot_units = LOT_UNITS.get(cfg.execution.default_lot_type, 10000)
        if trade.side == "buy":
            pnl = (close_price - trade.entry_price) * trade.volume * lot_units
        else:
            pnl = (trade.entry_price - close_price) * trade.volume * lot_units

        trade.status = "closed"
        trade.exit_price = close_price
        trade.exit_type = reason
        trade.pnl = pnl
        trade.closed_at = now
        session.commit()
        logger.info("Closed trade %d: %s P&L=%.2f", trade_id, reason, pnl)
        return {"trade_id": trade_id, "pnl": pnl}
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def _sync_positions_once():
    """Sync open LiveTrade records with broker/paper positions."""
    cfg = load_config()
    session = get_session()
    try:
        open_trades = session.query(LiveTrade).filter(LiveTrade.status == "open").all()
        if not open_trades:
            return

        if cfg.execution.mode == "live":
            # Sync each broker platform separately
            broker_positions = {}  # {platform: {ticket: position_dict}}

            for acc in cfg.accounts:
                if not acc.enabled:
                    continue
                service = _get_trade_service(acc.account_type)
                if not service or not service.is_available():
                    continue
                if not service.initialize(acc):
                    continue
                try:
                    positions = service.get_active_trades()
                    broker_positions[acc.account_type] = {
                        str(p["ticket"]): p for p in positions
                    }
                finally:
                    service.shutdown()

            for trade in open_trades:
                if trade.platform not in broker_positions:
                    continue
                if not trade.platform_ticket:
                    continue
                platform_pos = broker_positions[trade.platform]
                pos = platform_pos.get(str(trade.platform_ticket))

                # Safety: only close DxTrade trades if the WS is alive and has fresh data.
                # An empty positions list with a dead WS means stale data, not real closure.
                if not pos and trade.platform == "dxtrade" and len(platform_pos) == 0:
                    ws_alive = dxtrade_service._ws_obj is not None
                    has_metrics = "ACCOUNT_METRICS" in dxtrade_service._ws_cache
                    if not ws_alive or not has_metrics:
                        logger.debug("Skipping close check for trade %d — DxTrade WS not ready", trade.id)
                        continue
                if pos:
                    # Position still open — use broker-reported profit
                    trade.current_price = pos["current_price"]
                    trade.pnl = pos.get("profit", 0)
                    # Fix entry price if it was stored as 0
                    if not trade.entry_price and pos.get("price"):
                        trade.entry_price = pos["price"]
                    # Check DxTrade SL/TP — look at position fields AND working orders
                    if trade.platform == "dxtrade" and trade.stop_loss and trade.take_profit:
                        pos_sl = pos.get("sl", 0)
                        pos_tp = pos.get("tp", 0)
                        if not pos_sl or not pos_tp:
                            try:
                                ws_orders = dxtrade_service._ws_cache.get("ORDERS", {}).get("body", [])
                                for wo in (ws_orders if isinstance(ws_orders, list) else []):
                                    legs = wo.get("legs", [])
                                    for leg in (legs if isinstance(legs, list) else []):
                                        if str(leg.get("positionCode", "")) == str(trade.platform_ticket):
                                            if wo.get("type") == "STOP" and wo.get("status") == "WORKING":
                                                pos_sl = wo.get("stopPrice", 0)
                                            elif wo.get("type") == "LIMIT" and wo.get("status") == "WORKING":
                                                pos_tp = wo.get("limitPrice", 0)
                            except Exception:
                                pass
                        if not pos_sl or not pos_tp:
                            try:
                                acc = next((a for a in cfg.accounts if a.account_type == "dxtrade" and a.enabled), None)
                                if acc:
                                    fill = pos.get("price") or trade.entry_price
                                    logger.info("Trade %d missing SL/TP — setting via protections (fill=%.5f sl=%.5f tp=%.5f)",
                                                trade.id, fill, trade.stop_loss, trade.take_profit)
                                    dxtrade_service.modify_position(acc, trade.platform_ticket, trade.stop_loss, trade.take_profit)
                            except Exception as exc:
                                logger.error("Trade %d SL/TP safeguard error: %s", trade.id, exc)
                else:
                    # Position gone from broker — closed externally (SL/TP/manual)
                    trade.status = "closed"
                    trade.closed_at = datetime.now(timezone.utc)
                    # Look up actual PnL from broker deal history
                    if trade.platform == "mt5":
                        try:
                            import MetaTrader5 as mt5
                            deals = mt5.history_deals_get(position=trade.platform_ticket)
                            if deals:
                                close_deal = [d for d in deals if d.entry == 1]
                                if close_deal:
                                    trade.pnl = close_deal[0].profit
                                    trade.exit_price = close_deal[0].price
                                open_deal = [d for d in deals if d.entry == 0]
                                if open_deal and not trade.entry_price:
                                    trade.entry_price = open_deal[0].price
                        except Exception:
                            pass
                    elif trade.platform == "dxtrade":
                        # Look up PnL from DxTrade deal history
                        try:
                            dx_acc = next((a for a in cfg.accounts if a.account_type == "dxtrade" and a.enabled), None)
                            dx_deals = dxtrade_service.get_deal_history(
                                dx_acc, datetime.now(timezone.utc) - timedelta(days=7), datetime.now(timezone.utc)
                            ) if dx_acc else []
                            for dd in dx_deals:
                                if str(dd.get("ticket")) == str(trade.platform_ticket):
                                    trade.pnl = dd.get("profit", 0)
                                    trade.exit_price = dd.get("exit_price")
                                    break
                        except Exception:
                            pass
                    # Determine exit type and price
                    if trade.pnl is not None:
                        trade.exit_type = "tp" if trade.pnl >= 0 else "sl"
                    elif trade.current_price:
                        trade.exit_price = trade.current_price
                        if trade.stop_loss and trade.take_profit:
                            sl_dist = abs(trade.current_price - trade.stop_loss)
                            tp_dist = abs(trade.current_price - trade.take_profit)
                            trade.exit_type = "sl" if sl_dist < tp_dist else "tp"
                        else:
                            trade.exit_type = "unknown"
                    else:
                        trade.exit_type = "unknown"
                    logger.info("Trade %d closed externally (%s, pnl=%.2f)", trade.id, trade.exit_type, trade.pnl or 0)
        else:
            # Paper mode — update prices from latest candles
            from models.candle import Candle
            for trade in open_trades:
                latest = session.query(Candle).filter(
                    Candle.symbol == trade.symbol,
                ).order_by(Candle.ts.desc()).first()
                if latest:
                    trade.current_price = latest.close
                    pip_val = _pip_value(trade.symbol)
                    lot_units = LOT_UNITS.get(cfg.execution.default_lot_type, 10000)
                    if trade.side == "buy":
                        trade.pnl = (latest.close - trade.entry_price) * trade.volume * lot_units
                    else:
                        trade.pnl = (trade.entry_price - latest.close) * trade.volume * lot_units

                    # Check SL/TP in paper mode
                    if trade.side == "buy":
                        if latest.low <= trade.stop_loss:
                            trade.status = "closed"
                            trade.exit_price = trade.stop_loss
                            trade.exit_type = "sl"
                            trade.pnl = (trade.stop_loss - trade.entry_price) * trade.volume * lot_units
                            trade.closed_at = datetime.now(timezone.utc)
                        elif latest.high >= trade.take_profit:
                            trade.status = "closed"
                            trade.exit_price = trade.take_profit
                            trade.exit_type = "tp"
                            trade.pnl = (trade.take_profit - trade.entry_price) * trade.volume * lot_units
                            trade.closed_at = datetime.now(timezone.utc)
                    else:
                        if latest.high >= trade.stop_loss:
                            trade.status = "closed"
                            trade.exit_price = trade.stop_loss
                            trade.exit_type = "sl"
                            trade.pnl = (trade.entry_price - trade.stop_loss) * trade.volume * lot_units
                            trade.closed_at = datetime.now(timezone.utc)
                        elif latest.low <= trade.take_profit:
                            trade.status = "closed"
                            trade.exit_price = trade.take_profit
                            trade.exit_type = "tp"
                            trade.pnl = (trade.entry_price - trade.take_profit) * trade.volume * lot_units
                            trade.closed_at = datetime.now(timezone.utc)

        # Collect trades that just closed for post-trade analysis
        just_closed = [t for t in open_trades if t.status == "closed"]
        session.commit()

        # Kick off post-trade analysis in background threads (don't block sync)
        if just_closed and cfg.claude.enabled:
            for trade in just_closed:
                trade_id = trade.id
                threading.Thread(
                    target=_run_post_trade_analysis,
                    args=(trade_id,),
                    daemon=True,
                    name=f"trade-analysis-{trade_id}",
                ).start()

    except Exception as exc:
        session.rollback()
        logger.error("Position sync error: %s", exc)
    finally:
        session.close()


def _run_post_trade_analysis(trade_id: int):
    """Background: analyze a just-closed trade with Claude and store results."""
    import threading
    from services import claude_service, news_service
    from services.rule_engine import _compute_indicators
    from models.candle import Candle

    session = get_session()
    try:
        trade = session.query(LiveTrade).filter(LiveTrade.id == trade_id).first()
        if not trade or trade.status != "closed":
            return

        # Build context for Claude
        trade_data = {
            "symbol": trade.symbol,
            "side": trade.side,
            "entry_price": trade.entry_price,
            "exit_price": trade.exit_price or trade.current_price,
            "stop_loss": trade.stop_loss,
            "take_profit": trade.take_profit,
            "pnl": trade.pnl,
            "volume": trade.volume,
            "exit_type": trade.exit_type,
            "platform": trade.platform,
            "opened_at": trade.opened_at.isoformat() if trade.opened_at else "",
            "closed_at": trade.closed_at.isoformat() if trade.closed_at else "",
        }

        # Get candles around the trade period for context
        try:
            candles = session.query(Candle).filter(
                Candle.symbol == trade.symbol,
                Candle.timeframe == "15m",
            ).order_by(Candle.ts.desc()).limit(100).all()
            candles.reverse()
            if candles:
                candle_dicts = [{"time": c.ts.isoformat(), "open": c.open, "high": c.high,
                                 "low": c.low, "close": c.close} for c in candles]
                # Indicators at entry and exit
                trade_data["indicators_at_entry"] = "\n".join(
                    f"  {k}: {v}" for k, v in _compute_indicators(candle_dicts[:50]).items()
                    if isinstance(v, (bool, int, float, str)) and not str(v).startswith("0.")
                )[:500]
                trade_data["indicators_at_exit"] = "\n".join(
                    f"  {k}: {v}" for k, v in _compute_indicators(candle_dicts).items()
                    if isinstance(v, (bool, int, float, str)) and not str(v).startswith("0.")
                )[:500]
                # Recent price action summary
                recent = candle_dicts[-10:]
                trade_data["candles_context"] = "\n".join(
                    f"  {c['time']}: O={c['open']:.5f} H={c['high']:.5f} L={c['low']:.5f} C={c['close']:.5f}"
                    for c in recent
                )
        except Exception:
            pass

        # Get news events during the trade
        try:
            if trade.opened_at:
                events = news_service.get_events_at_time(trade.symbol, trade.opened_at, hours_before=1, hours_after=4)
                if events:
                    trade_data["news_context"] = "\n".join(
                        f"  {e.get('title', '?')} ({e.get('currency', '?')}, {e.get('impact', '?')}) "
                        f"{e.get('minutes_until', '?')}min from entry"
                        for e in events[:5]
                    )
        except Exception:
            pass

        # Call Claude for analysis
        analysis = claude_service.analyze_closed_trade(trade_data)

        # Store result
        trade.trade_analysis = analysis
        session.commit()
        logger.info("Trade %d post-analysis complete: %s (pnl=$%.2f)",
                     trade_id, analysis.get("outcome", "?"), trade.pnl or 0)

    except Exception as exc:
        session.rollback()
        logger.error("Post-trade analysis error for trade %d: %s", trade_id, exc)
    finally:
        session.close()


async def position_sync_loop(stop_event: asyncio.Event) -> None:
    """Background loop that syncs open positions with broker/paper state."""
    logger.info("Position sync loop started.")
    while not stop_event.is_set():
        cfg = load_config()
        try:
            await asyncio.to_thread(_sync_positions_once)
        except Exception as exc:
            logger.error("Position sync loop error: %s", exc)

        try:
            await asyncio.wait_for(
                stop_event.wait(), timeout=cfg.execution.position_sync_interval_sec
            )
        except asyncio.TimeoutError:
            pass
    logger.info("Position sync loop stopped.")
