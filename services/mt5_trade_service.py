"""MetaTrader 5 trade execution service.

Provides order placement, modification, and position management.
All functions are synchronous — callers use asyncio.to_thread().
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

try:
    import MetaTrader5 as mt5
except ImportError:  # pragma: no cover - optional dependency
    mt5 = None

from config import AccountConfig, load_config

logger = logging.getLogger(__name__)


def is_available() -> bool:
    return mt5 is not None


def initialize(account: AccountConfig) -> bool:
    if mt5 is None:
        logger.warning("MetaTrader5 package is not installed.")
        return False

    kwargs: Dict[str, Any] = {}
    if account.login:
        try:
            kwargs["login"] = int(account.login)
        except ValueError:
            kwargs["login"] = account.login
    if account.password:
        kwargs["password"] = account.password
    if account.server:
        kwargs["server"] = account.server

    if account.path:
        return mt5.initialize(account.path, **kwargs)
    return mt5.initialize(**kwargs)


def shutdown() -> None:
    if mt5 is None:
        return
    mt5.shutdown()


def _resolve_symbol(symbol: str) -> str:
    """Resolve a symbol name, trying suffix and wildcard match."""
    if mt5 is None:
        raise ValueError("MetaTrader5 package is not installed.")

    cfg = load_config()
    suffix = (cfg.mt5.symbol_suffix or "").strip()

    candidates = [symbol]
    if suffix:
        candidates.insert(0, f"{symbol}{suffix}")

    for candidate in candidates:
        info = mt5.symbol_info(candidate)
        if info is not None:
            mt5.symbol_select(candidate, True)
            return candidate

    matched = mt5.symbols_get(f"{symbol}*") or []
    if matched:
        selected = matched[0].name
        mt5.symbol_select(selected, True)
        return selected

    raise ValueError(f"Could not find MT5 symbol matching '{symbol}'")


def get_account_snapshot(account: AccountConfig) -> dict | None:
    if mt5 is None:
        return None
    info = mt5.account_info()
    if info is None:
        return None
    return {
        "balance": float(info.balance),
        "equity": float(info.equity),
        "currency": info.currency,
    }


def get_active_trades() -> List[dict]:
    if mt5 is None:
        return []
    positions = mt5.positions_get()
    if positions is None:
        return []
    results: List[dict] = []
    for pos in positions:
        side = "buy" if pos.type == 0 else "sell"
        results.append(
            {
                "symbol": pos.symbol,
                "side": side,
                "price": float(pos.price_open),
                "current_price": float(pos.price_current),
                "volume": float(pos.volume),
                "ticket": pos.ticket,
                "profit": float(pos.profit),
                "sl": float(pos.sl),
                "tp": float(pos.tp),
            }
        )
    return results


def get_position_by_ticket(ticket: int) -> dict | None:
    """Fetch a specific position by ticket number."""
    if mt5 is None:
        return None
    positions = mt5.positions_get(ticket=ticket)
    if not positions:
        return None
    pos = positions[0]
    side = "buy" if pos.type == 0 else "sell"
    return {
        "symbol": pos.symbol,
        "side": side,
        "price": float(pos.price_open),
        "current_price": float(pos.price_current),
        "volume": float(pos.volume),
        "ticket": pos.ticket,
        "profit": float(pos.profit),
        "sl": float(pos.sl),
        "tp": float(pos.tp),
    }


def get_symbol_info(symbol: str) -> dict | None:
    """Get symbol trading properties (pip value, spread, volume limits)."""
    if mt5 is None:
        return None
    try:
        resolved = _resolve_symbol(symbol)
    except ValueError:
        return None
    info = mt5.symbol_info(resolved)
    if info is None:
        return None
    return {
        "name": info.name,
        "digits": info.digits,
        "point": info.point,
        "spread": info.spread,
        "volume_min": info.volume_min,
        "volume_max": info.volume_max,
        "volume_step": info.volume_step,
        "trade_contract_size": info.trade_contract_size,
    }


def place_market_order(
    account: AccountConfig,
    symbol: str,
    side: str,
    volume: float,
    sl: float,
    tp: float,
    comment: str = "WeldonTrader",
) -> dict:
    """Place a market order. Returns result dict with 'ticket' on success or 'error' on failure."""
    if mt5 is None:
        return {"error": "MetaTrader5 not installed", "code": -1}

    if not initialize(account):
        return {"error": "MT5 initialization failed", "code": -2}

    try:
        resolved = _resolve_symbol(symbol)
        info = mt5.symbol_info(resolved)
        if info is None:
            return {"error": f"Symbol info not available for {resolved}", "code": -3}

        # Clamp volume to symbol limits
        vol = max(info.volume_min, min(info.volume_max, volume))
        # Round to volume step
        if info.volume_step > 0:
            vol = round(vol / info.volume_step) * info.volume_step
        vol = round(vol, 2)

        if side.lower() == "buy":
            order_type = mt5.ORDER_TYPE_BUY
            price = mt5.symbol_info_tick(resolved).ask
        else:
            order_type = mt5.ORDER_TYPE_SELL
            price = mt5.symbol_info_tick(resolved).bid

        # Pick a filling mode the broker actually supports
        # filling_mode bitmask: bit 0 (1)=FOK, bit 1 (2)=IOC
        filling = mt5.ORDER_FILLING_IOC
        if hasattr(info, "filling_mode"):
            fm = info.filling_mode
            if fm & 1:
                filling = mt5.ORDER_FILLING_FOK
            elif fm & 2:
                filling = mt5.ORDER_FILLING_IOC
            else:
                filling = mt5.ORDER_FILLING_RETURN

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": resolved,
            "volume": vol,
            "type": order_type,
            "price": price,
            "sl": sl,
            "tp": tp,
            "deviation": 20,
            "magic": 202500,
            "comment": comment,
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": filling,
        }

        result = mt5.order_send(request)
        if result is None:
            return {"error": "order_send returned None", "code": -4}

        if result.retcode != mt5.TRADE_RETCODE_DONE:
            return {
                "error": f"Order failed: {result.comment}",
                "code": result.retcode,
                "raw": str(result),
            }

        # MT5 market orders often return price=0 in OrderSendResult.
        # Look up actual fill price from the deal.
        fill_price = result.price
        if not fill_price and result.order:
            import time as _t
            _t.sleep(0.3)  # brief pause for deal to register
            deals = mt5.history_deals_get(position=result.order)
            if deals:
                fill_price = deals[0].price

        return {
            "ticket": result.order,
            "price": fill_price or price,  # fallback to requested price
            "volume": vol,
            "symbol": resolved,
            "raw": str(result),
        }
    finally:
        shutdown()


def modify_position(
    account: AccountConfig,
    ticket: int,
    sl: float,
    tp: float,
) -> dict:
    """Modify SL/TP on an open position."""
    if mt5 is None:
        return {"error": "MetaTrader5 not installed", "code": -1}

    if not initialize(account):
        return {"error": "MT5 initialization failed", "code": -2}

    try:
        position = get_position_by_ticket(ticket)
        if position is None:
            return {"error": f"Position {ticket} not found", "code": -3}

        request = {
            "action": mt5.TRADE_ACTION_SLTP,
            "position": ticket,
            "symbol": position["symbol"],
            "sl": sl,
            "tp": tp,
        }

        result = mt5.order_send(request)
        if result is None:
            return {"error": "order_send returned None", "code": -4}

        if result.retcode != mt5.TRADE_RETCODE_DONE:
            return {"error": f"Modify failed: {result.comment}", "code": result.retcode}

        return {"ticket": ticket, "sl": sl, "tp": tp}
    finally:
        shutdown()


def close_position(
    account: AccountConfig,
    ticket: int,
    volume: float | None = None,
) -> dict:
    """Close a position (or partial close if volume specified)."""
    if mt5 is None:
        return {"error": "MetaTrader5 not installed", "code": -1}

    if not initialize(account):
        return {"error": "MT5 initialization failed", "code": -2}

    try:
        position = get_position_by_ticket(ticket)
        if position is None:
            return {"error": f"Position {ticket} not found", "code": -3}

        close_vol = volume if volume else position["volume"]
        # Opposite order type to close
        if position["side"] == "buy":
            order_type = mt5.ORDER_TYPE_SELL
            price = mt5.symbol_info_tick(position["symbol"]).bid
        else:
            order_type = mt5.ORDER_TYPE_BUY
            price = mt5.symbol_info_tick(position["symbol"]).ask

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": position["symbol"],
            "volume": close_vol,
            "type": order_type,
            "position": ticket,
            "price": price,
            "deviation": 20,
            "magic": 202500,
            "comment": "WeldonTrader close",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        result = mt5.order_send(request)
        if result is None:
            return {"error": "order_send returned None", "code": -4}

        if result.retcode != mt5.TRADE_RETCODE_DONE:
            return {"error": f"Close failed: {result.comment}", "code": result.retcode}

        return {
            "ticket": ticket,
            "close_price": result.price,
            "volume_closed": close_vol,
        }
    finally:
        shutdown()
