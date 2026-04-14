"""DxTrade REST trading service.

Based on patterns from C:\\MT5ToDXTrade project which had a working
DxTrade integration. Uses REST endpoints with multiple fallback paths.
"""

from __future__ import annotations

import json
import logging
import re
import threading
import time
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List

import httpx

from config import AccountConfig

logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)

# Module-level session state
_session: httpx.Client | None = None
_csrf_token: str = ""
_account_id: str = ""
_server_url: str = ""
_logged_in: bool = False

# Persistent WebSocket — DxTrade requires an active message bus for REST orders
_ws_cache: dict = {}
_ws_cache_time: float = 0
_WS_CACHE_TTL = 300  # 5 minutes
_ws_lock = threading.Lock()
_ws_thread: threading.Thread | None = None
_ws_obj = None  # live websocket connection
_ws_stop = threading.Event()

# Endpoint patterns — try multiple variants (different DxTrade versions)
_EP_ACCOUNT = (
    "/dxsca-web/api/accounts/{aid}/summary",
    "/api/accounts/{aid}/summary",
)
_EP_POSITIONS = (
    "/dxsca-web/api/positions?accountId={aid}",
    "/api/positions?accountId={aid}",
)
_EP_ORDER = (
    "/api/orders/single",
    "/dxsca-web/api/orders/single",
)
_EP_CLOSE = (
    "/api/positions/close",
    "/dxsca-web/api/positions/close",
)
_EP_PROTECTIONS = (
    "/api/positions/protections",
    "/dxsca-web/api/positions/protections",
)
_EP_LOGIN = (
    "/api/auth/login",
    "/dxsca-web/login",
)


def is_available() -> bool:
    return True


_init_lock = threading.Lock()


def initialize(account: AccountConfig) -> bool:
    """Login to DxTrade and cache session. Reuses existing session if still valid."""
    global _session, _csrf_token, _account_id, _server_url, _logged_in

    # Already logged in — reuse session
    if _logged_in and _session is not None:
        return True

    # Prevent concurrent initialization from resetting a working session
    if not _init_lock.acquire(timeout=30):
        return _logged_in  # another thread is initializing

    try:
        # Double-check after acquiring lock
        if _logged_in and _session is not None:
            return True

        _server_url = account.server.rstrip("/")
        _account_id = account.api_key or account.login
        _logged_in = False

        _session = httpx.Client(timeout=30.0, follow_redirects=True)

        # Try login endpoints
        vendor = _server_url.split("//")[-1].split("/")[0]
        login_payloads = [
            {"username": account.login, "password": account.password, "vendor": vendor, "rememberMe": True},
            {"username": account.login, "password": account.password, "domain": "fxify", "rememberMe": True},
            {"username": account.login, "password": account.password},
        ]

        for ep in _EP_LOGIN:
            for payload in login_payloads:
                try:
                    resp = _session.post(
                        f"{_server_url}{ep}",
                        json=payload,
                        headers={"Content-Type": "application/json"},
                    )
                    if resp.status_code not in (200, 201, 204):
                        continue
                    if not resp.text:
                        continue
                    data = resp.json()

                    login_info = data.get("loginInfoTO", {})
                    if login_info.get("successful") or login_info.get("status"):
                        tok = data.get("accessToken") or data.get("token") or data.get("jwt")
                        if tok:
                            _session.headers["Authorization"] = f"Bearer {tok}"
                        logger.info("DxTrade login OK via %s", ep)
                        _logged_in = True
                        break
                except Exception:
                    continue
            if _logged_in:
                break

        if not _logged_in:
            logger.error("DxTrade login failed on all endpoints")
            return False

        # Get CSRF token
        try:
            page = _session.get(f"{_server_url}/")
            m = re.search(r'name=["\']csrf["\']\s+content=["\']([^"\']+)["\']', page.text, re.I)
            if m:
                _csrf_token = m.group(1)
            for name in ("XSRF-TOKEN", "xsrf-token", "X-XSRF-TOKEN"):
                val = _session.cookies.get(name)
                if val:
                    _csrf_token = val
                    break
        except Exception:
            pass

        _session.headers.update({
            "Accept": "application/json, text/plain, */*",
            "Content-Type": "application/json; charset=UTF-8",
            "X-Requested-With": "XMLHttpRequest",
            "Origin": _server_url,
            "Referer": f"{_server_url}/",
        })
        if _csrf_token:
            _session.headers["X-CSRF-Token"] = _csrf_token
            _session.headers["x-csrf-token"] = _csrf_token
        if _account_id:
            _session.headers["X-Account-Id"] = _account_id

        if not _account_id:
            _account_id = _discover_account_id()

        _ensure_ws_running()

        return True
    finally:
        _init_lock.release()


def _discover_account_id() -> str:
    """Try to auto-discover account ID from /api/accounts."""
    for ep in ("/api/accounts", "/dxsca-web/api/accounts"):
        try:
            r = _session.get(f"{_server_url}{ep}")
            if r.status_code == 200:
                data = r.json()
                accs = data if isinstance(data, list) else data.get("accounts", [])
                for a in accs:
                    aid = a.get("accountId") or a.get("id") or a.get("account")
                    if aid:
                        logger.info("DxTrade auto-discovered account ID: %s", aid)
                        return str(aid)
        except Exception:
            continue
    return ""


def shutdown() -> None:
    """No-op for DxTrade — keep session alive to avoid repeated logins.
    Call force_shutdown() to truly close."""
    pass


def force_shutdown() -> None:
    """Actually close the session and WS thread (called on app shutdown)."""
    global _session, _csrf_token, _account_id, _logged_in, _ws_cache, _ws_obj, _ws_thread, _last_good_snapshot
    _last_good_snapshot = None
    _ws_stop.set()
    if _ws_obj:
        try:
            _ws_obj.close()
        except Exception:
            pass
        _ws_obj = None
    if _ws_thread:
        _ws_thread.join(timeout=5)
        _ws_thread = None
    if _session:
        try:
            _session.close()
        except Exception:
            pass
    _session = None
    _csrf_token = ""
    _account_id = ""
    _logged_in = False
    _ws_cache = {}


# symbol → instrumentId map. Seeded with known FXIFY/Alchemy Markets IDs;
# dynamically extended from WS INSTRUMENTS/POSITIONS messages.
_instrument_cache: dict[str, int] = {
    "EURUSD.r": 1889,
    "GBPUSD.r": 1868,
    "USDJPY.r": 1910,
}


def _resolve_account_id():
    """Use WS ACCOUNTS message to find the correct internal account ID.

    DxTrade has two IDs: accountCode (e.g. '1823206') and internal id (e.g. '85993').
    The API requires the internal id, but config typically stores the accountCode.
    """
    global _account_id
    messages = _get_ws_data()
    accts_msg = messages.get("ACCOUNTS", {}).get("body", {})
    if not accts_msg:
        return

    # Check if current _account_id is actually an accountCode, not an id
    current = accts_msg.get("currentAccount", {})
    if current.get("id"):
        internal_id = str(current["id"])
        account_code = str(current.get("accountCode", ""))
        # If our stored ID matches the accountCode, switch to the internal id
        if _account_id == account_code and _account_id != internal_id:
            logger.info("DxTrade account ID resolved: %s (code) → %s (internal)", _account_id, internal_id)
            _account_id = internal_id
            if _session:
                _session.headers["X-Account-Id"] = _account_id


def _build_instrument_cache():
    """Populate instrument cache from WebSocket data.

    Different DxTrade versions send instrument data in different messages.
    Try INSTRUMENTS, LIMITS, POSITIONS, and USER_SETTINGS.
    """
    if _instrument_cache:
        return
    messages = _get_ws_data()

    # Source 1: INSTRUMENTS message (some servers send a full list)
    for msg_type in ("INSTRUMENTS", "LIMITS"):
        body = messages.get(msg_type, {}).get("body", [])
        if not isinstance(body, list):
            continue
        for item in body:
            iid = item.get("id") or item.get("instrumentId")
            sym = item.get("symbol")
            if iid is not None and sym:
                _instrument_cache[str(sym).strip()] = int(iid)

    # Source 2: POSITIONS — each has positionKey.instrumentId + symbol from MESSAGE logs
    positions = messages.get("POSITIONS", {}).get("body", [])
    if isinstance(positions, list):
        for pos in positions:
            pk = pos.get("positionKey", {})
            iid = pk.get("instrumentId")
            # Positions don't have symbol directly — match via MESSAGE trade logs
            if iid is not None:
                # Look up symbol from trade messages
                for msg in messages.get("MESSAGE", {}).get("body", []):
                    params = msg.get("parametersTO", {})
                    if str(params.get("orderKey")) == str(pos.get("uid")) and params.get("symbol"):
                        _instrument_cache[params["symbol"]] = int(iid)
                        break

    # Source 3: INSTRUMENT_METRICS — has instrumentId but no symbol, still useful for reverse lookup
    # (already handled via positions above)

    if _instrument_cache:
        logger.info("DxTrade instrument cache: %d instruments (%s)", len(_instrument_cache), dict(_instrument_cache))


def _resolve_instrument_id(symbol: str) -> int | None:
    """Look up the integer instrumentId for a symbol string."""
    _build_instrument_cache()

    # Exact match
    if symbol in _instrument_cache:
        return _instrument_cache[symbol]
    if symbol.upper() in _instrument_cache:
        return _instrument_cache[symbol.upper()]

    # Fuzzy: EURUSD matches EURUSD.r, EURUSD.RR, etc
    upper = symbol.upper()
    for cached_sym, cached_id in _instrument_cache.items():
        if cached_sym.upper().startswith(upper):
            _instrument_cache[upper] = cached_id
            return cached_id

    # REST search fallback — try instrument search endpoints
    if _session:
        for ep in ("/api/instruments/search?query={sym}", "/dxsca-web/api/instruments/search?query={sym}"):
            try:
                r = _session.get(f"{_server_url}{ep.format(sym=symbol)}", timeout=10)
                if r.status_code != 200:
                    continue
                data = r.json()
                items = data if isinstance(data, list) else (data.get("items") or data.get("result") or [])
                for item in items:
                    iid = item.get("id") or item.get("instrumentId")
                    sym_name = item.get("symbol", "")
                    if iid is not None and sym_name:
                        _instrument_cache[sym_name] = int(iid)
                        if sym_name.upper().startswith(upper):
                            return int(iid)
            except Exception:
                continue

    return None


def _try_get(endpoints: tuple, **fmt_kwargs) -> dict | list | None:
    """Try multiple endpoint patterns, return first successful JSON response."""
    if not _session:
        return None
    for ep in endpoints:
        url = f"{_server_url}{ep.format(**fmt_kwargs)}"
        try:
            r = _session.get(url)
            if r.status_code == 200 and r.text:
                return r.json()
        except Exception:
            continue
    return None


def _extract_balance(summary: dict) -> dict:
    """Extract balance/equity from various DxTrade response formats."""
    # Navigate into nested structures
    if "account" in summary and isinstance(summary["account"], dict):
        summary = summary["account"]
    elif "summary" in summary and isinstance(summary["summary"], dict):
        summary = summary["summary"]

    bal_keys = ["balance", "cashBalance", "cash", "availableCash",
                "walletBalance", "accountBalance", "balanceValue", "availableFunds"]
    eq_keys = ["equity", "equityValue", "accountValue", "netLiquidation",
               "netLiq", "netEquity", "totalEquity"]

    balance = 0
    for k in bal_keys:
        if k in summary and summary[k]:
            balance = float(summary[k])
            break

    equity = 0
    for k in eq_keys:
        if k in summary and summary[k]:
            equity = float(summary[k])
            break

    return {
        "balance": balance,
        "equity": equity or balance,
        "currency": summary.get("currency", "USD"),
    }


def _ws_build_url() -> str:
    return (
        f"{_server_url.replace('https://', 'wss://').replace('http://', 'ws://')}"
        "/client/connector"
        "?X-Atmosphere-tracking-id=0"
        "&X-Atmosphere-Framework=2.3.2-javascript"
        "&X-Atmosphere-Transport=websocket"
        "&X-Atmosphere-TrackMessageSize=true"
        "&Content-Type=text/x-gwt-rpc;%20charset=UTF-8"
        "&X-atmo-protocol=true"
        "&sessionState=dx-new&guest-mode=false"
    )


def _ws_cookie_string() -> str:
    seen = {}
    for c in _session.cookies.jar:
        seen[c.name] = c.value
    return "; ".join(f"{k}={v}" for k, v in seen.items())


def _ws_parse_frame(frame: str):
    """Parse an Atmosphere frame into typed messages, updating _ws_cache in place."""
    global _ws_cache_time
    for part in frame.split("|"):
        part = part.strip()
        if not part.startswith("{"):
            continue
        try:
            d = json.loads(part)
            dtype = d.get("type", "")
            if not dtype:
                continue
            if dtype == "MESSAGE":
                if "MESSAGE" not in _ws_cache:
                    _ws_cache["MESSAGE"] = {"type": "MESSAGE", "body": []}
                body = d.get("body", [])
                if isinstance(body, list):
                    trade_logs = [m for m in body if isinstance(m, dict) and m.get("messageCategory") == "TRADE_LOG"]
                    _ws_cache["MESSAGE"]["body"].extend(trade_logs)
            elif dtype == "INSTRUMENTS" and isinstance(d.get("body"), list):
                _ws_cache[dtype] = d
                for item in d["body"]:
                    iid = item.get("id") or item.get("instrumentId")
                    sym = item.get("symbol")
                    if iid is not None and sym:
                        _instrument_cache[str(sym).strip()] = int(iid)
            elif dtype in ("POSITIONS", "POSITION_METRICS"):
                # Merge incrementally — DxTrade sends partial updates, not full lists
                body = d.get("body", [])
                if not isinstance(body, list):
                    body = [body] if isinstance(body, dict) else []
                if dtype not in _ws_cache:
                    _ws_cache[dtype] = {"type": dtype, "body": []}
                existing = _ws_cache[dtype].get("body", [])
                if not isinstance(existing, list):
                    existing = []
                for item in body:
                    uid = item.get("uid") or (item.get("positionKey", {}).get("positionCode") if isinstance(item.get("positionKey"), dict) else None) or ""
                    if not uid:
                        continue
                    # Check if position was closed (quantity=0)
                    qty = item.get("quantity", None)
                    if qty is not None and qty == 0:
                        existing = [e for e in existing if (e.get("uid") or (e.get("positionKey", {}).get("positionCode") if isinstance(e.get("positionKey"), dict) else "")) != uid]
                    else:
                        # Update or add
                        replaced = False
                        for i, e in enumerate(existing):
                            e_uid = e.get("uid") or (e.get("positionKey", {}).get("positionCode") if isinstance(e.get("positionKey"), dict) else "")
                            if e_uid == uid:
                                existing[i] = item
                                replaced = True
                                break
                        if not replaced:
                            existing.append(item)
                _ws_cache[dtype]["body"] = existing
            else:
                _ws_cache[dtype] = d
            _ws_cache_time = time.time()
        except Exception:
            pass


def _ws_loop():
    """Persistent WebSocket loop — keeps message bus alive for REST order placement."""
    global _ws_obj
    try:
        import websocket as ws_lib
    except ImportError:
        logger.error("websocket-client package not installed")
        return

    while not _ws_stop.is_set():
        if not _session or not _server_url:
            _ws_stop.wait(5.0)
            continue
        try:
            cookie_str = _ws_cookie_string()
            ws_url = _ws_build_url()
            logger.info("DxTrade WS connecting (persistent)...")
            _ws_obj = ws_lib.create_connection(
                ws_url, header={"Cookie": cookie_str, "Origin": _server_url}, timeout=10,
            )
            logger.info("DxTrade WS connected, reading frames...")

            while not _ws_stop.is_set():
                _ws_obj.settimeout(10.0)
                try:
                    frame = _ws_obj.recv()
                    if not frame:
                        continue
                    # Skip very large frames (instrument lists) but still count as alive
                    if len(frame) > 200_000:
                        continue
                    _ws_parse_frame(frame)
                    # Signal initial load once we have metrics
                    if "ACCOUNT_METRICS" in _ws_cache and not _ws_initial_load.is_set():
                        _ws_initial_load.set()
                except ws_lib.WebSocketTimeoutException:
                    continue
                except ws_lib.WebSocketConnectionClosedException:
                    logger.warning("DxTrade WS closed by server, will reconnect")
                    break
                except Exception:
                    break

            try:
                _ws_obj.close()
            except Exception:
                pass
            _ws_obj = None
        except Exception as exc:
            logger.error("DxTrade WS connection error: %s", exc)
            _ws_obj = None

        # Backoff before reconnect
        if not _ws_stop.is_set():
            _ws_stop.wait(5.0)


def _ensure_ws_running():
    """Start the persistent WS thread if not already running."""
    global _ws_thread
    if _ws_thread and _ws_thread.is_alive():
        return
    _ws_stop.clear()
    _ws_thread = threading.Thread(target=_ws_loop, daemon=True, name="dxtrade-ws")
    _ws_thread.start()


_ws_initial_load = threading.Event()  # set once first full data burst is received


def _get_ws_data(wait: bool = False) -> dict:
    """Get WebSocket data. Starts persistent connection if needed.

    The persistent WS thread keeps _ws_cache updated continuously.
    If wait=True, blocks up to 15s for ACCOUNT_METRICS (used before placing orders).
    Otherwise returns immediately with whatever is cached.
    """
    _ensure_ws_running()

    if wait and "ACCOUNT_METRICS" not in _ws_cache:
        deadline = time.time() + 15
        while time.time() < deadline:
            if "ACCOUNT_METRICS" in _ws_cache:
                break
            time.sleep(0.5)

    return _ws_cache


_last_good_snapshot: dict | None = None


def get_account_snapshot(account: AccountConfig) -> dict | None:
    global _last_good_snapshot

    if not _logged_in:
        return _last_good_snapshot  # return last known good data instead of None

    # Try WebSocket data (the persistent WS keeps this updated)
    messages = _get_ws_data()
    metrics_msg = messages.get("ACCOUNT_METRICS", {})
    body = metrics_msg.get("body", {})
    all_metrics = body.get("allMetrics", body)

    if all_metrics and (all_metrics.get("equity") or all_metrics.get("cashBalance")):
        _last_good_snapshot = {
            "balance": float(all_metrics.get("cashBalance", 0) or all_metrics.get("availableFunds", 0)),
            "equity": float(all_metrics.get("equity", 0)),
            "currency": "USD",
        }
        return _last_good_snapshot

    # Metrics not yet received — return last known data if available
    return _last_good_snapshot


def get_active_trades() -> List[dict]:
    if not _logged_in:
        return []

    messages = _get_ws_data()
    pos_msg = messages.get("POSITIONS", {})
    pos_data = pos_msg.get("body", [])
    pos_list = pos_data if isinstance(pos_data, list) else []

    # Merge live PnL from POSITION_METRICS into each position
    metrics_msg = messages.get("POSITION_METRICS", {})
    metrics_body = metrics_msg.get("body", [])
    metrics_by_uid = {}
    if isinstance(metrics_body, list):
        for m in metrics_body:
            uid = m.get("uid") or str(m.get("positionId", ""))
            if uid:
                metrics_by_uid[uid] = m

    results = []
    for d in pos_list:
        if not isinstance(d, dict):
            continue
        uid = d.get("uid") or ""
        pm = metrics_by_uid.get(uid, {})
        if pm:
            d["_metrics"] = pm  # attach for normalizer
        results.append(_normalize_position(d))
    return results


def _extract_positions(data) -> list:
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in ("positions", "data", "result", "items", "body"):
            v = data.get(key)
            if isinstance(v, list):
                return v
    return []


def _normalize_position(d: dict) -> dict:
    # WS format: uid, positionKey.positionCode, positionKey.instrumentId, quantity, costBasis
    pk = d.get("positionKey", {})
    ticket = pk.get("positionCode") or d.get("uid") or d.get("positionCode") or d.get("id") or ""
    instrument_id = pk.get("instrumentId") or d.get("instrumentId")

    # Resolve symbol from instrument cache (reverse lookup)
    symbol = d.get("symbol") or d.get("instrument") or ""
    if not symbol and instrument_id is not None:
        for sym, iid in _instrument_cache.items():
            if iid == int(instrument_id):
                symbol = sym
                break
        if not symbol:
            symbol = str(instrument_id)

    qty = float(d.get("quantity") or d.get("qty") or d.get("size") or 0)
    side_raw = (d.get("side") or d.get("direction") or "").upper()
    if not side_raw and qty != 0:
        side_raw = "BUY" if qty > 0 else "SELL"
    side = "buy" if "BUY" in side_raw or "LONG" in side_raw else "sell"

    price = float(d.get("costBasis") or d.get("avgPrice") or d.get("price") or d.get("openPrice") or 0)
    sl_raw = d.get("stopLoss") or d.get("sl")
    tp_raw = d.get("takeProfit") or d.get("tp")
    sl = float(sl_raw.get("fixedPrice", 0) if isinstance(sl_raw, dict) else (sl_raw or 0))
    tp = float(tp_raw.get("fixedPrice", 0) if isinstance(tp_raw, dict) else (tp_raw or 0))

    # Use POSITION_METRICS for live PnL. For current price, plRate is a conversion
    # rate (not the instrument price) so only use it if it's close to the entry price.
    pm = d.get("_metrics", {})
    pl_rate = float(pm.get("plRate") or 0)
    # plRate is the instrument price only if it's within 20% of entry price
    if pl_rate and price and abs(pl_rate - price) / price < 0.2:
        current = pl_rate
    else:
        current = float(d.get("currentPrice") or d.get("lastPrice") or price)
    pnl_raw = pm.get("plOpen") or d.get("unrealizedPnl") or d.get("pnl") or d.get("fpl") or 0

    return {
        "symbol": str(symbol), "side": side, "price": price,
        "current_price": current, "volume": abs(qty),
        "ticket": str(ticket), "profit": float(pnl_raw),
        "sl": sl, "tp": tp,
        "instrument_id": instrument_id,
    }


def get_position_by_ticket(ticket) -> dict | None:
    for p in get_active_trades():
        if str(p.get("ticket")) == str(ticket):
            return p
    return None


def get_symbol_info(symbol: str) -> dict | None:
    return {
        "name": symbol,
        "digits": 5 if "JPY" not in symbol else 3,
        "point": 0.00001 if "JPY" not in symbol else 0.001,
        "spread": 0,
        "volume_min": 0.01,
        "volume_max": 100.0,
        "volume_step": 0.01,
        "trade_contract_size": 100000,
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
    global _ws_cache, _ws_obj

    if not _session or not _logged_in:
        if not initialize(account):
            return {"error": "DxTrade not initialized", "code": -1}

    # Warm the message bus FIRST — DxTrade requires an active WS session before orders
    # This also populates the instrument cache and resolves the internal account ID
    _get_ws_data(wait=True)
    _resolve_account_id()

    # Resolve integer instrumentId (required by DxTrade)
    instrument_id = _resolve_instrument_id(symbol)
    # Find the full symbol string the server uses (may have suffix like .r)
    server_symbol = symbol
    for cached_sym, cached_id in _instrument_cache.items():
        if cached_id == instrument_id:
            server_symbol = cached_sym
            break

    order_side = "BUY" if side.lower() == "buy" else "SELL"
    # DxTrade uses units, not lots (0.01 lots × 100,000 = 1,000 units)
    qty_units = round(volume * 100_000)
    qty = -qty_units if order_side == "SELL" else qty_units
    request_id = f"wt-{int(time.time() * 1000)}"

    legs = {
        "positionEffect": "OPENING",
        "ratioQuantity": 1,
        "symbol": server_symbol,
    }
    if instrument_id is not None:
        legs["instrumentId"] = instrument_id

    payload = {
        "directExchange": False,
        "legs": [legs],
        "limitPrice": 0,
        "orderSide": order_side,
        "orderType": "MARKET",
        "quantity": qty,
        "timeInForce": "GTC",
        "requestId": request_id,
    }
    if _account_id:
        payload["accountId"] = _account_id

    # Include SL/TP in the order as-is from the signal.
    # The caller (_execute_on_account) already recalculates from fill price.
    if sl and sl > 0:
        payload["stopLoss"] = {
            "orderType": "STOP",
            "fixedPrice": round(sl, 5),
            "priceFixed": True,
            "removed": False,
        }
    if tp and tp > 0:
        payload["takeProfit"] = {
            "orderType": "LIMIT",
            "fixedPrice": round(tp, 5),
            "priceFixed": True,
            "removed": False,
        }

    last_error = ""
    for ep in _EP_ORDER:
        try:
            # Use compact JSON (no spaces) — some DxTrade versions are strict
            resp = _session.post(
                f"{_server_url}{ep}",
                data=json.dumps(payload, separators=(",", ":")),
                headers={**_session.headers, "Content-Type": "application/json; charset=UTF-8"},
            )
            if resp.status_code in (200, 201):
                result_data = resp.json() if resp.text else {}
                order_id = (result_data.get("orderId") or result_data.get("id")
                            or result_data.get("orderChainId") or request_id)
                logger.info("DxTrade order placed: %s %s %s vol=%s via %s",
                            order_side, server_symbol, order_id, volume, ep)
                # Wait briefly for position to appear in WS, get real position code + fill price
                fill_price = 0
                position_code = str(order_id)
                for _ in range(6):
                    time.sleep(0.5)
                    pos_msg = _ws_cache.get("POSITIONS", {}).get("body", [])
                    if isinstance(pos_msg, list):
                        for pos in pos_msg:
                            pk = pos.get("positionKey", {})
                            if pk.get("instrumentId") == instrument_id:
                                position_code = str(pk.get("positionCode") or pos.get("uid", order_id))
                                fill_price = float(pos.get("costBasis", 0))
                                break
                    if fill_price:
                        break
                # If SL/TP were dropped (too close to fill price), recalculate
                # from the actual fill and set via protections endpoint
                if position_code and fill_price and (sl or tp):
                    time.sleep(1)
                    pos_check = _ws_cache.get("POSITIONS", {}).get("body", [])
                    for pc in (pos_check if isinstance(pos_check, list) else []):
                        uid = pc.get("uid") or (pc.get("positionKey", {}).get("positionCode") if isinstance(pc.get("positionKey"), dict) else "")
                        if str(uid) == str(position_code):
                            has_sl = pc.get("stopLoss") is not None
                            has_tp = pc.get("takeProfit") is not None
                            if not has_sl or not has_tp:
                                # Recalculate from fill price with correct distances
                                sl_dist = abs(sl - fill_price) if sl else 0
                                tp_dist = abs(tp - fill_price) if tp else 0
                                pip_val = 0.01 if "JPY" in symbol.upper() else 0.0001
                                min_dist = pip_val * 15
                                sl_dist = max(sl_dist, min_dist)
                                tp_dist = max(tp_dist, min_dist)
                                if order_side == "SELL":
                                    new_sl = fill_price + sl_dist
                                    new_tp = fill_price - tp_dist
                                else:
                                    new_sl = fill_price - sl_dist
                                    new_tp = fill_price + tp_dist
                                logger.info("DxTrade SL/TP dropped — setting via protections: fill=%.5f sl=%.5f tp=%.5f",
                                            fill_price, new_sl, new_tp)
                                try:
                                    modify_position(account, position_code, new_sl, new_tp)
                                except Exception as exc:
                                    logger.warning("DxTrade SL/TP fallback failed: %s", exc)
                            break

                return {
                    "ticket": position_code,
                    "price": fill_price,
                    "volume": volume,
                    "symbol": symbol,
                    "raw": result_data,
                }

            # Handle 409 conflicts with fallback strategies
            if resp.status_code == 409:
                body = (resp.text or "").lower()

                # Fallback: NO_MESSAGE_BUS — WS died, force reconnect and retry
                if "no_message_bus" in body or "no message bus" in body:
                    logger.warning("DxTrade NO_MESSAGE_BUS — forcing WS reconnect and retrying")
                    _ws_cache.clear()  # clear stale cache to force fresh connection
                    if _ws_obj:
                        try:
                            _ws_obj.close()
                        except Exception:
                            pass
                        _ws_obj = None
                    _ensure_ws_running()
                    # Wait for bus to come alive
                    for _ in range(10):
                        time.sleep(1)
                        if "ACCOUNT_METRICS" in _ws_cache:
                            break
                    _resolve_account_id()
                    # Retry the order
                    r2 = _session.post(
                        f"{_server_url}{ep}",
                        data=json.dumps(payload, separators=(",", ":")),
                        headers={**_session.headers, "Content-Type": "application/json; charset=UTF-8"},
                    )
                    if r2.status_code in (200, 201):
                        result_data = r2.json() if r2.text else {}
                        order_id = result_data.get("orderId") or result_data.get("id") or request_id
                        logger.info("DxTrade order placed (after WS reconnect): %s %s %s", order_side, server_symbol, order_id)
                        return {"ticket": order_id, "price": 0, "volume": volume, "symbol": symbol, "raw": result_data}

                # Fallback A: remove accountId from payload (keep in header)
                if "account" in body and "not been found" in body:
                    alt = dict(payload)
                    alt.pop("accountId", None)
                    r2 = _session.post(
                        f"{_server_url}{ep}",
                        data=json.dumps(alt, separators=(",", ":")),
                        headers={**_session.headers, "Content-Type": "application/json; charset=UTF-8"},
                    )
                    if r2.status_code in (200, 201):
                        result_data = r2.json() if r2.text else {}
                        order_id = result_data.get("orderId") or result_data.get("id") or request_id
                        logger.info("DxTrade order placed (no accountId): %s %s %s", order_side, server_symbol, order_id)
                        return {"ticket": order_id, "price": 0, "volume": volume, "symbol": symbol, "raw": result_data}

                # Fallback B: flip quantity sign
                if order_side == "SELL":
                    alt = dict(payload)
                    alt["quantity"] = -alt["quantity"]
                    r2 = _session.post(
                        f"{_server_url}{ep}",
                        data=json.dumps(alt, separators=(",", ":")),
                        headers={**_session.headers, "Content-Type": "application/json; charset=UTF-8"},
                    )
                    if r2.status_code in (200, 201):
                        result_data = r2.json() if r2.text else {}
                        order_id = result_data.get("orderId") or result_data.get("id") or request_id
                        logger.info("DxTrade order placed (flipped qty): %s %s %s", order_side, server_symbol, order_id)
                        return {"ticket": order_id, "price": 0, "volume": volume, "symbol": symbol, "raw": result_data}

            last_error = f"{ep} HTTP {resp.status_code}: {resp.text[:200]}"
            logger.warning("DxTrade order rejected: %s", last_error)
        except Exception as exc:
            last_error = f"{ep}: {exc}"
            logger.warning("DxTrade order exception: %s", last_error)
            continue

    return {"error": f"Order failed: {last_error}", "code": -4}


def modify_position(
    account: AccountConfig,
    ticket: int | str,
    sl: float,
    tp: float,
) -> dict:
    """Modify SL/TP on a DxTrade position using the protections endpoint.

    Uses the exact payload format from the DxTrade web UI:
    positionKey + protectionOrders array with quantityForProtection.
    """
    if not _session or not _logged_in:
        return {"error": "DxTrade not initialized", "code": -1}

    _get_ws_data(wait=True)
    _resolve_account_id()

    pos = get_position_by_ticket(ticket)
    if not pos:
        return {"error": f"Position {ticket} not found", "code": -3}

    instrument_id = pos.get("instrument_id")
    qty = pos.get("volume", 0)
    # quantityForProtection: negative for sell positions, positive for buy
    qty_prot = -abs(qty) if pos.get("side") == "sell" else abs(qty)
    fill_price = pos.get("price", 0)

    protection_orders = []
    if tp and tp > 0:
        protection_orders.append({
            "fixedOffset": round(abs(tp - fill_price), 5) if fill_price else 0,
            "fixedPrice": round(tp, 5),
            "orderType": "LIMIT",
            "priceFixed": True,
            "quantityForProtection": qty_prot,
            "removed": False,
        })
    if sl and sl > 0:
        protection_orders.append({
            "fixedOffset": round(abs(sl - fill_price), 5) if fill_price else 0,
            "fixedPrice": round(sl, 5),
            "orderType": "STOP",
            "priceFixed": True,
            "quantityForProtection": qty_prot,
            "removed": False,
        })

    if not protection_orders:
        return {"ticket": ticket, "sl": sl, "tp": tp}

    payload = {
        "positionKey": {
            "instrumentId": int(instrument_id) if instrument_id else 0,
            "positionCode": str(ticket),
        },
        "protectionOrders": protection_orders,
    }

    resp = _session.post(
        f"{_server_url}/api/positions/protections",
        data=json.dumps(payload, separators=(",", ":")),
        headers={**_session.headers, "Content-Type": "application/json; charset=UTF-8"},
    )
    if resp.status_code in (200, 201):
        logger.info("DxTrade SL/TP set for %s: sl=%.5f tp=%.5f", ticket, sl, tp)
        return {"ticket": ticket, "sl": sl, "tp": tp}

    logger.warning("DxTrade protections failed for %s: %s %s", ticket, resp.status_code, resp.text[:150])
    return {"error": f"Protections failed (HTTP {resp.status_code})", "code": -4}


def close_position(
    account: AccountConfig,
    ticket: int | str,
    volume: float | None = None,
) -> dict:
    if not _session or not _logged_in:
        return {"error": "DxTrade not initialized", "code": -1}

    pos = get_position_by_ticket(ticket)
    if not pos:
        # Fallback: check our DB
        from db import get_session
        from models.signal import LiveTrade
        session = get_session()
        try:
            trade = session.query(LiveTrade).filter(
                LiveTrade.platform_ticket == str(ticket),
                LiveTrade.platform == "dxtrade",
                LiveTrade.status == "open",
            ).first()
            if trade:
                pos = {"symbol": trade.symbol, "side": trade.side, "volume": trade.volume}
        finally:
            session.close()

    if not pos:
        return {"error": f"Position {ticket} not found", "code": -3}

    close_vol = volume if volume else pos["volume"]
    qty = close_vol if pos["side"] == "sell" else -close_vol

    # Build legs with integer instrumentId if available
    leg = {
        "positionCode": str(ticket),
        "positionEffect": "CLOSING",
        "ratioQuantity": 1,
        "symbol": pos["symbol"],
    }
    iid = pos.get("instrument_id")
    if iid is not None:
        leg["instrumentId"] = int(iid)

    payload = {
        "legs": [leg],
        "limitPrice": 0,
        "orderType": "MARKET",
        "quantity": qty,
        "timeInForce": "GTC",
    }
    if _account_id:
        payload["accountId"] = _account_id

    # Ensure WS bus is alive
    _get_ws_data(wait=True)
    _resolve_account_id()

    for ep in _EP_CLOSE:
        try:
            resp = _session.post(
                f"{_server_url}{ep}",
                data=json.dumps(payload, separators=(",", ":")),
                headers={**_session.headers, "Content-Type": "application/json; charset=UTF-8"},
            )
            if resp.status_code in (200, 201, 204):
                logger.info("DxTrade position closed: %s", ticket)
                return {
                    "ticket": ticket,
                    "close_price": pos.get("current_price", 0),
                    "volume_closed": close_vol,
                }
        except Exception:
            continue

    return {"error": "Close failed on all endpoints", "code": -4}


def get_deal_history(account: AccountConfig, start: datetime, end: datetime) -> list[dict]:
    """Extract trade history from WebSocket TRADE_LOG messages."""
    if not _logged_in:
        return []

    messages = _get_ws_data()
    msg_data = messages.get("MESSAGE", {})
    body = msg_data.get("body", [])
    if not isinstance(body, list):
        return []

    # Extract filled orders from TRADE_LOG
    # Group by positionCode — first fill is the open, last fill is the close
    positions: dict[str, list[dict]] = {}  # positionCode -> [fill, fill, ...]
    for m in body:
        if m.get("messageCategory") != "TRADE_LOG":
            continue
        p = m.get("parametersTO", {})
        status = p.get("orderStatus", "")
        if status != "FILLED":
            continue

        pos_code = p.get("positionCode") or p.get("openChainKey")
        if not pos_code:
            continue

        filled_price = p.get("filledPrice")
        if filled_price == "NaN" or not filled_price:
            continue

        ts = m.get("timeStamp", 0)

        entry = {
            "symbol": p.get("symbol", ""),
            "side": p.get("orderSide", ""),
            "price": float(filled_price),
            "volume": float(p.get("filledSize", 0) or p.get("quantity", 0)),
            "order_type": p.get("orderType", ""),
            "timestamp": ts,
            "protection": p.get("protection", False),
        }

        if pos_code not in positions:
            positions[pos_code] = []
        positions[pos_code].append(entry)

    # Build deal list — first fill is the open, last fill is the close
    deals = []
    for pos_code, fill_list in positions.items():
        if not fill_list:
            continue
        # Sort by timestamp to ensure correct order
        fill_list.sort(key=lambda f: f["timestamp"])
        open_fill = fill_list[0]
        close_fill = fill_list[-1] if len(fill_list) > 1 else None

        symbol = open_fill["symbol"]
        side = open_fill["side"].lower()
        entry_price = open_fill["price"]
        volume = open_fill["volume"]
        # Normalize volume (DxTrade uses units, not lots — always convert)
        lot_volume = volume / 100000

        if close_fill:
            exit_price = close_fill["price"]
            exit_type = "tp" if close_fill["order_type"] == "LIMIT" else "sl"
            # Compute P&L
            pip_val = 0.01 if "JPY" in symbol.upper() else 0.0001
            if side == "buy":
                pnl = (exit_price - entry_price) / pip_val * lot_volume * 10
            else:
                pnl = (entry_price - exit_price) / pip_val * lot_volume * 10
            ts = close_fill["timestamp"]
        else:
            # Position still open — skip, not a completed deal
            continue

        deals.append({
            "ticket": pos_code,
            "symbol": symbol,
            "type": side,
            "volume": lot_volume,
            "price": entry_price,
            "exit_price": exit_price,
            "profit": round(pnl, 2),
            "exit_type": exit_type,
            "account": account.name,
            "time": datetime.fromtimestamp(ts / 1000, tz=timezone.utc).isoformat() if ts else None,
        })

    # Filter by date range and sort
    deals = [d for d in deals if d.get("time")]
    deals.sort(key=lambda d: d["time"], reverse=True)
    return deals
