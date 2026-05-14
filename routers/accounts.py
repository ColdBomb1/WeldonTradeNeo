from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from config import AccountConfig, TradingBlackout, CONFIG_PATH, load_config, save_config
from services import mt5_trade_service
from services.performance_archive import after_cutoff, filter_query_after_cutoff

router = APIRouter(tags=["accounts"])

BASE_DIR = Path(__file__).resolve().parent.parent
TEMPLATES = Jinja2Templates(directory=str(BASE_DIR / "templates"))


@router.get("/accounts")
def accounts_page(request: Request):
    cfg = load_config()
    return TEMPLATES.TemplateResponse(
        "accounts.html",
        {
            "request": request,
            "config": cfg,
            "config_path": str(CONFIG_PATH),
            "mt5_available": mt5_trade_service.is_available(),
        },
    )


@router.post("/accounts")
async def add_account(request: Request):
    form = await request.form()
    cfg = load_config()

    account_name = str(form.get("account_name", "")).strip()
    if account_name:
        account = AccountConfig(
            name=account_name,
            account_type=str(form.get("account_type", "mt5")).strip() or "mt5",
            login=str(form.get("account_login", "")).strip(),
            password=str(form.get("account_password", "")).strip(),
            server=str(form.get("account_server", "")).strip(),
            path=str(form.get("account_path", "")).strip(),
            api_key=str(form.get("account_api_key", "")).strip(),
        )
        existing = [acct for acct in cfg.accounts if acct.name != account_name]
        existing.append(account)
        cfg.accounts = existing

    save_config(cfg)
    return RedirectResponse(url="/accounts", status_code=303)


@router.post("/accounts/remove")
async def remove_account(request: Request):
    """Remove an account by name. POST-only — destructive actions must never be GET
    (browsers prefetch links, security scanners follow links, users misclick)."""
    form = await request.form()
    name = str(form.get("name", "")).strip()
    if not name:
        return RedirectResponse(url="/accounts", status_code=303)
    cfg = load_config()
    cfg.accounts = [acct for acct in cfg.accounts if acct.name != name]
    save_config(cfg)
    return RedirectResponse(url="/accounts", status_code=303)


# ---------------------------------------------------------------------------
# Per-account controls (toggle enabled, close all positions)
# ---------------------------------------------------------------------------


@router.post("/api/accounts/{account_name}/toggle")
def toggle_account(account_name: str) -> JSONResponse:
    """Toggle an account's enabled flag. Disabled accounts cannot open new trades."""
    cfg = load_config()
    for acc in cfg.accounts:
        if acc.name == account_name:
            acc.enabled = not acc.enabled
            save_config(cfg)
            # Update cache inline so dashboard reflects change immediately
            with _cache_lock:
                for cached in _account_cache.get("accounts", []):
                    if cached["name"] == account_name:
                        cached["enabled"] = acc.enabled
                        if not acc.enabled:
                            cached["connected"] = False
            return JSONResponse({"ok": True, "account": account_name, "enabled": acc.enabled})
    return JSONResponse({"error": f"Account '{account_name}' not found"}, status_code=404)


@router.post("/api/accounts/{account_name}/close-all")
def close_all_positions(account_name: str) -> JSONResponse:
    """Close all open broker positions for an account. Works even when disabled."""
    import logging
    log = logging.getLogger("accounts.close_all")

    cfg = load_config()
    acc = next((a for a in cfg.accounts if a.name == account_name), None)
    if not acc:
        return JSONResponse({"error": f"Account '{account_name}' not found"}, status_code=404)

    service = _get_trade_service(acc.account_type)
    if not service:
        return JSONResponse({"error": f"No trade service for type '{acc.account_type}'"}, status_code=400)

    if not service.initialize(acc):
        return JSONResponse({"error": "Failed to connect to broker"}, status_code=502)

    try:
        positions = service.get_active_trades()
        if not positions:
            return JSONResponse({"ok": True, "account": account_name, "closed": 0, "errors": 0, "details": []})

        closed = 0
        errors = 0
        details = []
        for pos in positions:
            ticket = pos.get("ticket")
            try:
                result = service.close_position(acc, ticket)
                if "error" in result:
                    errors += 1
                    details.append({"ticket": ticket, "symbol": pos.get("symbol"), "error": result["error"]})
                    log.warning("close-all %s ticket %s failed: %s", account_name, ticket, result["error"])
                else:
                    closed += 1
                    details.append({"ticket": ticket, "symbol": pos.get("symbol"), "closed": True})
            except Exception as exc:
                errors += 1
                details.append({"ticket": ticket, "symbol": pos.get("symbol"), "error": str(exc)[:120]})
                log.error("close-all %s ticket %s exception: %s", account_name, ticket, exc)

        return JSONResponse({"ok": True, "account": account_name, "closed": closed, "errors": errors, "details": details})
    except Exception as exc:
        log.error("close-all %s failed: %s", account_name, exc)
        return JSONResponse({"error": f"Close all failed: {str(exc)[:120]}"}, status_code=500)


# ---------------------------------------------------------------------------
# Live account data API
# ---------------------------------------------------------------------------


def _get_trade_service(account_type):
    if account_type == "mt5":
        return mt5_trade_service
    elif account_type == "dxtrade":
        from services import dxtrade_service
        return dxtrade_service
    return None


# Cached account status — refreshed in background, never blocks page loads
import threading
import time as _time

_account_cache: dict = {"accounts": [], "mode": "paper", "paused": False, "last_update": 0}
_cache_lock = threading.Lock()
_refresh_started: float = 0  # timestamp-based guard instead of boolean


def _fetch_one_account(acc, service) -> dict:
    """Fetch status for a single account. Called from threads."""
    try:
        if not service.initialize(acc):
            return {"name": acc.name, "type": acc.account_type, "enabled": True, "connected": False, "error": "Broker login failed"}
        snap = service.get_account_snapshot(acc)
        if snap and (snap.get("balance", 0) > 0 or snap.get("equity", 0) > 0):
            return {
                "name": acc.name, "type": acc.account_type, "enabled": True, "connected": True,
                "balance": snap["balance"], "equity": snap["equity"],
                "currency": snap.get("currency", "USD"),
            }
        if snap is None:
            return {"name": acc.name, "type": acc.account_type, "enabled": True, "connected": False, "error": "Waiting for data"}
        return {"name": acc.name, "type": acc.account_type, "enabled": True, "connected": True, "balance": 0, "equity": 0, "currency": "USD"}
    except Exception as exc:
        return {"name": acc.name, "type": acc.account_type, "enabled": True, "connected": False, "error": str(exc)[:80]}


def _refresh_account_cache():
    """Fetch live account data. Accounts are queried in parallel so slow brokers don't block others."""
    global _refresh_started
    if _time.time() - _refresh_started < 10:
        return
    _refresh_started = _time.time()

    try:
        from concurrent.futures import ThreadPoolExecutor, as_completed

        cfg = load_config()
        accounts_status = []
        futures = {}

        with ThreadPoolExecutor(max_workers=4) as pool:
            for acc in cfg.accounts:
                if not acc.enabled:
                    accounts_status.append({
                        "name": acc.name, "type": acc.account_type, "enabled": False,
                        "connected": False, "balance": 0, "equity": 0, "currency": "USD",
                    })
                    continue
                service = _get_trade_service(acc.account_type)
                if not service:
                    accounts_status.append({"name": acc.name, "type": acc.account_type, "enabled": True, "connected": False, "error": "No service"})
                    continue
                futures[pool.submit(_fetch_one_account, acc, service)] = acc.name

            for future in as_completed(futures, timeout=60):
                try:
                    accounts_status.append(future.result())
                except Exception as exc:
                    name = futures[future]
                    accounts_status.append({"name": name, "type": "?", "enabled": True, "connected": False, "error": str(exc)[:80]})

        with _cache_lock:
            _account_cache["accounts"] = accounts_status
            _account_cache["mode"] = cfg.execution.mode
            _account_cache["paused"] = cfg.execution.paused
            _account_cache["last_update"] = _time.time()
    except Exception:
        pass
    finally:
        _refresh_started = 0


@router.get("/api/accounts/status")
def get_account_status() -> JSONResponse:
    """Get cached account status. Always returns instantly."""
    cfg = load_config()
    last_update = _account_cache.get("last_update", 0)
    cache_age = _time.time() - last_update if last_update > 0 else 999

    # Background refresh — but DON'T trigger if snapshot loop will handle it
    # The snapshot loop runs at +5s after startup; let it be the single refresh source
    if cache_age > 90 and _refresh_started == 0:
        thread = threading.Thread(target=_refresh_account_cache, daemon=True)
        thread.start()

    with _cache_lock:
        accounts = list(_account_cache.get("accounts", []))

    # Ensure DxTrade accounts always have fresh data from the persistent WS
    from services import dxtrade_service
    for dx_acc in cfg.accounts:
        if dx_acc.account_type != "dxtrade" or not dx_acc.enabled:
            continue
        # Always try to initialize — it returns immediately if already logged in
        try:
            dxtrade_service.initialize(dx_acc)
        except Exception:
            pass
        # Check WS cache for live data
        snap = dxtrade_service.get_account_snapshot(dx_acc)
        if snap and (snap.get("balance", 0) > 0 or snap.get("equity", 0) > 0):
            live_entry = {
                "name": dx_acc.name, "type": "dxtrade", "enabled": True, "connected": True,
                "balance": snap["balance"], "equity": snap["equity"],
                "currency": snap.get("currency", "USD"),
            }
        else:
            live_entry = {
                "name": dx_acc.name, "type": "dxtrade", "enabled": True, "connected": False,
                "balance": 0, "equity": 0, "currency": "USD",
                "error": "Connecting..." if dxtrade_service._logged_in else "Initializing...",
            }
        # Patch into the accounts list (replace or append)
        found = False
        for i, acct in enumerate(accounts):
            if acct.get("name") == dx_acc.name:
                accounts[i] = live_entry
                found = True
                break
        if not found:
            accounts.append(live_entry)
        # Update main cache when connected
        if live_entry.get("connected"):
            with _cache_lock:
                cached = _account_cache.get("accounts", [])
                updated = False
                for j, c in enumerate(cached):
                    if c.get("name") == dx_acc.name:
                        cached[j] = live_entry
                        updated = True
                        break
                if not updated:
                    cached.append(live_entry)
                    _account_cache["accounts"] = cached

    return JSONResponse({
            "accounts": accounts,
            "mode": cfg.execution.mode,
            "paused": cfg.execution.paused,
            "cache_age": round(cache_age, 1),
        })


_positions_cache: list = []
_positions_last: float = 0


def _refresh_positions_cache():
    global _positions_cache, _positions_last
    cfg = load_config()
    all_positions = []
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
            for p in positions:
                p["account"] = acc.name
                p["platform"] = acc.account_type
            all_positions.extend(positions)
        except Exception:
            pass
    _positions_cache = all_positions
    _positions_last = _time.time()


@router.get("/api/accounts/positions")
def get_positions() -> JSONResponse:
    """Get positions. Uses live DxTrade WS data + cached MT5 data."""
    cache_age = _time.time() - _positions_last if _positions_last > 0 else 999
    if cache_age > 30:
        thread = threading.Thread(target=_refresh_positions_cache, daemon=True)
        thread.start()

    # Build fresh position list — use live DxTrade data instead of stale cache
    positions = [p for p in _positions_cache if p.get("platform") != "dxtrade"]
    try:
        from services import dxtrade_service
        cfg = load_config()
        for acc in cfg.accounts:
            if acc.account_type == "dxtrade" and acc.enabled:
                dxtrade_service.initialize(acc)
                dx_pos = dxtrade_service.get_active_trades()
                # Filter out closed/empty positions (vol=0, no ticket)
                dx_pos = [p for p in dx_pos if p.get("volume", 0) > 0 and p.get("ticket")]
                for p in dx_pos:
                    p["account"] = acc.name
                    p["platform"] = "dxtrade"
                positions.extend(dx_pos)
                break
    except Exception:
        # Fall back to cached DxTrade positions
        positions.extend([p for p in _positions_cache if p.get("platform") == "dxtrade"])

    return JSONResponse({"positions": positions})


_deals_cache: list = []
_deals_last: float = 0


def _load_deals_from_db():
    """Load persisted deals from database into cache. Instant on startup."""
    global _deals_cache, _deals_last
    from models.account import AccountDeal
    from db import get_session as _gs
    session = _gs()
    try:
        rows = session.query(AccountDeal).order_by(AccountDeal.closed_at.desc()).limit(500).all()
        _deals_cache = [{
            "ticket": r.ticket, "symbol": _normalize_symbol(r.symbol), "type": r.side,
            "volume": r.volume, "price": r.price, "profit": r.profit,
            "account": r.account_name, "platform": r.platform,
            "time": r.closed_at.isoformat() if r.closed_at else None,
        } for r in rows]
        if _deals_cache:
            _deals_last = _time.time()
    finally:
        session.close()


def _normalize_symbol(symbol: str) -> str:
    """Strip broker suffixes from symbol names (e.g. EURUSD.sim -> EURUSD, GBPUSD.r -> GBPUSD)."""
    if "." in symbol:
        base = symbol.split(".")[0]
        # Only strip if the base looks like a forex pair (6 alpha chars)
        if len(base) == 6 and base.isalpha():
            return base
    return symbol


def _upsert_deal(session, account_name: str, platform: str, deal: dict):
    """Insert a deal if it doesn't already exist (by ticket)."""
    from models.account import AccountDeal
    ticket = str(deal.get("ticket", ""))
    if not ticket:
        return
    existing = session.query(AccountDeal).filter(
        AccountDeal.ticket == ticket,
        AccountDeal.account_name == account_name,
    ).first()
    if existing:
        return
    closed_str = deal.get("time")
    closed_at = datetime.fromisoformat(closed_str) if closed_str else datetime.now(timezone.utc)
    if closed_at.tzinfo is None:
        closed_at = closed_at.replace(tzinfo=timezone.utc)
    session.add(AccountDeal(
        account_name=account_name,
        platform=platform,
        ticket=ticket,
        symbol=_normalize_symbol(deal.get("symbol", "")),
        side=deal.get("type", deal.get("side", "")),
        volume=float(deal.get("volume", 0)),
        price=float(deal.get("price", 0)),
        profit=float(deal.get("profit", 0)),
        closed_at=closed_at,
    ))


def _refresh_deals_cache():
    """Fetch deals from brokers, persist new ones to DB, update cache."""
    global _deals_cache, _deals_last
    log = _logging.getLogger("accounts.deals")
    from db import get_session as _gs

    new_deals = []

    # MT5 deals
    try:
        import MetaTrader5 as mt5
        cfg = load_config()
        acc = next((a for a in cfg.accounts if a.account_type == "mt5" and a.enabled), None)
        if acc:
            kwargs = {}
            if acc.login:
                try:
                    kwargs["login"] = int(acc.login)
                except ValueError:
                    kwargs["login"] = acc.login
            if acc.password:
                kwargs["password"] = acc.password
            if acc.server:
                kwargs["server"] = acc.server
            ok = mt5.initialize(acc.path, **kwargs) if acc.path else mt5.initialize(**kwargs)
            if ok:
                # MT5 API expects broker server time, not UTC.
                # OANDA Prop uses UTC+3 (EET/EEST). Add offset to cover recent deals.
                now_server = datetime.now(timezone.utc) + timedelta(hours=3)
                deals = mt5.history_deals_get(now_server - timedelta(days=90), now_server)
                if deals:
                    for d in deals:
                        if d.type > 1:
                            continue  # skip balance/credit operations
                        if d.entry == 0:
                            continue  # skip opening fills (entry=0), keep closing fills (entry=1)
                        # For closing deals (entry=1), the type is the closing action.
                        # The original trade direction is the opposite.
                        original_side = "sell" if d.type == 0 else "buy"
                        new_deals.append({
                            "ticket": str(d.ticket), "symbol": _normalize_symbol(d.symbol),
                            "type": original_side,
                            "volume": d.volume, "price": d.price,
                            "profit": d.profit, "account": acc.name,
                            "platform": "mt5",
                            # d.time is in broker server time (UTC+3), convert to real UTC
                            "time": (datetime.fromtimestamp(d.time, tz=timezone.utc) - timedelta(hours=3)).isoformat(),
                        })
                mt5.shutdown()
    except Exception as exc:
        log.error("MT5 deal history error: %s", exc)

    # DxTrade deals
    cfg = load_config()
    for dx_acc in cfg.accounts:
        if dx_acc.account_type == "dxtrade" and dx_acc.enabled:
            try:
                from services import dxtrade_service
                dx_deals = dxtrade_service.get_deal_history(
                    dx_acc, datetime.now(timezone.utc) - timedelta(days=90), datetime.now(timezone.utc)
                )
                for d in dx_deals:
                    # Skip test trades (early testing with 1000-unit volumes that had wrong PnL)
                    raw_vol = d.get("volume", 0)
                    if raw_vol < 0.001:  # less than 0.001 lots = likely a micro test
                        continue
                    d["platform"] = "dxtrade"
                new_deals.extend(dx_deals)
            except Exception as exc:
                log.error("DxTrade deal history error: %s", exc)

    # Persist new deals to DB
    if new_deals:
        session = _gs()
        try:
            for deal in new_deals:
                _upsert_deal(session, deal.get("account", ""), deal.get("platform", ""), deal)
            session.commit()
        except Exception:
            session.rollback()
        finally:
            session.close()

    # Reload full history from DB (includes old + new)
    _load_deals_from_db()
    log.info("Deals cache: %d entries (%d new from brokers)", len(_deals_cache), len(new_deals))


@router.get("/api/accounts/history")
def get_trade_history() -> JSONResponse:
    """Get deal history — loads from DB instantly, refreshes from brokers in background."""
    # First call: load from DB (instant)
    if _deals_last == 0:
        _load_deals_from_db()
    # Trigger background broker refresh periodically
    cache_age = _time.time() - _deals_last if _deals_last > 0 else 999
    if cache_age > 60:
        thread = threading.Thread(target=_refresh_deals_cache, daemon=True)
        thread.start()
    return JSONResponse({"deals": [d for d in _deals_cache if after_cutoff(d.get("time"))]})


# ---------------------------------------------------------------------------
# Trading controls
# ---------------------------------------------------------------------------


@router.post("/api/accounts/pause")
def pause_trading() -> JSONResponse:
    cfg = load_config()
    cfg.execution.paused = True
    save_config(cfg)
    return JSONResponse({"ok": True, "paused": True})


@router.post("/api/accounts/resume")
def resume_trading() -> JSONResponse:
    cfg = load_config()
    cfg.execution.paused = False
    save_config(cfg)
    return JSONResponse({"ok": True, "paused": False})


@router.post("/api/accounts/mode")
async def set_mode(request: Request) -> JSONResponse:
    payload = await request.json()
    mode = payload.get("mode", "paper")
    if mode not in ("paper", "live"):
        return JSONResponse({"error": "Mode must be 'paper' or 'live'"}, status_code=400)
    cfg = load_config()
    cfg.execution.mode = mode
    save_config(cfg)
    return JSONResponse({"ok": True, "mode": mode})


@router.get("/api/accounts/blackouts")
def get_blackouts() -> JSONResponse:
    cfg = load_config()
    return JSONResponse({
        "blackouts": [
            {"start": b.start, "end": b.end, "reason": b.reason, "recurring": b.recurring}
            for b in cfg.execution.blackouts
        ]
    })


@router.post("/api/accounts/blackouts")
async def add_blackout(request: Request) -> JSONResponse:
    payload = await request.json()
    cfg = load_config()
    cfg.execution.blackouts.append(TradingBlackout(
        start=payload.get("start", ""),
        end=payload.get("end", ""),
        reason=payload.get("reason", ""),
        recurring=payload.get("recurring", False),
    ))
    save_config(cfg)
    return JSONResponse({"ok": True, "count": len(cfg.execution.blackouts)})


@router.post("/api/accounts/blackouts/remove")
async def remove_blackout(request: Request) -> JSONResponse:
    payload = await request.json()
    idx = payload.get("index", -1)
    cfg = load_config()
    if 0 <= idx < len(cfg.execution.blackouts):
        cfg.execution.blackouts.pop(idx)
        save_config(cfg)
    return JSONResponse({"ok": True, "count": len(cfg.execution.blackouts)})


# ---------------------------------------------------------------------------
# Account snapshot history
# ---------------------------------------------------------------------------

import logging as _logging

def store_account_snapshots():
    """Store a balance/equity snapshot for each connected account."""
    from models.account import AccountSnapshot
    from db import get_session as _gs

    with _cache_lock:
        accs = list(_account_cache.get("accounts", []))
    if not accs:
        return
    now = datetime.now(timezone.utc)
    session = _gs()
    try:
        for a in accs:
            if not a.get("connected") or a.get("balance", 0) <= 0:
                continue
            session.add(AccountSnapshot(
                server_id=a.get("type", ""), account_name=a.get("name", ""),
                ts=now, balance=a["balance"], equity=a.get("equity", a["balance"]),
            ))
        session.commit()
    except Exception:
        session.rollback()
    finally:
        session.close()


async def account_snapshot_loop(stop_event):
    """Background: store account snapshots every 5 minutes."""
    import asyncio as _aio
    log = _logging.getLogger("accounts.snapshot")
    log.info("Account snapshot loop started.")
    # Load persisted deals from DB immediately (instant, no broker calls)
    try:
        _load_deals_from_db()
        log.info("Loaded %d deals from DB", len(_deals_cache))
    except Exception:
        pass

    # Initial refresh: accounts first, then positions, then deals — sequential, no contention
    await _aio.sleep(3)
    try:
        await _aio.to_thread(_refresh_account_cache)
        log.info("Initial account cache populated")
    except Exception as exc:
        log.error("Initial account refresh error: %s", exc)

    await _aio.sleep(2)
    try:
        await _aio.to_thread(_refresh_positions_cache)
    except Exception:
        pass

    await _aio.sleep(5)
    try:
        await _aio.to_thread(_refresh_deals_cache)
    except Exception:
        pass

    while not stop_event.is_set():
        try:
            # All refreshes sequential — prevents DxTrade connection contention
            if _time.time() - _account_cache.get("last_update", 0) > 60:
                await _aio.to_thread(_refresh_account_cache)
            await _aio.to_thread(store_account_snapshots)
            if _time.time() - _positions_last > 60:
                await _aio.to_thread(_refresh_positions_cache)
            if _time.time() - _deals_last > 60:
                await _aio.to_thread(_refresh_deals_cache)
        except Exception as exc:
            log.error("Snapshot error: %s", exc)
        try:
            await _aio.wait_for(stop_event.wait(), timeout=300)
        except _aio.TimeoutError:
            pass
    log.info("Account snapshot loop stopped.")


@router.get("/api/accounts/{account_name}/history")
def get_account_equity_history(account_name: str, days: int = 0) -> JSONResponse:
    from models.account import AccountSnapshot
    from db import get_session as _gs
    session = _gs()
    try:
        query = session.query(AccountSnapshot).filter(
            AccountSnapshot.account_name == account_name,
        )
        query = filter_query_after_cutoff(query, AccountSnapshot.ts)
        if days > 0:
            cutoff = datetime.now(timezone.utc) - timedelta(days=days)
            query = query.filter(AccountSnapshot.ts >= cutoff)
        rows = query.order_by(AccountSnapshot.ts.asc()).all()
        return JSONResponse({
            "account": account_name,
            "points": [{"time": r.ts.isoformat(), "balance": r.balance, "equity": r.equity} for r in rows],
        })
    finally:
        session.close()


@router.get("/api/accounts/{account_name}/trades")
def get_account_trade_list(account_name: str, limit: int = 50) -> JSONResponse:
    from models.signal import LiveTrade
    from db import get_session as _gs

    result_trades = []

    # Use broker deal history as single source of truth (no LiveTrade duplicates)
    from models.account import AccountDeal
    session = _gs()
    try:
        broker_deals = session.query(AccountDeal).filter(
            AccountDeal.account_name == account_name,
        )
        broker_deals = filter_query_after_cutoff(broker_deals, AccountDeal.closed_at)
        broker_deals = broker_deals.order_by(AccountDeal.closed_at.desc()).limit(limit).all()
        for d in broker_deals:
            result_trades.append({
                "symbol": _normalize_symbol(d.symbol), "side": d.side,
                "volume": d.volume, "pnl": d.profit,
                "exit_type": "tp" if (d.profit or 0) >= 0 else "sl",
                "platform": d.platform, "source": "broker",
                "closed_at": d.closed_at.isoformat() if d.closed_at else None,
            })
    finally:
        session.close()

    return JSONResponse({"trades": result_trades[:limit]})
