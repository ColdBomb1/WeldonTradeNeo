from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from config import load_config
from db import get_session
from models.signal import LiveTrade
from services import trade_manager
from services.performance_archive import filter_query_after_cutoff

router = APIRouter(tags=["trades"])

BASE_DIR = Path(__file__).resolve().parent.parent
TEMPLATES = Jinja2Templates(directory=str(BASE_DIR / "templates"))


def _trade_to_dict(t: LiveTrade) -> dict:
    return {
        "id": t.id,
        "signal_id": t.signal_id,
        "symbol": t.symbol,
        "side": t.side,
        "entry_price": t.entry_price,
        "current_price": t.current_price,
        "stop_loss": t.stop_loss,
        "take_profit": t.take_profit,
        "volume": t.volume,
        "platform_ticket": t.platform_ticket,
        "platform": t.platform,
        "status": t.status,
        "pnl": t.pnl,
        "exit_price": t.exit_price,
        "exit_type": t.exit_type,
        "opened_at": t.opened_at.isoformat() if t.opened_at else None,
        "closed_at": t.closed_at.isoformat() if t.closed_at else None,
        "error_message": t.error_message,
    }


@router.get("/trades")
def trades_page(request: Request):
    return RedirectResponse(url="/operations")


@router.get("/api/trades")
def get_trades(
    status: str | None = None,
    symbol: str | None = None,
    limit: int = 100,
) -> JSONResponse:
    from models.account import AccountDeal
    session = get_session()
    try:
        if status == "closed":
            # For closed trades, use AccountDeal (broker truth) as the single source.
            # This avoids duplicates from LiveTrade records with calculated PnL.
            query = session.query(AccountDeal).order_by(AccountDeal.closed_at.desc())
            query = filter_query_after_cutoff(query, AccountDeal.closed_at)
            if symbol:
                query = query.filter(AccountDeal.symbol == symbol)
            deals = query.limit(min(limit, 500)).all()
            result = [{
                "id": None, "signal_id": None,
                "symbol": d.symbol, "side": d.side,
                "entry_price": d.price, "current_price": None,
                "stop_loss": None, "take_profit": None,
                "volume": d.volume, "platform_ticket": d.ticket,
                "platform": d.platform, "status": "closed",
                "pnl": d.profit, "exit_price": None,
                "exit_type": "tp" if (d.profit or 0) >= 0 else "sl",
                "opened_at": None,
                "closed_at": d.closed_at.isoformat() if d.closed_at else None,
                "error_message": None,
            } for d in deals]
        else:
            # For open/pending/error trades, use LiveTrade
            query = session.query(LiveTrade).order_by(LiveTrade.opened_at.desc())
            if status:
                query = query.filter(LiveTrade.status == status)
            if symbol:
                query = query.filter(LiveTrade.symbol == symbol)
            rows = query.limit(min(limit, 500)).all()
            result = [_trade_to_dict(t) for t in rows]

        return JSONResponse({
            "count": len(result),
            "trades": result,
        })
    finally:
        session.close()

@router.get("/api/trades/summary")
def get_trade_summary() -> JSONResponse:
    session = get_session()
    try:
        today_start = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)

        open_trades = session.query(LiveTrade).filter(LiveTrade.status == "open").all()
        closed_today = session.query(LiveTrade).filter(
            LiveTrade.status == "closed",
            LiveTrade.closed_at >= today_start,
        ).all()
        all_closed_query = session.query(LiveTrade).filter(LiveTrade.status == "closed")
        all_closed_query = filter_query_after_cutoff(all_closed_query, LiveTrade.closed_at)
        all_closed = all_closed_query.all()

        unrealized_pnl = sum(t.pnl or 0 for t in open_trades)
        daily_pnl = sum(t.pnl or 0 for t in closed_today)
        total_pnl = sum(t.pnl or 0 for t in all_closed)
        total_trades = len(all_closed)
        winning = sum(1 for t in all_closed if t.pnl and t.pnl > 0)
        win_rate = winning / total_trades if total_trades > 0 else 0

        return JSONResponse({
            "open_positions": len(open_trades),
            "unrealized_pnl": round(unrealized_pnl, 2),
            "daily_pnl": round(daily_pnl, 2),
            "daily_trades": len(closed_today),
            "total_pnl": round(total_pnl, 2),
            "total_trades": total_trades,
            "win_rate": round(win_rate, 4),
        })
    finally:
        session.close()


@router.get("/api/trades/{trade_id}")
def get_trade(trade_id: int) -> JSONResponse:
    session = get_session()
    try:
        t = session.query(LiveTrade).filter(LiveTrade.id == trade_id).first()
        if t is None:
            return JSONResponse({"error": "Trade not found"}, status_code=404)
        return JSONResponse(_trade_to_dict(t))
    finally:
        session.close()


@router.post("/api/trades/{trade_id}/close")
def close_trade(trade_id: int) -> JSONResponse:
    result = trade_manager.close_trade(trade_id, reason="manual")
    if "error" in result:
        return JSONResponse(result, status_code=400)
    return JSONResponse(result)


class ModifyRequest(BaseModel):
    sl: float | None = None
    tp: float | None = None


@router.post("/api/trades/{trade_id}/modify")
def modify_trade(trade_id: int, body: ModifyRequest) -> JSONResponse:
    session = get_session()
    try:
        t = session.query(LiveTrade).filter(LiveTrade.id == trade_id).first()
        if t is None:
            return JSONResponse({"error": "Trade not found"}, status_code=404)
        if t.status != "open":
            return JSONResponse({"error": f"Trade is {t.status}"}, status_code=400)

        if body.sl is not None:
            t.stop_loss = body.sl
        if body.tp is not None:
            t.take_profit = body.tp

        # If live MT5 trade, modify on broker
        if t.platform == "mt5" and t.platform_ticket:
            from services import mt5_trade_service
            account = trade_manager._get_mt5_account()
            if account:
                result = mt5_trade_service.modify_position(
                    account, t.platform_ticket, t.stop_loss, t.take_profit
                )
                if "error" in result:
                    session.rollback()
                    return JSONResponse(result, status_code=400)

        session.commit()
        return JSONResponse(_trade_to_dict(t))
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
