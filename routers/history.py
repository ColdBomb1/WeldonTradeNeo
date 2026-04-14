from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from fastapi.templating import Jinja2Templates

from config import load_config
from db import get_session
from models import AccountSnapshot, HistoryTick, Tick, TradeExecution, TradeRecommendation, TradeUpdate

router = APIRouter(tags=["history"])

BASE_DIR = Path(__file__).resolve().parent.parent
TEMPLATES = Jinja2Templates(directory=str(BASE_DIR / "templates"))


def _parse_dt(value: str | None) -> datetime:
    if not value:
        return datetime.now(tz=timezone.utc)
    dt = datetime.fromisoformat(value)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


@router.get("/charts")
def charts(request: Request, symbol: str | None = None, timeframe: str = "live"):
    cfg = load_config()
    selected_symbol = symbol or (cfg.symbols[0] if cfg.symbols else "")
    return TEMPLATES.TemplateResponse(
        "charts.html",
        {
            "request": request,
            "symbols": cfg.symbols,
            "timeframes": ["live"] + list(cfg.candle_timeframes),
            "selected_symbol": selected_symbol,
            "selected_timeframe": timeframe,
        },
    )


@router.get("/api/history-data")
def history_data(symbol: str, timeframe: str = "live") -> JSONResponse:
    session = get_session()
    points = []
    if timeframe == "live":
        rows = (
            session.query(Tick)
            .filter(Tick.symbol == symbol)
            .order_by(Tick.ts.desc())
            .limit(500)
            .all()
        )
    else:
        rows = (
            session.query(HistoryTick)
            .filter(HistoryTick.symbol == symbol, HistoryTick.timeframe == timeframe)
            .order_by(HistoryTick.ts.desc())
            .limit(2000)
            .all()
        )

    for row in reversed(rows):
        points.append({"time": row.ts.isoformat(), "bid": row.bid, "ask": row.ask})

    recs = (
        session.query(TradeRecommendation)
        .filter(TradeRecommendation.symbol == symbol)
        .order_by(TradeRecommendation.created_at.desc())
        .limit(50)
        .all()
    )
    executions = (
        session.query(TradeExecution)
        .filter(TradeExecution.symbol == symbol)
        .order_by(TradeExecution.created_at.desc())
        .limit(50)
        .all()
    )
    updates = (
        session.query(TradeUpdate)
        .filter(TradeUpdate.symbol == symbol)
        .order_by(TradeUpdate.updated_at.desc())
        .limit(50)
        .all()
    )
    session.close()

    return JSONResponse(
        {
            "symbol": symbol,
            "timeframe": timeframe,
            "points": points,
            "recommendations": [
                {"time": rec.created_at.isoformat(), "side": rec.side, "price": rec.price, "status": rec.status}
                for rec in recs
            ],
            "executions": [
                {"time": ex.created_at.isoformat(), "side": ex.side, "price": ex.price, "status": ex.status, "server_id": ex.server_id}
                for ex in executions
            ],
            "trade_updates": [
                {"time": tu.updated_at.isoformat(), "symbol": tu.symbol, "side": tu.side, "price": tu.price, "status": tu.status, "server_id": tu.server_id}
                for tu in updates
            ],
        }
    )


@router.get("/api/account-performance")
def account_performance(server_id: str | None = None) -> JSONResponse:
    session = get_session()
    query = session.query(AccountSnapshot).order_by(AccountSnapshot.ts.asc())
    if server_id:
        query = query.filter(AccountSnapshot.server_id == server_id)
    rows = query.limit(2000).all()
    session.close()

    return JSONResponse(
        {
            "server_id": server_id,
            "points": [
                {"time": row.ts.isoformat(), "balance": row.balance, "equity": row.equity}
                for row in rows
            ],
        }
    )


@router.post("/api/recommendations")
async def create_recommendation(request: Request) -> JSONResponse:
    payload = await request.json()

    session = get_session()
    rec = TradeRecommendation(
        symbol=payload.get("symbol", ""),
        side=payload.get("side", ""),
        price=float(payload.get("price", 0)),
        created_at=_parse_dt(payload.get("timestamp")),
        status=payload.get("status", "new"),
        details=payload.get("details"),
    )
    session.add(rec)
    session.commit()
    session.close()

    return JSONResponse({"ok": True})
