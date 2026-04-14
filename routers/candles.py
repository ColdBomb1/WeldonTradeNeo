from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy import func

from config import load_config
from db import get_session
from models.candle import Candle

router = APIRouter(tags=["candles"])

BASE_DIR = Path(__file__).resolve().parent.parent
TEMPLATES = Jinja2Templates(directory=str(BASE_DIR / "templates"))


@router.get("/candles")
def candles_page(request: Request):
    cfg = load_config()
    return TEMPLATES.TemplateResponse(
        "candles.html",
        {
            "request": request,
            "symbols": cfg.symbols,
            "timeframes": cfg.candle_timeframes,
        },
    )


@router.get("/api/candles")
def get_candles(
    symbol: str,
    timeframe: str = "1h",
    start: str | None = None,
    end: str | None = None,
    limit: int = 500,
) -> JSONResponse:
    session = get_session()
    query = (
        session.query(Candle)
        .filter(Candle.symbol == symbol, Candle.timeframe == timeframe)
    )

    if start:
        dt_start = datetime.fromisoformat(start)
        if dt_start.tzinfo is None:
            dt_start = dt_start.replace(tzinfo=timezone.utc)
        query = query.filter(Candle.ts >= dt_start)

    if end:
        dt_end = datetime.fromisoformat(end)
        if dt_end.tzinfo is None:
            dt_end = dt_end.replace(tzinfo=timezone.utc)
        query = query.filter(Candle.ts <= dt_end)

    rows = query.order_by(Candle.ts.asc()).limit(min(limit, 5000)).all()
    session.close()

    return JSONResponse({
        "symbol": symbol,
        "timeframe": timeframe,
        "count": len(rows),
        "candles": [
            {
                "time": row.ts.isoformat(),
                "open": row.open,
                "high": row.high,
                "low": row.low,
                "close": row.close,
                "volume": row.volume,
            }
            for row in rows
        ],
    })


@router.get("/api/candles/symbols")
def get_available_symbols() -> JSONResponse:
    session = get_session()
    rows = session.query(Candle.symbol).distinct().order_by(Candle.symbol).all()
    session.close()
    return JSONResponse({"symbols": [row[0] for row in rows]})


@router.get("/api/candles/timeframes")
def get_available_timeframes(symbol: str) -> JSONResponse:
    session = get_session()
    rows = (
        session.query(Candle.timeframe)
        .filter(Candle.symbol == symbol)
        .distinct()
        .all()
    )
    session.close()
    return JSONResponse({"symbol": symbol, "timeframes": [row[0] for row in rows]})


@router.get("/api/candles/stats")
def get_candle_stats() -> JSONResponse:
    session = get_session()
    rows = (
        session.query(
            Candle.symbol,
            Candle.timeframe,
            func.count(Candle.id),
            func.min(Candle.ts),
            func.max(Candle.ts),
        )
        .group_by(Candle.symbol, Candle.timeframe)
        .order_by(Candle.symbol, Candle.timeframe)
        .all()
    )
    session.close()
    return JSONResponse({
        "stats": [
            {
                "symbol": row[0],
                "timeframe": row[1],
                "count": row[2],
                "earliest": row[3].isoformat() if row[3] else None,
                "latest": row[4].isoformat() if row[4] else None,
            }
            for row in rows
        ]
    })
