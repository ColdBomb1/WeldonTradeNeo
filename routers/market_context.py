from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy import func

from config import load_config
from db import get_session
from models.candle import Candle
from models.cot import CotPosition
from models.news import SentimentScore
from services import cot_service, market_context_service, news_service

router = APIRouter(tags=["market-context"])

BASE_DIR = Path(__file__).resolve().parent.parent
TEMPLATES = Jinja2Templates(directory=str(BASE_DIR / "templates"))


def _iso(value: datetime | None) -> str | None:
    return value.isoformat() if value else None


def _parse_dt(value: datetime | str | None) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        dt = value
    else:
        dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _age_minutes(value: datetime | str | None, now: datetime) -> float | None:
    dt = _parse_dt(value)
    if dt is None:
        return None
    return round(max(0.0, (now - dt).total_seconds() / 60.0), 1)


def _symbol_currencies(symbols: list[str]) -> set[str]:
    currencies: set[str] = set()
    for symbol in symbols:
        base, quote = market_context_service.split_symbol(symbol)
        for currency in (base, quote):
            if currency in market_context_service.MAJOR_CURRENCIES:
                currencies.add(currency)
    return currencies


def _latest_cot_rows(currencies: set[str], now: datetime) -> dict[str, CotPosition]:
    if not currencies:
        return {}
    session = get_session()
    try:
        rows: dict[str, CotPosition] = {}
        for currency in sorted(currencies):
            row = (
                session.query(CotPosition)
                .filter(
                    CotPosition.currency == currency,
                    CotPosition.available_at <= now,
                )
                .order_by(CotPosition.available_at.desc())
                .first()
            )
            if row:
                rows[currency] = row
        return rows
    finally:
        session.close()


def _cot_table(rows: dict[str, CotPosition], now: datetime) -> list[dict]:
    result = []
    for currency, row in sorted(rows.items()):
        result.append(
            {
                "currency": currency,
                "market_name": row.market_name,
                "report_date": _iso(row.report_date),
                "available_at": _iso(row.available_at),
                "age_days": round(max(0.0, (now - _parse_dt(row.available_at)).total_seconds() / 86400.0), 1),
                "open_interest": row.open_interest,
                "noncommercial_net": row.noncommercial_net,
                "noncommercial_net_pct": row.noncommercial_net_pct,
                "source": row.source,
            }
        )
    return result


def _latest_sentiment(symbols: list[str], now: datetime) -> list[dict]:
    session = get_session()
    try:
        result = []
        for symbol in symbols:
            row = (
                session.query(SentimentScore)
                .filter(SentimentScore.symbol == symbol.upper())
                .order_by(SentimentScore.ts.desc())
                .first()
            )
            if not row:
                continue
            result.append(
                {
                    "symbol": symbol.upper(),
                    "score": row.score,
                    "source": row.source,
                    "ts": _iso(row.ts),
                    "age_minutes": _age_minutes(row.ts, now),
                    "details": row.details or {},
                }
            )
        return result
    finally:
        session.close()


@router.get("/market-context")
def market_context_page(request: Request):
    cfg = load_config()
    return TEMPLATES.TemplateResponse(
        "market_context.html",
        {
            "request": request,
            "symbols": cfg.symbols,
            "timeframes": cfg.candle_timeframes,
            "default_timeframe": "15m" if "15m" in cfg.candle_timeframes else (cfg.candle_timeframes[0] if cfg.candle_timeframes else "1h"),
            "default_source": cfg.default_source,
            "news_enabled": cfg.news.enabled,
            "sentiment_enabled": cfg.news.sentiment_enabled,
        },
    )


@router.get("/api/market-context/summary")
def market_context_summary(timeframe: str = "15m") -> JSONResponse:
    cfg = load_config()
    now = datetime.now(timezone.utc)
    symbols = [s.upper() for s in cfg.symbols]
    selected_timeframe = timeframe if timeframe in cfg.candle_timeframes else (
        cfg.candle_timeframes[0] if cfg.candle_timeframes else timeframe
    )

    candle_stats: list[dict] = []
    candles_by_symbol: dict[str, list[dict]] = {}
    session = get_session()
    try:
        for symbol in symbols:
            for tf in cfg.candle_timeframes:
                count, first_ts, latest_ts = (
                    session.query(func.count(Candle.id), func.min(Candle.ts), func.max(Candle.ts))
                    .filter(Candle.symbol == symbol, Candle.timeframe == tf)
                    .one()
                )
                latest = None
                if latest_ts is not None:
                    latest = (
                        session.query(Candle)
                        .filter(Candle.symbol == symbol, Candle.timeframe == tf, Candle.ts == latest_ts)
                        .order_by(Candle.id.desc())
                        .first()
                    )
                candle_stats.append(
                    {
                        "symbol": symbol,
                        "timeframe": tf,
                        "count": int(count or 0),
                        "first": _iso(first_ts),
                        "latest": _iso(latest_ts),
                        "age_minutes": _age_minutes(latest_ts, now),
                        "source": latest.source if latest else None,
                        "close": latest.close if latest else None,
                    }
                )

            rows = (
                session.query(Candle)
                .filter(Candle.symbol == symbol, Candle.timeframe == selected_timeframe)
                .order_by(Candle.ts.desc())
                .limit(128)
                .all()
            )
            candles_by_symbol[symbol] = [
                {
                    "time": row.ts.isoformat(),
                    "open": row.open,
                    "high": row.high,
                    "low": row.low,
                    "close": row.close,
                    "volume": row.volume,
                    "source": row.source,
                }
                for row in reversed(rows)
            ]
    finally:
        session.close()

    enriched = market_context_service.enrich_candles_by_symbol(
        candles_by_symbol,
        include_news=cfg.news.enabled,
        include_strength=True,
        include_cot=False,
        include_sentiment=True,
    )

    cot_rows = _latest_cot_rows(_symbol_currencies(symbols), now)
    symbol_context: list[dict] = []
    for symbol in symbols:
        candles = enriched.get(symbol) or []
        if not candles:
            symbol_context.append({"symbol": symbol, "time": None, "close": None, "context": {}})
            continue
        latest = candles[-1]
        ctx = dict(latest.get("context") or {})
        base, quote = market_context_service.split_symbol(symbol)
        base_cot = cot_rows.get(base)
        quote_cot = cot_rows.get(quote)
        if base_cot or quote_cot:
            base_net = float(base_cot.noncommercial_net_pct) if base_cot else 0.0
            quote_net = float(quote_cot.noncommercial_net_pct) if quote_cot else 0.0
            ctx["cot_bias"] = round(base_net - quote_net, 3)
            ctx["base_cot_net_pct"] = round(base_net, 3) if base_cot else None
            ctx["quote_cot_net_pct"] = round(quote_net, 3) if quote_cot else None

        symbol_context.append(
            {
                "symbol": symbol,
                "time": latest.get("time"),
                "source": latest.get("source"),
                "close": latest.get("close"),
                "session": ctx.get("session"),
                "session_quality": ctx.get("session_quality"),
                "currency_strength_bias": ctx.get("currency_strength_bias"),
                "cot_bias": ctx.get("cot_bias"),
                "base_cot_net_pct": ctx.get("base_cot_net_pct"),
                "quote_cot_net_pct": ctx.get("quote_cot_net_pct"),
                "high_impact_near": ctx.get("high_impact_near"),
                "minutes_to_high_impact": ctx.get("minutes_to_high_impact"),
                "next_high_impact_title": ctx.get("next_high_impact_title"),
                "next_high_impact_currency": ctx.get("next_high_impact_currency"),
                "next_event_minutes": ctx.get("next_event_minutes"),
                "sentiment_score": ctx.get("sentiment_score"),
                "sentiment_age_minutes": ctx.get("sentiment_age_minutes"),
            }
        )

    events = news_service.get_upcoming_events(hours_ahead=24) if cfg.news.enabled else []

    return JSONResponse(
        {
            "generated_at": now.isoformat(),
            "symbols": symbols,
            "timeframe": selected_timeframe,
            "default_source": cfg.default_source,
            "news_enabled": cfg.news.enabled,
            "sentiment_enabled": cfg.news.sentiment_enabled,
            "candle_stats": candle_stats,
            "symbol_context": symbol_context,
            "events": events[:80],
            "cot": _cot_table(cot_rows, now),
            "sentiment": _latest_sentiment(symbols, now),
        }
    )


@router.post("/api/market-context/cot/sync")
def sync_cot() -> JSONResponse:
    now = datetime.now(timezone.utc)
    start = now - timedelta(days=400)
    try:
        stored = cot_service.sync_legacy_currency_futures(start, now)
        return JSONResponse({"ok": True, "stored": stored, "start": start.isoformat(), "end": now.isoformat()})
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=500)
