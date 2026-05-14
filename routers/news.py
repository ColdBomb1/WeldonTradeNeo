from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

from config import load_config
from services import news_service, sentiment_service

router = APIRouter(tags=["news"])

BASE_DIR = Path(__file__).resolve().parent.parent
TEMPLATES = Jinja2Templates(directory=str(BASE_DIR / "templates"))


@router.get("/calendar")
def calendar_page(request: Request):
    return RedirectResponse(url="/market-context")


@router.get("/api/calendar")
def get_upcoming_events(
    currency: str | None = None,
    hours: int = 24,
) -> JSONResponse:
    events = news_service.get_upcoming_events(currency=currency, hours_ahead=hours)
    return JSONResponse({"count": len(events), "events": events})


@router.get("/api/calendar/week")
def get_week_calendar() -> JSONResponse:
    events = news_service.get_week_calendar()
    return JSONResponse({"count": len(events), "events": events})


@router.get("/api/calendar/check/{symbol}")
def check_high_impact(symbol: str) -> JSONResponse:
    blocked, reason = news_service.is_high_impact_window(symbol)
    events = news_service.get_events_for_symbol(symbol, hours_ahead=4)
    return JSONResponse({
        "symbol": symbol,
        "blocked": blocked,
        "reason": reason,
        "upcoming_events": events,
    })


@router.get("/api/sentiment/{symbol}")
def get_sentiment(symbol: str) -> JSONResponse:
    current = sentiment_service.get_current_sentiment(symbol)
    if current is None:
        # Compute fresh sentiment
        result = sentiment_service.compute_sentiment(symbol)
        return JSONResponse(result)
    return JSONResponse(current)


@router.get("/api/sentiment/{symbol}/history")
def get_sentiment_history(symbol: str, hours: int = 24) -> JSONResponse:
    history = sentiment_service.get_sentiment_history(symbol, hours=hours)
    return JSONResponse({"symbol": symbol, "history": history})


@router.post("/api/calendar/refresh")
def refresh_calendar() -> JSONResponse:
    """Trigger an immediate calendar fetch."""
    cfg = load_config()
    if not cfg.news.enabled:
        return JSONResponse({"error": "News/calendar is disabled"}, status_code=400)
    try:
        news_service.fetch_and_store_calendar()
        return JSONResponse({"status": "Calendar refreshed"})
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=500)


@router.post("/api/calendar/seed-historical")
async def seed_historical(request: Request) -> JSONResponse:
    """Seed historical economic calendar events for a date range."""
    from datetime import datetime, timezone
    payload = await request.json()
    start_str = payload.get("start_date", "")
    end_str = payload.get("end_date", "")
    if not start_str or not end_str:
        return JSONResponse({"error": "start_date and end_date required"}, status_code=400)
    start = datetime.fromisoformat(start_str)
    end = datetime.fromisoformat(end_str)
    if start.tzinfo is None:
        start = start.replace(tzinfo=timezone.utc)
    if end.tzinfo is None:
        end = end.replace(tzinfo=timezone.utc)
    try:
        count = news_service.seed_historical_calendar(start, end)
        return JSONResponse({"status": f"Seeded {count} events", "count": count})
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=500)
