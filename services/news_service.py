"""Economic calendar and news event service.

Fetches forex economic calendar data and provides high-impact event
detection to prevent trading during volatile news releases.

Primary source: Forex Factory calendar (HTML scraping via httpx).
Fallback: manually maintained events if scraping fails.
"""

from __future__ import annotations

import asyncio
import logging
import re
from datetime import datetime, timedelta, timezone
from typing import List

import httpx

from config import load_config
from db import get_session
from models.news import NewsEvent

logger = logging.getLogger(__name__)

# Major forex economic events and their typical currencies
_KNOWN_HIGH_IMPACT = {
    "Non-Farm Employment Change": "USD",
    "FOMC Statement": "USD",
    "Federal Funds Rate": "USD",
    "CPI m/m": None,  # multiple currencies
    "CPI y/y": None,
    "GDP q/q": None,
    "GDP y/y": None,
    "Unemployment Rate": None,
    "Retail Sales m/m": None,
    "Interest Rate Decision": None,
    "ECB Press Conference": "EUR",
    "BOE Official Bank Rate": "GBP",
    "BOJ Policy Rate": "JPY",
    "RBA Cash Rate": "AUD",
    "RBNZ Official Cash Rate": "NZD",
    "BOC Overnight Rate": "CAD",
    "SNB Policy Rate": "CHF",
}


def _parse_forex_factory_html(html: str) -> list[dict]:
    """Parse Forex Factory calendar HTML into event dicts.

    This is a best-effort parser — Forex Factory changes their HTML regularly.
    Falls back gracefully on parse failures.
    """
    events = []

    # Find calendar rows — FF uses class "calendar__row" or similar
    row_pattern = re.compile(
        r'class="calendar__cell calendar__date[^"]*"[^>]*>(.*?)</td>.*?'
        r'class="calendar__cell calendar__time[^"]*"[^>]*>(.*?)</td>.*?'
        r'class="calendar__cell calendar__currency[^"]*"[^>]*>(.*?)</td>.*?'
        r'class="calendar__cell calendar__impact[^"]*"[^>]*>(.*?)</td>.*?'
        r'class="calendar__cell calendar__event[^"]*"[^>]*>(.*?)</td>.*?'
        r'class="calendar__cell calendar__actual[^"]*"[^>]*>(.*?)</td>.*?'
        r'class="calendar__cell calendar__forecast[^"]*"[^>]*>(.*?)</td>.*?'
        r'class="calendar__cell calendar__previous[^"]*"[^>]*>(.*?)</td>',
        re.DOTALL,
    )

    for match in row_pattern.finditer(html):
        date_raw, time_raw, currency, impact_html, event, actual, forecast, previous = match.groups()

        # Strip HTML tags
        currency = re.sub(r"<[^>]+>", "", currency).strip()
        event = re.sub(r"<[^>]+>", "", event).strip()
        actual = re.sub(r"<[^>]+>", "", actual).strip()
        forecast = re.sub(r"<[^>]+>", "", forecast).strip()
        previous = re.sub(r"<[^>]+>", "", previous).strip()

        # Parse impact from icon classes
        impact = "low"
        if "high" in impact_html.lower() or "icon--ff-impact-red" in impact_html:
            impact = "high"
        elif "medium" in impact_html.lower() or "icon--ff-impact-ora" in impact_html:
            impact = "medium"

        if not event or not currency:
            continue

        events.append({
            "title": event,
            "currency": currency.upper(),
            "impact": impact,
            "actual": actual or None,
            "forecast": forecast or None,
            "previous": previous or None,
        })

    return events


_FF_JSON_URL = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"

# Map country names to currency codes
_COUNTRY_CURRENCY = {
    "USD": "USD", "EUR": "EUR", "GBP": "GBP", "JPY": "JPY",
    "AUD": "AUD", "NZD": "NZD", "CAD": "CAD", "CHF": "CHF",
    "CNY": "CNY", "All": "ALL",
}


def _fetch_calendar_json(timeout: float = 15.0) -> list | None:
    """Fetch Forex Factory calendar from JSON API (no Cloudflare)."""
    try:
        resp = httpx.get(_FF_JSON_URL, timeout=timeout, follow_redirects=True, headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        })
        resp.raise_for_status()
        data = resp.json()
        return data if isinstance(data, list) else None
    except Exception as exc:
        logger.debug("Calendar JSON fetch failed: %s", exc)
        return None


def fetch_and_store_calendar():
    """Fetch calendar events from JSON API and upsert into the database."""
    events = _fetch_calendar_json()
    if not events:
        logger.info("Calendar fetch returned no data, using fallback")
        _store_fallback_events()
        return

    now = datetime.now(timezone.utc)
    session = get_session()
    try:
        stored = 0
        for raw in events:
            title = raw.get("title", "").strip()
            country = raw.get("country", "").strip()
            impact_raw = (raw.get("impact") or "").strip().lower()
            date_str = raw.get("date", "")

            if not title or not date_str:
                continue

            # Map impact
            impact = "low"
            if impact_raw in ("high", "holiday"):
                impact = "high"
            elif impact_raw == "medium":
                impact = "medium"

            # Map country to currency
            currency = _COUNTRY_CURRENCY.get(country, country[:3].upper() if country else "")

            # Parse date
            try:
                event_time = datetime.fromisoformat(date_str)
                if event_time.tzinfo is None:
                    event_time = event_time.replace(tzinfo=timezone.utc)
            except Exception:
                continue

            # Upsert
            existing = session.query(NewsEvent).filter(
                NewsEvent.title == title,
                NewsEvent.currency == currency,
                NewsEvent.event_time == event_time,
            ).first()

            if existing:
                existing.actual = raw.get("actual") or existing.actual
                existing.forecast = raw.get("forecast") or existing.forecast
                existing.previous = raw.get("previous") or existing.previous
                existing.fetched_at = now
            else:
                record = NewsEvent(
                    title=title,
                    currency=currency,
                    impact=impact,
                    event_time=event_time,
                    actual=raw.get("actual"),
                    forecast=raw.get("forecast"),
                    previous=raw.get("previous"),
                    source="forex_factory",
                    fetched_at=now,
                )
                session.add(record)
                stored += 1

        session.commit()
        logger.info("Stored %d new calendar events (%d total in feed)", stored, len(events))
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def _store_fallback_events():
    """Store well-known recurring events as a baseline when scraping fails.

    These are the major events that forex traders universally avoid:
    - US NFP (first Friday of month, 8:30 ET)
    - FOMC (scheduled 8 times/year)
    - ECB rate decisions (scheduled 8 times/year)
    """
    now = datetime.now(timezone.utc)
    session = get_session()
    try:
        # Check if we already have events this week
        week_start = now - timedelta(days=now.weekday())
        existing = session.query(NewsEvent).filter(
            NewsEvent.event_time >= week_start,
            NewsEvent.source == "fallback",
        ).count()
        if existing > 0:
            return  # Already have fallback events

        # Add known high-impact events for the week
        # NFP is typically first Friday of the month at 12:30 UTC (8:30 ET)
        for day_offset in range(7):
            day = week_start + timedelta(days=day_offset)
            if day.weekday() == 4 and day.day <= 7:  # First Friday
                nfp_time = day.replace(hour=12, minute=30, second=0, microsecond=0, tzinfo=timezone.utc)
                if nfp_time > now - timedelta(hours=1):
                    session.add(NewsEvent(
                        title="Non-Farm Employment Change",
                        currency="USD",
                        impact="high",
                        event_time=nfp_time,
                        source="fallback",
                        fetched_at=now,
                    ))
                    session.add(NewsEvent(
                        title="Unemployment Rate",
                        currency="USD",
                        impact="high",
                        event_time=nfp_time,
                        source="fallback",
                        fetched_at=now,
                    ))

        session.commit()
    except Exception:
        session.rollback()
    finally:
        session.close()


def get_upcoming_events(currency: str | None = None, hours_ahead: int = 4) -> list[dict]:
    """Query upcoming economic events within the next N hours."""
    now = datetime.now(timezone.utc)
    cutoff = now + timedelta(hours=hours_ahead)

    session = get_session()
    try:
        query = session.query(NewsEvent).filter(
            NewsEvent.event_time >= now,
            NewsEvent.event_time <= cutoff,
        ).order_by(NewsEvent.event_time.asc())

        if currency:
            query = query.filter(NewsEvent.currency == currency.upper())

        rows = query.all()
        return [
            {
                "id": e.id,
                "title": e.title,
                "currency": e.currency,
                "impact": e.impact,
                "event_time": e.event_time.isoformat(),
                "actual": e.actual,
                "forecast": e.forecast,
                "previous": e.previous,
                "source": e.source,
            }
            for e in rows
        ]
    finally:
        session.close()


def is_high_impact_window(symbol: str, buffer_minutes: int | None = None) -> tuple[bool, str]:
    """Check if a high-impact event is within the buffer window for either
    currency in the pair.

    Returns (is_blocked, reason).
    """
    cfg = load_config()
    if buffer_minutes is None:
        buffer_minutes = cfg.news.high_impact_buffer_minutes

    if not cfg.news.enabled:
        return False, ""

    # Extract currencies from pair (e.g., EURUSD → EUR, USD)
    base = symbol[:3].upper()
    quote = symbol[3:6].upper()

    now = datetime.now(timezone.utc)
    window_start = now - timedelta(minutes=buffer_minutes)
    window_end = now + timedelta(minutes=buffer_minutes)

    session = get_session()
    try:
        event = session.query(NewsEvent).filter(
            NewsEvent.impact == "high",
            NewsEvent.event_time >= window_start,
            NewsEvent.event_time <= window_end,
            NewsEvent.currency.in_([base, quote]),
        ).first()

        if event:
            return True, f"High-impact event: {event.title} ({event.currency}) at {event.event_time.strftime('%H:%M UTC')}"
        return False, ""
    finally:
        session.close()


def get_week_calendar() -> list[dict]:
    """Return all events for the current week."""
    now = datetime.now(timezone.utc)
    week_start = now - timedelta(days=now.weekday())
    week_start = week_start.replace(hour=0, minute=0, second=0, microsecond=0)
    week_end = week_start + timedelta(days=7)

    session = get_session()
    try:
        rows = session.query(NewsEvent).filter(
            NewsEvent.event_time >= week_start,
            NewsEvent.event_time <= week_end,
        ).order_by(NewsEvent.event_time.asc()).all()

        return [
            {
                "id": e.id,
                "title": e.title,
                "currency": e.currency,
                "impact": e.impact,
                "event_time": e.event_time.isoformat(),
                "actual": e.actual,
                "forecast": e.forecast,
                "previous": e.previous,
                "source": e.source,
            }
            for e in rows
        ]
    finally:
        session.close()


def get_events_for_symbol(symbol: str, hours_ahead: int = 24) -> list[dict]:
    """Get upcoming events relevant to a specific currency pair."""
    base = symbol[:3].upper()
    quote = symbol[3:6].upper()

    now = datetime.now(timezone.utc)
    cutoff = now + timedelta(hours=hours_ahead)

    session = get_session()
    try:
        rows = session.query(NewsEvent).filter(
            NewsEvent.event_time >= now,
            NewsEvent.event_time <= cutoff,
            NewsEvent.currency.in_([base, quote]),
        ).order_by(NewsEvent.event_time.asc()).all()

        return [
            {
                "id": e.id,
                "title": e.title,
                "currency": e.currency,
                "impact": e.impact,
                "event_time": e.event_time.isoformat(),
                "minutes_until": int((e.event_time - now).total_seconds() / 60),
                "actual": e.actual,
                "forecast": e.forecast,
                "previous": e.previous,
            }
            for e in rows
        ]
    finally:
        session.close()


def get_events_at_time(
    symbol: str,
    timestamp: datetime,
    hours_before: int = 1,
    hours_after: int = 4,
) -> list[dict]:
    """Get news events near a specific timestamp (for backtesting).

    Returns events that occurred within hours_before..hours_after of the timestamp,
    with minutes_until calculated relative to the given timestamp (not now).
    """
    base = symbol[:3].upper()
    quote = symbol[3:6].upper()

    window_start = timestamp - timedelta(hours=hours_before)
    window_end = timestamp + timedelta(hours=hours_after)

    session = get_session()
    try:
        rows = session.query(NewsEvent).filter(
            NewsEvent.event_time >= window_start,
            NewsEvent.event_time <= window_end,
            NewsEvent.currency.in_([base, quote]),
        ).order_by(NewsEvent.event_time.asc()).all()

        return [
            {
                "title": e.title,
                "currency": e.currency,
                "impact": e.impact,
                "event_time": e.event_time.isoformat(),
                "minutes_until": int((e.event_time - timestamp).total_seconds() / 60),
                "actual": e.actual,
                "forecast": e.forecast,
                "previous": e.previous,
                "already_released": e.event_time <= timestamp,
            }
            for e in rows
        ]
    finally:
        session.close()


def is_high_impact_at_time(
    symbol: str,
    timestamp: datetime,
    buffer_minutes: int = 30,
) -> tuple[bool, str]:
    """Check if a high-impact event is near a specific timestamp (for backtesting)."""
    base = symbol[:3].upper()
    quote = symbol[3:6].upper()

    window_start = timestamp - timedelta(minutes=buffer_minutes)
    window_end = timestamp + timedelta(minutes=buffer_minutes)

    session = get_session()
    try:
        event = session.query(NewsEvent).filter(
            NewsEvent.impact == "high",
            NewsEvent.event_time >= window_start,
            NewsEvent.event_time <= window_end,
            NewsEvent.currency.in_([base, quote]),
        ).first()

        if event:
            return True, f"{event.title} ({event.currency}) at {event.event_time.strftime('%H:%M UTC')}"
        return False, ""
    finally:
        session.close()


def seed_historical_calendar(start_date: datetime, end_date: datetime) -> int:
    """Seed the news_events table with known recurring high-impact events.

    Generates NFP (first Friday), FOMC (roughly every 6 weeks), ECB/BOE/BOJ
    rate decisions for the given date range. This gives backtests a baseline
    of major events to consider even without scraping historical pages.

    Returns count of events inserted.
    """
    session = get_session()
    now = datetime.now(timezone.utc)
    count = 0

    try:
        current = start_date
        while current <= end_date:
            year, month = current.year, current.month

            # --- NFP: first Friday of month at 12:30 UTC ---
            first_day = current.replace(day=1)
            # Find first Friday
            days_until_friday = (4 - first_day.weekday()) % 7
            nfp_date = first_day + timedelta(days=days_until_friday)
            nfp_time = nfp_date.replace(hour=12, minute=30, second=0, microsecond=0, tzinfo=timezone.utc)
            if start_date <= nfp_time <= end_date:
                _insert_event(session, "Non-Farm Employment Change", "USD", "high", nfp_time, now)
                _insert_event(session, "Unemployment Rate", "USD", "high", nfp_time, now)
                _insert_event(session, "Average Hourly Earnings m/m", "USD", "medium", nfp_time, now)
                count += 3

            # --- CPI: usually 2nd or 3rd week, ~12:30 UTC ---
            cpi_date = first_day + timedelta(days=(1 - first_day.weekday()) % 7 + 7)  # 2nd Tuesday approx
            cpi_time = cpi_date.replace(hour=12, minute=30, second=0, microsecond=0, tzinfo=timezone.utc)
            if start_date <= cpi_time <= end_date:
                _insert_event(session, "CPI m/m", "USD", "high", cpi_time, now)
                _insert_event(session, "Core CPI m/m", "USD", "high", cpi_time, now)
                count += 2

            # --- FOMC: 8 meetings/year, roughly every 6 weeks ---
            # Approximate: Jan, Mar, May, Jun, Jul, Sep, Nov, Dec
            fomc_months = {1, 3, 5, 6, 7, 9, 11, 12}
            if month in fomc_months:
                # Usually Wed of 3rd or 4th week at 18:00 UTC
                fomc_date = first_day + timedelta(days=(2 - first_day.weekday()) % 7 + 14)  # 3rd Wednesday
                fomc_time = fomc_date.replace(hour=18, minute=0, second=0, microsecond=0, tzinfo=timezone.utc)
                if start_date <= fomc_time <= end_date:
                    _insert_event(session, "FOMC Statement", "USD", "high", fomc_time, now)
                    _insert_event(session, "Federal Funds Rate", "USD", "high", fomc_time, now)
                    count += 2

            # --- ECB: usually Thursday, roughly every 6 weeks ---
            ecb_months = {1, 3, 4, 6, 7, 9, 10, 12}
            if month in ecb_months:
                ecb_date = first_day + timedelta(days=(3 - first_day.weekday()) % 7 + 7)  # 2nd Thursday
                ecb_time = ecb_date.replace(hour=12, minute=15, second=0, microsecond=0, tzinfo=timezone.utc)
                if start_date <= ecb_time <= end_date:
                    _insert_event(session, "ECB Interest Rate Decision", "EUR", "high", ecb_time, now)
                    _insert_event(session, "ECB Press Conference", "EUR", "high",
                                  ecb_time.replace(hour=12, minute=45), now)
                    count += 2

            # --- BOE: usually Thursday ---
            boe_months = {2, 3, 5, 6, 8, 9, 11, 12}
            if month in boe_months:
                boe_date = first_day + timedelta(days=(3 - first_day.weekday()) % 7 + 7)
                boe_time = boe_date.replace(hour=12, minute=0, second=0, microsecond=0, tzinfo=timezone.utc)
                if start_date <= boe_time <= end_date:
                    _insert_event(session, "BOE Official Bank Rate", "GBP", "high", boe_time, now)
                    count += 1

            # --- BOJ: usually Friday ---
            boj_months = {1, 3, 4, 6, 7, 9, 10, 12}
            if month in boj_months:
                boj_date = first_day + timedelta(days=(4 - first_day.weekday()) % 7 + 14)
                boj_time = boj_date.replace(hour=3, minute=0, second=0, microsecond=0, tzinfo=timezone.utc)
                if start_date <= boj_time <= end_date:
                    _insert_event(session, "BOJ Policy Rate", "JPY", "high", boj_time, now)
                    count += 1

            # Advance to next month
            if month == 12:
                current = current.replace(year=year + 1, month=1, day=1)
            else:
                current = current.replace(month=month + 1, day=1)

        session.commit()
        logger.info("Seeded %d historical calendar events", count)
        return count
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def _insert_event(session, title, currency, impact, event_time, fetched_at):
    """Insert event if it doesn't already exist."""
    existing = session.query(NewsEvent).filter(
        NewsEvent.title == title,
        NewsEvent.currency == currency,
        NewsEvent.event_time == event_time,
    ).first()
    if not existing:
        session.add(NewsEvent(
            title=title, currency=currency, impact=impact,
            event_time=event_time, source="historical_seed", fetched_at=fetched_at,
        ))


def count_events_in_range(start_date: datetime, end_date: datetime) -> int:
    """Count how many news events exist in a date range."""
    session = get_session()
    try:
        return session.query(NewsEvent).filter(
            NewsEvent.event_time >= start_date,
            NewsEvent.event_time <= end_date,
        ).count()
    finally:
        session.close()


async def news_collection_loop(stop_event: asyncio.Event) -> None:
    """Background loop that periodically fetches calendar data."""
    logger.info("News collection loop started.")
    while not stop_event.is_set():
        cfg = load_config()
        if cfg.news.enabled:
            try:
                await asyncio.to_thread(fetch_and_store_calendar)
            except Exception as exc:
                logger.error("News collection error: %s", exc)

        try:
            await asyncio.wait_for(
                stop_event.wait(), timeout=cfg.news.collection_interval_sec
            )
        except asyncio.TimeoutError:
            pass
    logger.info("News collection loop stopped.")
