"""Point-in-time market context for live scans and research backtests."""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Iterable

from db import get_session
from models.cot import CotPosition
from models.news import NewsEvent, SentimentScore
from services import cot_service


MAJOR_CURRENCIES = {"USD", "EUR", "GBP", "JPY", "AUD", "NZD", "CAD", "CHF"}


def parse_dt(value) -> datetime:
    if isinstance(value, datetime):
        dt = value
    else:
        dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def split_symbol(symbol: str) -> tuple[str, str]:
    cleaned = str(symbol or "").upper().replace("/", "").replace("=X", "")
    return cleaned[:3], cleaned[3:6]


def session_context(ts: datetime) -> dict:
    dt = parse_dt(ts).astimezone(timezone.utc)
    hour = dt.hour
    if 13 <= hour < 16:
        session = "london_ny_overlap"
        quality = 1.0
    elif 7 <= hour < 13:
        session = "london"
        quality = 0.85
    elif 16 <= hour < 21:
        session = "new_york"
        quality = 0.75
    elif 0 <= hour < 7:
        session = "asia"
        quality = 0.55
    else:
        session = "off_hours"
        quality = 0.25

    return {
        "hour_utc": hour,
        "day_of_week": dt.weekday(),
        "session": session,
        "session_quality": quality,
        "is_weekend_close": dt.weekday() == 4 and hour >= 20,
    }


def enrich_candles_by_symbol(
    candles_by_symbol: dict[str, list[dict]],
    *,
    include_news: bool = True,
    include_strength: bool = True,
    include_cot: bool = False,
    include_sentiment: bool = False,
) -> dict[str, list[dict]]:
    """Attach context dicts to candle rows.

    The function mutates and returns the supplied candle dictionaries. It only
    uses information available at or before each candle timestamp except for
    scheduled future event times, which are known ahead of time.
    """
    if not candles_by_symbol:
        return candles_by_symbol

    events_by_currency: dict[str, list[NewsEvent]] = {}
    if include_news:
        events_by_currency = _load_events_for_candles(candles_by_symbol)

    strength_by_time: dict[str, dict[str, float]] = {}
    if include_strength:
        strength_by_time = _currency_strength_by_time(candles_by_symbol)

    cot_by_currency: dict[str, list[CotPosition]] = {}
    if include_cot:
        cot_by_currency = _load_cot_for_candles(candles_by_symbol)

    sentiment_by_symbol: dict[str, dict] = {}
    if include_sentiment:
        sentiment_by_symbol = _latest_sentiment_by_symbol(candles_by_symbol)

    for symbol, candles in candles_by_symbol.items():
        base, quote = split_symbol(symbol)
        symbol_sentiment = sentiment_by_symbol.get(symbol.upper())
        for candle in candles:
            ts = parse_dt(candle["time"])
            ctx = dict(candle.get("context") or {})
            ctx.update(session_context(ts))

            if include_news:
                relevant_events = (
                    events_by_currency.get(base, [])
                    + events_by_currency.get(quote, [])
                )
                ctx.update(event_context(ts, relevant_events))

            if include_strength:
                strengths = strength_by_time.get(candle["time"]) or {}
                base_strength = strengths.get(base, 0.0)
                quote_strength = strengths.get(quote, 0.0)
                ctx["base_strength"] = round(base_strength, 4)
                ctx["quote_strength"] = round(quote_strength, 4)
                ctx["currency_strength_bias"] = round(base_strength - quote_strength, 4)

            if include_cot:
                ctx.update(cot_context(ts, cot_by_currency.get(base, []), cot_by_currency.get(quote, [])))

            if symbol_sentiment:
                ctx["sentiment_score"] = symbol_sentiment["score"]
                ctx["sentiment_age_minutes"] = symbol_sentiment["age_minutes"]

            candle["context"] = ctx
    return candles_by_symbol


def event_context(ts: datetime, events: Iterable[NewsEvent]) -> dict:
    timestamp = parse_dt(ts)
    nearest_high_before = None
    nearest_high_after = None
    nearest_any_after = None
    released_surprises: list[float] = []

    for event in events:
        event_time = parse_dt(event.event_time)
        delta_min = int((event_time - timestamp).total_seconds() / 60)
        impact = (event.impact or "").lower()
        if nearest_any_after is None or (0 <= delta_min < nearest_any_after["minutes_until"]):
            if delta_min >= 0:
                nearest_any_after = _event_summary(event, delta_min)

        if impact == "high":
            if delta_min >= 0:
                if nearest_high_after is None or delta_min < nearest_high_after["minutes_until"]:
                    nearest_high_after = _event_summary(event, delta_min)
            else:
                minutes_since = abs(delta_min)
                if nearest_high_before is None or minutes_since < nearest_high_before["minutes_since"]:
                    nearest_high_before = _event_summary(event, delta_min)

        if event_time <= timestamp:
            surprise = _numeric_surprise(event.actual, event.forecast)
            if surprise is not None:
                released_surprises.append(surprise)

    minutes_to_high = nearest_high_after["minutes_until"] if nearest_high_after else None
    minutes_since_high = nearest_high_before["minutes_since"] if nearest_high_before else None
    near_high = (
        (minutes_to_high is not None and minutes_to_high <= 60)
        or (minutes_since_high is not None and minutes_since_high <= 45)
    )

    return {
        "minutes_to_high_impact": minutes_to_high,
        "minutes_since_high_impact": minutes_since_high,
        "high_impact_near": near_high,
        "next_high_impact_title": nearest_high_after["title"] if nearest_high_after else None,
        "next_high_impact_currency": nearest_high_after["currency"] if nearest_high_after else None,
        "next_event_minutes": nearest_any_after["minutes_until"] if nearest_any_after else None,
        "recent_event_surprise": round(sum(released_surprises[-3:]), 4) if released_surprises else None,
    }


def cot_context(ts: datetime, base_rows: Iterable[CotPosition], quote_rows: Iterable[CotPosition]) -> dict:
    timestamp = parse_dt(ts)
    base = _latest_cot_at(timestamp, base_rows)
    quote = _latest_cot_at(timestamp, quote_rows)
    base_net_pct = float(base.noncommercial_net_pct) if base else 0.0
    quote_net_pct = float(quote.noncommercial_net_pct) if quote else 0.0
    base_age = _age_days(timestamp, base.available_at) if base else None
    quote_age = _age_days(timestamp, quote.available_at) if quote else None
    return {
        "base_cot_net_pct": round(base_net_pct, 3) if base else None,
        "quote_cot_net_pct": round(quote_net_pct, 3) if quote else None,
        "cot_bias": round(base_net_pct - quote_net_pct, 3) if base or quote else None,
        "base_cot_age_days": base_age,
        "quote_cot_age_days": quote_age,
        "base_cot_report_date": base.report_date.date().isoformat() if base else None,
        "quote_cot_report_date": quote.report_date.date().isoformat() if quote else None,
    }


def _event_summary(event: NewsEvent, delta_min: int) -> dict:
    return {
        "title": event.title,
        "currency": event.currency,
        "impact": event.impact,
        "minutes_until": max(0, delta_min),
        "minutes_since": max(0, -delta_min),
    }


def _numeric_surprise(actual: str | None, forecast: str | None) -> float | None:
    actual_num = _extract_number(actual)
    forecast_num = _extract_number(forecast)
    if actual_num is None or forecast_num is None:
        return None
    scale = max(abs(forecast_num), 1.0)
    return (actual_num - forecast_num) / scale


def _extract_number(value: str | None) -> float | None:
    if value is None:
        return None
    text = str(value).replace(",", "").replace("%", "").strip()
    if not text:
        return None
    multiplier = 1.0
    suffix = text[-1:].upper()
    if suffix == "K":
        multiplier = 1_000.0
        text = text[:-1]
    elif suffix == "M":
        multiplier = 1_000_000.0
        text = text[:-1]
    elif suffix == "B":
        multiplier = 1_000_000_000.0
        text = text[:-1]
    try:
        return float(text) * multiplier
    except ValueError:
        return None


def _load_events_for_candles(candles_by_symbol: dict[str, list[dict]]) -> dict[str, list[NewsEvent]]:
    times = [
        parse_dt(c["time"])
        for candles in candles_by_symbol.values()
        for c in candles
        if c.get("time")
    ]
    if not times:
        return {}
    currencies = set()
    for symbol in candles_by_symbol:
        base, quote = split_symbol(symbol)
        currencies.update(ccy for ccy in (base, quote) if ccy in MAJOR_CURRENCIES)
    if not currencies:
        return {}

    start = min(times) - timedelta(hours=4)
    end = max(times) + timedelta(hours=24)
    session = get_session()
    try:
        rows = (
            session.query(NewsEvent)
            .filter(
                NewsEvent.event_time >= start,
                NewsEvent.event_time <= end,
                NewsEvent.currency.in_(sorted(currencies)),
            )
            .order_by(NewsEvent.event_time.asc())
            .all()
        )
        by_currency: dict[str, list[NewsEvent]] = defaultdict(list)
        for row in rows:
            by_currency[row.currency].append(row)
        return dict(by_currency)
    finally:
        session.close()


def _load_cot_for_candles(candles_by_symbol: dict[str, list[dict]]) -> dict[str, list[CotPosition]]:
    times = [
        parse_dt(c["time"])
        for candles in candles_by_symbol.values()
        for c in candles
        if c.get("time")
    ]
    if not times:
        return {}
    currencies = set()
    for symbol in candles_by_symbol:
        base, quote = split_symbol(symbol)
        currencies.update(ccy for ccy in (base, quote) if ccy in MAJOR_CURRENCIES)
    if not currencies:
        return {}

    start = min(times) - timedelta(days=120)
    end = max(times)
    rows = _query_cot_rows(currencies, start, end)
    if not rows:
        try:
            cot_service.sync_legacy_currency_futures(start, end)
            rows = _query_cot_rows(currencies, start, end)
        except Exception:
            rows = []

    by_currency: dict[str, list[CotPosition]] = defaultdict(list)
    for row in rows:
        by_currency[row.currency].append(row)
    for currency in by_currency:
        by_currency[currency].sort(key=lambda item: item.available_at)
    return dict(by_currency)


def _query_cot_rows(currencies: set[str], start: datetime, end: datetime) -> list[CotPosition]:
    session = get_session()
    try:
        return (
            session.query(CotPosition)
            .filter(
                CotPosition.currency.in_(sorted(currencies)),
                CotPosition.available_at >= start,
                CotPosition.available_at <= end,
            )
            .order_by(CotPosition.currency.asc(), CotPosition.available_at.asc())
            .all()
        )
    finally:
        session.close()


def _latest_cot_at(timestamp: datetime, rows: Iterable[CotPosition]) -> CotPosition | None:
    latest = None
    for row in rows:
        available_at = parse_dt(row.available_at)
        if available_at <= timestamp:
            latest = row
        else:
            break
    return latest


def _age_days(timestamp: datetime, available_at: datetime) -> float:
    return round(max(0.0, (timestamp - parse_dt(available_at)).total_seconds() / 86400.0), 2)


def _currency_strength_by_time(candles_by_symbol: dict[str, list[dict]], lookback: int = 16) -> dict[str, dict[str, float]]:
    totals: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    for symbol, candles in candles_by_symbol.items():
        base, quote = split_symbol(symbol)
        if base not in MAJOR_CURRENCIES or quote not in MAJOR_CURRENCIES:
            continue
        for idx, candle in enumerate(candles):
            if idx < lookback:
                continue
            prev_close = float(candles[idx - lookback]["close"] or 0)
            close = float(candle["close"] or 0)
            if prev_close <= 0:
                continue
            pct = (close - prev_close) / prev_close * 100.0
            ts = candle["time"]
            totals[ts][base] += pct
            totals[ts][quote] -= pct
            counts[ts][base] += 1
            counts[ts][quote] += 1

    result: dict[str, dict[str, float]] = {}
    for ts, currency_totals in totals.items():
        result[ts] = {
            currency: currency_totals[currency] / max(1, counts[ts][currency])
            for currency in currency_totals
        }
    return result


def _latest_sentiment_by_symbol(candles_by_symbol: dict[str, list[dict]]) -> dict[str, dict]:
    symbols = [symbol.upper() for symbol in candles_by_symbol]
    if not symbols:
        return {}
    session = get_session()
    try:
        result = {}
        now = datetime.now(timezone.utc)
        for symbol in symbols:
            row = (
                session.query(SentimentScore)
                .filter(SentimentScore.symbol == symbol)
                .order_by(SentimentScore.ts.desc())
                .first()
            )
            if row:
                result[symbol] = {
                    "score": float(row.score),
                    "age_minutes": round((now - parse_dt(row.ts)).total_seconds() / 60, 1),
                }
        return result
    finally:
        session.close()
