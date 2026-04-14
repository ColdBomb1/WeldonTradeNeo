from __future__ import annotations

import copy
from datetime import datetime, timezone
from threading import Lock
from typing import Any, Dict, Tuple

import httpx

DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/122.0.0.0 Safari/537.36"
)

SUPPORTED_RANGES = (
    "1d",
    "5d",
    "1mo",
    "3mo",
    "6mo",
    "1y",
    "2y",
    "5y",
    "10y",
    "ytd",
    "max",
)

SUPPORTED_INTERVALS = (
    "1m",
    "2m",
    "5m",
    "15m",
    "30m",
    "60m",
    "90m",
    "1h",
    "1d",
    "5d",
    "1wk",
    "1mo",
    "3mo",
)

_CACHE: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
_CACHE_LOCK = Lock()


def normalize_pair(pair: str) -> str:
    cleaned = pair.strip().upper().replace("/", "")
    if cleaned.endswith("=X"):
        cleaned = cleaned[:-2]

    if len(cleaned) != 6 or not cleaned.isalpha():
        raise ValueError("Currency pair must look like EURUSD or EUR/USD.")
    return cleaned


def _pick_latest_value(values: list[Any], index: int) -> float | None:
    if index < 0 or index >= len(values):
        return None
    value = values[index]
    if value is None:
        return None
    return float(value)


def _extract_points(result: dict) -> list[dict]:
    timestamps = result.get("timestamp") or []
    indicators = result.get("indicators") or {}
    quotes = indicators.get("quote") or [{}]
    quote = quotes[0] if quotes else {}

    closes = quote.get("close") or []
    opens = quote.get("open") or []
    highs = quote.get("high") or []
    lows = quote.get("low") or []
    volumes = quote.get("volume") or []

    points = []
    for idx, ts in enumerate(timestamps):
        price = (
            _pick_latest_value(closes, idx)
            or _pick_latest_value(opens, idx)
            or _pick_latest_value(highs, idx)
            or _pick_latest_value(lows, idx)
        )
        if price is None:
            continue

        dt = datetime.fromtimestamp(int(ts), tz=timezone.utc).isoformat()
        volume = _pick_latest_value(volumes, idx)
        points.append({"time": dt, "price": price, "volume": volume})
    return points


def extract_ohlcv_points(result: dict) -> list[dict]:
    """Extract full OHLCV candle data from a Yahoo Finance API result."""
    timestamps = result.get("timestamp") or []
    indicators = result.get("indicators") or {}
    quotes = indicators.get("quote") or [{}]
    quote = quotes[0] if quotes else {}

    closes = quote.get("close") or []
    opens = quote.get("open") or []
    highs = quote.get("high") or []
    lows = quote.get("low") or []
    volumes = quote.get("volume") or []

    points = []
    for idx, ts in enumerate(timestamps):
        o = _pick_latest_value(opens, idx)
        h = _pick_latest_value(highs, idx)
        l = _pick_latest_value(lows, idx)  # noqa: E741
        c = _pick_latest_value(closes, idx)

        if c is None and o is None:
            continue

        dt = datetime.fromtimestamp(int(ts), tz=timezone.utc).isoformat()
        volume = _pick_latest_value(volumes, idx)
        points.append({
            "time": dt,
            "open": o if o is not None else c,
            "high": h if h is not None else (c or o),
            "low": l if l is not None else (c or o),
            "close": c if c is not None else o,
            "volume": volume,
        })
    return points


def _fetch_yahoo_chart(symbol: str, range_name: str, interval: str, timeout_sec: float) -> dict:
    url = f"https://query2.finance.yahoo.com/v8/finance/chart/{symbol}=X"
    params = {
        "range": range_name,
        "interval": interval,
        "includePrePost": "false",
        "events": "history",
    }
    headers = {"User-Agent": DEFAULT_USER_AGENT, "Accept": "application/json"}

    response = httpx.get(url, params=params, headers=headers, timeout=timeout_sec)
    response.raise_for_status()
    payload = response.json()
    chart = payload.get("chart") or {}
    error = chart.get("error")
    if error:
        raise ValueError(error.get("description") or "Price request failed.")

    results = chart.get("result") or []
    if not results:
        raise ValueError("No price data returned for this symbol/timeframe.")
    return results[0]


def get_chart_data(symbol: str, range_name: str, interval: str, timeout_sec: float = 15.0) -> dict:
    pair = normalize_pair(symbol)
    selected_range = range_name if range_name in SUPPORTED_RANGES else "1d"
    selected_interval = interval if interval in SUPPORTED_INTERVALS else "1m"
    cache_key = (pair, selected_range, selected_interval)

    try:
        result = _fetch_yahoo_chart(pair, selected_range, selected_interval, timeout_sec)
        points = _extract_points(result)
        if not points:
            raise ValueError("No chart points available for this request.")

        meta = result.get("meta") or {}
        fetched_at = datetime.now(tz=timezone.utc).isoformat()
        latest = points[-1]
        volumes = [point.get("volume") for point in points if point.get("volume") is not None]
        has_nonzero_volume = any(volume > 0 for volume in volumes)
        if has_nonzero_volume:
            volume_state = "available"
        elif volumes:
            volume_state = "zero_only"
        else:
            volume_state = "missing"

        payload = {
            "symbol": pair,
            "range": selected_range,
            "interval": selected_interval,
            "source": "yahoo",
            "source_notice": "Free feed that may be delayed by up to about one minute.",
            "market_state": meta.get("marketState"),
            "currency": meta.get("currency"),
            "exchange_timezone": meta.get("exchangeTimezoneName"),
            "fetched_at": fetched_at,
            "is_stale": False,
            "volume_available": has_nonzero_volume,
            "volume_state": volume_state,
            "latest": latest,
            "points": points,
        }

        with _CACHE_LOCK:
            _CACHE[cache_key] = copy.deepcopy(payload)
        return payload
    except Exception:
        with _CACHE_LOCK:
            cached = _CACHE.get(cache_key)
        if cached:
            stale_payload = copy.deepcopy(cached)
            stale_payload["is_stale"] = True
            stale_payload["fetched_at"] = datetime.now(tz=timezone.utc).isoformat()
            stale_payload["warning"] = "Live fetch failed. Serving cached data."
            return stale_payload
        raise


def fetch_ohlcv(symbol: str, range_name: str, interval: str, timeout_sec: float = 15.0) -> list[dict]:
    """Fetch OHLCV candle data for a symbol. Returns list of candle dicts."""
    pair = normalize_pair(symbol)
    selected_range = range_name if range_name in SUPPORTED_RANGES else "1d"
    selected_interval = interval if interval in SUPPORTED_INTERVALS else "1m"

    result = _fetch_yahoo_chart(pair, selected_range, selected_interval, timeout_sec)
    return extract_ohlcv_points(result)
