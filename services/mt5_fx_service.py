from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict

try:
    import MetaTrader5 as mt5
except ImportError:  # pragma: no cover - optional dependency
    mt5 = None

from config import MT5Config

SUPPORTED_INTERVALS = (
    "1m",
    "2m",
    "5m",
    "15m",
    "30m",
    "60m",
    "1h",
    "4h",
    "1d",
    "1wk",
    "1mo",
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

_RANGE_SECONDS = {
    "1d": 86400,
    "5d": 86400 * 5,
    "1mo": 86400 * 30,
    "3mo": 86400 * 90,
    "6mo": 86400 * 180,
    "1y": 86400 * 365,
    "2y": 86400 * 365 * 2,
    "5y": 86400 * 365 * 5,
    "10y": 86400 * 365 * 10,
    "ytd": 86400 * 365,
    "max": 86400 * 365 * 10,
}

_INTERVAL_SECONDS = {
    "1m": 60,
    "2m": 120,
    "5m": 300,
    "15m": 900,
    "30m": 1800,
    "60m": 3600,
    "1h": 3600,
    "4h": 14400,
    "1d": 86400,
    "1wk": 86400 * 7,
    "1mo": 86400 * 30,
}


def is_available() -> bool:
    return mt5 is not None


def initialize(cfg: MT5Config) -> bool:
    if mt5 is None:
        return False

    kwargs: Dict[str, Any] = {}
    if cfg.login is not None:
        kwargs["login"] = int(cfg.login)
    if cfg.password:
        kwargs["password"] = cfg.password
    if cfg.server:
        kwargs["server"] = cfg.server

    if cfg.terminal_path:
        return bool(mt5.initialize(cfg.terminal_path, **kwargs))
    return bool(mt5.initialize(**kwargs))


def shutdown() -> None:
    if mt5 is not None:
        mt5.shutdown()


def _timeframe(interval: str):
    if mt5 is None:
        return None
    mapping = {
        "1m": "TIMEFRAME_M1",
        "2m": "TIMEFRAME_M2",
        "5m": "TIMEFRAME_M5",
        "15m": "TIMEFRAME_M15",
        "30m": "TIMEFRAME_M30",
        "60m": "TIMEFRAME_H1",
        "1h": "TIMEFRAME_H1",
        "4h": "TIMEFRAME_H4",
        "1d": "TIMEFRAME_D1",
        "1wk": "TIMEFRAME_W1",
        "1mo": "TIMEFRAME_MN1",
    }
    attr_name = mapping.get(interval)
    if not attr_name:
        return None
    return getattr(mt5, attr_name, None)


def _normalize_pair(symbol: str) -> str:
    cleaned = symbol.strip().upper().replace("/", "").replace("=X", "")
    if not cleaned:
        raise ValueError("Symbol is required.")
    return cleaned


def _resolve_symbol(pair: str, cfg: MT5Config) -> str:
    if mt5 is None:
        raise ValueError("MetaTrader5 package is not installed.")

    normalized = _normalize_pair(pair)
    suffix = (cfg.symbol_suffix or "").strip()

    candidates = [normalized]
    if suffix:
        candidates.insert(0, f"{normalized}{suffix}")

    for candidate in candidates:
        info = mt5.symbol_info(candidate)
        if info is not None:
            mt5.symbol_select(candidate, True)
            return candidate

    matched = mt5.symbols_get(f"{normalized}*") or []
    if matched:
        selected = matched[0].name
        mt5.symbol_select(selected, True)
        return selected

    raise ValueError(f"Could not find an MT5 symbol matching '{normalized}'.")


def _bar_count(range_name: str, interval: str) -> int:
    range_seconds = _RANGE_SECONDS.get(range_name, _RANGE_SECONDS["1d"])
    interval_seconds = _INTERVAL_SECONDS.get(interval, _INTERVAL_SECONDS["1m"])
    bars = int(range_seconds / max(interval_seconds, 1))
    return max(50, min(10000, bars + 20))


def _parse_points(rates) -> list[dict]:
    points = []
    if rates is None:
        return points

    for row in rates:
        close = float(row["close"])
        tick_volume = float(row["tick_volume"]) if "tick_volume" in row.dtype.names else None
        real_volume = float(row["real_volume"]) if "real_volume" in row.dtype.names else None
        volume = real_volume if real_volume and real_volume > 0 else tick_volume
        dt = datetime.fromtimestamp(int(row["time"]), tz=timezone.utc).isoformat()
        points.append({"time": dt, "price": close, "volume": volume})
    return points


def _parse_ohlcv(rates) -> list[dict]:
    candles = []
    if rates is None:
        return candles

    for row in rates:
        tick_volume = float(row["tick_volume"]) if "tick_volume" in row.dtype.names else None
        real_volume = float(row["real_volume"]) if "real_volume" in row.dtype.names else None
        volume = real_volume if real_volume and real_volume > 0 else tick_volume
        dt = datetime.fromtimestamp(int(row["time"]), tz=timezone.utc).isoformat()
        candles.append({
            "time": dt,
            "open": float(row["open"]),
            "high": float(row["high"]),
            "low": float(row["low"]),
            "close": float(row["close"]),
            "volume": volume,
        })
    return candles


def fetch_ohlcv(symbol: str, range_name: str, interval: str, cfg: MT5Config) -> list[dict]:
    """Fetch OHLCV candles from the configured MT5 terminal."""
    if mt5 is None:
        raise ValueError("MetaTrader5 package is not installed in this Python environment.")

    selected_range = range_name if range_name in SUPPORTED_RANGES else "1d"
    selected_interval = interval if interval in SUPPORTED_INTERVALS else "1m"
    timeframe = _timeframe(selected_interval)
    if timeframe is None:
        raise ValueError(f"MT5 does not support interval '{selected_interval}'.")

    if not initialize(cfg):
        raise ValueError("Failed to initialize MT5. Check terminal path/account settings.")

    try:
        resolved_symbol = _resolve_symbol(symbol, cfg)
        rates = mt5.copy_rates_from_pos(
            resolved_symbol,
            timeframe,
            0,
            _bar_count(selected_range, selected_interval),
        )
        candles = _parse_ohlcv(rates)
        if not candles:
            raise ValueError(f"No MT5 OHLCV data returned for symbol '{resolved_symbol}'.")
        return candles
    finally:
        shutdown()


def get_chart_data(symbol: str, range_name: str, interval: str, cfg: MT5Config) -> dict:
    if mt5 is None:
        raise ValueError("MetaTrader5 package is not installed in this Python environment.")

    selected_range = range_name if range_name in SUPPORTED_RANGES else "1d"
    if interval not in SUPPORTED_INTERVALS:
        raise ValueError(f"Interval '{interval}' is not supported for MT5 source.")
    selected_interval = interval

    timeframe = _timeframe(selected_interval)
    if timeframe is None:
        raise ValueError(f"MT5 does not support interval '{selected_interval}'.")

    if not initialize(cfg):
        raise ValueError("Failed to initialize MT5. Check terminal path/account settings.")

    try:
        resolved_symbol = _resolve_symbol(symbol, cfg)
        rates = mt5.copy_rates_from_pos(
            resolved_symbol,
            timeframe,
            0,
            _bar_count(selected_range, selected_interval),
        )
        points = _parse_points(rates)
        if not points:
            raise ValueError(f"No MT5 price data returned for symbol '{resolved_symbol}'.")

        latest = points[-1]
        volumes = [point.get("volume") for point in points if point.get("volume") is not None]
        has_nonzero_volume = any(volume > 0 for volume in volumes)
        if has_nonzero_volume:
            volume_state = "available"
        elif volumes:
            volume_state = "zero_only"
        else:
            volume_state = "missing"

        return {
            "symbol": _normalize_pair(symbol),
            "resolved_symbol": resolved_symbol,
            "range": selected_range,
            "interval": selected_interval,
            "source": "mt5",
            "source_notice": "Data from local MetaTrader 5 terminal.",
            "fetched_at": datetime.now(tz=timezone.utc).isoformat(),
            "is_stale": False,
            "volume_available": has_nonzero_volume,
            "volume_state": volume_state,
            "latest": latest,
            "points": points,
        }
    finally:
        shutdown()
