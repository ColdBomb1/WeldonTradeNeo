from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone

from config import load_config
from services import yahoo_fx_service

logger = logging.getLogger(__name__)

# Maps our candle timeframes to Yahoo Finance (interval, range) pairs.
# Yahoo has limits on how far back each interval goes.
_TIMEFRAME_MAP = {
    "1m": ("1m", "1d"),
    "5m": ("5m", "5d"),
    "15m": ("15m", "1mo"),
    "30m": ("30m", "1mo"),
    "1h": ("1h", "3mo"),
    "4h": ("1h", "3mo"),  # aggregate 4x 1h candles
    "1d": ("1d", "1y"),
    "1w": ("1wk", "5y"),
}


def _aggregate_4h_candles(hourly: list[dict]) -> list[dict]:
    """Aggregate 1-hour candles into 4-hour candles."""
    if not hourly:
        return []

    result = []
    bucket = []
    for candle in hourly:
        dt = datetime.fromisoformat(candle["time"])
        bucket_hour = (dt.hour // 4) * 4
        if bucket and datetime.fromisoformat(bucket[0]["time"]).hour // 4 * 4 != bucket_hour:
            result.append(_merge_bucket(bucket))
            bucket = []
        bucket.append(candle)

    if bucket:
        result.append(_merge_bucket(bucket))
    return result


def _merge_bucket(candles: list[dict]) -> dict:
    """Merge multiple candles into one OHLCV candle."""
    o = candles[0]["open"]
    h = max(c["high"] for c in candles)
    l = min(c["low"] for c in candles)  # noqa: E741
    c = candles[-1]["close"]
    vol = sum((c.get("volume") or 0) for c in candles)
    return {
        "time": candles[0]["time"],
        "open": o,
        "high": h,
        "low": l,
        "close": c,
        "volume": vol if vol > 0 else None,
    }


def _collect_and_store(symbol: str, timeframe: str, timeout_sec: float) -> int:
    """Fetch OHLCV data from Yahoo and upsert into the candles table.

    Returns the number of candles upserted.
    """
    from sqlalchemy.dialects.postgresql import insert as pg_insert
    from db import get_session
    from models.candle import Candle

    mapping = _TIMEFRAME_MAP.get(timeframe)
    if not mapping:
        return 0

    yahoo_interval, yahoo_range = mapping

    try:
        ohlcv = yahoo_fx_service.fetch_ohlcv(symbol, yahoo_range, yahoo_interval, timeout_sec)
    except Exception as exc:
        logger.warning("Yahoo OHLCV fetch failed for %s/%s: %s", symbol, timeframe, exc)
        return 0

    if not ohlcv:
        return 0

    if timeframe == "4h":
        ohlcv = _aggregate_4h_candles(ohlcv)

    session = get_session()
    try:
        count = 0
        for candle in ohlcv:
            ts_str = candle["time"]
            if isinstance(ts_str, str):
                ts = datetime.fromisoformat(ts_str)
            else:
                ts = ts_str
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)

            stmt = pg_insert(Candle).values(
                symbol=symbol,
                timeframe=timeframe,
                ts=ts,
                open=candle["open"],
                high=candle["high"],
                low=candle["low"],
                close=candle["close"],
                volume=candle.get("volume"),
                source="yahoo",
            ).on_conflict_do_update(
                constraint="uq_candle_symbol_tf_ts",
                set_={
                    "open": candle["open"],
                    "high": candle["high"],
                    "low": candle["low"],
                    "close": candle["close"],
                    "volume": candle.get("volume"),
                    "source": "yahoo",
                },
            )
            session.execute(stmt)
            count += 1

        session.commit()
        return count
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


async def candle_collection_loop(stop_event: asyncio.Event) -> None:
    """Background loop that periodically collects OHLCV candle data."""
    logger.info("Candle collection loop started.")
    while not stop_event.is_set():
        cfg = load_config()
        if cfg.candle_collection_enabled:
            for symbol in cfg.symbols:
                for timeframe in cfg.candle_timeframes:
                    if stop_event.is_set():
                        break
                    try:
                        count = await asyncio.to_thread(
                            _collect_and_store, symbol, timeframe, cfg.request_timeout_sec
                        )
                        if count:
                            logger.info("Stored %d candles: %s/%s", count, symbol, timeframe)
                    except Exception as exc:
                        logger.error("Candle collection error %s/%s: %s", symbol, timeframe, exc)
                    # Small delay between requests to avoid rate limiting
                    await asyncio.sleep(0.5)

        try:
            await asyncio.wait_for(stop_event.wait(), timeout=cfg.candle_collection_interval_sec)
        except asyncio.TimeoutError:
            pass
    logger.info("Candle collection loop stopped.")
