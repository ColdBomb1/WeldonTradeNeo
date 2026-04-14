from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from db import get_session
from models.candle import Candle
from services import indicator_service

router = APIRouter(tags=["indicators"])


def _parse_indicator_spec(spec: str) -> tuple[str, dict]:
    """Parse indicator spec like 'sma_20' or 'macd' or 'bbands_20_2'.

    Returns (indicator_name, params_dict).
    """
    parts = spec.lower().split("_")
    name = parts[0]

    if name == "sma":
        period = int(parts[1]) if len(parts) > 1 else 20
        return "sma", {"period": period}
    elif name == "ema":
        period = int(parts[1]) if len(parts) > 1 else 20
        return "ema", {"period": period}
    elif name == "rsi":
        period = int(parts[1]) if len(parts) > 1 else 14
        return "rsi", {"period": period}
    elif name == "macd":
        fast = int(parts[1]) if len(parts) > 1 else 12
        slow = int(parts[2]) if len(parts) > 2 else 26
        signal = int(parts[3]) if len(parts) > 3 else 9
        return "macd", {"fast": fast, "slow": slow, "signal": signal}
    elif name in ("bbands", "bollinger"):
        period = int(parts[1]) if len(parts) > 1 else 20
        std = float(parts[2]) if len(parts) > 2 else 2.0
        return "bbands", {"period": period, "std_dev": std}
    else:
        raise ValueError(f"Unknown indicator: {name}")


@router.get("/api/indicators")
def compute_indicators(
    symbol: str,
    timeframe: str = "1h",
    indicators: str = "sma_20",
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

    if not rows:
        return JSONResponse({"symbol": symbol, "timeframe": timeframe, "candles": [], "indicators": {}})

    candles = [
        {
            "time": row.ts.isoformat(),
            "open": row.open,
            "high": row.high,
            "low": row.low,
            "close": row.close,
            "volume": row.volume,
        }
        for row in rows
    ]
    closes = [row.close for row in rows]

    result_indicators = {}
    specs = [s.strip() for s in indicators.split(",") if s.strip()]

    for spec in specs:
        try:
            name, params = _parse_indicator_spec(spec)
        except ValueError:
            continue

        if name == "sma":
            values = indicator_service.sma(closes, params["period"])
            result_indicators[spec] = {"type": "overlay", "values": values}
        elif name == "ema":
            values = indicator_service.ema(closes, params["period"])
            result_indicators[spec] = {"type": "overlay", "values": values}
        elif name == "rsi":
            values = indicator_service.rsi(closes, params["period"])
            result_indicators[spec] = {"type": "subchart", "values": values, "range": [0, 100]}
        elif name == "macd":
            data = indicator_service.macd(closes, params["fast"], params["slow"], params["signal"])
            result_indicators[spec] = {"type": "subchart", "macd": data["macd"], "signal": data["signal"], "histogram": data["histogram"]}
        elif name == "bbands":
            data = indicator_service.bollinger_bands(closes, params["period"], params["std_dev"])
            result_indicators[spec] = {"type": "overlay", "upper": data["upper"], "middle": data["middle"], "lower": data["lower"]}

    return JSONResponse({
        "symbol": symbol,
        "timeframe": timeframe,
        "candles": candles,
        "indicators": result_indicators,
    })
