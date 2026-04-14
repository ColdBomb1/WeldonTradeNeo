"""Technical indicator computations.

All functions operate on plain lists of floats (close prices, etc.)
and return lists of the same length with None for periods where
the indicator cannot yet be computed.
"""

from __future__ import annotations

import math
from typing import List, Optional


def sma(closes: List[float], period: int) -> List[Optional[float]]:
    """Simple Moving Average."""
    result: List[Optional[float]] = [None] * len(closes)
    if period < 1 or len(closes) < period:
        return result
    window_sum = sum(closes[:period])
    result[period - 1] = window_sum / period
    for i in range(period, len(closes)):
        window_sum += closes[i] - closes[i - period]
        result[i] = window_sum / period
    return result


def ema(closes: List[float], period: int) -> List[Optional[float]]:
    """Exponential Moving Average."""
    result: List[Optional[float]] = [None] * len(closes)
    if period < 1 or len(closes) < period:
        return result
    k = 2.0 / (period + 1)
    # Seed with SMA of first `period` values
    seed = sum(closes[:period]) / period
    result[period - 1] = seed
    prev = seed
    for i in range(period, len(closes)):
        val = closes[i] * k + prev * (1 - k)
        result[i] = val
        prev = val
    return result


def rsi(closes: List[float], period: int = 14) -> List[Optional[float]]:
    """Relative Strength Index (Wilder's smoothing)."""
    result: List[Optional[float]] = [None] * len(closes)
    if period < 1 or len(closes) < period + 1:
        return result

    gains = []
    losses = []
    for i in range(1, len(closes)):
        diff = closes[i] - closes[i - 1]
        gains.append(max(diff, 0.0))
        losses.append(max(-diff, 0.0))

    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period

    if avg_loss == 0:
        result[period] = 100.0
    else:
        rs = avg_gain / avg_loss
        result[period] = 100.0 - (100.0 / (1.0 + rs))

    for i in range(period, len(gains)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        if avg_loss == 0:
            result[i + 1] = 100.0
        else:
            rs = avg_gain / avg_loss
            result[i + 1] = 100.0 - (100.0 / (1.0 + rs))
    return result


def macd(
    closes: List[float],
    fast: int = 12,
    slow: int = 26,
    signal_period: int = 9,
) -> dict:
    """MACD indicator.

    Returns dict with keys: "macd", "signal", "histogram" — each a list.
    """
    fast_ema = ema(closes, fast)
    slow_ema = ema(closes, slow)

    macd_line: List[Optional[float]] = [None] * len(closes)
    for i in range(len(closes)):
        if fast_ema[i] is not None and slow_ema[i] is not None:
            macd_line[i] = fast_ema[i] - slow_ema[i]

    # Signal line: EMA of the MACD line
    macd_values = [v for v in macd_line if v is not None]
    if len(macd_values) < signal_period:
        signal_line: List[Optional[float]] = [None] * len(closes)
        histogram: List[Optional[float]] = [None] * len(closes)
    else:
        signal_ema = ema(macd_values, signal_period)
        signal_line = [None] * len(closes)
        histogram = [None] * len(closes)

        macd_start = next(i for i, v in enumerate(macd_line) if v is not None)
        for j, val in enumerate(signal_ema):
            idx = macd_start + j
            if idx < len(closes):
                signal_line[idx] = val

        for i in range(len(closes)):
            if macd_line[i] is not None and signal_line[i] is not None:
                histogram[i] = macd_line[i] - signal_line[i]

    return {
        "macd": macd_line,
        "signal": signal_line,
        "histogram": histogram,
    }


def atr(candles: List[dict], period: int = 14) -> List[Optional[float]]:
    """Average True Range — measures volatility for stop-loss placement.

    Uses Wilder's smoothing (same as RSI).
    candles: list of dicts with 'high', 'low', 'close' keys.
    """
    n = len(candles)
    result: List[Optional[float]] = [None] * n
    if period < 1 or n < period + 1:
        return result

    # True Range for each bar (starting from index 1)
    tr: List[float] = []
    for i in range(1, n):
        h = candles[i]["high"]
        l = candles[i]["low"]
        pc = candles[i - 1]["close"]
        tr.append(max(h - l, abs(h - pc), abs(l - pc)))

    # First ATR = simple average of first `period` true ranges
    first_atr = sum(tr[:period]) / period
    result[period] = first_atr

    # Wilder's smoothing for the rest
    prev = first_atr
    for i in range(period, len(tr)):
        val = (prev * (period - 1) + tr[i]) / period
        result[i + 1] = val
        prev = val

    return result


def bollinger_bands(
    closes: List[float], period: int = 20, std_dev: float = 2.0
) -> dict:
    """Bollinger Bands.

    Returns dict with keys: "upper", "middle", "lower" — each a list.
    """
    middle = sma(closes, period)
    upper: List[Optional[float]] = [None] * len(closes)
    lower: List[Optional[float]] = [None] * len(closes)

    for i in range(period - 1, len(closes)):
        if middle[i] is None:
            continue
        window = closes[i - period + 1: i + 1]
        mean = middle[i]
        variance = sum((x - mean) ** 2 for x in window) / period
        sd = math.sqrt(variance)
        upper[i] = mean + std_dev * sd
        lower[i] = mean - std_dev * sd

    return {
        "upper": upper,
        "middle": middle,
        "lower": lower,
    }
