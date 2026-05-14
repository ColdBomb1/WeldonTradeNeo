"""Pluggable trading strategy framework.

Each strategy implements BaseStrategy and is registered in STRATEGIES.

Performance: strategies support precompute() + evaluate_at() to avoid
redundant indicator recomputation during backtests. The backtest engine
calls precompute() once for the full candle set, then evaluate_at() at
each bar — turning O(n^2) into O(n) per backtest.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import List, Optional

from services import indicator_service


class SignalType(Enum):
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"


@dataclass
class Signal:
    type: SignalType
    price: float
    timestamp: datetime
    reason: str = ""
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    confidence: Optional[float] = None


class BaseStrategy(ABC):
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def description(self) -> str: ...

    @abstractmethod
    def required_params(self) -> List[dict]: ...

    @abstractmethod
    def default_params(self) -> dict: ...

    @abstractmethod
    def evaluate(self, candles: List[dict], params: dict) -> Signal:
        """Evaluate the strategy on candle history up to the current bar.

        candles: list of dicts with keys time, open, high, low, close, volume
        params: strategy-specific parameters
        Returns a Signal for the current bar.
        """
        ...

    def precompute(self, candles: List[dict], params: dict) -> dict:
        """Precompute all indicator arrays for the full candle set.

        Override this for O(n) backtests instead of O(n^2).
        Returns a dict of precomputed data that evaluate_at() can use.
        """
        return {}

    def evaluate_at(self, i: int, candles: List[dict], pre: dict, params: dict) -> Signal:
        """Evaluate at bar index i using precomputed data.

        Default falls back to evaluate() with a slice (slow).
        Override alongside precompute() for fast backtesting.
        """
        return self.evaluate(candles[: i + 1], params)


# ---------------------------------------------------------------------------
# Single-indicator strategies
# ---------------------------------------------------------------------------


class SMACrossoverStrategy(BaseStrategy):
    def name(self) -> str:
        return "sma_crossover"

    def description(self) -> str:
        return "Buy when fast SMA crosses above slow SMA; sell when it crosses below."

    def required_params(self) -> List[dict]:
        return [
            {"name": "fast_period", "type": "int", "label": "Fast SMA Period", "default": 10},
            {"name": "slow_period", "type": "int", "label": "Slow SMA Period", "default": 30},
        ]

    def default_params(self) -> dict:
        return {"fast_period": 10, "slow_period": 30}

    def evaluate(self, candles: List[dict], params: dict) -> Signal:
        fast_period = int(params.get("fast_period", 10))
        slow_period = int(params.get("slow_period", 30))
        closes = [c["close"] for c in candles]
        current_price = closes[-1]
        ts = datetime.fromisoformat(candles[-1]["time"])

        if len(closes) < slow_period + 1:
            return Signal(SignalType.HOLD, current_price, ts, "Not enough data")

        fast_sma = indicator_service.sma(closes, fast_period)
        slow_sma = indicator_service.sma(closes, slow_period)

        curr_fast = fast_sma[-1]
        curr_slow = slow_sma[-1]
        prev_fast = fast_sma[-2]
        prev_slow = slow_sma[-2]

        if None in (curr_fast, curr_slow, prev_fast, prev_slow):
            return Signal(SignalType.HOLD, current_price, ts, "Indicator not ready")

        if prev_fast <= prev_slow and curr_fast > curr_slow:
            return Signal(SignalType.BUY, current_price, ts, "Fast SMA crossed above slow SMA")
        elif prev_fast >= prev_slow and curr_fast < curr_slow:
            return Signal(SignalType.SELL, current_price, ts, "Fast SMA crossed below slow SMA")

        return Signal(SignalType.HOLD, current_price, ts)

    def precompute(self, candles, params):
        closes = [c["close"] for c in candles]
        return {
            "closes": closes,
            "fast_sma": indicator_service.sma(closes, int(params.get("fast_period", 10))),
            "slow_sma": indicator_service.sma(closes, int(params.get("slow_period", 30))),
        }

    def evaluate_at(self, i, candles, pre, params):
        price = pre["closes"][i]
        ts = datetime.fromisoformat(candles[i]["time"])
        cf, cs = pre["fast_sma"][i], pre["slow_sma"][i]
        pf, ps = pre["fast_sma"][i - 1], pre["slow_sma"][i - 1]
        if None in (cf, cs, pf, ps):
            return Signal(SignalType.HOLD, price, ts)
        if pf <= ps and cf > cs:
            return Signal(SignalType.BUY, price, ts, "Fast SMA crossed above slow SMA")
        if pf >= ps and cf < cs:
            return Signal(SignalType.SELL, price, ts, "Fast SMA crossed below slow SMA")
        return Signal(SignalType.HOLD, price, ts)


class RSIReversalStrategy(BaseStrategy):
    def name(self) -> str:
        return "rsi_reversal"

    def description(self) -> str:
        return "Buy when RSI crosses above oversold level; sell when it crosses below overbought."

    def required_params(self) -> List[dict]:
        return [
            {"name": "rsi_period", "type": "int", "label": "RSI Period", "default": 14},
            {"name": "oversold", "type": "float", "label": "Oversold Level", "default": 30.0},
            {"name": "overbought", "type": "float", "label": "Overbought Level", "default": 70.0},
        ]

    def default_params(self) -> dict:
        return {"rsi_period": 14, "oversold": 30.0, "overbought": 70.0}

    def evaluate(self, candles: List[dict], params: dict) -> Signal:
        rsi_period = int(params.get("rsi_period", 14))
        oversold = float(params.get("oversold", 30.0))
        overbought = float(params.get("overbought", 70.0))
        closes = [c["close"] for c in candles]
        current_price = closes[-1]
        ts = datetime.fromisoformat(candles[-1]["time"])

        if len(closes) < rsi_period + 2:
            return Signal(SignalType.HOLD, current_price, ts, "Not enough data")

        rsi_values = indicator_service.rsi(closes, rsi_period)
        curr_rsi = rsi_values[-1]
        prev_rsi = rsi_values[-2]

        if curr_rsi is None or prev_rsi is None:
            return Signal(SignalType.HOLD, current_price, ts, "RSI not ready")

        if prev_rsi <= oversold and curr_rsi > oversold:
            return Signal(SignalType.BUY, current_price, ts, f"RSI crossed above {oversold}")
        elif prev_rsi >= overbought and curr_rsi < overbought:
            return Signal(SignalType.SELL, current_price, ts, f"RSI crossed below {overbought}")

        return Signal(SignalType.HOLD, current_price, ts)

    def precompute(self, candles, params):
        closes = [c["close"] for c in candles]
        return {
            "closes": closes,
            "rsi": indicator_service.rsi(closes, int(params.get("rsi_period", 14))),
        }

    def evaluate_at(self, i, candles, pre, params):
        price = pre["closes"][i]
        ts = datetime.fromisoformat(candles[i]["time"])
        oversold = float(params.get("oversold", 30.0))
        overbought = float(params.get("overbought", 70.0))
        cr, pr = pre["rsi"][i], pre["rsi"][i - 1]
        if cr is None or pr is None:
            return Signal(SignalType.HOLD, price, ts)
        if pr <= oversold and cr > oversold:
            return Signal(SignalType.BUY, price, ts, f"RSI crossed above {oversold}")
        if pr >= overbought and cr < overbought:
            return Signal(SignalType.SELL, price, ts, f"RSI crossed below {overbought}")
        return Signal(SignalType.HOLD, price, ts)


class MACDCrossoverStrategy(BaseStrategy):
    def name(self) -> str:
        return "macd_crossover"

    def description(self) -> str:
        return "Buy when MACD crosses above signal line; sell when it crosses below."

    def required_params(self) -> List[dict]:
        return [
            {"name": "fast_period", "type": "int", "label": "Fast EMA Period", "default": 12},
            {"name": "slow_period", "type": "int", "label": "Slow EMA Period", "default": 26},
            {"name": "signal_period", "type": "int", "label": "Signal Period", "default": 9},
        ]

    def default_params(self) -> dict:
        return {"fast_period": 12, "slow_period": 26, "signal_period": 9}

    def evaluate(self, candles: List[dict], params: dict) -> Signal:
        fast = int(params.get("fast_period", 12))
        slow = int(params.get("slow_period", 26))
        signal_period = int(params.get("signal_period", 9))
        closes = [c["close"] for c in candles]
        current_price = closes[-1]
        ts = datetime.fromisoformat(candles[-1]["time"])

        if len(closes) < slow + signal_period + 1:
            return Signal(SignalType.HOLD, current_price, ts, "Not enough data")

        result = indicator_service.macd(closes, fast, slow, signal_period)
        macd_line = result["macd"]
        signal_line = result["signal"]

        curr_macd = macd_line[-1]
        curr_signal = signal_line[-1]
        prev_macd = macd_line[-2]
        prev_signal = signal_line[-2]

        if None in (curr_macd, curr_signal, prev_macd, prev_signal):
            return Signal(SignalType.HOLD, current_price, ts, "MACD not ready")

        if prev_macd <= prev_signal and curr_macd > curr_signal:
            return Signal(SignalType.BUY, current_price, ts, "MACD crossed above signal")
        elif prev_macd >= prev_signal and curr_macd < curr_signal:
            return Signal(SignalType.SELL, current_price, ts, "MACD crossed below signal")

        return Signal(SignalType.HOLD, current_price, ts)

    def precompute(self, candles, params):
        closes = [c["close"] for c in candles]
        m = indicator_service.macd(
            closes,
            int(params.get("fast_period", 12)),
            int(params.get("slow_period", 26)),
            int(params.get("signal_period", 9)),
        )
        return {"closes": closes, "macd": m["macd"], "signal": m["signal"]}

    def evaluate_at(self, i, candles, pre, params):
        price = pre["closes"][i]
        ts = datetime.fromisoformat(candles[i]["time"])
        cm, cs_ = pre["macd"][i], pre["signal"][i]
        pm, ps_ = pre["macd"][i - 1], pre["signal"][i - 1]
        if None in (cm, cs_, pm, ps_):
            return Signal(SignalType.HOLD, price, ts)
        if pm <= ps_ and cm > cs_:
            return Signal(SignalType.BUY, price, ts, "MACD crossed above signal")
        if pm >= ps_ and cm < cs_:
            return Signal(SignalType.SELL, price, ts, "MACD crossed below signal")
        return Signal(SignalType.HOLD, price, ts)


# ---------------------------------------------------------------------------
# Composite / Multi-Indicator Strategies
# ---------------------------------------------------------------------------


class SMARSIStrategy(BaseStrategy):
    """SMA crossover for direction + RSI filter to avoid overbought buys / oversold sells."""

    def name(self) -> str:
        return "sma_rsi"

    def description(self) -> str:
        return "SMA crossover filtered by RSI — avoids buying overbought / selling oversold."

    def required_params(self) -> List[dict]:
        return [
            {"name": "fast_period", "type": "int", "label": "Fast SMA Period", "default": 10},
            {"name": "slow_period", "type": "int", "label": "Slow SMA Period", "default": 30},
            {"name": "rsi_period", "type": "int", "label": "RSI Period", "default": 14},
            {"name": "rsi_oversold", "type": "float", "label": "RSI Oversold", "default": 35.0},
            {"name": "rsi_overbought", "type": "float", "label": "RSI Overbought", "default": 65.0},
        ]

    def default_params(self) -> dict:
        return {"fast_period": 10, "slow_period": 30, "rsi_period": 14, "rsi_oversold": 35.0, "rsi_overbought": 65.0}

    def evaluate(self, candles: List[dict], params: dict) -> Signal:
        fast_period = int(params.get("fast_period", 10))
        slow_period = int(params.get("slow_period", 30))
        rsi_period = int(params.get("rsi_period", 14))
        rsi_oversold = float(params.get("rsi_oversold", 35.0))
        rsi_overbought = float(params.get("rsi_overbought", 65.0))
        closes = [c["close"] for c in candles]
        current_price = closes[-1]
        ts = datetime.fromisoformat(candles[-1]["time"])

        min_bars = max(slow_period + 1, rsi_period + 2)
        if len(closes) < min_bars:
            return Signal(SignalType.HOLD, current_price, ts, "Not enough data")

        fast_sma = indicator_service.sma(closes, fast_period)
        slow_sma = indicator_service.sma(closes, slow_period)
        rsi_values = indicator_service.rsi(closes, rsi_period)

        curr_fast, prev_fast = fast_sma[-1], fast_sma[-2]
        curr_slow, prev_slow = slow_sma[-1], slow_sma[-2]
        curr_rsi = rsi_values[-1]

        if None in (curr_fast, curr_slow, prev_fast, prev_slow, curr_rsi):
            return Signal(SignalType.HOLD, current_price, ts, "Indicators not ready")

        if prev_fast <= prev_slow and curr_fast > curr_slow and curr_rsi < rsi_overbought:
            return Signal(SignalType.BUY, current_price, ts,
                          f"SMA crossover UP (RSI={curr_rsi:.1f} < {rsi_overbought})")
        if prev_fast >= prev_slow and curr_fast < curr_slow and curr_rsi > rsi_oversold:
            return Signal(SignalType.SELL, current_price, ts,
                          f"SMA crossover DOWN (RSI={curr_rsi:.1f} > {rsi_oversold})")

        return Signal(SignalType.HOLD, current_price, ts)

    def precompute(self, candles, params):
        closes = [c["close"] for c in candles]
        return {
            "closes": closes,
            "fast_sma": indicator_service.sma(closes, int(params.get("fast_period", 10))),
            "slow_sma": indicator_service.sma(closes, int(params.get("slow_period", 30))),
            "rsi": indicator_service.rsi(closes, int(params.get("rsi_period", 14))),
        }

    def evaluate_at(self, i, candles, pre, params):
        price = pre["closes"][i]
        ts = datetime.fromisoformat(candles[i]["time"])
        rsi_oversold = float(params.get("rsi_oversold", 35.0))
        rsi_overbought = float(params.get("rsi_overbought", 65.0))
        cf, cs_ = pre["fast_sma"][i], pre["slow_sma"][i]
        pf, ps_ = pre["fast_sma"][i - 1], pre["slow_sma"][i - 1]
        cr = pre["rsi"][i]
        if None in (cf, cs_, pf, ps_, cr):
            return Signal(SignalType.HOLD, price, ts)
        if pf <= ps_ and cf > cs_ and cr < rsi_overbought:
            return Signal(SignalType.BUY, price, ts, f"SMA crossover UP (RSI={cr:.1f})")
        if pf >= ps_ and cf < cs_ and cr > rsi_oversold:
            return Signal(SignalType.SELL, price, ts, f"SMA crossover DOWN (RSI={cr:.1f})")
        return Signal(SignalType.HOLD, price, ts)


class MACDBollingerStrategy(BaseStrategy):
    """MACD for direction + Bollinger Band proximity for entry timing."""

    def name(self) -> str:
        return "macd_bbands"

    def description(self) -> str:
        return "MACD crossover confirmed by price near Bollinger Band edge."

    def required_params(self) -> List[dict]:
        return [
            {"name": "macd_fast", "type": "int", "label": "MACD Fast EMA", "default": 12},
            {"name": "macd_slow", "type": "int", "label": "MACD Slow EMA", "default": 26},
            {"name": "macd_signal", "type": "int", "label": "MACD Signal", "default": 9},
            {"name": "bb_period", "type": "int", "label": "Bollinger Period", "default": 20},
            {"name": "bb_std", "type": "float", "label": "Bollinger Std Dev", "default": 2.0},
        ]

    def default_params(self) -> dict:
        return {"macd_fast": 12, "macd_slow": 26, "macd_signal": 9, "bb_period": 20, "bb_std": 2.0}

    def evaluate(self, candles: List[dict], params: dict) -> Signal:
        macd_fast = int(params.get("macd_fast", 12))
        macd_slow = int(params.get("macd_slow", 26))
        macd_sig = int(params.get("macd_signal", 9))
        bb_period = int(params.get("bb_period", 20))
        bb_std = float(params.get("bb_std", 2.0))
        closes = [c["close"] for c in candles]
        current_price = closes[-1]
        ts = datetime.fromisoformat(candles[-1]["time"])

        min_bars = max(macd_slow + macd_sig + 1, bb_period + 1)
        if len(closes) < min_bars:
            return Signal(SignalType.HOLD, current_price, ts, "Not enough data")

        macd_data = indicator_service.macd(closes, macd_fast, macd_slow, macd_sig)
        bb_data = indicator_service.bollinger_bands(closes, bb_period, bb_std)

        curr_macd = macd_data["macd"][-1]
        curr_signal = macd_data["signal"][-1]
        prev_macd = macd_data["macd"][-2]
        prev_signal = macd_data["signal"][-2]
        bb_lower = bb_data["lower"][-1]
        bb_upper = bb_data["upper"][-1]
        bb_middle = bb_data["middle"][-1]

        if None in (curr_macd, curr_signal, prev_macd, prev_signal, bb_lower, bb_upper, bb_middle):
            return Signal(SignalType.HOLD, current_price, ts, "Indicators not ready")

        band_width = bb_upper - bb_lower
        near_threshold = band_width * 0.25

        if prev_macd <= prev_signal and curr_macd > curr_signal and current_price < bb_lower + near_threshold:
            return Signal(SignalType.BUY, current_price, ts,
                          f"MACD cross UP near lower BB (price={current_price:.5f}, BB lower={bb_lower:.5f})")
        if prev_macd >= prev_signal and curr_macd < curr_signal and current_price > bb_upper - near_threshold:
            return Signal(SignalType.SELL, current_price, ts,
                          f"MACD cross DOWN near upper BB (price={current_price:.5f}, BB upper={bb_upper:.5f})")

        return Signal(SignalType.HOLD, current_price, ts)

    def precompute(self, candles, params):
        closes = [c["close"] for c in candles]
        m = indicator_service.macd(
            closes,
            int(params.get("macd_fast", 12)),
            int(params.get("macd_slow", 26)),
            int(params.get("macd_signal", 9)),
        )
        bb = indicator_service.bollinger_bands(
            closes, int(params.get("bb_period", 20)), float(params.get("bb_std", 2.0)),
        )
        return {
            "closes": closes,
            "macd": m["macd"], "signal": m["signal"],
            "bb_lower": bb["lower"], "bb_upper": bb["upper"],
        }

    def evaluate_at(self, i, candles, pre, params):
        price = pre["closes"][i]
        ts = datetime.fromisoformat(candles[i]["time"])
        cm, cs_ = pre["macd"][i], pre["signal"][i]
        pm, ps_ = pre["macd"][i - 1], pre["signal"][i - 1]
        bl, bu = pre["bb_lower"][i], pre["bb_upper"][i]
        if None in (cm, cs_, pm, ps_, bl, bu):
            return Signal(SignalType.HOLD, price, ts)
        bw = bu - bl
        nt = bw * 0.25
        if pm <= ps_ and cm > cs_ and price < bl + nt:
            return Signal(SignalType.BUY, price, ts, "MACD cross UP near lower BB")
        if pm >= ps_ and cm < cs_ and price > bu - nt:
            return Signal(SignalType.SELL, price, ts, "MACD cross DOWN near upper BB")
        return Signal(SignalType.HOLD, price, ts)


class TripleScreenStrategy(BaseStrategy):
    """Alexander Elder's Triple Screen: EMA trend + MACD momentum + RSI entry."""

    def name(self) -> str:
        return "triple_screen"

    def description(self) -> str:
        return "EMA trend + MACD momentum + RSI entry — three-layer confirmation."

    def required_params(self) -> List[dict]:
        return [
            {"name": "trend_ema", "type": "int", "label": "Trend EMA Period", "default": 50},
            {"name": "macd_fast", "type": "int", "label": "MACD Fast", "default": 12},
            {"name": "macd_slow", "type": "int", "label": "MACD Slow", "default": 26},
            {"name": "macd_signal", "type": "int", "label": "MACD Signal", "default": 9},
            {"name": "rsi_period", "type": "int", "label": "RSI Period", "default": 14},
            {"name": "rsi_buy_zone", "type": "float", "label": "RSI Buy Zone (below)", "default": 45.0},
            {"name": "rsi_sell_zone", "type": "float", "label": "RSI Sell Zone (above)", "default": 55.0},
        ]

    def default_params(self) -> dict:
        return {
            "trend_ema": 50, "macd_fast": 12, "macd_slow": 26,
            "macd_signal": 9, "rsi_period": 14, "rsi_buy_zone": 45.0, "rsi_sell_zone": 55.0,
        }

    def evaluate(self, candles: List[dict], params: dict) -> Signal:
        trend_period = int(params.get("trend_ema", 50))
        macd_fast = int(params.get("macd_fast", 12))
        macd_slow = int(params.get("macd_slow", 26))
        macd_sig = int(params.get("macd_signal", 9))
        rsi_period = int(params.get("rsi_period", 14))
        rsi_buy_zone = float(params.get("rsi_buy_zone", 45.0))
        rsi_sell_zone = float(params.get("rsi_sell_zone", 55.0))
        closes = [c["close"] for c in candles]
        current_price = closes[-1]
        ts = datetime.fromisoformat(candles[-1]["time"])

        min_bars = max(trend_period + 1, macd_slow + macd_sig + 1, rsi_period + 2)
        if len(closes) < min_bars:
            return Signal(SignalType.HOLD, current_price, ts, "Not enough data")

        trend_ema = indicator_service.ema(closes, trend_period)
        macd_data = indicator_service.macd(closes, macd_fast, macd_slow, macd_sig)
        rsi_values = indicator_service.rsi(closes, rsi_period)

        curr_ema = trend_ema[-1]
        curr_hist = macd_data["histogram"][-1]
        prev_hist = macd_data["histogram"][-2]
        curr_rsi = rsi_values[-1]
        prev_rsi = rsi_values[-2]

        if None in (curr_ema, curr_hist, prev_hist, curr_rsi, prev_rsi):
            return Signal(SignalType.HOLD, current_price, ts, "Indicators not ready")

        uptrend = current_price > curr_ema
        downtrend = current_price < curr_ema
        macd_bullish = curr_hist > prev_hist
        macd_bearish = curr_hist < prev_hist
        rsi_buy_entry = prev_rsi <= rsi_buy_zone and curr_rsi > rsi_buy_zone
        rsi_sell_entry = prev_rsi >= rsi_sell_zone and curr_rsi < rsi_sell_zone

        if uptrend and macd_bullish and rsi_buy_entry:
            return Signal(SignalType.BUY, current_price, ts,
                          f"Triple screen BUY: uptrend + MACD rising + RSI pullback ({curr_rsi:.1f})")
        if downtrend and macd_bearish and rsi_sell_entry:
            return Signal(SignalType.SELL, current_price, ts,
                          f"Triple screen SELL: downtrend + MACD falling + RSI pullback ({curr_rsi:.1f})")

        return Signal(SignalType.HOLD, current_price, ts)

    def precompute(self, candles, params):
        closes = [c["close"] for c in candles]
        m = indicator_service.macd(
            closes,
            int(params.get("macd_fast", 12)),
            int(params.get("macd_slow", 26)),
            int(params.get("macd_signal", 9)),
        )
        return {
            "closes": closes,
            "trend_ema": indicator_service.ema(closes, int(params.get("trend_ema", 50))),
            "histogram": m["histogram"],
            "rsi": indicator_service.rsi(closes, int(params.get("rsi_period", 14))),
        }

    def evaluate_at(self, i, candles, pre, params):
        price = pre["closes"][i]
        ts = datetime.fromisoformat(candles[i]["time"])
        rsi_buy_zone = float(params.get("rsi_buy_zone", 45.0))
        rsi_sell_zone = float(params.get("rsi_sell_zone", 55.0))
        ce = pre["trend_ema"][i]
        ch, ph = pre["histogram"][i], pre["histogram"][i - 1]
        cr, pr = pre["rsi"][i], pre["rsi"][i - 1]
        if None in (ce, ch, ph, cr, pr):
            return Signal(SignalType.HOLD, price, ts)
        uptrend = price > ce
        downtrend = price < ce
        if uptrend and ch > ph and pr <= rsi_buy_zone and cr > rsi_buy_zone:
            return Signal(SignalType.BUY, price, ts, f"Triple screen BUY ({cr:.1f})")
        if downtrend and ch < ph and pr >= rsi_sell_zone and cr < rsi_sell_zone:
            return Signal(SignalType.SELL, price, ts, f"Triple screen SELL ({cr:.1f})")
        return Signal(SignalType.HOLD, price, ts)


class EMAConfluenceStrategy(BaseStrategy):
    """Multi-EMA confluence: fast/medium/slow EMAs must be properly stacked
    + RSI confirmation to enter."""

    def name(self) -> str:
        return "ema_confluence"

    def description(self) -> str:
        return "Fast/Medium/Slow EMA stacking + RSI confirmation — high-confluence entries."

    def required_params(self) -> List[dict]:
        return [
            {"name": "fast_ema", "type": "int", "label": "Fast EMA", "default": 8},
            {"name": "medium_ema", "type": "int", "label": "Medium EMA", "default": 21},
            {"name": "slow_ema", "type": "int", "label": "Slow EMA", "default": 55},
            {"name": "rsi_period", "type": "int", "label": "RSI Period", "default": 14},
            {"name": "rsi_min_buy", "type": "float", "label": "RSI Min for Buy", "default": 40.0},
            {"name": "rsi_max_buy", "type": "float", "label": "RSI Max for Buy", "default": 70.0},
            {"name": "rsi_min_sell", "type": "float", "label": "RSI Min for Sell", "default": 30.0},
            {"name": "rsi_max_sell", "type": "float", "label": "RSI Max for Sell", "default": 60.0},
        ]

    def default_params(self) -> dict:
        return {
            "fast_ema": 8, "medium_ema": 21, "slow_ema": 55,
            "rsi_period": 14, "rsi_min_buy": 40.0, "rsi_max_buy": 70.0,
            "rsi_min_sell": 30.0, "rsi_max_sell": 60.0,
        }

    def evaluate(self, candles: List[dict], params: dict) -> Signal:
        fast_p = int(params.get("fast_ema", 8))
        med_p = int(params.get("medium_ema", 21))
        slow_p = int(params.get("slow_ema", 55))
        rsi_period = int(params.get("rsi_period", 14))
        rsi_min_buy = float(params.get("rsi_min_buy", 40.0))
        rsi_max_buy = float(params.get("rsi_max_buy", 70.0))
        rsi_min_sell = float(params.get("rsi_min_sell", 30.0))
        rsi_max_sell = float(params.get("rsi_max_sell", 60.0))
        closes = [c["close"] for c in candles]
        current_price = closes[-1]
        ts = datetime.fromisoformat(candles[-1]["time"])

        min_bars = max(slow_p + 1, rsi_period + 2)
        if len(closes) < min_bars:
            return Signal(SignalType.HOLD, current_price, ts, "Not enough data")

        fast_vals = indicator_service.ema(closes, fast_p)
        med_vals = indicator_service.ema(closes, med_p)
        slow_vals = indicator_service.ema(closes, slow_p)
        rsi_vals = indicator_service.rsi(closes, rsi_period)

        cf, cm, cs = fast_vals[-1], med_vals[-1], slow_vals[-1]
        pf, pm, ps = fast_vals[-2], med_vals[-2], slow_vals[-2]
        curr_rsi = rsi_vals[-1]

        if None in (cf, cm, cs, pf, pm, ps, curr_rsi):
            return Signal(SignalType.HOLD, current_price, ts, "Indicators not ready")

        bullish_stack = cf > cm > cs
        prev_not_bullish = not (pf > pm > ps)
        bearish_stack = cf < cm < cs
        prev_not_bearish = not (pf < pm < ps)

        if bullish_stack and prev_not_bullish and rsi_min_buy <= curr_rsi <= rsi_max_buy:
            return Signal(SignalType.BUY, current_price, ts,
                          f"EMA confluence BUY: {fast_p}>{med_p}>{slow_p}, RSI={curr_rsi:.1f}")
        if bearish_stack and prev_not_bearish and rsi_min_sell <= curr_rsi <= rsi_max_sell:
            return Signal(SignalType.SELL, current_price, ts,
                          f"EMA confluence SELL: {fast_p}<{med_p}<{slow_p}, RSI={curr_rsi:.1f}")

        return Signal(SignalType.HOLD, current_price, ts)

    def precompute(self, candles, params):
        closes = [c["close"] for c in candles]
        return {
            "closes": closes,
            "fast": indicator_service.ema(closes, int(params.get("fast_ema", 8))),
            "med": indicator_service.ema(closes, int(params.get("medium_ema", 21))),
            "slow": indicator_service.ema(closes, int(params.get("slow_ema", 55))),
            "rsi": indicator_service.rsi(closes, int(params.get("rsi_period", 14))),
        }

    def evaluate_at(self, i, candles, pre, params):
        price = pre["closes"][i]
        ts = datetime.fromisoformat(candles[i]["time"])
        rsi_min_buy = float(params.get("rsi_min_buy", 40.0))
        rsi_max_buy = float(params.get("rsi_max_buy", 70.0))
        rsi_min_sell = float(params.get("rsi_min_sell", 30.0))
        rsi_max_sell = float(params.get("rsi_max_sell", 60.0))
        cf, cm, cs_ = pre["fast"][i], pre["med"][i], pre["slow"][i]
        pf, pm, ps_ = pre["fast"][i - 1], pre["med"][i - 1], pre["slow"][i - 1]
        cr = pre["rsi"][i]
        if None in (cf, cm, cs_, pf, pm, ps_, cr):
            return Signal(SignalType.HOLD, price, ts)
        bullish = cf > cm > cs_
        was_not = not (pf > pm > ps_)
        if bullish and was_not and rsi_min_buy <= cr <= rsi_max_buy:
            return Signal(SignalType.BUY, price, ts, f"EMA confluence BUY (RSI={cr:.1f})")
        bearish = cf < cm < cs_
        was_not_b = not (pf < pm < ps_)
        if bearish and was_not_b and rsi_min_sell <= cr <= rsi_max_sell:
            return Signal(SignalType.SELL, price, ts, f"EMA confluence SELL (RSI={cr:.1f})")
        return Signal(SignalType.HOLD, price, ts)


class BollingerRSIStrategy(BaseStrategy):
    """Mean reversion: Buy when price touches lower Bollinger Band AND RSI is oversold."""

    def name(self) -> str:
        return "bollinger_rsi"

    def description(self) -> str:
        return "Mean reversion — Bollinger Band touch + RSI extreme confirmation."

    def required_params(self) -> List[dict]:
        return [
            {"name": "bb_period", "type": "int", "label": "Bollinger Period", "default": 20},
            {"name": "bb_std", "type": "float", "label": "Bollinger Std Dev", "default": 2.0},
            {"name": "rsi_period", "type": "int", "label": "RSI Period", "default": 14},
            {"name": "rsi_oversold", "type": "float", "label": "RSI Oversold", "default": 30.0},
            {"name": "rsi_overbought", "type": "float", "label": "RSI Overbought", "default": 70.0},
        ]

    def default_params(self) -> dict:
        return {"bb_period": 20, "bb_std": 2.0, "rsi_period": 14, "rsi_oversold": 30.0, "rsi_overbought": 70.0}

    def evaluate(self, candles: List[dict], params: dict) -> Signal:
        bb_period = int(params.get("bb_period", 20))
        bb_std = float(params.get("bb_std", 2.0))
        rsi_period = int(params.get("rsi_period", 14))
        oversold = float(params.get("rsi_oversold", 30.0))
        overbought = float(params.get("rsi_overbought", 70.0))
        closes = [c["close"] for c in candles]
        current_price = closes[-1]
        ts = datetime.fromisoformat(candles[-1]["time"])

        min_bars = max(bb_period + 1, rsi_period + 2)
        if len(closes) < min_bars:
            return Signal(SignalType.HOLD, current_price, ts, "Not enough data")

        bb = indicator_service.bollinger_bands(closes, bb_period, bb_std)
        rsi_vals = indicator_service.rsi(closes, rsi_period)
        curr_rsi = rsi_vals[-1]
        prev_rsi = rsi_vals[-2]
        lower = bb["lower"][-1]
        upper = bb["upper"][-1]

        if None in (curr_rsi, prev_rsi, lower, upper):
            return Signal(SignalType.HOLD, current_price, ts, "Indicators not ready")

        if current_price <= lower and prev_rsi <= oversold and curr_rsi > oversold:
            return Signal(SignalType.BUY, current_price, ts,
                          f"BB+RSI BUY: price at lower BB, RSI rising from {oversold}")
        if current_price >= upper and prev_rsi >= overbought and curr_rsi < overbought:
            return Signal(SignalType.SELL, current_price, ts,
                          f"BB+RSI SELL: price at upper BB, RSI falling from {overbought}")

        return Signal(SignalType.HOLD, current_price, ts)

    def precompute(self, candles, params):
        closes = [c["close"] for c in candles]
        bb = indicator_service.bollinger_bands(
            closes, int(params.get("bb_period", 20)), float(params.get("bb_std", 2.0)),
        )
        return {
            "closes": closes,
            "bb_lower": bb["lower"], "bb_upper": bb["upper"],
            "rsi": indicator_service.rsi(closes, int(params.get("rsi_period", 14))),
        }

    def evaluate_at(self, i, candles, pre, params):
        price = pre["closes"][i]
        ts = datetime.fromisoformat(candles[i]["time"])
        oversold = float(params.get("rsi_oversold", 30.0))
        overbought = float(params.get("rsi_overbought", 70.0))
        cr, pr = pre["rsi"][i], pre["rsi"][i - 1]
        bl, bu = pre["bb_lower"][i], pre["bb_upper"][i]
        if None in (cr, pr, bl, bu):
            return Signal(SignalType.HOLD, price, ts)
        if price <= bl and pr <= oversold and cr > oversold:
            return Signal(SignalType.BUY, price, ts, "BB+RSI BUY")
        if price >= bu and pr >= overbought and cr < overbought:
            return Signal(SignalType.SELL, price, ts, "BB+RSI SELL")
        return Signal(SignalType.HOLD, price, ts)


class ResearchGenomeStrategy(BaseStrategy):
    """Deterministic strategy driven by a structured research genome.

    The genome is intentionally explicit: thresholds, weights, filters, and
    SL/TP policy are data. This lets the research engine mutate and validate
    candidates without relying on natural-language interpretation.
    """

    def name(self) -> str:
        return "research_genome"

    def description(self) -> str:
        return "Structured weighted setup model for deterministic research candidates."

    def required_params(self) -> List[dict]:
        return [
            {"name": "min_score", "type": "float", "label": "Minimum score", "default": 3.0},
            {"name": "score_margin", "type": "float", "label": "Score margin", "default": 0.35},
            {"name": "trend_ema", "type": "int", "label": "Trend EMA", "default": 100},
            {"name": "fast_ema", "type": "int", "label": "Fast EMA", "default": 20},
            {"name": "medium_ema", "type": "int", "label": "Medium EMA", "default": 50},
            {"name": "rsi_period", "type": "int", "label": "RSI period", "default": 14},
            {"name": "rsi_buy_min", "type": "float", "label": "RSI buy min", "default": 42.0},
            {"name": "rsi_buy_max", "type": "float", "label": "RSI buy max", "default": 68.0},
            {"name": "rsi_sell_min", "type": "float", "label": "RSI sell min", "default": 32.0},
            {"name": "rsi_sell_max", "type": "float", "label": "RSI sell max", "default": 58.0},
            {"name": "sl_atr_multiplier", "type": "float", "label": "SL ATR", "default": 2.0},
            {"name": "tp_rr", "type": "float", "label": "TP risk/reward", "default": 1.7},
            {"name": "session_enabled", "type": "bool", "label": "Entry session filter", "default": False},
            {"name": "session_start_hour", "type": "int", "label": "Session start UTC", "default": 7},
            {"name": "session_end_hour", "type": "int", "label": "Session end UTC", "default": 20},
        ]

    def default_params(self) -> dict:
        return {
            "schema_version": 1,
            "min_score": 3.0,
            "score_margin": 0.35,
            "long_enabled": True,
            "short_enabled": True,
            "trend_ema": 100,
            "fast_ema": 20,
            "medium_ema": 50,
            "rsi_period": 14,
            "rsi_buy_min": 42.0,
            "rsi_buy_max": 68.0,
            "rsi_sell_min": 32.0,
            "rsi_sell_max": 58.0,
            "macd_fast": 12,
            "macd_slow": 26,
            "macd_signal": 9,
            "bb_period": 20,
            "bb_std": 2.0,
            "bb_near_pct": 0.35,
            "atr_period": 14,
            "atr_min_ratio": 0.55,
            "atr_max_ratio": 2.8,
            "breakout_lookback": 20,
            "sl_atr_multiplier": 2.0,
            "tp_rr": 1.7,
            "session_enabled": False,
            "session_start_hour": 7,
            "session_end_hour": 20,
            "weights": {
                "trend": 1.0,
                "ema_stack": 0.9,
                "rsi_zone": 0.8,
                "rsi_turn": 0.8,
                "macd": 0.8,
                "bb_pullback": 0.6,
                "breakout": 0.5,
                "candle": 0.35,
                "volatility": 0.45,
            },
        }

    def evaluate(self, candles: List[dict], params: dict) -> Signal:
        pre = self.precompute(candles, params)
        return self.evaluate_at(len(candles) - 1, candles, pre, params)

    def precompute(self, candles, params):
        p = self._merged(params)
        closes = [c["close"] for c in candles]
        macd_data = indicator_service.macd(
            closes,
            int(p["macd_fast"]),
            int(p["macd_slow"]),
            int(p["macd_signal"]),
        )
        bb = indicator_service.bollinger_bands(
            closes,
            int(p["bb_period"]),
            float(p["bb_std"]),
        )
        atr_vals = indicator_service.atr(candles, int(p["atr_period"]))
        return {
            "closes": closes,
            "fast": indicator_service.ema(closes, int(p["fast_ema"])),
            "medium": indicator_service.ema(closes, int(p["medium_ema"])),
            "trend": indicator_service.ema(closes, int(p["trend_ema"])),
            "rsi": indicator_service.rsi(closes, int(p["rsi_period"])),
            "macd_hist": macd_data["histogram"],
            "bb_lower": bb["lower"],
            "bb_upper": bb["upper"],
            "atr": atr_vals,
            "atr_avg": self._rolling_avg(atr_vals, 20),
            "highs": [c["high"] for c in candles],
            "lows": [c["low"] for c in candles],
        }

    def evaluate_at(self, i, candles, pre, params):
        p = self._merged(params)
        price = pre["closes"][i]
        ts = datetime.fromisoformat(candles[i]["time"])
        if p.get("session_enabled", False) and not self._within_session(ts, p):
            return Signal(SignalType.HOLD, price, ts, "Outside entry session", confidence=0.0)
        if i < max(int(p["trend_ema"]), int(p["bb_period"]), int(p["macd_slow"]) + int(p["macd_signal"]), 30):
            return Signal(SignalType.HOLD, price, ts, "Indicators not ready", confidence=0.0)

        values = [
            pre["fast"][i], pre["medium"][i], pre["trend"][i], pre["rsi"][i],
            pre["macd_hist"][i], pre["bb_lower"][i], pre["bb_upper"][i], pre["atr"][i],
        ]
        if any(v is None for v in values):
            return Signal(SignalType.HOLD, price, ts, "Indicators not ready", confidence=0.0)

        buy_score, buy_parts = self._score_side("buy", i, candles, pre, p)
        sell_score, sell_parts = self._score_side("sell", i, candles, pre, p)
        min_score = float(p["min_score"])
        margin = float(p["score_margin"])
        max_possible = max(sum(float(v) for v in p["weights"].values()), min_score)

        side = SignalType.HOLD
        score = max(buy_score, sell_score)
        parts: list[str] = []
        if p.get("long_enabled", True) and buy_score >= min_score and buy_score >= sell_score + margin:
            side = SignalType.BUY
            parts = buy_parts
        elif p.get("short_enabled", True) and sell_score >= min_score and sell_score >= buy_score + margin:
            side = SignalType.SELL
            parts = sell_parts
        else:
            return Signal(
                SignalType.HOLD,
                price,
                ts,
                f"No edge: buy_score={buy_score:.2f}, sell_score={sell_score:.2f}",
                confidence=round(min(score / max_possible, 1.0), 3),
            )

        atr = pre["atr"][i] or price * 0.005
        sl_dist = max(atr * float(p["sl_atr_multiplier"]), price * 0.0005)
        tp_dist = sl_dist * float(p["tp_rr"])
        if side == SignalType.BUY:
            sl = price - sl_dist
            tp = price + tp_dist
        else:
            sl = price + sl_dist
            tp = price - tp_dist
        confidence = round(min(score / max_possible, 1.0), 3)
        return Signal(
            side,
            price,
            ts,
            f"{side.value.upper()} score={score:.2f}: {', '.join(parts[:5])}",
            stop_loss=sl,
            take_profit=tp,
            confidence=confidence,
        )

    def _score_side(self, side: str, i: int, candles: List[dict], pre: dict, p: dict) -> tuple[float, list[str]]:
        w = p["weights"]
        price = pre["closes"][i]
        prev_price = pre["closes"][i - 1]
        fast, medium, trend = pre["fast"][i], pre["medium"][i], pre["trend"][i]
        pfast, pmedium = pre["fast"][i - 1], pre["medium"][i - 1]
        rsi, prev_rsi = pre["rsi"][i], pre["rsi"][i - 1]
        hist, prev_hist = pre["macd_hist"][i], pre["macd_hist"][i - 1]
        lower, upper = pre["bb_lower"][i], pre["bb_upper"][i]
        atr, atr_avg = pre["atr"][i], pre["atr_avg"][i]
        lookback = int(p["breakout_lookback"])
        start = max(0, i - lookback)
        prior_high = max(pre["highs"][start:i]) if i > start else candles[i]["high"]
        prior_low = min(pre["lows"][start:i]) if i > start else candles[i]["low"]

        score = 0.0
        parts: list[str] = []
        is_buy = side == "buy"

        def add(key: str, condition: bool, label: str) -> None:
            nonlocal score
            if condition:
                score += float(w.get(key, 0))
                parts.append(label)

        add("trend", price > trend if is_buy else price < trend, "trend")
        add("ema_stack", fast > medium if is_buy else fast < medium, "ema stack")
        add("rsi_zone", float(p["rsi_buy_min"] if is_buy else p["rsi_sell_min"]) <= rsi <= float(p["rsi_buy_max"] if is_buy else p["rsi_sell_max"]), "rsi zone")
        add("rsi_turn", rsi > prev_rsi if is_buy else rsi < prev_rsi, "rsi turn")
        add("macd", hist is not None and prev_hist is not None and (hist > prev_hist if is_buy else hist < prev_hist), "macd")
        if upper is not None and lower is not None and upper > lower:
            band_width = upper - lower
            add("bb_pullback", price <= lower + band_width * float(p["bb_near_pct"]) if is_buy else price >= upper - band_width * float(p["bb_near_pct"]), "bb pullback")
        add("breakout", price > prior_high and prev_price <= prior_high if is_buy else price < prior_low and prev_price >= prior_low, "breakout")
        add("candle", candles[i]["close"] > candles[i]["open"] if is_buy else candles[i]["close"] < candles[i]["open"], "candle")
        if atr and atr_avg and atr_avg > 0:
            ratio = atr / atr_avg
            add("volatility", float(p["atr_min_ratio"]) <= ratio <= float(p["atr_max_ratio"]), "volatility")
        if pfast is not None and pmedium is not None:
            add("ema_stack", pfast <= pmedium and fast > medium if is_buy else pfast >= pmedium and fast < medium, "ema cross")

        return score, parts

    def _merged(self, params: dict) -> dict:
        merged = self.default_params()
        for key, value in (params or {}).items():
            if key == "weights" and isinstance(value, dict):
                merged["weights"] = {**merged["weights"], **value}
            else:
                merged[key] = value
        return merged

    @staticmethod
    def _within_session(ts: datetime, p: dict) -> bool:
        if ts.tzinfo is not None:
            ts = ts.astimezone(timezone.utc)
        start = max(0, min(23, int(p.get("session_start_hour", 7))))
        end = max(0, min(24, int(p.get("session_end_hour", 20))))
        hour = ts.hour
        if start == end:
            return True
        if start < end:
            return start <= hour < end
        return hour >= start or hour < end

    @staticmethod
    def _rolling_avg(values: List[Optional[float]], period: int) -> List[Optional[float]]:
        result: List[Optional[float]] = [None] * len(values)
        window: list[float] = []
        for i, value in enumerate(values):
            if value is not None:
                window.append(value)
            if len(window) > period:
                window.pop(0)
            if len(window) == period:
                result[i] = sum(window) / period
        return result


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

STRATEGIES: dict[str, BaseStrategy] = {
    # Single-indicator
    "sma_crossover": SMACrossoverStrategy(),
    "rsi_reversal": RSIReversalStrategy(),
    "macd_crossover": MACDCrossoverStrategy(),
    # Multi-indicator composite
    "sma_rsi": SMARSIStrategy(),
    "macd_bbands": MACDBollingerStrategy(),
    "triple_screen": TripleScreenStrategy(),
    "ema_confluence": EMAConfluenceStrategy(),
    "bollinger_rsi": BollingerRSIStrategy(),
    "research_genome": ResearchGenomeStrategy(),
}


def get_strategy(name: str) -> BaseStrategy | None:
    return STRATEGIES.get(name)


def list_strategies() -> list[dict]:
    return [
        {
            "name": s.name(),
            "description": s.description(),
            "params": s.required_params(),
            "defaults": s.default_params(),
        }
        for s in STRATEGIES.values()
    ]
