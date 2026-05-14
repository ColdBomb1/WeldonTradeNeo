"""AI rule-based strategy engine.

The configured AI provider interprets natural language rules
against market data to make trade decisions. Rules evolve through training.
"""

from __future__ import annotations

import logging
import math
from datetime import datetime, timezone
from typing import Any

from config import load_config
from db import get_session
from models.candle import Candle
from models.ruleset import RuleSet
from models.training import TrainingRun, TrainingIteration
from services import ai_service, indicator_service, lessons_service

logger = logging.getLogger(__name__)

# Stop flags
_active_trains: dict[int, bool] = {}
_active_backtests: dict[int, bool] = {}

# Live progress stores — polled by API
_backtest_progress: dict[int, dict] = {}
_training_progress: dict[int, dict] = {}

_next_backtest_id = 0


def _new_backtest_id() -> int:
    global _next_backtest_id
    _next_backtest_id += 1
    return _next_backtest_id


def _is_stopped(run_id: int) -> bool:
    return _active_trains.get(run_id, False)


def request_stop(run_id: int):
    _active_trains[run_id] = True


def request_stop_backtest(bt_id: int):
    _active_backtests[bt_id] = True


def get_backtest_progress(bt_id: int) -> dict | None:
    return _backtest_progress.get(bt_id)


def get_training_progress(run_id: int) -> dict | None:
    return _training_progress.get(run_id)


# ---------------------------------------------------------------------------
# Rule evaluation
# ---------------------------------------------------------------------------

# Module-level constant so it's one stable string object across calls. Content
# identical across every evaluate_rules invocation — cached on the API side
# by the cache_control mark in evaluate_rules().
_EVAL_SYSTEM_PROMPT = (
    "You are a discretionary forex trader analyzing current market conditions.\n\n"
    "THE CORE TASK: Pattern-match the current moment against the named SETUP TEMPLATES "
    "in the framework. When a template is a clean match (4-5 of its conditions present), "
    "recommend the trade. When no template fits, HOLD.\n\n"
    "Both types of errors cost money:\n"
    "- Taking a bad trade loses the SL amount.\n"
    "- Missing a real opportunity loses the move you didn't capture.\n"
    "Neither answer is 'safer'. Recommend the setup when you see one. HOLD when you don't.\n\n"
    "MULTI-TIMEFRAME ANALYSIS (critical):\n"
    "The prompt contains 4h and 1h higher-timeframe summaries. Use them to form "
    "DIRECTIONAL BIAS FIRST:\n"
    "  1. Look at the 4h: is it UPTREND, DOWNTREND, or RANGE? That's your primary bias.\n"
    "  2. Look at the 1h: is it aligned with the 4h, or showing a pullback/consolidation "
    "  within the 4h trend?\n"
    "  3. Only THEN look at the 15m entry timeframe for a trigger.\n"
    "A 15m setup that CONTRADICTS the 1h/4h structure is usually a trap. A 15m setup that "
    "ALIGNS with higher-TF bias is where real opportunities live. State your 4h/1h read "
    "explicitly in your reasoning, then explain why the 15m is or isn't confirming it.\n\n"
    "REPORT YOUR REAL CONVICTION. The confidence field is your honest estimate of setup "
    "quality on a 0.0-1.0 scale. Do not inflate it to justify a trade. Do not deflate it "
    "to avoid recommending one. Calibration matters.\n\n"
    "HOW TO USE RECENT TRADE MEMORY:\n"
    "- Lessons are DATA POINTS about patterns that worked or failed, not prohibitions.\n"
    "- A setup that rhymes with a past loss is worth extra scrutiny - but if the CONTEXT "
    "(trend direction, momentum regime, volatility, session) differs materially, the "
    "lesson does not apply.\n"
    "- If the memory block says no prior trades exist, operate from first principles. "
    "Do not manufacture caution.\n\n"
    "ANTI-HALLUCINATION: When you cite an indicator value in your reasoning (e.g. \"RSI "
    "at 54\"), quote it ACCURATELY from the indicators block. Past entries where reasoning "
    "claimed one value and the data showed another have lost money. If you're not sure of "
    "an exact value, say \"near X\" rather than a precise number. Be honest about what the "
    "data actually shows, not what you wish it showed.\n\n"
    "Indicators include pre-computed boolean flags. Trust them as facts - but a true flag "
    "is a DATA POINT, not a trigger. You decide whether it matters in this context.\n\n"
    "Always respond with valid JSON only."
)


def evaluate_rules(
    rules_text: str,
    symbol: str,
    timeframe: str,
    candles: list[dict],
    indicators: dict[str, Any] | None = None,
    news_events: list[dict] | None = None,
    sentiment: dict | None = None,
    lessons_text: str | None = None,
    multi_tf_candles: dict[str, list[dict]] | None = None,
) -> dict:
    """Evaluate current market state using the configured AI as a discretionary analyst.

    Rules are treated as a considerations framework + named setup templates —
    not hard triggers. Recent trade lessons are injected as memory, and when
    `multi_tf_candles` is provided (e.g. {'1h': [...], '4h': [...]}), higher-
    timeframe context is included so the model can form directional bias before
    scoring the entry-timeframe trigger.

    Returns: {"signal": "BUY/SELL/HOLD", "confidence": 0-1,
              "stop_loss": float|None, "take_profit": float|None,
              "reasoning": str}
    """
    cfg = load_config()
    if not ai_service.is_enabled(cfg):
        return {"signal": "HOLD", "confidence": 0, "reasoning": "AI provider not available"}

    # Build context
    if indicators is None:
        indicators = _compute_indicators(candles)

    # Pull lessons from recent closed trades (unless caller supplied their own,
    # e.g. backtest mode where the DB contains no relevant history).
    if lessons_text is None:
        try:
            lessons_text = lessons_service.format_lessons_for_prompt(symbol, days=14)
        except Exception as exc:
            logger.warning("Lessons fetch failed for %s: %s", symbol, exc)
            lessons_text = ""

    static_ctx = _build_static_context(rules_text)
    dynamic_ctx = _build_dynamic_context(
        symbol, timeframe, candles, indicators, news_events, sentiment,
        lessons_text=lessons_text,
        multi_tf_candles=multi_tf_candles,
    )

    try:
        parsed = ai_service.generate_json(
            system=_EVAL_SYSTEM_PROMPT,
            prompt=f"{static_ctx}\n\n{dynamic_ctx}",
            cfg=cfg,
            max_tokens=1024,
            temperature=0.2,
            fallback=None,
        )
        if parsed is None:
            return {"signal": "HOLD", "confidence": 0, "reasoning": "Could not parse response"}

        parsed.setdefault("signal", "HOLD")
        parsed.setdefault("confidence", 0)
        parsed.setdefault("reasoning", "")
        parsed["signal"] = parsed["signal"].upper()
        if parsed["signal"] not in ("BUY", "SELL", "HOLD"):
            parsed["signal"] = "HOLD"
        return parsed
    except Exception as exc:
        logger.error("Rule evaluation error: %s", exc)
        return {"signal": "HOLD", "confidence": 0, "reasoning": f"Error: {exc}"}


def _build_static_context(rules_text: str) -> str:
    """Static portion of the evaluation prompt: rules + analysis procedure +
    output schema. Identical across every call in a bar-close burst, so it
    lives behind a cache_control mark in evaluate_rules().
    """
    return (
        f"CONSIDERATIONS & FRAMEWORK:\n{rules_text}\n\n"
        f"ANALYSIS STEPS:\n"
        f"  1. Read the 4h/1h context (in the market data below). Is the broader trend UP, DOWN, or RANGE? Note key levels.\n"
        f"  2. Check which SETUP TEMPLATE (from the framework above) best matches THIS moment.\n"
        f"  3. Score the match: how many of that template's conditions are actually present?\n"
        f"  4. If 4+ conditions match cleanly, recommend the trade. If 3 or fewer, HOLD.\n\n"
        f"Reminders:\n"
        f"  - Your confidence is your honest estimate of setup quality on 0.0-1.0. Report what you "
        f"actually see.\n"
        f"  - Missing a real opportunity costs the same as taking a bad trade.\n"
        f"  - Quote indicator values from the data provided - do not cite values from memory or guess.\n\n"
        f"Respond with JSON (keep reasoning short - we store and read it):\n"
        f'{{\n'
        f'  "signal": "BUY" | "SELL" | "HOLD",\n'
        f'  "confidence": 0.0-1.0,\n'
        f'  "stop_loss": <price or null>,\n'
        f'  "take_profit": <price or null>,\n'
        f'  "reasoning": "<=200 chars: 4h/1h read, template matched (or none), and decision rationale"\n'
        f'}}'
    )


def _build_dynamic_context(
    symbol: str,
    timeframe: str,
    candles: list[dict],
    indicators: dict[str, Any],
    news_events: list[dict] | None,
    sentiment: dict | None,
    lessons_text: str | None = None,
    multi_tf_candles: dict[str, list[dict]] | None = None,
) -> str:
    """Dynamic per-call portion of the prompt: current market data, candles,
    indicators, events, sentiment, and per-symbol lessons. Placed AFTER the
    cached static context so different symbols share the cache prefix.
    """
    recent = candles[-30:] if len(candles) > 30 else candles
    candle_text = "\n".join(
        "  {time}: O={o:.5f} H={h:.5f} L={l:.5f} C={c:.5f}".format(
            time=c.get("time", "?"), o=c["open"], h=c["high"], l=c["low"], c=c["close"],
        )
        for c in recent
    )

    # Higher-timeframe block (1h, 4h) when available
    mtf_block = _format_multi_tf_block(symbol, multi_tf_candles or {})

    # Separate cross-pair context from technical indicators
    cross_pair_keys = {"cross_pair_summary", "usd_bias"}
    tech_indicators = {k: v for k, v in indicators.items() if k not in cross_pair_keys}
    indicator_text = "\n".join(f"  {k}: {v}" for k, v in tech_indicators.items())

    cross_pair_text = ""
    if "usd_bias" in indicators:
        cross_pair_text = f"\nCross-pair USD bias: {indicators['usd_bias']}\n"
        summaries = indicators.get("cross_pair_summary", [])
        if summaries:
            cross_pair_text += "Other pairs:\n" + "\n".join(
                "  {pair}: {dir} ({chg:+.3f}%){rsi}".format(
                    pair=s["pair"], dir=s["direction"], chg=s["change_pct"],
                    rsi=f", RSI={s['rsi']}" if s.get("rsi") else "",
                )
                for s in summaries
            ) + "\n"

    events_text = "  None"
    if news_events:
        events_text = "\n".join(
            "  {title} ({ccy}, {imp}) in {mins} min".format(
                title=e["title"], ccy=e["currency"], imp=e["impact"],
                mins=e.get("minutes_until", "?"),
            )
            for e in news_events[:5]
        )

    sent_text = "No data"
    if sentiment and sentiment.get("score") is not None:
        score = sentiment["score"]
        label = "bullish" if score > 0.2 else "bearish" if score < -0.2 else "neutral"
        sent_text = f"{label} ({score:+.2f})"

    news_guidance = ""
    if news_events:
        news_guidance = (
            "NEWS HANDLING: Only skip a trade due to news if a HIGH-impact event "
            "is within 15 minutes. Events 30+ minutes away should NOT affect your decision. "
            "Already-released events are informational only — do not skip trades because of them.\n\n"
        )

    lessons_block = ""
    if lessons_text:
        lessons_block = f"{lessons_text}\n\n"

    now_utc = datetime.now(timezone.utc)
    session_label = _classify_fx_session(now_utc)

    return (
        f"{lessons_block}"
        f"CURRENT MARKET DATA:\n"
        f"Current UTC time: {now_utc.strftime('%Y-%m-%dT%H:%M:%SZ')} ({session_label})\n"
        f"Symbol: {symbol}, Entry Timeframe: {timeframe}\n"
        f"Current price: {candles[-1]['close']:.5f}\n\n"
        f"{mtf_block}"
        f"ENTRY-TIMEFRAME DATA ({timeframe}) - use AFTER forming higher-TF bias above:\n"
        f"Recent {timeframe} candles:\n{candle_text}\n\n"
        f"{timeframe} indicators (data points - interpret them, don't just react to them):\n{indicator_text}\n\n"
        f"{cross_pair_text}"
        f"Nearby economic events:\n{events_text}\n\n"
        f"{news_guidance}"
        f"Market sentiment: {sent_text}\n"
    )


def _classify_fx_session(now_utc: datetime) -> str:
    """Label the current FX session based on UTC hour. Use this — do not let
    Claude infer it from candle timestamps, which it has been getting wrong.

    Reference (UTC): Tokyo 00-09, London 07-16, NY 13-22, Sydney 22-07.
    The London-NY overlap (13-16 UTC) carries the most volume.
    """
    h = now_utc.hour
    if 13 <= h < 16:
        return "London-NY overlap, peak volume window"
    if 7 <= h < 13:
        return "London session active, NY pre-market"
    if 16 <= h < 21:
        return "NY session, post-London"
    if 21 <= h < 24:
        return "post-NY, Sydney/Asian opening, low liquidity"
    if 0 <= h < 7:
        return "Asian session (Tokyo/Sydney), pre-London"
    return "session transition"


def _compute_indicators(candles: list[dict]) -> dict:
    """Compute indicator set with current AND previous values for crossover detection."""
    closes = [c["close"] for c in candles]
    result = {}

    # RSI with previous values and pre-computed crossover flags
    rsi = indicator_service.rsi(closes, 14)
    if rsi and rsi[-1] is not None:
        result["rsi_14"] = round(rsi[-1], 2)
        if len(rsi) >= 2 and rsi[-2] is not None:
            result["rsi_14_prev"] = round(rsi[-2], 2)
            # Pre-compute crossover flags so Claude doesn't have to
            if rsi[-2] < 30 and rsi[-1] >= 30:
                result["rsi_crossed_above_30"] = True
            if rsi[-2] > 70 and rsi[-1] <= 70:
                result["rsi_crossed_below_70"] = True
            if rsi[-2] < 50 and rsi[-1] >= 50:
                result["rsi_crossed_above_50"] = True
            if rsi[-2] > 50 and rsi[-1] <= 50:
                result["rsi_crossed_below_50"] = True
        # Recent RSI range (was RSI recently extreme?)
        recent_rsi = [v for v in rsi[-10:] if v is not None]
        if recent_rsi:
            result["rsi_min_10bars"] = round(min(recent_rsi), 2)
            result["rsi_max_10bars"] = round(max(recent_rsi), 2)

    # ATR with average
    atr = indicator_service.atr(candles, 14)
    if atr and atr[-1] is not None:
        result["atr_14"] = round(atr[-1], 6)
        atr_clean = [v for v in atr if v is not None]
        if len(atr_clean) >= 20:
            result["atr_20_avg"] = round(sum(atr_clean[-20:]) / 20, 6)
            result["atr_vs_avg"] = round(atr[-1] / (sum(atr_clean[-20:]) / 20), 2)

    # SMAs
    for period in (20, 50, 200):
        sma = indicator_service.sma(closes, period)
        if sma and sma[-1] is not None:
            result[f"sma_{period}"] = round(sma[-1], 5)

    # EMAs with slope detection
    ema_values = {}
    for period in (20, 50, 200):
        ema = indicator_service.ema(closes, period)
        if ema and ema[-1] is not None:
            result[f"ema_{period}"] = round(ema[-1], 5)
            ema_values[period] = ema[-1]
            if len(ema) >= 2 and ema[-2] is not None:
                result[f"ema_{period}_prev"] = round(ema[-2], 5)

    # Pre-compute EMA relationships
    if 20 in ema_values and 50 in ema_values:
        result["ema_20_above_50"] = ema_values[20] > ema_values[50]
    if 50 in ema_values and 200 in ema_values:
        result["ema_50_above_200"] = ema_values[50] > ema_values[200]

    # Price vs EMAs
    price = closes[-1]
    for period in (20, 50, 200):
        if period in ema_values:
            result[f"price_above_ema_{period}"] = price > ema_values[period]

    # Bollinger Bands with position
    bb = indicator_service.bollinger_bands(closes, 20, 2.0)
    if bb and bb["upper"][-1] is not None:
        result["bb_upper"] = round(bb["upper"][-1], 5)
        result["bb_lower"] = round(bb["lower"][-1], 5)
        result["bb_middle"] = round(bb["middle"][-1], 5)
        bb_width = bb["upper"][-1] - bb["lower"][-1]
        if bb_width > 0:
            result["bb_position"] = round((price - bb["lower"][-1]) / bb_width, 2)
        result["price_near_bb_upper"] = price >= bb["upper"][-1] * 0.999
        result["price_near_bb_lower"] = price <= bb["lower"][-1] * 1.001

    # MACD with crossover detection
    macd = indicator_service.macd(closes)
    if macd and macd["macd"][-1] is not None:
        result["macd"] = round(macd["macd"][-1], 6)
        result["macd_signal"] = round(macd["signal"][-1], 6)
        result["macd_histogram"] = round(macd["histogram"][-1], 6)
        if len(macd["histogram"]) >= 2 and macd["histogram"][-2] is not None:
            result["macd_histogram_prev"] = round(macd["histogram"][-2], 6)
            if macd["histogram"][-2] < 0 and macd["histogram"][-1] >= 0:
                result["macd_crossed_bullish"] = True
            if macd["histogram"][-2] > 0 and macd["histogram"][-1] <= 0:
                result["macd_crossed_bearish"] = True

    # Price action context
    result["current_price"] = price
    result["prev_close"] = closes[-2] if len(closes) >= 2 else price
    result["candle_direction"] = "bullish" if candles[-1]["close"] > candles[-1]["open"] else "bearish"
    if len(candles) >= 2:
        result["prev_candle_direction"] = "bullish" if candles[-2]["close"] > candles[-2]["open"] else "bearish"

    if len(closes) >= 20:
        result["high_20"] = round(max(c["high"] for c in candles[-20:]), 5)
        result["low_20"] = round(min(c["low"] for c in candles[-20:]), 5)
        result["new_20bar_high"] = candles[-1]["high"] >= result["high_20"]
        result["new_20bar_low"] = candles[-1]["low"] <= result["low_20"]

    # Volatility spike detection (proxy for news events when no calendar data)
    if "atr_14" in result and "atr_20_avg" in result and result["atr_20_avg"] > 0:
        current_range = candles[-1]["high"] - candles[-1]["low"]
        if current_range > result["atr_14"] * 2.5:
            result["volatility_spike"] = True
            result["volatility_spike_ratio"] = round(current_range / result["atr_14"], 2)

    # Trading session detection
    bar_time = candles[-1].get("time", "")
    if bar_time:
        try:
            ts = datetime.fromisoformat(bar_time)
            hour_utc = ts.hour
            # Sessions (UTC)
            if 7 <= hour_utc < 16:
                result["session"] = "London"
            if 12 <= hour_utc < 21:
                result["session"] = "London_NY_overlap" if 12 <= hour_utc < 16 else "New_York"
            elif 0 <= hour_utc < 8:
                result["session"] = "Asian"
            elif hour_utc >= 21 or hour_utc < 0:
                result["session"] = "off_hours"
            # Day of week
            result["day_of_week"] = ts.strftime("%A")
            result["is_weekend_close"] = ts.weekday() == 4 and hour_utc >= 20
        except (ValueError, AttributeError):
            pass

    return result


def _pip_value_for(symbol: str) -> float:
    """Match trade_manager._pip_value — kept local to avoid cross-module import."""
    return 0.01 if "JPY" in symbol.upper() else 0.0001


def _compute_tf_summary(symbol: str, candles: list[dict]) -> dict:
    """Compute a compact higher-timeframe summary block for inclusion in prompts.

    Returns a dict with:
      - 'ohlc_rows':    last 10 bars formatted for the prompt
      - 'trend':        'UPTREND' | 'DOWNTREND' | 'RANGE'
      - 'key_indicators': one-line indicator snapshot
      - 'structure':    swing high/low with pip distance from current price
      - 'last_time':    timestamp of latest bar
    """
    if not candles:
        return {}
    closes = [c["close"] for c in candles]
    price = closes[-1]
    pip = _pip_value_for(symbol)

    # Indicator snapshot
    ema20 = indicator_service.ema(closes, 20)
    ema50 = indicator_service.ema(closes, 50) if len(closes) >= 50 else None
    ema200 = indicator_service.ema(closes, 200) if len(closes) >= 200 else None
    rsi = indicator_service.rsi(closes, 14)
    macd = indicator_service.macd(closes)
    atr = indicator_service.atr(candles, 14)

    e20 = ema20[-1] if ema20 and ema20[-1] is not None else None
    e50 = ema50[-1] if ema50 and ema50[-1] is not None else None
    e200 = ema200[-1] if ema200 and ema200[-1] is not None else None
    r = rsi[-1] if rsi and rsi[-1] is not None else None
    mh = macd["histogram"][-1] if macd and macd["histogram"][-1] is not None else None
    a = atr[-1] if atr and atr[-1] is not None else None

    atr_avg = None
    atr_ratio = None
    if atr:
        atr_clean = [v for v in atr if v is not None]
        if len(atr_clean) >= 20:
            atr_avg = sum(atr_clean[-20:]) / 20
            if atr_avg > 0 and a is not None:
                atr_ratio = a / atr_avg

    # Trend classification from EMA alignment + price location
    trend = "RANGE"
    if e20 is not None and e50 is not None:
        if e200 is not None:
            if e20 > e50 > e200 and price > e50:
                trend = "UPTREND"
            elif e20 < e50 < e200 and price < e50:
                trend = "DOWNTREND"
        else:
            if e20 > e50 and price > e50:
                trend = "UPTREND"
            elif e20 < e50 and price < e50:
                trend = "DOWNTREND"

    # Structure: swing high / swing low over last 20 bars
    window = candles[-20:] if len(candles) >= 20 else candles
    swing_high = max(c["high"] for c in window)
    swing_low = min(c["low"] for c in window)
    sh_dist = (swing_high - price) / pip
    sl_dist = (price - swing_low) / pip

    # Last 10 bars text
    recent = candles[-10:]
    ohlc_rows = "\n".join(
        "    {time}: O={o:.5f} H={h:.5f} L={l:.5f} C={c:.5f}".format(
            time=c.get("time", "?"), o=c["open"], h=c["high"], l=c["low"], c=c["close"],
        )
        for c in recent
    )

    # One-line indicator text
    parts = []
    if e20 is not None:
        parts.append(f"ema_20={e20:.5f}")
    if e50 is not None:
        parts.append(f"ema_50={e50:.5f}")
    if e200 is not None:
        parts.append(f"ema_200={e200:.5f}")
    if r is not None:
        parts.append(f"rsi_14={r:.2f}")
    if mh is not None:
        parts.append(f"macd_hist={mh:+.6f}")
    if atr_ratio is not None:
        parts.append(f"atr_vs_avg={atr_ratio:.2f}x")
    key_indicators = ", ".join(parts) if parts else "(insufficient history)"

    structure = (
        f"swing_high={swing_high:.5f} ({sh_dist:+.1f} pips), "
        f"swing_low={swing_low:.5f} ({sl_dist:+.1f} pips below)"
    )

    return {
        "ohlc_rows": ohlc_rows,
        "trend": trend,
        "key_indicators": key_indicators,
        "structure": structure,
        "last_time": candles[-1].get("time", "?"),
    }


def _format_multi_tf_block(symbol: str, multi_tf_candles: dict[str, list[dict]]) -> str:
    """Format 4h/1h context summaries for the prompt. Returns empty string if nothing to show."""
    if not multi_tf_candles:
        return ""
    # Show 4h first (broader context), then 1h
    order = ["4h", "1h"]
    sections = []
    for tf in order:
        bars = multi_tf_candles.get(tf)
        if not bars or len(bars) < 20:
            continue
        s = _compute_tf_summary(symbol, bars)
        if not s:
            continue
        sections.append(
            f"  {tf} (as of {s['last_time']}, trend={s['trend']}):\n"
            f"    Structure: {s['structure']}\n"
            f"    Indicators: {s['key_indicators']}\n"
            f"    Last 10 bars:\n{s['ohlc_rows']}"
        )
    if not sections:
        return ""
    return "HIGHER TIMEFRAME CONTEXT:\n" + "\n\n".join(sections) + "\n\n"


def compute_cross_pair_context(
    symbol: str,
    all_candles: dict[str, list[dict]],
    bar_index: int | None = None,
) -> dict:
    """Compute cross-pair context showing what other USD pairs are doing.

    Gives Claude the 'USD is strengthening across the board' insight.
    all_candles: {symbol: [candle_dicts]} for all configured pairs.
    """
    result = {}
    base = symbol[:3]
    quote = symbol[3:6]

    pair_summaries = []
    usd_strength_signals = 0  # positive = USD strengthening
    total_pairs = 0

    for pair, candles in all_candles.items():
        if pair == symbol or not candles:
            continue
        # Use last 20 candles for momentum
        recent = candles[-20:] if len(candles) >= 20 else candles
        if len(recent) < 2:
            continue

        first_close = recent[0]["close"]
        last_close = recent[-1]["close"]
        change_pct = (last_close - first_close) / first_close * 100

        pair_base = pair[:3]
        pair_quote = pair[3:6]

        direction = "up" if change_pct > 0.05 else "down" if change_pct < -0.05 else "flat"

        # Compute RSI for the pair
        closes = [c["close"] for c in candles]
        rsi_vals = indicator_service.rsi(closes, 14)
        pair_rsi = round(rsi_vals[-1], 1) if rsi_vals and rsi_vals[-1] is not None else None

        pair_summaries.append({
            "pair": pair,
            "direction": direction,
            "change_pct": round(change_pct, 3),
            "rsi": pair_rsi,
            "price": last_close,
        })

        # Track USD strength
        total_pairs += 1
        if "USD" in pair:
            if pair_quote == "USD":
                # pair like EURUSD — if down, USD is strengthening
                usd_strength_signals += (-1 if change_pct > 0 else 1 if change_pct < 0 else 0)
            elif pair_base == "USD":
                # pair like USDJPY — if up, USD is strengthening
                usd_strength_signals += (1 if change_pct > 0 else -1 if change_pct < 0 else 0)

    if pair_summaries:
        result["cross_pair_summary"] = pair_summaries

    if total_pairs > 0:
        if usd_strength_signals > 0:
            result["usd_bias"] = "strengthening"
        elif usd_strength_signals < 0:
            result["usd_bias"] = "weakening"
        else:
            result["usd_bias"] = "neutral"

    return result


# ---------------------------------------------------------------------------
# Batch evaluation (20-50x faster backtesting)
# ---------------------------------------------------------------------------

def evaluate_rules_batch(
    rules_text: str,
    symbol: str,
    timeframe: str,
    candles: list[dict],
    chunk_candles: list[dict],
    indicators_at_start: dict[str, Any],
    position_state: str,
    news_events: list[dict] | None = None,
) -> list[dict]:
    """Evaluate rules across a batch of candles in a single AI call.

    Includes indicator snapshots at start, middle, and end of the chunk
    so the model can accurately detect crossovers throughout the window.

    Returns list of signal dicts: [{"bar_index": int, "signal": "BUY/SELL", ...}]
    """
    cfg = load_config()
    if not ai_service.is_enabled(cfg):
        return []

    # Format chunk candles compactly
    chunk_text = "\n".join(
        "  [{i}] {time}: O={o:.5f} H={h:.5f} L={l:.5f} C={c:.5f}".format(
            i=idx, time=c.get("time", "?"), o=c["open"], h=c["high"], l=c["low"], c=c["close"],
        )
        for idx, c in enumerate(chunk_candles)
    )

    # Compute indicator snapshots at start, middle, and end of chunk
    # This keeps indicators fresh even with larger batch sizes
    base_len = len(candles) - len(chunk_candles)
    snapshot_points = [0]
    if len(chunk_candles) > 10:
        snapshot_points.append(len(chunk_candles) // 2)
    snapshot_points.append(len(chunk_candles) - 1)

    indicator_sections = []
    for sp_idx, sp in enumerate(snapshot_points):
        label = ["Start", "Middle", "End"][min(sp_idx, 2)]
        bar_num = sp
        # Compute indicators up to this point in the chunk
        window = candles[:base_len + sp + 1]
        if len(window) < 50:
            continue
        ind = _compute_indicators(window)
        # Filter to key flags only (keep prompt concise)
        key_flags = {k: v for k, v in ind.items() if isinstance(v, bool) or k in (
            "rsi_14", "atr_14", "atr_vs_avg", "macd_histogram", "bb_position",
            "current_price", "session", "usd_bias",
        )}
        flag_text = ", ".join(f"{k}={v}" for k, v in key_flags.items())
        indicator_sections.append(f"  [Bar {bar_num}] {label}: {flag_text}")

    indicator_text = "\n".join(indicator_sections) if indicator_sections else \
        "\n".join(f"  {k}: {v}" for k, v in indicators_at_start.items())

    news_text = "  None nearby"
    if news_events:
        news_text = "\n".join(
            "  {title} ({ccy}, {imp}) {rel}".format(
                title=e["title"], ccy=e["currency"], imp=e["impact"],
                rel=f"{e['minutes_until']}min away" if not e.get("already_released") else "already released",
            )
            for e in news_events[:5]
        )

    prompt = (
        f"TRADING RULES:\n{rules_text}\n\n"
        f"CONTEXT:\nSymbol: {symbol}, Timeframe: {timeframe}\n"
        f"Current position: {position_state}\n\n"
        f"Indicator snapshots across window:\n{indicator_text}\n\n"
        f"Economic events near this window:\n{news_text}\n\n"
        f"CANDLE WINDOW ({len(chunk_candles)} bars):\n{chunk_text}\n\n"
        f"NEWS HANDLING: Only skip trades if a HIGH-impact event is within 15 minutes of "
        f"that bar. Events 30+ minutes away should NOT prevent signals.\n\n"
        f"Scan ALL bars in this window. Use the indicator snapshots to determine "
        f"crossovers and flag states at different points. For each bar where your "
        f"rules trigger a BUY or SELL signal, include it. Return empty array if none.\n\n"
        f"Respond with a JSON array (empty if no signals):\n"
        f'[\n'
        f'  {{"bar_index": <0-based index in the window>, "signal": "BUY"|"SELL", '
        f'"confidence": 0.0-1.0, "stop_loss": <price or null>, '
        f'"take_profit": <price or null>, "reasoning": "<brief>"}}\n'
        f']'
    )

    try:
        parsed = ai_service.generate_json(
            system=(
                "You are a forex trading rule interpreter. Scan the candle window and "
                "identify ALL bars where ANY entry rule condition is met. "
                "A single valid condition trigger is enough for a signal — you do NOT need "
                "multiple conditions. The indicators include pre-computed boolean flags "
                "(rsi_crossed_above_30, macd_crossed_bullish, etc.) — trust them directly. "
                "Return a JSON array. Empty array [] if no signals. No text outside JSON."
            ),
            prompt=prompt,
            cfg=cfg,
            max_tokens=2048,
            temperature=0.2,
            fallback=None,
        )
        if parsed is None:
            return []
        if isinstance(parsed, dict):
            parsed = [parsed]
        # Validate and normalize
        signals = []
        for item in parsed:
            if not isinstance(item, dict):
                continue
            sig = item.get("signal", "").upper()
            if sig not in ("BUY", "SELL"):
                continue
            bar_idx = item.get("bar_index", 0)
            if not isinstance(bar_idx, int) or bar_idx < 0 or bar_idx >= len(chunk_candles):
                continue
            signals.append({
                "bar_index": bar_idx,
                "signal": sig,
                "confidence": item.get("confidence", 0.5),
                "stop_loss": item.get("stop_loss"),
                "take_profit": item.get("take_profit"),
                "reasoning": item.get("reasoning", ""),
            })
        return signals
    except Exception as exc:
        logger.error("Batch evaluation error: %s", exc)
        return []


def _is_interesting_bar(candles: list[dict], i: int) -> bool:
    """Pre-filter: check if a bar is worth sending to Claude.

    Returns True if technical indicators suggest something notable is happening.
    Fast local computation — no API calls.
    """
    if i < 20:
        return False

    closes = [c["close"] for c in candles[:i + 1]]
    current = candles[i]
    prev = candles[i - 1]

    # RSI approaching extremes
    rsi = indicator_service.rsi(closes, 14)
    if rsi and rsi[-1] is not None:
        if rsi[-1] < 35 or rsi[-1] > 65:
            return True
        # RSI crossing 30 or 70
        if rsi[-2] is not None:
            if (rsi[-2] < 30 and rsi[-1] >= 30) or (rsi[-2] > 70 and rsi[-1] <= 70):
                return True
            if (rsi[-2] >= 30 and rsi[-1] < 30) or (rsi[-2] <= 70 and rsi[-1] > 70):
                return True

    # Price touching Bollinger Band
    bb = indicator_service.bollinger_bands(closes, 20, 2.0)
    if bb and bb["upper"][-1] is not None:
        if current["low"] <= bb["lower"][-1] or current["high"] >= bb["upper"][-1]:
            return True

    # SMA crossover (20/50)
    sma20 = indicator_service.sma(closes, 20)
    sma50 = indicator_service.sma(closes, 50)
    if (sma20 and sma50 and len(sma20) >= 2 and len(sma50) >= 2
            and sma20[-1] is not None and sma50[-1] is not None
            and sma20[-2] is not None and sma50[-2] is not None):
        if (sma20[-2] <= sma50[-2] and sma20[-1] > sma50[-1]) or \
           (sma20[-2] >= sma50[-2] and sma20[-1] < sma50[-1]):
            return True

    # New 20-bar high or low
    if i >= 20:
        recent_highs = [candles[j]["high"] for j in range(i - 19, i)]
        recent_lows = [candles[j]["low"] for j in range(i - 19, i)]
        if current["high"] > max(recent_highs) or current["low"] < min(recent_lows):
            return True

    # MACD histogram sign change
    macd = indicator_service.macd(closes)
    if macd and len(macd["histogram"]) >= 2:
        h1, h2 = macd["histogram"][-2], macd["histogram"][-1]
        if h1 is not None and h2 is not None:
            if (h1 < 0 and h2 >= 0) or (h1 >= 0 and h2 < 0):
                return True

    return False


# ---------------------------------------------------------------------------
# Rule backtesting
# ---------------------------------------------------------------------------

LOT_UNITS = {"micro": 1000, "mini": 10000, "standard": 100000}


def backtest_rules(
    rules_text: str,
    symbol: str,
    timeframe: str,
    candles: list[dict],
    initial_balance: float = 10000.0,
    context_window: int = 50,
    bt_id: int | None = None,
    mode: str = "batch",
    all_pair_candles: dict[str, list[dict]] | None = None,
    training_run_id: int | None = None,
) -> dict:
    """Backtest rules by walking through candle history with AI evaluation.

    Modes:
      - "bar_by_bar": One Claude call per bar (most accurate, slowest)
      - "batch": Evaluate chunks of bars per call (20-50x faster, recommended)
      - "filtered": Only call Claude on interesting bars (fastest, may miss signals)

    all_pair_candles: optional dict of {symbol: candles} for cross-pair context.
    Automatically seeds historical news events for the backtest period.

    Returns metrics dict compatible with training engine.
    """
    # Auto-seed historical news for the backtest period
    _ensure_historical_news(candles, symbol)

    if mode == "batch":
        return _backtest_batch(rules_text, symbol, timeframe, candles,
                               initial_balance, context_window, bt_id,
                               all_pair_candles=all_pair_candles, training_run_id=training_run_id)
    elif mode == "filtered":
        return _backtest_filtered(rules_text, symbol, timeframe, candles,
                                   initial_balance, context_window, bt_id, all_pair_candles=all_pair_candles)
    return _backtest_bar_by_bar(rules_text, symbol, timeframe, candles,
                                 initial_balance, context_window, bt_id, all_pair_candles=all_pair_candles)


def _ensure_historical_news(candles: list[dict], symbol: str):
    """Seed historical news events if none exist for the backtest period."""
    from services import news_service

    if not candles:
        return
    try:
        start_str = candles[0].get("time", "")
        end_str = candles[-1].get("time", "")
        start = datetime.fromisoformat(start_str)
        end = datetime.fromisoformat(end_str)
        if start.tzinfo is None:
            start = start.replace(tzinfo=timezone.utc)
        if end.tzinfo is None:
            end = end.replace(tzinfo=timezone.utc)

        existing = news_service.count_events_in_range(start, end)
        if existing < 5:
            count = news_service.seed_historical_calendar(start, end)
            logger.info("Seeded %d historical news events for backtest period %s to %s",
                        count, start.date(), end.date())
    except Exception as exc:
        logger.warning("Could not seed historical news: %s", exc)


def _get_news_for_bar(symbol: str, bar_time_str: str) -> list[dict]:
    """Get news events near a specific bar's timestamp for backtest context.

    Uses a tight window: only events within 30min before or 45min after.
    This prevents Claude from being spooked by events hours away that
    won't affect the trade outcome on intraday timeframes.
    """
    from services import news_service
    try:
        ts = datetime.fromisoformat(bar_time_str)
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        # Tight window — only truly imminent events
        events = news_service.get_events_at_time(symbol, ts, hours_before=0, hours_after=1)
        # Filter to only show events within 45 minutes
        return [e for e in events if abs(e.get("minutes_until", 999)) <= 45]
    except Exception:
        return []


def _backtest_bar_by_bar(
    rules_text: str, symbol: str, timeframe: str, candles: list[dict],
    initial_balance: float, context_window: int, bt_id: int | None,
    all_pair_candles: dict[str, list[dict]] | None = None,
) -> dict:
    """Original bar-by-bar backtest. Most accurate but slowest."""
    cfg = load_config()
    pip_value = 0.01 if "JPY" in symbol.upper() else 0.0001
    lot_units = LOT_UNITS.get(cfg.execution.default_lot_type, 10000)
    spread_cost = 1.5 * pip_value

    balance = initial_balance
    peak_balance = initial_balance
    max_drawdown = 0.0
    position = None
    trades = []
    equity_curve = []

    start_bar = max(context_window, 50)
    total_bars = len(candles) - start_bar

    for i in range(start_bar, len(candles)):
        # Check cancellation
        if bt_id is not None and _active_backtests.get(bt_id):
            logger.info("Backtest %d cancelled", bt_id)
            break
        window = candles[:i + 1]
        current = candles[i]
        price = current["close"]

        # Check SL/TP on open position
        if position:
            hit_sl = False
            hit_tp = False
            if position["side"] == "buy":
                if current["low"] <= position["sl"]:
                    hit_sl = True
                elif current["high"] >= position["tp"]:
                    hit_tp = True
            else:
                if current["high"] >= position["sl"]:
                    hit_sl = True
                elif current["low"] <= position["tp"]:
                    hit_tp = True

            if hit_sl or hit_tp:
                exit_price = position["sl"] if hit_sl else position["tp"]
                exit_type = "sl" if hit_sl else "tp"
                if position["side"] == "buy":
                    pnl = (exit_price - position["entry_price"]) * position["volume"] * lot_units
                else:
                    pnl = (position["entry_price"] - exit_price) * position["volume"] * lot_units
                pnl -= spread_cost * position["volume"] * lot_units
                balance += pnl
                trades.append({
                    "side": position["side"],
                    "entry_price": position["entry_price"],
                    "exit_price": exit_price,
                    "stop_loss": position["sl"],
                    "take_profit": position["tp"],
                    "pnl": round(pnl, 2),
                    "exit_type": exit_type,
                    "entry_ts": position["entry_ts"],
                    "exit_ts": current["time"],
                    "volume": position["volume"],
                })
                position = None

        # Track equity
        unrealized = 0
        if position:
            if position["side"] == "buy":
                unrealized = (price - position["entry_price"]) * position["volume"] * lot_units
            else:
                unrealized = (position["entry_price"] - price) * position["volume"] * lot_units
        equity = balance + unrealized
        if equity > peak_balance:
            peak_balance = equity
        dd = (peak_balance - equity) / peak_balance if peak_balance > 0 else 0
        if dd > max_drawdown:
            max_drawdown = dd
        equity_curve.append({"time": current["time"], "equity": round(equity, 2)})

        # Publish progress
        if bt_id is not None:
            bars_done = i - start_bar + 1
            _backtest_progress[bt_id] = {
                "bt_id": bt_id,
                "status": "running",
                "bar": bars_done,
                "total_bars": total_bars,
                "pct": round(bars_done / total_bars * 100, 1) if total_bars else 0,
                "balance": round(balance, 2),
                "equity": round(equity, 2),
                "trades": len(trades),
                "wins": sum(1 for t in trades if t["pnl"] > 0),
                "max_drawdown": round(max_drawdown, 4),
                "equity_curve": equity_curve[-50:],  # last 50 points for live chart
                "recent_trades": trades[-10:],
            }

        # Only evaluate for new positions when none open
        if position is not None:
            continue

        # Evaluate rules with Claude (including news + cross-pair context)
        indicators = _compute_indicators(window)
        if all_pair_candles:
            cross_ctx = compute_cross_pair_context(symbol, {
                p: cs[:i + 1] for p, cs in all_pair_candles.items() if len(cs) > i
            })
            indicators.update(cross_ctx)
        news_events = _get_news_for_bar(symbol, current["time"])
        result = evaluate_rules(rules_text, symbol, timeframe, window, indicators,
                                news_events=news_events)

        signal = result.get("signal", "HOLD")
        if signal not in ("BUY", "SELL"):
            continue

        confidence = result.get("confidence", 0)
        if confidence < cfg.signals.min_confidence:
            continue

        pos = _open_position(signal, price, window, pip_value, lot_units, balance, cfg,
                             result.get("stop_loss"), result.get("take_profit"))
        if pos:
            position = pos

    # Close remaining position at end
    if position:
        price = candles[-1]["close"]
        if position["side"] == "buy":
            pnl = (price - position["entry_price"]) * position["volume"] * lot_units
        else:
            pnl = (position["entry_price"] - price) * position["volume"] * lot_units
        pnl -= spread_cost * position["volume"] * lot_units
        balance += pnl
        trades.append({
            "side": position["side"],
            "entry_price": position["entry_price"],
            "exit_price": price,
            "stop_loss": position["sl"],
            "take_profit": position["tp"],
            "pnl": round(pnl, 2),
            "exit_type": "end",
            "entry_ts": position["entry_ts"],
            "exit_ts": candles[-1]["time"],
            "volume": position["volume"],
        })

    # Compute metrics
    total = len(trades)
    wins = sum(1 for t in trades if t["pnl"] > 0)
    losses = total - wins
    gross_profit = sum(t["pnl"] for t in trades if t["pnl"] > 0)
    gross_loss = abs(sum(t["pnl"] for t in trades if t["pnl"] <= 0))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else (999.0 if gross_profit > 0 else 0)

    # Sharpe ratio
    if len(equity_curve) >= 2:
        returns = []
        for j in range(1, len(equity_curve)):
            prev_eq = equity_curve[j - 1]["equity"]
            if prev_eq > 0:
                returns.append((equity_curve[j]["equity"] - prev_eq) / prev_eq)
        if returns:
            avg_r = sum(returns) / len(returns)
            var_r = sum((r - avg_r) ** 2 for r in returns) / len(returns)
            std_r = var_r ** 0.5
            sharpe = (avg_r / std_r * math.sqrt(252)) if std_r > 0 else 0
        else:
            sharpe = 0
    else:
        sharpe = 0

    result = {
        "final_balance": round(balance, 2),
        "total_trades": total,
        "winning_trades": wins,
        "losing_trades": losses,
        "win_rate": round(wins / total, 4) if total > 0 else 0,
        "max_drawdown": round(max_drawdown, 4),
        "profit_factor": round(min(profit_factor, 999.0), 2),
        "sharpe_ratio": round(sharpe, 2),
        "equity_curve": equity_curve,
        "trades": trades,
    }

    # Mark backtest complete in progress store
    if bt_id is not None:
        _backtest_progress[bt_id] = {
            "bt_id": bt_id,
            "status": "completed",
            "bar": total_bars,
            "total_bars": total_bars,
            "pct": 100,
            "balance": result["final_balance"],
            "equity": result["final_balance"],
            "trades": total,
            "wins": wins,
            "max_drawdown": result["max_drawdown"],
            "equity_curve": equity_curve,
            "recent_trades": trades[-10:],
            "results": {k: result[k] for k in
                        ("final_balance", "total_trades", "win_rate",
                         "max_drawdown", "profit_factor", "sharpe_ratio")},
        }
        _active_backtests.pop(bt_id, None)

    return result


def _manage_position_for_bar(
    position, current, balance, pip_value, lot_units, spread_cost, trades,
):
    """Check SL/TP on open position, return (position, balance) after processing."""
    if position is None:
        return position, balance

    hit_sl = hit_tp = False
    if position["side"] == "buy":
        if current["low"] <= position["sl"]:
            hit_sl = True
        elif current["high"] >= position["tp"]:
            hit_tp = True
    else:
        if current["high"] >= position["sl"]:
            hit_sl = True
        elif current["low"] <= position["tp"]:
            hit_tp = True

    if hit_sl or hit_tp:
        exit_price = position["sl"] if hit_sl else position["tp"]
        exit_type = "sl" if hit_sl else "tp"
        if position["side"] == "buy":
            pnl = (exit_price - position["entry_price"]) * position["volume"] * lot_units
        else:
            pnl = (position["entry_price"] - exit_price) * position["volume"] * lot_units
        pnl -= spread_cost * position["volume"] * lot_units
        balance += pnl
        trades.append({
            "side": position["side"], "entry_price": position["entry_price"],
            "exit_price": exit_price, "stop_loss": position["sl"],
            "take_profit": position["tp"], "pnl": round(pnl, 2),
            "exit_type": exit_type, "entry_ts": position["entry_ts"],
            "exit_ts": current["time"], "volume": position["volume"],
        })
        position = None

    return position, balance


def _open_position(signal_type, price, candles_up_to, pip_value, lot_units, balance, cfg, sl_override=None, tp_override=None):
    """Compute SL/TP and position size, return position dict."""
    atr_vals = indicator_service.atr(candles_up_to, 14)
    atr_val = atr_vals[-1] if atr_vals and atr_vals[-1] is not None else price * 0.005

    sl = sl_override
    tp = tp_override
    if sl is None:
        sl_dist = atr_val * cfg.execution.sl_atr_multiplier
        sl = price - sl_dist if signal_type == "BUY" else price + sl_dist
    if tp is None:
        tp_dist = atr_val * cfg.execution.tp_atr_multiplier
        tp = price + tp_dist if signal_type == "BUY" else price - tp_dist

    sl_distance = abs(price - sl)
    sl_pips = sl_distance / pip_value
    if sl_pips <= 0:
        return None
    # Enforce minimum SL distance to prevent oversized positions
    min_sl_pips = 10  # at least 10 pips SL
    sl_pips = max(sl_pips, min_sl_pips)
    risk_amount = balance * (cfg.execution.risk_per_trade_pct / 100.0)
    pip_value_per_lot = pip_value * lot_units
    volume = risk_amount / (sl_pips * pip_value_per_lot)
    # Cap volume to prevent any single trade from risking too much
    max_volume = balance / (price * lot_units) * 5  # max 5x leverage
    volume = min(volume, max_volume)
    volume = max(round(volume, 2), 0.01)

    return {
        "side": signal_type.lower(), "entry_price": price,
        "entry_ts": candles_up_to[-1]["time"], "volume": volume,
        "sl": sl, "tp": tp,
    }


def _finalize_backtest(position, candles, balance, pip_value, lot_units, spread_cost,
                        trades, equity_curve, initial_balance, peak_balance, max_drawdown, bt_id):
    """Close remaining position and compute metrics."""
    if position:
        price = candles[-1]["close"]
        if position["side"] == "buy":
            pnl = (price - position["entry_price"]) * position["volume"] * lot_units
        else:
            pnl = (position["entry_price"] - price) * position["volume"] * lot_units
        pnl -= spread_cost * position["volume"] * lot_units
        balance += pnl
        trades.append({
            "side": position["side"], "entry_price": position["entry_price"],
            "exit_price": price, "stop_loss": position["sl"], "take_profit": position["tp"],
            "pnl": round(pnl, 2), "exit_type": "end",
            "entry_ts": position["entry_ts"], "exit_ts": candles[-1]["time"],
            "volume": position["volume"],
        })

    total = len(trades)
    wins = sum(1 for t in trades if t["pnl"] > 0)
    gross_profit = sum(t["pnl"] for t in trades if t["pnl"] > 0)
    gross_loss = abs(sum(t["pnl"] for t in trades if t["pnl"] <= 0))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else (999.0 if gross_profit > 0 else 0)

    if len(equity_curve) >= 2:
        returns = []
        for j in range(1, len(equity_curve)):
            prev_eq = equity_curve[j - 1]["equity"]
            if prev_eq > 0:
                returns.append((equity_curve[j]["equity"] - prev_eq) / prev_eq)
        if returns:
            avg_r = sum(returns) / len(returns)
            var_r = sum((r - avg_r) ** 2 for r in returns) / len(returns)
            std_r = var_r ** 0.5
            sharpe = (avg_r / std_r * math.sqrt(252)) if std_r > 0 else 0
        else:
            sharpe = 0
    else:
        sharpe = 0

    result = {
        "final_balance": round(balance, 2),
        "total_trades": total, "winning_trades": wins, "losing_trades": total - wins,
        "win_rate": round(wins / total, 4) if total > 0 else 0,
        "max_drawdown": round(max_drawdown, 4),
        "profit_factor": round(min(profit_factor, 999.0), 2),
        "sharpe_ratio": round(sharpe, 2),
        "equity_curve": equity_curve, "trades": trades,
    }

    if bt_id is not None:
        _backtest_progress[bt_id] = {
            "bt_id": bt_id, "status": "completed", "bar": len(equity_curve),
            "total_bars": len(equity_curve), "pct": 100,
            "balance": result["final_balance"], "equity": result["final_balance"],
            "trades": total, "wins": wins, "max_drawdown": result["max_drawdown"],
            "equity_curve": equity_curve, "recent_trades": trades[-10:],
            "results": {k: result[k] for k in
                        ("final_balance", "total_trades", "win_rate",
                         "max_drawdown", "profit_factor", "sharpe_ratio")},
        }
        _active_backtests.pop(bt_id, None)

    return result


def _backtest_batch(
    rules_text: str, symbol: str, timeframe: str, candles: list[dict],
    initial_balance: float, context_window: int, bt_id: int | None,
    chunk_size: int = 40, all_pair_candles: dict[str, list[dict]] | None = None,
    training_run_id: int | None = None,
) -> dict:
    """Batch evaluation backtest. ~20x faster than bar-by-bar.

    Sends chunks of bars to Claude at once, Claude identifies all signals in each chunk.
    """
    cfg = load_config()
    pip_value = 0.01 if "JPY" in symbol.upper() else 0.0001
    lot_units = LOT_UNITS.get(cfg.execution.default_lot_type, 10000)
    spread_cost = 1.5 * pip_value

    balance = initial_balance
    peak_balance = initial_balance
    max_drawdown = 0.0
    position = None
    trades = []
    equity_curve = []

    start_bar = max(context_window, 50)
    total_bars = len(candles) - start_bar
    api_calls = 0

    i = start_bar
    while i < len(candles):
        if bt_id is not None and _active_backtests.get(bt_id):
            break

        chunk_end = min(i + chunk_size, len(candles))
        chunk = candles[i:chunk_end]

        # Process SL/TP for each bar in chunk (fast, no API)
        for bar_offset, current in enumerate(chunk):
            bar_idx = i + bar_offset
            position, balance = _manage_position_for_bar(
                position, current, balance, pip_value, lot_units, spread_cost, trades,
            )
            # Track equity
            unrealized = 0
            if position:
                price = current["close"]
                if position["side"] == "buy":
                    unrealized = (price - position["entry_price"]) * position["volume"] * lot_units
                else:
                    unrealized = (position["entry_price"] - price) * position["volume"] * lot_units
            equity = balance + unrealized
            if equity > peak_balance:
                peak_balance = equity
            dd = (peak_balance - equity) / peak_balance if peak_balance > 0 else 0
            if dd > max_drawdown:
                max_drawdown = dd
            equity_curve.append({"time": current["time"], "equity": round(equity, 2)})

        # Only call Claude if no position open (looking for entries)
        if position is None:
            indicators = _compute_indicators(candles[:i + 1])
            # Add cross-pair context
            if all_pair_candles:
                cross_ctx = compute_cross_pair_context(symbol, {
                    p: cs[:i + 1] for p, cs in all_pair_candles.items()
                    if len(cs) > i
                })
                indicators.update(cross_ctx)

            position_state = "no position"
            chunk_news = _get_news_for_bar(symbol, chunk[0]["time"]) if chunk else []

            signals = evaluate_rules_batch(
                rules_text, symbol, timeframe, candles[:i + 1],
                chunk, indicators, position_state, chunk_news,
            )
            api_calls += 1

            # Process signals from the batch
            for sig in sorted(signals, key=lambda s: s["bar_index"]):
                if position is not None:
                    break  # already opened from an earlier signal in this batch
                bar_idx = i + sig["bar_index"]
                if bar_idx >= len(candles):
                    continue
                confidence = sig.get("confidence", 0)
                if confidence < cfg.signals.min_confidence:
                    continue
                price = candles[bar_idx]["close"]
                pos = _open_position(
                    sig["signal"], price, candles[:bar_idx + 1],
                    pip_value, lot_units, balance, cfg,
                    sig.get("stop_loss"), sig.get("take_profit"),
                )
                if pos:
                    position = pos

        # Publish progress
        if bt_id is not None:
            bars_done = min(chunk_end - start_bar, total_bars)
            _backtest_progress[bt_id] = {
                "bt_id": bt_id, "status": "running",
                "bar": bars_done, "total_bars": total_bars,
                "pct": round(bars_done / total_bars * 100, 1) if total_bars else 0,
                "balance": round(balance, 2), "equity": round(equity_curve[-1]["equity"], 2) if equity_curve else balance,
                "trades": len(trades), "wins": sum(1 for t in trades if t["pnl"] > 0),
                "max_drawdown": round(max_drawdown, 4),
                "equity_curve": equity_curve[-50:],
                "recent_trades": trades[-10:],
                "api_calls": api_calls,
            }

        # Push live equity curve to training progress (updates chart per batch)
        if training_run_id is not None and training_run_id in _training_progress:
            sampled = equity_curve[::max(1, len(equity_curve) // 200)] if len(equity_curve) > 200 else equity_curve
            curves = _training_progress[training_run_id].get("per_symbol_curves") or {}
            curves[symbol] = sampled
            _training_progress[training_run_id]["per_symbol_curves"] = curves

        i = chunk_end

    logger.info("Batch backtest: %d API calls (vs %d bar-by-bar)", api_calls, total_bars)
    return _finalize_backtest(position, candles, balance, pip_value, lot_units, spread_cost,
                               trades, equity_curve, initial_balance, peak_balance, max_drawdown, bt_id)


def _backtest_filtered(
    rules_text: str, symbol: str, timeframe: str, candles: list[dict],
    initial_balance: float, context_window: int, bt_id: int | None,
    all_pair_candles: dict[str, list[dict]] | None = None,
) -> dict:
    """Pre-filtered backtest. Only calls Claude on interesting bars. ~40-60x faster."""
    cfg = load_config()
    pip_value = 0.01 if "JPY" in symbol.upper() else 0.0001
    lot_units = LOT_UNITS.get(cfg.execution.default_lot_type, 10000)
    spread_cost = 1.5 * pip_value

    balance = initial_balance
    peak_balance = initial_balance
    max_drawdown = 0.0
    position = None
    trades = []
    equity_curve = []

    start_bar = max(context_window, 50)
    total_bars = len(candles) - start_bar
    api_calls = 0

    for i in range(start_bar, len(candles)):
        if bt_id is not None and _active_backtests.get(bt_id):
            break

        current = candles[i]
        price = current["close"]

        # SL/TP check (always, no API)
        position, balance = _manage_position_for_bar(
            position, current, balance, pip_value, lot_units, spread_cost, trades,
        )

        # Track equity
        unrealized = 0
        if position:
            if position["side"] == "buy":
                unrealized = (price - position["entry_price"]) * position["volume"] * lot_units
            else:
                unrealized = (position["entry_price"] - price) * position["volume"] * lot_units
        equity = balance + unrealized
        if equity > peak_balance:
            peak_balance = equity
        dd = (peak_balance - equity) / peak_balance if peak_balance > 0 else 0
        if dd > max_drawdown:
            max_drawdown = dd
        equity_curve.append({"time": current["time"], "equity": round(equity, 2)})

        # Only call Claude if no position AND bar is interesting
        if position is None and _is_interesting_bar(candles, i):
            window = candles[:i + 1]
            indicators = _compute_indicators(window)
            if all_pair_candles:
                cross_ctx = compute_cross_pair_context(symbol, {
                    p: cs[:i + 1] for p, cs in all_pair_candles.items() if len(cs) > i
                })
                indicators.update(cross_ctx)
            news_events = _get_news_for_bar(symbol, current["time"])
            result = evaluate_rules(rules_text, symbol, timeframe, window, indicators,
                                    news_events=news_events)
            api_calls += 1

            signal = result.get("signal", "HOLD")
            confidence = result.get("confidence", 0)
            if signal in ("BUY", "SELL") and confidence >= cfg.signals.min_confidence:
                pos = _open_position(
                    signal, price, window, pip_value, lot_units, balance, cfg,
                    result.get("stop_loss"), result.get("take_profit"),
                )
                if pos:
                    position = pos

        # Publish progress
        if bt_id is not None and i % 20 == 0:
            bars_done = i - start_bar + 1
            _backtest_progress[bt_id] = {
                "bt_id": bt_id, "status": "running",
                "bar": bars_done, "total_bars": total_bars,
                "pct": round(bars_done / total_bars * 100, 1) if total_bars else 0,
                "balance": round(balance, 2), "equity": round(equity, 2),
                "trades": len(trades), "wins": sum(1 for t in trades if t["pnl"] > 0),
                "max_drawdown": round(max_drawdown, 4),
                "equity_curve": equity_curve[-50:],
                "recent_trades": trades[-10:],
                "api_calls": api_calls,
            }

    logger.info("Filtered backtest: %d API calls (vs %d bar-by-bar)", api_calls, total_bars)
    return _finalize_backtest(position, candles, balance, pip_value, lot_units, spread_cost,
                               trades, equity_curve, initial_balance, peak_balance, max_drawdown, bt_id)


# ---------------------------------------------------------------------------
# Rule evolution
# ---------------------------------------------------------------------------

def _symbol_analytics(sym: str, metrics: dict) -> str:
    """Build analytics text for one symbol's backtest results."""
    trades_list = metrics.get("trades", [])
    total = metrics.get("total_trades", 0)
    if total == 0:
        return f"  {sym}: 0 trades\n"

    sl_hits = sum(1 for t in trades_list if t.get("exit_type") == "sl")
    tp_hits = sum(1 for t in trades_list if t.get("exit_type") == "tp")
    wins = [t["pnl"] for t in trades_list if t.get("pnl", 0) > 0]
    losses = [t["pnl"] for t in trades_list if t.get("pnl", 0) <= 0]
    avg_win = sum(wins) / len(wins) if wins else 0
    avg_loss = sum(losses) / len(losses) if losses else 0

    return (
        f"  {sym}: {total} trades, WR={metrics.get('win_rate', 0):.1%}, "
        f"Sharpe={metrics.get('sharpe_ratio', 0):.2f}, PF={metrics.get('profit_factor', 0):.2f}, "
        f"DD={metrics.get('max_drawdown', 0):.1%}, "
        f"SL hits={sl_hits}/{total} ({sl_hits/total*100:.0f}%), "
        f"TP hits={tp_hits}/{total} ({tp_hits/total*100:.0f}%), "
        f"Avg win=${avg_win:+.2f}, Avg loss=${avg_loss:+.2f}\n"
    )


def evolve_rules(
    rules_text: str,
    per_symbol_metrics: dict[str, dict],
    trade_summary: str,
    iteration: int,
    prev_rules: str | None = None,
    consistency_text: str = "",
) -> dict:
    """Ask the configured AI provider to evolve rules based on cross-pair backtest performance.

    per_symbol_metrics: {"EURUSD": {metrics}, "GBPUSD": {metrics}, ...}
    consistency_text: pass-to-pass variance data for same rules
    Returns: {"evolved_rules": str, "changes_made": [str], "reasoning": str}
    """
    cfg = load_config()
    if not ai_service.is_enabled(cfg):
        return {"error": "AI provider not available"}

    prev_text = ""
    if prev_rules and prev_rules != rules_text:
        prev_text = f"\nPrevious rules (iteration {iteration - 1}):\n{prev_rules}\n"

    # Per-symbol breakdown
    symbol_section = "BACKTEST RESULTS BY SYMBOL:\n"
    for sym, m in per_symbol_metrics.items():
        symbol_section += _symbol_analytics(sym, m)

    # Aggregate
    agg = _aggregate_metrics(per_symbol_metrics, cfg.training.rank_by)
    total_trades = agg.get("total_trades", 0)

    # Cross-pair insight
    if len(per_symbol_metrics) > 1:
        sharpes = {s: m.get("sharpe_ratio", 0) or 0 for s, m in per_symbol_metrics.items()}
        best_sym = max(sharpes, key=sharpes.get)
        worst_sym = min(sharpes, key=sharpes.get)
        cross_pair = (
            f"\nCROSS-PAIR ANALYSIS:\n"
            f"  Best performing: {best_sym} (Sharpe {sharpes[best_sym]:.2f})\n"
            f"  Worst performing: {worst_sym} (Sharpe {sharpes[worst_sym]:.2f})\n"
            f"  Rules must work across ALL pairs — avoid changes that only help one pair.\n"
        )
    else:
        cross_pair = ""

    prompt = (
        f"You are evolving a forex trading strategy through iteration {iteration}.\n\n"
        f"CURRENT RULES:\n{rules_text}\n"
        f"{prev_text}\n"
        f"{symbol_section}\n"
        f"AGGREGATE ACROSS ALL PAIRS:\n"
        f"  Total trades: {total_trades}\n"
        f"  Average win rate: {agg.get('win_rate', 0):.1%}\n"
        f"  Average Sharpe: {agg.get('sharpe_ratio', 0):.2f}\n"
        f"  Average profit factor: {agg.get('profit_factor', 0):.2f}\n"
        f"  Worst drawdown: {agg.get('max_drawdown', 0):.1%}\n"
        f"  Initial balance: ${cfg.training.initial_balance:,.2f}\n\n"
        f"{cross_pair}\n"
        f"{consistency_text}\n"
        f"TRADE SUMMARY:\n{trade_summary}\n\n"
        f"ANALYSIS FRAMEWORK — Before evolving rules, analyze these specific questions:\n"
        f"1. SL HIT RATE: If SL hits are >50% on any pair, stops may be too tight.\n"
        f"2. CROSS-PAIR CONSISTENCY: If one pair performs much worse, identify what's different "
        f"about that pair's price action and adjust rules to handle it.\n"
        f"3. WIN CLUSTERING: If wins cluster together, the strategy works in trends but "
        f"fails in ranges. Add a ranging-market filter.\n"
        f"4. TRADE FREQUENCY: Aim for 2-5 trades per week per pair. "
        f"If too few, loosen conditions. If too many, tighten confluence requirements.\n"
        f"5. COUNTER-TREND LOSSES: Strengthen trend filters without eliminating opportunities.\n"
        f"6. OVERFITTING: Do NOT make rules pair-specific. "
        f"Prefer structural improvements that benefit ALL pairs. "
        f"Rules should be universal across currency pairs and time periods.\n\n"
        f"EVOLVE the rules. Output the COMPLETE new rule set (not a diff). "
        f"Keep the structured section format (Entry Signals, Trend Filter, Confirmation, "
        f"Trade Management, Filters, Confidence Scoring).\n\n"
        f"Respond with JSON:\n"
        f'{{\n'
        f'  "evolved_rules": "<the complete new rule set>",\n'
        f'  "changes_made": ["list of specific changes"],\n'
        f'  "reasoning": "why these changes address the problems identified above"\n'
        f'}}'
    )

    try:
        system = (
            "You are an expert quantitative forex researcher evolving trading rules "
            "through iterative backtesting. Your goal is consistent, modest profitability — "
            "not home runs.\n\n"
            "CRITICAL PRINCIPLES:\n"
            "- NEVER overfit: prefer robust structural changes over parameter tweaks\n"
            "- Focus on ENTRY QUALITY: requiring confluence (multiple conditions agreeing) "
            "is more important than finding the perfect indicator threshold\n"
            "- STOP LOSS SIZING: if SL is hit >50% of the time, it's too tight for the "
            "timeframe's noise. Wider stops with the same R:R ratio is usually better\n"
            "- CONFIRMATION: waiting for a confirmation candle after a signal reduces "
            "false entries dramatically, even though it slightly worsens entry price\n"
            "- TREND ALIGNMENT: trading with the higher-timeframe trend is the single "
            "biggest edge in forex. Make trend filters robust but not too restrictive\n"
            "- DON'T OVER-RESTRICT: rules that never trigger are worse than rules that "
            "sometimes lose. Target 2-5 trades per week per currency pair\n"
            "- Each rule must be specific enough for another AI to interpret unambiguously\n"
            "- Always respond with valid JSON only"
        )
        parsed = ai_service.generate_json(
            system=system,
            prompt=prompt,
            cfg=cfg,
            review=True,
            fallback=None,
        )
        if parsed is None:
            return {"error": "Could not parse response"}
        # Unwrap if Opus returned a list instead of a dict
        if isinstance(parsed, list):
            parsed = parsed[0] if parsed and isinstance(parsed[0], dict) else {"error": "Unexpected list response"}
        parsed.setdefault("evolved_rules", rules_text)
        parsed.setdefault("changes_made", [])
        parsed.setdefault("reasoning", "")
        return parsed
    except Exception as exc:
        logger.error("Rule evolution error: %s", exc)
        return {"error": str(exc)}


def _build_trade_summary(trades: list[dict], max_trades: int = 15) -> str:
    if not trades:
        return "No trades executed."
    total = len(trades)
    wins = sum(1 for t in trades if t.get("pnl", 0) > 0)
    exit_types = {}
    for t in trades:
        et = t.get("exit_type", "unknown")
        exit_types[et] = exit_types.get(et, 0) + 1
    exit_text = ", ".join(f"{k}: {v}" for k, v in sorted(exit_types.items()))

    lines = [
        f"Total: {total} trades ({wins} wins, {total - wins} losses)",
        f"Exit types: {exit_text}",
        f"Sample trades:",
    ]
    for t in trades[-max_trades:]:
        lines.append(
            "  {side} @ {entry:.5f} -> {exit:.5f} | P&L=${pnl:+.2f} ({et})".format(
                side=t.get("side", "?"),
                entry=t.get("entry_price", 0),
                exit=t.get("exit_price", 0),
                pnl=t.get("pnl", 0),
                et=t.get("exit_type", "?"),
            )
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def _aggregate_metrics(per_symbol: dict[str, dict], rank_by: str) -> dict:
    """Aggregate per-symbol backtest metrics into a combined summary."""
    all_trades = []
    total_balance = 0
    for sym, m in per_symbol.items():
        all_trades.extend(m.get("trades", []))
        total_balance += m.get("final_balance", 0)

    total = len(all_trades)
    wins = sum(1 for t in all_trades if t.get("pnl", 0) > 0)
    sharpes = [m.get("sharpe_ratio", 0) or 0 for m in per_symbol.values()]
    pfs = [m.get("profit_factor", 0) or 0 for m in per_symbol.values()]
    dds = [m.get("max_drawdown", 0) or 0 for m in per_symbol.values()]

    return {
        "final_balance": total_balance / len(per_symbol) if per_symbol else 0,
        "total_trades": total,
        "win_rate": wins / total if total else 0,
        "sharpe_ratio": sum(sharpes) / len(sharpes) if sharpes else 0,
        "profit_factor": sum(pfs) / len(pfs) if pfs else 0,
        "max_drawdown": max(dds) if dds else 0,
        "trades": all_trades,
    }


def train_ruleset(
    ruleset_id: int,
    symbols_candles: dict[str, list[dict]],
    timeframe: str,
    training_run_id: int,
    max_iterations: int | None = None,
    initial_balance: float | None = None,
    target_dd_pct: float | None = None,
    target_profit_pct: float | None = None,
) -> list[dict]:
    """Run iterative cross-pair training for a ruleset.

    Each iteration backtests ALL symbols, aggregates results, and evolves
    rules based on the full cross-pair picture. Stops when both drawdown
    and profit targets are met, or max passes hit.
    """
    cfg = load_config()
    tc = cfg.training
    if max_iterations is not None:
        tc.max_iterations = max_iterations
    start_balance = initial_balance or tc.initial_balance
    dd_target = target_dd_pct or 5.0    # default 5%
    profit_target = target_profit_pct or 10.0  # default 10%
    source_ruleset_name = f"Ruleset {ruleset_id}"
    session = get_session()

    try:
        ruleset = session.query(RuleSet).filter(RuleSet.id == ruleset_id).first()
        if not ruleset:
            return []
        current_rules = ruleset.rules_text
        source_ruleset_name = ruleset.name
    finally:
        session.close()

    symbols = list(symbols_candles.keys())
    prev_agg = None
    prev_rules = None
    # Track results per rule version to measure consistency
    rule_history: list[dict] = []  # [{rules_hash, iteration, sharpe, win_rate, trades, ...}]
    iterations = []

    # Initialize progress immediately so polling doesn't 404
    _training_progress[training_run_id] = {
        "run_id": training_run_id,
        "status": "running",
        "iteration": 0,
        "max_iterations": tc.max_iterations,
        "initial_balance": start_balance,
        "target_dd_pct": dd_target,
        "target_profit_pct": profit_target,
        "symbols": symbols,
        "current_rules": current_rules,
        "iterations": [],
        "equity_curve": [],
    }

    for iteration in range(1, tc.max_iterations + 1):
        if _is_stopped(training_run_id):
            break

        logger.info("Ruleset %d training: [%s]/%s iteration %d",
                     ruleset_id, ", ".join(symbols), timeframe, iteration)

        # Clear equity curves from previous pass so the chart resets
        if training_run_id in _training_progress:
            _training_progress[training_run_id]["per_symbol_curves"] = {}

        # Backtest ALL symbols with current rules, passing cross-pair context
        per_symbol_metrics: dict[str, dict] = {}
        for sym, candles in symbols_candles.items():
            if _is_stopped(training_run_id):
                break
            metrics = backtest_rules(
                current_rules, sym, timeframe, candles,
                initial_balance=start_balance,
                mode="batch",
                all_pair_candles=symbols_candles,
                training_run_id=training_run_id,
            )
            per_symbol_metrics[sym] = metrics
            logger.info("  %s: Sharpe=%.2f, WR=%.1f%%, Trades=%d",
                         sym, metrics.get("sharpe_ratio", 0),
                         (metrics.get("win_rate", 0) or 0) * 100,
                         metrics.get("total_trades", 0))
            # equity_curve is updated live per-batch-chunk inside _backtest_batch

        if _is_stopped(training_run_id):
            break

        agg = _aggregate_metrics(per_symbol_metrics, tc.rank_by)

        # Compute improvement on aggregate
        improvement = None
        if prev_agg:
            curr_val = agg.get(tc.rank_by, 0) or 0
            prev_val = prev_agg.get(tc.rank_by, 0) or 0
            if prev_val != 0:
                improvement = (curr_val - prev_val) / abs(prev_val)
            else:
                improvement = 1.0 if curr_val > 0 else 0.0

        # Build combined trade summary
        all_trades = agg.get("trades", [])
        trade_summary = _build_trade_summary(all_trades)

        # Track results for consistency measurement
        rules_hash = hash(current_rules)
        rule_history.append({
            "rules_hash": rules_hash, "iteration": iteration,
            "sharpe": agg.get("sharpe_ratio", 0), "win_rate": agg.get("win_rate", 0),
            "trades": agg.get("total_trades", 0), "max_drawdown": agg.get("max_drawdown", 0),
            "final_balance": agg.get("final_balance", start_balance),
        })

        # Build consistency report for same-rule passes
        same_rule_passes = [r for r in rule_history if r["rules_hash"] == rules_hash]
        consistency_text = ""
        if len(same_rule_passes) >= 2:
            sharpes = [r["sharpe"] for r in same_rule_passes]
            wrs = [r["win_rate"] for r in same_rule_passes]
            trades = [r["trades"] for r in same_rule_passes]
            avg_sharpe = sum(sharpes) / len(sharpes)
            sharpe_spread = max(sharpes) - min(sharpes)
            avg_wr = sum(wrs) / len(wrs)
            wr_spread = max(wrs) - min(wrs)
            high_variance = sharpe_spread > 0.5 or wr_spread > 0.15
            consistency_text = (
                f"\nCONSISTENCY CHECK ({len(same_rule_passes)} passes with same rules):\n"
            )
            for r in same_rule_passes:
                consistency_text += (
                    f"  Pass {r['iteration']}: Sharpe={r['sharpe']:.2f}, "
                    f"WR={r['win_rate']:.1%}, Trades={r['trades']}, DD={r['max_drawdown']:.1%}\n"
                )
            if high_variance:
                consistency_text += (
                    f"  *** HIGH VARIANCE (Sharpe spread={sharpe_spread:.2f}, WR spread={wr_spread:.1%}) ***\n"
                    f"  Rules are too ambiguous — Claude interprets them differently each time.\n"
                    f"  PRIORITY: Make entry conditions more specific and unambiguous.\n"
                    f"  Use exact indicator thresholds (e.g. 'RSI below 25' not 'RSI oversold').\n"
                )
            else:
                consistency_text += f"  Variance is LOW — rules are clear and consistent.\n"

        # Evolve rules with full cross-pair context + consistency data
        evolution = evolve_rules(
            current_rules, per_symbol_metrics, trade_summary, iteration, prev_rules,
            consistency_text,
        )

        # Store iteration (use aggregate metrics)
        session = get_session()
        try:
            iter_record = TrainingIteration(
                training_run_id=training_run_id,
                iteration_number=iteration,
                symbol=", ".join(symbols),
                timeframe=timeframe,
                strategy_type=f"ruleset_{ruleset_id}",
                parameters={"rules_text": current_rules},
                final_balance=agg.get("final_balance"),
                total_trades=agg.get("total_trades"),
                win_rate=agg.get("win_rate"),
                max_drawdown=agg.get("max_drawdown"),
                profit_factor=agg.get("profit_factor"),
                sharpe_ratio=agg.get("sharpe_ratio"),
                equity_curve=None,
                trades=None,  # too large for multi-symbol
                claude_analysis=evolution if "error" not in evolution else {"error": evolution.get("error")},
                suggested_params={"evolved_rules": evolution.get("evolved_rules", current_rules)},
                improvement_pct=improvement,
                created_at=datetime.now(timezone.utc),
            )
            session.add(iter_record)

            run = session.query(TrainingRun).filter(TrainingRun.id == training_run_id).first()
            if run:
                run.iterations_completed = iteration
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

        iter_data = {
            "iteration_number": iteration,
            "rules": current_rules,
            "symbols": symbols,
            "timeframe": timeframe,
            **{k: agg.get(k) for k in ("final_balance", "total_trades", "win_rate",
                                         "max_drawdown", "profit_factor", "sharpe_ratio")},
            "improvement_pct": improvement,
            "changes_made": evolution.get("changes_made", []),
            "reasoning": evolution.get("reasoning", ""),
            "evolved_rules": evolution.get("evolved_rules", ""),
        }
        iterations.append(iter_data)

        # Collect per-symbol equity curves (separate lines, not merged)
        per_symbol_curves = {}
        for sym, m in per_symbol_metrics.items():
            curve = m.get("equity_curve", [])
            if curve:
                step = max(1, len(curve) // 150)
                per_symbol_curves[sym] = curve[::step]

        # Publish training progress
        dd_val = agg.get("max_drawdown", 0)
        final_bal = agg.get("final_balance", start_balance)
        profit_pct = (final_bal - start_balance) / start_balance if start_balance > 0 else 0
        _training_progress[training_run_id] = {
            "run_id": training_run_id,
            "status": "running",
            "iteration": iteration,
            "max_iterations": tc.max_iterations,
            "initial_balance": start_balance,
            "target_dd_pct": dd_target,
            "target_profit_pct": profit_target,
            "symbols": symbols,
            "current_rules": current_rules,
            "per_symbol_curves": per_symbol_curves,
            "iterations": [
                {k: it.get(k) for k in ("iteration_number", "final_balance", "total_trades",
                                          "win_rate", "sharpe_ratio", "profit_factor",
                                          "max_drawdown", "improvement_pct", "changes_made", "reasoning")}
                for it in iterations
            ],
        }

        logger.info(
            "  AGGREGATE -> Bal=$%s, DD=%.1f%% (target <=%.1f%%), Profit=%.1f%% (target >=%.1f%%), Sharpe=%.2f, Trades=%d",
            f"{final_bal:,.0f}", dd_val * 100, dd_target,
            profit_pct * 100, profit_target,
            agg.get("sharpe_ratio", 0), agg.get("total_trades", 0),
        )

        # Check target convergence: both drawdown and profit targets met
        dd_ok = dd_val <= dd_target / 100
        profit_ok = profit_pct >= profit_target / 100
        if dd_ok and profit_ok:
            logger.info("Training targets met! DD=%.1f%% <= %.1f%%, Profit=%.1f%% >= %.1f%%",
                         dd_val * 100, dd_target, profit_pct * 100, profit_target)
            break

        # Prepare next iteration
        prev_agg = agg
        prev_rules = current_rules

        if "evolved_rules" in evolution and evolution["evolved_rules"] != current_rules:
            current_rules = evolution["evolved_rules"]
            changes = evolution.get("changes_made", [])
            logger.info("  Rules evolved (%d changes): %s", len(changes), "; ".join(changes[:3]))
        elif "error" in evolution:
            # API error — log but continue to next iteration with same rules
            logger.warning("Evolution failed: %s — retrying next pass with current rules", evolution["error"])
        else:
            logger.info("  No rule changes suggested — continuing with current rules")

    # Write the best pass as a validation-gated child ruleset instead of
    # mutating the source ruleset.
    candidate_id = None
    if iterations and not _is_stopped(training_run_id):
        dd_limit = dd_target / 100
        within_dd = [it for it in iterations if (it.get("max_drawdown") or 1) <= dd_limit]
        if within_dd:
            best = max(within_dd, key=lambda it: it.get("final_balance", 0))
        else:
            # None met drawdown target — pick lowest drawdown
            best = min(iterations, key=lambda it: it.get("max_drawdown", 1))
        session = get_session()
        try:
            ruleset = session.query(RuleSet).filter(RuleSet.id == ruleset_id).first()
            if ruleset:
                now = datetime.now(timezone.utc)
                base_params = dict(ruleset.parameters or {})
                base_params.pop("validation", None)
                base_params.pop("promotion", None)
                candidate_params = {
                    **base_params,
                    "candidate": True,
                    "source_ruleset_id": ruleset_id,
                    "source_ruleset_name": ruleset.name,
                    "training_run_id": training_run_id,
                    "selected_iteration": best.get("iteration_number"),
                    "training": {
                        "symbols": symbols,
                        "timeframe": timeframe,
                        "initial_balance": start_balance,
                        "target_dd_pct": dd_target,
                        "target_profit_pct": profit_target,
                        "max_iterations": tc.max_iterations,
                    },
                    "validation": {"status": "not_run", "passed": False},
                }
                training_metrics = {
                    k: best.get(k) for k in ("final_balance", "total_trades", "win_rate",
                                               "max_drawdown", "profit_factor", "sharpe_ratio")
                }
                candidate = RuleSet(
                    name=f"{ruleset.name} candidate run {training_run_id}"[:128],
                    description=(
                        f"Candidate evolved from ruleset {ruleset.id} by training run "
                        f"{training_run_id}. Requires validation before promotion."
                    )[:512],
                    status="candidate",
                    rules_text=best.get("rules", ""),
                    parameters=candidate_params,
                    symbols=symbols,
                    timeframes=[timeframe],
                    version=1,
                    parent_id=ruleset_id,
                    performance_metrics={
                        **training_metrics,
                        "selection_basis": "training_in_sample",
                        "validated": False,
                        "validation_status": "not_run",
                        "source_training_run_id": training_run_id,
                    },
                    created_at=now,
                    updated_at=now,
                )
                session.add(candidate)
                session.commit()
                candidate_id = candidate.id
                logger.info(
                    "Created candidate ruleset %d from source ruleset %d training run %d",
                    candidate_id, ruleset_id, training_run_id,
                )
            else:
                session.commit()
        except Exception:
            logger.exception(
                "Failed to create candidate ruleset from source ruleset %d training run %d",
                ruleset_id, training_run_id,
            )
            session.rollback()
        finally:
            session.close()

    # Mark training complete or stopped
    if training_run_id in _training_progress:
        if candidate_id:
            _training_progress[training_run_id]["candidate_ruleset_id"] = candidate_id
            _training_progress[training_run_id]["candidate_ruleset_name"] = (
                f"{source_ruleset_name} candidate run {training_run_id}"[:128]
            )
        if _is_stopped(training_run_id):
            _training_progress[training_run_id]["status"] = "stopped"
        else:
            _training_progress[training_run_id]["status"] = "completed"

    return iterations
