"""Autonomous signal generation loop.

Scans configured symbols/timeframes using registered strategies,
optionally validates signals with Claude, and dispatches confirmed
signals to the trade manager for execution.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta, timezone

from config import load_config
from db import get_session
from models.candle import Candle
from models.signal import Signal as SignalModel
from services.strategy_service import STRATEGIES, SignalType
from services import indicator_service, claude_service, trade_manager, news_service, sentiment_service, rule_engine

logger = logging.getLogger(__name__)


# Per (scan_key, timeframe, symbol) latest bar timestamp we've already scanned.
# Used so a 15m ruleset checked every 5min only burns one Claude call per bar
# (i.e. once every 15min) instead of three. Resets on process restart, which is
# fine — at worst the first scan after a restart re-scores a bar already seen
# by the previous instance.
_last_scanned_bar: dict[tuple[str, str, str], str] = {}


# Minutes per timeframe. All are divisors of 1440 (one day), which lets us
# floor wall-clock time to a bar boundary by dividing minutes-since-midnight.
_TIMEFRAME_MINUTES = {"1m": 1, "5m": 5, "15m": 15, "1h": 60, "4h": 240, "1d": 1440}


def _current_bar_key(timeframe: str, now: datetime) -> str:
    """Return a stable ISO string for the bar currently in progress at `now`.

    Earlier versions keyed the gate off candles[-1]["time"], but the candle
    collector rewrites the in-progress bar's ts on every tick (seen in the DB
    as rows at 07:29:39, 07:32:39, 07:35:40, …). That made every scan look
    like a new bar and defeated the gate. Computing the bar start from
    wall-clock + timeframe is independent of collector behavior and stable
    for the full bar period.
    """
    tf_min = _TIMEFRAME_MINUTES.get(timeframe, 15)
    midnight = now.replace(hour=0, minute=0, second=0, microsecond=0)
    elapsed_min = int((now - midnight).total_seconds() // 60)
    bar_offset = (elapsed_min // tf_min) * tf_min
    return (midnight + timedelta(minutes=bar_offset)).isoformat()


def _claim_new_bar(scan_key: str, timeframe: str, symbol: str, bar_ts: str) -> bool:
    """Return True the first time we see (scan_key, tf, symbol, bar_ts), False
    after that. Marks the bar as claimed before the caller spends any work on
    it — if the downstream Claude call fails, the bar isn't retried until the
    next bar closes (acceptable, prevents retry storms during API outages)."""
    key = (scan_key, timeframe, symbol)
    if _last_scanned_bar.get(key) == bar_ts:
        return False
    _last_scanned_bar[key] = bar_ts
    return True


def _query_candles(symbol: str, timeframe: str, limit: int = 200) -> list[dict]:
    """Fetch recent candles from the database as dicts."""
    session = get_session()
    try:
        rows = (
            session.query(Candle)
            .filter(Candle.symbol == symbol, Candle.timeframe == timeframe)
            .order_by(Candle.ts.desc())
            .limit(limit)
            .all()
        )
        if not rows:
            return []
        rows.reverse()
        return [
            {
                "time": row.ts.isoformat(),
                "open": row.open,
                "high": row.high,
                "low": row.low,
                "close": row.close,
                "volume": row.volume or 0,
            }
            for row in rows
        ]
    finally:
        session.close()


def _check_cooldown(symbol: str, cool_down_minutes: int) -> bool:
    """Return True if a recent signal exists for this symbol (cooldown active)."""
    session = get_session()
    try:
        cutoff = datetime.now(timezone.utc) - timedelta(minutes=cool_down_minutes)
        recent = (
            session.query(SignalModel)
            .filter(
                SignalModel.symbol == symbol,
                SignalModel.created_at >= cutoff,
                SignalModel.status.in_(["confirmed", "executed"]),
            )
            .first()
        )
        return recent is not None
    finally:
        session.close()


def _count_signals_today() -> int:
    """Count signals generated today."""
    session = get_session()
    try:
        today_start = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
        return (
            session.query(SignalModel)
            .filter(SignalModel.created_at >= today_start)
            .count()
        )
    finally:
        session.close()


def _check_candle_freshness(candles: list[dict], timeframe: str) -> bool:
    """Return True if candle data is fresh enough for the given timeframe."""
    if not candles:
        return False
    # Maximum staleness: 2x the timeframe interval
    tf_minutes = {
        "1m": 1, "5m": 5, "15m": 15, "30m": 30,
        "1h": 60, "4h": 240, "1d": 1440,
    }
    max_age = timedelta(minutes=tf_minutes.get(timeframe, 60) * 2)
    last_time = datetime.fromisoformat(candles[-1]["time"])
    if last_time.tzinfo is None:
        last_time = last_time.replace(tzinfo=timezone.utc)
    return (datetime.now(timezone.utc) - last_time) < max_age


def _compute_indicator_context(candles: list[dict]) -> dict:
    """Compute current indicator values for Claude context."""
    closes = [c["close"] for c in candles]
    result = {}

    rsi = indicator_service.rsi(closes, 14)
    if rsi and rsi[-1] is not None:
        result["rsi_14"] = round(rsi[-1], 2)

    atr = indicator_service.atr(candles, 14)
    if atr and atr[-1] is not None:
        result["atr_14"] = round(atr[-1], 6)

    sma20 = indicator_service.sma(closes, 20)
    if sma20 and sma20[-1] is not None:
        result["sma_20"] = round(sma20[-1], 5)

    sma50 = indicator_service.sma(closes, 50)
    if sma50 and sma50[-1] is not None:
        result["sma_50"] = round(sma50[-1], 5)

    bb = indicator_service.bollinger_bands(closes, 20, 2.0)
    if bb and bb["upper"][-1] is not None:
        result["bb_upper"] = round(bb["upper"][-1], 5)
        result["bb_lower"] = round(bb["lower"][-1], 5)
        result["bb_middle"] = round(bb["middle"][-1], 5)

    macd = indicator_service.macd(closes)
    if macd and macd["macd"][-1] is not None:
        result["macd"] = round(macd["macd"][-1], 6)
        result["macd_signal"] = round(macd["signal"][-1], 6)
        result["macd_histogram"] = round(macd["histogram"][-1], 6)

    result["current_price"] = closes[-1]
    return result


def _scan_once():
    """Run one full scan across all configured symbols/timeframes/strategies."""
    cfg = load_config()
    sig_cfg = cfg.signals

    if not sig_cfg.enabled:
        return

    symbols = sig_cfg.symbols if sig_cfg.symbols else cfg.symbols
    timeframes = sig_cfg.timeframes
    strategy_names = sig_cfg.strategies if sig_cfg.strategies else list(STRATEGIES.keys())

    # Trading session window check (EST/EDT)
    from zoneinfo import ZoneInfo
    est_now = datetime.now(ZoneInfo("America/New_York"))
    est_time = est_now.hour * 60 + est_now.minute  # minutes since midnight EST
    session_1 = (6 * 60 + 30, 16 * 60)           # 6:30 AM - 4:00 PM EST (London + full NY)
    session_2 = (21 * 60, 23 * 60 + 30)         # 9:00 PM - 11:30 PM EST (Tokyo open)
    if not (session_1[0] <= est_time <= session_1[1] or session_2[0] <= est_time <= session_2[1]):
        return

    # Daily limit check (0 = unlimited)
    if sig_cfg.max_signals_per_day > 0 and _count_signals_today() >= sig_cfg.max_signals_per_day:
        logger.info("Daily signal limit (%d) reached, skipping scan", sig_cfg.max_signals_per_day)
        return

    logger.info(
        "Strategy scan tick: %d strategies x %d symbols x %d timeframes",
        len(strategy_names), len(symbols), len(timeframes),
    )

    for symbol in symbols:
        for timeframe in timeframes:
            candles = _query_candles(symbol, timeframe)
            if not candles or len(candles) < 50:
                logger.debug("Not enough candles for %s/%s (%d)", symbol, timeframe, len(candles))
                continue

            if not _check_candle_freshness(candles, timeframe):
                logger.debug("Stale candle data for %s/%s, skipping", symbol, timeframe)
                continue

            # Bar-close gating: skip if we've already scanned this bar.
            # All strategies see the same candles for this (sym, tf) so one
            # gate covers all of them.
            latest_bar_ts = _current_bar_key(timeframe, datetime.now(timezone.utc))
            if not _claim_new_bar("strategy", timeframe, symbol, latest_bar_ts):
                continue

            for strategy_name in strategy_names:
                strategy = STRATEGIES.get(strategy_name)
                if strategy is None:
                    continue

                try:
                    params = strategy.default_params()
                    signal = strategy.evaluate(candles, params)
                except Exception as exc:
                    logger.error("Strategy %s error on %s/%s: %s", strategy_name, symbol, timeframe, exc)
                    continue

                if signal.type == SignalType.HOLD:
                    continue

                # Cooldown check
                if _check_cooldown(symbol, sig_cfg.cool_down_minutes):
                    logger.debug("Cooldown active for %s, skipping signal", symbol)
                    continue

                # News high-impact window check
                blocked, block_reason = news_service.is_high_impact_window(symbol)
                if blocked:
                    logger.info("Signal blocked for %s: %s", symbol, block_reason)
                    continue

                # Daily limit recheck
                if sig_cfg.max_signals_per_day > 0 and _count_signals_today() >= sig_cfg.max_signals_per_day:
                    return

                side = signal.type.value  # "buy" or "sell"
                now = datetime.now(timezone.utc)

                # Compute ATR for SL/TP if strategy didn't provide them
                sl = signal.stop_loss
                tp = signal.take_profit
                if sl is None or tp is None:
                    atr_vals = indicator_service.atr(candles, 14)
                    atr_val = atr_vals[-1] if atr_vals and atr_vals[-1] is not None else signal.price * 0.005
                    if sl is None:
                        sl_dist = atr_val * cfg.execution.sl_atr_multiplier
                        sl = signal.price - sl_dist if side == "buy" else signal.price + sl_dist
                    if tp is None:
                        tp_dist = atr_val * cfg.execution.tp_atr_multiplier
                        tp = signal.price + tp_dist if side == "buy" else signal.price - tp_dist

                # Claude confirmation
                claude_analysis = None
                confidence = None
                status = "confirmed"
                reject_reason = ""

                if sig_cfg.require_claude_confirmation and cfg.claude.enabled:
                    indicators = _compute_indicator_context(candles)

                    # Gather news and sentiment context for Claude
                    upcoming_events = news_service.get_events_for_symbol(symbol, hours_ahead=4) if cfg.news.enabled else []
                    sentiment = sentiment_service.get_current_sentiment(symbol)

                    analysis = claude_service.analyze_trade_signal(
                        symbol=symbol,
                        timeframe=timeframe,
                        strategy_name=strategy_name,
                        signal_side=side,
                        signal_reason=signal.reason,
                        price=signal.price,
                        stop_loss=sl,
                        take_profit=tp,
                        candle_summary=candles,
                        indicators=indicators,
                        upcoming_events=upcoming_events,
                        sentiment=sentiment,
                    )
                    claude_analysis = analysis
                    confidence = analysis.get("confidence", 0.0)

                    rec = analysis.get("recommendation", "skip")
                    if rec == "skip":
                        status = "rejected"
                        reject_reason = "Claude: skip"
                    elif confidence < sig_cfg.min_confidence:
                        status = "rejected"
                        reject_reason = f"Low confidence ({confidence:.0%} < {sig_cfg.min_confidence:.0%})"
                    else:
                        status = "confirmed"
                        # Apply Claude's adjusted SL/TP if provided
                        if analysis.get("adjusted_sl") is not None:
                            sl = analysis["adjusted_sl"]
                        if analysis.get("adjusted_tp") is not None:
                            tp = analysis["adjusted_tp"]

                # Persist signal
                session = get_session()
                try:
                    sig_record = SignalModel(
                        symbol=symbol,
                        timeframe=timeframe,
                        strategy_type=strategy_name,
                        side=side,
                        price=signal.price,
                        stop_loss=sl,
                        take_profit=tp,
                        confidence=confidence,
                        reason=signal.reason,
                        claude_analysis={**(claude_analysis or {}), "reject_reason": reject_reason} if reject_reason else claude_analysis,
                        status=status,
                        created_at=now,
                        resolved_at=now if status == "rejected" else None,
                    )
                    session.add(sig_record)
                    session.commit()
                    signal_id = sig_record.id
                except Exception:
                    session.rollback()
                    raise
                finally:
                    session.close()

                logger.info(
                    "Signal: %s %s %s via %s (confidence=%.2f, status=%s)",
                    side, symbol, timeframe, strategy_name,
                    confidence or 0.0, status,
                )

                # Execute confirmed signals
                if status == "confirmed":
                    try:
                        trade_manager.execute_signal(signal_id)
                    except Exception as exc:
                        logger.error("Trade execution error for signal %d: %s", signal_id, exc)


def _scan_rulesets():
    """Scan active rulesets using Claude rule evaluation."""
    cfg = load_config()
    sig_cfg = cfg.signals

    if not cfg.claude.enabled:
        return

    # Trading session window check (EST/EDT)
    from zoneinfo import ZoneInfo
    est_now = datetime.now(ZoneInfo("America/New_York"))
    est_time = est_now.hour * 60 + est_now.minute
    session_1 = (6 * 60 + 30, 16 * 60)           # 6:30 AM - 4:00 PM EST (London + full NY)
    session_2 = (21 * 60, 23 * 60 + 30)         # 9:00 PM - 11:30 PM EST (Tokyo open)
    if not (session_1[0] <= est_time <= session_1[1] or session_2[0] <= est_time <= session_2[1]):
        return

    from models.ruleset import RuleSet
    session = get_session()
    try:
        active_rulesets = session.query(RuleSet).filter(RuleSet.status == "active").all()
        rulesets = [(rs.id, rs.name, rs.rules_text, rs.symbols, rs.timeframes) for rs in active_rulesets]
    finally:
        session.close()

    if not rulesets:
        return

    logger.info("Ruleset scan tick: %d active ruleset(s)", len(rulesets))

    for rs_id, rs_name, rules_text, rs_symbols, rs_timeframes in rulesets:
        symbols = rs_symbols if rs_symbols else cfg.symbols
        timeframes = rs_timeframes if rs_timeframes else sig_cfg.timeframes

        for timeframe in timeframes:
            # Load candles for all symbols at this timeframe (for cross-pair context)
            all_candles = {}
            for sym in symbols:
                c = _query_candles(sym, timeframe)
                if c and len(c) >= 50:
                    all_candles[sym] = c

            # Fetch higher-timeframe context (1h, 4h) per symbol so Claude can
            # form directional bias from the broader picture before scoring the
            # entry-timeframe trigger. Skip redundant fetch if primary TF already
            # matches a higher TF.
            higher_tfs = [tf for tf in ("1h", "4h") if tf != timeframe]
            multi_tf_by_symbol: dict[str, dict[str, list[dict]]] = {}
            for sym in symbols:
                mtf: dict[str, list[dict]] = {}
                for htf in higher_tfs:
                    hc = _query_candles(sym, htf, limit=60)
                    if hc and len(hc) >= 20:
                        mtf[htf] = hc
                if mtf:
                    multi_tf_by_symbol[sym] = mtf

            # Per-tick counters — the summary log at the end of this (ruleset,
            # timeframe) iteration shows exactly how many symbols actually got
            # a Claude call vs were gated out. If evaluated=0 across ticks
            # until a new bar closes, the bar-close gate is doing its job.
            evaluated = 0
            skipped_same_bar = 0
            skipped_cooldown = 0
            skipped_news = 0

            for symbol in symbols:
                if sig_cfg.max_signals_per_day > 0 and _count_signals_today() >= sig_cfg.max_signals_per_day:
                    return

                candles = all_candles.get(symbol)
                if not candles:
                    continue
                if not _check_candle_freshness(candles, timeframe):
                    continue

                # Bar-close gating: skip if we've already scanned this bar
                # for this ruleset. Keeps a 15m ruleset on a 5min scan_interval
                # firing Claude once per bar instead of three.
                latest_bar_ts = _current_bar_key(timeframe, datetime.now(timezone.utc))
                if not _claim_new_bar(f"ruleset:{rs_id}", timeframe, symbol, latest_bar_ts):
                    skipped_same_bar += 1
                    continue

                # Cooldown check
                if _check_cooldown(symbol, sig_cfg.cool_down_minutes):
                    skipped_cooldown += 1
                    continue

                # News check
                blocked, _ = news_service.is_high_impact_window(symbol)
                if blocked:
                    skipped_news += 1
                    continue

                # Evaluate rules with Claude (including cross-pair context)
                indicators = rule_engine._compute_indicators(candles)
                if len(all_candles) > 1:
                    cross_ctx = rule_engine.compute_cross_pair_context(symbol, all_candles)
                    indicators.update(cross_ctx)

                upcoming_events = news_service.get_events_for_symbol(symbol, hours_ahead=1) if cfg.news.enabled else []
                # Filter to tight window like backtest
                upcoming_events = [e for e in upcoming_events if abs(e.get("minutes_until", 999)) <= 45]
                sentiment = sentiment_service.get_current_sentiment(symbol) if cfg.news.enabled else None

                evaluated += 1
                result = rule_engine.evaluate_rules(
                    rules_text, symbol, timeframe, candles,
                    indicators, upcoming_events, sentiment,
                    multi_tf_candles=multi_tf_by_symbol.get(symbol),
                )

                signal_type = result.get("signal", "HOLD")
                confidence = result.get("confidence", 0)

                # Visibility: log EVERY scan outcome, not just the ones that pass.
                # This is the only way to see whether Claude is scoring near-misses
                # vs seeing nothing at all.
                if signal_type not in ("BUY", "SELL"):
                    logger.info(
                        "Ruleset scan: %s %s via '%s' -> HOLD (conf=%.2f) | reason: %s",
                        symbol, timeframe, rs_name, confidence,
                        (result.get("reasoning") or "")[:300],
                    )
                    continue

                if confidence < sig_cfg.min_confidence:
                    logger.info(
                        "Ruleset scan: %s %s via '%s' -> %s below threshold (conf=%.2f < %.2f) - dropped | reason: %s",
                        symbol, timeframe, rs_name, signal_type, confidence, sig_cfg.min_confidence,
                        (result.get("reasoning") or "")[:300],
                    )
                    continue

                side = signal_type.lower()
                now = datetime.now(timezone.utc)
                price = candles[-1]["close"]

                sl = result.get("stop_loss")
                tp = result.get("take_profit")
                if sl is None or tp is None:
                    atr_vals = indicator_service.atr(candles, 14)
                    atr_val = atr_vals[-1] if atr_vals and atr_vals[-1] is not None else price * 0.005
                    if sl is None:
                        sl_dist = atr_val * cfg.execution.sl_atr_multiplier
                        sl = price - sl_dist if side == "buy" else price + sl_dist
                    if tp is None:
                        tp_dist = atr_val * cfg.execution.tp_atr_multiplier
                        tp = price + tp_dist if side == "buy" else price - tp_dist

                # R:R guardrail. Claude's SL/TP placements have run sub-1.5 R:R
                # often enough to bleed real money (EURUSD SELL: 15/25 trades had
                # TP closer than SL). The ATR fallback above also defaults to a
                # 0.8 R:R (sl_atr=2.5, tp_atr=2.0). With ~50% live win rate,
                # anything below 1.5 R:R is mathematically a losing strategy.
                # Widen TP to 1.5x risk before persisting.
                risk_dist = abs(price - sl)
                reward_dist = abs(price - tp)
                if risk_dist > 0 and reward_dist < 1.5 * risk_dist:
                    original_tp = tp
                    tp = price + 1.5 * risk_dist if side == "buy" else price - 1.5 * risk_dist
                    logger.info(
                        "R:R guardrail: %s %s %s reward/risk %.2f < 1.5; widened TP %.5f -> %.5f",
                        symbol, timeframe, side,
                        reward_dist / risk_dist, original_tp, tp,
                    )

                # Persist signal
                session = get_session()
                try:
                    sig_record = SignalModel(
                        symbol=symbol,
                        timeframe=timeframe,
                        strategy_type=f"ruleset_{rs_id}",
                        side=side,
                        price=price,
                        stop_loss=sl,
                        take_profit=tp,
                        confidence=confidence,
                        reason=result.get("reasoning", ""),
                        claude_analysis=result,
                        status="confirmed",
                        created_at=now,
                    )
                    session.add(sig_record)
                    session.commit()
                    signal_id = sig_record.id
                except Exception:
                    session.rollback()
                    raise
                finally:
                    session.close()

                logger.info(
                    "Ruleset signal: %s %s %s via '%s' (confidence=%.2f)",
                    side, symbol, timeframe, rs_name, confidence,
                )

                try:
                    trade_manager.execute_signal(signal_id)
                except Exception as exc:
                    logger.error("Trade execution error for signal %d: %s", signal_id, exc)

            logger.info(
                "Ruleset '%s' on %s done: evaluated=%d (claude calls), "
                "skipped_same_bar=%d, skipped_cooldown=%d, skipped_news=%d",
                rs_name, timeframe, evaluated,
                skipped_same_bar, skipped_cooldown, skipped_news,
            )


async def signal_generation_loop(stop_event: asyncio.Event) -> None:
    """Background loop that scans for trading signals."""
    logger.info("Signal generation loop started.")
    while not stop_event.is_set():
        cfg = load_config()
        if cfg.signals.enabled:
            # Scan indicator-based strategies (legacy — only if no active rulesets)
            if cfg.signals.strategies:
                try:
                    await asyncio.to_thread(_scan_once)
                except Exception as exc:
                    logger.error("Strategy scan error: %s", exc)

            # Scan active rulesets (primary signal source)
            try:
                await asyncio.to_thread(_scan_rulesets)
            except Exception as exc:
                logger.error("Ruleset scan error: %s", exc)

        try:
            await asyncio.wait_for(
                stop_event.wait(), timeout=cfg.signals.scan_interval_sec
            )
        except asyncio.TimeoutError:
            pass
    logger.info("Signal generation loop stopped.")
