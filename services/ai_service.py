"""Provider-neutral AI integration for trade analysis.

This module is the app-facing AI boundary.  Database fields and a few routes
still use "claude" in their names for compatibility, but new decision code
should call this service instead of provider-specific SDKs.
"""

from __future__ import annotations

import logging
from typing import Any

import httpx

from config import AppConfig, load_config
from services import claude_service

logger = logging.getLogger(__name__)


class AIServiceError(RuntimeError):
    """Raised when the configured AI provider cannot service a request."""


def is_enabled(cfg: AppConfig | None = None) -> bool:
    cfg = cfg or load_config()
    return bool(cfg.ai.enabled)


def reset_client() -> None:
    """Reset cached provider clients after settings change."""
    claude_service.reset_client()


def _provider(cfg: AppConfig) -> str:
    return (cfg.ai.provider or "claude").strip().lower()


def _model(cfg: AppConfig, review: bool = False) -> str:
    model = cfg.ai.review_model if review else cfg.ai.model
    return (model or cfg.ai.model or "local-model").strip()


def _max_tokens(cfg: AppConfig, review: bool = False, override: int | None = None) -> int:
    if override is not None:
        return max(1, int(override))
    return int(cfg.ai.review_max_tokens if review else cfg.ai.max_tokens)


def _openai_chat_url(base_url: str) -> str:
    base = (base_url or "").strip().rstrip("/")
    if not base:
        base = "http://127.0.0.1:8000/v1"
    if base.endswith("/chat/completions"):
        return base
    if base.endswith("/v1"):
        return f"{base}/chat/completions"
    return f"{base}/v1/chat/completions"


def generate_text(
    *,
    system: str,
    prompt: str,
    cfg: AppConfig | None = None,
    max_tokens: int | None = None,
    temperature: float | None = None,
    review: bool = False,
    json_mode: bool = False,
) -> str:
    """Generate text from the configured provider."""
    cfg = cfg or load_config()
    if not cfg.ai.enabled:
        raise AIServiceError("AI provider is disabled")

    temp = cfg.ai.temperature if temperature is None else temperature
    provider = _provider(cfg)

    if provider == "claude":
        client = claude_service._get_client()
        if client is None:
            raise AIServiceError("Claude API not available")
        response = client.messages.create(
            model=_model(cfg, review=review),
            max_tokens=_max_tokens(cfg, review=review, override=max_tokens),
            temperature=temp,
            system=system,
            messages=[{"role": "user", "content": prompt}],
        )
        return claude_service._extract_text(response)

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": prompt},
    ]

    if provider == "ollama":
        url = f"{cfg.ai.base_url.rstrip('/')}/api/chat"
        payload: dict[str, Any] = {
            "model": _model(cfg, review=review),
            "messages": messages,
            "stream": False,
            "think": False,
            "options": {
                "temperature": temp,
                "num_predict": _max_tokens(cfg, review=review, override=max_tokens),
            },
        }
        if json_mode:
            payload["format"] = "json"
        with httpx.Client(timeout=cfg.ai.timeout_sec) as client:
            resp = client.post(url, json=payload)
            resp.raise_for_status()
            data = resp.json()
        content = (data.get("message") or {}).get("content")
        if content is None:
            content = data.get("response")
        if not content:
            raise AIServiceError("Ollama response did not include message content")
        return str(content)

    if provider == "openai_compatible":
        headers = {"Content-Type": "application/json"}
        if cfg.ai.api_key:
            headers["Authorization"] = f"Bearer {cfg.ai.api_key}"
        payload = {
            "model": _model(cfg, review=review),
            "messages": messages,
            "temperature": temp,
            "max_tokens": _max_tokens(cfg, review=review, override=max_tokens),
        }
        # Do not force OpenAI JSON-object mode here because several trading
        # workflows legitimately return JSON arrays.
        with httpx.Client(timeout=cfg.ai.timeout_sec) as client:
            resp = client.post(_openai_chat_url(cfg.ai.base_url), headers=headers, json=payload)
            resp.raise_for_status()
            data = resp.json()
        try:
            return str(data["choices"][0]["message"]["content"])
        except (KeyError, IndexError, TypeError) as exc:
            raise AIServiceError("OpenAI-compatible response did not include message content") from exc

    raise AIServiceError(f"Unsupported AI provider: {provider}")


def generate_json(
    *,
    system: str,
    prompt: str,
    cfg: AppConfig | None = None,
    max_tokens: int | None = None,
    temperature: float | None = None,
    review: bool = False,
    fallback: Any = None,
) -> Any:
    """Generate and parse JSON. Returns fallback when parsing or provider calls fail."""
    try:
        text = generate_text(
            system=system,
            prompt=prompt,
            cfg=cfg,
            max_tokens=max_tokens,
            temperature=temperature,
            review=review,
            json_mode=True,
        )
    except Exception as exc:
        logger.error("AI provider call failed: %s", exc)
        return fallback

    parsed = claude_service._parse_json_response(text)
    if parsed is None:
        logger.warning("AI provider returned non-JSON response")
        return fallback
    return parsed


def parse_json_response(text: str) -> dict | list | None:
    return claude_service._parse_json_response(text)


def compute_price_summary(candles: list[dict]) -> dict:
    return claude_service.compute_price_summary(candles)


def _build_system_prompt() -> str:
    return (
        "You are an expert forex trading analyst integrated into an automated trading platform. "
        "Your job is to evaluate trade ideas, identify market context, and return structured JSON. "
        "You are not the final risk manager; deterministic account risk checks run after your review. "
        "Prefer clear, calibrated confidence over forced trade frequency. "
        "Always respond with valid JSON only."
    )


def analyze_trade_signal(
    symbol: str,
    timeframe: str,
    strategy_name: str,
    signal_side: str,
    signal_reason: str,
    price: float,
    stop_loss: float | None,
    take_profit: float | None,
    candle_summary: list[dict],
    indicators: dict[str, Any],
    account_equity: float | None = None,
    open_positions: list[dict] | None = None,
    upcoming_events: list[dict] | None = None,
    sentiment: dict | None = None,
) -> dict:
    cfg = load_config()
    if not cfg.ai.enabled:
        return {"confidence": 0.0, "recommendation": "skip", "reasoning": "AI provider disabled"}
    if _provider(cfg) == "claude":
        return claude_service.analyze_trade_signal(
            symbol=symbol,
            timeframe=timeframe,
            strategy_name=strategy_name,
            signal_side=signal_side,
            signal_reason=signal_reason,
            price=price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            candle_summary=candle_summary,
            indicators=indicators,
            account_equity=account_equity,
            open_positions=open_positions,
            upcoming_events=upcoming_events,
            sentiment=sentiment,
        )

    recent = candle_summary[-20:] if len(candle_summary) > 20 else candle_summary
    candle_text = "\n".join(
        f"  {c.get('time', '?')}: O={c['open']:.5f} H={c['high']:.5f} "
        f"L={c['low']:.5f} C={c['close']:.5f}"
        for c in recent
    )
    indicator_text = "\n".join(f"  {k}: {v}" for k, v in indicators.items())
    position_text = "None"
    if open_positions:
        position_text = "\n".join(
            f"  {p['symbol']} {p['side']} @ {p.get('price', '?')} vol={p.get('volume', '?')}"
            for p in open_positions
        )

    prompt = (
        f"Evaluate this forex trade candidate.\n\n"
        f"Symbol: {symbol}\nTimeframe: {timeframe}\nStrategy: {strategy_name}\n"
        f"Signal: {signal_side.upper()}\nPrice: {price:.5f}\n"
        f"Stop Loss: {f'{stop_loss:.5f}' if stop_loss else 'not set'}\n"
        f"Take Profit: {f'{take_profit:.5f}' if take_profit else 'not set'}\n"
        f"Candidate reason: {signal_reason}\n\n"
        f"Recent candles:\n{candle_text}\n\n"
        f"Indicators:\n{indicator_text}\n\n"
        f"Account equity: {account_equity or 'unknown'}\nOpen positions:\n{position_text}\n\n"
        f"Upcoming economic events:\n{claude_service._format_events(upcoming_events)}\n\n"
        f"Market sentiment: {claude_service._format_sentiment(sentiment)}\n\n"
        f"Return JSON:\n"
        f'{{"confidence": 0.0, "recommendation": "execute|skip|wait", '
        f'"adjusted_sl": null, "adjusted_tp": null, "reasoning": "<=200 chars"}}'
    )
    parsed = generate_json(
        system=_build_system_prompt(),
        prompt=prompt,
        cfg=cfg,
        max_tokens=cfg.ai.max_tokens,
        fallback={"confidence": 0.0, "recommendation": "skip", "reasoning": "AI analysis failed"},
    )
    if not isinstance(parsed, dict):
        return {"confidence": 0.0, "recommendation": "skip", "reasoning": "AI returned invalid shape"}
    parsed.setdefault("confidence", 0.0)
    parsed.setdefault("recommendation", "skip")
    parsed.setdefault("reasoning", "")
    return parsed


def analyze_optimization_results(
    symbol: str,
    timeframe: str,
    candle_count: int,
    results_markdown: str,
) -> dict:
    cfg = load_config()
    if not cfg.ai.enabled:
        return {"error": "AI provider not enabled"}
    if _provider(cfg) == "claude":
        return claude_service.analyze_optimization_results(symbol, timeframe, candle_count, results_markdown)

    prompt = (
        f"Analyze these forex strategy optimization results:\n\n"
        f"Symbol: {symbol}, Timeframe: {timeframe}, Candles: {candle_count}\n\n"
        f"{results_markdown}\n\n"
        f"Respond with JSON containing analysis, best_config, warnings, and refined_suggestions."
    )
    parsed = generate_json(system=_build_system_prompt(), prompt=prompt, cfg=cfg, review=True, fallback=None)
    return parsed if isinstance(parsed, dict) else {"error": "Could not parse AI response"}


def analyze_strategy_performance(scorecard: list[dict], signal_accuracy: dict) -> dict:
    cfg = load_config()
    if not cfg.ai.enabled:
        return {"error": "AI provider not enabled"}
    if _provider(cfg) == "claude":
        return claude_service.analyze_strategy_performance(scorecard, signal_accuracy)
    if not scorecard:
        return {"error": "No performance data to analyze"}

    scorecard_text = "\n".join(
        "  {name}: {trades} trades, {wr:.1%} win rate, PF={pf:.2f}, "
        "P&L=${pnl:.2f}, MaxDD=${dd:.2f}".format(
            name=s["strategy"], trades=s["trades"], wr=s["win_rate"],
            pf=s["profit_factor"], pnl=s["total_pnl"], dd=s["max_drawdown"],
        )
        for s in scorecard
    )
    prompt = (
        f"Review the live trading performance of the strategy portfolio.\n\n"
        f"Strategy scorecard:\n{scorecard_text}\n\n"
        f"Signal confidence accuracy:\n{signal_accuracy}\n\n"
        f"Respond with JSON containing summary, degraded_strategies, strong_strategies, "
        f"recommendations, and confidence_calibration."
    )
    parsed = generate_json(system=_build_system_prompt(), prompt=prompt, cfg=cfg, review=True, fallback=None)
    return parsed if isinstance(parsed, dict) else {"error": "Could not parse AI response"}


def analyze_training_iteration(
    strategy_type: str,
    parameters: dict,
    metrics: dict,
    trade_summary: str,
    candle_count: int,
    iteration_number: int,
    previous_metrics: dict | None = None,
    previous_params: dict | None = None,
) -> dict:
    cfg = load_config()
    if not cfg.ai.enabled:
        return {"error": "AI provider not enabled"}
    if _provider(cfg) == "claude":
        return claude_service.analyze_training_iteration(
            strategy_type=strategy_type,
            parameters=parameters,
            metrics=metrics,
            trade_summary=trade_summary,
            candle_count=candle_count,
            iteration_number=iteration_number,
            previous_metrics=previous_metrics,
            previous_params=previous_params,
        )

    prompt = (
        f"Training iteration {iteration_number} for strategy {strategy_type}.\n"
        f"Parameters: {parameters}\nCandles: {candle_count}\n"
        f"Metrics: {metrics}\nPrevious metrics: {previous_metrics}\n"
        f"Previous params: {previous_params}\nTrade summary:\n{trade_summary}\n\n"
        f"Respond with JSON containing adjusted_parameters, reasoning, data_feedback, "
        f"confidence, and risk_adjustments."
    )
    parsed = generate_json(system=_build_system_prompt(), prompt=prompt, cfg=cfg, review=True, fallback=None)
    if not isinstance(parsed, dict):
        return {"error": "Could not parse AI response"}
    parsed.setdefault("adjusted_parameters", parameters)
    parsed.setdefault("confidence", 0.5)
    return parsed


def generate_training_report(iterations: list[dict]) -> dict:
    cfg = load_config()
    if not cfg.ai.enabled:
        return {"error": "AI provider not enabled"}
    if _provider(cfg) == "claude":
        return claude_service.generate_training_report(iterations)
    if not iterations:
        return {"error": "No iterations to analyze"}

    prompt = (
        f"Review this complete forex strategy training run:\n\n{iterations}\n\n"
        f"Respond with JSON containing best_iteration, best_params, deploy_recommendation, "
        f"confidence, improvements_found, data_insights, overfitting_risk, and summary."
    )
    parsed = generate_json(system=_build_system_prompt(), prompt=prompt, cfg=cfg, review=True, fallback=None)
    return parsed if isinstance(parsed, dict) else {"error": "Could not parse AI response"}


def analyze_closed_trade(trade_data: dict) -> dict:
    cfg = load_config()
    if not cfg.ai.enabled:
        return {"error": "AI provider not enabled"}
    if _provider(cfg) == "claude":
        return claude_service.analyze_closed_trade(trade_data)

    prompt = (
        f"Analyze this completed forex trade and extract reusable lessons.\n\n"
        f"Trade data:\n{trade_data}\n\n"
        f"Respond with JSON containing outcome, pnl_assessment, entry_quality, exit_quality, "
        f"market_context, news_impact, what_went_right, what_went_wrong, lesson, "
        f"and confidence_in_analysis."
    )
    parsed = generate_json(system=_build_system_prompt(), prompt=prompt, cfg=cfg, max_tokens=2048, fallback=None)
    if not isinstance(parsed, dict):
        return {"error": "Could not parse AI response"}
    parsed.setdefault("outcome", "win" if (trade_data.get("pnl") or 0) >= 0 else "loss")
    parsed.setdefault("confidence_in_analysis", 0.5)
    return parsed
