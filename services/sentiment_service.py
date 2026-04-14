"""Market sentiment analysis via Claude and news headlines.

Computes a sentiment score (-1.0 bearish to +1.0 bullish) for currency
pairs based on recent news and economic context.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

from config import load_config
from db import get_session
from models.news import SentimentScore
from services import claude_service, news_service

logger = logging.getLogger(__name__)


def compute_sentiment(symbol: str) -> dict:
    """Compute sentiment for a currency pair using available context.

    Uses Claude (if enabled) to analyze upcoming events and market context.
    Returns {"score": float, "reasoning": str, "source": str}.
    """
    cfg = load_config()

    # Gather context for the pair
    events = news_service.get_events_for_symbol(symbol, hours_ahead=24)

    high_events = [e for e in events if e["impact"] == "high"]
    medium_events = [e for e in events if e["impact"] == "medium"]

    # If Claude is enabled, use it for sentiment analysis
    if cfg.claude.enabled and cfg.news.sentiment_enabled:
        return _claude_sentiment(symbol, events)

    # Fallback: rule-based sentiment from event count and direction
    return _rule_based_sentiment(symbol, high_events, medium_events)


def _claude_sentiment(symbol: str, events: list[dict]) -> dict:
    """Use Claude to analyze sentiment based on upcoming events."""
    client = claude_service._get_client()
    if client is None:
        return {"score": 0.0, "reasoning": "Claude not available", "source": "fallback"}

    cfg = load_config()
    base = symbol[:3]
    quote = symbol[3:6]

    events_text = "None upcoming" if not events else "\n".join(
        "  - {title} ({ccy}, {imp} impact) in {mins} min{fc}{pv}".format(
            title=e["title"], ccy=e["currency"], imp=e["impact"],
            mins=e.get("minutes_until", "?"),
            fc=f" | Forecast: {e['forecast']}" if e.get("forecast") else "",
            pv=f" | Previous: {e['previous']}" if e.get("previous") else "",
        )
        for e in events
    )

    prompt = (
        f"Analyze the current market sentiment for {symbol} ({base}/{quote}).\n\n"
        f"Upcoming economic events:\n{events_text}\n\n"
        f"Based on the economic calendar and typical market behavior, "
        f"provide a sentiment assessment.\n\n"
        f"Respond with JSON:\n"
        f'{{\n'
        f'  "score": -1.0 to 1.0 (negative=bearish, positive=bullish for {base} vs {quote}),\n'
        f'  "reasoning": "brief explanation"\n'
        f'}}'
    )

    try:
        response = client.messages.create(
            model=cfg.claude.model,
            max_tokens=512,
            temperature=0.2,
            system="You are a forex market sentiment analyst. Provide concise JSON assessments.",
            messages=[{"role": "user", "content": prompt}],
        )
        text = response.content[0].text
        parsed = claude_service._parse_json_response(text)
        if parsed and "score" in parsed:
            score = max(-1.0, min(1.0, float(parsed["score"])))
            result = {
                "score": score,
                "reasoning": parsed.get("reasoning", ""),
                "source": "claude_analysis",
            }
            _store_sentiment(symbol, score, "claude_analysis", parsed)
            return result
    except Exception as exc:
        logger.error("Claude sentiment error: %s", exc)

    return {"score": 0.0, "reasoning": "Analysis failed", "source": "fallback"}


def _rule_based_sentiment(
    symbol: str,
    high_events: list[dict],
    medium_events: list[dict],
) -> dict:
    """Simple rule-based sentiment when Claude is unavailable.

    - High-impact events nearby → neutral (uncertainty)
    - No events → slightly bullish bias (risk-on default)
    """
    if high_events:
        score = 0.0
        reasoning = f"{len(high_events)} high-impact event(s) upcoming — elevated uncertainty"
    elif medium_events:
        score = 0.0
        reasoning = f"{len(medium_events)} medium-impact event(s) upcoming"
    else:
        score = 0.1
        reasoning = "No significant events upcoming — low-volatility environment"

    _store_sentiment(symbol, score, "rule_based", {"high": len(high_events), "medium": len(medium_events)})
    return {"score": score, "reasoning": reasoning, "source": "rule_based"}


def _store_sentiment(symbol: str, score: float, source: str, details: dict | None):
    """Persist a sentiment score to the database."""
    session = get_session()
    try:
        record = SentimentScore(
            symbol=symbol,
            score=score,
            source=source,
            details=details,
            ts=datetime.now(timezone.utc),
        )
        session.add(record)
        session.commit()
    except Exception:
        session.rollback()
    finally:
        session.close()


def get_current_sentiment(symbol: str) -> dict | None:
    """Get the most recent sentiment score for a symbol."""
    session = get_session()
    try:
        row = session.query(SentimentScore).filter(
            SentimentScore.symbol == symbol,
        ).order_by(SentimentScore.ts.desc()).first()

        if row is None:
            return None

        age_minutes = (datetime.now(timezone.utc) - row.ts).total_seconds() / 60
        return {
            "symbol": row.symbol,
            "score": row.score,
            "source": row.source,
            "details": row.details,
            "ts": row.ts.isoformat(),
            "age_minutes": round(age_minutes, 1),
        }
    finally:
        session.close()


def get_sentiment_history(symbol: str, hours: int = 24) -> list[dict]:
    """Get sentiment score history for a symbol."""
    cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
    session = get_session()
    try:
        rows = session.query(SentimentScore).filter(
            SentimentScore.symbol == symbol,
            SentimentScore.ts >= cutoff,
        ).order_by(SentimentScore.ts.asc()).all()

        return [
            {"score": r.score, "source": r.source, "ts": r.ts.isoformat()}
            for r in rows
        ]
    finally:
        session.close()
