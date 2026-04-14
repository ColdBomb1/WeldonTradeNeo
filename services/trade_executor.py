from __future__ import annotations

from datetime import datetime, timezone

from config import AppConfig


def execute_recommendation(cfg: AppConfig, recommendation: dict) -> dict:
    """Stub trade execution. Replace with live broker calls."""

    return {
        "symbol": recommendation.get("symbol", ""),
        "side": recommendation.get("side", ""),
        "price": float(recommendation.get("price", 0)),
        "volume": float(recommendation.get("volume", 0)),
        "status": "simulated",
        "server_id": cfg.server_id,
        "timestamp": datetime.now(tz=timezone.utc).isoformat(),
        "source": {"app": "weldon_trader", "server_id": cfg.server_id},
        "recommendation": recommendation,
    }
