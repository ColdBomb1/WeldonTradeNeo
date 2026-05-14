from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from fastapi.templating import Jinja2Templates

from config import load_config
from services import ai_service, performance_service, risk_manager

router = APIRouter(tags=["analytics"])

BASE_DIR = Path(__file__).resolve().parent.parent
TEMPLATES = Jinja2Templates(directory=str(BASE_DIR / "templates"))


@router.get("/analytics")
def analytics_page(request: Request):
    cfg = load_config()
    return TEMPLATES.TemplateResponse(
        "analytics.html",
        {
            "request": request,
            "symbols": cfg.symbols,
        },
    )


@router.get("/api/analytics/daily-pnl")
def get_daily_pnl(days: int = 30) -> JSONResponse:
    data = performance_service.get_daily_pnl(days=days)
    return JSONResponse({"days": days, "daily_pnl": data})


@router.get("/api/analytics/equity-curve")
def get_equity_curve() -> JSONResponse:
    curve = performance_service.get_equity_curve()
    return JSONResponse({"points": len(curve), "curve": curve})


@router.get("/api/analytics/strategies")
def get_strategy_scorecard() -> JSONResponse:
    scorecard = performance_service.get_strategy_scorecard()
    return JSONResponse({"strategies": scorecard})


@router.get("/api/analytics/symbols")
def get_symbol_breakdown() -> JSONResponse:
    breakdown = performance_service.get_symbol_breakdown()
    return JSONResponse({"symbols": breakdown})


@router.get("/api/analytics/signal-accuracy")
def get_signal_accuracy() -> JSONResponse:
    accuracy = performance_service.get_signal_accuracy()
    return JSONResponse(accuracy)


@router.get("/api/analytics/risk")
def get_risk_summary() -> JSONResponse:
    summary = risk_manager.get_risk_summary()
    return JSONResponse(summary)


@router.get("/api/analytics/live-vs-backtest")
def get_live_vs_backtest(strategy: str, symbol: str) -> JSONResponse:
    comparison = performance_service.get_live_vs_backtest(strategy, symbol)
    return JSONResponse(comparison)


@router.post("/api/analytics/claude-review")
def run_claude_performance_review() -> JSONResponse:
    """Ask the configured AI provider to analyze overall trading performance."""
    cfg = load_config()
    if not ai_service.is_enabled(cfg):
        return JSONResponse({"error": "AI provider not enabled"}, status_code=400)

    scorecard = performance_service.get_strategy_scorecard()
    accuracy = performance_service.get_signal_accuracy()

    if not scorecard:
        return JSONResponse({"error": "No closed trades to analyze yet"}, status_code=400)

    analysis = ai_service.analyze_strategy_performance(scorecard, accuracy)
    if "error" in analysis and "raw" not in analysis:
        return JSONResponse(analysis, status_code=500)

    return JSONResponse({"ok": True, "analysis": analysis})
