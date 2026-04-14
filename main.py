"""WeldonTrader — Unified forex trading platform.

Run with:
    uvicorn main:app --reload --port 8000
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles

from config import load_config
from db import init_db
from services.candle_collector import candle_collection_loop
from services.signal_generator import signal_generation_loop
from services.trade_manager import position_sync_loop
from services.news_service import news_collection_loop
from routers.accounts import account_snapshot_loop

from routers import marketdata, history, candles, indicators, accounts, settings, trade_plans, signals, trades, news, analytics, training, rulesets

logging.basicConfig(level=logging.INFO)

BASE_DIR = Path(__file__).resolve().parent

app = FastAPI(title="WeldonTrader")
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")

# Include routers
app.include_router(marketdata.router)
app.include_router(history.router)
app.include_router(candles.router)
app.include_router(indicators.router)
app.include_router(accounts.router)
app.include_router(settings.router)
app.include_router(trade_plans.router)
app.include_router(signals.router)
app.include_router(trades.router)
app.include_router(news.router)
app.include_router(analytics.router)
app.include_router(training.router)
app.include_router(rulesets.router)


@app.on_event("startup")
async def startup():
    init_db()
    # Clean up orphaned training runs from previous sessions
    try:
        from db import get_session
        from models.training import TrainingRun
        _s = get_session()
        stuck = _s.query(TrainingRun).filter(TrainingRun.status == "running").all()
        for r in stuck:
            r.status = "stopped"
        if stuck:
            _s.commit()
            import logging
            logging.getLogger("startup").info("Cleaned %d orphaned training runs", len(stuck))
        _s.close()
    except Exception:
        pass
    app.state.stop_event = asyncio.Event()
    app.state.candle_task = asyncio.create_task(
        candle_collection_loop(app.state.stop_event)
    )
    app.state.signal_task = asyncio.create_task(
        signal_generation_loop(app.state.stop_event)
    )
    app.state.position_task = asyncio.create_task(
        position_sync_loop(app.state.stop_event)
    )
    app.state.news_task = asyncio.create_task(
        news_collection_loop(app.state.stop_event)
    )
    app.state.snapshot_task = asyncio.create_task(
        account_snapshot_loop(app.state.stop_event)
    )


@app.on_event("shutdown")
async def shutdown():
    if hasattr(app.state, "stop_event"):
        app.state.stop_event.set()
    from services.training_engine import stop_all
    stop_all()
    from services import rule_engine
    for bt_id in list(rule_engine._active_backtests):
        rule_engine._active_backtests[bt_id] = True
    for run_id in list(rule_engine._active_trains):
        rule_engine._active_trains[run_id] = True


@app.get("/")
def root():
    return RedirectResponse(url="/dashboard")


@app.get("/healthz")
def healthcheck():
    cfg = load_config()
    return {
        "status": "ok",
        "symbols": cfg.symbols,
        "candle_collection": cfg.candle_collection_enabled,
        "accounts": len(cfg.accounts),
    }
