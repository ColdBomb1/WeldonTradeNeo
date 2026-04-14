from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from fastapi.templating import Jinja2Templates

from config import load_config
from db import get_session
from models.training import TrainingRun, TrainingIteration
from services import training_engine
from services.strategy_service import list_strategies

router = APIRouter(tags=["training"])

BASE_DIR = Path(__file__).resolve().parent.parent
TEMPLATES = Jinja2Templates(directory=str(BASE_DIR / "templates"))


@router.get("/training")
def training_page(request: Request):
    cfg = load_config()
    return TEMPLATES.TemplateResponse(
        "training.html",
        {
            "request": request,
            "symbols": cfg.symbols,
            "timeframes": cfg.candle_timeframes,
            "strategies": list_strategies(),
            "training_config": cfg.training,
        },
    )


@router.get("/api/training/runs")
def get_training_runs() -> JSONResponse:
    session = get_session()
    try:
        runs = session.query(TrainingRun).order_by(TrainingRun.created_at.desc()).all()
        return JSONResponse({
            "runs": [
                {
                    "id": r.id,
                    "name": r.name,
                    "status": r.status,
                    "symbols": r.symbols,
                    "timeframes": r.timeframes,
                    "strategies": r.strategies,
                    "start_date": r.start_date.isoformat(),
                    "end_date": r.end_date.isoformat(),
                    "iterations_completed": r.iterations_completed,
                    "created_at": r.created_at.isoformat(),
                    "completed_at": r.completed_at.isoformat() if r.completed_at else None,
                }
                for r in runs
            ]
        })
    finally:
        session.close()


@router.post("/api/training/start")
async def start_training(request: Request) -> JSONResponse:
    payload = await request.json()

    symbols = payload.get("symbols", [])
    timeframes = payload.get("timeframes", ["1h"])
    strategies = payload.get("strategies", [])
    start_date_str = payload.get("start_date", "")
    end_date_str = payload.get("end_date", "")
    name = payload.get("name", "")

    if not symbols:
        return JSONResponse({"error": "No symbols selected"}, status_code=400)
    if not strategies:
        return JSONResponse({"error": "No strategies selected"}, status_code=400)
    if not start_date_str or not end_date_str:
        return JSONResponse({"error": "Start and end dates required"}, status_code=400)

    start_date = datetime.fromisoformat(start_date_str)
    end_date = datetime.fromisoformat(end_date_str)
    if start_date.tzinfo is None:
        start_date = start_date.replace(tzinfo=timezone.utc)
    if end_date.tzinfo is None:
        end_date = end_date.replace(tzinfo=timezone.utc)

    if not name:
        name = f"Training {', '.join(symbols[:3])} {', '.join(strategies[:3])}"
    name = name[:128]

    # Create the run record first so we can return the ID immediately
    from models.training import TrainingRun as TRModel
    now = datetime.now(timezone.utc)
    session = get_session()
    try:
        run = TRModel(
            name=name,
            status="pending",
            symbols=symbols,
            timeframes=timeframes,
            strategies=strategies,
            start_date=start_date,
            end_date=end_date,
            iterations_completed=0,
            config_snapshot=load_config().training.model_dump(),
            created_at=now,
        )
        session.add(run)
        session.commit()
        run_id = run.id
    except Exception as exc:
        session.rollback()
        import traceback
        traceback.print_exc()
        return JSONResponse({"error": f"Failed to create training run: {exc}"}, status_code=500)
    finally:
        session.close()

    # Run training in background thread
    async def _run():
        await asyncio.to_thread(
            training_engine.run_training,
            name, symbols, timeframes, strategies, start_date, end_date,
            run_id,
        )

    asyncio.create_task(_run())

    return JSONResponse({"ok": True, "run_id": run_id, "message": "Training started"})


@router.get("/api/training/runs/{run_id}")
def get_training_run(run_id: int) -> JSONResponse:
    session = get_session()
    try:
        run = session.query(TrainingRun).filter(TrainingRun.id == run_id).first()
        if not run:
            return JSONResponse({"error": "Run not found"}, status_code=404)

        iterations = (
            session.query(TrainingIteration)
            .filter(TrainingIteration.training_run_id == run_id)
            .order_by(TrainingIteration.iteration_number.asc())
            .all()
        )

        return JSONResponse({
            "run": {
                "id": run.id,
                "name": run.name,
                "status": run.status,
                "symbols": run.symbols,
                "timeframes": run.timeframes,
                "strategies": run.strategies,
                "start_date": run.start_date.isoformat(),
                "end_date": run.end_date.isoformat(),
                "iterations_completed": run.iterations_completed,
                "final_report": run.final_report,
                "created_at": run.created_at.isoformat(),
                "completed_at": run.completed_at.isoformat() if run.completed_at else None,
            },
            "iterations": [
                {
                    "id": it.id,
                    "iteration_number": it.iteration_number,
                    "symbol": it.symbol,
                    "timeframe": it.timeframe,
                    "strategy_type": it.strategy_type,
                    "parameters": it.parameters,
                    "final_balance": it.final_balance,
                    "total_trades": it.total_trades,
                    "win_rate": it.win_rate,
                    "max_drawdown": it.max_drawdown,
                    "profit_factor": it.profit_factor,
                    "sharpe_ratio": it.sharpe_ratio,
                    "avg_monthly_return": it.avg_monthly_return,
                    "improvement_pct": it.improvement_pct,
                    "claude_analysis": it.claude_analysis,
                    "suggested_params": it.suggested_params,
                    "created_at": it.created_at.isoformat(),
                }
                for it in iterations
            ],
        })
    finally:
        session.close()


@router.post("/api/training/runs/{run_id}/stop")
def stop_training(run_id: int) -> JSONResponse:
    training_engine.request_stop(run_id)
    return JSONResponse({"ok": True, "message": "Stop signal sent"})


@router.post("/api/training/runs/{run_id}/deploy")
def deploy_training(run_id: int) -> JSONResponse:
    result = training_engine.deploy_best_params(run_id)
    if "error" in result:
        return JSONResponse(result, status_code=400)
    return JSONResponse({"ok": True, "deployed": result})


@router.get("/api/training/runs/{run_id}/report")
def get_training_report(run_id: int) -> JSONResponse:
    session = get_session()
    try:
        run = session.query(TrainingRun).filter(TrainingRun.id == run_id).first()
        if not run:
            return JSONResponse({"error": "Run not found"}, status_code=404)
        if not run.final_report:
            return JSONResponse({"error": "No report available yet"}, status_code=400)
        return JSONResponse({"report": run.final_report})
    finally:
        session.close()
