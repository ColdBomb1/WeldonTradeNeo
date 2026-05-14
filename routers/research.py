from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from fastapi.templating import Jinja2Templates

from config import load_config
from db import get_session
from models.candle import Candle
from models.research import ResearchCandidate, ResearchRun
from models.ruleset import RuleSet
from services import research_engine

router = APIRouter(tags=["research"])

BASE_DIR = Path(__file__).resolve().parent.parent
TEMPLATES = Jinja2Templates(directory=str(BASE_DIR / "templates"))


@router.get("/research")
def research_page(request: Request):
    cfg = load_config()
    return TEMPLATES.TemplateResponse(
        "research.html",
        {
            "request": request,
            "symbols": cfg.symbols,
            "timeframes": cfg.candle_timeframes,
            "defaults": research_engine.DEFAULT_SETTINGS,
        },
    )


@router.get("/api/research/runs")
def list_research_runs() -> JSONResponse:
    session = get_session()
    try:
        runs = session.query(ResearchRun).order_by(ResearchRun.created_at.desc()).all()
        return JSONResponse({
            "runs": [
                {
                    "id": r.id,
                    "name": r.name,
                    "status": r.status,
                    "symbols": r.symbols,
                    "timeframe": r.timeframe,
                    "start_date": r.start_date.isoformat(),
                    "end_date": r.end_date.isoformat(),
                    "candidate_ruleset_id": r.candidate_ruleset_id,
                    "created_at": r.created_at.isoformat(),
                    "completed_at": r.completed_at.isoformat() if r.completed_at else None,
                    "best": (r.result or {}).get("best_candidate") if r.result else None,
                }
                for r in runs
            ]
        })
    finally:
        session.close()


@router.get("/api/research/runs/{run_id}")
def get_research_run(run_id: int) -> JSONResponse:
    session = get_session()
    try:
        run = session.query(ResearchRun).filter(ResearchRun.id == run_id).first()
        if not run:
            return JSONResponse({"error": "Research run not found"}, status_code=404)
        candidates = (
            session.query(ResearchCandidate)
            .filter(ResearchCandidate.research_run_id == run_id)
            .order_by(ResearchCandidate.rank.asc())
            .all()
        )
        result_candidates = {
            int(item.get("rank") or 0): item
            for item in ((run.result or {}).get("top_candidates") or [])
        }
        return JSONResponse({
            "run": {
                "id": run.id,
                "name": run.name,
                "status": run.status,
                "symbols": run.symbols,
                "timeframe": run.timeframe,
                "start_date": run.start_date.isoformat(),
                "end_date": run.end_date.isoformat(),
                "settings": run.settings,
                "result": run.result,
                "candidate_ruleset_id": run.candidate_ruleset_id,
                "created_at": run.created_at.isoformat(),
                "completed_at": run.completed_at.isoformat() if run.completed_at else None,
            },
            "candidates": [
                {
                    "id": c.id,
                    "rank": c.rank,
                    "generation": c.generation,
                    "genome": c.genome,
                    "train_metrics": c.train_metrics,
                    "validation_metrics": c.validation_metrics,
                    "validation_stress_metrics": result_candidates.get(c.rank, {}).get("validation_stress_metrics"),
                    "holdout_metrics": result_candidates.get(c.rank, {}).get("holdout_metrics"),
                    "holdout_stress_metrics": result_candidates.get(c.rank, {}).get("holdout_stress_metrics"),
                    "gates": result_candidates.get(c.rank, {}).get("gates"),
                    "reasons": result_candidates.get(c.rank, {}).get("reasons", []),
                    "score": c.score,
                    "passed": c.passed,
                    "ruleset_id": c.ruleset_id,
                    "created_at": c.created_at.isoformat(),
                }
                for c in candidates
            ],
        })
    finally:
        session.close()


@router.post("/api/research/start")
async def start_research(request: Request) -> JSONResponse:
    payload = await request.json()
    cfg = load_config()
    symbols = payload.get("symbols") or cfg.symbols
    if isinstance(symbols, str):
        symbols = cfg.symbols if symbols.lower() == "all" else [symbols]
    timeframe = payload.get("timeframe", "15m")
    start_date_str = payload.get("start_date", "")
    end_date_str = payload.get("end_date", "")
    if not start_date_str or not end_date_str:
        return JSONResponse({"error": "start_date and end_date required"}, status_code=400)
    start_date = _parse_dt(start_date_str)
    end_date = _parse_dt(end_date_str)
    settings = research_engine.normalize_settings(payload.get("settings") or {})

    session = get_session()
    try:
        candles_by_symbol: dict[str, list[dict]] = {}
        for symbol in symbols:
            rows = (
                session.query(Candle)
                .filter(
                    Candle.symbol == symbol,
                    Candle.timeframe == timeframe,
                    Candle.ts >= start_date,
                    Candle.ts <= end_date,
                )
                .order_by(Candle.ts.asc())
                .all()
            )
            if len(rows) >= 260:
                candles_by_symbol[symbol] = [
                    {
                        "time": c.ts.isoformat(),
                        "open": c.open,
                        "high": c.high,
                        "low": c.low,
                        "close": c.close,
                        "volume": c.volume or 0,
                    }
                    for c in rows
                ]
        if not candles_by_symbol:
            return JSONResponse({"error": "Not enough candle data for selected symbols/timeframe"}, status_code=400)

        now = datetime.now(timezone.utc)
        run = ResearchRun(
            name=(payload.get("name") or f"Research {', '.join(candles_by_symbol.keys())}/{timeframe}")[:128],
            status="running",
            symbols=list(candles_by_symbol.keys()),
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date,
            settings=settings,
            created_at=now,
        )
        session.add(run)
        session.commit()
        run_id = run.id
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()

    async def _run():
        try:
            result = await asyncio.to_thread(
                research_engine.run_research,
                run_id=run_id,
                candles_by_symbol=candles_by_symbol,
                timeframe=timeframe,
                settings=settings,
            )
            _persist_result(run_id, result)
        except Exception as exc:
            import logging
            logging.getLogger(__name__).exception("Research run %d failed: %s", run_id, exc)
            sess = get_session()
            try:
                run = sess.query(ResearchRun).filter(ResearchRun.id == run_id).first()
                if run:
                    run.status = "failed"
                    run.completed_at = datetime.now(timezone.utc)
                    run.result = {"error": str(exc)}
                sess.commit()
            except Exception:
                sess.rollback()
            finally:
                sess.close()

    asyncio.create_task(_run())
    return JSONResponse({"ok": True, "run_id": run_id})


@router.get("/api/research/runs/{run_id}/progress")
def research_progress(run_id: int) -> JSONResponse:
    progress = research_engine.get_progress(run_id)
    if progress is None:
        return JSONResponse({"error": "Not found"}, status_code=404)
    return JSONResponse(progress)


@router.post("/api/research/runs/{run_id}/stop")
def stop_research(run_id: int) -> JSONResponse:
    research_engine.request_stop(run_id)
    return JSONResponse({"ok": True})


def _persist_result(run_id: int, result: dict) -> None:
    now = datetime.now(timezone.utc)
    session = get_session()
    try:
        run = session.query(ResearchRun).filter(ResearchRun.id == run_id).first()
        if not run:
            return
        run.status = result.get("status", "completed")
        run.result = result
        run.completed_at = now

        best = result.get("best_candidate")
        ruleset_id = None
        if best:
            validation = {
                "status": "passed" if best.get("passed") else "failed",
                "passed": bool(best.get("passed")),
                "validated_at": now.isoformat(),
                "method": "walk_forward_train_validation_holdout_research",
                "criteria": result.get("settings", {}),
                "reasons": best.get("reasons", []),
                "gates": best.get("gates", {}),
                "metrics": {
                    "train": best.get("train_metrics"),
                    "validation": best.get("validation_metrics"),
                    "validation_stress": best.get("validation_stress_metrics"),
                    "holdout": best.get("holdout_metrics"),
                    "holdout_stress": best.get("holdout_stress_metrics"),
                },
            }
            genome = best.get("genome") or {}
            rs = RuleSet(
                name=f"Research candidate run {run_id}"[:128],
                description="Structured walk-forward research candidate. Promote only after review.",
                status="validated" if validation["passed"] else "candidate",
                rules_text=research_engine.genome_to_rules_text(genome),
                parameters={
                    "candidate": True,
                    "execution_engine": "research_genome",
                    "strategy_schema": genome,
                    "research_run_id": run_id,
                    "validation": validation,
                },
                symbols=result.get("symbols") or run.symbols,
                timeframes=[result.get("timeframe") or run.timeframe],
                version=1,
                performance_metrics={
                    **(best.get("holdout_metrics") or best.get("validation_metrics") or {}),
                    "train_metrics": best.get("train_metrics"),
                    "validation_metrics": best.get("validation_metrics"),
                    "validation_stress_metrics": best.get("validation_stress_metrics"),
                    "holdout_metrics": best.get("holdout_metrics"),
                    "holdout_stress_metrics": best.get("holdout_stress_metrics"),
                    "gates": best.get("gates", {}),
                    "failure_reasons": best.get("reasons", []),
                    "validated": validation["passed"],
                    "validation_status": validation["status"],
                    "selection_basis": "train_validation_holdout_cost_stress",
                    "research_run_id": run_id,
                },
                created_at=now,
                updated_at=now,
            )
            session.add(rs)
            session.flush()
            ruleset_id = rs.id
            run.candidate_ruleset_id = ruleset_id
            result["candidate_ruleset_id"] = ruleset_id
            run.result = result

        for item in result.get("top_candidates", []):
            candidate = ResearchCandidate(
                research_run_id=run_id,
                generation=int(item.get("generation") or 0),
                rank=int(item.get("rank") or 0),
                genome=item.get("genome") or {},
                train_metrics=item.get("train_metrics"),
                validation_metrics=item.get("validation_metrics"),
                score=float(item.get("score") or 0.0),
                passed=bool(item.get("passed")),
                ruleset_id=ruleset_id if int(item.get("rank") or 0) == 1 else None,
                created_at=now,
            )
            session.add(candidate)
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def _parse_dt(value: str) -> datetime:
    dt = datetime.fromisoformat(value)
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
