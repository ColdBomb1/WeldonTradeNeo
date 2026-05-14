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
from models.ruleset import RuleSet
from models.training import TrainingRun
from services import ai_service, rule_engine
from services.strategy_service import list_strategies

router = APIRouter(tags=["rulesets"])

BASE_DIR = Path(__file__).resolve().parent.parent
TEMPLATES = Jinja2Templates(directory=str(BASE_DIR / "templates"))


def _ruleset_to_dict(r: RuleSet) -> dict:
    return {
        "id": r.id,
        "name": r.name,
        "description": r.description,
        "status": r.status,
        "rules_text": r.rules_text,
        "parameters": r.parameters,
        "symbols": r.symbols,
        "timeframes": r.timeframes,
        "version": r.version,
        "parent_id": r.parent_id,
        "performance_metrics": r.performance_metrics,
        "created_at": r.created_at.isoformat() if r.created_at else None,
        "updated_at": r.updated_at.isoformat() if r.updated_at else None,
    }


@router.get("/rulesets")
def rulesets_page(request: Request):
    cfg = load_config()
    return TEMPLATES.TemplateResponse(
        "rulesets.html",
        {
            "request": request,
            "config": cfg,
            "symbols": cfg.symbols,
            "timeframes": cfg.candle_timeframes,
        },
    )


@router.get("/api/rulesets")
def get_rulesets() -> JSONResponse:
    session = get_session()
    try:
        rows = session.query(RuleSet).order_by(RuleSet.updated_at.desc()).all()
        return JSONResponse({"rulesets": [_ruleset_to_dict(r) for r in rows]})
    finally:
        session.close()


@router.post("/api/rulesets")
async def create_ruleset(request: Request) -> JSONResponse:
    payload = await request.json()
    now = datetime.now(timezone.utc)
    session = get_session()
    try:
        rs = RuleSet(
            name=payload.get("name", "New Rule Set")[:128],
            description=payload.get("description", "")[:512],
            status="inactive",
            rules_text=payload.get("rules_text", ""),
            symbols=payload.get("symbols", []),
            timeframes=payload.get("timeframes", ["1h", "4h"]),
            version=1,
            created_at=now,
            updated_at=now,
        )
        session.add(rs)
        session.commit()
        return JSONResponse({"ok": True, "id": rs.id})
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


@router.get("/api/rulesets/{rs_id}")
def get_ruleset(rs_id: int) -> JSONResponse:
    session = get_session()
    try:
        rs = session.query(RuleSet).filter(RuleSet.id == rs_id).first()
        if not rs:
            return JSONResponse({"error": "Not found"}, status_code=404)
        return JSONResponse(_ruleset_to_dict(rs))
    finally:
        session.close()


@router.put("/api/rulesets/{rs_id}")
async def update_ruleset(rs_id: int, request: Request) -> JSONResponse:
    payload = await request.json()
    session = get_session()
    try:
        rs = session.query(RuleSet).filter(RuleSet.id == rs_id).first()
        if not rs:
            return JSONResponse({"error": "Not found"}, status_code=404)
        if "name" in payload:
            rs.name = payload["name"][:128]
        if "description" in payload:
            rs.description = payload["description"][:512]
        if "rules_text" in payload:
            rs.rules_text = payload["rules_text"]
            rs.version += 1
        if "symbols" in payload:
            rs.symbols = payload["symbols"]
        if "timeframes" in payload:
            rs.timeframes = payload["timeframes"]
        rs.updated_at = datetime.now(timezone.utc)
        session.commit()
        return JSONResponse({"ok": True})
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


@router.delete("/api/rulesets/{rs_id}")
def delete_ruleset(rs_id: int) -> JSONResponse:
    session = get_session()
    try:
        rs = session.query(RuleSet).filter(RuleSet.id == rs_id).first()
        if not rs:
            return JSONResponse({"error": "Not found"}, status_code=404)
        if rs.status == "active":
            return JSONResponse({"error": "Cannot delete an active ruleset. Deactivate it first."}, status_code=400)
        session.delete(rs)
        session.commit()
        return JSONResponse({"ok": True})
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


@router.post("/api/rulesets/{rs_id}/activate")
def activate_ruleset(rs_id: int) -> JSONResponse:
    session = get_session()
    try:
        rs = session.query(RuleSet).filter(RuleSet.id == rs_id).first()
        if not rs:
            return JSONResponse({"error": "Not found"}, status_code=404)
        rs.status = "active"
        rs.updated_at = datetime.now(timezone.utc)
        session.commit()
        return JSONResponse({"ok": True, "status": "active"})
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


@router.post("/api/rulesets/{rs_id}/deactivate")
def deactivate_ruleset(rs_id: int) -> JSONResponse:
    session = get_session()
    try:
        rs = session.query(RuleSet).filter(RuleSet.id == rs_id).first()
        if not rs:
            return JSONResponse({"error": "Not found"}, status_code=404)
        rs.status = "inactive"
        rs.updated_at = datetime.now(timezone.utc)
        session.commit()
        return JSONResponse({"ok": True, "status": "inactive"})
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


@router.post("/api/rulesets/{rs_id}/duplicate")
def duplicate_ruleset(rs_id: int) -> JSONResponse:
    session = get_session()
    try:
        rs = session.query(RuleSet).filter(RuleSet.id == rs_id).first()
        if not rs:
            return JSONResponse({"error": "Not found"}, status_code=404)
        now = datetime.now(timezone.utc)
        clone = RuleSet(
            name=f"{rs.name} (copy)"[:128],
            description=rs.description,
            status="inactive",
            rules_text=rs.rules_text,
            symbols=rs.symbols,
            timeframes=rs.timeframes,
            version=1,
            parent_id=rs.id,
            created_at=now,
            updated_at=now,
        )
        session.add(clone)
        session.commit()
        return JSONResponse({"ok": True, "id": clone.id})
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


@router.post("/api/rulesets/{rs_id}/backtest")
async def backtest_ruleset(rs_id: int, request: Request) -> JSONResponse:
    payload = await request.json()
    # Support both single symbol and multi-symbol ("all")
    symbols_input = payload.get("symbols") or payload.get("symbol", "EURUSD")
    if isinstance(symbols_input, str):
        if symbols_input.lower() == "all":
            cfg = load_config()
            symbols = cfg.symbols
        else:
            symbols = [symbols_input]
    else:
        symbols = symbols_input

    timeframe = payload.get("timeframe", "4h")
    start_date_str = payload.get("start_date", "")
    end_date_str = payload.get("end_date", "")

    if not start_date_str or not end_date_str:
        return JSONResponse({"error": "start_date and end_date required"}, status_code=400)

    start_date = datetime.fromisoformat(start_date_str)
    end_date = datetime.fromisoformat(end_date_str)
    if start_date.tzinfo is None:
        start_date = start_date.replace(tzinfo=timezone.utc)
    if end_date.tzinfo is None:
        end_date = end_date.replace(tzinfo=timezone.utc)

    session = get_session()
    try:
        rs = session.query(RuleSet).filter(RuleSet.id == rs_id).first()
        if not rs:
            return JSONResponse({"error": "Not found"}, status_code=404)
        rules_text = rs.rules_text

        # Load candles for ALL symbols (primary + cross-pair context)
        all_pair_candles = {}
        for sym in symbols:
            rows = (
                session.query(Candle)
                .filter(Candle.symbol == sym, Candle.timeframe == timeframe,
                        Candle.ts >= start_date, Candle.ts <= end_date)
                .order_by(Candle.ts.asc()).all()
            )
            all_pair_candles[sym] = [
                {"time": c.ts.isoformat(), "open": c.open, "high": c.high,
                 "low": c.low, "close": c.close, "volume": c.volume or 0}
                for c in rows
            ]
    finally:
        session.close()

    # Validate at least one symbol has enough data
    valid_symbols = [s for s in symbols if len(all_pair_candles.get(s, [])) >= 50]
    if not valid_symbols:
        return JSONResponse({"error": "Not enough candle data for any selected symbol"}, status_code=400)

    mode = payload.get("mode", "batch")
    if mode not in ("batch", "filtered", "bar_by_bar"):
        mode = "batch"

    bt_id = rule_engine._new_backtest_id()
    rule_engine._active_backtests[bt_id] = False
    total_bars = sum(len(all_pair_candles[s]) for s in valid_symbols)
    rule_engine._backtest_progress[bt_id] = {
        "bt_id": bt_id, "status": "starting", "bar": 0,
        "total_bars": total_bars, "pct": 0, "mode": mode,
        "symbols": valid_symbols,
    }

    async def _run():
        try:
            # Run backtests across all symbols with cross-pair context
            combined_trades = []
            combined_equity = []
            combined_balance = load_config().training.initial_balance
            per_symbol_results = {}

            for sym in valid_symbols:
                candle_dicts = all_pair_candles[sym]
                result = await asyncio.to_thread(
                    rule_engine.backtest_rules, rules_text, sym, timeframe,
                    candle_dicts, bt_id=bt_id, mode=mode,
                    all_pair_candles=all_pair_candles,
                )
                per_symbol_results[sym] = {
                    k: result.get(k) for k in
                    ("final_balance", "total_trades", "win_rate",
                     "max_drawdown", "profit_factor", "sharpe_ratio")
                }
                combined_trades.extend(result.get("trades", []))
                combined_equity.extend(result.get("equity_curve", []))

            # Compute combined metrics
            total_t = len(combined_trades)
            wins = sum(1 for t in combined_trades if t.get("pnl", 0) > 0)
            gross_p = sum(t["pnl"] for t in combined_trades if t.get("pnl", 0) > 0)
            gross_l = abs(sum(t["pnl"] for t in combined_trades if t.get("pnl", 0) <= 0))
            pf = gross_p / gross_l if gross_l > 0 else (999.0 if gross_p > 0 else 0)
            total_pnl = sum(t.get("pnl", 0) for t in combined_trades)

            combined_result = {
                "total_trades": total_t,
                "win_rate": round(wins / total_t, 4) if total_t > 0 else 0,
                "profit_factor": round(min(pf, 999.0), 2),
                "final_balance": round(10000 + total_pnl, 2),
                "per_symbol": per_symbol_results,
            }
        except asyncio.CancelledError:
            return
        except Exception as exc:
            import logging
            logging.getLogger(__name__).error("Backtest error: %s", exc)
            return
        # Save combined metrics to ruleset
        sess = get_session()
        try:
            r = sess.query(RuleSet).filter(RuleSet.id == rs_id).first()
            if r:
                r.performance_metrics = combined_result
                r.updated_at = datetime.now(timezone.utc)
                sess.commit()
        except Exception:
            sess.rollback()
        finally:
            sess.close()

    asyncio.create_task(_run())

    return JSONResponse({"ok": True, "bt_id": bt_id})


@router.post("/api/rulesets/{rs_id}/train")
async def train_ruleset(rs_id: int, request: Request) -> JSONResponse:
    payload = await request.json()
    symbols_input = payload.get("symbols") or payload.get("symbol", "EURUSD")
    if isinstance(symbols_input, str):
        if symbols_input.lower() == "all":
            symbols = load_config().symbols
        else:
            symbols = [symbols_input]
    else:
        symbols = symbols_input
    timeframe = payload.get("timeframe", "4h")
    start_date_str = payload.get("start_date", "")
    end_date_str = payload.get("end_date", "")
    max_iterations = payload.get("max_iterations")
    initial_balance = payload.get("initial_balance")
    target_dd_pct = payload.get("target_dd_pct")
    target_profit_pct = payload.get("target_profit_pct")

    if not start_date_str or not end_date_str:
        return JSONResponse({"error": "start_date and end_date required"}, status_code=400)

    start_date = datetime.fromisoformat(start_date_str)
    end_date = datetime.fromisoformat(end_date_str)
    if start_date.tzinfo is None:
        start_date = start_date.replace(tzinfo=timezone.utc)
    if end_date.tzinfo is None:
        end_date = end_date.replace(tzinfo=timezone.utc)

    session = get_session()
    try:
        rs = session.query(RuleSet).filter(RuleSet.id == rs_id).first()
        if not rs:
            return JSONResponse({"error": "Not found"}, status_code=404)

        # Load candles for each symbol
        all_candle_dicts: dict[str, list[dict]] = {}
        for sym in symbols:
            candles = (
                session.query(Candle)
                .filter(Candle.symbol == sym, Candle.timeframe == timeframe,
                        Candle.ts >= start_date, Candle.ts <= end_date)
                .order_by(Candle.ts.asc()).all()
            )
            dicts = [
                {"time": c.ts.isoformat(), "open": c.open, "high": c.high,
                 "low": c.low, "close": c.close, "volume": c.volume or 0}
                for c in candles
            ]
            if len(dicts) >= 50:
                all_candle_dicts[sym] = dicts
    finally:
        session.close()

    if not all_candle_dicts:
        return JSONResponse({"error": "Not enough candle data for any selected symbol"}, status_code=400)
    valid_symbols = list(all_candle_dicts.keys())

    # Create training run
    now = datetime.now(timezone.utc)
    session = get_session()
    try:
        label = ", ".join(valid_symbols)
        run = TrainingRun(
            name=f"Ruleset '{rs.name}' training on {label}/{timeframe}"[:128],
            status="running",
            symbols=valid_symbols,
            timeframes=[timeframe],
            strategies=[f"ruleset_{rs_id}"],
            start_date=start_date,
            end_date=end_date,
            iterations_completed=0,
            config_snapshot=load_config().training.model_dump(),
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
        rule_engine._active_trains[run_id] = False
        try:
            # Train all symbols together — each iteration backtests all pairs
            iterations = await asyncio.to_thread(
                rule_engine.train_ruleset, rs_id, all_candle_dicts, timeframe,
                run_id, max_iterations, initial_balance, target_dd_pct, target_profit_pct,
            )
            # Generate final report
            final_report = None
            cfg = load_config()
            if iterations and ai_service.is_enabled(cfg):
                final_report = await asyncio.to_thread(
                    ai_service.generate_training_report, iterations,
                )
            sess = get_session()
            try:
                r = sess.query(TrainingRun).filter(TrainingRun.id == run_id).first()
                if r:
                    r.status = "completed"
                    r.completed_at = datetime.now(timezone.utc)
                    r.final_report = final_report
                sess.commit()
            except Exception:
                sess.rollback()
            finally:
                sess.close()
        except Exception as exc:
            import logging
            logging.getLogger(__name__).error("Ruleset training failed: %s", exc)
            sess = get_session()
            try:
                r = sess.query(TrainingRun).filter(TrainingRun.id == run_id).first()
                if r:
                    r.status = "failed"
                    r.completed_at = datetime.now(timezone.utc)
                sess.commit()
            except Exception:
                sess.rollback()
            finally:
                sess.close()
        finally:
            rule_engine._active_trains.pop(run_id, None)

    asyncio.create_task(_run())

    return JSONResponse({"ok": True, "run_id": run_id})


# ---------------------------------------------------------------------------
# Progress & stop endpoints
# ---------------------------------------------------------------------------


@router.get("/api/rulesets/backtest/{bt_id}/progress")
def get_backtest_progress(bt_id: int) -> JSONResponse:
    progress = rule_engine.get_backtest_progress(bt_id)
    if progress is None:
        return JSONResponse({"error": "Not found"}, status_code=404)
    return JSONResponse(progress)


@router.post("/api/rulesets/backtest/{bt_id}/stop")
def stop_backtest(bt_id: int) -> JSONResponse:
    rule_engine.request_stop_backtest(bt_id)
    return JSONResponse({"ok": True})


@router.get("/api/rulesets/training/{run_id}/progress")
def get_training_progress(run_id: int) -> JSONResponse:
    progress = rule_engine.get_training_progress(run_id)
    if progress is None:
        return JSONResponse({"error": "Not found"}, status_code=404)
    return JSONResponse(progress)


@router.get("/api/rulesets/training/active")
def get_active_training() -> JSONResponse:
    """Return any currently running training runs (for resuming UI on page reload)."""
    active = []
    for run_id, progress in rule_engine._training_progress.items():
        if progress.get("status") == "running":
            active.append(progress)
    return JSONResponse({"active": active})


@router.post("/api/rulesets/training/{run_id}/stop")
def stop_training(run_id: int) -> JSONResponse:
    rule_engine.request_stop(run_id)
    return JSONResponse({"ok": True})
