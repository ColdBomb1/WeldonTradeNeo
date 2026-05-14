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


def _validation_passed(rs: RuleSet) -> bool:
    params = rs.parameters or {}
    validation = params.get("validation") or {}
    if validation.get("passed") is True or validation.get("status") == "passed":
        return True
    metrics = rs.performance_metrics or {}
    return metrics.get("validated") is True or metrics.get("validation_status") == "passed"


def _validation_criteria(payload: dict, rs: RuleSet, initial_balance: float) -> dict:
    cfg = load_config()
    params = rs.parameters or {}
    training = params.get("training") or {}
    default_dd_pct = training.get("target_dd_pct")
    if default_dd_pct is None:
        default_dd_pct = max(
            0.1,
            cfg.execution.max_relative_drawdown_pct - cfg.execution.drawdown_hard_stop_buffer_pct,
        )
    return {
        "min_trades": int(payload.get("min_trades") or 20),
        "max_drawdown": float(payload.get("max_drawdown_pct") or default_dd_pct) / 100.0,
        "min_profit_factor": float(payload.get("min_profit_factor") or 1.15),
        "min_sharpe": float(payload.get("min_sharpe") or 0.0),
        "require_profit": bool(payload.get("require_profit", True)),
        "initial_balance": initial_balance,
    }


def _score_validation(metrics: dict, criteria: dict) -> dict:
    reasons: list[str] = []
    total_trades = int(metrics.get("total_trades") or 0)
    max_drawdown = float(metrics.get("max_drawdown") or 0)
    profit_factor = float(metrics.get("profit_factor") or 0)
    sharpe_ratio = float(metrics.get("sharpe_ratio") or 0)
    final_balance = float(metrics.get("final_balance") or 0)
    initial_balance = float(criteria.get("initial_balance") or 0)

    if total_trades < criteria["min_trades"]:
        reasons.append(f"Needs at least {criteria['min_trades']} trades; got {total_trades}.")
    if max_drawdown > criteria["max_drawdown"]:
        reasons.append(
            f"Max drawdown {(max_drawdown * 100):.2f}% exceeds "
            f"{(criteria['max_drawdown'] * 100):.2f}%."
        )
    if profit_factor < criteria["min_profit_factor"]:
        reasons.append(
            f"Profit factor {profit_factor:.2f} is below {criteria['min_profit_factor']:.2f}."
        )
    if sharpe_ratio < criteria["min_sharpe"]:
        reasons.append(f"Sharpe {sharpe_ratio:.2f} is below {criteria['min_sharpe']:.2f}.")
    if criteria["require_profit"] and final_balance <= initial_balance:
        reasons.append(
            f"Final balance ${final_balance:,.2f} did not exceed "
            f"${initial_balance:,.2f}."
        )

    return {
        "passed": not reasons,
        "status": "passed" if not reasons else "failed",
        "reasons": reasons,
        "criteria": criteria,
    }


def _mark_validation_stale(rs: RuleSet, reason: str) -> None:
    params = dict(rs.parameters or {})
    validation = dict(params.get("validation") or {})
    if validation:
        validation["status"] = "stale"
        validation["passed"] = False
        validation["stale_reason"] = reason
        validation["stale_at"] = datetime.now(timezone.utc).isoformat()
        params["validation"] = validation
        rs.parameters = params

    metrics = dict(rs.performance_metrics or {})
    if metrics.get("validation") or metrics.get("validated") is True:
        metrics["validated"] = False
        metrics["validation_status"] = "stale"
        rs.performance_metrics = metrics

    if rs.status == "validated":
        rs.status = "candidate"


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
        validation_sensitive_change = False
        if "name" in payload:
            rs.name = payload["name"][:128]
        if "description" in payload:
            rs.description = payload["description"][:512]
        if "rules_text" in payload:
            rs.rules_text = payload["rules_text"]
            rs.version += 1
            validation_sensitive_change = True
        if "symbols" in payload:
            rs.symbols = payload["symbols"]
            validation_sensitive_change = True
        if "timeframes" in payload:
            rs.timeframes = payload["timeframes"]
            validation_sensitive_change = True
        if validation_sensitive_change:
            _mark_validation_stale(rs, "Rules, symbols, or timeframes changed after validation.")
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
        if rs.status in {"candidate", "validated"}:
            return JSONResponse(
                {"error": "Candidate rulesets must pass validation and be promoted explicitly."},
                status_code=400,
            )
        rs.status = "active"
        rs.updated_at = datetime.now(timezone.utc)
        session.commit()
        return JSONResponse({"ok": True, "status": "active"})
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


@router.post("/api/rulesets/{rs_id}/promote")
def promote_ruleset(rs_id: int) -> JSONResponse:
    session = get_session()
    try:
        rs = session.query(RuleSet).filter(RuleSet.id == rs_id).first()
        if not rs:
            return JSONResponse({"error": "Not found"}, status_code=404)
        if not _validation_passed(rs):
            return JSONResponse(
                {"error": "Ruleset must pass a validation backtest before promotion."},
                status_code=400,
            )

        now = datetime.now(timezone.utc)
        for other in session.query(RuleSet).filter(RuleSet.status == "active").all():
            if other.id != rs.id:
                other.status = "inactive"
                other.updated_at = now

        params = dict(rs.parameters or {})
        params["promotion"] = {
            "promoted_at": now.isoformat(),
            "previous_status": rs.status,
            "deactivated_other_active_rulesets": True,
        }
        rs.parameters = params
        rs.status = "active"
        rs.updated_at = now
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
    purpose = payload.get("purpose", "backtest")
    if purpose not in {"backtest", "validation"}:
        purpose = "backtest"
    initial_balance = float(payload.get("initial_balance") or load_config().training.initial_balance)
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
        "symbols": valid_symbols, "purpose": purpose,
    }

    async def _run():
        try:
            # Run backtests across all symbols with cross-pair context
            combined_trades = []
            combined_equity = []
            per_symbol_full = {}
            per_symbol_results = {}

            for sym in valid_symbols:
                if rule_engine._active_backtests.get(bt_id):
                    break
                candle_dicts = all_pair_candles[sym]
                result = await asyncio.to_thread(
                    rule_engine.backtest_rules, rules_text, sym, timeframe,
                    candle_dicts, initial_balance=initial_balance, bt_id=bt_id, mode=mode,
                    all_pair_candles=all_pair_candles,
                )
                if rule_engine._active_backtests.get(bt_id):
                    break
                per_symbol_full[sym] = result
                per_symbol_results[sym] = {
                    k: result.get(k) for k in
                    ("final_balance", "total_trades", "win_rate",
                     "max_drawdown", "profit_factor", "sharpe_ratio")
                }
                combined_trades.extend(result.get("trades", []))
                combined_equity.extend(result.get("equity_curve", []))

            if not per_symbol_full:
                rule_engine._backtest_progress[bt_id] = {
                    **rule_engine._backtest_progress.get(bt_id, {}),
                    "status": "stopped",
                    "pct": 100,
                }
                rule_engine._active_backtests.pop(bt_id, None)
                return

            # Compute combined metrics
            total_t = len(combined_trades)
            wins = sum(1 for t in combined_trades if t.get("pnl", 0) > 0)
            gross_p = sum(t["pnl"] for t in combined_trades if t.get("pnl", 0) > 0)
            gross_l = abs(sum(t["pnl"] for t in combined_trades if t.get("pnl", 0) <= 0))
            pf = gross_p / gross_l if gross_l > 0 else (999.0 if gross_p > 0 else 0)
            sharpes = [m.get("sharpe_ratio", 0) or 0 for m in per_symbol_full.values()]
            drawdowns = [m.get("max_drawdown", 0) or 0 for m in per_symbol_full.values()]
            final_balances = [m.get("final_balance", initial_balance) for m in per_symbol_full.values()]

            combined_result = {
                "total_trades": total_t,
                "winning_trades": wins,
                "losing_trades": total_t - wins,
                "win_rate": round(wins / total_t, 4) if total_t > 0 else 0,
                "profit_factor": round(min(pf, 999.0), 2),
                "sharpe_ratio": round(sum(sharpes) / len(sharpes), 2) if sharpes else 0,
                "max_drawdown": round(max(drawdowns), 4) if drawdowns else 0,
                "final_balance": round(sum(final_balances) / len(final_balances), 2),
                "initial_balance": initial_balance,
                "symbols": list(per_symbol_full.keys()),
                "timeframe": timeframe,
                "purpose": purpose,
                "per_symbol": per_symbol_results,
            }
            validation_result = None
            if purpose == "validation":
                criteria = _validation_criteria(payload, rs, initial_balance)
                metrics_snapshot = dict(combined_result)
                validation_result = {
                    **_score_validation(combined_result, criteria),
                    "validated_at": datetime.now(timezone.utc).isoformat(),
                    "bt_id": bt_id,
                    "symbols": list(per_symbol_full.keys()),
                    "timeframe": timeframe,
                    "start_date": start_date.isoformat(),
                    "end_date": end_date.isoformat(),
                    "metrics": metrics_snapshot,
                }
                combined_result["validated"] = validation_result["passed"]
                combined_result["validation_status"] = validation_result["status"]
                combined_result["validation_reasons"] = validation_result["reasons"]
                combined_result["validation"] = validation_result
        except asyncio.CancelledError:
            return
        except Exception as exc:
            import logging
            logging.getLogger(__name__).error("Backtest error: %s", exc)
            rule_engine._backtest_progress[bt_id] = {
                **rule_engine._backtest_progress.get(bt_id, {}),
                "status": "failed",
                "error": str(exc),
                "pct": 100,
            }
            rule_engine._active_backtests.pop(bt_id, None)
            return
        # Save combined metrics to ruleset
        sess = get_session()
        try:
            r = sess.query(RuleSet).filter(RuleSet.id == rs_id).first()
            if r:
                existing_metrics = dict(r.performance_metrics or {})
                if validation_result:
                    params = dict(r.parameters or {})
                    params["validation"] = validation_result
                    r.parameters = params
                    r.performance_metrics = combined_result
                    if r.status in {"candidate", "validated"}:
                        r.status = "validated" if validation_result["passed"] else "candidate"
                else:
                    preserved = {}
                    if "validation" in existing_metrics:
                        preserved["validation"] = existing_metrics["validation"]
                    for key in ("validated", "validation_status"):
                        if key in existing_metrics:
                            preserved[key] = existing_metrics[key]
                    r.performance_metrics = {**combined_result, **preserved}
                r.updated_at = datetime.now(timezone.utc)
                sess.commit()
        except Exception:
            sess.rollback()
        finally:
            sess.close()

        rule_engine._backtest_progress[bt_id] = {
            **rule_engine._backtest_progress.get(bt_id, {}),
            "bt_id": bt_id,
            "status": "completed",
            "bar": total_bars,
            "total_bars": total_bars,
            "pct": 100,
            "balance": combined_result["final_balance"],
            "equity": combined_result["final_balance"],
            "trades": combined_result["total_trades"],
            "wins": combined_result["winning_trades"],
            "max_drawdown": combined_result["max_drawdown"],
            "equity_curve": combined_equity[-200:],
            "results": combined_result,
        }
        rule_engine._active_backtests.pop(bt_id, None)

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
            progress = rule_engine.get_training_progress(run_id) or {}
            candidate_id = progress.get("candidate_ruleset_id")
            if candidate_id:
                if isinstance(final_report, dict):
                    final_report = {**final_report, "candidate_ruleset_id": candidate_id}
                else:
                    final_report = {"candidate_ruleset_id": candidate_id}
            sess = get_session()
            try:
                r = sess.query(TrainingRun).filter(TrainingRun.id == run_id).first()
                if r:
                    r.status = progress.get("status") if progress.get("status") in {"completed", "stopped"} else "completed"
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
