from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from fastapi.templating import Jinja2Templates

from config import load_config
from db import get_session
from models.candle import Candle
from models.trade_plan import BacktestRun, TradePlan
from services.backtest_engine import BacktestConfig, run_backtest
from services.ai_service import compute_price_summary
from services.optimizer import (
    DEFAULT_SEARCH_SPACES,
    OptimizationConfig,
    ParamRange,
    format_results_for_analysis,
    request_cancel,
    reset_cancel,
    run_optimization,
)
from services.strategy_service import list_strategies

router = APIRouter(tags=["trade-plans"])

BASE_DIR = Path(__file__).resolve().parent.parent
TEMPLATES = Jinja2Templates(directory=str(BASE_DIR / "templates"))


@router.get("/trade-plans")
def trade_plans_page(request: Request):
    cfg = load_config()
    session = get_session()
    plans = session.query(TradePlan).order_by(TradePlan.created_at.desc()).all()
    session.close()
    return TEMPLATES.TemplateResponse(
        "trade_plans.html",
        {
            "request": request,
            "plans": plans,
            "strategies": list_strategies(),
            "symbols": cfg.symbols,
            "timeframes": cfg.candle_timeframes,
        },
    )


@router.get("/api/trade-plans")
def api_list_trade_plans() -> JSONResponse:
    session = get_session()
    plans = session.query(TradePlan).order_by(TradePlan.created_at.desc()).all()
    session.close()
    return JSONResponse({
        "plans": [
            {
                "id": p.id,
                "name": p.name,
                "symbol": p.symbol,
                "strategy_type": p.strategy_type,
                "parameters": p.parameters,
                "status": p.status,
                "created_at": p.created_at.isoformat(),
            }
            for p in plans
        ]
    })


@router.post("/api/trade-plans")
async def create_trade_plan(request: Request) -> JSONResponse:
    payload = await request.json()
    now = datetime.now(tz=timezone.utc)

    session = get_session()
    plan = TradePlan(
        name=payload.get("name", "Untitled"),
        symbol=payload.get("symbol", "EURUSD"),
        strategy_type=payload.get("strategy_type", "sma_crossover"),
        parameters=payload.get("parameters", {}),
        status="draft",
        created_at=now,
        updated_at=now,
    )
    session.add(plan)
    session.commit()
    plan_id = plan.id
    session.close()

    return JSONResponse({"ok": True, "id": plan_id}, status_code=201)


@router.get("/api/trade-plans/{plan_id}")
def get_trade_plan(plan_id: int) -> JSONResponse:
    session = get_session()
    plan = session.query(TradePlan).filter(TradePlan.id == plan_id).first()
    if not plan:
        session.close()
        return JSONResponse({"error": "Trade plan not found"}, status_code=404)

    backtests = (
        session.query(BacktestRun)
        .filter(BacktestRun.trade_plan_id == plan_id)
        .order_by(BacktestRun.created_at.desc())
        .all()
    )
    session.close()

    return JSONResponse({
        "plan": {
            "id": plan.id,
            "name": plan.name,
            "symbol": plan.symbol,
            "strategy_type": plan.strategy_type,
            "parameters": plan.parameters,
            "status": plan.status,
            "created_at": plan.created_at.isoformat(),
            "updated_at": plan.updated_at.isoformat(),
        },
        "backtests": [
            {
                "id": b.id,
                "timeframe": b.timeframe,
                "start_date": b.start_date.isoformat(),
                "end_date": b.end_date.isoformat(),
                "initial_balance": b.initial_balance,
                "final_balance": b.final_balance,
                "total_trades": b.total_trades,
                "win_rate": b.win_rate,
                "max_drawdown": b.max_drawdown,
                "profit_factor": b.profit_factor,
                "sharpe_ratio": b.sharpe_ratio,
                "status": b.status,
                "created_at": b.created_at.isoformat(),
            }
            for b in backtests
        ],
    })


@router.put("/api/trade-plans/{plan_id}")
async def update_trade_plan(plan_id: int, request: Request) -> JSONResponse:
    payload = await request.json()
    session = get_session()
    plan = session.query(TradePlan).filter(TradePlan.id == plan_id).first()
    if not plan:
        session.close()
        return JSONResponse({"error": "Trade plan not found"}, status_code=404)

    if "name" in payload:
        plan.name = payload["name"]
    if "symbol" in payload:
        plan.symbol = payload["symbol"]
    if "strategy_type" in payload:
        plan.strategy_type = payload["strategy_type"]
    if "parameters" in payload:
        plan.parameters = payload["parameters"]
    if "status" in payload:
        plan.status = payload["status"]
    plan.updated_at = datetime.now(tz=timezone.utc)

    session.commit()
    session.close()
    return JSONResponse({"ok": True})


@router.delete("/api/trade-plans/{plan_id}")
def delete_trade_plan(plan_id: int) -> JSONResponse:
    session = get_session()
    plan = session.query(TradePlan).filter(TradePlan.id == plan_id).first()
    if not plan:
        session.close()
        return JSONResponse({"error": "Trade plan not found"}, status_code=404)

    session.query(BacktestRun).filter(BacktestRun.trade_plan_id == plan_id).delete()
    session.delete(plan)
    session.commit()
    session.close()
    return JSONResponse({"ok": True})


@router.post("/api/trade-plans/{plan_id}/backtest")
async def run_backtest_for_plan(plan_id: int, request: Request) -> JSONResponse:
    payload = await request.json()
    session = get_session()
    plan = session.query(TradePlan).filter(TradePlan.id == plan_id).first()
    if not plan:
        session.close()
        return JSONResponse({"error": "Trade plan not found"}, status_code=404)

    timeframe = payload.get("timeframe", "1h")
    start_date = datetime.fromisoformat(
        payload.get("start_date", "2024-01-01T00:00:00+00:00"))
    end_date = datetime.fromisoformat(
        payload.get("end_date", datetime.now(tz=timezone.utc).isoformat()))
    initial_balance = float(payload.get("initial_balance", 10000.0))

    # Risk management fields
    lot_type = payload.get("lot_type", "mini")
    risk_per_trade_pct = float(payload.get("risk_per_trade_pct", 1.0))
    sl_atr_multiplier = float(payload.get("sl_atr_multiplier", 1.5))
    tp_atr_multiplier = float(payload.get("tp_atr_multiplier", 2.0))
    monthly_max_loss_pct = float(payload.get("monthly_max_loss_pct", 7.0))

    if start_date.tzinfo is None:
        start_date = start_date.replace(tzinfo=timezone.utc)
    if end_date.tzinfo is None:
        end_date = end_date.replace(tzinfo=timezone.utc)

    # Create backtest run record
    now = datetime.now(tz=timezone.utc)
    bt_run = BacktestRun(
        trade_plan_id=plan_id,
        timeframe=timeframe,
        start_date=start_date,
        end_date=end_date,
        initial_balance=initial_balance,
        status="running",
        created_at=now,
    )
    session.add(bt_run)
    session.commit()
    run_id = bt_run.id

    # Load candles
    candles_rows = (
        session.query(Candle)
        .filter(
            Candle.symbol == plan.symbol,
            Candle.timeframe == timeframe,
            Candle.ts >= start_date,
            Candle.ts <= end_date,
        )
        .order_by(Candle.ts.asc())
        .all()
    )

    if len(candles_rows) < 2:
        bt_run.status = "failed"
        bt_run.completed_at = datetime.now(tz=timezone.utc)
        session.commit()
        session.close()
        return JSONResponse(
            {"error": "Not enough candle data for backtest",
             "run_id": run_id}, status_code=400)

    candles = [
        {
            "time": row.ts.isoformat(),
            "open": row.open,
            "high": row.high,
            "low": row.low,
            "close": row.close,
            "volume": row.volume,
        }
        for row in candles_rows
    ]

    pip_value = 0.01 if "JPY" in plan.symbol.upper() else 0.0001

    config = BacktestConfig(
        symbol=plan.symbol,
        timeframe=timeframe,
        strategy_type=plan.strategy_type,
        parameters=plan.parameters,
        start_date=start_date,
        end_date=end_date,
        initial_balance=initial_balance,
        pip_value=pip_value,
        lot_type=lot_type,
        risk_per_trade_pct=risk_per_trade_pct,
        sl_atr_multiplier=sl_atr_multiplier,
        tp_atr_multiplier=tp_atr_multiplier,
        monthly_max_loss_pct=monthly_max_loss_pct,
    )

    try:
        results = run_backtest(config, candles)
    except Exception as exc:
        bt_run.status = "failed"
        bt_run.completed_at = datetime.now(tz=timezone.utc)
        session.commit()
        session.close()
        return JSONResponse(
            {"error": str(exc), "run_id": run_id}, status_code=500)

    bt_run.final_balance = results.final_balance
    bt_run.total_trades = results.total_trades
    bt_run.win_rate = results.win_rate
    bt_run.max_drawdown = results.max_drawdown
    bt_run.profit_factor = results.profit_factor
    bt_run.sharpe_ratio = results.sharpe_ratio
    bt_run.equity_curve = results.equity_curve
    bt_run.trades = results.trades
    bt_run.status = "completed"
    bt_run.completed_at = datetime.now(tz=timezone.utc)
    session.commit()
    session.close()

    return JSONResponse({
        "ok": True,
        "run_id": run_id,
        "results": {
            "final_balance": results.final_balance,
            "total_trades": results.total_trades,
            "winning_trades": results.winning_trades,
            "losing_trades": results.losing_trades,
            "win_rate": results.win_rate,
            "max_drawdown": results.max_drawdown,
            "profit_factor": results.profit_factor,
            "sharpe_ratio": results.sharpe_ratio,
            "monthly_pnl": results.monthly_pnl,
            "avg_monthly_return": results.avg_monthly_return,
            "max_monthly_drawdown": results.max_monthly_drawdown,
            "months_above_target": results.months_above_target,
        },
    })


@router.get("/api/trade-plans/{plan_id}/backtests/{run_id}")
def get_backtest_result(plan_id: int, run_id: int) -> JSONResponse:
    session = get_session()
    bt = (
        session.query(BacktestRun)
        .filter(BacktestRun.id == run_id, BacktestRun.trade_plan_id == plan_id)
        .first()
    )
    if not bt:
        session.close()
        return JSONResponse({"error": "Backtest run not found"}, status_code=404)

    session.close()
    return JSONResponse({
        "id": bt.id,
        "trade_plan_id": bt.trade_plan_id,
        "timeframe": bt.timeframe,
        "start_date": bt.start_date.isoformat(),
        "end_date": bt.end_date.isoformat(),
        "initial_balance": bt.initial_balance,
        "final_balance": bt.final_balance,
        "total_trades": bt.total_trades,
        "win_rate": bt.win_rate,
        "max_drawdown": bt.max_drawdown,
        "profit_factor": bt.profit_factor,
        "sharpe_ratio": bt.sharpe_ratio,
        "equity_curve": bt.equity_curve,
        "trades": bt.trades,
        "status": bt.status,
        "created_at": bt.created_at.isoformat(),
        "completed_at": bt.completed_at.isoformat() if bt.completed_at else None,
    })


@router.get("/api/strategies")
def api_list_strategies() -> JSONResponse:
    return JSONResponse({"strategies": list_strategies()})


# ---------- Optimizer endpoints ----------


@router.get("/api/optimizer/search-spaces")
def get_search_spaces() -> JSONResponse:
    """Return default parameter search spaces for each strategy."""
    return JSONResponse({
        name: [{"name": pr.name, "values": pr.values} for pr in ranges]
        for name, ranges in DEFAULT_SEARCH_SPACES.items()
    })


@router.post("/api/optimizer/run")
async def run_optimizer(request: Request) -> JSONResponse:
    """Run parameter optimization for a strategy.

    Body: {
        symbol, timeframe, strategy_type,
        start_date, end_date, initial_balance,
        rank_by (optional), top_n (optional),
        param_ranges (optional, overrides defaults)
    }
    """
    payload = await request.json()
    symbol = payload.get("symbol", "EURUSD")
    timeframe = payload.get("timeframe", "1h")
    strategy_type = payload.get("strategy_type", "sma_crossover")
    start_date = datetime.fromisoformat(
        payload.get("start_date", "2024-01-01T00:00:00+00:00"))
    end_date = datetime.fromisoformat(
        payload.get("end_date", datetime.now(tz=timezone.utc).isoformat()))
    initial_balance = float(payload.get("initial_balance", 10000.0))
    rank_by = payload.get("rank_by", "sharpe_ratio")
    top_n = int(payload.get("top_n", 10))

    # Risk management fields
    lot_type = payload.get("lot_type", "mini")
    risk_per_trade_pct = float(payload.get("risk_per_trade_pct", 1.0))
    sl_atr_multiplier = float(payload.get("sl_atr_multiplier", 1.5))
    tp_atr_multiplier = float(payload.get("tp_atr_multiplier", 2.0))
    monthly_max_loss_pct = float(payload.get("monthly_max_loss_pct", 7.0))

    if start_date.tzinfo is None:
        start_date = start_date.replace(tzinfo=timezone.utc)
    if end_date.tzinfo is None:
        end_date = end_date.replace(tzinfo=timezone.utc)

    # Build param ranges
    custom_ranges = payload.get("param_ranges")
    if custom_ranges:
        param_ranges = [ParamRange(name=pr["name"], values=pr["values"]) for pr in custom_ranges]
    else:
        param_ranges = DEFAULT_SEARCH_SPACES.get(strategy_type, [])

    if not param_ranges:
        return JSONResponse({"error": f"No search space defined for strategy '{strategy_type}'"}, status_code=400)

    # Load candles
    session = get_session()
    candle_rows = (
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
    session.close()

    if len(candle_rows) < 50:
        return JSONResponse({
            "error": f"Not enough candle data ({len(candle_rows)} candles). Need at least 50 for meaningful optimization.",
        }, status_code=400)

    candles = [
        {
            "time": row.ts.isoformat(),
            "open": row.open,
            "high": row.high,
            "low": row.low,
            "close": row.close,
            "volume": row.volume,
        }
        for row in candle_rows
    ]

    opt_config = OptimizationConfig(
        symbol=symbol,
        timeframe=timeframe,
        strategy_type=strategy_type,
        param_ranges=param_ranges,
        start_date=start_date,
        end_date=end_date,
        initial_balance=initial_balance,
        rank_by=rank_by,
        lot_type=lot_type,
        risk_per_trade_pct=risk_per_trade_pct,
        sl_atr_multiplier=sl_atr_multiplier,
        tp_atr_multiplier=tp_atr_multiplier,
        monthly_max_loss_pct=monthly_max_loss_pct,
    )

    try:
        results = run_optimization(opt_config, candles, top_n=top_n)
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=500)

    analysis_text = format_results_for_analysis(
        opt_config, results, len(candles))

    return JSONResponse({
        "ok": True,
        "symbol": symbol,
        "timeframe": timeframe,
        "strategy_type": strategy_type,
        "candle_count": len(candles),
        "combinations_tested": len(param_ranges),
        "results": [
            {
                "rank": i + 1,
                "parameters": r.parameters,
                "final_balance": r.final_balance,
                "total_trades": r.total_trades,
                "win_rate": r.win_rate,
                "max_drawdown": r.max_drawdown,
                "profit_factor": r.profit_factor,
                "sharpe_ratio": r.sharpe_ratio,
                "score": r.score,
                "avg_monthly_return": r.avg_monthly_return,
                "max_monthly_drawdown": r.max_monthly_drawdown,
                "months_above_target": r.months_above_target,
            }
            for i, r in enumerate(results)
        ],
        "analysis_markdown": analysis_text,
    })


@router.post("/api/optimizer/stop")
async def stop_optimizer() -> JSONResponse:
    """Signal any running optimizer to stop after its current iteration."""
    request_cancel()
    return JSONResponse({"ok": True, "message": "Stop signal sent."})


# ---------- Claude API workflow ----------


@router.post("/api/claude/analyze")
async def claude_analyze_results(request: Request) -> JSONResponse:
    """Send optimization results to Claude for analysis via API.

    Body: { symbol, timeframe, results_markdown }
    """
    from services import ai_service

    payload = await request.json()
    symbol = payload.get("symbol", "EURUSD")
    timeframe = payload.get("timeframe", "1h")
    results_markdown = payload.get("results_markdown", "")

    if not results_markdown:
        return JSONResponse({"error": "results_markdown is required"}, status_code=400)

    # Count candles for context
    session = get_session()
    candle_count = session.query(Candle).filter(
        Candle.symbol == symbol, Candle.timeframe == timeframe,
    ).count()
    session.close()

    analysis = ai_service.analyze_optimization_results(
        symbol=symbol,
        timeframe=timeframe,
        candle_count=candle_count,
        results_markdown=results_markdown,
    )

    if "error" in analysis and "raw" not in analysis:
        return JSONResponse(analysis, status_code=500)

    return JSONResponse({"ok": True, "analysis": analysis})


@router.post("/api/claude/run-suggestions")
async def run_claude_suggestions(request: Request) -> JSONResponse:
    """Parse Claude's response, run optimizer for each suggestion, return results.

    Body: {
        claude_response: str,   -- raw JSON array of strategy suggestions
        symbol: str,
        timeframe: str,
        start_date: str,
        end_date: str,
        initial_balance: float (optional),
        rank_by: str (optional),
    }
    """
    from services.ai_service import parse_json_response
    from services.strategy_service import STRATEGIES

    payload = await request.json()
    claude_response = payload.get("claude_response", "")
    symbol = payload.get("symbol", "EURUSD")
    timeframe = payload.get("timeframe", "1h")
    start_date = datetime.fromisoformat(
        payload.get("start_date", "2024-01-01T00:00:00+00:00"))
    end_date = datetime.fromisoformat(
        payload.get("end_date", datetime.now(tz=timezone.utc).isoformat()))
    initial_balance = float(payload.get("initial_balance", 10000.0))
    rank_by = payload.get("rank_by", "sharpe_ratio")

    # Risk management fields
    lot_type = payload.get("lot_type", "mini")
    risk_per_trade_pct = float(payload.get("risk_per_trade_pct", 1.0))
    sl_atr_multiplier = float(payload.get("sl_atr_multiplier", 1.5))
    tp_atr_multiplier = float(payload.get("tp_atr_multiplier", 2.0))
    monthly_max_loss_pct = float(payload.get("monthly_max_loss_pct", 7.0))

    if start_date.tzinfo is None:
        start_date = start_date.replace(tzinfo=timezone.utc)
    if end_date.tzinfo is None:
        end_date = end_date.replace(tzinfo=timezone.utc)

    # Parse suggestions from Claude response
    parsed = parse_json_response(claude_response)
    if parsed is None:
        return JSONResponse({"error": "Could not parse strategy suggestions from response."}, status_code=400)

    if isinstance(parsed, dict):
        parsed = [parsed]

    suggestions = []
    for item in parsed:
        st = item.get("strategy_type", "")
        if st not in STRATEGIES:
            continue
        suggestions.append({
            "strategy_type": st,
            "parameters": item.get("parameters", {}),
            "rationale": item.get("rationale", ""),
        })

    if not suggestions:
        return JSONResponse({"error": "No valid strategy suggestions found."}, status_code=400)

    # Load candles once
    session = get_session()
    candle_rows = (
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
    session.close()

    if len(candle_rows) < 50:
        return JSONResponse({
            "error": f"Not enough candle data ({len(candle_rows)} candles). Need at least 50.",
        }, status_code=400)

    candles = [
        {"time": r.ts.isoformat(), "open": r.open, "high": r.high,
         "low": r.low, "close": r.close, "volume": r.volume}
        for r in candle_rows
    ]

    from services.optimizer import is_cancelled

    all_results = []
    for suggestion in suggestions:
        if is_cancelled():
            break

        param_ranges = _build_ranges_from_suggestion_dict(suggestion)

        if not param_ranges:
            all_results.append({
                "strategy_type": suggestion["strategy_type"],
                "rationale": suggestion["rationale"],
                "base_parameters": suggestion["parameters"],
                "top_results": [],
                "error": "Could not build parameter ranges",
            })
            continue

        opt_config = OptimizationConfig(
            symbol=symbol,
            timeframe=timeframe,
            strategy_type=suggestion["strategy_type"],
            param_ranges=param_ranges,
            start_date=start_date,
            end_date=end_date,
            initial_balance=initial_balance,
            rank_by=rank_by,
            lot_type=lot_type,
            risk_per_trade_pct=risk_per_trade_pct,
            sl_atr_multiplier=sl_atr_multiplier,
            tp_atr_multiplier=tp_atr_multiplier,
            monthly_max_loss_pct=monthly_max_loss_pct,
        )

        try:
            results = run_optimization(opt_config, candles, top_n=5)
        except Exception as exc:
            all_results.append({
                "strategy_type": suggestion["strategy_type"],
                "rationale": suggestion["rationale"],
                "base_parameters": suggestion["parameters"],
                "top_results": [],
                "error": str(exc),
            })
            continue

        all_results.append({
            "strategy_type": suggestion["strategy_type"],
            "rationale": suggestion["rationale"],
            "base_parameters": suggestion["parameters"],
            "combinations_tested": len(param_ranges),
            "top_results": [
                {
                    "rank": i + 1,
                    "parameters": r.parameters,
                    "final_balance": r.final_balance,
                    "total_trades": r.total_trades,
                    "win_rate": r.win_rate,
                    "max_drawdown": r.max_drawdown,
                    "profit_factor": r.profit_factor,
                    "sharpe_ratio": r.sharpe_ratio,
                    "avg_monthly_return": r.avg_monthly_return,
                    "max_monthly_drawdown": r.max_monthly_drawdown,
                    "months_above_target": r.months_above_target,
                }
                for i, r in enumerate(results)
            ],
        })

    return JSONResponse({
        "ok": True,
        "suggestions_parsed": len(suggestions),
        "results": all_results,
    })


def _build_ranges_from_suggestion_dict(suggestion: dict) -> list[ParamRange]:
    """Build optimizer parameter ranges around a suggestion dict.

    Uses 3 values per parameter to keep combos manageable:
      - int params:   [value-1, value, value+1]
      - float params: [value*0.9, value, value*1.1]
    """
    from services.strategy_service import STRATEGIES

    strategy = STRATEGIES.get(suggestion["strategy_type"])
    if not strategy:
        return []

    param_meta = {p["name"]: p for p in strategy.required_params()}
    ranges = []

    for name, value in suggestion["parameters"].items():
        meta = param_meta.get(name)
        if meta is None:
            continue

        if meta["type"] == "float":
            fv = float(value)
            values = sorted(set([
                round(fv * 0.9, 1),
                round(fv, 1),
                round(fv * 1.1, 1),
            ]))
        else:
            iv = int(value)
            values = sorted(set([
                max(1, iv - 1),
                iv,
                iv + 1,
            ]))

        ranges.append(ParamRange(name=name, values=values))

    return ranges
