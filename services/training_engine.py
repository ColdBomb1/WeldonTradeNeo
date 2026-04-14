"""Autonomous training engine: replay historical data, iterate with Claude feedback.

Runs backtest → evaluate → Claude suggests adjustments → re-run → compare → repeat
until convergence or max iterations. Produces a deployment-ready configuration.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone

from config import load_config
from db import get_session
from models.candle import Candle
from models.training import TrainingRun, TrainingIteration
from services.backtest_engine import BacktestConfig, run_backtest
from services.strategy_service import STRATEGIES
from services import claude_service

logger = logging.getLogger(__name__)

# Active training runs that can be cancelled
_active_runs: dict[int, bool] = {}
_global_stop = False


def request_stop(run_id: int):
    _active_runs[run_id] = True


def stop_all():
    """Stop all running training runs (called on app shutdown)."""
    global _global_stop
    _global_stop = True
    for run_id in list(_active_runs):
        _active_runs[run_id] = True


def _is_stopped(run_id: int) -> bool:
    return _global_stop or _active_runs.get(run_id, False)


def _heuristic_adjust(strategy, current_params: dict, metrics: dict) -> dict | None:
    """Generate parameter adjustments without Claude, based on performance metrics.

    Returns adjusted params dict, or None if no more adjustments to try.
    """
    required = strategy.required_params()
    if not required:
        return None

    new_params = dict(current_params)
    win_rate = metrics.get("win_rate", 0) or 0
    profit_factor = metrics.get("profit_factor", 0) or 0
    max_dd = metrics.get("max_drawdown", 0) or 0

    # Heuristic: adjust period-type params based on performance
    # Low win rate → try longer periods (more filtering)
    # High drawdown → try longer periods (fewer trades)
    # Good win rate but low PF → try shorter periods (more trades)
    adjusted = False
    for p in required:
        name = p["name"]
        val = current_params.get(name, p["default"])

        if p["type"] == "int" and "period" in name.lower():
            if win_rate < 0.45 or max_dd > 0.15:
                # Increase periods to filter more
                new_val = int(val * 1.2) + 1
                if new_val != val:
                    new_params[name] = new_val
                    adjusted = True
            elif win_rate > 0.55 and profit_factor < 1.3:
                # Decrease periods to capture more trades
                new_val = max(2, int(val * 0.85))
                if new_val != val:
                    new_params[name] = new_val
                    adjusted = True

        elif p["type"] == "float" and ("level" in name.lower() or "zone" in name.lower()):
            if win_rate < 0.45:
                # Tighten threshold levels
                if "oversold" in name.lower() or "buy" in name.lower():
                    new_params[name] = round(val - 3, 1)
                    adjusted = True
                elif "overbought" in name.lower() or "sell" in name.lower():
                    new_params[name] = round(val + 3, 1)
                    adjusted = True

    return new_params if adjusted else None


def _load_candles(symbol: str, timeframe: str, start_date: datetime, end_date: datetime) -> list[dict]:
    """Load candles from database for the replay period."""
    session = get_session()
    try:
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
        return [
            {
                "time": row.ts.isoformat(),
                "open": row.open,
                "high": row.high,
                "low": row.low,
                "close": row.close,
                "volume": row.volume or 0,
            }
            for row in rows
        ]
    finally:
        session.close()


def _build_trade_summary(trades: list[dict], max_trades: int = 20) -> str:
    """Format trade list into a concise summary for Claude."""
    if not trades:
        return "No trades executed."

    total = len(trades)
    wins = sum(1 for t in trades if t.get("pnl", 0) > 0)
    losses = total - wins

    # Summarize exit types
    exit_types = {}
    for t in trades:
        et = t.get("exit_type", "unknown")
        exit_types[et] = exit_types.get(et, 0) + 1

    exit_text = ", ".join(f"{k}: {v}" for k, v in sorted(exit_types.items()))

    lines = [
        f"Total: {total} trades ({wins} wins, {losses} losses)",
        f"Exit types: {exit_text}",
        "",
        "Sample trades (last {n}):".format(n=min(max_trades, total)),
    ]

    for t in trades[-max_trades:]:
        pnl = t.get("pnl", 0)
        lines.append(
            "  {side} @ {entry:.5f} → {exit:.5f} | "
            "SL={sl:.5f} TP={tp:.5f} | "
            "P&L=${pnl:+.2f} ({exit_type})".format(
                side=t.get("side", "?"),
                entry=t.get("entry_price", 0),
                exit=t.get("exit_price", 0),
                sl=t.get("stop_loss", 0),
                tp=t.get("take_profit", 0),
                pnl=pnl,
                exit_type=t.get("exit_type", "?"),
            )
        )

    return "\n".join(lines)


def _get_metric(results, metric_name: str) -> float:
    """Safely extract a metric from BacktestResults."""
    return getattr(results, metric_name, 0) or 0


def _results_to_metrics(results) -> dict:
    """Convert BacktestResults to a flat metrics dict."""
    return {
        "final_balance": results.final_balance,
        "total_trades": results.total_trades,
        "winning_trades": results.winning_trades,
        "losing_trades": results.losing_trades,
        "win_rate": results.win_rate,
        "max_drawdown": results.max_drawdown,
        "profit_factor": results.profit_factor,
        "sharpe_ratio": results.sharpe_ratio,
        "avg_monthly_return": results.avg_monthly_return,
        "max_monthly_drawdown": results.max_monthly_drawdown,
    }


def _compute_improvement(current: dict, previous: dict, rank_by: str) -> float:
    """Compute improvement percentage between two iterations."""
    curr_val = current.get(rank_by, 0) or 0
    prev_val = previous.get(rank_by, 0) or 0
    if prev_val == 0:
        return 1.0 if curr_val > 0 else 0.0
    return (curr_val - prev_val) / abs(prev_val)


def run_training_for_combo(
    run_id: int,
    symbol: str,
    timeframe: str,
    strategy_type: str,
    candles: list[dict],
    start_date: datetime,
    end_date: datetime,
) -> list[dict]:
    """Run the iterative training loop for one symbol/timeframe/strategy combo.

    Returns list of iteration result dicts.
    """
    cfg = load_config()
    tc = cfg.training
    strategy = STRATEGIES.get(strategy_type)
    if strategy is None:
        logger.error("Unknown strategy: %s", strategy_type)
        return []

    if len(candles) < 50:
        logger.warning("Not enough candles for %s/%s (%d)", symbol, timeframe, len(candles))
        return []

    pip_value = 0.01 if "JPY" in symbol.upper() else 0.0001
    params = strategy.default_params()
    prev_metrics = None
    prev_params = None
    iterations = []
    claude_available = True  # disable after first API failure

    for iteration in range(1, tc.max_iterations + 1):
        if _is_stopped(run_id):
            logger.info("Training run %d stopped by user", run_id)
            break

        logger.info(
            "Training run %d: %s %s/%s iteration %d params=%s",
            run_id, strategy_type, symbol, timeframe, iteration, params,
        )

        # Run backtest
        bt_config = BacktestConfig(
            symbol=symbol,
            timeframe=timeframe,
            strategy_type=strategy_type,
            parameters=params,
            start_date=start_date,
            end_date=end_date,
            initial_balance=tc.initial_balance,
            pip_value=pip_value,
            lot_type=cfg.execution.default_lot_type,
            risk_per_trade_pct=cfg.execution.risk_per_trade_pct,
            sl_atr_multiplier=cfg.execution.sl_atr_multiplier,
            tp_atr_multiplier=cfg.execution.tp_atr_multiplier,
        )

        try:
            results = run_backtest(bt_config, candles)
        except Exception as exc:
            logger.error("Backtest failed: %s", exc)
            break

        metrics = _results_to_metrics(results)

        # Compute improvement
        improvement = None
        if prev_metrics is not None:
            improvement = _compute_improvement(metrics, prev_metrics, tc.rank_by)

        # Build trade summary for Claude
        trade_summary = _build_trade_summary(results.trades)

        # Claude analysis (if enabled)
        claude_analysis = None
        suggested_params = None

        if tc.use_claude_evaluation and cfg.claude.enabled and claude_available:
            claude_analysis = claude_service.analyze_training_iteration(
                strategy_type=strategy_type,
                parameters=params,
                metrics=metrics,
                trade_summary=trade_summary,
                candle_count=len(candles),
                iteration_number=iteration,
                previous_metrics=prev_metrics,
                previous_params=prev_params,
            )

            if "error" in claude_analysis:
                logger.warning("Claude unavailable, switching to heuristic mode: %s",
                               claude_analysis["error"])
                claude_available = False
            elif "adjusted_parameters" in claude_analysis:
                suggested_params = claude_analysis["adjusted_parameters"]

        # Store iteration in DB
        session = get_session()
        try:
            iter_record = TrainingIteration(
                training_run_id=run_id,
                iteration_number=iteration,
                symbol=symbol,
                timeframe=timeframe,
                strategy_type=strategy_type,
                parameters=params,
                final_balance=results.final_balance,
                total_trades=results.total_trades,
                win_rate=results.win_rate,
                max_drawdown=results.max_drawdown,
                profit_factor=results.profit_factor,
                sharpe_ratio=results.sharpe_ratio,
                avg_monthly_return=results.avg_monthly_return,
                equity_curve=results.equity_curve,
                trades=results.trades,
                claude_analysis=claude_analysis,
                suggested_params=suggested_params,
                improvement_pct=improvement,
                created_at=datetime.now(timezone.utc),
            )
            session.add(iter_record)

            # Update run progress
            run = session.query(TrainingRun).filter(TrainingRun.id == run_id).first()
            if run:
                run.iterations_completed = iteration
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

        iterations.append({
            "iteration_number": iteration,
            "symbol": symbol,
            "timeframe": timeframe,
            "strategy_type": strategy_type,
            "parameters": params,
            "final_balance": results.final_balance,
            "total_trades": results.total_trades,
            "win_rate": results.win_rate,
            "max_drawdown": results.max_drawdown,
            "profit_factor": results.profit_factor,
            "sharpe_ratio": results.sharpe_ratio,
            "avg_monthly_return": results.avg_monthly_return,
            "improvement_pct": improvement,
        })

        logger.info(
            "  → Sharpe=%.2f, WR=%.1f%%, PF=%.2f, DD=%.1f%%, Trades=%d%s",
            results.sharpe_ratio, results.win_rate * 100,
            results.profit_factor, results.max_drawdown * 100,
            results.total_trades,
            f" (improvement: {improvement:+.1%})" if improvement is not None else "",
        )

        # Check convergence
        if iteration > 1 and improvement is not None:
            if abs(improvement) < tc.improvement_threshold:
                logger.info("Training converged (improvement %.1f%% < threshold %.1f%%)",
                            improvement * 100, tc.improvement_threshold * 100)
                break

        # Prepare next iteration
        prev_metrics = metrics
        prev_params = params

        if suggested_params:
            # Validate suggested params against strategy's required params
            valid_params = {}
            for p in strategy.required_params():
                pname = p["name"]
                if pname in suggested_params:
                    try:
                        if p["type"] == "float":
                            valid_params[pname] = float(suggested_params[pname])
                        else:
                            valid_params[pname] = int(suggested_params[pname])
                    except (ValueError, TypeError):
                        valid_params[pname] = params.get(pname, p["default"])
                else:
                    valid_params[pname] = params.get(pname, p["default"])
            params = valid_params
        else:
            # No Claude suggestions — use heuristic adjustments
            params = _heuristic_adjust(strategy, params, metrics)
            if params is None:
                break  # No more adjustments possible

    return iterations


def run_training(
    name: str,
    symbols: list[str],
    timeframes: list[str],
    strategies: list[str],
    start_date: datetime,
    end_date: datetime,
    run_id: int | None = None,
) -> int:
    """Start a training run. Returns the run ID.

    This is synchronous — call via asyncio.to_thread() for non-blocking execution.
    If run_id is provided, uses an existing TrainingRun record.
    """
    cfg = load_config()
    now = datetime.now(timezone.utc)

    if run_id is None:
        # Create run record
        session = get_session()
        try:
            run = TrainingRun(
                name=name,
                status="running",
                symbols=symbols,
                timeframes=timeframes,
                strategies=strategies,
                start_date=start_date,
                end_date=end_date,
                iterations_completed=0,
                config_snapshot=cfg.training.model_dump(),
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
    else:
        # Mark pre-created run as running
        session = get_session()
        try:
            run = session.query(TrainingRun).filter(TrainingRun.id == run_id).first()
            if run:
                run.status = "running"
                session.commit()
        except Exception:
            session.rollback()
        finally:
            session.close()

    _active_runs[run_id] = False
    all_iterations = []

    try:
        for symbol in symbols:
            for timeframe in timeframes:
                if _is_stopped(run_id):
                    break

                # Load candles once for all strategies on this symbol/timeframe
                candles = _load_candles(symbol, timeframe, start_date, end_date)
                if len(candles) < 50:
                    logger.warning(
                        "Skipping %s/%s: only %d candles", symbol, timeframe, len(candles)
                    )
                    continue

                for strategy_type in strategies:
                    if _is_stopped(run_id):
                        break

                    iterations = run_training_for_combo(
                        run_id, symbol, timeframe, strategy_type,
                        candles, start_date, end_date,
                    )
                    all_iterations.extend(iterations)

        # Generate final report
        final_report = None
        if all_iterations and cfg.training.use_claude_evaluation and cfg.claude.enabled:
            final_report = claude_service.generate_training_report(all_iterations)

        # Update run as completed
        session = get_session()
        try:
            run = session.query(TrainingRun).filter(TrainingRun.id == run_id).first()
            if run:
                run.status = "completed"
                run.completed_at = datetime.now(timezone.utc)
                run.final_report = final_report
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

        logger.info("Training run %d completed with %d iterations", run_id, len(all_iterations))

    except Exception as exc:
        logger.error("Training run %d failed: %s", run_id, exc)
        session = get_session()
        try:
            run = session.query(TrainingRun).filter(TrainingRun.id == run_id).first()
            if run:
                run.status = "failed"
                run.completed_at = datetime.now(timezone.utc)
            session.commit()
        except Exception:
            session.rollback()
        finally:
            session.close()
    finally:
        _active_runs.pop(run_id, None)

    return run_id


def deploy_best_params(run_id: int) -> dict:
    """Deploy the best iteration's params to the live signal config.

    Finds the iteration with the best ranking metric and updates
    the strategy's default parameters in the signal config.
    """
    cfg = load_config()
    session = get_session()
    try:
        iterations = (
            session.query(TrainingIteration)
            .filter(TrainingIteration.training_run_id == run_id)
            .all()
        )
        if not iterations:
            return {"error": "No iterations found"}

        # Find best by configured ranking metric
        rank_by = cfg.training.rank_by
        best = max(iterations, key=lambda it: getattr(it, rank_by, 0) or 0)

        return {
            "strategy_type": best.strategy_type,
            "symbol": best.symbol,
            "timeframe": best.timeframe,
            "parameters": best.parameters,
            "iteration": best.iteration_number,
            "sharpe_ratio": best.sharpe_ratio,
            "win_rate": best.win_rate,
            "profit_factor": best.profit_factor,
        }
    finally:
        session.close()
