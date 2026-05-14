"""Structured strategy research and walk-forward evolution.

This engine searches explicit strategy genomes instead of asking the model to
rewrite free-form rules. The local model can still review results, but the
thing being optimized is deterministic JSON: thresholds, weights, filters, and
risk policy.
"""

from __future__ import annotations

import copy
import json
import logging
import random
from datetime import datetime, timedelta, timezone
from typing import Any

from config import load_config
from services import ai_service
from services.backtest_engine import BacktestConfig, BacktestResults, run_backtest
from services.strategy_service import get_strategy

logger = logging.getLogger(__name__)

_active_research: dict[int, bool] = {}
_research_progress: dict[int, dict] = {}


DEFAULT_SETTINGS = {
    "symbol_mode": "per_symbol",
    "population_size": 18,
    "generations": 5,
    "elite_count": 4,
    "validation_top_n": 5,
    "folds": 3,
    "initial_balance": 10000.0,
    "spread_pips": 1.5,
    "risk_per_trade_pct": 0.35,
    "lot_type": "mini",
    "realistic_execution": True,
    "entry_timing": "next_open",
    "slippage_pips": 0.2,
    "min_sl_pips": 10.0,
    "enforce_live_exit_policy": True,
    "broker_lot_units": 100000,
    "min_train_trades": 20,
    "min_train_profit_factor": 1.0,
    "min_train_return_pct": 0.0,
    "min_train_profitable_test_ratio": 0.4,
    "min_validation_trades": 20,
    "max_drawdown_pct": 6.0,
    "min_profit_factor": 1.15,
    "min_return_pct": 0.25,
    "min_profitable_test_ratio": 0.5,
    "max_symbol_loss_pct": 3.0,
    "holdout_enabled": True,
    "holdout_days": 14,
    "holdout_pct": 0.0,
    "min_holdout_bars": 60,
    "min_holdout_trades": 20,
    "min_holdout_profit_factor": 1.15,
    "min_holdout_return_pct": 0.1,
    "min_holdout_profitable_test_ratio": 0.5,
    "rolling_holdout_enabled": True,
    "rolling_holdout_windows": 4,
    "rolling_holdout_days": 14,
    "rolling_holdout_step_days": 14,
    "min_rolling_holdout_pass_ratio": 0.67,
    "confirmation_enabled": True,
    "confirmation_variants": 6,
    "confirmation_mutation_rate": 0.08,
    "confirmation_stress_multiplier": 1.25,
    "min_confirmation_pass_ratio": 0.67,
    "cost_stress_multiplier": 1.5,
    "min_stress_profit_factor": 1.05,
    "min_stress_return_pct": 0.0,
    "min_positive_month_ratio": 0.45,
    "max_month_loss_pct": 3.0,
    "min_monthly_tests_for_gate": 3,
    "min_session_hours": 4,
    "persist_group_winners": True,
    "mutation_rate": 0.35,
    "seed": None,
    "ai_review": False,
}

PRESET_SYMBOL_GROUPS = [
    {"name": "eur-gbp", "symbols": ["EURUSD", "GBPUSD"]},
    {"name": "aud-nzd", "symbols": ["AUDUSD", "NZDUSD"]},
    {"name": "jpy", "symbols": ["USDJPY"]},
    {"name": "cad-chf", "symbols": ["USDCAD", "USDCHF"]},
]

PARAM_SPACE = {
    "min_score": (2.2, 4.6, 0.1),
    "score_margin": (0.1, 0.9, 0.05),
    "trend_ema": (50, 200, 5),
    "fast_ema": (8, 34, 1),
    "medium_ema": (20, 80, 5),
    "rsi_buy_min": (35.0, 52.0, 1.0),
    "rsi_buy_max": (58.0, 78.0, 1.0),
    "rsi_sell_min": (22.0, 42.0, 1.0),
    "rsi_sell_max": (48.0, 65.0, 1.0),
    "bb_near_pct": (0.15, 0.55, 0.05),
    "atr_min_ratio": (0.35, 0.95, 0.05),
    "atr_max_ratio": (1.8, 4.0, 0.1),
    "breakout_lookback": (12, 48, 2),
    "sl_atr_multiplier": (1.2, 3.2, 0.1),
    "tp_rr": (1.2, 2.6, 0.1),
    "session_start_hour": (0, 23, 1),
    "session_end_hour": (1, 24, 1),
}

WEIGHT_SPACE = {
    "trend": (0.4, 1.6, 0.1),
    "ema_stack": (0.3, 1.4, 0.1),
    "rsi_zone": (0.2, 1.2, 0.1),
    "rsi_turn": (0.2, 1.2, 0.1),
    "macd": (0.2, 1.3, 0.1),
    "bb_pullback": (0.0, 1.0, 0.1),
    "breakout": (0.0, 1.0, 0.1),
    "candle": (0.0, 0.8, 0.05),
    "volatility": (0.0, 1.0, 0.1),
}


def request_stop(run_id: int) -> None:
    _active_research[run_id] = True


def get_progress(run_id: int) -> dict | None:
    return _research_progress.get(run_id)


def normalize_settings(raw: dict | None) -> dict:
    raw = raw or {}
    settings = dict(DEFAULT_SETTINGS)
    settings.update({k: v for k, v in raw.items() if v is not None})
    if str(settings.get("symbol_mode", "")).lower() not in {"global", "per_symbol", "clustered", "custom"}:
        settings["symbol_mode"] = DEFAULT_SETTINGS["symbol_mode"]
    else:
        settings["symbol_mode"] = str(settings["symbol_mode"]).lower()
    settings["population_size"] = max(6, min(80, int(settings["population_size"])))
    settings["generations"] = max(1, min(40, int(settings["generations"])))
    settings["elite_count"] = max(1, min(settings["population_size"], int(settings["elite_count"])))
    settings["validation_top_n"] = max(1, min(settings["population_size"], int(settings["validation_top_n"])))
    settings["folds"] = max(1, min(8, int(settings["folds"])))
    settings["initial_balance"] = max(1000.0, float(settings["initial_balance"]))
    settings["spread_pips"] = max(0.0, min(20.0, float(settings["spread_pips"])))
    settings["risk_per_trade_pct"] = max(0.05, min(5.0, float(settings["risk_per_trade_pct"])))
    settings["realistic_execution"] = _coerce_bool(settings["realistic_execution"])
    if str(settings.get("entry_timing", "")).lower() not in {"signal_close", "next_open"}:
        settings["entry_timing"] = DEFAULT_SETTINGS["entry_timing"]
    else:
        settings["entry_timing"] = str(settings["entry_timing"]).lower()
    settings["slippage_pips"] = max(0.0, min(10.0, float(settings["slippage_pips"])))
    settings["min_sl_pips"] = max(0.0, min(100.0, float(settings["min_sl_pips"])))
    settings["enforce_live_exit_policy"] = _coerce_bool(settings["enforce_live_exit_policy"])
    broker_lot_units = settings.get("broker_lot_units")
    settings["broker_lot_units"] = (
        None if broker_lot_units in {None, "", 0, "0"}
        else max(1000, min(1000000, int(broker_lot_units)))
    )
    settings["min_train_trades"] = max(1, min(10000, int(settings["min_train_trades"])))
    settings["min_train_profit_factor"] = max(0.1, min(10.0, float(settings["min_train_profit_factor"])))
    settings["min_train_return_pct"] = max(-50.0, min(100.0, float(settings["min_train_return_pct"])))
    settings["min_train_profitable_test_ratio"] = max(0.0, min(1.0, float(settings["min_train_profitable_test_ratio"])))
    settings["min_validation_trades"] = max(1, min(10000, int(settings["min_validation_trades"])))
    settings["max_drawdown_pct"] = max(0.1, min(50.0, float(settings["max_drawdown_pct"])))
    settings["min_profit_factor"] = max(0.1, min(10.0, float(settings["min_profit_factor"])))
    settings["min_return_pct"] = max(-50.0, min(100.0, float(settings["min_return_pct"])))
    settings["min_profitable_test_ratio"] = max(0.0, min(1.0, float(settings["min_profitable_test_ratio"])))
    settings["max_symbol_loss_pct"] = max(0.1, min(50.0, float(settings["max_symbol_loss_pct"])))
    settings["holdout_enabled"] = _coerce_bool(settings["holdout_enabled"])
    settings["holdout_days"] = max(0, min(120, int(settings["holdout_days"])))
    settings["holdout_pct"] = max(0.0, min(50.0, float(settings["holdout_pct"])))
    settings["min_holdout_bars"] = max(10, min(10000, int(settings["min_holdout_bars"])))
    settings["min_holdout_trades"] = max(1, min(10000, int(settings["min_holdout_trades"])))
    settings["min_holdout_profit_factor"] = max(0.1, min(10.0, float(settings["min_holdout_profit_factor"])))
    settings["min_holdout_return_pct"] = max(-50.0, min(100.0, float(settings["min_holdout_return_pct"])))
    settings["min_holdout_profitable_test_ratio"] = max(0.0, min(1.0, float(settings["min_holdout_profitable_test_ratio"])))
    settings["rolling_holdout_enabled"] = _coerce_bool(settings["rolling_holdout_enabled"])
    settings["rolling_holdout_windows"] = max(1, min(12, int(settings["rolling_holdout_windows"])))
    settings["rolling_holdout_days"] = max(1, min(120, int(settings["rolling_holdout_days"])))
    settings["rolling_holdout_step_days"] = max(1, min(120, int(settings["rolling_holdout_step_days"])))
    settings["min_rolling_holdout_pass_ratio"] = max(
        0.0,
        min(1.0, float(settings["min_rolling_holdout_pass_ratio"])),
    )
    settings["confirmation_enabled"] = _coerce_bool(settings["confirmation_enabled"])
    settings["confirmation_variants"] = max(1, min(24, int(settings["confirmation_variants"])))
    settings["confirmation_mutation_rate"] = max(0.01, min(0.35, float(settings["confirmation_mutation_rate"])))
    settings["confirmation_stress_multiplier"] = max(1.0, min(5.0, float(settings["confirmation_stress_multiplier"])))
    settings["min_confirmation_pass_ratio"] = max(
        0.0,
        min(1.0, float(settings["min_confirmation_pass_ratio"])),
    )
    settings["cost_stress_multiplier"] = max(1.0, min(5.0, float(settings["cost_stress_multiplier"])))
    settings["min_stress_profit_factor"] = max(0.1, min(10.0, float(settings["min_stress_profit_factor"])))
    settings["min_stress_return_pct"] = max(-50.0, min(100.0, float(settings["min_stress_return_pct"])))
    settings["min_positive_month_ratio"] = max(0.0, min(1.0, float(settings["min_positive_month_ratio"])))
    settings["max_month_loss_pct"] = max(0.1, min(50.0, float(settings["max_month_loss_pct"])))
    settings["min_monthly_tests_for_gate"] = max(1, min(1000, int(settings["min_monthly_tests_for_gate"])))
    settings["min_session_hours"] = max(1, min(23, int(settings["min_session_hours"])))
    settings["persist_group_winners"] = _coerce_bool(settings["persist_group_winners"])
    settings["mutation_rate"] = max(0.01, min(1.0, float(settings["mutation_rate"])))
    settings["ai_review"] = _coerce_bool(settings["ai_review"])
    return settings


def _coerce_bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() not in {"0", "false", "no", "off"}
    return bool(value)


def _backtest_execution_kwargs(settings: dict) -> dict:
    if not _coerce_bool(settings.get("realistic_execution")):
        return {}
    return {
        "entry_timing": settings.get("entry_timing", "next_open"),
        "slippage_pips": float(settings.get("slippage_pips") or 0.0),
        "min_sl_pips": float(settings.get("min_sl_pips") or 0.0),
        "enforce_live_exit_policy": _coerce_bool(settings.get("enforce_live_exit_policy")),
        "broker_lot_units": settings.get("broker_lot_units"),
    }


def default_genome() -> dict:
    strategy = get_strategy("research_genome")
    if strategy is None:
        raise RuntimeError("research_genome strategy is not registered")
    return copy.deepcopy(strategy.default_params())


def _genome_key(genome: dict) -> str:
    return json.dumps(genome, sort_keys=True, separators=(",", ":"))


def run_research(
    *,
    run_id: int,
    candles_by_symbol: dict[str, list[dict]],
    timeframe: str,
    settings: dict,
) -> dict:
    settings = normalize_settings(settings)
    _active_research[run_id] = False

    windows = {
        symbol: _build_windows(candles, settings["folds"], settings)
        for symbol, candles in candles_by_symbol.items()
    }
    usable_symbols = [
        symbol for symbol, w in windows.items()
        if w["train"] and w["validation"] and (not settings["holdout_enabled"] or w["holdout"])
    ]
    candles_by_symbol = {s: candles_by_symbol[s] for s in usable_symbols}
    windows = {s: windows[s] for s in usable_symbols}
    if not candles_by_symbol:
        raise ValueError("Not enough candle data to build train/validation/holdout windows")
    split_summary = {
        symbol: _window_summary(candles_by_symbol[symbol], windows[symbol])
        for symbol in usable_symbols
    }
    symbol_groups = _build_symbol_groups(usable_symbols, settings)
    if not symbol_groups:
        raise ValueError("No usable symbol groups were available for research")

    generations: list[dict] = []
    candidates: list[dict] = []
    best_progress: dict | None = None

    _research_progress[run_id] = {
        "run_id": run_id,
        "status": "running",
        "generation": 0,
        "generations": settings["generations"],
        "symbols": usable_symbols,
        "symbol_mode": settings["symbol_mode"],
        "symbol_groups": symbol_groups,
        "group": None,
        "group_index": 0,
        "group_count": len(symbol_groups),
        "split_summary": split_summary,
        "best_score": None,
        "best_metrics": None,
    }

    for group_idx, group in enumerate(symbol_groups, start=1):
        if _active_research.get(run_id):
            break
        group_symbols = group["symbols"]
        group_candles = {symbol: candles_by_symbol[symbol] for symbol in group_symbols}
        group_windows = {symbol: windows[symbol] for symbol in group_symbols}
        group_rng = random.Random(f"{settings.get('seed') or run_id}:{group['name']}")
        population = _initial_population(group_rng, settings["population_size"], settings)
        best_seen: dict | None = None
        last_scored: list[dict] = []

        for generation in range(1, settings["generations"] + 1):
            if _active_research.get(run_id):
                break
            scored = []
            for idx, genome in enumerate(population, start=1):
                if _active_research.get(run_id):
                    break
                metrics = evaluate_genome(
                    genome,
                    group_candles,
                    group_windows,
                    "train",
                    timeframe,
                    settings,
                )
                scored.append({"genome": genome, "train_metrics": metrics, "score": metrics["score"]})
                if best_seen is None or metrics["score"] > best_seen["score"]:
                    best_seen = scored[-1]
                if best_progress is None or metrics["score"] > best_progress["score"]:
                    best_progress = scored[-1]
                _research_progress[run_id].update({
                    "group": group["name"],
                    "group_index": group_idx,
                    "group_symbols": group_symbols,
                    "generation": generation,
                    "candidate": idx,
                    "population_size": len(population),
                    "best_score": round(best_progress["score"], 4) if best_progress else None,
                    "best_metrics": best_progress.get("train_metrics") if best_progress else None,
                })

            scored.sort(key=lambda item: item["score"], reverse=True)
            last_scored = scored
            gen_summary = {
                "group": group["name"],
                "group_index": group_idx,
                "symbols": group_symbols,
                "generation": generation,
                "best_score": scored[0]["score"] if scored else None,
                "best_metrics": scored[0]["train_metrics"] if scored else None,
                "avg_score": sum(item["score"] for item in scored) / len(scored) if scored else 0,
            }
            generations.append(gen_summary)
            if not scored:
                break

            elites = scored[: settings["elite_count"]]
            next_population = [copy.deepcopy(item["genome"]) for item in elites]
            while len(next_population) < settings["population_size"]:
                parent = group_rng.choice(elites)["genome"]
                next_population.append(mutate_genome(parent, group_rng, settings["mutation_rate"], settings))
            population = next_population

        top_train = []
        seen_genomes = set()
        for item in ([best_seen] if best_seen else []) + last_scored:
            if not item:
                continue
            key = _genome_key(item.get("genome") or {})
            if key in seen_genomes:
                continue
            seen_genomes.add(key)
            top_train.append(item)
        top_train = sorted(
            top_train,
            key=lambda item: item["train_metrics"]["score"],
            reverse=True,
        )[: settings["validation_top_n"]]

        group_candidates = []
        for item in top_train:
            candidate = _evaluate_candidate(
                item=item,
                group=group,
                group_index=group_idx,
                group_candles=group_candles,
                group_windows=group_windows,
                timeframe=timeframe,
                settings=settings,
            )
            group_candidates.append(candidate)
        group_candidates.sort(key=lambda item: (item["passed"], item["score"]), reverse=True)
        for group_rank, candidate in enumerate(group_candidates, start=1):
            candidate["group_rank"] = group_rank
            candidate["is_group_winner"] = group_rank == 1
        candidates.extend(group_candidates)

    candidates.sort(key=lambda item: (item["passed"], item["score"]), reverse=True)
    for idx, candidate in enumerate(candidates, start=1):
        candidate["rank"] = idx

    result = {
        "status": "stopped" if _active_research.get(run_id) else "completed",
        "timeframe": timeframe,
        "symbols": usable_symbols,
        "symbol_mode": settings["symbol_mode"],
        "symbol_groups": symbol_groups,
        "settings": settings,
        "split_summary": split_summary,
        "generations": generations,
        "top_candidates": candidates,
        "best_candidate": candidates[0] if candidates else None,
    }
    result["ai_review"] = _ai_review(result) if settings["ai_review"] and candidates else None
    _research_progress[run_id].update({
        "status": result["status"],
        "generation": settings["generations"],
        "completed_group_generations": len(generations),
        "best_score": candidates[0]["score"] if candidates else None,
        "best_metrics": (
            candidates[0].get("holdout_metrics")
            or candidates[0].get("validation_metrics")
            if candidates
            else None
        ),
        "passed": candidates[0]["passed"] if candidates else False,
    })
    _active_research.pop(run_id, None)
    return result


def _evaluate_candidate(
    *,
    item: dict,
    group: dict,
    group_index: int,
    group_candles: dict[str, list[dict]],
    group_windows: dict[str, dict[str, list[tuple[int, int]]]],
    timeframe: str,
    settings: dict,
) -> dict:
    genome = item["genome"]
    train_gate = score_train(item["train_metrics"], settings)
    validation = evaluate_genome(genome, group_candles, group_windows, "validation", timeframe, settings)
    validation_gate = score_validation(validation, settings)
    stress_multiplier = float(settings["cost_stress_multiplier"])
    validation_stress = None
    validation_stress_gate = {"passed": True, "status": "passed", "reasons": []}
    if stress_multiplier > 1.0:
        validation_stress = evaluate_genome(
            genome,
            group_candles,
            group_windows,
            "validation",
            timeframe,
            settings,
            spread_multiplier=stress_multiplier,
        )
        validation_stress_gate = score_cost_stress(validation_stress, settings, "Validation stress")

    holdout = None
    holdout_gate = {"passed": True, "status": "passed", "reasons": []}
    holdout_stress = None
    holdout_stress_gate = {"passed": True, "status": "passed", "reasons": []}
    if settings["holdout_enabled"]:
        if any(w["holdout"] for w in group_windows.values()):
            holdout = evaluate_genome(genome, group_candles, group_windows, "holdout", timeframe, settings)
            holdout_gate = score_holdout(holdout, settings)
            if stress_multiplier > 1.0:
                holdout_stress = evaluate_genome(
                    genome,
                    group_candles,
                    group_windows,
                    "holdout",
                    timeframe,
                    settings,
                    spread_multiplier=stress_multiplier,
                )
                holdout_stress_gate = score_cost_stress(holdout_stress, settings, "Holdout stress")
        else:
            holdout_gate = {
                "passed": False,
                "status": "failed",
                "reasons": ["Holdout enabled but no holdout windows were available."],
            }

    rolling_holdout = None
    rolling_holdout_gate = {"passed": True, "status": "passed", "reasons": []}
    if settings["rolling_holdout_enabled"]:
        rolling_holdout = evaluate_rolling_holdout(genome, group_candles, timeframe, settings)
        rolling_holdout_gate = score_rolling_holdout(rolling_holdout, settings)

    gates = {
        "train": train_gate,
        "validation": validation_gate,
        "validation_stress": validation_stress_gate,
        "holdout": holdout_gate,
        "holdout_stress": holdout_stress_gate,
        "rolling_holdout": rolling_holdout_gate,
    }
    confirmation = None
    if settings["confirmation_enabled"]:
        base_reasons = _collect_gate_reasons(gates)
        if not base_reasons:
            confirmation = evaluate_confirmation(genome, group_candles, timeframe, settings)
            gates["confirmation"] = score_confirmation(confirmation, settings)
        else:
            gates["confirmation"] = {"passed": False, "status": "skipped", "reasons": []}
    reasons = _collect_gate_reasons(gates)
    return {
        "rank": 0,
        "group_rank": 0,
        "is_group_winner": False,
        "symbol_group": group["name"],
        "group_index": group_index,
        "symbols": list(group["symbols"]),
        "generation": settings["generations"],
        "genome": genome,
        "train_metrics": item["train_metrics"],
        "validation_metrics": validation,
        "validation_stress_metrics": validation_stress,
        "holdout_metrics": holdout,
        "holdout_stress_metrics": holdout_stress,
        "rolling_holdout_metrics": rolling_holdout,
        "confirmation_metrics": confirmation,
        "gates": gates,
        "score": _candidate_selection_score(
            train=item["train_metrics"],
            validation=validation,
            validation_stress=validation_stress,
            holdout=holdout,
            holdout_stress=holdout_stress,
            rolling_holdout=rolling_holdout,
            confirmation=confirmation,
        ),
        "passed": not reasons,
        "reasons": reasons,
    }


def evaluate_genome(
    genome: dict,
    candles_by_symbol: dict[str, list[dict]],
    windows_by_symbol: dict[str, dict[str, list[tuple[int, int]]]],
    split: str,
    timeframe: str,
    settings: dict,
    spread_multiplier: float = 1.0,
) -> dict:
    results: list[tuple[str, BacktestResults]] = []
    min_bars = 40 if split in {"holdout", "rolling_holdout"} else 80
    for symbol, candles in candles_by_symbol.items():
        for start, end in windows_by_symbol[symbol][split]:
            window_candles = candles[start:end]
            if len(window_candles) < min_bars:
                continue
            config = BacktestConfig(
                symbol=symbol,
                timeframe=timeframe,
                strategy_type="research_genome",
                parameters=genome,
                start_date=_parse_dt(window_candles[0]["time"]),
                end_date=_parse_dt(window_candles[-1]["time"]),
                initial_balance=settings["initial_balance"],
                spread_pips=float(settings["spread_pips"]) * spread_multiplier,
                pip_value=0.01 if "JPY" in symbol.upper() else 0.0001,
                lot_type=settings["lot_type"],
                risk_per_trade_pct=settings["risk_per_trade_pct"],
                sl_atr_multiplier=float(genome.get("sl_atr_multiplier", 2.0)),
                tp_atr_multiplier=float(genome.get("sl_atr_multiplier", 2.0)) * float(genome.get("tp_rr", 1.7)),
                monthly_max_loss_pct=settings["max_drawdown_pct"],
                **_backtest_execution_kwargs(settings),
            )
            results.append((symbol, run_backtest(config, window_candles)))
    return aggregate_results(results, settings)


def evaluate_rolling_holdout(
    genome: dict,
    candles_by_symbol: dict[str, list[dict]],
    timeframe: str,
    settings: dict,
    spread_multiplier: float = 1.0,
) -> dict:
    results: list[tuple[str, BacktestResults]] = []
    window_summaries: list[dict] = []
    for symbol, candles in candles_by_symbol.items():
        for window in _build_rolling_holdout_windows(candles, settings):
            window_candles = candles[window["start_idx"]:window["end_idx"]]
            if len(window_candles) < int(settings["min_holdout_bars"]):
                continue
            config = BacktestConfig(
                symbol=symbol,
                timeframe=timeframe,
                strategy_type="research_genome",
                parameters=genome,
                start_date=_parse_dt(window_candles[0]["time"]),
                end_date=_parse_dt(window_candles[-1]["time"]),
                initial_balance=settings["initial_balance"],
                spread_pips=float(settings["spread_pips"]) * spread_multiplier,
                pip_value=0.01 if "JPY" in symbol.upper() else 0.0001,
                lot_type=settings["lot_type"],
                risk_per_trade_pct=settings["risk_per_trade_pct"],
                sl_atr_multiplier=float(genome.get("sl_atr_multiplier", 2.0)),
                tp_atr_multiplier=float(genome.get("sl_atr_multiplier", 2.0)) * float(genome.get("tp_rr", 1.7)),
                monthly_max_loss_pct=settings["max_drawdown_pct"],
                **_backtest_execution_kwargs(settings),
            )
            result = run_backtest(config, window_candles)
            results.append((symbol, result))
            metrics = aggregate_results([(symbol, result)], settings)
            gate = score_holdout(metrics, settings)
            window_summaries.append({
                "symbol": symbol,
                "window": window["window"],
                "start": window_candles[0]["time"],
                "end": window_candles[-1]["time"],
                "bars": len(window_candles),
                "return_pct": metrics["return_pct"],
                "profit_factor": metrics["profit_factor"],
                "max_drawdown": metrics["max_drawdown"],
                "total_trades": metrics["total_trades"],
                "win_rate": metrics["win_rate"],
                "passed": gate["passed"],
                "reasons": gate["reasons"],
            })

    metrics = aggregate_results(results, settings)
    passes = sum(1 for item in window_summaries if item["passed"])
    count = len(window_summaries)
    metrics["rolling_window_count"] = count
    metrics["rolling_window_passes"] = passes
    metrics["rolling_window_pass_ratio"] = round(passes / count, 3) if count else 0.0
    metrics["rolling_windows"] = window_summaries
    return metrics


def evaluate_confirmation(
    genome: dict,
    candles_by_symbol: dict[str, list[dict]],
    timeframe: str,
    settings: dict,
) -> dict:
    seed = f"{settings.get('seed')}:confirmation:{_genome_key(genome)}"
    rng = random.Random(seed)
    exact = evaluate_rolling_holdout(genome, candles_by_symbol, timeframe, settings)
    exact_gate = score_rolling_holdout(exact, settings)

    stress_multiplier = float(settings.get("confirmation_stress_multiplier") or 1.0)
    exact_stress = None
    exact_stress_gate = None
    if stress_multiplier > 1.0:
        stress_settings = _confirmation_stress_settings(settings)
        exact_stress = evaluate_rolling_holdout(
            genome,
            candles_by_symbol,
            timeframe,
            settings,
            spread_multiplier=stress_multiplier,
        )
        exact_stress_gate = score_rolling_holdout(exact_stress, stress_settings)

    variants = []
    for idx in range(int(settings["confirmation_variants"])):
        neighbor = _neighbor_genome(genome, rng, settings)
        metrics = evaluate_rolling_holdout(neighbor, candles_by_symbol, timeframe, settings)
        gate = score_rolling_holdout(metrics, settings)
        variants.append({
            "index": idx + 1,
            "passed": gate["passed"],
            "gate": gate,
            "metrics": metrics,
            "delta": _genome_delta(genome, neighbor),
        })

    neighbor_count = len(variants)
    neighbor_passes = sum(1 for item in variants if item["passed"])
    neighbor_metrics = [item["metrics"] for item in variants]
    neighbor_scores = [float(m.get("score") or 0.0) for m in neighbor_metrics]
    returns = [float(m.get("return_pct") or 0.0) for m in neighbor_metrics]
    profit_factors = [float(m.get("profit_factor") or 0.0) for m in neighbor_metrics]
    drawdowns = [float(m.get("max_drawdown") or 0.0) for m in neighbor_metrics]
    score_parts = [float(exact.get("score") or 0.0)] + neighbor_scores

    return {
        "exact": exact,
        "exact_gate": exact_gate,
        "exact_stress": exact_stress,
        "exact_stress_gate": exact_stress_gate,
        "stress_multiplier": stress_multiplier,
        "neighbor_count": neighbor_count,
        "neighbor_passes": neighbor_passes,
        "neighbor_pass_ratio": round(neighbor_passes / neighbor_count, 3) if neighbor_count else 0.0,
        "avg_neighbor_return_pct": round(sum(returns) / len(returns), 3) if returns else 0.0,
        "worst_neighbor_return_pct": round(min(returns), 3) if returns else 0.0,
        "worst_neighbor_profit_factor": round(min(profit_factors), 3) if profit_factors else 0.0,
        "max_neighbor_drawdown": round(max(drawdowns), 4) if drawdowns else 0.0,
        "score": round(sum(score_parts) / len(score_parts), 4) if score_parts else 0.0,
        "variants": variants,
    }


def aggregate_results(results: list[tuple[str, BacktestResults]], settings: dict) -> dict:
    if not results:
        return {
            "score": -9999.0,
            "tests": 0,
            "total_trades": 0,
            "return_pct": -100.0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "max_drawdown": 1.0,
            "sharpe_ratio": 0.0,
            "profitable_test_ratio": 0.0,
            "worst_symbol_return_pct": -100.0,
            "expectancy": 0.0,
            "avg_trades_per_test": 0.0,
            "months": 0,
            "positive_month_ratio": 0.0,
            "worst_month_pct": -100.0,
            "per_symbol": {},
        }

    initial = settings["initial_balance"]
    all_trades = []
    total_pnl = 0.0
    drawdown = 0.0
    sharpes = []
    profitable_tests = 0
    monthly_returns = []
    per_symbol: dict[str, dict] = {}

    for symbol, result in results:
        pnl = result.final_balance - initial
        total_pnl += pnl
        all_trades.extend(result.trades)
        drawdown = max(drawdown, result.max_drawdown)
        sharpes.append(result.sharpe_ratio)
        if pnl > 0:
            profitable_tests += 1
        monthly_returns.extend(float(m.get("pct", 0.0)) for m in result.monthly_pnl)
        bucket = per_symbol.setdefault(symbol, {"tests": 0, "trades": 0, "pnl": 0.0})
        bucket["tests"] += 1
        bucket["trades"] += result.total_trades
        bucket["pnl"] += pnl

    wins = sum(1 for t in all_trades if t.get("pnl", 0) > 0)
    gross_profit = sum(t.get("pnl", 0) for t in all_trades if t.get("pnl", 0) > 0)
    gross_loss = abs(sum(t.get("pnl", 0) for t in all_trades if t.get("pnl", 0) <= 0))
    total_trades = len(all_trades)
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else (999.0 if gross_profit > 0 else 0.0)
    return_pct = total_pnl / (initial * len(results)) * 100.0
    profitable_ratio = profitable_tests / len(results)
    positive_month_ratio = (
        sum(1 for value in monthly_returns if value > 0) / len(monthly_returns)
        if monthly_returns
        else 0.0
    )
    worst_month_pct = min(monthly_returns) if monthly_returns else 0.0
    for symbol, bucket in per_symbol.items():
        bucket["return_pct"] = bucket["pnl"] / (initial * bucket["tests"]) * 100.0
        bucket["pnl"] = round(bucket["pnl"], 2)
        bucket["return_pct"] = round(bucket["return_pct"], 3)
    worst_symbol_return = min((b["return_pct"] for b in per_symbol.values()), default=-100.0)

    score = _fitness_score(
        return_pct=return_pct,
        profit_factor=profit_factor,
        sharpe=sum(sharpes) / len(sharpes) if sharpes else 0.0,
        max_drawdown=drawdown,
        total_trades=total_trades,
        profitable_ratio=profitable_ratio,
        worst_symbol_return=worst_symbol_return,
        settings=settings,
    )

    return {
        "score": round(score, 4),
        "tests": len(results),
        "total_trades": total_trades,
        "return_pct": round(return_pct, 3),
        "win_rate": round(wins / total_trades, 4) if total_trades else 0.0,
        "profit_factor": round(min(profit_factor, 999.0), 3),
        "max_drawdown": round(drawdown, 4),
        "sharpe_ratio": round(sum(sharpes) / len(sharpes), 4) if sharpes else 0.0,
        "profitable_test_ratio": round(profitable_ratio, 3),
        "worst_symbol_return_pct": round(worst_symbol_return, 3),
        "expectancy": round(total_pnl / total_trades, 2) if total_trades else 0.0,
        "avg_trades_per_test": round(total_trades / len(results), 2),
        "months": len(monthly_returns),
        "positive_month_ratio": round(positive_month_ratio, 3),
        "worst_month_pct": round(worst_month_pct, 3),
        "per_symbol": per_symbol,
    }


def score_train(metrics: dict, settings: dict) -> dict:
    return _score_split(
        "Train",
        metrics,
        settings,
        min_trades=int(settings["min_train_trades"]),
        min_profit_factor=float(settings["min_train_profit_factor"]),
        min_return_pct=float(settings["min_train_return_pct"]),
        min_profitable_test_ratio=float(settings["min_train_profitable_test_ratio"]),
    )


def score_validation(metrics: dict, settings: dict) -> dict:
    return _score_split(
        "Validation",
        metrics,
        settings,
        min_trades=int(settings["min_validation_trades"]),
        min_profit_factor=float(settings["min_profit_factor"]),
        min_return_pct=float(settings["min_return_pct"]),
        min_profitable_test_ratio=float(settings["min_profitable_test_ratio"]),
    )


def score_holdout(metrics: dict, settings: dict) -> dict:
    return _score_split(
        "Holdout",
        metrics,
        settings,
        min_trades=int(settings["min_holdout_trades"]),
        min_profit_factor=float(settings["min_holdout_profit_factor"]),
        min_return_pct=float(settings["min_holdout_return_pct"]),
        min_profitable_test_ratio=float(settings["min_holdout_profitable_test_ratio"]),
    )


def score_rolling_holdout(metrics: dict, settings: dict) -> dict:
    reasons = []
    count = int(metrics.get("rolling_window_count") or 0)
    expected = int(settings["rolling_holdout_windows"])
    pass_ratio = float(metrics.get("rolling_window_pass_ratio") or 0.0)
    min_pass_ratio = float(settings["min_rolling_holdout_pass_ratio"])
    if count < expected:
        reasons.append(f"Rolling holdout windows {count} below {expected}.")
    if pass_ratio < min_pass_ratio:
        reasons.append(f"Rolling holdout pass ratio {pass_ratio:.2f} below {min_pass_ratio:.2f}.")
    aggregate_gate = _score_split(
        "Rolling holdout",
        metrics,
        settings,
        min_trades=int(settings["min_holdout_trades"]) * max(1, count),
        min_profit_factor=float(settings["min_holdout_profit_factor"]),
        min_return_pct=float(settings["min_holdout_return_pct"]),
        min_profitable_test_ratio=min_pass_ratio,
    )
    reasons.extend(aggregate_gate["reasons"])
    return {"passed": not reasons, "status": "passed" if not reasons else "failed", "reasons": reasons}


def score_confirmation(metrics: dict | None, settings: dict) -> dict:
    reasons = []
    if not metrics:
        return {
            "passed": False,
            "status": "failed",
            "reasons": ["Confirmation metrics were not produced."],
        }

    exact_gate = metrics.get("exact_gate") or {}
    if exact_gate.get("passed") is not True:
        reasons.extend(f"Confirmation exact: {reason}" for reason in exact_gate.get("reasons") or [])

    stress_gate = metrics.get("exact_stress_gate") or {}
    if stress_gate and stress_gate.get("passed") is not True:
        reasons.extend(f"Confirmation stress: {reason}" for reason in stress_gate.get("reasons") or [])

    expected_neighbors = int(settings["confirmation_variants"])
    neighbor_count = int(metrics.get("neighbor_count") or 0)
    neighbor_pass_ratio = float(metrics.get("neighbor_pass_ratio") or 0.0)
    min_pass_ratio = float(settings["min_confirmation_pass_ratio"])
    if neighbor_count < expected_neighbors:
        reasons.append(f"Confirmation variants {neighbor_count} below {expected_neighbors}.")
    if neighbor_pass_ratio < min_pass_ratio:
        reasons.append(
            f"Confirmation neighbor pass ratio {neighbor_pass_ratio:.3f} below {min_pass_ratio:.3f}."
        )
    if float(metrics.get("worst_neighbor_return_pct") or 0.0) < -float(settings["max_symbol_loss_pct"]):
        reasons.append(
            f"Confirmation worst neighbor {metrics['worst_neighbor_return_pct']:.2f}% breaches "
            f"-{settings['max_symbol_loss_pct']:.2f}%."
        )
    return {"passed": not reasons, "status": "passed" if not reasons else "failed", "reasons": reasons}


def score_cost_stress(metrics: dict, settings: dict, label: str) -> dict:
    reasons = []
    if metrics["max_drawdown"] > float(settings["max_drawdown_pct"]) / 100.0:
        reasons.append(f"{label} drawdown {metrics['max_drawdown'] * 100:.2f}% exceeds {settings['max_drawdown_pct']:.2f}%.")
    if metrics["profit_factor"] < float(settings["min_stress_profit_factor"]):
        reasons.append(f"{label} PF {metrics['profit_factor']:.2f} below {settings['min_stress_profit_factor']:.2f}.")
    if metrics["return_pct"] < float(settings["min_stress_return_pct"]):
        reasons.append(f"{label} return {metrics['return_pct']:.2f}% below {settings['min_stress_return_pct']:.2f}%.")
    if metrics["worst_symbol_return_pct"] < -float(settings["max_symbol_loss_pct"]):
        reasons.append(f"{label} worst symbol {metrics['worst_symbol_return_pct']:.2f}% breaches -{settings['max_symbol_loss_pct']:.2f}%.")
    return {"passed": not reasons, "status": "passed" if not reasons else "failed", "reasons": reasons}


def _score_split(
    label: str,
    metrics: dict,
    settings: dict,
    *,
    min_trades: int,
    min_profit_factor: float,
    min_return_pct: float,
    min_profitable_test_ratio: float,
) -> dict:
    reasons = []
    if metrics["total_trades"] < min_trades:
        reasons.append(f"{label} trades {metrics['total_trades']} below {min_trades}.")
    if metrics["max_drawdown"] > float(settings["max_drawdown_pct"]) / 100.0:
        reasons.append(f"{label} drawdown {metrics['max_drawdown'] * 100:.2f}% exceeds {settings['max_drawdown_pct']:.2f}%.")
    if metrics["profit_factor"] < min_profit_factor:
        reasons.append(f"{label} PF {metrics['profit_factor']:.2f} below {min_profit_factor:.2f}.")
    if metrics["return_pct"] < min_return_pct:
        reasons.append(f"{label} return {metrics['return_pct']:.2f}% below {min_return_pct:.2f}%.")
    if metrics["profitable_test_ratio"] < min_profitable_test_ratio:
        reasons.append(f"{label} profitable test ratio {metrics['profitable_test_ratio']:.2f} below {min_profitable_test_ratio:.2f}.")
    if metrics["worst_symbol_return_pct"] < -float(settings["max_symbol_loss_pct"]):
        reasons.append(f"{label} worst symbol {metrics['worst_symbol_return_pct']:.2f}% breaches -{settings['max_symbol_loss_pct']:.2f}%.")
    if int(metrics.get("months") or 0) >= int(settings["min_monthly_tests_for_gate"]):
        if metrics.get("positive_month_ratio", 0.0) < float(settings["min_positive_month_ratio"]):
            reasons.append(
                f"{label} positive month ratio {metrics['positive_month_ratio']:.2f} below "
                f"{settings['min_positive_month_ratio']:.2f}."
            )
        if metrics.get("worst_month_pct", 0.0) < -float(settings["max_month_loss_pct"]):
            reasons.append(
                f"{label} worst month {metrics['worst_month_pct']:.2f}% breaches "
                f"-{settings['max_month_loss_pct']:.2f}%."
            )
    return {"passed": not reasons, "status": "passed" if not reasons else "failed", "reasons": reasons}


def _collect_gate_reasons(gates: dict[str, dict]) -> list[str]:
    reasons: list[str] = []
    for gate in gates.values():
        reasons.extend(gate.get("reasons") or [])
    return reasons


def _candidate_selection_score(
    *,
    train: dict,
    validation: dict,
    validation_stress: dict | None,
    holdout: dict | None,
    holdout_stress: dict | None,
    rolling_holdout: dict | None = None,
    confirmation: dict | None = None,
) -> float:
    weighted_score = train.get("score", 0.0) * 0.20 + validation.get("score", 0.0) * 0.35
    weight = 0.55
    if holdout:
        weighted_score += holdout.get("score", 0.0) * 0.45
        weight += 0.45
    if validation_stress:
        weighted_score += validation_stress.get("score", 0.0) * 0.15
        weight += 0.15
    if holdout_stress:
        weighted_score += holdout_stress.get("score", 0.0) * 0.25
        weight += 0.25
    if rolling_holdout:
        weighted_score += rolling_holdout.get("score", 0.0) * 0.30
        weight += 0.30
    if confirmation:
        weighted_score += confirmation.get("score", 0.0) * 0.20
        weight += 0.20
    return round(weighted_score / weight, 4)


def mutate_genome(parent: dict, rng: random.Random, mutation_rate: float, settings: dict | None = None) -> dict:
    child = copy.deepcopy(parent)
    for key, spec in PARAM_SPACE.items():
        if rng.random() < mutation_rate:
            child[key] = _mutate_value(child.get(key), spec, rng)
    weights = dict(child.get("weights") or {})
    for key, spec in WEIGHT_SPACE.items():
        if rng.random() < mutation_rate:
            weights[key] = _mutate_value(weights.get(key), spec, rng)
    child["weights"] = weights
    if rng.random() < mutation_rate * 0.25:
        child["long_enabled"] = not bool(child.get("long_enabled", True))
    if rng.random() < mutation_rate * 0.25:
        child["short_enabled"] = not bool(child.get("short_enabled", True))
    if rng.random() < mutation_rate * 0.5:
        child["session_enabled"] = not bool(child.get("session_enabled", False))
    if not child.get("long_enabled") and not child.get("short_enabled"):
        child["long_enabled"] = True
        child["short_enabled"] = True
    _repair_genome(child, settings)
    return child


def _neighbor_genome(parent: dict, rng: random.Random, settings: dict) -> dict:
    child = copy.deepcopy(parent)
    mutation_rate = float(settings["confirmation_mutation_rate"])
    changed = False
    param_keys = [
        key for key in PARAM_SPACE
        if key not in {"session_start_hour", "session_end_hour"} or child.get("session_enabled")
    ]
    for key in param_keys:
        if rng.random() < mutation_rate:
            child[key] = _mutate_value(child.get(key), PARAM_SPACE[key], rng)
            changed = True

    weights = dict(child.get("weights") or {})
    for key, spec in WEIGHT_SPACE.items():
        if rng.random() < mutation_rate:
            weights[key] = _mutate_value(weights.get(key), spec, rng)
            changed = True
    child["weights"] = weights

    if not changed:
        key = rng.choice(param_keys or list(PARAM_SPACE))
        child[key] = _mutate_value(child.get(key), PARAM_SPACE[key], rng)

    child["long_enabled"] = bool(parent.get("long_enabled", True))
    child["short_enabled"] = bool(parent.get("short_enabled", True))
    child["session_enabled"] = bool(parent.get("session_enabled", False))
    _repair_genome(child, settings)
    return child


def _genome_delta(base: dict, candidate: dict) -> dict:
    delta = {
        key: candidate.get(key)
        for key in PARAM_SPACE
        if candidate.get(key) != base.get(key)
    }
    weight_delta = {
        key: (candidate.get("weights") or {}).get(key)
        for key in WEIGHT_SPACE
        if (candidate.get("weights") or {}).get(key) != (base.get("weights") or {}).get(key)
    }
    if weight_delta:
        delta["weights"] = weight_delta
    return delta


def _confirmation_stress_settings(settings: dict) -> dict:
    stress_settings = dict(settings)
    stress_settings["min_holdout_profit_factor"] = settings["min_stress_profit_factor"]
    stress_settings["min_holdout_return_pct"] = settings["min_stress_return_pct"]
    return stress_settings


def genome_to_rules_text(genome: dict) -> str:
    weights = genome.get("weights") or {}
    return (
        "STRUCTURED RESEARCH GENOME - deterministic execution\n\n"
        "This candidate was evolved by walk-forward research. The JSON schema in "
        "parameters.strategy_schema is the source of truth for execution and backtesting.\n\n"
        f"Direction enabled: long={bool(genome.get('long_enabled', True))}, "
        f"short={bool(genome.get('short_enabled', True))}\n"
        f"Minimum score: {genome.get('min_score')}, score margin: {genome.get('score_margin')}\n"
        f"Trend EMA: {genome.get('trend_ema')}, fast/medium EMA: "
        f"{genome.get('fast_ema')}/{genome.get('medium_ema')}\n"
        f"RSI buy zone: {genome.get('rsi_buy_min')}-{genome.get('rsi_buy_max')}; "
        f"RSI sell zone: {genome.get('rsi_sell_min')}-{genome.get('rsi_sell_max')}\n"
        f"Bollinger period/std/proximity: {genome.get('bb_period')}/"
        f"{genome.get('bb_std')}/{genome.get('bb_near_pct')}\n"
        f"ATR ratio filter: {genome.get('atr_min_ratio')}-{genome.get('atr_max_ratio')}\n"
        f"SL: {genome.get('sl_atr_multiplier')}x ATR; TP: {genome.get('tp_rr')}R\n\n"
        f"Entry session filter: enabled={bool(genome.get('session_enabled', False))}, "
        f"UTC {genome.get('session_start_hour', 7)}-{genome.get('session_end_hour', 20)}\n\n"
        "Weights:\n"
        + "\n".join(f"- {k}: {v}" for k, v in sorted(weights.items()))
    )


def _initial_population(rng: random.Random, size: int, settings: dict | None = None) -> list[dict]:
    base = default_genome()
    population = [base]
    while len(population) < size:
        population.append(mutate_genome(base, rng, mutation_rate=0.9, settings=settings))
    return population


def _build_symbol_groups(usable_symbols: list[str], settings: dict) -> list[dict]:
    selected = list(dict.fromkeys(usable_symbols))
    selected_set = set(selected)
    mode = settings.get("symbol_mode", "global")
    if mode == "global":
        return [{"name": "global", "symbols": selected}] if selected else []
    if mode == "per_symbol":
        return [{"name": symbol, "symbols": [symbol]} for symbol in selected]
    if mode == "custom":
        groups = _normalize_custom_symbol_groups(settings.get("symbol_groups"), selected_set)
        if groups:
            return groups

    groups: list[dict] = []
    assigned: set[str] = set()
    for group in PRESET_SYMBOL_GROUPS:
        symbols = [symbol for symbol in group["symbols"] if symbol in selected_set]
        if symbols:
            groups.append({"name": group["name"], "symbols": symbols})
            assigned.update(symbols)
    for symbol in selected:
        if symbol not in assigned:
            groups.append({"name": symbol, "symbols": [symbol]})
    return groups


def _normalize_custom_symbol_groups(raw_groups: Any, selected_set: set[str]) -> list[dict]:
    if not isinstance(raw_groups, list):
        return []
    groups = []
    assigned: set[str] = set()
    for idx, raw in enumerate(raw_groups, start=1):
        if isinstance(raw, dict):
            name = str(raw.get("name") or f"group-{idx}")[:64]
            raw_symbols = raw.get("symbols") or []
        else:
            name = f"group-{idx}"
            raw_symbols = raw
        if not isinstance(raw_symbols, list):
            continue
        symbols = []
        for symbol in raw_symbols:
            symbol = str(symbol).upper()
            if symbol in selected_set and symbol not in assigned:
                symbols.append(symbol)
                assigned.add(symbol)
        if symbols:
            groups.append({"name": name, "symbols": symbols})
    return groups


def _build_windows(candles: list[dict], folds: int, settings: dict | None = None) -> dict[str, list[tuple[int, int]]]:
    settings = settings or DEFAULT_SETTINGS
    n = len(candles)
    min_train = 180
    min_val = 80
    min_holdout = int(settings.get("min_holdout_bars", 60))
    holdout: list[tuple[int, int]] = []
    research_end = n

    if settings.get("holdout_enabled", True):
        holdout_start = _find_holdout_start(candles, settings)
        if holdout_start is None or n - holdout_start < min_holdout:
            return {"train": [], "validation": [], "holdout": []}
        holdout = [(holdout_start, n)]
        research_end = holdout_start

    if research_end < min_train + min_val:
        return {"train": [], "validation": [], "holdout": holdout}
    folds = max(1, min(folds, 5))
    seg = research_end // (folds + 1)
    train = []
    validation = []
    for idx in range(folds):
        train_end = seg * (idx + 1)
        val_start = train_end
        val_end = seg * (idx + 2) if idx < folds - 1 else research_end
        if train_end >= min_train and val_end - val_start >= min_val:
            train.append((0, train_end))
            validation.append((val_start, val_end))
    if not train:
        split = int(research_end * 0.7)
        train = [(0, split)]
        validation = [(split, research_end)]
    return {"train": train, "validation": validation, "holdout": holdout}


def _find_holdout_start(candles: list[dict], settings: dict) -> int | None:
    if not candles:
        return None

    holdout_days = int(settings.get("holdout_days") or 0)
    if holdout_days > 0:
        try:
            cutoff = _parse_dt(candles[-1]["time"]) - timedelta(days=holdout_days)
            for idx, candle in enumerate(candles):
                if _parse_dt(candle["time"]) >= cutoff:
                    return idx
        except Exception:
            logger.warning("Unable to build date-based holdout; falling back to percent split", exc_info=True)

    holdout_pct = float(settings.get("holdout_pct") or 0.0)
    if holdout_pct > 0:
        holdout_bars = max(int(settings.get("min_holdout_bars", 60)), int(len(candles) * holdout_pct / 100.0))
        return max(0, len(candles) - holdout_bars)

    return None


def _build_rolling_holdout_windows(candles: list[dict], settings: dict) -> list[dict]:
    if not candles:
        return []
    try:
        last_ts = _parse_dt(candles[-1]["time"])
    except Exception:
        logger.warning("Unable to build rolling holdout windows", exc_info=True)
        return []

    windows = []
    window_days = int(settings.get("rolling_holdout_days") or settings.get("holdout_days") or 14)
    step_days = int(settings.get("rolling_holdout_step_days") or window_days)
    requested = int(settings.get("rolling_holdout_windows") or 1)
    min_bars = int(settings.get("min_holdout_bars", 60))
    for window_idx in range(requested):
        end_ts = last_ts - timedelta(days=step_days * window_idx)
        start_ts = end_ts - timedelta(days=window_days)
        start_idx = None
        end_idx = None
        for idx, candle in enumerate(candles):
            ts = _parse_dt(candle["time"])
            if start_idx is None and ts >= start_ts:
                start_idx = idx
            if ts <= end_ts:
                end_idx = idx + 1
            elif start_idx is not None:
                break
        if start_idx is None or end_idx is None or end_idx <= start_idx:
            continue
        if end_idx - start_idx < min_bars:
            continue
        windows.append({
            "window": window_idx,
            "start_idx": start_idx,
            "end_idx": end_idx,
            "start": candles[start_idx]["time"],
            "end": candles[end_idx - 1]["time"],
            "bars": end_idx - start_idx,
        })
    return windows


def _window_summary(candles: list[dict], windows: dict[str, list[tuple[int, int]]]) -> dict:
    summary: dict[str, list[dict]] = {}
    for split, ranges in windows.items():
        summary[split] = []
        for start, end in ranges:
            if start < 0 or end > len(candles) or start >= end:
                continue
            summary[split].append({
                "start": candles[start]["time"],
                "end": candles[end - 1]["time"],
                "bars": end - start,
            })
    return summary


def _fitness_score(
    *,
    return_pct: float,
    profit_factor: float,
    sharpe: float,
    max_drawdown: float,
    total_trades: int,
    profitable_ratio: float,
    worst_symbol_return: float,
    settings: dict,
) -> float:
    dd_pct = max_drawdown * 100.0
    low_trade_penalty = max(0, int(settings["min_train_trades"]) - total_trades) * 0.25
    dd_penalty = max(0.0, dd_pct - float(settings["max_drawdown_pct"])) * 2.0
    worst_symbol_penalty = max(0.0, -float(worst_symbol_return) - float(settings["max_symbol_loss_pct"])) * 1.5
    pf_component = min(profit_factor, 4.0) * 1.5
    pf_shortfall_penalty = max(0.0, float(settings["min_profit_factor"]) - profit_factor) * 4.0
    negative_return_penalty = max(0.0, -return_pct) * 1.5
    return (
        return_pct * 1.2
        + pf_component
        + sharpe * 0.8
        + profitable_ratio * 4.0
        - dd_pct * 0.9
        - low_trade_penalty
        - dd_penalty
        - worst_symbol_penalty
        - pf_shortfall_penalty
        - negative_return_penalty
    )


def _mutate_value(current: Any, spec: tuple[float, float, float], rng: random.Random) -> float | int:
    low, high, step = spec
    if current is None:
        current = rng.uniform(low, high)
    span = high - low
    delta = rng.gauss(0, span * 0.12)
    value = min(high, max(low, float(current) + delta))
    steps = round((value - low) / step)
    value = low + steps * step
    return int(round(value)) if float(step).is_integer() and float(low).is_integer() else round(value, 4)


def _repair_genome(genome: dict, settings: dict | None = None) -> None:
    if int(genome.get("fast_ema", 20)) >= int(genome.get("medium_ema", 50)):
        genome["fast_ema"] = max(5, int(genome["medium_ema"]) // 2)
    if int(genome.get("medium_ema", 50)) >= int(genome.get("trend_ema", 100)):
        genome["medium_ema"] = max(int(genome["fast_ema"]) + 2, int(genome["trend_ema"]) // 2)
    if float(genome.get("rsi_buy_min", 42)) >= float(genome.get("rsi_buy_max", 68)):
        genome["rsi_buy_min"], genome["rsi_buy_max"] = 42.0, 68.0
    if float(genome.get("rsi_sell_min", 32)) >= float(genome.get("rsi_sell_max", 58)):
        genome["rsi_sell_min"], genome["rsi_sell_max"] = 32.0, 58.0
    genome["session_start_hour"] = max(0, min(23, int(genome.get("session_start_hour", 7))))
    genome["session_end_hour"] = max(1, min(24, int(genome.get("session_end_hour", 20))))
    if genome.get("session_enabled"):
        min_session_hours = int((settings or DEFAULT_SETTINGS).get("min_session_hours", 1))
        _enforce_min_session_hours(genome, min_session_hours)


def _session_duration_hours(start_hour: int, end_hour: int) -> int:
    if start_hour == 0 and end_hour == 24:
        return 24
    if end_hour == start_hour:
        return 0
    if end_hour > start_hour:
        return end_hour - start_hour
    return (24 - start_hour) + end_hour


def _enforce_min_session_hours(genome: dict, min_session_hours: int) -> None:
    min_session_hours = max(1, min(23, int(min_session_hours)))
    start = max(0, min(23, int(genome.get("session_start_hour", 7))))
    end = max(1, min(24, int(genome.get("session_end_hour", 20))))
    if _session_duration_hours(start, end) >= min_session_hours:
        genome["session_start_hour"] = start
        genome["session_end_hour"] = end
        return
    new_end = start + min_session_hours
    genome["session_start_hour"] = start
    genome["session_end_hour"] = new_end if new_end <= 24 else new_end - 24


def _parse_dt(value: str) -> datetime:
    dt = datetime.fromisoformat(value)
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def _ai_review(result: dict) -> dict | None:
    cfg = load_config()
    if not ai_service.is_enabled(cfg):
        return None
    compact = {
        "best_candidate": {
            "passed": result["best_candidate"]["passed"],
            "reasons": result["best_candidate"].get("reasons", []),
            "train_metrics": result["best_candidate"]["train_metrics"],
            "validation_metrics": result["best_candidate"]["validation_metrics"],
            "validation_stress_metrics": result["best_candidate"].get("validation_stress_metrics"),
            "holdout_metrics": result["best_candidate"].get("holdout_metrics"),
            "holdout_stress_metrics": result["best_candidate"].get("holdout_stress_metrics"),
            "rolling_holdout_metrics": result["best_candidate"].get("rolling_holdout_metrics"),
            "confirmation_metrics": result["best_candidate"].get("confirmation_metrics"),
            "gates": result["best_candidate"].get("gates", {}),
            "symbol_group": result["best_candidate"].get("symbol_group"),
            "symbols": result["best_candidate"].get("symbols", []),
            "genome": result["best_candidate"]["genome"],
        },
        "settings": result["settings"],
        "symbol_groups": result.get("symbol_groups", []),
        "split_summary": result.get("split_summary", {}),
    }
    return ai_service.generate_json(
        system=(
            "You are a forex research reviewer. Analyze deterministic walk-forward "
            "strategy results. Do not claim profitability; identify robustness risks "
            "and next mutation priorities. Return valid JSON only."
        ),
        prompt=(
            "Review this research result. Return JSON with keys: summary, robustness_risks, "
            "next_tests, mutation_priorities, promote_recommendation.\n\n"
            f"{compact}"
        ),
        cfg=cfg,
        max_tokens=1200,
        temperature=0.1,
        review=True,
        fallback=None,
    )
