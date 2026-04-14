"""Local strategy parameter optimizer.

Grid-searches parameter combinations across backtests to find
profitable configurations without any external API calls.

Performance: precomputed indicators via strategy.precompute() — O(n) per backtest.
"""

from __future__ import annotations

import itertools
import logging
import threading
from dataclasses import dataclass
from datetime import datetime
from typing import List

from services.backtest_engine import BacktestConfig, BacktestResults, run_backtest
from services.strategy_service import STRATEGIES

logger = logging.getLogger(__name__)


class OptimizationCancelled(Exception):
    """Raised when the user cancels a running optimization."""


# Module-level cancellation event — set by the /api/optimizer/stop endpoint.
_cancel_event = threading.Event()


def request_cancel() -> None:
    """Signal the running optimizer to stop."""
    _cancel_event.set()


def is_cancelled() -> bool:
    return _cancel_event.is_set()


def reset_cancel() -> None:
    _cancel_event.clear()


@dataclass
class ParamRange:
    """Defines a range of values to search for a single parameter."""
    name: str
    values: list  # explicit list of values to try


@dataclass
class OptimizationConfig:
    symbol: str
    timeframe: str
    strategy_type: str
    param_ranges: List[ParamRange]
    start_date: datetime
    end_date: datetime
    initial_balance: float = 10000.0
    rank_by: str = "sharpe_ratio"

    # Risk management (passed through to BacktestConfig)
    lot_type: str = "mini"
    risk_per_trade_pct: float = 1.0
    sl_atr_multiplier: float = 1.5
    tp_atr_multiplier: float = 2.0
    monthly_max_loss_pct: float = 7.0


@dataclass
class OptimizationResult:
    parameters: dict
    final_balance: float
    total_trades: int
    win_rate: float
    max_drawdown: float
    profit_factor: float
    sharpe_ratio: float
    score: float  # the ranking metric value
    avg_monthly_return: float = 0.0
    max_monthly_drawdown: float = 0.0
    months_above_target: int = 0


# Predefined search spaces for each strategy
DEFAULT_SEARCH_SPACES: dict[str, List[ParamRange]] = {
    # --- Single-indicator ---
    "sma_crossover": [
        ParamRange("fast_period", [5, 8, 10, 13, 15, 20]),
        ParamRange("slow_period", [20, 25, 30, 40, 50, 60, 80, 100]),
    ],
    "rsi_reversal": [
        ParamRange("rsi_period", [7, 10, 14, 21]),
        ParamRange("oversold", [20.0, 25.0, 30.0, 35.0]),
        ParamRange("overbought", [65.0, 70.0, 75.0, 80.0]),
    ],
    "macd_crossover": [
        ParamRange("fast_period", [8, 10, 12, 15]),
        ParamRange("slow_period", [21, 26, 30, 35]),
        ParamRange("signal_period", [7, 9, 12]),
    ],
    # --- Multi-indicator composite ---
    "sma_rsi": [
        ParamRange("fast_period", [5, 8, 10, 15, 20]),
        ParamRange("slow_period", [25, 30, 40, 50]),
        ParamRange("rsi_period", [10, 14, 21]),
        ParamRange("rsi_oversold", [30.0, 35.0, 40.0]),
        ParamRange("rsi_overbought", [60.0, 65.0, 70.0]),
    ],
    "macd_bbands": [
        ParamRange("macd_fast", [8, 10, 12]),
        ParamRange("macd_slow", [21, 26, 30]),
        ParamRange("macd_signal", [7, 9]),
        ParamRange("bb_period", [15, 20, 25]),
        ParamRange("bb_std", [1.5, 2.0, 2.5]),
    ],
    "triple_screen": [
        ParamRange("trend_ema", [30, 40, 50, 65]),
        ParamRange("macd_fast", [10, 12]),
        ParamRange("macd_slow", [24, 26]),
        ParamRange("macd_signal", [7, 9]),
        ParamRange("rsi_period", [10, 14]),
        ParamRange("rsi_buy_zone", [35.0, 40.0, 45.0]),
        ParamRange("rsi_sell_zone", [55.0, 60.0, 65.0]),
    ],
    "ema_confluence": [
        ParamRange("fast_ema", [5, 8, 10, 13]),
        ParamRange("medium_ema", [18, 21, 25]),
        ParamRange("slow_ema", [45, 55, 65]),
        ParamRange("rsi_period", [10, 14]),
        ParamRange("rsi_min_buy", [35.0, 40.0]),
        ParamRange("rsi_max_buy", [65.0, 70.0]),
    ],
    "bollinger_rsi": [
        ParamRange("bb_period", [15, 20, 25, 30]),
        ParamRange("bb_std", [1.5, 2.0, 2.5]),
        ParamRange("rsi_period", [10, 14, 21]),
        ParamRange("rsi_oversold", [20.0, 25.0, 30.0]),
        ParamRange("rsi_overbought", [70.0, 75.0, 80.0]),
    ],
}


def _generate_param_combos(param_ranges: List[ParamRange]) -> list[dict]:
    """Generate all combinations from parameter ranges."""
    if not param_ranges:
        return [{}]

    names = [pr.name for pr in param_ranges]
    value_lists = [pr.values for pr in param_ranges]

    combos = []
    for values in itertools.product(*value_lists):
        combo = dict(zip(names, values))
        combos.append(combo)
    return combos


def _is_valid_combo(strategy_type: str, params: dict) -> bool:
    """Filter out invalid parameter combinations."""
    # Fast period must be less than slow period
    if strategy_type in ("sma_crossover", "sma_rsi"):
        if params.get("fast_period", 0) >= params.get("slow_period", 1):
            return False
    if strategy_type in ("macd_crossover", "macd_bbands", "triple_screen"):
        fast_key = "macd_fast" if "macd_fast" in params else "fast_period"
        slow_key = "macd_slow" if "macd_slow" in params else "slow_period"
        if params.get(fast_key, 0) >= params.get(slow_key, 1):
            return False

    # Oversold must be less than overbought
    if strategy_type == "rsi_reversal":
        if params.get("oversold", 0) >= params.get("overbought", 100):
            return False
    if strategy_type in ("sma_rsi",):
        if params.get("rsi_oversold", 0) >= params.get("rsi_overbought", 100):
            return False
    if strategy_type in ("bollinger_rsi",):
        if params.get("rsi_oversold", 0) >= params.get("rsi_overbought", 100):
            return False
    if strategy_type in ("triple_screen",):
        if params.get("rsi_buy_zone", 0) >= params.get("rsi_sell_zone", 100):
            return False

    # EMA confluence: fast < medium < slow
    if strategy_type == "ema_confluence":
        if params.get("fast_ema", 0) >= params.get("medium_ema", 1):
            return False
        if params.get("medium_ema", 0) >= params.get("slow_ema", 1):
            return False
        if params.get("rsi_min_buy", 0) >= params.get("rsi_max_buy", 100):
            return False

    return True


def _get_score(result: BacktestResults, rank_by: str) -> float:
    """Extract the ranking metric from backtest results."""
    if rank_by == "sharpe_ratio":
        return result.sharpe_ratio
    elif rank_by == "profit_factor":
        return min(result.profit_factor, 100.0)  # cap infinity
    elif rank_by == "win_rate":
        return result.win_rate
    elif rank_by == "final_balance":
        return result.final_balance
    elif rank_by == "avg_monthly_return":
        return result.avg_monthly_return
    return result.sharpe_ratio


def run_optimization(
    config: OptimizationConfig,
    candles: list[dict],
    top_n: int = 10,
    progress_callback=None,
) -> list[OptimizationResult]:
    """Run grid search optimization over parameter combinations.

    Each individual backtest uses precomputed indicators (O(n) per run)
    so the full grid search stays fast without needing multiprocessing.

    Args:
        config: Optimization configuration
        candles: Pre-loaded candle data
        top_n: Number of top results to return
        progress_callback: Optional callable(current, total) for progress

    Returns:
        List of top OptimizationResult sorted by score descending
    """
    strategy = STRATEGIES.get(config.strategy_type)
    if strategy is None:
        raise ValueError(f"Unknown strategy: {config.strategy_type}")

    combos = _generate_param_combos(config.param_ranges)
    valid_combos = [
        c for c in combos
        if _is_valid_combo(config.strategy_type, c)
    ]

    if not valid_combos:
        raise ValueError("No valid parameter combinations to test.")

    logger.info(
        "Optimizer: testing %d combinations for %s on %s/%s (%d candles)",
        len(valid_combos),
        config.strategy_type,
        config.symbol,
        config.timeframe,
        len(candles),
    )

    pip_value = 0.01 if "JPY" in config.symbol.upper() else 0.0001
    results: list[OptimizationResult] = []
    total = len(valid_combos)

    # Clear any stale cancellation before starting
    reset_cancel()

    for idx, params in enumerate(valid_combos):
        # Check for cancellation every iteration
        if _cancel_event.is_set():
            logger.info("Optimizer cancelled after %d/%d combinations", idx, total)
            break

        if progress_callback:
            progress_callback(idx + 1, total)

        try:
            bt_config = BacktestConfig(
                symbol=config.symbol,
                timeframe=config.timeframe,
                strategy_type=config.strategy_type,
                parameters=params,
                start_date=config.start_date,
                end_date=config.end_date,
                initial_balance=config.initial_balance,
                pip_value=pip_value,
                lot_type=config.lot_type,
                risk_per_trade_pct=config.risk_per_trade_pct,
                sl_atr_multiplier=config.sl_atr_multiplier,
                tp_atr_multiplier=config.tp_atr_multiplier,
                monthly_max_loss_pct=config.monthly_max_loss_pct,
            )
            bt_result = run_backtest(bt_config, candles)

            if bt_result.total_trades == 0:
                continue

            score = _get_score(bt_result, config.rank_by)
            results.append(OptimizationResult(
                parameters=params,
                final_balance=bt_result.final_balance,
                total_trades=bt_result.total_trades,
                win_rate=bt_result.win_rate,
                max_drawdown=bt_result.max_drawdown,
                profit_factor=bt_result.profit_factor,
                sharpe_ratio=bt_result.sharpe_ratio,
                score=score,
                avg_monthly_return=bt_result.avg_monthly_return,
                max_monthly_drawdown=bt_result.max_monthly_drawdown,
                months_above_target=bt_result.months_above_target,
            ))
        except Exception as exc:
            logger.warning("Backtest failed for params %s: %s", params, exc)
            continue

    results.sort(key=lambda r: r.score, reverse=True)
    return results[:top_n]


# ---------------------------------------------------------------------------
# Analysis export
# ---------------------------------------------------------------------------


def format_results_for_analysis(
    config: OptimizationConfig,
    results: list[OptimizationResult],
    candle_count: int,
) -> str:
    """Format optimization results as markdown for Claude analysis."""
    lines = [
        "# Strategy Optimization Results",
        "",
        f"**Symbol:** {config.symbol}",
        f"**Timeframe:** {config.timeframe}",
        f"**Strategy:** {config.strategy_type}",
        f"**Period:** {config.start_date.strftime('%Y-%m-%d')} to "
        f"{config.end_date.strftime('%Y-%m-%d')}",
        f"**Candles:** {candle_count}",
        f"**Initial Balance:** ${config.initial_balance:,.2f}",
        f"**Ranked By:** {config.rank_by}",
        f"**Combinations Tested:** "
        f"{len(_generate_param_combos(config.param_ranges))}",
        f"**Profitable Results:** {len(results)}",
        "",
        "## Top Results",
        "",
        "| Rank | Parameters | Final Balance | Trades "
        "| Win Rate | Max DD | PF | Sharpe "
        "| Avg Mo. | Mo. DD | Mo.>5% |",
        "|------|-----------|--------------|--------"
        "|----------|--------|----|--------"
        "|---------|--------|--------|",
    ]

    for i, r in enumerate(results):
        ps = ", ".join(f"{k}={v}" for k, v in r.parameters.items())
        lines.append(
            f"| {i+1} | {ps} | ${r.final_balance:,.2f} "
            f"| {r.total_trades} | {r.win_rate:.1%} "
            f"| {r.max_drawdown:.1%} | {r.profit_factor:.2f} "
            f"| {r.sharpe_ratio:.2f} "
            f"| {r.avg_monthly_return:.1f}% "
            f"| {r.max_monthly_drawdown:.1f}% "
            f"| {r.months_above_target} |"
        )

    lines.extend([
        "",
        "## Analysis Prompt",
        "",
        "Please analyze these backtest results and suggest:",
        "1. Which parameter set looks most robust (not just highest return)?",
        "2. Are there signs of overfitting in any of the top results?",
        "3. What additional strategies or parameter ranges should I test?",
        "4. What risk management rules would you recommend?",
    ])

    return "\n".join(lines)
