"""Walk-forward backtesting engine.

Simulates trading a strategy against historical candle data
and computes performance metrics.

Performance: uses strategy.precompute() + evaluate_at() to compute
indicators once per backtest instead of at every bar (O(n) vs O(n^2)).

Risk management features:
  - ATR-based stop-loss and take-profit placement
  - Lot-based position sizing (micro/mini/standard)
  - Monthly P&L tracking with circuit breaker
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional

from services import indicator_service
from services.strategy_service import SignalType, get_strategy

LOT_UNITS = {"micro": 1000, "mini": 10000, "standard": 100000}
MAX_SL_PIPS = {
    "EURUSD": 15, "GBPUSD": 20, "USDJPY": 25,
    "USDCHF": 15, "AUDUSD": 18, "NZDUSD": 20, "USDCAD": 18,
}
MAX_TP_PIPS = {
    "EURUSD": 25, "GBPUSD": 35, "USDJPY": 40,
    "USDCHF": 25, "AUDUSD": 30, "NZDUSD": 35, "USDCAD": 30,
}
MAX_LOTS = {
    "EURUSD": 3.0, "GBPUSD": 2.25, "USDJPY": 2.86,
    "USDCHF": 2.5, "AUDUSD": 2.25, "NZDUSD": 2.25, "USDCAD": 3.5,
}


@dataclass
class BacktestConfig:
    symbol: str
    timeframe: str
    strategy_type: str
    parameters: dict
    start_date: datetime
    end_date: datetime
    initial_balance: float = 10000.0
    spread_pips: float = 1.5
    pip_value: float = 0.0001  # 0.01 for JPY pairs

    # --- Risk management ---
    lot_type: str = "mini"  # "micro" (1K), "mini" (10K), "standard" (100K)
    risk_per_trade_pct: float = 1.0  # % of balance risked per trade
    sl_atr_period: int = 14
    sl_atr_multiplier: float = 1.5  # SL distance = ATR * multiplier
    tp_atr_multiplier: float = 2.0  # TP distance = ATR * multiplier
    max_open_positions: int = 1  # for future multi-position support
    monthly_max_loss_pct: float = 7.0  # circuit breaker: stop if month DD exceeds this
    entry_timing: str = "signal_close"  # "signal_close" | "next_open"
    slippage_pips: float = 0.0
    min_sl_pips: float = 0.0
    enforce_live_exit_policy: bool = False
    min_reward_risk: float = 1.5
    broker_lot_units: Optional[int] = None

    # Legacy field for backward compatibility with optimizer
    position_size_pct: float = 1.0


@dataclass
class TradeRecord:
    entry_ts: str
    exit_ts: str
    side: str
    entry_price: float
    exit_price: float
    stop_loss: float
    take_profit: float
    volume: float  # in lots
    pnl: float
    exit_type: str  # "sl", "tp", "signal", "end"


@dataclass
class BacktestResults:
    final_balance: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    max_drawdown: float
    profit_factor: float
    sharpe_ratio: float
    equity_curve: List[dict]
    trades: List[dict]

    # Monthly tracking
    monthly_pnl: List[dict] = field(default_factory=list)
    max_monthly_drawdown: float = 0.0
    avg_monthly_return: float = 0.0
    months_above_target: int = 0  # months with >= 5% return


@dataclass
class _Position:
    side: str
    entry_price: float
    entry_ts: str
    volume: float  # in lots
    stop_loss: float
    take_profit: float


def _parse_time(ts: str) -> datetime:
    """Parse ISO timestamp, handling both aware and naive formats."""
    try:
        return datetime.fromisoformat(ts)
    except ValueError:
        return datetime.strptime(ts, "%Y-%m-%d %H:%M:%S")


def _month_key(ts: str) -> str:
    """Extract YYYY-MM from timestamp string."""
    dt = _parse_time(ts)
    return f"{dt.year}-{dt.month:02d}"


def run_backtest(config: BacktestConfig, candles: List[dict]) -> BacktestResults:
    """Run a backtest over provided candle data.

    candles: list of dicts with time, open, high, low, close, volume.
    Already filtered to the requested date range.
    """
    strategy = get_strategy(config.strategy_type)
    if strategy is None:
        raise ValueError(f"Unknown strategy: {config.strategy_type}")

    if not candles or len(candles) < 2:
        return _empty_results(config.initial_balance)

    balance = config.initial_balance
    position: Optional[_Position] = None
    trades: List[TradeRecord] = []
    equity_curve: List[dict] = []
    peak_equity = balance
    max_drawdown = 0.0

    spread_cost = config.spread_pips * config.pip_value
    lot_units = LOT_UNITS.get(config.lot_type, 10000)
    pnl_lot_units = int(config.broker_lot_units or lot_units)

    # Precompute strategy indicators (O(n) total)
    precomputed = strategy.precompute(candles, config.parameters)
    use_fast_path = bool(precomputed)

    # Precompute ATR for SL/TP placement
    atr_values = indicator_service.atr(candles, config.sl_atr_period)
    entry_timing = config.entry_timing if config.entry_timing in {"signal_close", "next_open"} else "signal_close"
    slippage = max(0.0, float(config.slippage_pips)) * config.pip_value

    def _entry_price(raw_price: float, side: SignalType) -> float:
        if side == SignalType.BUY:
            return raw_price + slippage
        if side == SignalType.SELL:
            return raw_price - slippage
        return raw_price

    def _exit_price(raw_price: float, side: str) -> float:
        if side == "buy":
            return raw_price - slippage
        if side == "sell":
            return raw_price + slippage
        return raw_price

    def _account_conversion(price: float) -> float:
        return _quote_to_account_rate(config.symbol, price)

    def _build_position(signal, raw_entry_price: float, entry_time: str, atr_index: int) -> _Position:
        entry_price = _entry_price(raw_entry_price, signal.type)
        signal_price = float(signal.price or raw_entry_price)
        cur_atr = atr_values[atr_index] if atr_index < len(atr_values) else None

        if cur_atr is not None and cur_atr > 0:
            sl_distance = cur_atr * config.sl_atr_multiplier
            tp_distance = cur_atr * config.tp_atr_multiplier
        else:
            sl_distance = config.spread_pips * config.pip_value * 20
            tp_distance = config.spread_pips * config.pip_value * 30

        if signal.stop_loss is not None:
            sl_distance = abs(signal_price - signal.stop_loss)
        if signal.take_profit is not None:
            tp_distance = abs(signal_price - signal.take_profit)

        if signal.type == SignalType.BUY:
            stop_loss = entry_price - sl_distance
            take_profit = entry_price + tp_distance
        else:
            stop_loss = entry_price + sl_distance
            take_profit = entry_price - tp_distance

        sizing_sl_distance = abs(entry_price - stop_loss)
        if config.enforce_live_exit_policy:
            risk_dist = abs(entry_price - stop_loss)
            reward_dist = abs(entry_price - take_profit)
            if risk_dist > 0 and reward_dist < float(config.min_reward_risk) * risk_dist:
                take_profit = (
                    entry_price + float(config.min_reward_risk) * risk_dist
                    if signal.type == SignalType.BUY
                    else entry_price - float(config.min_reward_risk) * risk_dist
                )

            pip_val = config.pip_value
            sl_cap = MAX_SL_PIPS.get(config.symbol.upper(), 20) * pip_val
            tp_cap = MAX_TP_PIPS.get(config.symbol.upper(), 30) * pip_val
            if abs(entry_price - stop_loss) > sl_cap:
                stop_loss = entry_price - sl_cap if signal.type == SignalType.BUY else entry_price + sl_cap
            if abs(entry_price - take_profit) > tp_cap:
                take_profit = entry_price + tp_cap if signal.type == SignalType.BUY else entry_price - tp_cap

        sl_distance = sizing_sl_distance if config.enforce_live_exit_policy else abs(entry_price - stop_loss)
        sl_pips = sl_distance / config.pip_value
        if sl_pips <= 0:
            sl_pips = 1.0
        if config.enforce_live_exit_policy:
            sl_pips = max(sl_pips, 10.0)
        sl_pips = max(sl_pips, float(config.min_sl_pips))

        risk_amount = balance * (config.risk_per_trade_pct / 100.0)
        pip_value_per_lot = config.pip_value * lot_units
        volume_lots = risk_amount / (sl_pips * pip_value_per_lot)
        if config.enforce_live_exit_policy:
            if "JPY" in config.symbol.upper():
                volume_lots *= 159.0 / 2
            volume_lots = min(volume_lots, MAX_LOTS.get(config.symbol.upper(), 3.0))
        volume_lots = max(round(volume_lots, 2), 0.01)

        return _Position(
            side=signal.type.value,
            entry_price=entry_price,
            entry_ts=entry_time,
            volume=volume_lots,
            stop_loss=stop_loss,
            take_profit=take_profit,
        )

    # Monthly tracking
    monthly_start_balance: dict[str, float] = {}  # month -> balance at start
    monthly_trades: dict[str, list] = {}
    month_stopped: set[str] = set()  # months where circuit breaker triggered
    pending_entry: tuple[object, int] | None = None

    for i in range(1, len(candles)):
        current = candles[i]
        current_price = current["close"]
        bar_time = current["time"]
        current_month = _month_key(bar_time)

        # Track month start balance
        if current_month not in monthly_start_balance:
            monthly_start_balance[current_month] = balance
            monthly_trades[current_month] = []

        if entry_timing == "next_open" and pending_entry is not None and position is None:
            if current_month not in month_stopped and balance > 0:
                pending_signal, signal_index = pending_entry
                position = _build_position(pending_signal, current["open"], bar_time, signal_index)
            pending_entry = None

        # Strategy signal
        if use_fast_path:
            signal = strategy.evaluate_at(i, candles, precomputed, config.parameters)
        else:
            signal = strategy.evaluate(candles[: i + 1], config.parameters)

        # --- Check SL/TP hits on open positions ---
        if position is not None:
            exit_price = None
            exit_type = None

            # Check SL hit (worst-case: check SL before TP)
            if position.side == "buy":
                if current["low"] <= position.stop_loss:
                    exit_price = _exit_price(position.stop_loss, position.side)
                    exit_type = "sl"
                elif current["high"] >= position.take_profit:
                    exit_price = _exit_price(position.take_profit, position.side)
                    exit_type = "tp"
            else:  # sell
                if current["high"] >= position.stop_loss:
                    exit_price = _exit_price(position.stop_loss, position.side)
                    exit_type = "sl"
                elif current["low"] <= position.take_profit:
                    exit_price = _exit_price(position.take_profit, position.side)
                    exit_type = "tp"

            # Close on opposing signal (if SL/TP didn't trigger)
            if exit_price is None:
                if (position.side == "buy" and signal.type == SignalType.SELL) or \
                   (position.side == "sell" and signal.type == SignalType.BUY):
                    exit_price = _exit_price(current_price, position.side)
                    exit_type = "signal"

            if exit_price is not None:
                pnl = _compute_pnl(
                    position.side, position.entry_price, exit_price,
                    position.volume, pnl_lot_units, spread_cost, exit_type,
                    _account_conversion(exit_price),
                )
                balance += pnl
                trade = TradeRecord(
                    entry_ts=position.entry_ts,
                    exit_ts=bar_time,
                    side=position.side,
                    entry_price=position.entry_price,
                    exit_price=exit_price,
                    stop_loss=position.stop_loss,
                    take_profit=position.take_profit,
                    volume=position.volume,
                    pnl=round(pnl, 2),
                    exit_type=exit_type,
                )
                trades.append(trade)
                monthly_trades.setdefault(current_month, []).append(trade)
                position = None

        # --- Monthly circuit breaker check ---
        month_start_bal = monthly_start_balance.get(current_month, balance)
        if month_start_bal > 0:
            month_dd = (month_start_bal - balance) / month_start_bal
            if month_dd >= config.monthly_max_loss_pct / 100.0:
                month_stopped.add(current_month)

        # --- Open new position ---
        can_open = (
            position is None
            and signal.type in (SignalType.BUY, SignalType.SELL)
            and current_month not in month_stopped
            and balance > 0
        )

        if can_open:
            if entry_timing == "next_open":
                pending_entry = (signal, i)
            else:
                position = _build_position(signal, current_price, bar_time, i)

        # --- Track equity ---
        equity = balance
        if position is not None:
            if position.side == "buy":
                unrealized = (
                    (current_price - position.entry_price)
                    * position.volume
                    * pnl_lot_units
                    * _account_conversion(current_price)
                )
            else:
                unrealized = (
                    (position.entry_price - current_price)
                    * position.volume
                    * pnl_lot_units
                    * _account_conversion(current_price)
                )
            equity += unrealized

        equity_curve.append({"time": bar_time, "equity": round(equity, 2)})

        if equity > peak_equity:
            peak_equity = equity
        dd = (peak_equity - equity) / peak_equity if peak_equity > 0 else 0
        if dd > max_drawdown:
            max_drawdown = dd

    # Close any remaining position at last price
    if position is not None and candles:
        last = candles[-1]
        exit_price = _exit_price(last["close"], position.side)
        pnl = _compute_pnl(
            position.side, position.entry_price, exit_price,
            position.volume, pnl_lot_units, spread_cost, "end",
            _account_conversion(exit_price),
        )
        balance += pnl
        trade = TradeRecord(
            entry_ts=position.entry_ts,
            exit_ts=last["time"],
            side=position.side,
            entry_price=position.entry_price,
            exit_price=exit_price,
            stop_loss=position.stop_loss,
            take_profit=position.take_profit,
            volume=position.volume,
            pnl=round(pnl, 2),
            exit_type="end",
        )
        trades.append(trade)
        last_month = _month_key(last["time"])
        monthly_trades.setdefault(last_month, []).append(trade)

    # --- Compute metrics ---
    total = len(trades)
    winners = [t for t in trades if t.pnl > 0]
    losers = [t for t in trades if t.pnl <= 0]
    win_rate = len(winners) / total if total > 0 else 0.0

    gross_profit = sum(t.pnl for t in winners) if winners else 0.0
    gross_loss = abs(sum(t.pnl for t in losers)) if losers else 0.0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else (
        float("inf") if gross_profit > 0 else 0.0
    )

    # Sharpe ratio (annualized)
    sharpe_ratio = _compute_sharpe(equity_curve)

    # Monthly P&L
    monthly_pnl = _compute_monthly_pnl(
        monthly_start_balance, monthly_trades, config.initial_balance,
    )
    max_monthly_dd = 0.0
    monthly_returns = []
    months_above_5 = 0
    for m in monthly_pnl:
        monthly_returns.append(m["pct"])
        if m["pct"] < 0 and abs(m["pct"]) > max_monthly_dd:
            max_monthly_dd = abs(m["pct"])
        if m["pct"] >= 5.0:
            months_above_5 += 1
    avg_monthly = sum(monthly_returns) / len(monthly_returns) if monthly_returns else 0.0

    return BacktestResults(
        final_balance=round(balance, 2),
        total_trades=total,
        winning_trades=len(winners),
        losing_trades=len(losers),
        win_rate=round(win_rate, 4),
        max_drawdown=round(max_drawdown, 4),
        profit_factor=round(profit_factor, 4) if profit_factor != float("inf") else 999.99,
        sharpe_ratio=round(sharpe_ratio, 4),
        equity_curve=equity_curve,
        trades=[
            {
                "entry_ts": t.entry_ts,
                "exit_ts": t.exit_ts,
                "side": t.side,
                "entry_price": t.entry_price,
                "exit_price": t.exit_price,
                "stop_loss": t.stop_loss,
                "take_profit": t.take_profit,
                "volume": t.volume,
                "pnl": t.pnl,
                "exit_type": t.exit_type,
            }
            for t in trades
        ],
        monthly_pnl=monthly_pnl,
        max_monthly_drawdown=round(max_monthly_dd, 2),
        avg_monthly_return=round(avg_monthly, 2),
        months_above_target=months_above_5,
    )


def _compute_pnl(
    side: str,
    entry_price: float,
    exit_price: float,
    volume_lots: float,
    lot_units: int,
    spread_cost: float,
    exit_type: str,
    account_conversion: float = 1.0,
) -> float:
    """Compute P&L for a closed position in account currency.

    P&L = price_diff * volume_lots * lot_units
    Spread is deducted on entry (already factored into the price difference).
    """
    if side == "buy":
        raw_pnl = (exit_price - entry_price) * volume_lots * lot_units
    else:
        raw_pnl = (entry_price - exit_price) * volume_lots * lot_units

    # Spread cost: applied once per round trip
    spread_total = spread_cost * volume_lots * lot_units
    return (raw_pnl - spread_total) * account_conversion


def _quote_to_account_rate(symbol: str, price: float) -> float:
    """Convert quote-currency P&L into USD account currency for USD majors."""
    sym = symbol.upper()
    if sym.endswith("USD") or price <= 0:
        return 1.0
    if sym.startswith("USD"):
        return 1.0 / price
    return 1.0


def _compute_sharpe(equity_curve: List[dict]) -> float:
    """Annualized Sharpe ratio from equity curve."""
    if len(equity_curve) <= 1:
        return 0.0

    returns = []
    for j in range(1, len(equity_curve)):
        prev_eq = equity_curve[j - 1]["equity"]
        curr_eq = equity_curve[j]["equity"]
        if prev_eq > 0:
            returns.append((curr_eq - prev_eq) / prev_eq)

    if not returns:
        return 0.0

    avg_ret = sum(returns) / len(returns)
    if len(returns) > 1:
        std_ret = math.sqrt(sum((r - avg_ret) ** 2 for r in returns) / len(returns))
    else:
        std_ret = 0.0

    return (avg_ret / std_ret * math.sqrt(252)) if std_ret > 0 else 0.0


def _compute_monthly_pnl(
    monthly_start_balance: dict[str, float],
    monthly_trades: dict[str, list],
    initial_balance: float,
) -> List[dict]:
    """Compute monthly P&L breakdown."""
    months = sorted(monthly_start_balance.keys())
    result = []
    running_balance = initial_balance

    for month in months:
        start_bal = monthly_start_balance[month]
        month_pnl = sum(t.pnl for t in monthly_trades.get(month, []))
        end_bal = start_bal + month_pnl
        pct = (month_pnl / start_bal * 100) if start_bal > 0 else 0.0
        trade_count = len(monthly_trades.get(month, []))

        result.append({
            "month": month,
            "start_balance": round(start_bal, 2),
            "pnl": round(month_pnl, 2),
            "pct": round(pct, 2),
            "trades": trade_count,
        })
        running_balance = end_bal

    return result


def _empty_results(initial_balance: float) -> BacktestResults:
    """Return empty results when there's not enough data."""
    return BacktestResults(
        final_balance=initial_balance,
        total_trades=0,
        winning_trades=0,
        losing_trades=0,
        win_rate=0.0,
        max_drawdown=0.0,
        profit_factor=0.0,
        sharpe_ratio=0.0,
        equity_curve=[],
        trades=[],
        monthly_pnl=[],
        max_monthly_drawdown=0.0,
        avg_monthly_return=0.0,
        months_above_target=0,
    )
