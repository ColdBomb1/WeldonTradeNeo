# WeldonTrader Neo

Forex trading platform: strategy backtesting, parameter optimization, and Claude-powered workflow integration.

## Stack

- **Backend:** FastAPI + Jinja2 templates, Python 3
- **Database:** PostgreSQL via synchronous SQLAlchemy 2.0 (psycopg2)
- **Config:** Pydantic models persisted as `data/config.json`
- **Frontend:** Server-rendered HTML + vanilla JS, IBM Plex fonts, cream/teal theme

## Run

```bash
uvicorn main:app --reload --port 8000
```

## Project Structure

```
main.py            # FastAPI app, startup/shutdown lifecycle, router registration
config.py          # Pydantic AppConfig, MT5Config, AccountConfig; JSON persistence
db.py              # SQLAlchemy engine + session factory (lazy singleton)
models/            # ORM: Candle, Tick, Trade*, AccountSnapshot, TradePlan, BacktestRun
services/          # Business logic: yahoo_fx, mt5_fx, candle_collector, indicator, strategy, backtest_engine, optimizer, claude_workflow
routers/           # API: marketdata, history, candles, indicators, accounts, settings, trade_plans
templates/         # Jinja2: base.html + dashboard, trade_plans, candles, charts, accounts, settings
static/            # app.css
data/              # config.json (runtime), created on first run
apps/              # Legacy microservice architecture (archived, not used)
```

## Key Patterns

- **Strategies:** BaseStrategy ABC with `STRATEGIES` registry dict. 8 strategies registered (sma_crossover, rsi_reversal, macd_crossover, sma_rsi, macd_bbands, triple_screen, ema_confluence, bollinger_rsi). All implement `precompute()` + `evaluate_at()` for O(n) backtests.
- **Indicators:** Pure Python math (no numpy). Functions return lists aligned with input; `None` for bars where indicator can't be computed.
- **Backtest:** Walk-forward, ATR-based SL/TP, monthly circuit breaker, lot-based position sizing, spread cost modeling.
- **Optimizer:** Grid search over parameter ranges. Claude suggests initial parameters, optimizer refines with tight +-1 ranges.
- **Candle collection:** Background `asyncio.create_task` on startup; uses `asyncio.to_thread()` for blocking Yahoo I/O.
- **Upsert:** PostgreSQL `ON CONFLICT DO UPDATE` on (symbol, timeframe, ts) unique constraint.
- **4h candles:** Aggregated from 1h Yahoo data since Yahoo doesn't provide 4h directly.
- **JPY pairs:** pip_value = 0.01 (vs 0.0001 for others).

## Database

PostgreSQL required. Tables auto-created via `Base.metadata.create_all()` on startup (no Alembic).

Default URL: `postgresql+psycopg2://postgres:postgres@localhost:5432/weldon_trader`

## Testing

No test suite yet.
