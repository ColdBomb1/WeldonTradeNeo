from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import List

from pydantic import BaseModel, Field, field_validator

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
CONFIG_PATH = DATA_DIR / "config.json"


class MT5Config(BaseModel):
    terminal_path: str = ""
    login: int | None = None
    password: str = ""
    server: str = ""
    symbol_suffix: str = "!"


class AccountConfig(BaseModel):
    name: str
    account_type: str = "mt5"  # mt5, dxtrade, matchtrader, tradelocker
    login: str = ""
    password: str = ""
    server: str = ""
    path: str = ""
    api_key: str = ""
    enabled: bool = True


class ClaudeConfig(BaseModel):
    api_key: str = ""
    model: str = "claude-sonnet-4-20250514"
    review_model: str = "claude-opus-4-6"
    max_tokens: int = 4096
    review_max_tokens: int = 16000
    temperature: float = 0.3
    enabled: bool = False


class SignalConfig(BaseModel):
    enabled: bool = False
    scan_interval_sec: int = 300
    symbols: List[str] = Field(default_factory=list)
    timeframes: List[str] = Field(default_factory=lambda: ["1h", "4h"])
    strategies: List[str] = Field(default_factory=list)
    require_claude_confirmation: bool = True
    min_confidence: float = 0.6
    max_signals_per_day: int = 10
    cool_down_minutes: int = 60


class TradingBlackout(BaseModel):
    start: str = ""  # ISO datetime or cron-like "HH:MM"
    end: str = ""
    reason: str = ""
    recurring: bool = False  # True = daily/weekly pattern


class ExecutionConfig(BaseModel):
    mode: str = "paper"  # "paper" | "live"
    paused: bool = False  # manual pause — blocks all new trades

    # Prop firm risk management
    risk_per_trade_pct: float = 1.0  # % of balance risked per trade (1% = safe for 4% daily limit)
    max_open_positions: int = 3  # total across all pairs (1 per pair)
    max_daily_loss_pct: float = 4.0  # prop firm daily drawdown limit
    max_total_loss_pct: float = 8.0  # prop firm max account drawdown

    # Position sizing (used by trade manager when rules don't specify SL/TP)
    default_lot_type: str = "mini"
    sl_atr_multiplier: float = 2.5  # fallback SL — rules override this
    tp_atr_multiplier: float = 2.0  # fallback TP — rules override this
    position_sync_interval_sec: int = 30
    blackouts: List[TradingBlackout] = Field(default_factory=list)


class TrainingConfig(BaseModel):
    max_iterations: int = 5
    improvement_threshold: float = 0.05
    initial_balance: float = 10000.0
    use_claude_evaluation: bool = True
    use_claude_signals: bool = False
    rank_by: str = "sharpe_ratio"
    auto_deploy_threshold: float = 1.5


class NewsConfig(BaseModel):
    enabled: bool = False
    collection_interval_sec: int = 1800
    high_impact_buffer_minutes: int = 30
    sentiment_enabled: bool = False
    news_api_key: str = ""  # optional, for NewsAPI.org


class AppConfig(BaseModel):
    server_id: str = ""
    database_url: str = "postgresql+psycopg2://postgres:postgres@localhost:5432/weldon_trader"
    symbols: List[str] = Field(default_factory=lambda: ["EURUSD", "GBPUSD", "USDJPY"])
    poll_interval_sec: int = 30
    default_source: str = "yahoo"
    default_range: str = "1d"
    default_interval: str = "1m"
    request_timeout_sec: float = 15.0
    mt5: MT5Config = Field(default_factory=MT5Config)
    accounts: List[AccountConfig] = Field(default_factory=list)
    candle_collection_enabled: bool = True
    candle_collection_interval_sec: int = 60
    candle_timeframes: List[str] = Field(
        default_factory=lambda: ["1m", "5m", "15m", "1h", "4h", "1d"]
    )
    account_poll_interval_sec: float = 5.0
    claude: ClaudeConfig = Field(default_factory=ClaudeConfig)
    signals: SignalConfig = Field(default_factory=SignalConfig)
    execution: ExecutionConfig = Field(default_factory=ExecutionConfig)
    news: NewsConfig = Field(default_factory=NewsConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)

    @field_validator("symbols", mode="before")
    @classmethod
    def _normalize_symbols(cls, value):
        if value is None:
            return ["EURUSD"]

        raw_items: List[str]
        if isinstance(value, str):
            raw_items = [token.strip() for token in value.replace("\r", "").replace(",", "\n").split("\n")]
        else:
            raw_items = [str(token).strip() for token in value]

        normalized: List[str] = []
        for item in raw_items:
            if not item:
                continue
            pair = item.replace("/", "").replace("=X", "").strip().upper()
            if len(pair) == 6 and pair.isalpha():
                normalized.append(pair)

        if not normalized:
            return ["EURUSD"]

        seen = set()
        deduped: List[str] = []
        for pair in normalized:
            if pair in seen:
                continue
            seen.add(pair)
            deduped.append(pair)
        return deduped

    @field_validator("poll_interval_sec", mode="after")
    @classmethod
    def _clamp_poll_interval(cls, value: int) -> int:
        return max(5, min(300, int(value)))

    @field_validator("request_timeout_sec", mode="after")
    @classmethod
    def _clamp_timeout(cls, value: float) -> float:
        return max(5.0, min(60.0, float(value)))

    @field_validator("default_source", mode="after")
    @classmethod
    def _validate_default_source(cls, value: str) -> str:
        cleaned = (value or "yahoo").strip().lower()
        if cleaned not in {"yahoo", "mt5"}:
            return "yahoo"
        return cleaned

    @field_validator("candle_collection_interval_sec", mode="after")
    @classmethod
    def _clamp_candle_interval(cls, value: int) -> int:
        return max(30, min(3600, int(value)))


def load_config() -> AppConfig:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not CONFIG_PATH.exists():
        cfg = AppConfig()
        if not cfg.server_id:
            cfg.server_id = str(uuid.uuid4())
        save_config(cfg)
        return cfg

    data = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    cfg = AppConfig.model_validate(data)
    if not cfg.server_id:
        cfg.server_id = str(uuid.uuid4())
        save_config(cfg)
    return cfg


def save_config(cfg: AppConfig) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    CONFIG_PATH.write_text(
        json.dumps(cfg.model_dump(), indent=2),
        encoding="utf-8",
    )
