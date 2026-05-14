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
    model: str = "claude-sonnet-4-6"
    # review_model previously was an Opus variant for deep-review calls. It is
    # now forced to Sonnet — see services/claude_service._enforce_sonnet_only.
    # Any value other than Sonnet is rejected by the validator below.
    review_model: str = "claude-sonnet-4-6"
    max_tokens: int = 4096
    review_max_tokens: int = 4096
    temperature: float = 0.3
    enabled: bool = False

    @field_validator("model", "review_model", mode="after")
    @classmethod
    def _sonnet_only(cls, value: str) -> str:
        # Hard lock: only Sonnet models allowed. Anything else — including an
        # old config.json with Opus in it — gets rewritten at load time. This
        # pairs with the runtime guard in claude_service._enforce_sonnet_only.
        allowed = "claude-sonnet-4-6"
        if value != allowed:
            import logging
            logging.getLogger(__name__).error(
                "Claude model %r is not allowed; forcing to %s", value, allowed,
            )
            return allowed
        return value


class AIConfig(BaseModel):
    enabled: bool = False
    provider: str = "claude"  # claude, openai_compatible, ollama
    base_url: str = "http://127.0.0.1:11434"
    api_key: str = ""
    model: str = "claude-sonnet-4-6"
    review_model: str = "claude-sonnet-4-6"
    max_tokens: int = 2048
    review_max_tokens: int = 4096
    temperature: float = 0.2
    timeout_sec: float = 120.0

    @field_validator("provider", mode="after")
    @classmethod
    def _validate_provider(cls, value: str) -> str:
        cleaned = (value or "claude").strip().lower()
        if cleaned not in {"claude", "openai_compatible", "ollama"}:
            return "claude"
        return cleaned


class SignalConfig(BaseModel):
    enabled: bool = False
    scan_interval_sec: int = 300
    symbols: List[str] = Field(default_factory=list)
    timeframes: List[str] = Field(default_factory=lambda: ["1h", "4h"])
    strategies: List[str] = Field(default_factory=list)
    require_claude_confirmation: bool = True
    require_model_review: bool = True
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
    risk_per_trade_pct: float = 0.5  # % of balance risked per trade before drawdown scaling
    min_risk_per_trade_pct: float = 0.25
    max_open_positions: int = 3  # total across all pairs (1 per pair)
    max_daily_loss_pct: float = 2.0
    max_total_loss_pct: float = 7.0
    account_start_balance: float = 0.0
    max_relative_drawdown_pct: float = 7.0
    drawdown_hard_stop_buffer_pct: float = 0.25
    risk_reduction_drawdown_pct: float = 4.0
    max_aggregate_open_risk_pct: float = 2.0

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
    use_model_evaluation: bool = True
    use_model_signals: bool = False
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
    ai: AIConfig = Field(default_factory=AIConfig)
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
    _migrate_legacy_config(cfg, data)
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


def _migrate_legacy_config(cfg: AppConfig, raw_data: dict) -> None:
    """Bridge old Claude-specific config to the provider-neutral AI config."""
    changed = False

    if "ai" not in raw_data:
        cfg.ai.enabled = cfg.claude.enabled
        cfg.ai.provider = "claude"
        cfg.ai.api_key = cfg.claude.api_key
        cfg.ai.model = cfg.claude.model
        cfg.ai.review_model = cfg.claude.review_model
        cfg.ai.max_tokens = cfg.claude.max_tokens
        cfg.ai.review_max_tokens = cfg.claude.review_max_tokens
        cfg.ai.temperature = cfg.claude.temperature
        changed = True

    signals_raw = raw_data.get("signals", {})
    if "require_model_review" not in signals_raw:
        cfg.signals.require_model_review = cfg.signals.require_claude_confirmation
        changed = True

    training_raw = raw_data.get("training", {})
    if "use_model_evaluation" not in training_raw:
        cfg.training.use_model_evaluation = cfg.training.use_claude_evaluation
        changed = True
    if "use_model_signals" not in training_raw:
        cfg.training.use_model_signals = cfg.training.use_claude_signals
        changed = True

    if changed:
        save_config(cfg)
