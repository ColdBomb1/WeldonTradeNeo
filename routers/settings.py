from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Request
from fastapi.responses import RedirectResponse
from fastapi.templating import Jinja2Templates

from config import CONFIG_PATH, load_config, save_config
from services import mt5_fx_service, yahoo_fx_service

router = APIRouter(tags=["settings"])

BASE_DIR = Path(__file__).resolve().parent.parent
TEMPLATES = Jinja2Templates(directory=str(BASE_DIR / "templates"))

SUPPORTED_SOURCES = ("yahoo", "mt5")
SUPPORTED_RANGES = yahoo_fx_service.SUPPORTED_RANGES
SUPPORTED_INTERVALS = tuple(
    list(yahoo_fx_service.SUPPORTED_INTERVALS)
    + [v for v in mt5_fx_service.SUPPORTED_INTERVALS if v not in yahoo_fx_service.SUPPORTED_INTERVALS]
)


@router.get("/settings")
def settings_page(request: Request):
    cfg = load_config()
    return TEMPLATES.TemplateResponse(
        "settings.html",
        {
            "request": request,
            "config": cfg,
            "config_path": str(CONFIG_PATH),
            "supported_sources": SUPPORTED_SOURCES,
            "supported_ranges": SUPPORTED_RANGES,
            "supported_intervals": SUPPORTED_INTERVALS,
            "mt5_available": mt5_fx_service.is_available(),
        },
    )


def _parse_symbols(value: str) -> list[str]:
    raw = value.replace("\r", "").replace(",", "\n").split("\n")
    cleaned = []
    seen = set()
    for token in raw:
        pair = token.strip().upper().replace("/", "").replace("=X", "")
        if len(pair) != 6 or not pair.isalpha():
            continue
        if pair in seen:
            continue
        seen.add(pair)
        cleaned.append(pair)
    return cleaned


@router.post("/settings")
async def update_settings(request: Request):
    form = await request.form()
    cfg = load_config()

    # Market data settings
    parsed_symbols = _parse_symbols(str(form.get("symbols", "")))
    if parsed_symbols:
        cfg.symbols = parsed_symbols

    poll_raw = str(form.get("poll_interval_sec", cfg.poll_interval_sec)).strip()
    try:
        cfg.poll_interval_sec = max(5, min(300, int(poll_raw)))
    except ValueError:
        pass

    chosen_range = str(form.get("default_range", cfg.default_range)).strip()
    if chosen_range in SUPPORTED_RANGES:
        cfg.default_range = chosen_range

    chosen_interval = str(form.get("default_interval", cfg.default_interval)).strip()
    if chosen_interval in SUPPORTED_INTERVALS:
        cfg.default_interval = chosen_interval

    chosen_source = str(form.get("default_source", cfg.default_source)).strip().lower()
    if chosen_source in SUPPORTED_SOURCES:
        cfg.default_source = chosen_source

    timeout_raw = str(form.get("request_timeout_sec", cfg.request_timeout_sec)).strip()
    try:
        cfg.request_timeout_sec = max(5.0, min(60.0, float(timeout_raw)))
    except ValueError:
        pass

    # MT5 settings
    cfg.mt5.terminal_path = str(form.get("mt5_terminal_path", cfg.mt5.terminal_path)).strip()
    mt5_login_raw = str(form.get("mt5_login", "")).strip()
    cfg.mt5.login = int(mt5_login_raw) if mt5_login_raw else None
    cfg.mt5.password = str(form.get("mt5_password", cfg.mt5.password)).strip()
    cfg.mt5.server = str(form.get("mt5_server", cfg.mt5.server)).strip()
    cfg.mt5.symbol_suffix = str(form.get("mt5_symbol_suffix", cfg.mt5.symbol_suffix)).strip()

    # Database
    db_url = str(form.get("database_url", cfg.database_url)).strip()
    if db_url:
        cfg.database_url = db_url

    # Candle collection
    cfg.candle_collection_enabled = bool(form.get("candle_collection_enabled"))
    candle_interval_raw = str(form.get("candle_collection_interval_sec", cfg.candle_collection_interval_sec)).strip()
    try:
        cfg.candle_collection_interval_sec = max(30, min(3600, int(candle_interval_raw)))
    except ValueError:
        pass

    # AI provider
    provider = str(form.get("ai_provider", cfg.ai.provider)).strip().lower()
    if provider in {"claude", "ollama", "openai_compatible"}:
        cfg.ai.provider = provider
    cfg.ai.base_url = str(form.get("ai_base_url", cfg.ai.base_url)).strip() or cfg.ai.base_url
    cfg.ai.api_key = str(form.get("ai_api_key", cfg.ai.api_key)).strip()
    cfg.ai.model = str(form.get("ai_model", cfg.ai.model)).strip() or cfg.ai.model
    cfg.ai.review_model = str(form.get("ai_review_model", cfg.ai.review_model)).strip() or cfg.ai.review_model
    try:
        cfg.ai.temperature = max(0.0, min(1.0, float(form.get("ai_temperature", cfg.ai.temperature))))
    except (ValueError, TypeError):
        pass
    try:
        cfg.ai.timeout_sec = max(5.0, min(600.0, float(form.get("ai_timeout_sec", cfg.ai.timeout_sec))))
    except (ValueError, TypeError):
        pass
    cfg.ai.enabled = bool(form.get("ai_enabled"))

    # Keep the legacy Claude config in sync when Claude is the selected provider.
    if cfg.ai.provider == "claude":
        cfg.claude.enabled = cfg.ai.enabled
        cfg.claude.api_key = cfg.ai.api_key
        cfg.claude.model = cfg.ai.model
        cfg.claude.review_model = cfg.ai.review_model
        cfg.claude.temperature = cfg.ai.temperature
    else:
        cfg.claude.enabled = False

    from services import ai_service
    ai_service.reset_client()

    # Signal generation
    cfg.signals.enabled = bool(form.get("signals_enabled"))
    try:
        cfg.signals.scan_interval_sec = max(30, min(3600, int(form.get("signals_scan_interval_sec", cfg.signals.scan_interval_sec))))
    except (ValueError, TypeError):
        pass
    tf_raw = str(form.get("signals_timeframes", "")).strip()
    if tf_raw:
        cfg.signals.timeframes = [t.strip() for t in tf_raw.split(",") if t.strip()]
    try:
        cfg.signals.min_confidence = max(0.0, min(1.0, float(form.get("signals_min_confidence", cfg.signals.min_confidence))))
    except (ValueError, TypeError):
        pass
    try:
        cfg.signals.max_signals_per_day = max(0, min(100, int(form.get("signals_max_per_day", cfg.signals.max_signals_per_day))))
    except (ValueError, TypeError):
        pass
    try:
        cfg.signals.cool_down_minutes = max(5, min(1440, int(form.get("signals_cool_down_minutes", cfg.signals.cool_down_minutes))))
    except (ValueError, TypeError):
        pass
    cfg.signals.require_model_review = bool(form.get("signals_require_model"))
    cfg.signals.require_claude_confirmation = cfg.signals.require_model_review

    # News & Calendar
    cfg.news.enabled = bool(form.get("news_enabled"))
    try:
        cfg.news.collection_interval_sec = max(300, min(7200, int(form.get("news_collection_interval_sec", cfg.news.collection_interval_sec))))
    except (ValueError, TypeError):
        pass
    try:
        cfg.news.high_impact_buffer_minutes = max(5, min(120, int(form.get("news_buffer_minutes", cfg.news.high_impact_buffer_minutes))))
    except (ValueError, TypeError):
        pass
    cfg.news.sentiment_enabled = bool(form.get("news_sentiment_enabled"))

    # Risk management (prop firm)
    try:
        cfg.execution.risk_per_trade_pct = max(0.1, min(5.0, float(form.get("execution_risk_pct", cfg.execution.risk_per_trade_pct))))
    except (ValueError, TypeError):
        pass
    try:
        cfg.execution.min_risk_per_trade_pct = max(0.0, min(cfg.execution.risk_per_trade_pct, float(form.get("execution_min_risk_pct", cfg.execution.min_risk_per_trade_pct))))
    except (ValueError, TypeError):
        pass
    try:
        cfg.execution.max_daily_loss_pct = max(0.1, min(10.0, float(form.get("execution_max_daily_loss", cfg.execution.max_daily_loss_pct))))
    except (ValueError, TypeError):
        pass
    try:
        cfg.execution.max_total_loss_pct = max(0.1, min(20.0, float(form.get("execution_max_total_loss", cfg.execution.max_total_loss_pct))))
    except (ValueError, TypeError):
        pass
    try:
        cfg.execution.account_start_balance = max(0.0, float(form.get("execution_account_start_balance", cfg.execution.account_start_balance)))
    except (ValueError, TypeError):
        pass
    try:
        cfg.execution.max_relative_drawdown_pct = max(0.1, min(20.0, float(form.get("execution_max_relative_drawdown", cfg.execution.max_relative_drawdown_pct))))
    except (ValueError, TypeError):
        pass
    try:
        cfg.execution.drawdown_hard_stop_buffer_pct = max(0.0, min(5.0, float(form.get("execution_drawdown_buffer", cfg.execution.drawdown_hard_stop_buffer_pct))))
    except (ValueError, TypeError):
        pass
    try:
        cfg.execution.risk_reduction_drawdown_pct = max(0.0, min(20.0, float(form.get("execution_risk_reduction_drawdown", cfg.execution.risk_reduction_drawdown_pct))))
    except (ValueError, TypeError):
        pass
    cfg.execution.allow_drawdown_override = bool(form.get("execution_allow_drawdown_override"))
    try:
        cfg.execution.max_aggregate_open_risk_pct = max(0.0, min(10.0, float(form.get("execution_max_aggregate_risk", cfg.execution.max_aggregate_open_risk_pct))))
    except (ValueError, TypeError):
        pass
    try:
        cfg.execution.max_currency_open_risk_pct = max(0.0, min(10.0, float(form.get("execution_max_currency_risk", cfg.execution.max_currency_open_risk_pct))))
    except (ValueError, TypeError):
        pass
    try:
        cfg.execution.max_same_currency_positions = max(0, min(10, int(form.get("execution_max_same_currency_positions", cfg.execution.max_same_currency_positions))))
    except (ValueError, TypeError):
        pass
    try:
        cfg.execution.max_open_positions = max(1, min(10, int(form.get("execution_max_positions", cfg.execution.max_open_positions))))
    except (ValueError, TypeError):
        pass

    save_config(cfg)
    return RedirectResponse(url="/settings", status_code=303)
