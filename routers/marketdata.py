from __future__ import annotations

from pathlib import Path

import httpx
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from fastapi.templating import Jinja2Templates

from config import CONFIG_PATH, load_config, save_config
from services import mt5_fx_service, yahoo_fx_service

router = APIRouter(tags=["marketdata"])

BASE_DIR = Path(__file__).resolve().parent.parent
TEMPLATES = Jinja2Templates(directory=str(BASE_DIR / "templates"))

SUPPORTED_SOURCES = ("yahoo", "mt5")
SUPPORTED_RANGES = yahoo_fx_service.SUPPORTED_RANGES
SUPPORTED_INTERVALS = tuple(
    list(yahoo_fx_service.SUPPORTED_INTERVALS)
    + [v for v in mt5_fx_service.SUPPORTED_INTERVALS if v not in yahoo_fx_service.SUPPORTED_INTERVALS]
)


@router.get("/dashboard")
def dashboard(request: Request):
    cfg = load_config()
    return TEMPLATES.TemplateResponse(
        "dashboard.html",
        {
            "request": request,
            "config": cfg,
        },
    )


@router.get("/api/chart-data")
def chart_data(
    symbol: str,
    range: str | None = None,
    interval: str | None = None,
    source: str | None = None,
) -> JSONResponse:
    cfg = load_config()
    selected_source = (source or cfg.default_source).strip().lower()
    selected_range = range or cfg.default_range
    selected_interval = interval or cfg.default_interval

    if selected_source not in SUPPORTED_SOURCES:
        return JSONResponse({"error": f"Unsupported source '{selected_source}'."}, status_code=400)

    try:
        if selected_source == "mt5":
            data = mt5_fx_service.get_chart_data(
                symbol=symbol,
                range_name=selected_range,
                interval=selected_interval,
                cfg=cfg.mt5,
            )
        else:
            data = yahoo_fx_service.get_chart_data(
                symbol=symbol,
                range_name=selected_range,
                interval=selected_interval,
                timeout_sec=cfg.request_timeout_sec,
            )
    except ValueError as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)
    except httpx.HTTPError as exc:
        return JSONResponse({"error": f"Data source request failed: {exc}"}, status_code=502)
    except Exception as exc:
        return JSONResponse({"error": f"Unexpected error: {exc}"}, status_code=500)

    return JSONResponse(data)


@router.get("/api/config")
def api_config() -> JSONResponse:
    cfg = load_config()
    return JSONResponse(
        {
            "symbols": cfg.symbols,
            "default_source": cfg.default_source,
            "poll_interval_sec": cfg.poll_interval_sec,
            "default_range": cfg.default_range,
            "default_interval": cfg.default_interval,
            "request_timeout_sec": cfg.request_timeout_sec,
            "supported_sources": list(SUPPORTED_SOURCES),
            "supported_ranges": list(SUPPORTED_RANGES),
            "supported_intervals": list(SUPPORTED_INTERVALS),
            "mt5_supported_intervals": list(mt5_fx_service.SUPPORTED_INTERVALS),
            "mt5_available": mt5_fx_service.is_available(),
            "mt5": {
                "terminal_path": cfg.mt5.terminal_path,
                "login": cfg.mt5.login,
                "server": cfg.mt5.server,
                "symbol_suffix": cfg.mt5.symbol_suffix,
            },
        }
    )
