from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Request
from fastapi.templating import Jinja2Templates

from config import load_config

router = APIRouter(tags=["operations"])

BASE_DIR = Path(__file__).resolve().parent.parent
TEMPLATES = Jinja2Templates(directory=str(BASE_DIR / "templates"))


@router.get("/operations")
def operations_page(request: Request):
    cfg = load_config()
    return TEMPLATES.TemplateResponse(
        "operations.html",
        {
            "request": request,
            "symbols": cfg.symbols,
            "signals_enabled": cfg.signals.enabled,
            "execution_mode": cfg.execution.mode,
            "paused": cfg.execution.paused,
        },
    )
