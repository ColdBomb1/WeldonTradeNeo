from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from fastapi.templating import Jinja2Templates

from config import load_config
from db import get_session
from models.signal import Signal
from services.strategy_service import list_strategies

router = APIRouter(tags=["signals"])

BASE_DIR = Path(__file__).resolve().parent.parent
TEMPLATES = Jinja2Templates(directory=str(BASE_DIR / "templates"))


def _signal_to_dict(sig: Signal) -> dict:
    analysis = sig.claude_analysis or {}
    reject_reason = analysis.get("reject_reason", "") if isinstance(analysis, dict) else ""
    return {
        "id": sig.id,
        "symbol": sig.symbol,
        "timeframe": sig.timeframe,
        "strategy_type": sig.strategy_type,
        "side": sig.side,
        "price": sig.price,
        "stop_loss": sig.stop_loss,
        "take_profit": sig.take_profit,
        "confidence": sig.confidence,
        "reason": sig.reason,
        "reject_reason": reject_reason,
        "claude_analysis": sig.claude_analysis,
        "status": sig.status,
        "created_at": sig.created_at.isoformat() if sig.created_at else None,
        "resolved_at": sig.resolved_at.isoformat() if sig.resolved_at else None,
    }


@router.get("/signals")
def signals_page(request: Request):
    cfg = load_config()
    from models.ruleset import RuleSet
    session = get_session()
    try:
        active_rs = session.query(RuleSet).filter(RuleSet.status == "active").all()
        active_names = ", ".join(f"{r.name} (v{r.version})" for r in active_rs)
    finally:
        session.close()

    return TEMPLATES.TemplateResponse(
        "signals.html",
        {
            "request": request,
            "symbols": cfg.symbols,
            "signals_enabled": cfg.signals.enabled,
            "scan_interval": cfg.signals.scan_interval_sec,
            "active_rulesets": active_names or "None",
        },
    )


@router.get("/api/signals")
def get_signals(
    symbol: str | None = None,
    status: str | None = None,
    limit: int = 100,
) -> JSONResponse:
    session = get_session()
    try:
        query = session.query(Signal).order_by(Signal.created_at.desc())

        if symbol:
            query = query.filter(Signal.symbol == symbol)
        if status:
            query = query.filter(Signal.status == status)

        rows = query.limit(min(limit, 500)).all()
        return JSONResponse({
            "count": len(rows),
            "signals": [_signal_to_dict(s) for s in rows],
        })
    finally:
        session.close()


@router.get("/api/signals/{signal_id}")
def get_signal(signal_id: int) -> JSONResponse:
    session = get_session()
    try:
        sig = session.query(Signal).filter(Signal.id == signal_id).first()
        if sig is None:
            return JSONResponse({"error": "Signal not found"}, status_code=404)
        return JSONResponse(_signal_to_dict(sig))
    finally:
        session.close()


@router.post("/api/signals/{signal_id}/execute")
def execute_signal(signal_id: int) -> JSONResponse:
    from services import trade_manager
    session = get_session()
    try:
        sig = session.query(Signal).filter(Signal.id == signal_id).first()
        if sig is None:
            return JSONResponse({"error": "Signal not found"}, status_code=404)
        if sig.status not in ("pending", "confirmed"):
            return JSONResponse({"error": f"Signal is {sig.status}, cannot execute"}, status_code=400)
        sig.status = "confirmed"
        sig.resolved_at = datetime.now(timezone.utc)
        session.commit()
    finally:
        session.close()

    result = trade_manager.execute_signal(signal_id)
    if "error" in result:
        return JSONResponse(result, status_code=400)
    return JSONResponse(result)


@router.post("/api/signals/{signal_id}/reject")
def reject_signal(signal_id: int) -> JSONResponse:
    session = get_session()
    try:
        sig = session.query(Signal).filter(Signal.id == signal_id).first()
        if sig is None:
            return JSONResponse({"error": "Signal not found"}, status_code=404)
        sig.status = "rejected"
        sig.resolved_at = datetime.now(timezone.utc)
        session.commit()
        return JSONResponse({"id": sig.id, "status": "rejected"})
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


@router.post("/api/signals/scan-now")
def scan_now() -> JSONResponse:
    """Trigger an immediate signal scan (rulesets + legacy strategies)."""
    from services.signal_generator import _scan_rulesets, _scan_once
    cfg = load_config()
    if not cfg.signals.enabled:
        return JSONResponse({"error": "Signal generation is disabled"}, status_code=400)
    try:
        _scan_rulesets()
        if cfg.signals.strategies:
            _scan_once()
        return JSONResponse({"status": "scan completed"})
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=500)


@router.post("/api/signals/clear")
def clear_old_signals() -> JSONResponse:
    """Clear all rejected/expired signals older than 24 hours."""
    from datetime import timedelta
    cutoff = datetime.now(timezone.utc) - timedelta(hours=24)
    session = get_session()
    try:
        deleted = session.query(Signal).filter(
            Signal.status.in_(["rejected", "expired"]),
            Signal.created_at < cutoff,
        ).delete(synchronize_session=False)
        session.commit()
        return JSONResponse({"ok": True, "deleted": deleted})
    except Exception:
        session.rollback()
        return JSONResponse({"error": "Failed to clear"}, status_code=500)
    finally:
        session.close()
