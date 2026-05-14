"""Non-destructive performance archive cutoffs.

The archive workflow keeps broker and app trade rows intact, writes a small
snapshot for audit, and sets a cutoff so performance views default to new
results only.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from config import DATA_DIR, load_config, save_config
from db import get_session
from models.account import AccountDeal, AccountSnapshot
from models.signal import LiveTrade

ARCHIVE_DIR = DATA_DIR / "performance_archives"


def parse_dt(value: str | datetime | None) -> datetime | None:
    if not value:
        return None
    if isinstance(value, datetime):
        dt = value
    else:
        dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def get_archive_cutoff() -> datetime | None:
    cfg = load_config()
    return parse_dt(cfg.performance_archive_cutoff)


def filter_query_after_cutoff(query, column):
    cutoff = get_archive_cutoff()
    if cutoff is None:
        return query
    return query.filter(column >= cutoff)


def after_cutoff(value: str | datetime | None) -> bool:
    cutoff = get_archive_cutoff()
    if cutoff is None:
        return True
    dt = parse_dt(value)
    return dt is not None and dt >= cutoff


def cutoff_iso() -> str | None:
    cutoff = get_archive_cutoff()
    return cutoff.isoformat() if cutoff else None


def archive_current_performance(note: str = "") -> dict:
    cutoff = datetime.now(timezone.utc)
    session = get_session()
    try:
        account_deals = session.query(AccountDeal).filter(AccountDeal.closed_at < cutoff).all()
        live_trades = session.query(LiveTrade).filter(
            LiveTrade.status == "closed",
            LiveTrade.closed_at < cutoff,
            LiveTrade.pnl.isnot(None),
        ).all()
        snapshots_count = session.query(AccountSnapshot).filter(AccountSnapshot.ts < cutoff).count()

        by_symbol: dict[str, dict] = {}
        for deal in account_deals:
            sym = (deal.symbol or "").upper()
            row = by_symbol.setdefault(sym, {"trades": 0, "pnl": 0.0})
            row["trades"] += 1
            row["pnl"] += float(deal.profit or 0.0)

        snapshot = {
            "archived_at": cutoff.isoformat(),
            "note": note,
            "account_deals": {
                "count": len(account_deals),
                "pnl": round(sum(float(d.profit or 0.0) for d in account_deals), 2),
            },
            "live_trades": {
                "count": len(live_trades),
                "pnl": round(sum(float(t.pnl or 0.0) for t in live_trades), 2),
            },
            "account_snapshots": {"count": int(snapshots_count or 0)},
            "symbols": {
                sym: {"trades": data["trades"], "pnl": round(data["pnl"], 2)}
                for sym, data in sorted(by_symbol.items())
            },
        }
    finally:
        session.close()

    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    archive_path = ARCHIVE_DIR / f"performance_archive_{cutoff.strftime('%Y%m%d_%H%M%S')}.json"
    archive_path.write_text(json.dumps(snapshot, indent=2), encoding="utf-8")

    cfg = load_config()
    cfg.performance_archive_cutoff = cutoff.isoformat()
    save_config(cfg)

    return {**snapshot, "archive_path": str(archive_path)}


def clear_archive_cutoff() -> dict:
    cfg = load_config()
    old = cfg.performance_archive_cutoff
    cfg.performance_archive_cutoff = ""
    save_config(cfg)
    return {"ok": True, "previous_cutoff": old or None}


def archive_status() -> dict:
    cutoff = get_archive_cutoff()
    archives = []
    if ARCHIVE_DIR.exists():
        for path in sorted(ARCHIVE_DIR.glob("performance_archive_*.json"), reverse=True):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                archives.append(
                    {
                        "file": str(path),
                        "archived_at": data.get("archived_at"),
                        "note": data.get("note", ""),
                        "account_deals": data.get("account_deals", {}),
                        "live_trades": data.get("live_trades", {}),
                    }
                )
            except Exception:
                archives.append({"file": str(path), "error": "Could not read archive"})

    session = get_session()
    try:
        live_query = session.query(LiveTrade).filter(
            LiveTrade.status == "closed",
            LiveTrade.pnl.isnot(None),
        )
        deal_query = session.query(AccountDeal)
        if cutoff:
            live_query = live_query.filter(LiveTrade.closed_at >= cutoff)
            deal_query = deal_query.filter(AccountDeal.closed_at >= cutoff)
        current_live = live_query.all()
        current_deals = deal_query.all()
    finally:
        session.close()

    return {
        "cutoff": cutoff.isoformat() if cutoff else None,
        "archives": archives[:20],
        "current_window": {
            "live_trades": len(current_live),
            "live_pnl": round(sum(float(t.pnl or 0.0) for t in current_live), 2),
            "account_deals": len(current_deals),
            "account_deal_pnl": round(sum(float(d.profit or 0.0) for d in current_deals), 2),
        },
    }
