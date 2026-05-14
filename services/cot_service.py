"""CFTC Commitments of Traders currency futures ingestion.

The research engine uses these rows as weekly, point-in-time positioning
context. Availability is intentionally conservative so historical tests do not
use a report before it would have been public.
"""

from __future__ import annotations

import csv
import io
import logging
import zipfile
from datetime import datetime, timedelta, timezone

import httpx

from db import get_session
from models.cot import CotPosition

logger = logging.getLogger(__name__)

CFTC_LEGACY_ZIP_URL = "https://www.cftc.gov/files/dea/history/deacot{year}.zip"
SOURCE = "cftc_legacy_futures"

CFTC_MARKET_TO_CURRENCY = {
    "098662": "USD",  # USD Index - ICE Futures U.S.
    "099741": "EUR",  # Euro FX - CME
    "096742": "GBP",  # British Pound - CME
    "097741": "JPY",  # Japanese Yen - CME
    "092741": "CHF",  # Swiss Franc - CME
    "090741": "CAD",  # Canadian Dollar - CME
    "232741": "AUD",  # Australian Dollar - CME
    "112741": "NZD",  # NZ Dollar - CME
}


def count_positions_in_range(start_date: datetime, end_date: datetime) -> int:
    session = get_session()
    try:
        return session.query(CotPosition).filter(
            CotPosition.available_at >= start_date,
            CotPosition.available_at <= end_date,
        ).count()
    finally:
        session.close()


def sync_legacy_currency_futures(start_date: datetime, end_date: datetime) -> int:
    start = _ensure_aware(start_date) - timedelta(days=90)
    end = _ensure_aware(end_date)
    years = range(start.year, end.year + 1)
    total = 0
    for year in years:
        try:
            rows = _fetch_year_rows(year)
            total += _store_rows(rows, start, end)
        except Exception as exc:
            logger.warning("COT sync failed for %s: %s", year, exc)
    return total


def latest_for_currencies(currencies: set[str], end_date: datetime) -> dict[str, CotPosition]:
    if not currencies:
        return {}
    session = get_session()
    try:
        result = {}
        for currency in sorted(currencies):
            row = (
                session.query(CotPosition)
                .filter(
                    CotPosition.currency == currency,
                    CotPosition.available_at <= end_date,
                )
                .order_by(CotPosition.available_at.desc())
                .first()
            )
            if row:
                result[currency] = row
        return result
    finally:
        session.close()


def _fetch_year_rows(year: int) -> list[dict]:
    url = CFTC_LEGACY_ZIP_URL.format(year=year)
    resp = httpx.get(url, timeout=45.0, follow_redirects=True)
    resp.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(resp.content)) as archive:
        entry_name = next(
            (name for name in archive.namelist() if name.lower().endswith(".txt")),
            None,
        )
        if not entry_name:
            return []
        text = archive.read(entry_name).decode("latin-1")
    reader = csv.DictReader(io.StringIO(text))
    rows = []
    for raw in reader:
        row = {str(k).strip(): str(v).strip() for k, v in raw.items() if k is not None}
        code = row.get("CFTC Contract Market Code", "").zfill(6)
        if code in CFTC_MARKET_TO_CURRENCY:
            rows.append(row)
    return rows


def _store_rows(rows: list[dict], start_date: datetime, end_date: datetime) -> int:
    now = datetime.now(timezone.utc)
    session = get_session()
    stored = 0
    try:
        for row in rows:
            code = row.get("CFTC Contract Market Code", "").zfill(6)
            currency = CFTC_MARKET_TO_CURRENCY.get(code)
            if not currency:
                continue
            report_date = _parse_report_date(row.get("As of Date in Form YYYY-MM-DD"))
            if not report_date:
                continue
            available_at = _release_available_at(report_date)
            if available_at < start_date or available_at > end_date + timedelta(days=14):
                continue

            noncommercial_long = _num(row.get("Noncommercial Positions-Long (All)"))
            noncommercial_short = _num(row.get("Noncommercial Positions-Short (All)"))
            open_interest = max(_num(row.get("Open Interest (All)")), 0.0)
            noncommercial_net = noncommercial_long - noncommercial_short
            net_pct = (noncommercial_net / open_interest * 100.0) if open_interest > 0 else 0.0

            existing = (
                session.query(CotPosition)
                .filter(
                    CotPosition.currency == currency,
                    CotPosition.report_date == report_date,
                    CotPosition.source == SOURCE,
                )
                .first()
            )
            target = existing or CotPosition(
                currency=currency,
                report_date=report_date,
                source=SOURCE,
                fetched_at=now,
                market_name=row.get("Market and Exchange Names", "")[:128],
                available_at=available_at,
                open_interest=open_interest,
                noncommercial_long=noncommercial_long,
                noncommercial_short=noncommercial_short,
                commercial_long=_num(row.get("Commercial Positions-Long (All)")),
                commercial_short=_num(row.get("Commercial Positions-Short (All)")),
                pct_noncommercial_long=_num(row.get("% of OI-Noncommercial-Long (All)")),
                pct_noncommercial_short=_num(row.get("% of OI-Noncommercial-Short (All)")),
                noncommercial_net=noncommercial_net,
                noncommercial_net_pct=net_pct,
            )
            target.market_name = row.get("Market and Exchange Names", "")[:128]
            target.available_at = available_at
            target.open_interest = open_interest
            target.noncommercial_long = noncommercial_long
            target.noncommercial_short = noncommercial_short
            target.commercial_long = _num(row.get("Commercial Positions-Long (All)"))
            target.commercial_short = _num(row.get("Commercial Positions-Short (All)"))
            target.pct_noncommercial_long = _num(row.get("% of OI-Noncommercial-Long (All)"))
            target.pct_noncommercial_short = _num(row.get("% of OI-Noncommercial-Short (All)"))
            target.noncommercial_net = noncommercial_net
            target.noncommercial_net_pct = net_pct
            target.fetched_at = now
            if existing is None:
                session.add(target)
                stored += 1

        session.commit()
        if stored:
            logger.info("Stored %d COT currency futures rows", stored)
        return stored
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def _parse_report_date(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        dt = datetime.strptime(value.strip(), "%Y-%m-%d")
        return dt.replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def _release_available_at(report_date: datetime) -> datetime:
    # Official COT releases are usually Friday 3:30 p.m. Eastern for Tuesday
    # data, but holidays can delay by one or two days. Use the following
    # Tuesday 20:30 UTC as a conservative point-in-time availability cutoff.
    return (report_date + timedelta(days=7)).replace(hour=20, minute=30, second=0, microsecond=0)


def _num(value: str | None) -> float:
    if value is None:
        return 0.0
    text = str(value).replace(",", "").strip()
    if not text:
        return 0.0
    try:
        return float(text)
    except ValueError:
        return 0.0


def _ensure_aware(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)
