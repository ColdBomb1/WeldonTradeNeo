"""Portfolio-level milestone evaluation and candidate search."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

from models.candle import Candle
from models.ruleset import RuleSet
from services import market_context_service
from services.backtest_engine import BacktestConfig, BacktestResults, run_backtest


DEFAULT_TARGETS = {
    "short_days": 14,
    "long_days": 30,
    "min_short_return_pct": 2.5,
    "min_long_return_pct": 6.0,
    "max_drawdown_pct": 6.0,
    "min_short_trades": 1,
    "min_long_trades": 1,
}

LIVE_READINESS_DEFAULTS = {
    "windows": [
        {"days": 7, "min_return_pct": 0.0, "max_drawdown_pct": 4.0, "min_trades": 8, "min_profit_factor": 1.05},
        {"days": 14, "min_return_pct": 2.5, "max_drawdown_pct": 5.0, "min_trades": 20, "min_profit_factor": 1.20},
        {"days": 30, "min_return_pct": 6.0, "max_drawdown_pct": 5.0, "min_trades": 50, "min_profit_factor": 1.20},
        {"days": 60, "min_return_pct": 10.0, "max_drawdown_pct": 6.0, "min_trades": 100, "min_profit_factor": 1.15},
    ],
    "stress_windows": [
        {"days": 14, "min_return_pct": 1.5, "max_drawdown_pct": 6.0, "min_trades": 20, "min_profit_factor": 1.05},
        {"days": 30, "min_return_pct": 4.0, "max_drawdown_pct": 6.0, "min_trades": 50, "min_profit_factor": 1.10},
    ],
    "cost_stress_multiplier": 1.5,
    "min_symbols": 4,
    "max_symbol_14d_loss_pct": 1.0,
    "min_symbol_30d_return_pct": 0.0,
    "min_symbol_60d_return_pct": 0.0,
    "max_symbol_30d_drawdown_pct": 3.5,
    "max_symbol_60d_drawdown_pct": 4.0,
    "max_30d_pnl_concentration": 0.45,
}


def normalize_targets(payload: dict | None = None) -> dict:
    payload = payload or {}
    targets = dict(DEFAULT_TARGETS)
    for key in targets:
        if payload.get(key) is not None:
            targets[key] = payload[key]

    targets["short_days"] = max(1, min(120, int(targets["short_days"])))
    targets["long_days"] = max(targets["short_days"], min(180, int(targets["long_days"])))
    targets["min_short_return_pct"] = float(targets["min_short_return_pct"])
    targets["min_long_return_pct"] = float(targets["min_long_return_pct"])
    targets["max_drawdown_pct"] = max(0.1, min(50.0, float(targets["max_drawdown_pct"])))
    targets["min_short_trades"] = max(0, int(targets["min_short_trades"]))
    targets["min_long_trades"] = max(0, int(targets["min_long_trades"]))
    return targets


def normalize_live_readiness(raw: dict | None = None) -> dict:
    raw = raw or {}
    readiness = dict(LIVE_READINESS_DEFAULTS)
    readiness["windows"] = [
        _normalize_readiness_window(item)
        for item in (raw.get("windows") or LIVE_READINESS_DEFAULTS["windows"])
    ]
    readiness["stress_windows"] = [
        _normalize_readiness_window(item)
        for item in (raw.get("stress_windows") or LIVE_READINESS_DEFAULTS["stress_windows"])
    ]
    readiness["cost_stress_multiplier"] = max(
        1.0,
        min(5.0, float(raw.get("cost_stress_multiplier") or readiness["cost_stress_multiplier"])),
    )
    readiness["min_symbols"] = max(1, min(20, int(raw.get("min_symbols") or readiness["min_symbols"])))
    readiness["max_symbol_14d_loss_pct"] = max(
        0.0,
        min(20.0, float(raw.get("max_symbol_14d_loss_pct") or readiness["max_symbol_14d_loss_pct"])),
    )
    readiness["min_symbol_30d_return_pct"] = max(
        -20.0,
        min(50.0, float(raw.get("min_symbol_30d_return_pct") or readiness["min_symbol_30d_return_pct"])),
    )
    readiness["min_symbol_60d_return_pct"] = max(
        -20.0,
        min(50.0, float(raw.get("min_symbol_60d_return_pct") or readiness["min_symbol_60d_return_pct"])),
    )
    readiness["max_symbol_30d_drawdown_pct"] = max(
        0.1,
        min(50.0, float(raw.get("max_symbol_30d_drawdown_pct") or readiness["max_symbol_30d_drawdown_pct"])),
    )
    readiness["max_symbol_60d_drawdown_pct"] = max(
        0.1,
        min(50.0, float(raw.get("max_symbol_60d_drawdown_pct") or readiness["max_symbol_60d_drawdown_pct"])),
    )
    readiness["max_30d_pnl_concentration"] = max(
        0.1,
        min(1.0, float(raw.get("max_30d_pnl_concentration") or readiness["max_30d_pnl_concentration"])),
    )
    return readiness


def _normalize_readiness_window(raw: dict) -> dict:
    return {
        "days": max(1, min(180, int(raw.get("days") or 14))),
        "min_return_pct": max(-50.0, min(100.0, float(raw.get("min_return_pct") or 0.0))),
        "max_drawdown_pct": max(0.1, min(50.0, float(raw.get("max_drawdown_pct") or 6.0))),
        "min_trades": max(0, min(10000, int(raw.get("min_trades") or 0))),
        "min_profit_factor": max(0.1, min(10.0, float(raw.get("min_profit_factor") or 1.0))),
    }


def normalize_options(payload: dict | None, cfg) -> dict:
    payload = payload or {}
    realistic = _coerce_bool(payload.get("realistic"), True)
    return {
        "initial_balance": float(
            payload.get("initial_balance")
            or cfg.execution.account_start_balance
            or cfg.training.initial_balance
        ),
        "realistic": realistic,
        "entry_timing": payload.get("entry_timing") or ("next_open" if realistic else "signal_close"),
        "slippage_pips": float(
            payload.get("slippage_pips")
            if payload.get("slippage_pips") is not None
            else (0.2 if realistic else 0.0)
        ),
        "min_sl_pips": float(
            payload.get("min_sl_pips")
            if payload.get("min_sl_pips") is not None
            else (10.0 if realistic else 0.0)
        ),
        "enforce_live_exit_policy": _coerce_bool(payload.get("enforce_live_exit_policy"), realistic),
        "end_date": _parse_dt(payload["end_date"]) if payload.get("end_date") else None,
        "use_validation_balance": _coerce_bool(payload.get("use_validation_balance"), False),
    }


def evaluate_active_milestone(session, payload: dict | None, cfg) -> dict:
    payload = payload or {}
    targets = normalize_targets(payload)
    options = normalize_options(payload, cfg)
    active = (
        session.query(RuleSet)
        .filter(RuleSet.status == "active")
        .order_by(RuleSet.id.asc())
        .all()
    )
    scopes = build_structured_scopes(session, active)
    if not scopes:
        return {"error": "No active structured research rulesets found"}

    short = evaluate_portfolio_scopes(
        session,
        scopes,
        days=targets["short_days"],
        options=options,
    )
    long = evaluate_portfolio_scopes(
        session,
        scopes,
        days=targets["long_days"],
        options=options,
        common_end=_parse_dt(short["portfolio"]["window_end"]),
    )
    status = score_milestone(short["portfolio"], long["portfolio"], targets)
    return {
        "targets": targets,
        "status": status,
        "short": short,
        "long": long,
    }


def validate_portfolio(session, payload: dict | None, cfg) -> dict:
    payload = payload or {}
    targets = normalize_targets(payload)
    options = normalize_options(payload, cfg)
    selected_ids = _normalize_ruleset_ids(payload.get("ruleset_ids"))
    selection_source = "explicit"
    search_result = None
    if not selected_ids and _coerce_bool(payload.get("auto_select"), False):
        search_payload = {
            **payload,
            "include_candidates": _coerce_bool(payload.get("include_candidates"), True),
            "top_portfolios": max(1, int(payload.get("top_portfolios") or 10)),
        }
        search_result = search_candidate_portfolios(session, search_payload, cfg)
        best = search_result.get("best_passing")
        if not best:
            return {
                "targets": targets,
                "status": {
                    "passed": False,
                    "status": "failed",
                    "reasons": ["No passing candidate portfolio was found."],
                },
                "search": _compact_search_result(search_result),
                "selected_rulesets": [],
            }
        selected_ids = [int(item["ruleset_id"]) for item in best.get("strategies") or []]
        selection_source = "auto_select_best_passing"

    if not selected_ids:
        return {
            "targets": targets,
            "status": {
                "passed": False,
                "status": "failed",
                "reasons": ["ruleset_ids are required unless auto_select is enabled."],
            },
            "selected_rulesets": [],
        }

    rulesets = (
        session.query(RuleSet)
        .filter(RuleSet.id.in_(selected_ids))
        .order_by(RuleSet.id.asc())
        .all()
    )
    found_ids = {int(rs.id) for rs in rulesets}
    missing_ids = [ruleset_id for ruleset_id in selected_ids if ruleset_id not in found_ids]
    scopes = build_structured_scopes(session, rulesets)
    selected_conflicts = _selected_scope_conflicts(scopes)
    if missing_ids or selected_conflicts:
        reasons = []
        if missing_ids:
            reasons.append(f"Missing rulesets: {', '.join(str(item) for item in missing_ids)}.")
        reasons.extend(selected_conflicts)
        return {
            "targets": targets,
            "status": {"passed": False, "status": "failed", "reasons": reasons},
            "selected_rulesets": _selected_rulesets_summary(rulesets),
        }
    if not scopes:
        return {
            "targets": targets,
            "status": {
                "passed": False,
                "status": "failed",
                "reasons": ["No selected rulesets contain structured research strategies."],
            },
            "selected_rulesets": _selected_rulesets_summary(rulesets),
        }

    common_end = options["end_date"] or min(item["latest"] for item in scopes)
    short = evaluate_portfolio_scopes(
        session,
        scopes,
        days=targets["short_days"],
        options=options,
        common_end=common_end,
    )
    long = evaluate_portfolio_scopes(
        session,
        scopes,
        days=targets["long_days"],
        options=options,
        common_end=common_end,
    )
    status = score_milestone(short["portfolio"], long["portfolio"], targets)
    validation_id = f"portfolio_validation_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    return {
        "validation_id": validation_id,
        "validated_at": datetime.now(timezone.utc).isoformat(),
        "selection_source": selection_source,
        "targets": targets,
        "options": _public_options(options),
        "status": status,
        "selected_ruleset_ids": [int(rs.id) for rs in rulesets],
        "selected_rulesets": _selected_rulesets_summary(rulesets),
        "short": short,
        "long": long,
        "search": _compact_search_result(search_result) if search_result else None,
    }


def evaluate_live_readiness(session, payload: dict | None, cfg) -> dict:
    payload = payload or {}
    readiness = normalize_live_readiness(payload.get("readiness") or payload)
    options = normalize_options(payload, cfg)
    selected_ids = _normalize_ruleset_ids(payload.get("ruleset_ids"))
    selection_source = "explicit" if selected_ids else "active"

    query = session.query(RuleSet)
    if selected_ids:
        rulesets = query.filter(RuleSet.id.in_(selected_ids)).order_by(RuleSet.id.asc()).all()
        found_ids = {int(rs.id) for rs in rulesets}
        missing_ids = [ruleset_id for ruleset_id in selected_ids if ruleset_id not in found_ids]
    else:
        rulesets = query.filter(RuleSet.status == "active").order_by(RuleSet.id.asc()).all()
        missing_ids = []

    scopes = build_structured_scopes(session, rulesets)
    selected_conflicts = _selected_scope_conflicts(scopes)
    if missing_ids or selected_conflicts:
        reasons = []
        if missing_ids:
            reasons.append(f"Missing rulesets: {', '.join(str(item) for item in missing_ids)}.")
        reasons.extend(selected_conflicts)
        return {
            "selection_source": selection_source,
            "status": {"passed": False, "status": "failed", "reasons": reasons},
            "readiness": readiness,
            "selected_rulesets": _selected_rulesets_summary(rulesets),
        }
    if not scopes:
        return {
            "selection_source": selection_source,
            "status": {
                "passed": False,
                "status": "failed",
                "reasons": ["No selected rulesets contain structured research strategies."],
            },
            "readiness": readiness,
            "selected_rulesets": _selected_rulesets_summary(rulesets),
        }

    common_end = options["end_date"] or min(item["latest"] for item in scopes)
    baseline = _evaluate_readiness_windows(session, scopes, readiness["windows"], options, common_end)
    stress_options = dict(options)
    stress_options["spread_multiplier"] = readiness["cost_stress_multiplier"]
    stress_options["slippage_pips"] = float(options["slippage_pips"]) * readiness["cost_stress_multiplier"]
    stress = _evaluate_readiness_windows(session, scopes, readiness["stress_windows"], stress_options, common_end)
    status = score_live_readiness(baseline, stress, readiness, scopes)
    return {
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "selection_source": selection_source,
        "selected_ruleset_ids": [int(rs.id) for rs in rulesets],
        "selected_rulesets": _selected_rulesets_summary(rulesets),
        "options": _public_options(options),
        "readiness": readiness,
        "status": status,
        "baseline": baseline,
        "stress": stress,
    }


def _evaluate_readiness_windows(
    session,
    scopes: list[dict],
    windows: list[dict],
    options: dict,
    common_end: datetime,
) -> dict:
    evaluations = {}
    for window in windows:
        days = int(window["days"])
        result = evaluate_portfolio_scopes(
            session,
            scopes,
            days=days,
            options=options,
            common_end=common_end,
        )
        evaluations[str(days)] = _compact_portfolio_evaluation(result)
    return evaluations


def score_live_readiness(
    baseline: dict,
    stress: dict,
    readiness: dict,
    scopes: list[dict],
) -> dict:
    reasons = []
    gates = {
        "baseline": _score_readiness_windows(baseline, readiness["windows"], "baseline"),
        "stress": _score_readiness_windows(stress, readiness["stress_windows"], "stress"),
        "symbol_balance": _score_symbol_balance(baseline, readiness, scopes),
    }
    for gate_name, gate in gates.items():
        reasons.extend(f"{gate_name}: {reason}" for reason in gate["reasons"])
    return {
        "passed": not reasons,
        "status": "ready" if not reasons else "not_ready",
        "reasons": reasons,
        "gates": gates,
    }


def _score_readiness_windows(evaluations: dict, windows: list[dict], label: str) -> dict:
    reasons = []
    per_window = {}
    for window in windows:
        days = int(window["days"])
        key = str(days)
        portfolio = (evaluations.get(key) or {}).get("portfolio") or {}
        gate = _score_window(
            portfolio,
            window["min_return_pct"],
            window["max_drawdown_pct"],
            window["min_trades"],
        )
        if float(portfolio.get("profit_factor") or 0.0) < float(window["min_profit_factor"]):
            gate["reasons"].append(
                f"profit factor {float(portfolio.get('profit_factor') or 0.0):.2f} below {float(window['min_profit_factor']):.2f}"
            )
            gate["passed"] = False
            gate["status"] = "failed"
        per_window[key] = gate
        reasons.extend(f"{days}d {reason}" for reason in gate["reasons"])
    return {
        "passed": not reasons,
        "status": "passed" if not reasons else "failed",
        "label": label,
        "windows": per_window,
        "reasons": reasons,
    }


def _score_symbol_balance(baseline: dict, readiness: dict, scopes: list[dict]) -> dict:
    reasons = []
    symbols = sorted({str(scope["symbol"]).upper() for scope in scopes})
    if len(symbols) < int(readiness["min_symbols"]):
        reasons.append(f"symbols {len(symbols)} below {int(readiness['min_symbols'])}")

    window_14 = baseline.get("14") or {}
    window_30 = baseline.get("30") or {}
    window_60 = baseline.get("60") or {}
    for strategy in window_14.get("strategies") or []:
        if float(strategy.get("return_pct") or 0.0) < -float(readiness["max_symbol_14d_loss_pct"]):
            reasons.append(
                f"{strategy.get('symbol')} 14d return {float(strategy.get('return_pct') or 0.0):.2f}% below -{float(readiness['max_symbol_14d_loss_pct']):.2f}%"
            )
    for strategy in window_30.get("strategies") or []:
        if float(strategy.get("return_pct") or 0.0) < float(readiness["min_symbol_30d_return_pct"]):
            reasons.append(
                f"{strategy.get('symbol')} 30d return {float(strategy.get('return_pct') or 0.0):.2f}% below {float(readiness['min_symbol_30d_return_pct']):.2f}%"
            )
        if float(strategy.get("max_drawdown_pct") or 0.0) > float(readiness["max_symbol_30d_drawdown_pct"]):
            reasons.append(
                f"{strategy.get('symbol')} 30d drawdown {float(strategy.get('max_drawdown_pct') or 0.0):.2f}% exceeds {float(readiness['max_symbol_30d_drawdown_pct']):.2f}%"
            )
    for strategy in window_60.get("strategies") or []:
        if float(strategy.get("return_pct") or 0.0) < float(readiness["min_symbol_60d_return_pct"]):
            reasons.append(
                f"{strategy.get('symbol')} 60d return {float(strategy.get('return_pct') or 0.0):.2f}% below {float(readiness['min_symbol_60d_return_pct']):.2f}%"
            )
        if float(strategy.get("max_drawdown_pct") or 0.0) > float(readiness["max_symbol_60d_drawdown_pct"]):
            reasons.append(
                f"{strategy.get('symbol')} 60d drawdown {float(strategy.get('max_drawdown_pct') or 0.0):.2f}% exceeds {float(readiness['max_symbol_60d_drawdown_pct']):.2f}%"
            )

    strategies_30 = window_30.get("strategies") or []
    total_pnl = sum(max(0.0, float(item.get("pnl") or 0.0)) for item in strategies_30)
    if total_pnl > 0.0 and strategies_30:
        concentration = max(max(0.0, float(item.get("pnl") or 0.0)) for item in strategies_30) / total_pnl
        if concentration > float(readiness["max_30d_pnl_concentration"]):
            reasons.append(
                f"30d PnL concentration {concentration:.2f} exceeds {float(readiness['max_30d_pnl_concentration']):.2f}"
            )
    return {
        "passed": not reasons,
        "status": "passed" if not reasons else "failed",
        "symbols": symbols,
        "reasons": reasons,
    }


def persist_portfolio_validation(
    session,
    validation: dict,
    *,
    note: str = "",
    promoted: bool = False,
) -> None:
    selected_ids = [int(item) for item in validation.get("selected_ruleset_ids") or []]
    if not selected_ids:
        return
    compact = compact_validation_for_storage(validation, note=note, promoted=promoted)
    for rs in session.query(RuleSet).filter(RuleSet.id.in_(selected_ids)).all():
        params = dict(rs.parameters or {})
        history = list(params.get("portfolio_validation_history") or [])
        history.append(compact)
        params["portfolio_validation_history"] = history[-10:]
        params["portfolio_validation"] = compact
        rs.parameters = params
        metrics = dict(rs.performance_metrics or {})
        metrics["portfolio_validation"] = compact
        rs.performance_metrics = metrics


def compact_validation_for_storage(validation: dict, *, note: str = "", promoted: bool = False) -> dict:
    return {
        "validation_id": validation.get("validation_id"),
        "validated_at": validation.get("validated_at"),
        "selection_source": validation.get("selection_source"),
        "note": note,
        "promoted": promoted,
        "promoted_at": datetime.now(timezone.utc).isoformat() if promoted else None,
        "status": validation.get("status"),
        "targets": validation.get("targets"),
        "options": validation.get("options"),
        "selected_ruleset_ids": validation.get("selected_ruleset_ids") or [],
        "selected_rulesets": validation.get("selected_rulesets") or [],
        "short": validation.get("short", {}).get("portfolio"),
        "long": validation.get("long", {}).get("portfolio"),
    }


def search_candidate_portfolios(session, payload: dict | None, cfg) -> dict:
    payload = payload or {}
    targets = normalize_targets(payload)
    options = normalize_options(payload, cfg)
    include_candidates = _coerce_bool(payload.get("include_candidates"), False)
    statuses = payload.get("statuses")
    if statuses is None:
        statuses = ["active", "validated"]
        if include_candidates:
            statuses.append("candidate")
    statuses = {str(status).lower() for status in statuses}

    rows = (
        session.query(RuleSet)
        .filter(RuleSet.status.in_(sorted(statuses)))
        .order_by(RuleSet.id.asc())
        .all()
    )
    scopes = build_structured_scopes(session, rows)
    if not scopes:
        return {
            "targets": targets,
            "error": "No structured rulesets found for the requested statuses.",
            "candidate_count": 0,
            "top_portfolios": [],
        }
    prefilter_per_group = int(
        payload.get("prefilter_per_group")
        if payload.get("prefilter_per_group") is not None
        else (5 if include_candidates else 0)
    )
    if prefilter_per_group > 0:
        scopes = _prefilter_scopes(scopes, prefilter_per_group)

    common_end = options["end_date"] or min(item["latest"] for item in scopes)
    short_eval = evaluate_scope_windows(
        session,
        scopes,
        days=targets["short_days"],
        options=options,
        common_end=common_end,
    )
    long_eval = evaluate_scope_windows(
        session,
        scopes,
        days=targets["long_days"],
        options=options,
        common_end=common_end,
    )

    candidates = []
    for key, long_result in long_eval["scopes_by_key"].items():
        short_result = short_eval["scopes_by_key"].get(key)
        if not short_result:
            continue
        candidate = {
            "key": key,
            "ruleset_id": long_result["ruleset_id"],
            "name": long_result["name"],
            "status": long_result["status"],
            "symbol": long_result["symbol"],
            "timeframe": long_result["timeframe"],
            "short": short_result,
            "long": long_result,
        }
        candidate["score"] = _candidate_score(candidate, targets)
        candidate["gates"] = {
            "short": _score_window(short_result, targets["min_short_return_pct"], targets["max_drawdown_pct"], targets["min_short_trades"]),
            "long": _score_window(long_result, targets["min_long_return_pct"], targets["max_drawdown_pct"], targets["min_long_trades"]),
        }
        candidates.append(candidate)

    top_per_group = max(1, min(20, int(payload.get("top_per_group") or 8)))
    beam_width = max(10, min(1000, int(payload.get("beam_width") or 200)))
    groups: dict[tuple[str, str], list[dict]] = {}
    for candidate in candidates:
        groups.setdefault((candidate["symbol"], candidate["timeframe"]), []).append(candidate)
    grouped_candidates = []
    for group_key, group_items in sorted(groups.items()):
        sorted_group = sorted(group_items, key=lambda item: item["score"], reverse=True)[:top_per_group]
        grouped_candidates.append({"group": group_key, "candidates": sorted_group})

    beam: list[tuple[dict, ...]] = [tuple()]
    for group in grouped_candidates:
        expanded: list[tuple[dict, ...]] = []
        for existing in beam:
            expanded.append(existing)
            for candidate in group["candidates"]:
                expanded.append(existing + (candidate,))

        ranked = []
        seen_portfolios: set[tuple[tuple[int, str, str], ...]] = set()
        for combo in expanded:
            key = tuple(sorted(
                (int(item["ruleset_id"]), str(item["symbol"]), str(item["timeframe"]))
                for item in combo
            ))
            if key in seen_portfolios:
                continue
            seen_portfolios.add(key)
            ranked.append((_portfolio_score(combo, targets, options["initial_balance"]), combo))
        ranked.sort(key=lambda item: item[0], reverse=True)
        beam = [combo for _, combo in ranked[:beam_width]]

    portfolios = []
    for combo in beam:
        portfolios.append(_portfolio_candidate_summary(combo, targets, options["initial_balance"]))
    portfolios.sort(
        key=lambda item: (
            item["status"]["passed"],
            item["score"],
            item["long"]["return_pct"],
            item["short"]["return_pct"],
        ),
        reverse=True,
    )
    top_limit = max(1, min(50, int(payload.get("top_portfolios") or 10)))
    return {
        "targets": targets,
        "statuses": sorted(statuses),
        "include_candidates": include_candidates,
        "prefiltered_scope_count": len(scopes),
        "candidate_count": len(candidates),
        "groups": [
            {
                "symbol": symbol,
                "timeframe": timeframe,
                "candidate_count": len(groups.get((symbol, timeframe), [])),
            }
            for symbol, timeframe in sorted(groups)
        ],
        "best_passing": next((p for p in portfolios if p["status"]["passed"]), None),
        "top_portfolios": portfolios[:top_limit],
    }


def _prefilter_scopes(scopes: list[dict], max_per_group: int) -> list[dict]:
    grouped: dict[tuple[str, str], list[dict]] = {}
    for scope in scopes:
        grouped.setdefault((scope["symbol"], scope["timeframe"]), []).append(scope)
    selected = []
    for group_scopes in grouped.values():
        selected.extend(
            sorted(group_scopes, key=_stored_scope_score, reverse=True)[:max_per_group]
        )
    return selected


def _stored_scope_score(scope: dict) -> float:
    rs = scope["ruleset"]
    metrics = rs.performance_metrics or {}
    params = rs.parameters or {}
    validation = params.get("validation") or {}
    validation_metrics = validation.get("metrics") or {}
    metric_sets = [
        metrics,
        _as_dict(metrics.get("holdout_metrics")),
        _as_dict(metrics.get("rolling_holdout_metrics")),
        _as_dict(validation_metrics.get("holdout")),
        _as_dict(validation_metrics.get("rolling_holdout")),
    ]
    best_return = max((float(m.get("return_pct") or -100.0) for m in metric_sets), default=-100.0)
    best_pf = max((float(m.get("profit_factor") or 0.0) for m in metric_sets), default=0.0)
    best_dd = min((float(m.get("max_drawdown") or 1.0) for m in metric_sets), default=1.0)
    status_bonus = {"active": 0.2, "validated": 0.5, "candidate": 0.0}.get(str(rs.status), 0.0)
    passed_bonus = 0.4 if (validation.get("passed") is True or metrics.get("validated") is True) else 0.0
    return best_return * 2.0 + min(best_pf, 4.0) - best_dd * 100.0 * 0.15 + status_bonus + passed_bonus


def _as_dict(value: Any) -> dict:
    return value if isinstance(value, dict) else {}


def build_structured_scopes(session, rulesets: list[RuleSet]) -> list[dict]:
    scopes = []
    for rs in rulesets:
        params = rs.parameters or {}
        schema = params.get("strategy_schema") or {}
        if params.get("execution_engine") != "research_genome" or not schema:
            continue
        validation = params.get("validation") or {}
        criteria = validation.get("criteria") or {}
        for symbol in rs.symbols or []:
            for timeframe in rs.timeframes or []:
                latest = (
                    session.query(Candle.ts)
                    .filter(Candle.symbol == symbol, Candle.timeframe == timeframe)
                    .order_by(Candle.ts.desc())
                    .first()
                )
                if not latest:
                    continue
                scopes.append({
                    "ruleset": rs,
                    "schema": schema,
                    "criteria": criteria,
                    "symbol": str(symbol).upper(),
                    "timeframe": str(timeframe),
                    "latest": latest[0],
                    "key": f"{rs.id}:{str(symbol).upper()}:{str(timeframe)}",
                })
    return scopes


def evaluate_portfolio_scopes(
    session,
    scopes: list[dict],
    *,
    days: int,
    options: dict,
    common_end: datetime | None = None,
) -> dict:
    scope_eval = evaluate_scope_windows(
        session,
        scopes,
        days=days,
        options=options,
        common_end=common_end,
    )
    portfolio = combine_scope_results(
        list(scope_eval["scopes_by_key"].values()),
        initial_balance=float(options["initial_balance"]),
        days=days,
        common_start=scope_eval["window_start"],
        common_end=scope_eval["window_end"],
    )
    return {
        "portfolio": portfolio,
        "strategies": list(scope_eval["scopes_by_key"].values()),
        "combined_curve": portfolio.pop("_combined_curve", []),
        "combined_pnl_curve": portfolio.pop("_combined_pnl_curve", []),
        "symbol_curves": portfolio.pop("_symbol_curves", []),
        "trades": portfolio.pop("_trades", [])[-200:],
    }


def evaluate_scope_windows(
    session,
    scopes: list[dict],
    *,
    days: int,
    options: dict,
    common_end: datetime | None = None,
) -> dict:
    if not scopes:
        return {"window_start": None, "window_end": None, "scopes_by_key": {}}
    end = common_end or options.get("end_date") or min(item["latest"] for item in scopes)
    start = end - timedelta(days=days)

    candles_by_symbol_timeframe: dict[tuple[str, str], list[dict]] = {}
    for item in scopes:
        symbol = item["symbol"]
        timeframe = item["timeframe"]
        key = (symbol, timeframe)
        if key in candles_by_symbol_timeframe:
            continue
        rows = (
            session.query(Candle)
            .filter(
                Candle.symbol == symbol,
                Candle.timeframe == timeframe,
                Candle.ts >= start,
                Candle.ts <= end,
            )
            .order_by(Candle.ts.asc())
            .all()
        )
        candles_by_symbol_timeframe[key] = [
            {
                "time": c.ts.isoformat(),
                "open": c.open,
                "high": c.high,
                "low": c.low,
                "close": c.close,
                "volume": c.volume or 0,
            }
            for c in rows
        ]

    candles_by_timeframe: dict[str, dict[str, list[dict]]] = {}
    for (symbol, timeframe), candles in candles_by_symbol_timeframe.items():
        candles_by_timeframe.setdefault(timeframe, {})[symbol] = candles
    for candles_by_symbol in candles_by_timeframe.values():
        market_context_service.enrich_candles_by_symbol(
            candles_by_symbol,
            include_news=True,
            include_strength=True,
            include_cot=True,
            include_sentiment=False,
        )

    scopes_by_key = {}
    for item in scopes:
        symbol = item["symbol"]
        timeframe = item["timeframe"]
        candles = candles_by_symbol_timeframe.get((symbol, timeframe), [])
        if len(candles) < 50:
            continue
        result = _run_structured_scope(
            item=item,
            candles=candles,
            start=start,
            end=end,
            options=options,
        )
        scopes_by_key[item["key"]] = result
    return {
        "window_start": start,
        "window_end": end,
        "scopes_by_key": scopes_by_key,
    }


def combine_scope_results(
    scope_results: list[dict],
    *,
    initial_balance: float,
    days: int,
    common_start: datetime,
    common_end: datetime,
) -> dict:
    curves_by_key = {item["key"]: item["equity_curve"] for item in scope_results}
    starts_by_key = {item["key"]: float(item["initial_balance"]) for item in scope_results}
    all_trades = []
    for item in scope_results:
        for trade in item.get("trades") or []:
            all_trades.append({
                **trade,
                "ruleset_id": item["ruleset_id"],
                "ruleset_name": item["name"],
                "symbol": item["symbol"],
                "timeframe": item["timeframe"],
            })
    all_trades.sort(key=lambda t: t.get("exit_ts") or t.get("entry_ts") or "")
    combined_curve = _combine_equity_curves(curves_by_key, starts_by_key, initial_balance)
    combined_pnl_curve = _combine_pnl_curves(curves_by_key, starts_by_key)
    gross_profit = sum(float(t.get("pnl") or 0.0) for t in all_trades if float(t.get("pnl") or 0.0) > 0)
    gross_loss = abs(sum(float(t.get("pnl") or 0.0) for t in all_trades if float(t.get("pnl") or 0.0) <= 0))
    wins = sum(1 for t in all_trades if float(t.get("pnl") or 0.0) > 0)
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else (999.0 if gross_profit > 0 else 0.0)
    total_pnl = round(sum(float(item.get("pnl") or 0.0) for item in scope_results), 2)
    symbol_curves = _symbol_curves(scope_results)
    max_drawdown = _max_drawdown_from_curve(combined_curve)
    return {
        "window_start": common_start.isoformat(),
        "window_end": common_end.isoformat(),
        "days": days,
        "initial_balance": initial_balance,
        "final_balance_estimate": round(initial_balance + total_pnl, 2),
        "pnl": total_pnl,
        "return_pct": round(total_pnl / initial_balance * 100.0, 3) if initial_balance else 0.0,
        "total_trades": len(all_trades),
        "winning_trades": wins,
        "losing_trades": len(all_trades) - wins,
        "win_rate": round(wins / len(all_trades), 4) if all_trades else 0.0,
        "profit_factor": round(min(profit_factor, 999.0), 4),
        "max_drawdown": max_drawdown,
        "max_drawdown_pct": round(max_drawdown * 100.0, 3),
        "_combined_curve": [
            {
                "time": point["time"],
                "equity": point["equity"],
                "pnl": round(float(point["equity"]) - initial_balance, 2),
            }
            for point in combined_curve
        ],
        "_combined_pnl_curve": combined_pnl_curve,
        "_symbol_curves": symbol_curves,
        "_trades": all_trades,
    }


def score_milestone(short_portfolio: dict, long_portfolio: dict, targets: dict) -> dict:
    short_gate = _score_window(
        short_portfolio,
        targets["min_short_return_pct"],
        targets["max_drawdown_pct"],
        targets["min_short_trades"],
    )
    long_gate = _score_window(
        long_portfolio,
        targets["min_long_return_pct"],
        targets["max_drawdown_pct"],
        targets["min_long_trades"],
    )
    reasons = [f"14d: {reason}" for reason in short_gate["reasons"]]
    reasons.extend(f"30d: {reason}" for reason in long_gate["reasons"])
    return {
        "passed": not reasons,
        "status": "passed" if not reasons else "failed",
        "short": short_gate,
        "long": long_gate,
        "reasons": reasons,
    }


def _run_structured_scope(
    *,
    item: dict,
    candles: list[dict],
    start: datetime,
    end: datetime,
    options: dict,
) -> dict:
    rs = item["ruleset"]
    schema = item["schema"]
    criteria = item["criteria"]
    symbol = item["symbol"]
    timeframe = item["timeframe"]
    starting_balance = float(
        criteria.get("initial_balance")
        if options.get("use_validation_balance") and criteria.get("initial_balance")
        else options["initial_balance"]
    )
    result = run_backtest(
        BacktestConfig(
            symbol=symbol,
            timeframe=timeframe,
            strategy_type="research_genome",
            parameters=schema,
            start_date=start,
            end_date=end,
            initial_balance=starting_balance,
            spread_pips=float(criteria.get("spread_pips") or 1.5) * float(options.get("spread_multiplier") or 1.0),
            pip_value=0.01 if "JPY" in symbol.upper() else 0.0001,
            lot_type=str(criteria.get("lot_type") or "mini"),
            risk_per_trade_pct=float(criteria.get("risk_per_trade_pct") or 0.35),
            sl_atr_multiplier=float(schema.get("sl_atr_multiplier", 2.0)),
            tp_atr_multiplier=float(schema.get("sl_atr_multiplier", 2.0)) * float(schema.get("tp_rr", 1.7)),
            monthly_max_loss_pct=float(criteria.get("max_drawdown_pct") or 6.0),
            entry_timing=options["entry_timing"],
            slippage_pips=float(options["slippage_pips"]),
            min_sl_pips=float(options["min_sl_pips"]),
            enforce_live_exit_policy=bool(options["enforce_live_exit_policy"]),
            broker_lot_units=100000 if options["realistic"] else None,
        ),
        candles,
    )
    return _scope_result_dict(item, result, starting_balance, len(candles))


def _scope_result_dict(item: dict, result: BacktestResults, starting_balance: float, candle_count: int) -> dict:
    pnl = round(result.final_balance - starting_balance, 2)
    rs = item["ruleset"]
    return {
        "key": item["key"],
        "ruleset_id": rs.id,
        "name": rs.name,
        "status": rs.status,
        "symbol": item["symbol"],
        "timeframe": item["timeframe"],
        "candles": candle_count,
        "initial_balance": starting_balance,
        "final_balance": result.final_balance,
        "pnl": pnl,
        "return_pct": round(pnl / starting_balance * 100.0, 3) if starting_balance else 0.0,
        "total_trades": result.total_trades,
        "winning_trades": result.winning_trades,
        "losing_trades": result.losing_trades,
        "win_rate": result.win_rate,
        "profit_factor": result.profit_factor,
        "max_drawdown": result.max_drawdown,
        "max_drawdown_pct": round(result.max_drawdown * 100.0, 3),
        "sharpe_ratio": result.sharpe_ratio,
        "equity_curve": result.equity_curve,
        "trades": result.trades,
    }


def _compact_portfolio_evaluation(evaluation: dict) -> dict:
    return {
        "portfolio": evaluation.get("portfolio") or {},
        "strategies": [
            _compact_strategy_result(item)
            for item in evaluation.get("strategies") or []
        ],
    }


def _compact_strategy_result(item: dict) -> dict:
    return {
        key: value
        for key, value in item.items()
        if key not in {"equity_curve", "trades"}
    }


def _portfolio_candidate_summary(combo: tuple[dict, ...], targets: dict, initial_balance: float) -> dict:
    short_portfolio = _combine_candidate_window(combo, "short", initial_balance, targets["short_days"])
    long_portfolio = _combine_candidate_window(combo, "long", initial_balance, targets["long_days"])
    status = score_milestone(short_portfolio, long_portfolio, targets)
    return {
        "score": round(_portfolio_score(combo, targets, initial_balance), 4),
        "status": status,
        "short": _public_portfolio(short_portfolio),
        "long": _public_portfolio(long_portfolio),
        "strategies": [
            {
                "ruleset_id": item["ruleset_id"],
                "name": item["name"],
                "status": item["status"],
                "symbol": item["symbol"],
                "timeframe": item["timeframe"],
                "short_return_pct": item["short"]["return_pct"],
                "long_return_pct": item["long"]["return_pct"],
                "short_trades": item["short"]["total_trades"],
                "long_trades": item["long"]["total_trades"],
                "short_max_drawdown_pct": item["short"]["max_drawdown_pct"],
                "long_max_drawdown_pct": item["long"]["max_drawdown_pct"],
            }
            for item in combo
        ],
    }


def _normalize_ruleset_ids(raw: Any) -> list[int]:
    if raw is None:
        return []
    if isinstance(raw, str):
        raw = [part.strip() for part in raw.split(",") if part.strip()]
    if not isinstance(raw, list):
        return []
    ids = []
    for item in raw:
        try:
            ruleset_id = int(item)
        except (TypeError, ValueError):
            continue
        if ruleset_id > 0 and ruleset_id not in ids:
            ids.append(ruleset_id)
    return ids


def _selected_scope_conflicts(scopes: list[dict]) -> list[str]:
    reasons = []
    seen: dict[tuple[str, str], int] = {}
    for scope in scopes:
        key = (str(scope["symbol"]).upper(), str(scope["timeframe"]).lower())
        previous = seen.get(key)
        ruleset_id = int(scope["ruleset"].id)
        if previous is not None and previous != ruleset_id:
            reasons.append(
                f"Selected rulesets #{previous} and #{ruleset_id} both target {key[0]}/{key[1]}."
            )
        seen[key] = ruleset_id
    return reasons


def _selected_rulesets_summary(rulesets: list[RuleSet]) -> list[dict]:
    return [
        {
            "ruleset_id": int(rs.id),
            "name": rs.name,
            "status": rs.status,
            "symbols": rs.symbols or [],
            "timeframes": rs.timeframes or [],
        }
        for rs in rulesets
    ]


def _public_options(options: dict) -> dict:
    return {
        key: (value.isoformat() if isinstance(value, datetime) else value)
        for key, value in options.items()
        if key != "end_date" or value is not None
    }


def _compact_search_result(search_result: dict | None) -> dict | None:
    if not search_result:
        return None
    best = search_result.get("best_passing")
    return {
        "candidate_count": search_result.get("candidate_count"),
        "prefiltered_scope_count": search_result.get("prefiltered_scope_count"),
        "best_passing": {
            "score": best.get("score"),
            "short": best.get("short"),
            "long": best.get("long"),
            "strategies": best.get("strategies"),
        } if best else None,
    }


def _combine_candidate_window(combo: tuple[dict, ...], window: str, initial_balance: float, days: int) -> dict:
    if not combo:
        now = datetime.now(timezone.utc)
        return combine_scope_results([], initial_balance=initial_balance, days=days, common_start=now, common_end=now)
    scope_results = [item[window] for item in combo]
    start = min(_parse_dt(item["equity_curve"][0]["time"]) for item in scope_results if item["equity_curve"])
    end = max(_parse_dt(item["equity_curve"][-1]["time"]) for item in scope_results if item["equity_curve"])
    return combine_scope_results(scope_results, initial_balance=initial_balance, days=days, common_start=start, common_end=end)


def _public_portfolio(portfolio: dict) -> dict:
    return {
        key: value
        for key, value in portfolio.items()
        if not key.startswith("_")
    }


def _portfolio_score(combo: tuple[dict, ...], targets: dict, initial_balance: float) -> float:
    if not combo:
        return -9999.0
    short_pnl = sum(float(item["short"].get("pnl") or 0.0) for item in combo)
    long_pnl = sum(float(item["long"].get("pnl") or 0.0) for item in combo)
    short_return = short_pnl / initial_balance * 100.0 if initial_balance else 0.0
    long_return = long_pnl / initial_balance * 100.0 if initial_balance else 0.0
    short_trades = sum(int(item["short"].get("total_trades") or 0) for item in combo)
    long_trades = sum(int(item["long"].get("total_trades") or 0) for item in combo)
    avg_short_pf = sum(float(item["short"].get("profit_factor") or 0.0) for item in combo) / len(combo)
    avg_long_pf = sum(float(item["long"].get("profit_factor") or 0.0) for item in combo) / len(combo)
    dd = max(
        max(float(item["short"].get("max_drawdown_pct") or 0.0) for item in combo),
        max(float(item["long"].get("max_drawdown_pct") or 0.0) for item in combo),
    )
    short_gap = short_return - float(targets["min_short_return_pct"])
    long_gap = long_return - float(targets["min_long_return_pct"])
    dd_penalty = max(0.0, dd - float(targets["max_drawdown_pct"])) * 6.0
    trade_penalty = 0.0 if short_trades and long_trades else 5.0
    passed = (
        short_return >= float(targets["min_short_return_pct"])
        and long_return >= float(targets["min_long_return_pct"])
        and dd <= float(targets["max_drawdown_pct"])
        and short_trades >= int(targets["min_short_trades"])
        and long_trades >= int(targets["min_long_trades"])
    )
    return (
        long_gap * 2.0
        + short_gap * 1.5
        + min(avg_long_pf, 3.0) * 0.8
        + min(avg_short_pf, 3.0) * 0.5
        - dd * 0.25
        - dd_penalty
        - trade_penalty
        + (8.0 if passed else 0.0)
    )


def _candidate_score(candidate: dict, targets: dict) -> float:
    short = candidate["short"]
    long = candidate["long"]
    return (
        float(long["return_pct"]) * 2.0
        + float(short["return_pct"]) * 1.5
        + min(float(long["profit_factor"]), 4.0) * 0.7
        + min(float(short["profit_factor"]), 4.0) * 0.4
        - max(float(long["max_drawdown_pct"]), float(short["max_drawdown_pct"])) * 0.35
        + min(float(long["total_trades"]), 30.0) * 0.02
    )


def _score_window(metrics: dict, min_return_pct: float, max_drawdown_pct: float, min_trades: int) -> dict:
    reasons = []
    if float(metrics.get("return_pct") or 0.0) < float(min_return_pct):
        reasons.append(
            f"return {float(metrics.get('return_pct') or 0.0):.2f}% below {float(min_return_pct):.2f}%"
        )
    if float(metrics.get("max_drawdown_pct") or 0.0) > float(max_drawdown_pct):
        reasons.append(
            f"drawdown {float(metrics.get('max_drawdown_pct') or 0.0):.2f}% exceeds {float(max_drawdown_pct):.2f}%"
        )
    if int(metrics.get("total_trades") or 0) < int(min_trades):
        reasons.append(f"trades {int(metrics.get('total_trades') or 0)} below {int(min_trades)}")
    return {"passed": not reasons, "status": "passed" if not reasons else "failed", "reasons": reasons}


def _combine_equity_curves(
    curves_by_key: dict[str, list[dict]],
    starting_balance_by_key: dict[str, float],
    initial_balance: float,
) -> list[dict]:
    times = sorted({point["time"] for curve in curves_by_key.values() for point in curve}, key=_parse_dt)
    last_equity = {
        key: float(starting_balance_by_key.get(key, initial_balance))
        for key in curves_by_key
    }
    index_by_key = {key: 0 for key in curves_by_key}
    combined = []
    for ts in times:
        for key, curve in curves_by_key.items():
            idx = index_by_key[key]
            while idx < len(curve) and _parse_dt(curve[idx]["time"]) <= _parse_dt(ts):
                last_equity[key] = float(curve[idx]["equity"])
                idx += 1
            index_by_key[key] = idx
        equity = initial_balance + sum(
            equity - float(starting_balance_by_key.get(key, initial_balance))
            for key, equity in last_equity.items()
        )
        combined.append({"time": ts, "equity": round(equity, 2)})
    return combined


def _combine_pnl_curves(
    curves_by_key: dict[str, list[dict]],
    starting_balance_by_key: dict[str, float],
) -> list[dict]:
    times = sorted({point["time"] for curve in curves_by_key.values() for point in curve}, key=_parse_dt)
    last_equity = {
        key: float(starting_balance_by_key.get(key, 0.0))
        for key in curves_by_key
    }
    index_by_key = {key: 0 for key in curves_by_key}
    combined = []
    for ts in times:
        for key, curve in curves_by_key.items():
            idx = index_by_key[key]
            while idx < len(curve) and _parse_dt(curve[idx]["time"]) <= _parse_dt(ts):
                last_equity[key] = float(curve[idx]["equity"])
                idx += 1
            index_by_key[key] = idx
        pnl = sum(
            equity - float(starting_balance_by_key.get(key, 0.0))
            for key, equity in last_equity.items()
        )
        combined.append({"time": ts, "pnl": round(pnl, 2)})
    return combined


def _symbol_curves(scope_results: list[dict]) -> list[dict]:
    curves_by_symbol: dict[str, dict[str, list[dict]]] = {}
    starts_by_symbol: dict[str, dict[str, float]] = {}
    for item in scope_results:
        curves_by_symbol.setdefault(item["symbol"], {})[item["key"]] = item["equity_curve"]
        starts_by_symbol.setdefault(item["symbol"], {})[item["key"]] = float(item["initial_balance"])
    return [
        {
            "symbol": symbol,
            "points": _combine_pnl_curves(curves, starts_by_symbol.get(symbol, {})),
        }
        for symbol, curves in sorted(curves_by_symbol.items())
    ]


def _max_drawdown_from_curve(equity_curve: list[dict]) -> float:
    peak = None
    max_drawdown = 0.0
    for point in equity_curve:
        equity = float(point.get("equity") or 0.0)
        if peak is None or equity > peak:
            peak = equity
        if peak and peak > 0:
            max_drawdown = max(max_drawdown, (peak - equity) / peak)
    return round(max_drawdown, 4)


def _parse_dt(value: Any) -> datetime:
    if isinstance(value, datetime):
        dt = value
    else:
        dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _coerce_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, str):
        return value.strip().lower() not in {"0", "false", "no", "off", ""}
    return bool(value)
