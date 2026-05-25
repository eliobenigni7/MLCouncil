"""Sharpe remediation ablation protocol (E.1 baseline + E.2 intervention matrix).

Runs ``one_year_backtest`` with optional env interventions and writes JSON
reports for before/after comparisons.

Example::

    python scripts/run_sharpe_ablation.py --help
    python scripts/run_sharpe_ablation.py --year-start 2024-01-01 --year-end 2025-01-01
    python scripts/run_sharpe_ablation.py --ledoit-wolf --cumulative
    python scripts/run_sharpe_ablation.py --moe --dcc --cumulative --output data/reports/ablation.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.one_year_backtest import run_one_year_backtest  # noqa: E402

# Intervention name → env overrides applied for that flag (baseline unchanged when absent).
INTERVENTION_ENV: dict[str, dict[str, str]] = {
    "ledoit_wolf": {"MLCOUNCIL_COVARIANCE_ESTIMATOR": "ledoit"},
    "dcc": {"MLCOUNCIL_COVARIANCE_ESTIMATOR": "dcc"},
    "moe": {"MLCOUNCIL_AGGREGATOR_MODE": "moe"},
    "linear_agg": {"MLCOUNCIL_AGGREGATOR_MODE": "linear"},
    "conformal": {"MLCOUNCIL_POSITION_SIZING": "conformal"},
    "cqr": {"MLCOUNCIL_POSITION_SIZING": "cqr"},
    "hrp_prior": {"MLCOUNCIL_HRP_SOFT_PRIOR": "true"},
    "mv_objective": {"MLCOUNCIL_TC_LAMBDA": "3.0", "MLCOUNCIL_RISK_LAMBDA": "1.0"},
    "tight_turnover": {"MLCOUNCIL_MAX_TURNOVER": "0.15"},
    "diff_portfolio": {"MLCOUNCIL_PORTFOLIO_MODE": "diff"},
    "stacking_shadow": {"MLCOUNCIL_STACKING_SHADOW": "true"},
}

INTERVENTION_FLAGS: dict[str, str] = {
    "ledoit_wolf": "--ledoit-wolf",
    "dcc": "--dcc",
    "moe": "--moe",
    "linear_agg": "--linear-agg",
    "conformal": "--conformal",
    "cqr": "--cqr",
    "hrp_prior": "--hrp-prior",
    "mv_objective": "--mv-objective",
    "tight_turnover": "--tight-turnover",
    "diff_portfolio": "--diff-portfolio",
    "stacking_shadow": "--stacking-shadow",
}

SCHEMA_VERSION = "1.0"


def _add_intervention_args(parser: argparse.ArgumentParser) -> None:
    for name, flag in INTERVENTION_FLAGS.items():
        parser.add_argument(
            flag,
            dest=name,
            action="store_true",
            help=f"Enable intervention: {name} ({INTERVENTION_ENV[name]})",
        )


def _selected_interventions(args: argparse.Namespace) -> list[str]:
    return [name for name in INTERVENTION_FLAGS if getattr(args, name, False)]


def _env_for_interventions(names: list[str], *, cumulative: bool) -> dict[str, str]:
    if not names:
        return {}
    ordered = names if cumulative else [names[-1]]
    merged: dict[str, str] = {}
    for name in ordered:
        merged.update(INTERVENTION_ENV.get(name, {}))
    return merged


def _apply_env(overrides: dict[str, str]) -> dict[str, str | None]:
    """Apply env overrides; return prior values for restoration."""
    prior: dict[str, str | None] = {}
    for key, value in overrides.items():
        prior[key] = os.environ.get(key)
        os.environ[key] = value
    return prior


def _restore_env(prior: dict[str, str | None]) -> None:
    for key, value in prior.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


def build_report_payload(
    *,
    label: str,
    window: dict[str, str],
    config: dict[str, Any],
    interventions: list[str],
    env_applied: dict[str, str],
    metrics: dict[str, Any],
) -> dict[str, Any]:
    """E.2 JSON output schema (versioned)."""
    return {
        "schema_version": SCHEMA_VERSION,
        "label": label,
        "window": window,
        "config": config,
        "interventions": interventions,
        "env_applied": env_applied,
        "metrics": metrics,
        "status": "error" if metrics.get("error") else "ok",
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Sharpe ablation: baseline + optional intervention env flags"
    )
    parser.add_argument("--year-start", default="2024-01-01")
    parser.add_argument("--year-end", default="2025-01-01")
    parser.add_argument("--train-months", type=int, default=6)
    parser.add_argument("--rebalance-every", type=int, default=3)
    parser.add_argument("--max-turnover", type=float, default=0.15)
    parser.add_argument(
        "--cumulative",
        action="store_true",
        help="When multiple intervention flags are set, apply all in flag order (not last-only).",
    )
    parser.add_argument(
        "--baseline-only",
        action="store_true",
        help="Ignore intervention flags; export baseline metrics only.",
    )
    parser.add_argument(
        "--output",
        default=str(ROOT / "data" / "reports" / "baseline_2026Q2.json"),
    )
    parser.add_argument(
        "--list-interventions",
        action="store_true",
        help="Print available intervention keys and exit.",
    )
    _add_intervention_args(parser)
    args = parser.parse_args()

    if args.list_interventions:
        for name, flag in INTERVENTION_FLAGS.items():
            print(f"{flag:22} {name:16} {INTERVENTION_ENV[name]}")
        return 0

    selected = [] if args.baseline_only else _selected_interventions(args)
    env_overrides = _env_for_interventions(selected, cumulative=args.cumulative)

    run_config = {
        "train_months": args.train_months,
        "rebalance_every": args.rebalance_every,
        "max_turnover": args.max_turnover,
        "cumulative": args.cumulative,
        "baseline_only": args.baseline_only,
    }

    prior = _apply_env(env_overrides)
    try:
        stats = run_one_year_backtest(
            args.year_start,
            args.year_end,
            train_window_months=args.train_months,
            rebalance_every=args.rebalance_every,
            max_turnover_env=args.max_turnover,
            force_linear=env_overrides.get("MLCOUNCIL_AGGREGATOR_MODE", "linear") != "moe",
        )
    finally:
        _restore_env(prior)

    label = "sharpe_ablation_baseline" if not selected else f"sharpe_ablation_{'+'.join(selected)}"
    payload = build_report_payload(
        label=label,
        window={"start": args.year_start, "end": args.year_end},
        config=run_config,
        interventions=selected,
        env_applied=env_overrides,
        metrics=stats,
    )

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(f"Wrote {out}")
    if stats.get("error"):
        print(f"ERROR: {stats['error']}", file=sys.stderr)
        return 1
    print(f"Sharpe: {stats.get('sharpe', 'n/a')} | interventions: {selected or 'none'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
