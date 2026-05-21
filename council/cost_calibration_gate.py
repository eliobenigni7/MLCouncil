"""Promotion gate for self-calibrating transaction costs (ADR-0003).

Runs after ``cost_calibration_artifact`` materializes: lightweight cost A/B on
cached strategy weights, fill-quality checks, then auto-revert via
``config/runtime_override.env`` when validation fails.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from backtest.validation import (
    revert_to_static_cost_calibration,
    validate_cost_calibration_promotion,
)
from council.cost_calibration import (
    DEFAULT_CALIBRATION_PATH,
    compute_is_bps,
    load_calibration,
)
from council.transaction_costs import estimate_slippage_bps

logger = logging.getLogger(__name__)

_ROOT = Path(__file__).resolve().parents[1]
_RESULTS_DIR = _ROOT / "data" / "results"
_GATE_REPORT_PATH = _ROOT / "data" / "operations" / "cost_calibration_gate.json"
_ORDERS_DIR = _ROOT / "data" / "orders"


def _latest_orders_lineage(daily_orders: Optional[pd.DataFrame] = None) -> dict[str, str]:
    if daily_orders is not None and not daily_orders.empty:
        row = daily_orders.iloc[0]
        return {
            "pipeline_run_id": str(row.get("pipeline_run_id", "") or ""),
            "cost_calibration_version": str(row.get("cost_calibration_version", "") or ""),
        }
    if not _ORDERS_DIR.exists():
        return {}
    files = sorted(_ORDERS_DIR.glob("*.parquet"))
    if not files:
        return {}
    df = pd.read_parquet(files[-1])
    if df.empty:
        return {}
    row = df.iloc[0]
    return {
        "pipeline_run_id": str(row.get("pipeline_run_id", "") or ""),
        "cost_calibration_version": str(row.get("cost_calibration_version", "") or ""),
    }


def _load_backtest_inputs(root: Path) -> tuple[pd.DataFrame, pd.DataFrame] | None:
    weights_path = root / "data" / "results" / "strategy_weights.parquet"
    returns_path = root / "data" / "results" / "walk_forward_oos_returns.parquet"

    if not weights_path.exists():
        return None

    weights = pd.read_parquet(weights_path)
    if isinstance(weights.index, pd.DatetimeIndex):
        weights.index = pd.to_datetime(weights.index)
    else:
        weights.index = pd.to_datetime(weights.index)

    if returns_path.exists():
        returns = pd.read_parquet(returns_path)
        if returns.shape[1] == 1 and returns.columns[0] in {"oos_return", "return", 0}:
            col = returns.columns[0]
            returns = returns.rename(columns={col: "portfolio"})
            returns = returns.to_frame() if returns.ndim == 1 else returns
        if returns.shape[1] == 1:
            # Single-column OOS returns: broadcast to weight columns for cost-only sim
            col = returns.columns[0]
            returns_wide = pd.DataFrame(
                {t: returns[col].values for t in weights.columns},
                index=returns.index,
            )
        else:
            returns_wide = returns
        returns_wide.index = pd.to_datetime(returns_wide.index)
        return weights, returns_wide

    return None


def _fill_quality_metrics() -> tuple[Optional[float], Optional[float]]:
    try:
        from execution.fill_log import read_fills

        fills = read_fills()
        if fills.height == 0:
            return None, None
        fills = compute_is_bps(fills)
        pdf = fills.to_pandas()
        if "is_bps" not in pdf.columns or pdf["is_bps"].isna().all():
            return None, None
        median_is = float(pdf["is_bps"].median())
        if "slippage_bps_assumed" in pdf.columns:
            median_lookup = float(pdf["slippage_bps_assumed"].median())
        else:
            median_lookup = float(
                pdf["ticker"].map(estimate_slippage_bps).median()
            )
        return median_is, median_lookup
    except Exception as exc:  # noqa: BLE001
        logger.warning("Fill quality metrics unavailable: %s", exc)
        return None, None


def run_cost_calibration_promotion_gate(
    *,
    calibration_summary: dict[str, Any],
    daily_orders: Optional[pd.DataFrame] = None,
    root: Path | None = None,
    min_fills_per_tier: int = 30,
) -> dict[str, Any]:
    """Evaluate promotion; revert to static lookup automatically on failure."""
    base = root or _ROOT
    lineage = _latest_orders_lineage(daily_orders)
    report: dict[str, Any] = {
        "status": "skipped",
        "lineage": lineage,
        "calibration_summary": calibration_summary,
        "promotion_passed": None,
        "reasons": [],
        "reverted": False,
        "override_path": "",
    }

    if calibration_summary.get("status") != "ok":
        report["status"] = "skipped_no_calibration"
        report["reasons"] = [
            f"Calibration artifact status={calibration_summary.get('status')!r}; gate not applied."
        ]
        _write_gate_report(report, base)
        return report

    calib_path = DEFAULT_CALIBRATION_PATH
    if not calib_path.exists():
        report["status"] = "failed"
        report["reasons"] = ["cost_calibration.json missing after artifact run"]
        revert_to_static_cost_calibration(base, reason="; ".join(report["reasons"]))
        report["reverted"] = True
        report["override_path"] = str(base / "config" / "runtime_override.env")
        _write_gate_report(report, base)
        return report

    try:
        artifact = load_calibration(calib_path)
    except Exception as exc:  # noqa: BLE001
        report["status"] = "failed"
        report["reasons"] = [f"Cannot load calibration: {exc}"]
        revert_to_static_cost_calibration(base, reason=report["reasons"][0])
        report["reverted"] = True
        report["override_path"] = str(base / "config" / "runtime_override.env")
        _write_gate_report(report, base)
        return report

    backtest_inputs = _load_backtest_inputs(base)
    if backtest_inputs is None:
        report["status"] = "skipped_no_backtest_artifacts"
        report["reasons"] = [
            "Missing data/results/strategy_weights.parquet for cost A/B; "
            "run scripts/run_strategy_backtest.py first."
        ]
        _write_gate_report(report, base)
        return report

    weights, forward_returns = backtest_inputs
    from backtest.simulator import compare_cost_modes

    ab = compare_cost_modes(
        weights=weights,
        forward_returns=forward_returns.reindex(weights.index).fillna(0.0),
        initial_capital=100_000.0,
    )
    median_is, median_lookup = _fill_quality_metrics()

    result = validate_cost_calibration_promotion(
        ab["static_stats"],
        ab["calibrated_stats"],
        artifact=artifact,
        median_is_bps=median_is,
        median_lookup_bps=median_lookup,
        min_fills_per_tier=min_fills_per_tier,
    )

    report["ab_summary"] = {
        "net_sharpe_static_costs": ab["net_sharpe_static_costs"],
        "net_sharpe_calibrated_costs": ab["net_sharpe_calibrated_costs"],
        "net_sharpe_delta": ab["net_sharpe_delta"],
        "median_is_bps": median_is,
        "median_lookup_bps": median_lookup,
    }
    report["promotion_passed"] = result.passed
    report["reasons"] = result.reasons

    if result.passed:
        report["status"] = "promoted"
        logger.info("Cost calibration promotion gate PASSED")
    else:
        report["status"] = "reverted"
        override = revert_to_static_cost_calibration(
            base,
            reason="; ".join(result.reasons),
        )
        report["reverted"] = True
        report["override_path"] = str(override)
        logger.warning(
            "Cost calibration promotion gate FAILED — reverted to static lookup: %s",
            result.reasons,
        )

    _write_gate_report(report, base)
    return report


def _write_gate_report(report: dict[str, Any], root: Path) -> None:
    path = root / "data" / "operations" / "cost_calibration_gate.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
