"""Evidently-based drift and model performance reports.

Two public functions
--------------------
generate_drift_report(reference, current, output_path)
    Dataset-level drift overview + per-feature KS test + distribution plots.
    Saves an HTML report and returns a summary dict.

generate_model_performance_report(y_pred, y_true, reference_pred, reference_true, output_path)
    Compares regression performance today vs reference period.
    Saves an HTML report and returns a summary dict.

Both functions fall back to a lightweight scipy-based implementation when
the ``evidently`` package is not installed (e.g. in CI without the full
dependency set).

Output paths
------------
HTML reports are written to:
    data/monitoring/reports/{date}_drift.html
    data/monitoring/reports/{date}_performance.html

Usage
-----
    from council.evidently_reports import generate_drift_report

    summary = generate_drift_report(
        reference=baseline_df,
        current=today_df,
        output_path="data/monitoring/reports/2024-01-15_drift.html",
    )
    print(summary["n_drifted_features"], summary["dataset_drift"])
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

# Optional evidently import
try:
    from evidently import ColumnMapping
    from evidently.metric_preset import DataDriftPreset, RegressionPreset
    from evidently.report import Report
    _EVIDENTLY_AVAILABLE = True
except ImportError:  # pragma: no cover
    _EVIDENTLY_AVAILABLE = False
    logger.debug(
        "evidently not installed — drift reports will use scipy fallback. "
        "Install with: pip install evidently"
    )

_MONITORING_DIR = Path(__file__).parents[1] / "data" / "monitoring"
_REPORTS_DIR = _MONITORING_DIR / "reports"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_drift_report(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    """Generate an Evidently dataset-drift report (HTML + summary dict).

    Compares ``current`` against ``reference`` using the DataDriftPreset
    (KS test for numerical columns, chi-squared for categoricals).

    Parameters
    ----------
    reference:
        Reference / baseline DataFrame (e.g. rolling 30–60 days of features).
    current:
        Today's feature DataFrame.
    output_path:
        Where to save the HTML report.  If None, defaults to
        ``data/monitoring/reports/{today}_drift.html``.

    Returns
    -------
    dict with keys:
        n_features        — total number of numeric features tested
        n_drifted_features — features flagged as drifted
        dataset_drift     — True if majority of features drifted
        drift_fraction    — fraction of drifted features
        drifted_columns   — list of drifted column names
        report_path       — path to the saved HTML file (str)
    """
    output_path = _resolve_output_path(output_path, suffix="drift")

    if _EVIDENTLY_AVAILABLE:
        return _evidently_drift_report(reference, current, output_path)
    return _scipy_drift_report(reference, current, output_path)


def generate_model_performance_report(
    y_pred: pd.Series,
    y_true: pd.Series,
    reference_pred: pd.Series,
    reference_true: pd.Series,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    """Generate an Evidently regression performance comparison report.

    Compares today's prediction quality against a reference period to
    identify statistically significant model degradation.

    Parameters
    ----------
    y_pred:
        Today's model predictions (pd.Series).
    y_true:
        Today's realized outcomes (pd.Series).
    reference_pred:
        Reference period predictions.
    reference_true:
        Reference period realized outcomes.
    output_path:
        HTML output path.  Defaults to
        ``data/monitoring/reports/{today}_performance.html``.

    Returns
    -------
    dict with keys:
        current_mae, current_rmse, current_ic
        reference_mae, reference_rmse, reference_ic
        mae_change_pct, rmse_change_pct, ic_change_pct
        significant_degradation — True if RMSE worsened > 20 %
        report_path
    """
    output_path = _resolve_output_path(output_path, suffix="performance")

    if _EVIDENTLY_AVAILABLE:
        return _evidently_performance_report(
            y_pred, y_true, reference_pred, reference_true, output_path
        )
    return _scipy_performance_report(
        y_pred, y_true, reference_pred, reference_true, output_path
    )


# ---------------------------------------------------------------------------
# Evidently implementations
# ---------------------------------------------------------------------------

def _evidently_drift_report(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    output_path: Path,
) -> dict[str, Any]:
    """Full Evidently HTML drift report with KS test per feature."""
    # Evidently requires a shared column set
    shared_cols = [c for c in reference.columns if c in current.columns]
    ref = reference[shared_cols].copy()
    cur = current[shared_cols].copy()

    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=ref, current_data=cur, column_mapping=ColumnMapping())

    _ensure_reports_dir()
    report.save_html(str(output_path))
    logger.info(f"Evidently drift report saved → {output_path}")

    # Extract summary from the report dict
    report_dict = report.as_dict()
    drift_metrics = _extract_evidently_drift_summary(report_dict, shared_cols)
    drift_metrics["report_path"] = str(output_path)
    return drift_metrics


def _evidently_performance_report(
    y_pred: pd.Series,
    y_true: pd.Series,
    reference_pred: pd.Series,
    reference_true: pd.Series,
    output_path: Path,
) -> dict[str, Any]:
    """Full Evidently HTML regression performance report."""
    current_df = pd.DataFrame({"target": y_true.values, "prediction": y_pred.values})
    reference_df = pd.DataFrame(
        {"target": reference_true.values, "prediction": reference_pred.values}
    )

    col_mapping = ColumnMapping(target="target", prediction="prediction")
    report = Report(metrics=[RegressionPreset()])
    report.run(
        reference_data=reference_df,
        current_data=current_df,
        column_mapping=col_mapping,
    )

    _ensure_reports_dir()
    report.save_html(str(output_path))
    logger.info(f"Evidently performance report saved → {output_path}")

    summary = _compute_performance_summary(
        y_pred, y_true, reference_pred, reference_true
    )
    summary["report_path"] = str(output_path)
    return summary


def _extract_evidently_drift_summary(
    report_dict: dict,
    columns: list[str],
) -> dict[str, Any]:
    """Parse Evidently report dict to extract per-feature drift flags."""
    try:
        metrics = report_dict.get("metrics", [])
        drifted_cols: list[str] = []
        for metric in metrics:
            result = metric.get("result", {})
            drift_by_col = result.get("drift_by_columns", {})
            for col, info in drift_by_col.items():
                if isinstance(info, dict) and info.get("drift_detected", False):
                    drifted_cols.append(col)

        n_features = len(columns)
        n_drifted = len(drifted_cols)
        drift_fraction = n_drifted / n_features if n_features > 0 else 0.0
        dataset_drift = drift_fraction > 0.5

        return {
            "n_features": n_features,
            "n_drifted_features": n_drifted,
            "dataset_drift": dataset_drift,
            "drift_fraction": drift_fraction,
            "drifted_columns": drifted_cols,
        }
    except Exception as exc:  # noqa: BLE001
        logger.debug(f"Could not parse Evidently report dict: {exc}")
        return {
            "n_features": len(columns),
            "n_drifted_features": 0,
            "dataset_drift": False,
            "drift_fraction": 0.0,
            "drifted_columns": [],
        }


# ---------------------------------------------------------------------------
# Scipy fallback implementations
# ---------------------------------------------------------------------------

def _scipy_drift_report(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    output_path: Path,
) -> dict[str, Any]:
    """Lightweight KS-test-based drift report (no evidently dependency)."""
    from scipy.stats import ks_2samp

    shared_numeric = [
        c for c in reference.columns
        if c in current.columns
        and pd.api.types.is_numeric_dtype(reference[c])
    ]

    drifted_cols: list[str] = []
    feature_stats: list[dict] = []
    for col in shared_numeric:
        ref_vals = reference[col].dropna().values
        cur_vals = current[col].dropna().values
        if len(ref_vals) < 5 or len(cur_vals) < 5:
            continue
        stat, p_value = ks_2samp(ref_vals, cur_vals)
        drifted = p_value < 0.05
        if drifted:
            drifted_cols.append(col)
        feature_stats.append({
            "column": col,
            "ks_statistic": round(float(stat), 4),
            "p_value": round(float(p_value), 4),
            "drifted": drifted,
            "ref_mean": round(float(np.mean(ref_vals)), 4),
            "cur_mean": round(float(np.mean(cur_vals)), 4),
            "ref_std": round(float(np.std(ref_vals)), 4),
            "cur_std": round(float(np.std(cur_vals)), 4),
        })

    n_features = len(feature_stats)
    n_drifted = len(drifted_cols)
    drift_fraction = n_drifted / n_features if n_features > 0 else 0.0
    dataset_drift = drift_fraction > 0.5

    # Save minimal HTML
    _ensure_reports_dir()
    _save_fallback_drift_html(feature_stats, output_path)

    return {
        "n_features": n_features,
        "n_drifted_features": n_drifted,
        "dataset_drift": dataset_drift,
        "drift_fraction": round(drift_fraction, 4),
        "drifted_columns": drifted_cols,
        "feature_stats": feature_stats,
        "report_path": str(output_path),
    }


def _scipy_performance_report(
    y_pred: pd.Series,
    y_true: pd.Series,
    reference_pred: pd.Series,
    reference_true: pd.Series,
    output_path: Path,
) -> dict[str, Any]:
    """Compute performance metrics without Evidently."""
    summary = _compute_performance_summary(
        y_pred, y_true, reference_pred, reference_true
    )

    _ensure_reports_dir()
    _save_fallback_performance_html(summary, output_path)
    summary["report_path"] = str(output_path)
    return summary


def _compute_performance_summary(
    y_pred: pd.Series,
    y_true: pd.Series,
    reference_pred: pd.Series,
    reference_true: pd.Series,
) -> dict[str, Any]:
    """Compute MAE, RMSE, and Spearman IC for current vs reference."""
    from scipy.stats import spearmanr

    def _metrics(pred: pd.Series, true: pd.Series) -> dict[str, float]:
        residuals = true.values - pred.values
        mae = float(np.mean(np.abs(residuals)))
        rmse = float(np.sqrt(np.mean(residuals ** 2)))
        ic, _ = spearmanr(pred.values, true.values)
        return {
            "mae": round(mae, 6),
            "rmse": round(rmse, 6),
            "ic": round(float(ic) if not np.isnan(ic) else 0.0, 4),
        }

    cur = _metrics(y_pred, y_true)
    ref = _metrics(reference_pred, reference_true)

    mae_change_pct = (cur["mae"] - ref["mae"]) / (ref["mae"] + 1e-9) * 100
    rmse_change_pct = (cur["rmse"] - ref["rmse"]) / (ref["rmse"] + 1e-9) * 100
    ic_change_pct = (cur["ic"] - ref["ic"]) / (abs(ref["ic"]) + 1e-9) * 100
    significant_degradation = rmse_change_pct > 20.0

    return {
        "current_mae": cur["mae"],
        "current_rmse": cur["rmse"],
        "current_ic": cur["ic"],
        "reference_mae": ref["mae"],
        "reference_rmse": ref["rmse"],
        "reference_ic": ref["ic"],
        "mae_change_pct": round(mae_change_pct, 2),
        "rmse_change_pct": round(rmse_change_pct, 2),
        "ic_change_pct": round(ic_change_pct, 2),
        "significant_degradation": significant_degradation,
    }


# ---------------------------------------------------------------------------
# HTML helpers (fallback reports)
# ---------------------------------------------------------------------------

def _save_fallback_drift_html(
    feature_stats: list[dict],
    output_path: Path,
) -> None:
    """Write a minimal HTML table summarising per-feature drift."""
    rows = "".join(
        f"<tr style='background:{'#fdd' if f['drifted'] else 'white'}'>"
        f"<td>{f['column']}</td>"
        f"<td>{f['ks_statistic']}</td>"
        f"<td>{f['p_value']}</td>"
        f"<td>{'<b>YES</b>' if f['drifted'] else 'no'}</td>"
        f"<td>{f['ref_mean']} ± {f['ref_std']}</td>"
        f"<td>{f['cur_mean']} ± {f['cur_std']}</td>"
        f"</tr>"
        for f in feature_stats
    )
    html = (
        "<html><head><title>Feature Drift Report</title></head><body>"
        "<h2>Feature Drift Report (KS test)</h2>"
        "<table border='1' cellpadding='4' style='border-collapse:collapse'>"
        "<tr><th>Feature</th><th>KS stat</th><th>p-value</th><th>Drifted?</th>"
        "<th>Ref mean±std</th><th>Cur mean±std</th></tr>"
        f"{rows}"
        "</table></body></html>"
    )
    try:
        output_path.write_text(html)
        logger.info(f"Fallback drift report saved → {output_path}")
    except OSError as exc:
        logger.warning(f"Could not save fallback drift report: {exc}")


def _save_fallback_performance_html(
    summary: dict[str, Any],
    output_path: Path,
) -> None:
    """Write a minimal HTML table summarising performance comparison."""
    rows = "".join(
        f"<tr><td>{k}</td><td>{v}</td></tr>"
        for k, v in summary.items()
        if k != "report_path"
    )
    html = (
        "<html><head><title>Model Performance Report</title></head><body>"
        "<h2>Model Performance Report</h2>"
        "<table border='1' cellpadding='4' style='border-collapse:collapse'>"
        "<tr><th>Metric</th><th>Value</th></tr>"
        f"{rows}"
        "</table></body></html>"
    )
    try:
        output_path.write_text(html)
        logger.info(f"Fallback performance report saved → {output_path}")
    except OSError as exc:
        logger.warning(f"Could not save fallback performance report: {exc}")


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _resolve_output_path(
    output_path: str | Path | None,
    suffix: str,
) -> Path:
    """Return a resolved output path, defaulting to the reports directory."""
    if output_path is not None:
        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    from datetime import date
    _ensure_reports_dir()
    return _REPORTS_DIR / f"{date.today().isoformat()}_{suffix}.html"


def _ensure_reports_dir() -> None:
    _REPORTS_DIR.mkdir(parents=True, exist_ok=True)
