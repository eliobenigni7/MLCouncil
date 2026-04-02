"""CouncilMonitor — daily alpha decay, feature drift, SHAP stability, regime change.

Three families of checks
------------------------
1. Alpha decay      IC rolling mean drops below threshold for 5+ consecutive days.
2. Feature drift    KS test on each feature column; alert if >20 % of top-10 SHAP
                    features have p-value < 0.05.
3. SHAP stability   Overlap of today's top-10 features with the 30-day baseline;
                    alert if overlap falls below 70 %.
4. Regime change    HMM emits a new state AND transition probability > 0.70.

All results are returned as ``AlertResult`` instances (see alerts.py).
``run_daily_checks`` optionally logs scalar metrics to MLflow when available.

Usage
-----
    monitor = CouncilMonitor()

    # Individual checks
    result = monitor.check_alpha_decay("lgbm", ic_series, window=30)

    # Full daily run (logs to MLflow, returns all alerts)
    alerts = monitor.run_daily_checks(date.today())
"""

from __future__ import annotations

from datetime import date
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

try:
    import polars as pl
    _POLARS_AVAILABLE = True
except ImportError:  # pragma: no cover
    _POLARS_AVAILABLE = False

try:
    import mlflow
    _MLFLOW_AVAILABLE = True
except ImportError:  # pragma: no cover
    _MLFLOW_AVAILABLE = False

from council.alerts import AlertResult, AlertDispatcher, Severity


# ---------------------------------------------------------------------------
# CouncilMonitor
# ---------------------------------------------------------------------------

class CouncilMonitor:
    """Monitor alpha decay, feature drift, SHAP stability, and regime changes.

    Parameters
    ----------
    ic_threshold:
        Minimum acceptable rolling-IC value. Default 0.01.
    shap_stability_threshold:
        Maximum fraction of top-10 features that may change before an alert
        is raised (0.30 means >30 % turnover → alert). Default 0.30.
    drift_pvalue_threshold:
        KS-test p-value below which a single feature is considered drifted.
        Default 0.05.
    ic_alert_consecutive_days:
        Number of consecutive days IC must stay below threshold before an
        alert fires. Default 5.
    drift_feature_fraction:
        Fraction of monitored features that must be drifted to trigger the
        feature-drift alert. Default 0.20.
    shap_overlap_min:
        Minimum required overlap fraction between today's top-10 SHAP features
        and the 30-day baseline set. Default 0.70.
    regime_transition_prob_min:
        Minimum transition probability required (alongside a regime change) to
        fire a regime-change alert. Default 0.70.
    """

    def __init__(
        self,
        ic_threshold: float = 0.01,
        shap_stability_threshold: float = 0.30,
        drift_pvalue_threshold: float = 0.05,
        ic_alert_consecutive_days: int = 5,
        drift_feature_fraction: float = 0.20,
        shap_overlap_min: float = 0.70,
        regime_transition_prob_min: float = 0.70,
    ) -> None:
        self.ic_threshold = ic_threshold
        self.shap_stability_threshold = shap_stability_threshold
        self.drift_pvalue_threshold = drift_pvalue_threshold
        self.ic_alert_consecutive_days = ic_alert_consecutive_days
        self.drift_feature_fraction = drift_feature_fraction
        self.shap_overlap_min = shap_overlap_min
        self.regime_transition_prob_min = regime_transition_prob_min

    # ------------------------------------------------------------------
    # 1. Alpha decay
    # ------------------------------------------------------------------

    def check_alpha_decay(
        self,
        model_name: str,
        ic_history: pd.Series,
        window: int = 30,
    ) -> AlertResult:
        """Detect alpha decay from rolling IC history.

        The check fires when the rolling IC mean (over ``window`` days) has
        stayed below ``ic_threshold`` for at least ``ic_alert_consecutive_days``
        consecutive days.

        Parameters
        ----------
        model_name:
            Model identifier (e.g. "lgbm", "sentiment", "hmm").
        ic_history:
            DatetimeIndex or integer-indexed Series of daily IC values.
        window:
            Rolling window length (days) for computing the rolling mean IC.

        Returns
        -------
        AlertResult
        """
        if len(ic_history) < window:
            return AlertResult(
                is_alert=False,
                severity=Severity.INFO,
                model_name=model_name,
                check_type="alpha_decay",
                message=f"Insufficient IC history ({len(ic_history)} < {window} days).",
                recommendation="Wait for more data before drawing conclusions.",
                metric_value=float(ic_history.mean()) if len(ic_history) > 0 else float("nan"),
                threshold=self.ic_threshold,
            )

        rolling_ic = ic_history.rolling(window=window, min_periods=window).mean()
        # Drop leading NaNs
        rolling_ic_valid = rolling_ic.dropna()

        if rolling_ic_valid.empty:
            return AlertResult(
                is_alert=False,
                severity=Severity.INFO,
                model_name=model_name,
                check_type="alpha_decay",
                message="No valid rolling IC values to evaluate.",
                recommendation="Check IC computation pipeline.",
                metric_value=float("nan"),
                threshold=self.ic_threshold,
            )

        latest_ic = float(rolling_ic_valid.iloc[-1])

        # Count consecutive days at the tail where IC < threshold
        below_threshold = rolling_ic_valid < self.ic_threshold
        consecutive = _count_trailing_true(below_threshold)

        if consecutive >= self.ic_alert_consecutive_days:
            severity = (
                Severity.CRITICAL if consecutive >= self.ic_alert_consecutive_days * 2
                else Severity.WARNING
            )
            return AlertResult(
                is_alert=True,
                severity=severity,
                model_name=model_name,
                check_type="alpha_decay",
                message=(
                    f"{model_name}: rolling IC ({window}d) = {latest_ic:.4f} has been below "
                    f"threshold {self.ic_threshold:.4f} for {consecutive} consecutive days."
                ),
                recommendation=(
                    f"Consider retraining {model_name} on a more recent data window. "
                    "Review feature set for stale signals and check for structural break "
                    "in the target variable."
                ),
                metric_value=latest_ic,
                threshold=self.ic_threshold,
            )

        return AlertResult(
            is_alert=False,
            severity=Severity.INFO,
            model_name=model_name,
            check_type="alpha_decay",
            message=(
                f"{model_name}: rolling IC ({window}d) = {latest_ic:.4f} "
                f"(above threshold {self.ic_threshold:.4f})."
            ),
            recommendation="No action required.",
            metric_value=latest_ic,
            threshold=self.ic_threshold,
        )

    # ------------------------------------------------------------------
    # 2. Feature drift
    # ------------------------------------------------------------------

    def check_feature_drift(
        self,
        features_today: "pd.DataFrame | Any",
        features_baseline: "pd.DataFrame | Any",
        model_name: str,
        top_shap_features: list[str] | None = None,
    ) -> AlertResult:
        """Detect feature distribution drift using the KS test.

        Converts Polars DataFrames to pandas automatically when polars is
        available.  Focuses on ``top_shap_features`` when provided; otherwise
        tests all numeric columns.

        Parameters
        ----------
        features_today:
            Today's feature matrix (pandas or polars DataFrame).
        features_baseline:
            Reference/baseline feature matrix (past 60-day average distribution).
        model_name:
            Model identifier (used in the returned AlertResult).
        top_shap_features:
            Optional list of the top-10 SHAP feature column names to focus on.
            When None, all shared numeric columns are tested.

        Returns
        -------
        AlertResult
        """
        from scipy.stats import ks_2samp

        # Normalise to pandas
        today_df = _to_pandas(features_today)
        baseline_df = _to_pandas(features_baseline)

        if today_df.empty or baseline_df.empty:
            return AlertResult(
                is_alert=False,
                severity=Severity.INFO,
                model_name=model_name,
                check_type="feature_drift",
                message="Empty feature DataFrame — drift check skipped.",
                recommendation="Check data pipeline output.",
                metric_value=0.0,
                threshold=self.drift_pvalue_threshold,
            )

        # Select numeric columns present in both DataFrames
        numeric_cols = [
            c for c in today_df.columns
            if c in baseline_df.columns
            and pd.api.types.is_numeric_dtype(today_df[c])
        ]

        if top_shap_features:
            focus_cols = [c for c in top_shap_features if c in numeric_cols]
            if focus_cols:
                numeric_cols = focus_cols

        if not numeric_cols:
            return AlertResult(
                is_alert=False,
                severity=Severity.INFO,
                model_name=model_name,
                check_type="feature_drift",
                message="No shared numeric columns to test.",
                recommendation="Verify feature schema consistency.",
                metric_value=0.0,
                threshold=self.drift_pvalue_threshold,
            )

        drifted: list[str] = []
        for col in numeric_cols:
            t_vals = today_df[col].dropna().values
            b_vals = baseline_df[col].dropna().values
            if len(t_vals) < 5 or len(b_vals) < 5:
                continue
            _, p_value = ks_2samp(t_vals, b_vals)
            if p_value < self.drift_pvalue_threshold:
                drifted.append(col)

        n_tested = len(numeric_cols)
        n_drifted = len(drifted)
        drift_fraction = n_drifted / n_tested if n_tested > 0 else 0.0

        logger.debug(
            f"[{model_name}] Feature drift: {n_drifted}/{n_tested} features drifted "
            f"(threshold {self.drift_feature_fraction:.0%})"
        )

        if drift_fraction > self.drift_feature_fraction:
            severity = (
                Severity.CRITICAL if drift_fraction > self.drift_feature_fraction * 2
                else Severity.WARNING
            )
            drifted_preview = ", ".join(drifted[:5])
            if len(drifted) > 5:
                drifted_preview += f" … (+{len(drifted) - 5} more)"
            return AlertResult(
                is_alert=True,
                severity=severity,
                model_name=model_name,
                check_type="feature_drift",
                message=(
                    f"{model_name}: {n_drifted}/{n_tested} features "
                    f"({drift_fraction:.0%}) show significant distribution drift "
                    f"(KS p < {self.drift_pvalue_threshold}). "
                    f"Drifted: {drifted_preview}."
                ),
                recommendation=(
                    "Inspect the drifted features for data quality issues. "
                    "Consider rebuilding the feature baseline window, "
                    "or retraining the model on the updated distribution."
                ),
                metric_value=drift_fraction,
                threshold=self.drift_feature_fraction,
            )

        return AlertResult(
            is_alert=False,
            severity=Severity.INFO,
            model_name=model_name,
            check_type="feature_drift",
            message=(
                f"{model_name}: {n_drifted}/{n_tested} features drifted "
                f"({drift_fraction:.0%} ≤ threshold {self.drift_feature_fraction:.0%})."
            ),
            recommendation="No action required.",
            metric_value=drift_fraction,
            threshold=self.drift_feature_fraction,
        )

    # ------------------------------------------------------------------
    # 3. SHAP stability
    # ------------------------------------------------------------------

    def check_shap_stability(
        self,
        shap_today: pd.DataFrame,
        shap_baseline: pd.DataFrame,
        model_name: str = "council",
        top_n: int = 10,
    ) -> AlertResult:
        """Detect model behavioural change via SHAP feature importance shift.

        Compares the set of top-``top_n`` features by mean absolute SHAP value
        today vs the 30-day baseline.  If the Jaccard overlap of the two sets
        falls below ``shap_overlap_min``, an alert is raised.

        Parameters
        ----------
        shap_today:
            DataFrame where columns are feature names and rows are observations;
            values are SHAP values for today's predictions.
        shap_baseline:
            Same structure but for the 30-day baseline period.
        model_name:
            Model identifier.
        top_n:
            Number of top features to compare. Default 10.

        Returns
        -------
        AlertResult
        """
        if shap_today.empty or shap_baseline.empty:
            return AlertResult(
                is_alert=False,
                severity=Severity.INFO,
                model_name=model_name,
                check_type="shap_stability",
                message="Empty SHAP DataFrame — stability check skipped.",
                recommendation="Ensure SHAP values are computed and persisted daily.",
                metric_value=1.0,
                threshold=self.shap_overlap_min,
            )

        # Compute mean absolute SHAP importance
        importance_today = shap_today.abs().mean()
        importance_baseline = shap_baseline.abs().mean()

        # Select shared features
        shared = importance_today.index.intersection(importance_baseline.index)
        if len(shared) == 0:
            return AlertResult(
                is_alert=False,
                severity=Severity.INFO,
                model_name=model_name,
                check_type="shap_stability",
                message="No shared features between today and baseline SHAP DataFrames.",
                recommendation="Verify feature schema is consistent across days.",
                metric_value=0.0,
                threshold=self.shap_overlap_min,
            )

        n = min(top_n, len(shared))
        top_today = set(
            importance_today[shared].nlargest(n).index.tolist()
        )
        top_baseline = set(
            importance_baseline[shared].nlargest(n).index.tolist()
        )

        overlap_count = len(top_today & top_baseline)
        union_count = len(top_today | top_baseline)
        jaccard = overlap_count / union_count if union_count > 0 else 1.0

        logger.debug(
            f"[{model_name}] SHAP stability: top-{n} overlap = {jaccard:.2%} "
            f"(min {self.shap_overlap_min:.0%})"
        )

        if jaccard < self.shap_overlap_min:
            new_features = sorted(top_today - top_baseline)
            dropped_features = sorted(top_baseline - top_today)
            return AlertResult(
                is_alert=True,
                severity=Severity.WARNING,
                model_name=model_name,
                check_type="shap_stability",
                message=(
                    f"{model_name}: top-{n} SHAP feature overlap = {jaccard:.0%} "
                    f"(min {self.shap_overlap_min:.0%}). "
                    f"New features: {new_features}. "
                    f"Dropped: {dropped_features}."
                ),
                recommendation=(
                    "Model is relying on a different feature set than the baseline. "
                    "Investigate whether new features are spurious (regime artifact) "
                    "or reflect genuine signal shift. Consider retraining."
                ),
                metric_value=jaccard,
                threshold=self.shap_overlap_min,
            )

        return AlertResult(
            is_alert=False,
            severity=Severity.INFO,
            model_name=model_name,
            check_type="shap_stability",
            message=(
                f"{model_name}: top-{n} SHAP overlap = {jaccard:.0%} "
                f"(≥ threshold {self.shap_overlap_min:.0%})."
            ),
            recommendation="No action required.",
            metric_value=jaccard,
            threshold=self.shap_overlap_min,
        )

    # ------------------------------------------------------------------
    # 4. Regime change
    # ------------------------------------------------------------------

    def check_regime_change(
        self,
        regime_today: str,
        regime_yesterday: str,
        transition_prob: float,
        model_name: str = "hmm",
    ) -> AlertResult:
        """Alert when the HMM regime changes with high confidence.

        An alert fires when *both* conditions hold:
        1. ``regime_today != regime_yesterday``
        2. ``transition_prob >= regime_transition_prob_min``

        Parameters
        ----------
        regime_today:
            Current regime label (e.g. "bull", "bear", "transition").
        regime_yesterday:
            Previous day's regime label.
        transition_prob:
            Posterior probability of the new state emitted by the HMM.
        model_name:
            Model identifier. Default "hmm".

        Returns
        -------
        AlertResult
        """
        regime_changed = regime_today != regime_yesterday
        high_confidence = transition_prob >= self.regime_transition_prob_min

        if regime_changed and high_confidence:
            return AlertResult(
                is_alert=True,
                severity=Severity.WARNING,
                model_name=model_name,
                check_type="regime_change",
                message=(
                    f"Market regime changed: {regime_yesterday} → {regime_today} "
                    f"(transition probability {transition_prob:.2%})."
                ),
                recommendation=(
                    f"Update council weights for the new '{regime_today}' regime. "
                    "Review position sizing and stop-loss levels. "
                    "Consider reducing gross exposure until regime is confirmed."
                ),
                metric_value=transition_prob,
                threshold=self.regime_transition_prob_min,
            )

        if regime_changed and not high_confidence:
            return AlertResult(
                is_alert=False,
                severity=Severity.INFO,
                model_name=model_name,
                check_type="regime_change",
                message=(
                    f"Tentative regime change: {regime_yesterday} → {regime_today} "
                    f"(low confidence: {transition_prob:.2%} < {self.regime_transition_prob_min:.0%})."
                ),
                recommendation="Monitor over the next 2–3 days before adjusting weights.",
                metric_value=transition_prob,
                threshold=self.regime_transition_prob_min,
            )

        # No regime change
        return AlertResult(
            is_alert=False,
            severity=Severity.INFO,
            model_name=model_name,
            check_type="regime_change",
            message=f"Regime stable: {regime_today} (prob {transition_prob:.2%}).",
            recommendation="No action required.",
            metric_value=transition_prob,
            threshold=self.regime_transition_prob_min,
        )

    # ------------------------------------------------------------------
    # 5. Full daily run
    # ------------------------------------------------------------------

    def run_daily_checks(
        self,
        check_date: date,
        *,
        ic_histories: dict[str, pd.Series] | None = None,
        features_today: "pd.DataFrame | None" = None,
        features_baseline: "pd.DataFrame | None" = None,
        shap_today: pd.DataFrame | None = None,
        shap_baseline: pd.DataFrame | None = None,
        regime_today: str | None = None,
        regime_yesterday: str | None = None,
        transition_prob: float | None = None,
        top_shap_features: list[str] | None = None,
        dispatch: bool = True,
    ) -> list[AlertResult]:
        """Run all monitoring checks for a given date.

        Parameters
        ----------
        check_date:
            Date of the monitoring run (used for logging and file naming).
        ic_histories:
            model_name → pd.Series of daily IC values.
            When None, alpha-decay checks are skipped.
        features_today, features_baseline:
            Feature DataFrames for drift detection.
            When either is None, feature-drift check is skipped.
        shap_today, shap_baseline:
            SHAP value DataFrames for stability check.
            When either is None, SHAP-stability check is skipped.
        regime_today, regime_yesterday, transition_prob:
            Regime state inputs.  All three must be provided to run the
            regime-change check; otherwise it is skipped.
        top_shap_features:
            Top-10 SHAP feature names to focus the drift check on.
        dispatch:
            When True (default), alerts are routed through AlertDispatcher.

        Returns
        -------
        list[AlertResult]
            All results (alert and non-alert) from every executed check.
        """
        results: list[AlertResult] = []

        logger.info(f"CouncilMonitor.run_daily_checks — {check_date}")

        # Alpha decay per model
        if ic_histories:
            for model_name, ic_series in ic_histories.items():
                result = self.check_alpha_decay(model_name, ic_series)
                results.append(result)
                _log_check(result, check_date)

        # Feature drift
        if features_today is not None and features_baseline is not None:
            result = self.check_feature_drift(
                features_today,
                features_baseline,
                model_name="council",
                top_shap_features=top_shap_features,
            )
            results.append(result)
            _log_check(result, check_date)

        # SHAP stability
        if shap_today is not None and shap_baseline is not None:
            result = self.check_shap_stability(shap_today, shap_baseline)
            results.append(result)
            _log_check(result, check_date)

        # Regime change
        if (
            regime_today is not None
            and regime_yesterday is not None
            and transition_prob is not None
        ):
            result = self.check_regime_change(
                regime_today, regime_yesterday, transition_prob
            )
            results.append(result)
            _log_check(result, check_date)

        # Severity escalation: if multiple CRITICAL/WARNING alerts fire simultaneously
        results = _escalate_severity(results)

        # Log to MLflow when available
        _log_to_mlflow(results, check_date)

        # Dispatch through AlertDispatcher
        if dispatch:
            dispatcher = AlertDispatcher()
            dispatcher.dispatch(results)

        n_alerts = sum(1 for r in results if r.is_alert)
        logger.info(
            f"run_daily_checks complete: {len(results)} checks run, "
            f"{n_alerts} alert(s) triggered."
        )
        return results


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _count_trailing_true(series: pd.Series) -> int:
    """Count how many trailing True values a boolean Series ends with."""
    values = series.values[::-1]
    count = 0
    for v in values:
        if v:
            count += 1
        else:
            break
    return count


def _to_pandas(df: Any) -> pd.DataFrame:
    """Convert polars DataFrame to pandas; return as-is if already pandas."""
    if isinstance(df, pd.DataFrame):
        return df
    if _POLARS_AVAILABLE:
        import polars as pl  # noqa: PLC0415
        if isinstance(df, pl.DataFrame):
            return df.to_pandas()
    return pd.DataFrame()


def _log_check(result: AlertResult, check_date: date) -> None:
    """Emit a structured log line for the check result."""
    level = "WARNING" if result.is_alert else "DEBUG"
    getattr(logger, level.lower())(
        f"[{check_date}] {result.check_type} / {result.model_name}: "
        f"alert={result.is_alert} severity={result.severity.value} "
        f"metric={result.metric_value:.4f} threshold={result.threshold:.4f}"
    )


def _escalate_severity(results: list[AlertResult]) -> list[AlertResult]:
    """Escalate to CRITICAL when 3+ WARNING/CRITICAL alerts fire simultaneously."""
    active_alerts = [r for r in results if r.is_alert]
    if len(active_alerts) >= 3:
        for r in active_alerts:
            if r.severity in (Severity.WARNING, Severity.CRITICAL):
                r.severity = Severity.CRITICAL
    return results


def _log_to_mlflow(results: list[AlertResult], check_date: date) -> None:
    """Log alert metrics to MLflow (no-op when mlflow is not installed)."""
    if not _MLFLOW_AVAILABLE:
        return
    try:
        with mlflow.start_run(run_name=f"monitor_{check_date.isoformat()}"):
            for r in results:
                prefix = f"monitor/{r.check_type}/{r.model_name}"
                mlflow.log_metric(f"{prefix}/metric_value", r.metric_value)
                mlflow.log_metric(f"{prefix}/is_alert", int(r.is_alert))
            n_alerts = sum(1 for r in results if r.is_alert)
            mlflow.log_metric("monitor/total_alerts", n_alerts)
    except Exception as exc:  # noqa: BLE001
        logger.debug(f"MLflow logging skipped: {exc}")
