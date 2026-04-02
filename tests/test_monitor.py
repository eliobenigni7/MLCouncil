"""Tests for council/monitor.py and council/alerts.py (Agent 09).

Coverage
--------
1. test_alpha_decay_detected          IC < 0.01 for 5+ days → WARNING alert
2. test_no_false_alert_on_good_ic     IC > 0.03 consistently → no alert
3. test_drift_detected_on_shifted_distribution  features shifted 2σ → drift alert
4. test_shap_instability_detected     completely different top-10 → SHAP alert
5. test_severity_escalation           3 simultaneous alerts → all CRITICAL
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from council.alerts import AlertResult, Severity
from council.monitor import CouncilMonitor, _count_trailing_true, _escalate_severity


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_ic_series(values: list[float], start: date | None = None) -> pd.Series:
    """Build a DatetimeIndex IC Series from a list of daily values."""
    if start is None:
        start = date(2024, 1, 2)
    idx = [start + timedelta(days=i) for i in range(len(values))]
    return pd.Series(values, index=pd.DatetimeIndex(idx), name="ic")


def _make_feature_df(
    n_rows: int = 200,
    n_cols: int = 20,
    mean_shift: float = 0.0,
    seed: int = 0,
) -> pd.DataFrame:
    """Synthetic feature DataFrame with optional mean shift."""
    rng = np.random.default_rng(seed)
    data = rng.standard_normal((n_rows, n_cols)) + mean_shift
    cols = [f"feat_{i:03d}" for i in range(n_cols)]
    return pd.DataFrame(data, columns=cols)


def _make_shap_df(
    features: list[str],
    n_rows: int = 100,
    seed: int = 0,
) -> pd.DataFrame:
    """Synthetic SHAP value DataFrame with given feature columns."""
    rng = np.random.default_rng(seed)
    data = np.abs(rng.standard_normal((n_rows, len(features))))
    return pd.DataFrame(data, columns=features)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def monitor() -> CouncilMonitor:
    return CouncilMonitor()


# ---------------------------------------------------------------------------
# 1. Alpha decay detected
# ---------------------------------------------------------------------------

class TestAlphaDecay:
    def test_alpha_decay_detected(self, monitor: CouncilMonitor) -> None:
        """IC < 0.01 for ≥ 5 days at the end of the window → WARNING alert."""
        # 30 good days followed by 6 bad days
        good = [0.04] * 30
        bad = [-0.005] * 6  # all below threshold 0.01
        ic_series = _make_ic_series(good + bad)

        result = monitor.check_alpha_decay("lgbm", ic_series, window=30)

        assert result.is_alert, "Expected alert when IC decays for 6 consecutive days"
        assert result.check_type == "alpha_decay"
        assert result.model_name == "lgbm"
        assert result.severity in (Severity.WARNING, Severity.CRITICAL)
        assert result.metric_value < monitor.ic_threshold

    def test_alpha_decay_exactly_at_threshold(self, monitor: CouncilMonitor) -> None:
        """Exactly 5 days below threshold (boundary condition) → alert fires."""
        good = [0.05] * 30
        bad = [0.005] * 5  # exactly ic_alert_consecutive_days
        ic_series = _make_ic_series(good + bad)

        result = monitor.check_alpha_decay("sentiment", ic_series, window=30)
        assert result.is_alert

    def test_no_false_alert_on_good_ic(self, monitor: CouncilMonitor) -> None:
        """IC consistently > 0.03 → no alert."""
        values = [0.035 + 0.001 * (i % 5) for i in range(60)]
        ic_series = _make_ic_series(values)

        result = monitor.check_alpha_decay("lgbm", ic_series, window=30)

        assert not result.is_alert, (
            f"False alert: IC={result.metric_value:.4f} is above threshold "
            f"{monitor.ic_threshold:.4f}"
        )
        assert result.severity == Severity.INFO

    def test_insufficient_history_no_alert(self, monitor: CouncilMonitor) -> None:
        """Fewer days than window length → no alert (not enough data)."""
        ic_series = _make_ic_series([0.001] * 10)  # only 10 days, window=30
        result = monitor.check_alpha_decay("hmm", ic_series, window=30)
        assert not result.is_alert

    def test_decay_fewer_than_threshold_days_no_alert(self, monitor: CouncilMonitor) -> None:
        """IC below threshold for only 3 days (< 5 required) → no alert."""
        good = [0.04] * 35
        bad = [0.005] * 3  # only 3 consecutive bad days
        ic_series = _make_ic_series(good + bad)

        result = monitor.check_alpha_decay("lgbm", ic_series, window=30)
        assert not result.is_alert


# ---------------------------------------------------------------------------
# 2. Feature drift
# ---------------------------------------------------------------------------

class TestFeatureDrift:
    def test_drift_detected_on_shifted_distribution(
        self, monitor: CouncilMonitor
    ) -> None:
        """Features shifted by 2σ → >20 % of features flagged → DRIFT alert."""
        baseline = _make_feature_df(n_rows=300, n_cols=20, mean_shift=0.0, seed=1)
        today = _make_feature_df(n_rows=300, n_cols=20, mean_shift=2.0, seed=2)

        result = monitor.check_feature_drift(today, baseline, model_name="lgbm")

        assert result.is_alert, (
            f"Expected drift alert but got no alert. "
            f"drift_fraction={result.metric_value:.2%}"
        )
        assert result.check_type == "feature_drift"
        assert result.metric_value > monitor.drift_feature_fraction

    def test_no_drift_on_same_distribution(self, monitor: CouncilMonitor) -> None:
        """Same distribution (only seed differs) → no drift alert."""
        baseline = _make_feature_df(n_rows=500, n_cols=10, mean_shift=0.0, seed=10)
        today = _make_feature_df(n_rows=500, n_cols=10, mean_shift=0.0, seed=11)

        result = monitor.check_feature_drift(today, baseline, model_name="lgbm")

        assert not result.is_alert, (
            f"False drift alert: drift_fraction={result.metric_value:.2%}"
        )

    def test_drift_focuses_on_top_shap_features(self, monitor: CouncilMonitor) -> None:
        """When top_shap_features is given, only those columns are tested."""
        # Shift features 0-4, keep features 5-19 unchanged
        rng = np.random.default_rng(42)
        baseline = pd.DataFrame(
            rng.standard_normal((200, 20)),
            columns=[f"feat_{i:03d}" for i in range(20)],
        )
        current = baseline.copy()
        for i in range(5):
            current[f"feat_{i:03d}"] = rng.standard_normal(200) + 3.0  # shifted

        # Focus on only the stable features (5-19) — should NOT alert
        top_shap = [f"feat_{i:03d}" for i in range(5, 15)]
        result = monitor.check_feature_drift(
            current, baseline, model_name="lgbm", top_shap_features=top_shap
        )
        assert not result.is_alert, "Should not alert when only stable features are in top-SHAP"

    def test_empty_dataframe_skips_gracefully(self, monitor: CouncilMonitor) -> None:
        result = monitor.check_feature_drift(
            pd.DataFrame(), pd.DataFrame(), model_name="lgbm"
        )
        assert not result.is_alert


# ---------------------------------------------------------------------------
# 3. SHAP stability
# ---------------------------------------------------------------------------

class TestShapStability:
    def test_shap_instability_detected(self, monitor: CouncilMonitor) -> None:
        """Top-10 features today completely different from baseline → alert."""
        # 20 features — today uses features 0-9, baseline uses features 10-19
        features_today = [f"feat_{i:03d}" for i in range(10)]
        features_baseline = [f"feat_{i:03d}" for i in range(10, 20)]

        # Create SHAP DataFrames with shared column set but very different importances
        # today: features 0-9 have high values, features 10-19 have near-zero values
        rng = np.random.default_rng(0)
        all_features = features_today + features_baseline

        shap_today_data = np.zeros((50, 20))
        shap_today_data[:, :10] = np.abs(rng.standard_normal((50, 10))) + 2.0
        shap_today_data[:, 10:] = np.abs(rng.standard_normal((50, 10))) * 0.01
        shap_today = pd.DataFrame(shap_today_data, columns=all_features)

        shap_baseline_data = np.zeros((50, 20))
        shap_baseline_data[:, :10] = np.abs(rng.standard_normal((50, 10))) * 0.01
        shap_baseline_data[:, 10:] = np.abs(rng.standard_normal((50, 10))) + 2.0
        shap_baseline = pd.DataFrame(shap_baseline_data, columns=all_features)

        result = monitor.check_shap_stability(shap_today, shap_baseline, "lgbm")

        assert result.is_alert, (
            f"Expected SHAP instability alert. Overlap={result.metric_value:.2%}"
        )
        assert result.check_type == "shap_stability"
        assert result.metric_value < monitor.shap_overlap_min

    def test_shap_stability_no_alert_when_identical(
        self, monitor: CouncilMonitor
    ) -> None:
        """Same SHAP importances → full overlap → no alert."""
        features = [f"feat_{i:03d}" for i in range(20)]
        rng = np.random.default_rng(7)
        shap_df = _make_shap_df(features, n_rows=100, seed=7)

        result = monitor.check_shap_stability(shap_df, shap_df, "lgbm")
        assert not result.is_alert
        assert result.metric_value == pytest.approx(1.0)

    def test_shap_stability_partial_overlap(
        self, monitor: CouncilMonitor
    ) -> None:
        """80 % overlap (above 70 % threshold) → no alert."""
        all_features = [f"feat_{i:03d}" for i in range(20)]
        rng = np.random.default_rng(3)

        # Today's importance: features 0-9 rank highest
        today_data = np.zeros((100, 20))
        today_data[:, :10] = np.abs(rng.standard_normal((100, 10))) + 3.0
        today_data[:, 10:] = np.abs(rng.standard_normal((100, 10))) * 0.1
        shap_today = pd.DataFrame(today_data, columns=all_features)

        # Baseline: features 0-7 + 10,11 rank highest (8/10 overlap = 80 %)
        base_data = np.zeros((100, 20))
        base_data[:, :8] = np.abs(rng.standard_normal((100, 8))) + 3.0
        base_data[:, 8:10] = np.abs(rng.standard_normal((100, 2))) * 0.1
        base_data[:, 10:12] = np.abs(rng.standard_normal((100, 2))) + 3.0
        base_data[:, 12:] = np.abs(rng.standard_normal((100, 8))) * 0.1
        shap_baseline = pd.DataFrame(base_data, columns=all_features)

        result = monitor.check_shap_stability(shap_today, shap_baseline, "lgbm")
        assert not result.is_alert


# ---------------------------------------------------------------------------
# 4. Regime change
# ---------------------------------------------------------------------------

class TestRegimeChange:
    def test_regime_change_detected_high_confidence(
        self, monitor: CouncilMonitor
    ) -> None:
        """Regime changes with prob > 0.7 → WARNING alert."""
        result = monitor.check_regime_change(
            regime_today="bear",
            regime_yesterday="bull",
            transition_prob=0.85,
        )
        assert result.is_alert
        assert result.check_type == "regime_change"
        assert result.severity == Severity.WARNING

    def test_regime_change_low_confidence_no_alert(
        self, monitor: CouncilMonitor
    ) -> None:
        """Regime changes but prob < 0.7 → tentative INFO, no alert."""
        result = monitor.check_regime_change(
            regime_today="bear",
            regime_yesterday="bull",
            transition_prob=0.55,
        )
        assert not result.is_alert
        assert result.severity == Severity.INFO

    def test_stable_regime_no_alert(self, monitor: CouncilMonitor) -> None:
        """Same regime → no alert regardless of transition prob."""
        result = monitor.check_regime_change(
            regime_today="bull",
            regime_yesterday="bull",
            transition_prob=0.95,
        )
        assert not result.is_alert


# ---------------------------------------------------------------------------
# 5. Severity escalation
# ---------------------------------------------------------------------------

class TestSeverityEscalation:
    def test_severity_escalation_three_concurrent_alerts(
        self, monitor: CouncilMonitor
    ) -> None:
        """Three or more simultaneous alerts → all escalated to CRITICAL."""
        # Build 3 synthetic WARNING alerts
        alerts = [
            AlertResult(
                is_alert=True,
                severity=Severity.WARNING,
                model_name=f"model_{i}",
                check_type="alpha_decay",
                message=f"Alert {i}",
                recommendation="Investigate",
                metric_value=0.005,
                threshold=0.01,
            )
            for i in range(3)
        ]
        escalated = _escalate_severity(alerts)

        for r in escalated:
            assert r.severity == Severity.CRITICAL, (
                f"Expected CRITICAL after escalation, got {r.severity}"
            )

    def test_no_escalation_below_three(self) -> None:
        """Fewer than 3 alerts → severity unchanged."""
        alerts = [
            AlertResult(
                is_alert=True,
                severity=Severity.WARNING,
                model_name="lgbm",
                check_type="alpha_decay",
                message="Single alert",
                recommendation="Check model",
                metric_value=0.005,
                threshold=0.01,
            )
        ]
        result = _escalate_severity(alerts)
        assert result[0].severity == Severity.WARNING

    def test_non_alerts_not_escalated(self) -> None:
        """Non-alert results must not be escalated even when 3+ alerts exist."""
        # 3 alerts + 2 non-alerts
        active = [
            AlertResult(
                is_alert=True,
                severity=Severity.WARNING,
                model_name=f"m{i}",
                check_type="alpha_decay",
                message="alert",
                recommendation="fix",
                metric_value=0.0,
                threshold=0.01,
            )
            for i in range(3)
        ]
        inactive = [
            AlertResult(
                is_alert=False,
                severity=Severity.INFO,
                model_name=f"ok{i}",
                check_type="feature_drift",
                message="ok",
                recommendation="nothing",
                metric_value=0.5,
                threshold=1.0,
            )
            for i in range(2)
        ]
        escalated = _escalate_severity(active + inactive)
        for r in escalated:
            if not r.is_alert:
                assert r.severity == Severity.INFO, "Non-alert severity must not change"


# ---------------------------------------------------------------------------
# AlertResult dataclass tests
# ---------------------------------------------------------------------------

class TestAlertResult:
    def test_to_dict_serializes_severity_as_string(self) -> None:
        ar = AlertResult(
            is_alert=True,
            severity=Severity.CRITICAL,
            model_name="lgbm",
            check_type="alpha_decay",
            message="test",
            recommendation="retrain",
            metric_value=0.005,
            threshold=0.01,
        )
        d = ar.to_dict()
        assert d["severity"] == "critical"
        assert isinstance(d["metric_value"], float)

    def test_timestamp_auto_set(self) -> None:
        ar = AlertResult(
            is_alert=False,
            severity=Severity.INFO,
            model_name="hmm",
            check_type="regime_change",
            message="stable",
            recommendation="nothing",
            metric_value=0.9,
            threshold=0.7,
        )
        assert ar.timestamp.endswith("Z"), f"Unexpected timestamp: {ar.timestamp}"


# ---------------------------------------------------------------------------
# Utility helper tests
# ---------------------------------------------------------------------------

class TestInternalHelpers:
    def test_count_trailing_true_all_true(self) -> None:
        s = pd.Series([True, True, True, True])
        assert _count_trailing_true(s) == 4

    def test_count_trailing_true_mixed(self) -> None:
        s = pd.Series([True, False, True, True, True])
        assert _count_trailing_true(s) == 3

    def test_count_trailing_true_none_true(self) -> None:
        s = pd.Series([False, False, False])
        assert _count_trailing_true(s) == 0

    def test_count_trailing_true_single_false(self) -> None:
        s = pd.Series([True, True, False])
        assert _count_trailing_true(s) == 0
