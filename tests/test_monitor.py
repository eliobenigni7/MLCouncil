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
from unittest.mock import MagicMock

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

class TestICMonitoring:
    def test_rolling_ic_computed(self, monitor: CouncilMonitor) -> None:
        ic = _make_ic_series([0.02] * 80)
        roll = monitor.compute_rolling_ic(ic, window=60)
        assert not roll.dropna().empty
        assert float(roll.dropna().iloc[-1]) == pytest.approx(0.02, abs=1e-6)

    def test_model_correlation_matrix(self, monitor: CouncilMonitor) -> None:
        ic_a = _make_ic_series(np.linspace(0.01, 0.05, 70).tolist())
        ic_b = _make_ic_series((np.linspace(0.01, 0.05, 70) * 0.9).tolist())
        corr = monitor.compute_model_correlation({"lgbm": ic_a, "sentiment": ic_b}, window=60)
        assert not corr.empty
        assert corr.iloc[0]["rho"] > 0.5

    def test_ic_sustained_decay_alert(self, monitor: CouncilMonitor) -> None:
        ic = _make_ic_series([-0.01] * 70)
        result = monitor.check_ic_sustained_decay("lgbm", ic, window=60)
        assert result.is_alert
        assert result.check_type == "ic_monitoring"
        assert result.severity == Severity.WARNING

    def test_ic_monitoring_healthy_ic_no_alert(self, monitor: CouncilMonitor) -> None:
        ic = _make_ic_series([0.02] * 50)
        result = monitor.check_ic_sustained_decay("lgbm", ic)
        assert not result.is_alert
        assert result.check_type == "ic_monitoring"


class TestCostCalibrationDivergence:
    def test_no_artifact_is_info(self, monitor: CouncilMonitor, tmp_path, monkeypatch):
        monkeypatch.setenv("MLCOUNCIL_COST_CALIBRATION_PATH", str(tmp_path / "missing.json"))
        result = monitor.check_cost_calibration_divergence()
        assert not result.is_alert
        assert result.check_type == "cost_calibration_divergence"

    def test_warning_on_sustained_divergence(self, monitor: CouncilMonitor, tmp_path, monkeypatch):
        from council.cost_calibration import write_calibration

        art = _artifact_for_monitor(kappa_by_tier={"mega": 10.0})
        calib = tmp_path / "cost_calibration.json"
        write_calibration(art, path=calib)
        monkeypatch.setenv("MLCOUNCIL_COST_CALIBRATION_PATH", str(calib))

        result = monitor.check_cost_calibration_divergence(
            streak_by_tier={"mega": 5},
        )
        assert result.is_alert
        assert result.severity in {Severity.WARNING, Severity.CRITICAL}


def _artifact_for_monitor(**kwargs):
    from datetime import datetime, timezone

    from council.cost_calibration import CalibrationArtifact

    defaults = dict(
        generated_at=datetime(2026, 5, 21, tzinfo=timezone.utc),
        calibration_window_end=datetime(2026, 5, 21, tzinfo=timezone.utc),
        fill_sample_count=60,
        min_fills=30,
        kappa_by_ticker={},
        fill_count_by_ticker={},
        kappa_by_tier={"mega": 20.0},
        fill_count_by_tier={"mega": 60},
    )
    defaults.update(kwargs)
    return CalibrationArtifact(**defaults)


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


# ---------------------------------------------------------------------------
# 6. Causal graph drift (T4.4)
# ---------------------------------------------------------------------------

class TestCausalGraphDrift:
    """check_causal_graph_drift: flag, baseline init e alert su cambio struttura."""

    def test_causal_drift_disabled_is_info(self, monitor: CouncilMonitor, monkeypatch) -> None:
        """Senza MLCOUNCIL_CAUSAL_DRIFT_ENABLED il check è INFO, mai alert."""
        monkeypatch.delenv("MLCOUNCIL_CAUSAL_DRIFT_ENABLED", raising=False)
        features = _make_feature_df(n_rows=80, n_cols=4, seed=3)
        returns = pd.Series(
            np.random.default_rng(4).standard_normal(80), index=features.index
        )

        result = monitor.check_causal_graph_drift(features, returns)

        assert not result.is_alert
        assert result.check_type == "causal_drift"
        assert result.severity == Severity.INFO

    def test_causal_drift_alert_on_structure_change(
        self, monitor: CouncilMonitor, monkeypatch
    ) -> None:
        """Con baseline condivisa, un cambio di struttura feature→return → WARNING."""
        from council.causal_drift import PCMCIDriftDetector

        monkeypatch.setenv("MLCOUNCIL_CAUSAL_DRIFT_ENABLED", "true")
        rng = np.random.default_rng(1)
        n = 80
        base_f = pd.DataFrame(
            {"f1": rng.standard_normal(n), "f2": rng.standard_normal(n)}
        )
        base_r = pd.Series(base_f["f1"] * 0.2 + rng.standard_normal(n) * 0.01)

        detector = PCMCIDriftDetector(corr_threshold=0.1, link_change_fraction=0.25)
        first = monitor.check_causal_graph_drift(base_f, base_r, detector=detector)
        assert not first.is_alert, "La prima chiamata inizializza solo la baseline"
        assert first.metric_value == 0.0

        shifted_f = pd.DataFrame(
            {"f2": rng.standard_normal(n), "f3": rng.standard_normal(n)}
        )
        shifted_r = pd.Series(shifted_f["f2"] * 0.25)
        second = monitor.check_causal_graph_drift(shifted_f, shifted_r, detector=detector)

        assert second.is_alert
        assert second.check_type == "causal_drift"
        assert second.severity == Severity.WARNING
        assert second.metric_value >= detector.link_change_fraction

    def test_causal_drift_fresh_detector_never_alerts_on_first_call(
        self, monitor: CouncilMonitor, monkeypatch
    ) -> None:
        """Senza detector condiviso, la prima chiamata crea la baseline (ok)."""
        monkeypatch.setenv("MLCOUNCIL_CAUSAL_DRIFT_ENABLED", "true")
        features = _make_feature_df(n_rows=80, n_cols=3, seed=5)
        returns = pd.Series(
            np.random.default_rng(6).standard_normal(80), index=features.index
        )

        result = monitor.check_causal_graph_drift(features, returns)

        assert not result.is_alert
        assert result.check_type == "causal_drift"


# ---------------------------------------------------------------------------
# 7. Unified health signals (F-0.2)
# ---------------------------------------------------------------------------

class TestHealthSignals:
    """collect_health_signals: aggregazione dei quattro famiglie di drift."""

    def test_all_ok_with_healthy_inputs(self) -> None:
        from council.alerting import collect_health_signals

        signals = collect_health_signals(
            tda_alert={"is_alert": False, "beta1_proxy": 0.20, "threshold": 0.35},
            causal_drift={"change_fraction": 0.10},
            adwin_drift={"drift_detected": False},
            ddm_drift={"drift_detected": False},
            evidently_drift={"drift_fraction": 0.30},
        )
        assert set(signals) == {
            "tda_warning",
            "causal_drift",
            "adwin_drift",
            "ddm_drift",
            "evidently_drift",
        }
        assert all(s["level"] == "ok" for s in signals.values())
        assert signals["causal_drift"]["value"] == pytest.approx(0.10)
        assert signals["evidently_drift"]["threshold"] == pytest.approx(0.5)

    def test_causal_drift_breach_alerts(self) -> None:
        from council.alerting import collect_health_signals

        signals = collect_health_signals(causal_drift={"change_fraction": 0.33})

        assert signals["causal_drift"]["level"] == "alert"
        assert signals["causal_drift"]["value"] == pytest.approx(0.33)
        assert signals["causal_drift"]["threshold"] == pytest.approx(0.25)

    def test_causal_drift_exactly_at_threshold_alerts(self) -> None:
        from council.alerting import collect_health_signals

        signals = collect_health_signals(causal_drift={"change_fraction": 0.25})
        assert signals["causal_drift"]["level"] == "alert"

    def test_evidently_drift_breach_alerts(self) -> None:
        from council.alerting import collect_health_signals

        signals = collect_health_signals(evidently_drift={"drift_fraction": 0.6})

        assert signals["evidently_drift"]["level"] == "alert"
        assert signals["evidently_drift"]["threshold"] == pytest.approx(0.5)

    def test_tda_alert_flag(self) -> None:
        from council.alerting import collect_health_signals

        signals = collect_health_signals(
            tda_alert={"is_alert": True, "beta1_proxy": 0.42}
        )
        assert signals["tda_warning"]["level"] == "alert"
        assert signals["tda_warning"]["value"] is True

    def test_adwin_ddm_flags(self) -> None:
        from council.alerting import collect_health_signals

        signals = collect_health_signals(
            adwin_drift={"drift_detected": True},
            ddm_drift={"drift_detected": True},
        )
        assert signals["adwin_drift"]["level"] == "alert"
        assert signals["ddm_drift"]["level"] == "alert"

    def test_bool_flags_accepted(self) -> None:
        """Input boolean nudi per ADWIN/DDM devono funzionare come i payload dict."""
        from council.alerting import collect_health_signals

        signals = collect_health_signals(adwin_drift=True, ddm_drift=False)
        assert signals["adwin_drift"]["level"] == "alert"
        assert signals["ddm_drift"]["level"] == "ok"

    def test_missing_inputs_ok_with_note(self) -> None:
        from council.alerting import collect_health_signals

        signals = collect_health_signals()

        for name in ("tda_warning", "causal_drift", "adwin_drift", "ddm_drift", "evidently_drift"):
            assert signals[name]["level"] == "ok", name
            assert signals[name]["value"] is None
            assert signals[name]["note"], f"{name} deve avere una nota"

    def test_empty_dict_inputs_ok_with_note(self) -> None:
        from council.alerting import collect_health_signals

        signals = collect_health_signals(
            tda_alert={}, causal_drift={}, evidently_drift={}
        )
        assert signals["tda_warning"]["level"] == "ok"
        assert signals["causal_drift"]["level"] == "ok"
        assert signals["evidently_drift"]["level"] == "ok"

    def test_malformed_inputs_no_exception(self) -> None:
        from council.alerting import collect_health_signals

        signals = collect_health_signals(
            tda_alert="garbage",
            causal_drift=[1, 2, 3],
            evidently_drift={"drift_fraction": "x"},
            adwin_drift={"drift_detected": "yes"},
        )

        assert signals["tda_warning"]["level"] == "ok"
        assert signals["causal_drift"]["level"] == "ok"
        assert signals["evidently_drift"]["level"] == "ok"
        assert signals["adwin_drift"]["level"] == "alert"  # "yes" è truthy → flag attivo

    def test_custom_thresholds(self) -> None:
        from council.alerting import collect_health_signals

        signals = collect_health_signals(
            causal_drift={"change_fraction": 0.20},
            evidently_drift={"drift_fraction": 0.45},
            causal_threshold=0.30,
            evidently_threshold=0.6,
        )
        assert signals["causal_drift"]["level"] == "ok"
        assert signals["evidently_drift"]["level"] == "ok"


# ---------------------------------------------------------------------------
# 8. Health dispatch (F-0.2) — ponte verso AlertDispatcher
# ---------------------------------------------------------------------------

class TestHealthDispatch:
    """dispatch_health_alerts: health dict → AlertResult → AlertDispatcher."""

    def test_alert_level_dispatches_critical(self) -> None:
        from council.alerting import dispatch_health_alerts

        health = {
            "causal_drift": {"level": "alert", "value": 0.4, "threshold": 0.25, "note": None},
            "tda_warning": {
                "level": "alert",
                "value": True,
                "threshold": 0.35,
                "note": "beta1_proxy=0.4100",
            },
            "evidently_drift": {"level": "ok", "value": 0.3, "threshold": 0.5, "note": None},
        }
        dispatcher = MagicMock()

        results = dispatch_health_alerts(
            health, dispatcher=dispatcher, check_date="2026-08-17"
        )

        assert len(results) == 2
        assert all(r.is_alert for r in results)
        assert all(r.model_name == "council" for r in results)
        assert all(r.severity == Severity.CRITICAL for r in results)
        assert {r.check_type for r in results} == {"causal_drift", "tda_warning"}

        causal = next(r for r in results if r.check_type == "causal_drift")
        assert causal.metric_value == pytest.approx(0.4)
        assert causal.threshold == pytest.approx(0.25)
        assert "health level=alert" in causal.message
        assert causal.timestamp.startswith("2026-08-17")

        flag = next(r for r in results if r.check_type == "tda_warning")
        assert flag.metric_value == pytest.approx(1.0)  # bool True → 1.0
        assert "beta1_proxy=0.4100" in flag.message

        dispatcher.dispatch.assert_called_once_with(results)

    def test_warn_level_dispatches_warning(self) -> None:
        from council.alerting import dispatch_health_alerts

        health = {
            "evidently_drift": {"level": "warn", "value": 0.6, "threshold": 0.5, "note": None}
        }
        dispatcher = MagicMock()

        results = dispatch_health_alerts(health, dispatcher=dispatcher)

        assert len(results) == 1
        assert results[0].severity == Severity.WARNING
        assert results[0].check_type == "evidently_drift"
        dispatcher.dispatch.assert_called_once()

    def test_ok_only_no_dispatch(self) -> None:
        from council.alerting import dispatch_health_alerts

        health = {
            "causal_drift": {"level": "ok", "value": 0.1, "threshold": 0.25, "note": None},
            "adwin_drift": {"level": "ok", "value": False, "threshold": 1.0, "note": None},
        }
        dispatcher = MagicMock()

        results = dispatch_health_alerts(health, dispatcher=dispatcher)

        assert results == []
        dispatcher.dispatch.assert_not_called()

    def test_none_or_malformed_health_graceful(self) -> None:
        from council.alerting import dispatch_health_alerts

        dispatcher = MagicMock()

        assert dispatch_health_alerts(None, dispatcher=dispatcher) == []
        assert dispatch_health_alerts("garbage", dispatcher=dispatcher) == []
        assert dispatch_health_alerts({"causal_drift": "garbage"}, dispatcher=dispatcher) == []
        dispatcher.dispatch.assert_not_called()

    def test_dispatch_from_disk_roundtrip(self, tmp_path, monkeypatch) -> None:
        """collect_health_signals_from_disk → dispatch: alert dal file JSON."""
        import json as _json

        from council.alerting import collect_health_signals_from_disk, dispatch_health_alerts

        results = tmp_path / "results"
        results.mkdir()
        (results / "causal_drift_latest.json").write_text(
            _json.dumps({"change_fraction": 0.33, "status": "alert", "is_alert": True})
        )
        dispatcher = MagicMock()

        health = collect_health_signals_from_disk(results)
        assert health["causal_drift"]["level"] == "alert"
        results_out = dispatch_health_alerts(health, dispatcher=dispatcher)
        assert len(results_out) == 1
        assert results_out[0].check_type == "causal_drift"
        assert results_out[0].severity == Severity.CRITICAL

    def test_real_dispatcher_writes_log_and_dashboard(self, monkeypatch, tmp_path) -> None:
        """Integrazione con AlertDispatcher reale: log + dashboard state, niente email."""
        import json as _json

        from council import alerts as alerts_mod
        from council.alerting import dispatch_health_alerts

        monkeypatch.setattr(alerts_mod, "_ALERTS_DIR", tmp_path / "alerts")
        monkeypatch.setattr(alerts_mod, "_MONITORING_DIR", tmp_path / "monitoring")
        monkeypatch.setattr(alerts_mod, "_DEADLETTER_DIR", tmp_path / "deadletter")
        monkeypatch.delenv("ALERT_EMAIL", raising=False)

        health = {
            "causal_drift": {"level": "alert", "value": 0.4, "threshold": 0.25, "note": None}
        }
        results = dispatch_health_alerts(health, check_date="2026-08-17")

        assert len(results) == 1
        log_path = tmp_path / "alerts" / f"{date.today().isoformat()}.json"
        assert log_path.exists()
        entries = _json.loads(log_path.read_text(encoding="utf-8"))
        assert entries[-1]["check_type"] == "causal_drift"
        assert entries[-1]["severity"] == "critical"
        assert entries[-1]["model_name"] == "council"

        dashboard = tmp_path / "monitoring" / "current_alerts.json"
        assert dashboard.exists()
        state = _json.loads(dashboard.read_text(encoding="utf-8"))
        assert state[-1]["check_type"] == "causal_drift"
