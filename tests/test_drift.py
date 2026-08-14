"""Tests for council/risk/drift.py River ADWIN and DDM detectors."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

river = pytest.importorskip("river")


class TestADWINDetector:
    def test_no_drift_on_stable_series(self):
        from council.risk.drift import ADWINDetector

        detector = ADWINDetector(window_days=60)
        stable = pd.Series(np.full(60, 0.0005))
        assert detector.update_series(stable) is False

    def test_update_reflects_detector_drift_flag(self):
        from unittest.mock import MagicMock

        from council.risk.drift import ADWINDetector

        inner = MagicMock()
        inner.drift_detected = True
        detector = ADWINDetector()
        detector._detector = inner
        assert detector.update(0.01) is True
        assert detector.drift_detected is True
        inner.update.assert_called_once()

    def test_window_bounded(self):
        from council.risk.drift import ADWINDetector

        detector = ADWINDetector(window_days=10)
        detector.update_series(pd.Series(np.random.default_rng(0).normal(0, 0.01, 100)))
        assert detector.window_size == 10


class TestDDMDetector:
    def test_ddm_reads_binary_drift_flag(self):
        from unittest.mock import MagicMock

        from council.risk.drift import DDMDetector

        inner = MagicMock()
        inner.drift_detected = True
        inner.in_warning_zone = False
        detector = DDMDetector(warm_start=5)
        detector._detector = inner
        assert detector.update(1) is True

    def test_returns_to_error_indicators(self):
        from council.risk.drift import DDMDetector

        errs = DDMDetector.returns_to_error_indicators(
            pd.Series([0.01, -0.02, 0.0, -0.001])
        )
        assert list(errs) == [0, 1, 0, 1]

    def test_empty_series_is_noop(self):
        from council.risk.drift import ADWINDetector, DDMDetector

        assert ADWINDetector().update_series(pd.Series(dtype=float)) is False
        assert DDMDetector().update_series(pd.Series(dtype=float)) is False
