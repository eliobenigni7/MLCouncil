"""Tests for council.risk.tda_warning (T4.5)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from council.risk.tda_warning import PersistentHomologyAnalyser


def test_tda_analyser_returns_result():
    rng = np.random.default_rng(42)
    dates = pd.bdate_range("2024-01-01", periods=40)
    wide = pd.DataFrame(
        {f"T{i}": rng.standard_normal(40) * 0.01 for i in range(4)},
        index=dates,
    )
    analyser = PersistentHomologyAnalyser(window_days=30, beta1_threshold=0.99)
    result = analyser.analyse(wide)
    assert 0.0 <= result.beta1_proxy <= 1.0
    assert result.window_days == 30
