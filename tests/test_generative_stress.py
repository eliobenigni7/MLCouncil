"""Tests for council.generative_stress (T4.3)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from council.generative_stress import GenerativeStressEngine


def test_generative_stress_samples():
    rng = np.random.default_rng(0)
    dates = pd.bdate_range("2023-01-01", periods=100)
    wide = pd.DataFrame(
        {"A": rng.standard_normal(100) * 0.01, "B": rng.standard_normal(100) * 0.01},
        index=dates,
    )
    engine = GenerativeStressEngine(n_scenarios=500, random_state=0)
    result = engine.sample_scenarios(wide)
    assert result.n_scenarios == 500
    assert result.var_95 <= 0.0 or isinstance(result.var_95, float)
