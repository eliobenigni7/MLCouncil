"""Tests for council.risk.causal_drift (T4.4)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from council.risk.causal_drift import PCMCIDriftDetector


def test_causal_drift_baseline_then_stable():
    rng = np.random.default_rng(0)
    n = 80
    features = pd.DataFrame({"f1": rng.standard_normal(n), "f2": rng.standard_normal(n)})
    returns = pd.Series(features["f1"] * 0.05 + rng.standard_normal(n) * 0.01)

    det = PCMCIDriftDetector(corr_threshold=0.1, link_change_fraction=0.5)
    det.fit_baseline(features, returns)
    alert, diag = det.check(features, returns)
    assert not alert
    assert diag["status"] in ("ok", "baseline_initialized")


def test_causal_drift_detects_structure_change():
    rng = np.random.default_rng(1)
    n = 80
    base_f = pd.DataFrame({"f1": rng.standard_normal(n)})
    base_r = pd.Series(base_f["f1"] * 0.2)

    det = PCMCIDriftDetector(corr_threshold=0.1, link_change_fraction=0.2)
    det.fit_baseline(base_f, base_r)

    shifted_f = pd.DataFrame({"f2": rng.standard_normal(n)})
    shifted_r = pd.Series(shifted_f["f2"] * 0.25)
    alert, _ = det.check(shifted_f, shifted_r)
    assert alert
