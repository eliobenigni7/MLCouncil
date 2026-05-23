"""Tests for MoE shadow logging (T3.1)."""

from __future__ import annotations

import pandas as pd


def test_log_moe_shadow_writes_parquet(tmp_path):
    from council.moe_gating import log_moe_shadow

    linear = pd.Series([0.1, -0.2], index=["AAPL", "MSFT"])
    moe = pd.Series([0.2, -0.1], index=["AAPL", "MSFT"])
    path = log_moe_shadow(
        "2024-01-15",
        linear_signal=linear,
        moe_signal=moe,
        gate_weights=[0.6, 0.4],
        expert_order=["lgbm", "sentiment"],
        effective_weights={"lgbm": 0.55, "sentiment": 0.45},
        out_dir=tmp_path,
    )
    df = pd.read_parquet(path)
    assert len(df) == 2
    assert "moe_signal" in df.columns
