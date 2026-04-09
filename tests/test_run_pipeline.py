from __future__ import annotations

import importlib.util
import sys
from datetime import date
from pathlib import Path

import pandas as pd
import polars as pl


ROOT = Path(__file__).resolve().parents[1]


def _load_run_pipeline():
    module_name = "run_pipeline_module_test"
    if module_name in sys.modules:
        return sys.modules[module_name]

    spec = importlib.util.spec_from_file_location(
        module_name, ROOT / "scripts" / "run_pipeline.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_step_portfolio_does_not_shadow_pandas_import(monkeypatch, tmp_path):
    module = _load_run_pipeline()

    class DummySizer:
        def compute_position_multipliers(self, signal_live, x_live):
            return pd.Series(1.0, index=signal_live.index)

        def filter_low_confidence(self, signal_live, x_live, threshold_percentile=80):
            return signal_live

    class DummyPortfolioConstructor:
        def optimize(
            self,
            *,
            alpha_signals,
            position_multipliers,
            current_weights,
            returns_covariance,
        ):
            return pd.Series([0.6, 0.4], index=current_weights.index, dtype=float)

        def compute_orders(self, target_w, current_w, portfolio_value):
            return pd.DataFrame(
                {
                    "direction": ["buy", "sell"],
                    "quantity": [10, 5],
                    "target_weight": [0.6, 0.4],
                },
                index=target_w.index,
            )

    monkeypatch.setattr(module, "ORDERS_DIR", tmp_path)

    import council.portfolio as portfolio_module

    monkeypatch.setattr(portfolio_module, "PortfolioConstructor", DummyPortfolioConstructor)

    last_date = date(2026, 4, 8)
    council_signal = pd.Series({"AAPL": 0.8, "MSFT": 0.2}, dtype=float)
    feat_cols = ["feature_1"]
    feat_test = pl.DataFrame(
        {
            "ticker": ["AAPL", "MSFT"],
            "valid_time": [last_date, last_date],
            "feature_1": [1.0, 2.0],
        }
    )
    ohlcv = pl.DataFrame(
        {
            "ticker": ["AAPL", "AAPL", "AAPL", "MSFT", "MSFT", "MSFT"],
            "valid_time": [
                date(2026, 4, 6),
                date(2026, 4, 7),
                date(2026, 4, 8),
                date(2026, 4, 6),
                date(2026, 4, 7),
                date(2026, 4, 8),
            ],
            "adj_close": [100.0, 101.0, 102.0, 200.0, 198.0, 201.0],
        }
    )

    target = module.step_portfolio(
        council_signal=council_signal,
        sizer=DummySizer(),
        feat_cols=feat_cols,
        feat_test=feat_test,
        ohlcv=ohlcv,
        last_date=last_date,
        portfolio_value=100_000.0,
    )

    assert list(target.index) == ["AAPL", "MSFT"]
    assert (tmp_path / f"{last_date}.parquet").exists()
