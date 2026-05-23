"""Smoke tests for the Backtest Playground orchestrator.

These tests inject synthetic OHLCV via monkeypatching so the suite never
touches yfinance / network. They check:

  - Proxy signal generation produces non-trivial, finite values.
  - Snapshot persistence round-trip (params + equity curve + stats).
  - End-to-end ``run_playground_backtest`` returns a populated result.
  - Changing regime weights actually shifts the equity curve.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from backtest import playground  # noqa: E402
from backtest.playground import (
    PlaygroundParams,
    _build_proxy_signals,
    _classify_regime,
    list_snapshots,
    run_playground_backtest,
)


# ---------------------------------------------------------------------------
# Synthetic OHLCV
# ---------------------------------------------------------------------------

def _make_prices(tickers: list[str], n_days: int = 320, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2022-01-03", periods=n_days)
    drift = rng.normal(0.0006, 0.002, size=len(tickers))
    rets = rng.normal(0.0, 0.015, size=(n_days, len(tickers))) + drift
    prices = 100.0 * np.cumprod(1.0 + rets, axis=0)
    return pd.DataFrame(prices, index=dates, columns=tickers)


def _make_benchmark(n_days: int = 320, seed: int = 17) -> pd.Series:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2022-01-03", periods=n_days)
    rets = rng.normal(0.0004, 0.01, size=n_days)
    return pd.Series(100.0 * np.cumprod(1.0 + rets), index=dates, name="SPY")


@pytest.fixture
def patched_data(monkeypatch):
    universe = ["AAA", "BBB", "CCC", "DDD", "EEE"]
    prices = _make_prices(universe)
    bench = _make_benchmark()

    def fake_panel(tickers, start, end, progress_cb=None):
        cols = [t for t in tickers if t in prices.columns]
        if not cols:
            return pd.DataFrame()
        df = prices[cols].copy()
        df = df.loc[df.index <= pd.to_datetime(end)]
        return df

    def fake_bench(start, end):
        return bench.loc[bench.index <= pd.to_datetime(end)]

    monkeypatch.setattr(playground, "_load_universe_panel", fake_panel)
    monkeypatch.setattr(playground, "_load_benchmark", fake_bench)
    return universe, prices, bench


# ---------------------------------------------------------------------------
# Proxy signals
# ---------------------------------------------------------------------------

def test_proxy_signals_finite_and_nonzero():
    prices = _make_prices(["AAA", "BBB", "CCC", "DDD"], n_days=120)
    bench = _make_benchmark(n_days=120)
    sigs = _build_proxy_signals(prices, bench)
    assert set(sigs) == {"lgbm", "sentiment", "hmm"}
    for name, df in sigs.items():
        assert df.shape == prices.shape
        assert np.isfinite(df.values).all(), f"{name} has non-finite values"
        # After warmup window the signal should not be uniformly zero.
        late = df.iloc[40:]
        assert late.abs().sum().sum() > 0.0, f"{name} is degenerate"


def test_regime_classifier_returns_known_label():
    bench = _make_benchmark(n_days=300)
    label = _classify_regime(bench, bench.index[-1])
    assert label in {"bull", "bear", "transition"}


# ---------------------------------------------------------------------------
# Snapshot persistence
# ---------------------------------------------------------------------------

def test_snapshot_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(playground, "RESULTS_DIR", tmp_path)
    params = PlaygroundParams(
        start_date="2023-01-01",
        end_date="2023-06-30",
        universe=["AAA", "BBB"],
        note="unit-test",
    )
    equity = pd.Series(np.linspace(100_000, 110_000, 50),
                       index=pd.bdate_range("2023-01-03", periods=50),
                       name="equity")
    weights = pd.DataFrame(0.5, index=equity.index, columns=["AAA", "BBB"])
    stats = {"sharpe": 1.2, "max_drawdown": -0.05, "cagr": 0.10, "final_equity": 110_000}

    out = playground._persist_snapshot(  # noqa: SLF001
        params=params,
        equity=equity,
        gross_equity=equity,
        weights=weights,
        stats=stats,
        contributions=pd.DataFrame(),
        benchmark=pd.Series(dtype=float),
        base_dir=tmp_path,
    )
    assert out.exists()
    assert (out / "params.yaml").exists()
    assert (out / "stats.json").exists()
    assert (out / "equity_curve.parquet").exists()

    snaps = list_snapshots(base_dir=tmp_path)
    assert len(snaps) == 1
    row = snaps.iloc[0]
    assert row["start_date"] == "2023-01-01"
    assert row["note"] == "unit-test"
    assert abs(row["sharpe"] - 1.2) < 1e-6


# ---------------------------------------------------------------------------
# End-to-end smoke
# ---------------------------------------------------------------------------

def _smoke_params(universe: list[str]) -> PlaygroundParams:
    return PlaygroundParams(
        start_date="2022-06-01",
        end_date="2022-12-30",
        universe=list(universe),
        initial_capital=100_000.0,
        slippage_bps=2.0,
        commission_bps=0.5,
        max_position=0.30,
        max_turnover=1.0,
        max_vol_ann=0.80,
        sector_cap=1.0,
        min_signal_strength=0.0,
        use_orthogonality=False,
    )


def test_run_playground_backtest_end_to_end(patched_data, tmp_path, monkeypatch):
    universe, _, _ = patched_data
    monkeypatch.setattr(playground, "RESULTS_DIR", tmp_path)

    params = _smoke_params(universe)
    result = run_playground_backtest(params)

    assert result.equity_curve is not None
    assert not result.equity_curve.empty
    assert not result.weights.empty
    assert "sharpe" in result.stats
    assert "max_drawdown" in result.stats
    assert result.snapshot_path is not None
    assert result.snapshot_path.exists()


def test_changing_regime_weights_changes_equity(patched_data, tmp_path, monkeypatch):
    universe, _, _ = patched_data
    monkeypatch.setattr(playground, "RESULTS_DIR", tmp_path)

    base = _smoke_params(universe)
    r1 = run_playground_backtest(base)

    flipped = _smoke_params(universe)
    flipped.regime_weights = {
        "bull":       {"lgbm": 0.05, "sentiment": 0.05, "hmm": 0.90},
        "bear":       {"lgbm": 0.05, "sentiment": 0.05, "hmm": 0.90},
        "transition": {"lgbm": 0.05, "sentiment": 0.05, "hmm": 0.90},
    }
    r2 = run_playground_backtest(flipped)

    common = r1.equity_curve.index.intersection(r2.equity_curve.index)
    assert len(common) > 10
    diff = (r1.equity_curve.reindex(common) - r2.equity_curve.reindex(common)).abs()
    # Equity curves should not be identical when council weights flip drastically.
    assert diff.max() > 1.0, "Changing regime weights had no visible effect on equity"


def test_empty_universe_rejected():
    params = PlaygroundParams(
        start_date="2023-01-01",
        end_date="2023-06-30",
        universe=[],
    )
    with pytest.raises(ValueError):
        run_playground_backtest(params)
