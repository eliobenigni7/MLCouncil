"""Tests for council/ (Agent 05): aggregator, conformal prediction, portfolio.

Coverage
--------
1. test_weights_sum_to_one           — aggregated weights sum to 1.0 per regime
2. test_regime_shift_changes_weights — bull vs bear produce different weight vectors
3. test_conformal_coverage           — empirical coverage >= target on held-out data
4. test_position_multiplier_range    — all multipliers are in [0.2, 2.0]
5. test_portfolio_budget_constraint  — tier policy budget behavior
6. test_turnover_constraint          — |new_w - old_w|_1 <= max_turnover
"""

from __future__ import annotations

from datetime import date, timedelta
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Synthetic data helpers
# ---------------------------------------------------------------------------

def _make_signals(
    tickers: list[str] | None = None,
    seed: int = 42,
) -> dict[str, pd.Series]:
    """Dummy z-scored signals for the three standard council models."""
    if tickers is None:
        tickers = [f"S{i:03d}" for i in range(15)]
    rng = np.random.default_rng(seed)
    return {
        "lgbm":      pd.Series(rng.standard_normal(len(tickers)), index=tickers),
        "sentiment": pd.Series(rng.standard_normal(len(tickers)), index=tickers),
        "hmm":       pd.Series(rng.standard_normal(len(tickers)), index=tickers),
    }


def _make_covariance(n: int = 15, seed: int = 1) -> pd.DataFrame:
    """Return a valid PSD daily covariance matrix for n tickers."""
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((n, n)) * 0.005
    cov = A.T @ A + np.eye(n) * 1e-4
    tickers = [f"S{i:03d}" for i in range(n)]
    return pd.DataFrame(cov, index=tickers, columns=tickers)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def aggregator():
    from council.aggregator import CouncilAggregator
    return CouncilAggregator()


@pytest.fixture(scope="module")
def fitted_sizer():
    """ConformalPositionSizer fitted on synthetic linear data (400 samples)."""
    from council.conformal import ConformalPositionSizer

    rng = np.random.default_rng(42)
    n_calib, n_features = 400, 10
    X_calib = rng.standard_normal((n_calib, n_features))
    y_calib = (
        2.0 * X_calib[:, 0]
        - 1.5 * X_calib[:, 1]
        + rng.standard_normal(n_calib) * 0.5
    )

    sizer = ConformalPositionSizer(coverage=0.90)
    sizer.fit(X_calib, y_calib)
    return sizer


@pytest.fixture(scope="module")
def constructor():
    from council.portfolio import PortfolioConstructor
    return PortfolioConstructor()


# ---------------------------------------------------------------------------
# CouncilAggregator tests
# ---------------------------------------------------------------------------

class TestCouncilAggregator:
    def test_weights_sum_to_one(self, aggregator):
        """Aggregated model weights must sum to 1.0 for every regime."""
        signals = _make_signals()
        base_date = date(2024, 1, 15)

        for i, regime in enumerate(("bull", "bear", "transition")):
            d = base_date + timedelta(days=i)
            aggregator.aggregate(signals, regime, date=d)
            weights = aggregator._weights_log[d]["weights"]
            total = sum(weights.values())
            assert abs(total - 1.0) < 1e-9, (
                f"regime={regime}: weights sum to {total:.9f}, expected 1.0"
            )

    def test_regime_shift_changes_weights(self, aggregator):
        """Bull and bear regimes must produce different weight compositions."""
        signals = _make_signals(seed=11)
        d_bull = date(2024, 2, 1)
        d_bear = date(2024, 2, 2)

        aggregator.aggregate(signals, "bull", date=d_bull)
        aggregator.aggregate(signals, "bear", date=d_bear)

        bull_w = aggregator._weights_log[d_bull]["weights"]
        bear_w = aggregator._weights_log[d_bear]["weights"]

        all_models = set(bull_w) | set(bear_w)
        max_diff = max(
            abs(bull_w.get(m, 0.0) - bear_w.get(m, 0.0)) for m in all_models
        )
        assert max_diff > 1e-6, (
            "Bull and bear weights are identical — regime conditioning is not working. "
            f"bull={bull_w}, bear={bear_w}"
        )

    def test_aggregate_returns_zscore_structure(self, aggregator):
        """Output should be a pd.Series with mean ≈ 0 and positive std."""
        signals = _make_signals(seed=99)
        result = aggregator.aggregate(signals, "transition", date=date(2024, 3, 1))

        assert isinstance(result, pd.Series), "aggregate() must return pd.Series"
        assert float(result.std()) > 1e-6, "z-score std should be positive"
        # Mean should be numerically close to 0 after z-scoring
        assert abs(float(result.mean())) < 1e-9, (
            f"z-score mean = {result.mean():.6f}, expected 0.0"
        )

    def test_aggregate_output_index_is_tickers(self, aggregator):
        """Output index should be the union of tickers across all signal Series."""
        tickers = ["AAPL", "MSFT", "GOOGL"]
        signals = _make_signals(tickers=tickers, seed=7)
        result = aggregator.aggregate(signals, "bull", date=date(2024, 3, 2))
        assert set(result.index) == set(tickers)
        assert result.index.name == "ticker"

    def test_aggregate_handles_first_signal_history_row_with_misaligned_tickers(self):
        """First aggregate call should initialize history frames without crashing."""
        from council.aggregator import CouncilAggregator

        agg = CouncilAggregator()
        signals = {
            "lgbm": pd.Series([1.0, -0.2], index=["AAPL", "MSFT"]),
            "sentiment": pd.Series([0.5, 0.8], index=["MSFT", "GOOGL"]),
        }

        result = agg.aggregate(signals, "bull", date=date(2024, 3, 3))

        assert not result.empty
        assert set(result.index) == {"AAPL", "MSFT", "GOOGL"}
        assert set(agg._ortho_monitor._signal_history["lgbm"].columns) == {"AAPL", "MSFT"}
        assert set(agg._ortho_monitor._signal_history["sentiment"].columns) == {"MSFT", "GOOGL"}

    def test_update_performance_populates_ic_history(self):
        """update_performance should fill _ic_by_date with 30+ entries."""
        from council.aggregator import CouncilAggregator

        rng = np.random.default_rng(0)
        tickers = [f"T{i}" for i in range(10)]
        n_days = 40
        dates = [date(2023, 1, 2) + timedelta(days=i) for i in range(n_days)]

        signals_df = pd.DataFrame(
            rng.standard_normal((n_days, 10)), index=dates, columns=tickers
        )
        returns_df = pd.DataFrame(
            rng.standard_normal((n_days, 10)), index=dates, columns=tickers
        )

        agg = CouncilAggregator()
        agg.update_performance(
            {"lgbm": signals_df},
            returns_df,
            date=dates[-1],
        )

        assert "lgbm" in agg._ic_by_date, "IC history missing for 'lgbm'"
        assert len(agg._ic_by_date["lgbm"]) >= 30, (
            f"Expected >= 30 IC entries, got {len(agg._ic_by_date['lgbm'])}"
        )

    def test_adaptive_weights_kick_in_after_history(self):
        """With 60+ days of positive IC, adaptive weighting should deviate from base."""
        from council.aggregator import CouncilAggregator

        rng = np.random.default_rng(5)
        tickers = [f"T{i}" for i in range(20)]
        n_days = 70
        dates = [date(2022, 1, 3) + timedelta(days=i) for i in range(n_days)]

        # Give lgbm strong positive IC by making its signal predictive
        returns_mat = rng.standard_normal((n_days, 20))
        signals_lgbm = pd.DataFrame(
            returns_mat + rng.standard_normal((n_days, 20)) * 0.1,
            index=dates, columns=tickers,
        )
        signals_other = pd.DataFrame(
            rng.standard_normal((n_days, 20)), index=dates, columns=tickers
        )
        returns_df = pd.DataFrame(returns_mat, index=dates, columns=tickers)

        agg = CouncilAggregator()
        agg.update_performance(
            {"lgbm": signals_lgbm, "sentiment": signals_other, "hmm": signals_other},
            returns_df,
            date=dates[-1],
        )

        signals_live = _make_signals(tickers=tickers[:15], seed=3)
        d_test = date(2022, 4, 1)
        agg.aggregate(signals_live, "bull", date=d_test)
        weights = agg._weights_log[d_test]["weights"]

        # Weights must still sum to 1
        assert abs(sum(weights.values()) - 1.0) < 1e-9

    def test_get_attribution_columns(self):
        """get_attribution should return a DataFrame with the required columns."""
        from council.aggregator import CouncilAggregator

        agg = CouncilAggregator()
        signals = _make_signals()
        d = date(2024, 5, 1)
        agg.aggregate(signals, "bull", date=d)

        attr = agg.get_attribution(d)
        required = {"model_name", "weight", "ic_rolling_30d", "sharpe_rolling_60d", "pnl_contribution"}
        assert required.issubset(set(attr.columns)), (
            f"Missing columns: {required - set(attr.columns)}"
        )


# ---------------------------------------------------------------------------
# ConformalPositionSizer tests
# ---------------------------------------------------------------------------

class TestConformalPositionSizer:
    def test_conformal_coverage(self, fitted_sizer):
        """Empirical coverage on held-out data must be >= 0.85 (target 0.90).

        The jackknife+ method guarantees coverage >= 1 - 2*alpha*(n/(n+1)).
        For n=300 and alpha=0.10 that is ~ 0.897.  We allow a small tolerance
        for finite-sample variation.
        """
        rng = np.random.default_rng(123)
        n_test, n_features = 300, 10
        X_test = rng.standard_normal((n_test, n_features))
        y_test = (
            2.0 * X_test[:, 0]
            - 1.5 * X_test[:, 1]
            + rng.standard_normal(n_test) * 0.5
        )

        preds, lower, upper = fitted_sizer.get_intervals(X_test)

        assert preds.shape == (n_test,), "predictions shape mismatch"
        assert lower.shape == (n_test,), "lower bound shape mismatch"
        assert upper.shape == (n_test,), "upper bound shape mismatch"
        assert (upper >= lower).all(), "upper bound must be >= lower bound"

        covered = float(np.mean((y_test >= lower) & (y_test <= upper)))
        assert covered >= 0.85, (
            f"Empirical coverage {covered:.3f} < 0.85 "
            "(target 0.90; jackknife+ guarantee violated)"
        )

    def test_position_multiplier_range(self, fitted_sizer):
        """All position multipliers must lie in [0.2, 2.0]."""
        rng = np.random.default_rng(7)
        n = 20
        tickers = [f"S{i:03d}" for i in range(n)]
        X = rng.standard_normal((n, 10))
        council_signal = pd.Series(rng.standard_normal(n), index=tickers)

        multipliers = fitted_sizer.compute_position_multipliers(council_signal, X)

        assert isinstance(multipliers, pd.Series)
        assert len(multipliers) == n
        assert (multipliers >= 0.2 - 1e-9).all(), (
            f"Multipliers below 0.2: {multipliers[multipliers < 0.2].to_dict()}"
        )
        assert (multipliers <= 2.0 + 1e-9).all(), (
            f"Multipliers above 2.0: {multipliers[multipliers > 2.0].to_dict()}"
        )

    def test_multiplier_index_matches_signal(self, fitted_sizer):
        """Multiplier index must match council_signal index exactly."""
        rng = np.random.default_rng(9)
        tickers = ["AAPL", "MSFT", "GOOGL", "AMZN"]
        X = rng.standard_normal((4, 10))
        signal = pd.Series(rng.standard_normal(4), index=tickers)

        mult = fitted_sizer.compute_position_multipliers(signal, X)
        assert list(mult.index) == tickers

    def test_filter_low_confidence_zeros_wide_intervals(self, fitted_sizer):
        """filter_low_confidence at p80 should zero at least 15 % of signals."""
        rng = np.random.default_rng(15)
        n = 50
        tickers = [f"S{i:03d}" for i in range(n)]
        X = rng.standard_normal((n, 10))
        signal = pd.Series(rng.standard_normal(n), index=tickers)

        filtered = fitted_sizer.filter_low_confidence(signal, X, threshold_percentile=80)

        n_zeroed = int((filtered == 0.0).sum())
        assert n_zeroed >= int(n * 0.15), (
            f"Only {n_zeroed}/{n} signals zeroed at p80 — expected >= {int(n * 0.15)}"
        )

    def test_fit_raises_on_bad_input(self):
        """fit() must raise ValueError on mismatched or 1-D X."""
        from council.conformal import ConformalPositionSizer

        sizer = ConformalPositionSizer()
        with pytest.raises(ValueError):
            sizer.fit(np.ones(10), np.ones(10))  # 1-D X

    def test_get_intervals_raises_before_fit(self):
        """get_intervals() must raise RuntimeError if called before fit()."""
        from council.conformal import ConformalPositionSizer

        sizer = ConformalPositionSizer()
        with pytest.raises(RuntimeError):
            sizer.get_intervals(np.ones((5, 3)))


# ---------------------------------------------------------------------------
# PortfolioConstructor tests
# ---------------------------------------------------------------------------

class TestPortfolioConstructor:
    def _inputs(self, n: int = 15, seed: int = 42):
        rng = np.random.default_rng(seed)
        tickers = [f"S{i:03d}" for i in range(n)]
        alpha = pd.Series(rng.standard_normal(n), index=tickers)
        multipliers = pd.Series(np.ones(n), index=tickers)
        current_w = pd.Series(np.ones(n) / n, index=tickers)
        cov = _make_covariance(n, seed=seed)
        return alpha, multipliers, current_w, cov, tickers

    def test_portfolio_budget_constraint(self, constructor):
        """Cash reserve should apply only to the large-portfolio tier."""
        alpha, mult, current_w, cov, _ = self._inputs()
        target_mid = constructor.optimize(alpha, mult, current_w, cov, portfolio_value=50_000)
        target_large = constructor.optimize(alpha, mult, current_w, cov, portfolio_value=100_000)

        assert isinstance(target_mid, pd.Series), "optimize() must return pd.Series"
        assert abs(target_mid.sum() - 1.0) < 1e-4, (
            f"Mid-tier weights sum to {target_mid.sum():.6f}, expected 1.0"
        )
        assert abs(target_large.sum() - 0.85) < 1e-4, (
            f"Large-tier weights sum to {target_large.sum():.6f}, expected 0.85"
        )

    def test_turnover_constraint(self, constructor):
        """One-way turnover |new - old|_1 must be <= max_turnover."""
        alpha, mult, current_w, cov, _ = self._inputs()
        target = constructor.optimize(alpha, mult, current_w, cov)

        aligned_current = current_w.reindex(target.index).fillna(0.0)
        turnover = float((target - aligned_current).abs().sum())
        assert turnover <= constructor.max_turnover + 1e-4, (
            f"Turnover {turnover:.4f} exceeds max_turnover={constructor.max_turnover}"
        )

    def test_long_only_constraint(self, constructor):
        """All target weights must be >= 0 (long-only PoC)."""
        alpha, mult, current_w, cov, _ = self._inputs()
        target = constructor.optimize(alpha, mult, current_w, cov)

        assert (target >= -1e-6).all(), (
            f"Short positions found: {target[target < -1e-6].to_dict()}"
        )

    def test_max_position_constraint(self, constructor):
        """No single weight may exceed max_position after optimization."""
        alpha, mult, current_w, cov, _ = self._inputs()
        target = constructor.optimize(alpha, mult, current_w, cov)

        assert (target <= constructor.max_position + 1e-4).all(), (
            f"Weights exceeding max_position={constructor.max_position}: "
            f"{target[target > constructor.max_position + 1e-4].to_dict()}"
        )

    def test_portfolio_index_matches_alpha(self, constructor):
        """Output tickers must be selected from the alpha universe."""
        alpha, mult, current_w, cov, tickers = self._inputs()
        target = constructor.optimize(alpha, mult, current_w, cov)
        assert set(target.index).issubset(set(tickers))
        assert len(target.index) > 0

    def test_compute_orders_direction(self, constructor):
        """compute_orders must correctly classify buy/sell directions."""
        tickers = ["A", "B", "C"]
        # A increases, B decreases, C decreases
        target  = pd.Series([0.50, 0.30, 0.20], index=tickers)
        current = pd.Series([0.30, 0.40, 0.30], index=tickers)

        orders = constructor.compute_orders(target, current, portfolio_value=100_000.0)

        assert isinstance(orders, pd.DataFrame)
        order_map = orders.set_index("ticker")["direction"].to_dict()
        assert order_map.get("A") == "buy",  "A weight increases → should be buy"
        assert order_map.get("B") == "sell", "B weight decreases → should be sell"
        assert order_map.get("C") == "sell", "C weight decreases → should be sell"

    def test_compute_orders_quantity_positive(self, constructor):
        """All order quantities must be positive USD amounts."""
        tickers = ["X", "Y"]
        target  = pd.Series([0.70, 0.30], index=tickers)
        current = pd.Series([0.50, 0.50], index=tickers)

        orders = constructor.compute_orders(target, current, portfolio_value=50_000.0)
        assert (orders["quantity"] > 0).all(), "All quantities must be positive"

    def test_drawdown_scale_reduces_exposure_without_renormalization(self, constructor):
        """Drawdown scaling should reduce gross exposure (cash increases)."""
        target = pd.Series(
            [0.40, 0.30, 0.15],
            index=["AAPL", "MSFT", "GOOGL"],
            name="target_weight",
        )
        scaled = constructor.apply_drawdown_scale(
            target_weights=target,
            portfolio_return_5d=-0.14,
        )

        assert scaled.sum() < target.sum()
        assert scaled.sum() == pytest.approx(target.sum() * 0.25, rel=1e-6)

    def test_compute_orders_empty_on_no_change(self, constructor):
        """compute_orders returns empty DataFrame when no meaningful trades exist."""
        tickers = ["A", "B"]
        weights = pd.Series([0.5, 0.5], index=tickers)
        orders = constructor.compute_orders(weights, weights, portfolio_value=100.0)
        assert len(orders) == 0, "No orders expected when weights are identical"

    def test_sector_aware_fallback_respects_dynamic_sector_cap(self, constructor):
        """Il fallback deve restare investibile senza concentrare tutto nel tech."""
        import builtins
        from data.features.sector_exposure import (
            compute_effective_sector_cap,
            compute_sector_exposures,
        )

        tickers = [
            "AAPL", "MSFT", "GOOGL", "NVDA", "META",
            "AMZN", "ABNB", "UBER", "ETSY",
            "ROKU",
        ]
        alpha = pd.Series(range(len(tickers), 0, -1), index=tickers, dtype=float)
        multipliers = pd.Series(1.0, index=tickers)
        current_w = pd.Series(0.0, index=tickers)
        cov = pd.DataFrame(np.eye(len(tickers)) * 0.0001, index=tickers, columns=tickers)
        real_import = builtins.__import__

        def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "cvxpy":
                raise ModuleNotFoundError("No module named 'cvxpy'")
            return real_import(name, globals, locals, fromlist, level)

        with patch("builtins.__import__", side_effect=guarded_import):
            target = constructor.optimize(alpha, multipliers, current_w, cov)

        sector_weights = compute_sector_exposures(target)
        effective_sector_cap = compute_effective_sector_cap(
            tickers,
            base_sector_cap=constructor.sector_cap,
            max_position=max(constructor.max_position, 0.13),
        )
        assert abs(target.sum() - 0.85) < 1e-6
        assert sector_weights["Technology"] <= effective_sector_cap + 1e-6
        assert sector_weights["Consumer Discretionary"] <= effective_sector_cap + 1e-6
