"""Tests for models/base.py, models/technical.py, models/regime.py.

Synthetic data strategy
-----------------------
* Features have cross-sectional structure (correlations, vol clustering) to
  resemble real Alpha158 output without requiring live market data.
* Macro data has simulated regime shifts (low-vol bull / high-vol bear periods)
  so HMM has signal to recover.

Coverage
--------
1. test_lgbm_fit_predict          — fit 2y synthetic, predict 6m, IC > -0.5
2. test_cpcv_no_lookahead         — no test date in training for its fold
3. test_embargo_applied           — embargo_days positions before test excluded
4. test_hmm_states_labeled        — bull state mean return > bear state
5. test_regime_probabilities_sum  — probabilities sum to 1.0
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
import polars as pl
import pytest


# ---------------------------------------------------------------------------
# Synthetic data generators
# ---------------------------------------------------------------------------

def make_synthetic_features(
    n_stocks: int = 20,
    n_days: int = 500,
    seed: int = 42,
) -> tuple[pl.DataFrame, pd.Series]:
    """Generate realistic synthetic features and aligned targets.

    Features have cross-sectional correlation structure (factor model).
    Returns (features_df, targets_series) where targets are cross-sectional
    rank percentiles indexed by a MultiIndex (ticker, valid_time).
    """
    rng = np.random.default_rng(seed)
    tickers = [f"STK{i:03d}" for i in range(n_stocks)]

    # Business days only
    start = date(2021, 1, 4)
    all_dates: list[date] = []
    d = start
    while len(all_dates) < n_days:
        if d.weekday() < 5:
            all_dates.append(d)
        d += timedelta(days=1)

    n_feat = 30

    # Factor model: common factor + idiosyncratic noise
    common_factor = rng.standard_normal((n_days, 3))            # 3 latent factors
    loadings = rng.standard_normal((n_stocks, 3)) * 0.5         # per-stock loadings
    idiosync = rng.standard_normal((n_days, n_stocks)) * 0.8

    # n_days × n_stocks return matrix
    returns_matrix = common_factor @ loadings.T + idiosync      # (T, N)

    feat_names = [f"feat_{i:02d}" for i in range(n_feat)]

    rows = []
    target_index = []
    target_values = []

    for t_idx, dt in enumerate(all_dates):
        # Cross-sectional returns for this day
        day_returns = returns_matrix[t_idx]  # (N,)

        # Build n_feat features as noisy functions of the return factors
        # (some predictive, some noise) — no look-ahead
        if t_idx < 5:
            # Warm-up: not enough history
            continue

        history_5d = returns_matrix[max(0, t_idx - 5) : t_idx]  # up to 5 prior rows
        history_20d = returns_matrix[max(0, t_idx - 20) : t_idx]

        mom_5d = history_5d.mean(axis=0) if len(history_5d) > 0 else np.zeros(n_stocks)
        mom_20d = history_20d.mean(axis=0) if len(history_20d) > 0 else np.zeros(n_stocks)
        vol_5d = history_5d.std(axis=0) + 1e-6 if len(history_5d) > 1 else np.ones(n_stocks)

        noise = rng.standard_normal((n_stocks, n_feat)) * 0.5

        feat_matrix = np.column_stack(
            [
                mom_5d,
                mom_20d,
                vol_5d,
                mom_5d / vol_5d,           # momentum / vol
                mom_20d / vol_5d,
            ]
            + [noise[:, i] for i in range(n_feat - 5)]
        )  # (N, n_feat)

        # Cross-sectional rank of next-day return (target)
        next_idx = t_idx + 1
        if next_idx >= n_days:
            continue
        next_ret = returns_matrix[next_idx]
        ranks = pd.Series(next_ret).rank(pct=True).values

        for s_idx, ticker in enumerate(tickers):
            row = {"ticker": ticker, "valid_time": dt}
            for f_idx, fname in enumerate(feat_names):
                row[fname] = float(feat_matrix[s_idx, f_idx])
            rows.append(row)
            target_index.append((ticker, dt))
            target_values.append(float(ranks[s_idx]))

    feat_df = pl.DataFrame(rows).with_columns(
        pl.col("valid_time").cast(pl.Date)
    )
    targets = pd.Series(
        target_values,
        index=pd.MultiIndex.from_tuples(target_index, names=["ticker", "valid_time"]),
        name="rank_fwd_1d",
    )
    return feat_df, targets


def make_synthetic_macro(n_days: int = 500, seed: int = 7) -> pl.DataFrame:
    """Generate macro series with two clear regimes (bull / bear) and a transition.

    Columns: valid_time, sp500_ret_20d, vix, yield_spread
    Regime structure:
      - First third:   bull (positive drift, low vol, low VIX)
      - Middle third:  bear (negative drift, high vol, high VIX)
      - Final third:   transition (mixed)
    """
    rng = np.random.default_rng(seed)
    start = date(2021, 1, 4)
    all_dates: list[date] = []
    d = start
    while len(all_dates) < n_days:
        if d.weekday() < 5:
            all_dates.append(d)
        d += timedelta(days=1)

    n = len(all_dates)
    cut1, cut2 = n // 3, 2 * n // 3

    # sp500 cumulative return → daily return proxy
    drift = np.concatenate(
        [
            rng.normal(0.003, 0.008, cut1),           # bull
            rng.normal(-0.005, 0.018, cut2 - cut1),   # bear
            rng.normal(0.001, 0.012, n - cut2),        # transition
        ]
    )
    sp500_ret_20d = np.convolve(drift, np.ones(20) / 20, mode="same")

    vix = np.concatenate(
        [
            rng.normal(14, 2, cut1),
            rng.normal(28, 4, cut2 - cut1),
            rng.normal(20, 3, n - cut2),
        ]
    ).clip(10, 80)

    yield_spread = np.concatenate(
        [
            rng.normal(1.5, 0.15, cut1),
            rng.normal(0.2, 0.1, cut2 - cut1),
            rng.normal(0.9, 0.2, n - cut2),
        ]
    )

    return pl.DataFrame(
        {
            "valid_time": all_dates,
            "sp500_ret_20d": sp500_ret_20d.tolist(),
            "vix": vix.tolist(),
            "yield_spread": yield_spread.tolist(),
        }
    ).with_columns(pl.col("valid_time").cast(pl.Date))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def synthetic_data():
    """(features_df, targets) tuple — module-scoped for speed."""
    return make_synthetic_features(n_stocks=20, n_days=500)


@pytest.fixture(scope="module")
def macro_df():
    return make_synthetic_macro(n_days=500)


@pytest.fixture(scope="module")
def fitted_lgbm(synthetic_data):
    """Pre-fitted TechnicalModel shared across LightGBM tests."""
    from models.technical import TechnicalModel

    features, targets = synthetic_data
    model = TechnicalModel()
    # Use faster settings for tests
    model._params = {
        "n_estimators": 100,
        "learning_rate": 0.1,
        "num_leaves": 16,
        "min_child_samples": 5,
        "verbose": -1,
        "random_state": 42,
    }
    model._n_splits = 4
    model._embargo_days = 3
    model._n_test_folds = 2
    model.fit(features, targets)
    return model


@pytest.fixture(scope="module")
def fitted_hmm(macro_df):
    """Pre-fitted RegimeModel shared across HMM tests."""
    from models.regime import RegimeModel

    model = RegimeModel(n_states=3)
    model._n_iter = 50  # faster for tests
    model.fit(macro_df)
    return model


# ---------------------------------------------------------------------------
# LightGBM tests
# ---------------------------------------------------------------------------

class TestLGBM:
    def test_lgbm_predict_zero_fills_missing_features(self):
        from unittest.mock import MagicMock

        from models.technical import TechnicalModel

        features = pl.DataFrame(
            {
                "ticker": ["AAA", "BBB"],
                "valid_time": [date(2026, 4, 27), date(2026, 4, 27)],
                "feat_a": [1.0, 2.0],
            }
        )

        model = TechnicalModel.__new__(TechnicalModel)
        model._model = MagicMock()
        model._feature_cols = ["feat_a", "vix"]
        model._model.predict.side_effect = lambda X: X["feat_a"].to_numpy() + X["vix"].to_numpy()

        preds = model.predict(features)

        assert isinstance(preds, pd.Series)
        assert len(preds) == 2
        passed = model._model.predict.call_args.args[0]
        assert list(passed.columns) == ["feat_a", "vix"]
        assert (passed["vix"] == 0.0).all()

    def test_lgbm_fit_predict(self, synthetic_data, fitted_lgbm):
        """fit on 2y synthetic, predict on 6m hold-out; IC must be > -0.5."""
        from scipy.stats import spearmanr

        features, targets = synthetic_data
        model = fitted_lgbm

        # Hold-out: last 6 months (~125 business days) of the data
        all_dates = sorted(features["valid_time"].unique().to_list())
        cutoff = all_dates[-125]
        holdout = features.filter(pl.col("valid_time") >= cutoff)

        preds = model.predict(holdout)
        assert isinstance(preds, pd.Series), "predict() must return pd.Series"
        assert not preds.isna().all(), "All predictions are NaN"

        # Align with targets on hold-out tickers/dates
        holdout_pd = holdout.to_pandas()
        # Normalize valid_time to Python date for merge compatibility
        if pd.api.types.is_datetime64_any_dtype(holdout_pd["valid_time"]):
            holdout_pd["valid_time"] = holdout_pd["valid_time"].dt.date
        holdout_pd["score"] = preds.values

        # Check IC is not catastrophic (model should at minimum not anti-predict)
        t_reset = targets.reset_index()
        if pd.api.types.is_datetime64_any_dtype(t_reset["valid_time"]):
            t_reset["valid_time"] = t_reset["valid_time"].dt.date
        holdout_pd = holdout_pd.merge(t_reset, on=["ticker", "valid_time"], how="inner")
        if len(holdout_pd) > 10:
            ic, _ = spearmanr(holdout_pd["score"], holdout_pd["rank_fwd_1d"])
            assert ic > -0.5, f"IC too negative ({ic:.3f}): model is badly misconfigured"

    def test_predict_returns_zscore_structure(self, fitted_lgbm, synthetic_data):
        """Z-scores on a single date should have mean ≈ 0 and std > 0."""
        features, _ = synthetic_data
        last_date = features["valid_time"].max()
        single_day = features.filter(pl.col("valid_time") == last_date)

        preds = fitted_lgbm.predict(single_day)
        assert abs(preds.mean()) < 1.0, "Z-score mean should be near 0"
        assert preds.std() > 0, "Z-score std should be positive"

    def test_predict_index_is_tickers(self, fitted_lgbm, synthetic_data):
        features, _ = synthetic_data
        last_date = features["valid_time"].max()
        single_day = features.filter(pl.col("valid_time") == last_date)

        preds = fitted_lgbm.predict(single_day)
        assert preds.index.name == "ticker"
        assert set(preds.index) == set(single_day["ticker"].to_list())

    def test_get_metadata(self, fitted_lgbm):
        meta = fitted_lgbm.get_metadata()
        assert meta["name"] == "lgbm"
        assert meta["last_trained"] is not None
        assert meta["n_features"] > 0

    def test_shap_importance_top20(self, fitted_lgbm):
        imp = fitted_lgbm.get_shap_importance()
        assert isinstance(imp, pd.DataFrame)
        assert "feature" in imp.columns
        assert "shap_importance" in imp.columns
        assert len(imp) <= 20
        # Importance should be non-negative (mean |SHAP|)
        assert (imp["shap_importance"] >= 0).all()


# ---------------------------------------------------------------------------
# CPCV correctness tests
# ---------------------------------------------------------------------------

class TestCPCV:
    def _get_splits(self, n=200, n_splits=6, embargo=5, n_test=2):
        from models.technical import cpcv_split

        dates = [date(2020, 1, 1) + timedelta(days=i) for i in range(n)]
        return list(cpcv_split(dates, n_splits=n_splits, embargo_days=embargo, n_test_folds=n_test))

    def test_cpcv_no_lookahead(self):
        """No test date should appear in the training set of its own fold."""
        splits = self._get_splits()
        for train_dates, test_dates in splits:
            overlap = set(train_dates) & set(test_dates)
            assert len(overlap) == 0, (
                f"Lookahead bias: {len(overlap)} dates appear in both train and test"
            )

    def test_embargo_applied(self):
        """The embargo_days positions immediately before each test fold
        start must not appear in the training set."""
        from models.technical import cpcv_split

        n = 300
        embargo_days = 5
        dates = [date(2020, 1, 1) + timedelta(days=i) for i in range(n)]

        for train_dates, test_dates in cpcv_split(
            dates, n_splits=6, embargo_days=embargo_days, n_test_folds=2
        ):
            train_set = set(train_dates)
            test_sorted = sorted(test_dates)

            # Find the start of each contiguous test segment
            test_starts: list[date] = []
            for i, d in enumerate(test_sorted):
                if i == 0 or (d - test_sorted[i - 1]).days > 1:
                    test_starts.append(d)

            for test_start in test_starts:
                test_start_idx = dates.index(test_start)
                embargo_zone = set(
                    dates[max(0, test_start_idx - embargo_days) : test_start_idx]
                )
                leaked = embargo_zone & train_set
                assert len(leaked) == 0, (
                    f"Embargo violated: {len(leaked)} embargoed dates in training "
                    f"(test starts {test_start})"
                )

    def test_combinatorial_count(self):
        """C(n_splits, n_test_folds) folds should be generated."""
        from math import comb
        from models.technical import cpcv_split

        n_splits, n_test = 6, 2
        dates = list(range(300))  # use ints for simplicity
        splits = list(cpcv_split(dates, n_splits=n_splits, embargo_days=0, n_test_folds=n_test))
        expected = comb(n_splits, n_test)
        assert len(splits) == expected, (
            f"Expected {expected} CPCV folds, got {len(splits)}"
        )

    def test_cpcv_train_always_non_empty(self):
        """Every split must have a non-empty training set."""
        splits = self._get_splits()
        for i, (train, test) in enumerate(splits):
            assert len(train) > 0, f"Fold {i}: empty training set"
            assert len(test) > 0, f"Fold {i}: empty test set"


# ---------------------------------------------------------------------------
# HMM tests
# ---------------------------------------------------------------------------

class TestHMM:
    def test_hmm_states_labeled(self, macro_df, fitted_hmm):
        """Bull state should have higher mean return than bear state."""
        hist = fitted_hmm.get_regime_history(macro_df)

        df_pd = macro_df.to_pandas()
        df_pd = df_pd.merge(
            hist[["valid_time", "regime"]],
            on="valid_time",
            how="inner",
        )

        # Use the first feature column (equity return proxy)
        return_col = fitted_hmm._feature_cols[0]

        bull_mean = df_pd.loc[df_pd["regime"] == "bull", return_col].mean()
        bear_mean = df_pd.loc[df_pd["regime"] == "bear", return_col].mean()

        assert bull_mean > bear_mean, (
            f"Bull state mean ({bull_mean:.4f}) is not > bear state mean ({bear_mean:.4f}). "
            "State labelling may be inverted."
        )

    def test_regime_probabilities_sum_to_one(self, macro_df, fitted_hmm):
        """Regime probabilities for the last observation must sum to 1.0."""
        probs = fitted_hmm.predict_probabilities(macro_df)

        total = sum(probs.values())
        assert abs(total - 1.0) < 1e-6, (
            f"Probabilities do not sum to 1.0: {probs} (sum={total:.8f})"
        )

    def test_regime_probabilities_labels(self, macro_df, fitted_hmm):
        """Keys must be exactly the three expected regime labels."""
        probs = fitted_hmm.predict_probabilities(macro_df)
        assert set(probs.keys()) == {"bull", "bear", "transition"}, (
            f"Unexpected probability keys: {set(probs.keys())}"
        )

    def test_predict_regime_valid_label(self, macro_df, fitted_hmm):
        """predict_regime() must return one of the three label strings."""
        regime = fitted_hmm.predict_regime(macro_df)
        assert regime in {"bull", "bear", "transition"}, (
            f"predict_regime() returned unexpected label: '{regime}'"
        )

    def test_regime_history_shape(self, macro_df, fitted_hmm):
        """get_regime_history() must have one row per observation."""
        hist = fitted_hmm.get_regime_history(macro_df)
        assert isinstance(hist, pd.DataFrame)
        assert "valid_time" in hist.columns
        assert "regime" in hist.columns
        assert len(hist) == len(macro_df)

    def test_regime_history_prob_columns(self, macro_df, fitted_hmm):
        """Probability columns must exist and be in [0, 1]."""
        hist = fitted_hmm.get_regime_history(macro_df)
        prob_cols = [c for c in hist.columns if c.startswith("prob_")]
        assert len(prob_cols) == fitted_hmm.n_states, (
            f"Expected {fitted_hmm.n_states} prob_ columns, got {len(prob_cols)}"
        )
        for col in prob_cols:
            vals = hist[col].dropna()
            assert (vals >= 0).all() and (vals <= 1.0 + 1e-9).all(), (
                f"{col} has values outside [0, 1]"
            )

    def test_hmm_metadata(self, fitted_hmm):
        meta = fitted_hmm.get_metadata()
        assert meta["name"] == "hmm_regime"
        assert meta["n_states"] == 3
        assert meta["last_trained"] is not None
        assert set(meta["state_map"].values()) == {"bull", "bear", "transition"}

    def test_fit_with_alternative_columns(self, fitted_hmm):
        """Model should work with alternative column names (vix, sp500_ret_20d)."""
        # Make a macro df with sp500_ret_20d, vix, yield_spread columns
        macro = make_synthetic_macro(n_days=200)
        from models.regime import RegimeModel
        m = RegimeModel(n_states=3)
        m._n_iter = 20
        m.fit(macro)
        regime = m.predict_regime(macro)
        assert regime in {"bull", "bear", "transition"}

    def test_regime_model_fallback_without_hmmlearn(self, macro_df, monkeypatch):
        """Il modello di regime deve funzionare anche senza hmmlearn installato."""
        import builtins
        import importlib
        import sys

        monkeypatch.delitem(sys.modules, "models.regime", raising=False)
        real_import = builtins.__import__

        def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "hmmlearn":
                raise ModuleNotFoundError("No module named 'hmmlearn'")
            return real_import(name, globals, locals, fromlist, level)

        monkeypatch.setattr("builtins.__import__", guarded_import)

        regime_module = importlib.import_module("models.regime")
        model = regime_module.RegimeModel(n_states=3)
        model._n_iter = 20
        model.fit(macro_df)

        probs = model.predict_probabilities(macro_df)
        assert set(probs) == {"bull", "bear", "transition"}
        assert abs(sum(probs.values()) - 1.0) < 1e-6


# ---------------------------------------------------------------------------
# BaseModel interface tests
# ---------------------------------------------------------------------------

class TestBaseModel:
    def test_base_model_is_abstract(self):
        """Cannot instantiate BaseModel directly."""
        from models.base import BaseModel
        with pytest.raises(TypeError):
            BaseModel()  # type: ignore[abstract]

    def test_technical_model_satisfies_interface(self, fitted_lgbm):
        from models.base import BaseModel
        assert isinstance(fitted_lgbm, BaseModel)

    def test_save_load_roundtrip(self, fitted_lgbm, synthetic_data, tmp_path):
        """Saved model should produce identical predictions after load."""
        features, _ = synthetic_data
        last_date = features["valid_time"].max()
        single_day = features.filter(pl.col("valid_time") == last_date)

        preds_before = fitted_lgbm.predict(single_day)

        path = str(tmp_path / "lgbm_test.pkl")
        fitted_lgbm.save(path)

        from models.technical import TechnicalModel
        loaded = TechnicalModel()
        loaded.load(path)
        preds_after = loaded.predict(single_day)

        pd.testing.assert_series_equal(preds_before, preds_after)
