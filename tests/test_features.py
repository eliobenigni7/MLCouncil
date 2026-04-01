"""Tests for data/features — Alpha158 and target computation.

Coverage
--------
1. No look-ahead bias: feature at T uses only prices ≤ T-1
2. No NaN in feature rows after warm-up is dropped
3. Cross-sectional rank percentiles are bounded in [0, 1]
4. Cross-sectional ranks are monotonically consistent with input values
5. Target forward returns use future prices (shift(-n))
6. Output shape is correct for different universe sizes
7. Macro features join correctly when macro_df is provided
"""

from __future__ import annotations

from datetime import date, timedelta

import polars as pl
import pytest


# ---------------------------------------------------------------------------
# Synthetic data helpers
# ---------------------------------------------------------------------------

def _make_ohlcv(
    tickers: list[str],
    n_days: int = 120,
    start: date = date(2023, 1, 2),
    seed: int = 42,
) -> pl.DataFrame:
    """Generate synthetic OHLCV data for multiple tickers."""
    import random
    rng = random.Random(seed)

    rows = []
    for ticker in tickers:
        price = 100.0
        vol = 1_000_000
        for i in range(n_days):
            d = start + timedelta(days=i)
            # Skip weekends
            if d.weekday() >= 5:
                continue
            ret = rng.gauss(0.0002, 0.015)
            price = max(price * (1 + ret), 1.0)
            high = price * (1 + abs(rng.gauss(0, 0.005)))
            low = price * (1 - abs(rng.gauss(0, 0.005)))
            vol_day = int(vol * rng.uniform(0.5, 1.5))
            rows.append({
                "ticker": ticker,
                "valid_time": d,
                "transaction_time": None,
                "open": price * (1 + rng.gauss(0, 0.001)),
                "high": high,
                "low": low,
                "close": price,
                "adj_close": price,
                "volume": vol_day,
            })

    df = pl.DataFrame(rows).with_columns([
        pl.col("valid_time").cast(pl.Date),
        pl.col("transaction_time").cast(pl.Datetime("us", "UTC")),
        pl.col("open").cast(pl.Float64),
        pl.col("high").cast(pl.Float64),
        pl.col("low").cast(pl.Float64),
        pl.col("close").cast(pl.Float64),
        pl.col("adj_close").cast(pl.Float64),
        pl.col("volume").cast(pl.Int64),
    ])
    return df.sort(["ticker", "valid_time"])


def _make_macro(n_days: int = 120, start: date = date(2023, 1, 2)) -> pl.DataFrame:
    """Generate synthetic macro context DataFrame."""
    import random
    rng = random.Random(0)

    rows = []
    vix = 20.0
    spread = 0.5
    for i in range(n_days):
        d = start + timedelta(days=i)
        if d.weekday() >= 5:
            continue
        vix = max(vix + rng.gauss(0, 0.5), 10.0)
        spread = spread + rng.gauss(0, 0.02)
        rows.append({
            "valid_time": d,
            "vix": vix,
            "vix_chg": rng.gauss(0, 0.02),
            "yield_spread": spread,
            "yield_spread_chg": rng.gauss(0, 0.01),
            "sp500_ret_5d": rng.gauss(0.001, 0.02),
            "sp500_ret_20d": rng.gauss(0.004, 0.04),
        })

    return pl.DataFrame(rows).with_columns(pl.col("valid_time").cast(pl.Date))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "META"]


@pytest.fixture(scope="module")
def ohlcv_df():
    return _make_ohlcv(TICKERS, n_days=120)


@pytest.fixture(scope="module")
def macro_df():
    return _make_macro(n_days=120)


@pytest.fixture(scope="module")
def features_df(ohlcv_df, macro_df):
    from data.features.alpha158 import compute_alpha158
    return compute_alpha158(ohlcv_df, macro_df)


@pytest.fixture(scope="module")
def targets_df(ohlcv_df):
    from data.features.target import compute_targets
    return compute_targets(ohlcv_df, horizons=[1, 5])


# ---------------------------------------------------------------------------
# Alpha158 tests
# ---------------------------------------------------------------------------

class TestAlpha158:
    def test_output_is_dataframe(self, features_df):
        assert isinstance(features_df, pl.DataFrame)
        assert not features_df.is_empty()

    def test_required_columns_present(self, features_df):
        required = {"ticker", "valid_time", "ret_1d", "ret_5d", "vol_5d", "rsi_14"}
        missing = required - set(features_df.columns)
        assert not missing, f"Missing columns: {missing}"

    def test_no_raw_ohlcv_columns(self, features_df):
        forbidden = {"open", "high", "low", "close", "adj_close", "volume"}
        leaked = forbidden & set(features_df.columns)
        assert not leaked, f"Raw OHLCV columns leaked into features: {leaked}"

    def test_no_nan_after_warmup(self, features_df):
        """After dropna (warm-up rows removed), key features should have no NaN."""
        for col in ("ret_1d", "vol_5d", "close_ma5_ratio"):
            if col in features_df.columns:
                null_count = features_df[col].null_count()
                assert null_count == 0, f"{col} has {null_count} NaN rows after warm-up drop"

    def test_no_lookahead_ret1d(self, ohlcv_df):
        """ret_1d at day T must equal adj_close[T-2]/adj_close[T-3] - 1, not use T's price.

        Method: for a single ticker, verify that ret_1d[i] = close[i-1]/close[i-2] - 1.
        """
        from data.features.alpha158 import compute_alpha158

        single = ohlcv_df.filter(pl.col("ticker") == "AAPL")
        feats = compute_alpha158(single, None)

        # Sort both by valid_time
        feats_sorted = feats.sort("valid_time")
        raw_sorted = single.sort("valid_time")

        # Align on common valid_time
        joined = feats_sorted.join(
            raw_sorted.select(["valid_time", "adj_close"]),
            on="valid_time",
            how="inner",
        )

        # ret_1d[i] should equal adj_close_raw[i-1] / adj_close_raw[i-2] - 1
        # i.e., feature[T] uses close[T-1] and close[T-2]
        closes = raw_sorted["adj_close"].to_list()
        valid_times = raw_sorted["valid_time"].to_list()

        feat_map = dict(zip(joined["valid_time"].to_list(), joined["ret_1d"].to_list()))

        errors = []
        for i in range(2, len(closes)):
            t = valid_times[i]
            if t not in feat_map or feat_map[t] is None:
                continue
            expected = closes[i - 1] / closes[i - 2] - 1.0
            actual = feat_map[t]
            if abs(actual - expected) > 1e-9:
                errors.append(f"T={t}: expected {expected:.6f}, got {actual:.6f}")

        assert not errors, f"Look-ahead bias detected in ret_1d:\n" + "\n".join(errors[:5])

    def test_cross_sectional_rank_bounded(self, features_df):
        """Cross-sectional rank percentiles must be in [0, 1]."""
        for col in ("cs_ret_rank", "cs_vol_rank", "cs_volume_rank"):
            if col not in features_df.columns:
                continue
            col_data = features_df[col].drop_nulls()
            assert col_data.min() >= -1e-9, f"{col} has values below 0"
            assert col_data.max() <= 1.0 + 1e-9, f"{col} has values above 1"

    def test_cross_sectional_rank_monotone(self, features_df):
        """On each date, higher ret_1d should produce higher cs_ret_rank."""
        if "cs_ret_rank" not in features_df.columns:
            pytest.skip("cs_ret_rank not available")

        sample_date = features_df["valid_time"].max()
        day = features_df.filter(pl.col("valid_time") == sample_date).drop_nulls(
            subset=["ret_1d", "cs_ret_rank"]
        )
        if len(day) < 2:
            pytest.skip("Not enough tickers for rank test")

        day_sorted = day.sort("ret_1d")
        rank_sorted = day.sort("cs_ret_rank")
        # Tickers sorted by ret_1d and by cs_ret_rank should be in same order
        assert day_sorted["ticker"].to_list() == rank_sorted["ticker"].to_list()

    def test_shape_single_ticker(self, ohlcv_df):
        from data.features.alpha158 import compute_alpha158

        single = ohlcv_df.filter(pl.col("ticker") == "AAPL")
        feats = compute_alpha158(single, None)
        assert feats["ticker"].unique().to_list() == ["AAPL"]
        assert len(feats) > 0

    def test_shape_full_universe(self, features_df):
        tickers_in = set(features_df["ticker"].unique().to_list())
        assert tickers_in == set(TICKERS)

    def test_macro_columns_present_when_provided(self, features_df):
        macro_cols = {"vix", "yield_spread", "sp500_ret_5d"}
        present = macro_cols & set(features_df.columns)
        assert present, "No macro columns found even though macro_df was passed"

    def test_macro_columns_absent_without_macro(self, ohlcv_df):
        from data.features.alpha158 import compute_alpha158

        feats = compute_alpha158(ohlcv_df, None)
        for col in ("vix", "yield_spread", "sp500_ret_5d"):
            assert col not in feats.columns, f"{col} present without macro_df"


# ---------------------------------------------------------------------------
# Target tests
# ---------------------------------------------------------------------------

class TestTargets:
    def test_output_columns(self, targets_df):
        expected = {"ticker", "valid_time", "ret_fwd_1d", "ret_fwd_5d", "rank_fwd_1d", "rank_fwd_5d"}
        missing = expected - set(targets_df.columns)
        assert not missing, f"Missing target columns: {missing}"

    def test_forward_return_uses_future_price(self, ohlcv_df):
        """ret_fwd_1d[T] must equal adj_close[T+1] / adj_close[T] - 1."""
        from data.features.target import compute_targets

        single = ohlcv_df.filter(pl.col("ticker") == "AAPL").sort("valid_time")
        targets = compute_targets(single, horizons=[1])

        raw_prices = dict(zip(single["valid_time"].to_list(), single["adj_close"].to_list()))
        target_map = {
            row["valid_time"]: row["ret_fwd_1d"]
            for row in targets.to_dicts()
            if row["ret_fwd_1d"] is not None
        }

        dates = sorted(raw_prices.keys())
        errors = []
        for i, d in enumerate(dates[:-1]):
            if d not in target_map:
                continue
            expected = raw_prices[dates[i + 1]] / raw_prices[d] - 1.0
            actual = target_map[d]
            if abs(actual - expected) > 1e-9:
                errors.append(f"T={d}: expected {expected:.6f}, got {actual:.6f}")

        assert not errors, "Forward return look-forward check failed:\n" + "\n".join(errors[:5])

    def test_rank_bounded(self, targets_df):
        for col in ("rank_fwd_1d", "rank_fwd_5d"):
            data = targets_df[col].drop_nulls()
            assert data.min() >= -1e-9
            assert data.max() <= 1.0 + 1e-9

    def test_no_raw_ohlcv_in_targets(self, targets_df):
        forbidden = {"open", "high", "low", "close", "adj_close", "volume"}
        leaked = forbidden & set(targets_df.columns)
        assert not leaked, f"OHLCV columns leaked into targets: {leaked}"

    def test_last_rows_are_nan(self, ohlcv_df):
        """The last 5 rows for each ticker should have NaN for ret_fwd_5d."""
        from data.features.target import compute_targets

        single = ohlcv_df.filter(pl.col("ticker") == "AAPL").sort("valid_time")
        targets = compute_targets(single, horizons=[5])

        last_5 = targets.tail(5)
        null_count = last_5["ret_fwd_5d"].null_count()
        assert null_count > 0, "Expected NaN in last rows for 5d forward return"

    def test_custom_horizons(self, ohlcv_df):
        from data.features.target import compute_targets

        t = compute_targets(ohlcv_df, horizons=[2, 10])
        assert "ret_fwd_2d" in t.columns
        assert "ret_fwd_10d" in t.columns
        assert "ret_fwd_1d" not in t.columns

    def test_features_and_targets_stay_separate(self, features_df, targets_df):
        """Ensure no forward-return columns leaked into the feature DataFrame."""
        for col in targets_df.columns:
            if col in ("ticker", "valid_time"):
                continue
            assert col not in features_df.columns, (
                f"Target column '{col}' found in features_df — potential training leak"
            )
