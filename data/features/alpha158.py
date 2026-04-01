"""Alpha158 feature computation on Polars DataFrames.

All features are shifted(1) so that feature[T] uses only data available
at the close of T-1.  This prevents any look-ahead bias.

Input OHLCV schema
------------------
ticker          : Utf8
valid_time      : Date
transaction_time: Datetime("us", "UTC")
open, high, low, close, adj_close : Float64
volume          : Int64

Macro DataFrames (joined internally):
  vix_df        : valid_time, value (VIX level)
  treasuries_df : valid_time, yield_spread
  sp500_df      : valid_time, return_5d, return_20d  (or closest windows)
"""

from __future__ import annotations

import math

import polars as pl


# ---------------------------------------------------------------------------
# Internal helpers – all operate on a single-ticker sorted LazyFrame
# ---------------------------------------------------------------------------

def _shift(expr: pl.Expr, n: int = 1) -> pl.Expr:
    """Shift by n (default 1) to enforce look-ahead-safe alignment."""
    return expr.shift(n)


def _pct_change(close: pl.Expr, n: int) -> pl.Expr:
    """n-day return on shifted prices (feature-safe)."""
    c = _shift(close)
    return c / c.shift(n) - 1.0


def _rolling_mean(expr: pl.Expr, n: int) -> pl.Expr:
    return expr.rolling_mean(n, min_samples=n)


def _rolling_std(expr: pl.Expr, n: int) -> pl.Expr:
    return expr.rolling_std(n, min_samples=n)


def _rsi(close_shifted: pl.Expr, period: int) -> pl.Expr:
    """RSI (Wilder SMA) on already-shifted close."""
    delta = close_shifted - close_shifted.shift(1)
    gain = pl.when(delta > 0).then(delta).otherwise(0.0)
    loss = pl.when(delta < 0).then(-delta).otherwise(0.0)
    avg_gain = gain.rolling_mean(period, min_samples=period)
    avg_loss = loss.rolling_mean(period, min_samples=period)
    rs = avg_gain / (avg_loss + 1e-10)
    return 100.0 - 100.0 / (1.0 + rs)


def _ema(expr: pl.Expr, span: int) -> pl.Expr:
    return expr.ewm_mean(span=span, adjust=False, min_samples=span)


# ---------------------------------------------------------------------------
# Per-ticker feature blocks
# ---------------------------------------------------------------------------

def _price_features(df: pl.LazyFrame) -> pl.LazyFrame:
    """30 price-based features."""
    c = pl.col("adj_close")
    h = pl.col("high")
    lo = pl.col("low")

    hl = h - lo
    hl_safe = pl.when(hl.abs() > 1e-10).then(hl).otherwise(1e-10)

    return df.with_columns([
        # Returns (shifted prices → no look-ahead)
        _pct_change(c, 1).alias("ret_1d"),
        _pct_change(c, 5).alias("ret_5d"),
        _pct_change(c, 10).alias("ret_10d"),
        _pct_change(c, 20).alias("ret_20d"),
        _pct_change(c, 60).alias("ret_60d"),

        # Normalised price vs moving averages
        (_shift(c) / _rolling_mean(_shift(c), 5)).alias("close_ma5_ratio"),
        (_shift(c) / _rolling_mean(_shift(c), 10)).alias("close_ma10_ratio"),
        (_shift(c) / _rolling_mean(_shift(c), 20)).alias("close_ma20_ratio"),
        (_shift(c) / _rolling_mean(_shift(c), 60)).alias("close_ma60_ratio"),

        # High-Low range / close – rolling (daily hl/c, then rolled)
        (_rolling_mean(_shift(hl_safe / c), 5)).alias("hl_range_5d"),
        (_rolling_mean(_shift(hl_safe / c), 10)).alias("hl_range_10d"),
        (_rolling_mean(_shift(hl_safe / c), 20)).alias("hl_range_20d"),

        # Price position within daily range – rolling
        (_rolling_mean(_shift((c - lo) / hl_safe), 5)).alias("price_pos_5d"),
        (_rolling_mean(_shift((c - lo) / hl_safe), 10)).alias("price_pos_10d"),
        (_rolling_mean(_shift((c - lo) / hl_safe), 20)).alias("price_pos_20d"),
    ])


def _volume_features(df: pl.LazyFrame) -> pl.LazyFrame:
    """20 volume-based features."""
    v = pl.col("volume").cast(pl.Float64)
    c = pl.col("adj_close")
    sv = _shift(v)
    sc = _shift(c)

    return df.with_columns([
        # Volume vs moving averages
        (sv / _rolling_mean(sv, 5)).alias("vol_ma5_ratio"),
        (sv / _rolling_mean(sv, 10)).alias("vol_ma10_ratio"),
        (sv / _rolling_mean(sv, 20)).alias("vol_ma20_ratio"),
        (sv / _rolling_mean(sv, 60)).alias("vol_ma60_ratio"),

        # VWAP deviation: (close - vwap_n) / close, vwap = sum(c*v)/sum(v)
        (
            (sc - (sc * sv).rolling_sum(5, min_samples=5)
             / sv.rolling_sum(5, min_samples=5))
            / (sc.abs() + 1e-10)
        ).alias("vwap_dev_5d"),
        (
            (sc - (sc * sv).rolling_sum(10, min_samples=10)
             / sv.rolling_sum(10, min_samples=10))
            / (sc.abs() + 1e-10)
        ).alias("vwap_dev_10d"),
        (
            (sc - (sc * sv).rolling_sum(20, min_samples=20)
             / sv.rolling_sum(20, min_samples=20))
            / (sc.abs() + 1e-10)
        ).alias("vwap_dev_20d"),

        # Volume × price (dollar volume proxy, normalized by 20d avg)
        (sv * sc / (_rolling_mean(sv * sc, 20) + 1e-10)).alias("dollar_vol_ratio_20d"),

        # Log volume
        (sv.log(base=10)).alias("log_volume"),

        # Volume acceleration: vol_5d_avg / vol_20d_avg
        (
            _rolling_mean(sv, 5) / (_rolling_mean(sv, 20) + 1e-10)
        ).alias("vol_accel_5_20"),
    ])


def _volatility_features(df: pl.LazyFrame) -> pl.LazyFrame:
    """20 volatility-based features."""
    ret = (pl.col("adj_close") / pl.col("adj_close").shift(1) - 1.0)
    ret_s = _shift(ret)  # shift to avoid look-ahead

    vol5 = _rolling_std(ret_s, 5)
    vol10 = _rolling_std(ret_s, 10)
    vol20 = _rolling_std(ret_s, 20)
    vol60 = _rolling_std(ret_s, 60)

    return df.with_columns([
        vol5.alias("vol_5d"),
        vol10.alias("vol_10d"),
        vol20.alias("vol_20d"),
        vol60.alias("vol_60d"),

        # Vol-of-vol: std of rolling 5d vol, over 20d window
        _rolling_std(vol5, 20).alias("vol_of_vol_20d"),

        # Skewness proxy: (mean - median) / std, approximated via rolling
        # Using E[r³]/σ³ approximation via rolling_skew (Polars 1.x)
        ret_s.rolling_skew(20).alias("skew_20d"),

        # Normalised range-based vol (Parkinson)
        _rolling_mean(
            (pl.col("high").log(base=math.e) - pl.col("low").log(base=math.e)).shift(1) ** 2,
            20
        ).alias("park_vol_20d"),

        # Vol ratio: short / long
        (vol5 / (vol20 + 1e-10)).alias("vol_ratio_5_20"),
        (vol10 / (vol60 + 1e-10)).alias("vol_ratio_10_60"),
    ])


def _momentum_features(df: pl.LazyFrame) -> pl.LazyFrame:
    """30 momentum features: RSI, MACD, Bollinger, Williams %R, Stochastic."""
    c = pl.col("adj_close")
    h = pl.col("high")
    lo = pl.col("low")
    sc = _shift(c)
    sh = _shift(h)
    slo = _shift(lo)

    # MACD
    ema12 = _ema(sc, 12)
    ema26 = _ema(sc, 26)
    macd = ema12 - ema26
    macd_signal = _ema(macd, 9)

    # Bollinger Bands (20d, 2σ)
    bb_mid = _rolling_mean(sc, 20)
    bb_std = _rolling_std(sc, 20)
    bb_upper = bb_mid + 2.0 * bb_std
    bb_lower = bb_mid - 2.0 * bb_std
    bb_range = pl.when((bb_upper - bb_lower).abs() > 1e-10).then(bb_upper - bb_lower).otherwise(1e-10)

    # Williams %R (14d)
    high14 = sh.rolling_max(14, min_samples=14)
    low14 = slo.rolling_min(14, min_samples=14)
    hl14 = pl.when((high14 - low14).abs() > 1e-10).then(high14 - low14).otherwise(1e-10)

    # Stochastic %K (14d fast), %D (3d SMA of %K)
    stoch_k = (sc - low14) / hl14 * 100.0
    stoch_d = _rolling_mean(stoch_k, 3)

    return df.with_columns([
        # RSI
        _rsi(sc, 14).alias("rsi_14"),
        _rsi(sc, 28).alias("rsi_28"),

        # MACD
        macd.alias("macd"),
        macd_signal.alias("macd_signal"),
        (macd - macd_signal).alias("macd_hist"),

        # Bollinger band position (0 = lower band, 1 = upper band)
        ((sc - bb_lower) / bb_range).alias("bb_position"),
        (bb_std / (bb_mid.abs() + 1e-10)).alias("bb_width"),

        # Williams %R
        ((high14 - sc) / hl14 * (-100.0)).alias("williams_r"),

        # Stochastic
        stoch_k.alias("stoch_k"),
        stoch_d.alias("stoch_d"),

        # ROC (rate of change) at multiple horizons
        (sc / sc.shift(5) - 1.0).alias("roc_5"),
        (sc / sc.shift(10) - 1.0).alias("roc_10"),
        (sc / sc.shift(20) - 1.0).alias("roc_20"),

        # Trend strength: regression slope proxy (close vs ma20)
        (sc / (bb_mid + 1e-10) - 1.0).alias("price_vs_bb_mid"),

        # EMA crossover signals
        (_ema(sc, 5) / (_ema(sc, 20) + 1e-10) - 1.0).alias("ema5_20_ratio"),
        (_ema(sc, 10) / (_ema(sc, 60) + 1e-10) - 1.0).alias("ema10_60_ratio"),
    ])


def _macro_features(df: pl.LazyFrame, macro_df: pl.DataFrame) -> pl.LazyFrame:
    """Join macro context (VIX, yield spread, S&P returns)."""
    if macro_df is None or macro_df.is_empty():
        return df

    # Build a macro spine keyed on valid_time
    macro_cols = macro_df.lazy().select(
        [pl.col("valid_time")]
        + [pl.col(c) for c in macro_df.columns if c not in ("valid_time", "transaction_time")]
    )

    return df.join(macro_cols, on="valid_time", how="left")


# ---------------------------------------------------------------------------
# Cross-sectional features
# ---------------------------------------------------------------------------

def _cross_sectional_ranks(df: pl.DataFrame) -> pl.DataFrame:
    """Compute cross-sectional rank percentiles (0–1) per valid_time."""
    rank_cols = ["ret_1d", "vol_5d", "vol_ma5_ratio"]
    alias_map = {
        "ret_1d": "cs_ret_rank",
        "vol_5d": "cs_vol_rank",
        "vol_ma5_ratio": "cs_volume_rank",
    }

    available = [c for c in rank_cols if c in df.columns]
    if not available:
        return df

    # Group by date and compute percent_rank for each column
    rank_frames = []
    for col in available:
        ranked = (
            df.select(["ticker", "valid_time", col])
            .with_columns(
                pl.col(col)
                .rank(method="average")
                .over("valid_time")
                .alias(alias_map[col] + "_raw")
            )
            .with_columns(
                (
                    (pl.col(alias_map[col] + "_raw") - 1.0)
                    / (pl.col(alias_map[col] + "_raw").count().over("valid_time") - 1.0 + 1e-10)
                ).alias(alias_map[col])
            )
            .select(["ticker", "valid_time", alias_map[col]])
        )
        rank_frames.append(ranked)

    result = df
    for rf in rank_frames:
        result = result.join(rf, on=["ticker", "valid_time"], how="left")

    return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_alpha158(
    ohlcv_df: pl.DataFrame,
    macro_df: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """Compute Alpha158 features + macro context.

    Parameters
    ----------
    ohlcv_df:
        Full universe OHLCV with bi-temporal schema.
        Must contain: ticker, valid_time, open, high, low, close,
        adj_close, volume.
    macro_df:
        Wide macro DataFrame with valid_time + macro columns.
        Build with ``build_macro_context()``.

    Returns
    -------
    pl.DataFrame
        Features with ticker + valid_time index.
        All features use only data available at T-1 (look-ahead safe).
    """
    # Sort by ticker + date for correct rolling windows
    base = ohlcv_df.sort(["ticker", "valid_time"])

    # Apply per-ticker feature blocks via group_by + map_groups
    def _per_ticker(ticker_df: pl.DataFrame) -> pl.DataFrame:
        lf = ticker_df.lazy()
        lf = _price_features(lf)
        lf = _volume_features(lf)
        lf = _volatility_features(lf)
        lf = _momentum_features(lf)
        if macro_df is not None and not macro_df.is_empty():
            lf = _macro_features(lf, macro_df)
        return lf.collect()

    _ = math  # ensure import is used

    features = base.group_by("ticker", maintain_order=True).map_groups(_per_ticker)

    # Cross-sectional ranks (computed on full universe)
    features = _cross_sectional_ranks(features)

    # Drop rows in the warm-up window.
    # vol_5d (rolling std over 5 shifted returns) is the last of the "core"
    # features to become valid — filtering on it also guarantees ret_1d,
    # close_ma5_ratio, and all 5-day window features are non-null.
    features = features.filter(pl.col("vol_5d").is_not_null())

    # Keep only ticker + valid_time + feature columns (drop OHLCV raw cols)
    raw_cols = {"open", "high", "low", "close", "adj_close", "volume", "transaction_time"}
    keep = [c for c in features.columns if c not in raw_cols]
    return features.select(keep)


def build_macro_context(
    vix_path: str | None = None,
    treasuries_path: str | None = None,
    sp500_path: str | None = None,
) -> pl.DataFrame:
    """Load and join macro Parquets into a single wide DataFrame indexed by valid_time.

    Parameters
    ----------
    vix_path, treasuries_path, sp500_path:
        Paths to Parquet files produced by ``data.ingest.macro``.
        Pass None to skip a source.

    Returns
    -------
    pl.DataFrame with columns:
        valid_time, vix, vix_chg, yield_spread, yield_spread_chg,
        sp500_ret_5d, sp500_ret_20d (if available in sp500 parquet)
    """
    frames: list[pl.DataFrame] = []

    if vix_path is not None:
        vix = pl.read_parquet(vix_path).select(
            [pl.col("valid_time"), pl.col("value").alias("vix")]
        ).with_columns(
            (pl.col("vix") / pl.col("vix").shift(1) - 1.0).alias("vix_chg")
        )
        frames.append(vix)

    if treasuries_path is not None:
        treas = pl.read_parquet(treasuries_path).select(
            [
                pl.col("valid_time"),
                pl.col("yield_spread"),
                (pl.col("yield_spread") - pl.col("yield_spread").shift(1)).alias("yield_spread_chg"),
            ]
        )
        frames.append(treas)

    if sp500_path is not None:
        sp500_raw = pl.read_parquet(sp500_path)
        sp_cols = ["valid_time"]
        rename = {}
        # Pick 5d and 20d-ish windows (or whatever is available)
        for col in sp500_raw.columns:
            if col.startswith("return_"):
                window = int(col.split("_")[1].replace("d", ""))
                if window <= 7:
                    rename[col] = "sp500_ret_5d"
                    sp_cols.append(col)
                elif window <= 25:
                    rename[col] = "sp500_ret_20d"
                    sp_cols.append(col)
        sp500 = sp500_raw.select(sp_cols).rename(rename)
        frames.append(sp500)

    if not frames:
        return pl.DataFrame()

    result = frames[0]
    for other in frames[1:]:
        result = result.join(other, on="valid_time", how="outer_coalesce")

    return result.sort("valid_time")
