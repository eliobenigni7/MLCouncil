"""Forward-return target computation.

Targets use shift(-horizon) so that target[T] = return from T to T+horizon.
These MUST NOT be included in the feature DataFrame used for training.
Keep feature and target DataFrames separate until the final join at train time,
and align on (ticker, valid_time).

Supports both raw forward returns and risk-adjusted (volatility-scaled) targets.
Risk-adjusted targets divide forward return by rolling volatility, producing
more stable signals with better IC persistence (Gu, Kelly, Xiu approach).
"""

from __future__ import annotations

import numpy as np
import polars as pl


def compute_targets(
    ohlcv_df: pl.DataFrame,
    horizons: list[int] | None = None,
    risk_adjusted: bool = False,
    vol_window: int = 21,
) -> pl.DataFrame:
    """Compute forward returns and their cross-sectional ranks.

    Parameters
    ----------
    ohlcv_df:
        OHLCV DataFrame with ticker, valid_time, adj_close columns.
    horizons:
        List of forward-return horizons in days. Defaults to [1, 5].
    risk_adjusted:
        If True, compute risk-adjusted (volatility-scaled) targets.
        target = forward_return / rolling_volatility
        Default True - produces more stable signals.
    vol_window:
        Rolling window for volatility estimation in days. Default 21.

    Returns
    -------
    pl.DataFrame
        Columns: ticker, valid_time,
                 ret_fwd_{n}d  (raw forward return),
                 rank_fwd_{n}d (cross-sectional rank percentile 0-1),
                 risk_adj_fwd_{n}d (risk-adjusted forward return, if risk_adjusted=True).
    """
    if horizons is None:
        horizons = [1, 5]

    df = ohlcv_df.sort(["ticker", "valid_time"])

    def _compute_returns(ticker_df: pl.DataFrame) -> pl.DataFrame:
        c = pl.col("adj_close")
        exprs = []

        for h in horizons:
            fwd_price = c.shift(-h)
            exprs.append((fwd_price / c - 1.0).alias(f"ret_fwd_{h}d"))

        if risk_adjusted:
            # Compute rolling vol within this ticker's partition to avoid
            # cross-ticker contamination (pct_change across ticker boundaries).
            ticker_df = (
                ticker_df.sort("valid_time")
                .with_columns(
                    pl.col("adj_close").pct_change().alias("daily_ret")
                )
                .with_columns(
                    (pl.col("daily_ret") ** 2)
                    .rolling(index_column="valid_time", period=f"{vol_window}d", min_periods=vol_window)
                    .mean()
                    .alias("ret_sq_rolling"),
                    pl.col("daily_ret")
                    .shift(1)
                    .rolling(index_column="valid_time", period=f"{vol_window}d", min_periods=vol_window)
                    .mean()
                    .alias("ret_rolling"),
                )
                .with_columns(
                    ((pl.col("ret_sq_rolling") - pl.col("ret_rolling") ** 2).clip(1e-10))
                    .alias("rolling_var")
                )
                .with_columns(
                    (pl.col("rolling_var") ** 0.5 * np.sqrt(252)).alias("rolling_vol")
                )
            )
            for h in horizons:
                fwd_ret = c.shift(-h) / c - 1.0
                exprs.append(
                    (fwd_ret / (pl.col("rolling_vol") + 1e-8)).clip(-5, 5).alias(f"risk_adj_fwd_{h}d")
                )

        cols_to_drop = ["daily_ret", "ret_sq_rolling", "ret_rolling", "rolling_var", "rolling_vol"]
        existing_cols = [col for col in cols_to_drop if col in ticker_df.columns]
        return ticker_df.with_columns(exprs).drop(existing_cols)

    df = df.group_by("ticker", maintain_order=True).map_groups(_compute_returns)

    for h in horizons:
        col = f"ret_fwd_{h}d"
        rank_col = f"rank_fwd_{h}d"
        df = df.with_columns(
            pl.col(col)
            .rank(method="average")
            .over("valid_time")
            .alias(f"_rank_raw_{h}d")
        ).with_columns(
            (
                (pl.col(f"_rank_raw_{h}d") - 1.0)
                / (pl.col(f"_rank_raw_{h}d").count().over("valid_time") - 1.0 + 1e-10)
            ).alias(rank_col)
        ).drop(f"_rank_raw_{h}d")

    keep = ["ticker", "valid_time"] + [f"ret_fwd_{h}d" for h in horizons] + [f"rank_fwd_{h}d" for h in horizons]
    if risk_adjusted:
        keep += [f"risk_adj_fwd_{h}d" for h in horizons]

    available = [c for c in keep if c in df.columns]
    return df.select(available)
