"""Forward-return target computation.

Targets use shift(-horizon) so that target[T] = return from T to T+horizon.
These MUST NOT be included in the feature DataFrame used for training.
Keep feature and target DataFrames separate until the final join at train time,
and align on (ticker, valid_time).
"""

from __future__ import annotations

import polars as pl


def compute_targets(
    ohlcv_df: pl.DataFrame,
    horizons: list[int] | None = None,
) -> pl.DataFrame:
    """Compute forward returns and their cross-sectional ranks.

    Parameters
    ----------
    ohlcv_df:
        OHLCV DataFrame with ticker, valid_time, adj_close columns.
    horizons:
        List of forward-return horizons in days. Defaults to [1, 5].

    Returns
    -------
    pl.DataFrame
        Columns: ticker, valid_time,
                 ret_fwd_{n}d  (raw forward return),
                 rank_fwd_{n}d (cross-sectional rank percentile 0–1).

    Notes
    -----
    - Forward returns use shift(-horizon) on adj_close, so target[T] = future price / current price - 1.
    - Rows near the end of the series will be NaN for longer horizons.
    - Cross-sectional rank: for each date, rank tickers by forward return.
      Rank 0.0 = worst performer, 1.0 = best performer on that date.
    - NEVER merge targets into the feature DataFrame before model training.
    """
    if horizons is None:
        horizons = [1, 5]

    df = ohlcv_df.sort(["ticker", "valid_time"])

    # Compute raw forward returns per ticker
    def _fwd_returns(ticker_df: pl.DataFrame) -> pl.DataFrame:
        c = pl.col("adj_close")
        exprs = []
        for h in horizons:
            # shift(-h) gives price h days in the future → look-forward (correct for target)
            fwd_price = c.shift(-h)
            exprs.append(
                (fwd_price / c - 1.0).alias(f"ret_fwd_{h}d")
            )
        return ticker_df.with_columns(exprs)

    df = df.group_by("ticker", maintain_order=True).map_groups(_fwd_returns)

    # Cross-sectional rank per date for each horizon
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
    available = [c for c in keep if c in df.columns]
    return df.select(available)
