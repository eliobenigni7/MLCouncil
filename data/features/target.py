"""Forward-return and triple-barrier target computation.

Targets use shift(-horizon) so that target[T] = label/return from T over the
forward window. These MUST NOT be included in the feature DataFrame used for
training. Keep feature and target DataFrames separate until the final join at
train time, and align on (ticker, valid_time).

Supports:
- Raw / risk-adjusted forward returns (default; ``MLCOUNCIL_TARGET_MODE=forward_return``)
- López de Prado triple-barrier labels (+kσ, -kσ, vertical barrier T_max)
  (``MLCOUNCIL_TARGET_MODE=triple_barrier``)
"""

from __future__ import annotations

import os

import numpy as np
import polars as pl

_TARGET_MODES = frozenset({"forward_return", "triple_barrier"})


def get_target_mode() -> str:
    """Active target mode from ``MLCOUNCIL_TARGET_MODE`` (default ``forward_return``)."""
    mode = os.getenv("MLCOUNCIL_TARGET_MODE", "forward_return").strip().lower()
    if mode not in _TARGET_MODES:
        return "forward_return"
    return mode


def training_rank_column(horizon: int) -> str:
    """Cross-sectional rank column name for LGBM training at the given horizon."""
    if get_target_mode() == "triple_barrier":
        return f"rank_tb_{horizon}d"
    return f"rank_fwd_{horizon}d"


def _resolve_target_mode(target_mode: str | None) -> str:
    mode = (target_mode or get_target_mode()).strip().lower()
    return mode if mode in _TARGET_MODES else "forward_return"


def _env_float(key: str, default: float) -> float:
    raw = os.getenv(key)
    if raw is None or not raw.strip():
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def compute_targets(
    ohlcv_df: pl.DataFrame,
    horizons: list[int] | None = None,
    risk_adjusted: bool = True,
    vol_window: int = 21,
    target_mode: str | None = None,
) -> pl.DataFrame:
    """Compute training targets (forward return or triple-barrier).

    Parameters
    ----------
    ohlcv_df:
        OHLCV DataFrame with ticker, valid_time, adj_close (and high/low for
        triple-barrier).
    horizons:
        Forward horizons in days (vertical barrier for triple-barrier).
    risk_adjusted:
        Forward-return mode only: volatility-scaled targets.
    vol_window:
        Rolling window for volatility (days).
    target_mode:
        Override ``MLCOUNCIL_TARGET_MODE`` (``forward_return`` | ``triple_barrier``).

    Returns
    -------
    pl.DataFrame
        Forward-return mode: ret_fwd_*, rank_fwd_*, optional risk_adj_fwd_*.
        Triple-barrier mode: tb_label_*, tb_ret_*, rank_tb_*.
    """
    mode = _resolve_target_mode(target_mode)
    if mode == "triple_barrier":
        return compute_triple_barrier_targets(
            ohlcv_df,
            horizons=horizons,
            vol_window=vol_window,
        )
    return _compute_forward_return_targets(
        ohlcv_df,
        horizons=horizons,
        risk_adjusted=risk_adjusted,
        vol_window=vol_window,
    )


def compute_triple_barrier_targets(
    ohlcv_df: pl.DataFrame,
    horizons: list[int] | None = None,
    k: float | None = None,
    vol_window: int | None = None,
) -> pl.DataFrame:
    """Triple-barrier labels (+kσ upper, -kσ lower, vertical barrier T_max).

    Labels: +1 (upper touched first), -1 (lower), 0 (vertical / timeout).
    ``rank_tb_{h}d`` is the cross-sectional percentile rank of ``tb_label_{h}d``.
    """
    if horizons is None:
        horizons = [1, 5]
    k_mult = k if k is not None else _env_float("MLCOUNCIL_TB_K", 2.0)
    vwin = vol_window if vol_window is not None else int(_env_float("MLCOUNCIL_TB_VOL_WINDOW", 21))

    df = ohlcv_df.sort(["ticker", "valid_time"])

    def _per_ticker(ticker_df: pl.DataFrame) -> pl.DataFrame:
        ticker_df = ticker_df.sort("valid_time")
        closes = ticker_df["adj_close"].to_numpy()
        highs = ticker_df["high"].to_numpy() if "high" in ticker_df.columns else closes
        lows = ticker_df["low"].to_numpy() if "low" in ticker_df.columns else closes
        dates = ticker_df["valid_time"].to_list()
        n = len(closes)

        daily_ret = np.empty(n, dtype=float)
        daily_ret[0] = np.nan
        daily_ret[1:] = closes[1:] / closes[:-1] - 1.0

        rolling_vol = np.full(n, np.nan)
        for i in range(vwin, n):
            window = daily_ret[i - vwin + 1 : i + 1]
            valid = window[~np.isnan(window)]
            if len(valid) >= max(5, vwin // 2):
                rolling_vol[i] = float(np.std(valid, ddof=1))

        out: dict[str, list] = {f"tb_label_{h}d": [None] * n for h in horizons}
        out.update({f"tb_ret_{h}d": [None] * n for h in horizons})

        for i in range(n):
            vol = rolling_vol[i]
            if vol is None or np.isnan(vol) or vol < 1e-8:
                continue
            start = closes[i]
            if start <= 0:
                continue
            upper = start * (1.0 + k_mult * vol)
            lower = start * (1.0 - k_mult * vol)

            for h in horizons:
                if i + h >= n:
                    continue
                end_date = dates[i + h]
                gap = end_date - dates[i]
                if hasattr(gap, "days") and gap.days > 7 * h:
                    continue

                label: int | None = None
                exit_ret: float | None = None
                for j in range(1, h + 1):
                    if highs[i + j] >= upper:
                        label = 1
                        exit_ret = closes[i + j] / start - 1.0
                        break
                    if lows[i + j] <= lower:
                        label = -1
                        exit_ret = closes[i + j] / start - 1.0
                        break

                if label is None:
                    exit_ret = closes[i + h] / start - 1.0
                    label = 0 if abs(exit_ret) < 1e-6 else (1 if exit_ret > 0 else -1)

                out[f"tb_label_{h}d"][i] = float(label)
                out[f"tb_ret_{h}d"][i] = float(exit_ret)

        exprs = [
            pl.Series(name, values, dtype=pl.Float64)
            for name, values in out.items()
        ]
        return ticker_df.with_columns(exprs)

    df = df.group_by("ticker", maintain_order=True).map_groups(_per_ticker)

    for h in horizons:
        label_col = f"tb_label_{h}d"
        rank_col = f"rank_tb_{h}d"
        df = df.with_columns(
            pl.col(label_col)
            .rank(method="average")
            .over("valid_time")
            .alias(f"_rank_raw_tb_{h}d")
        ).with_columns(
            (
                (pl.col(f"_rank_raw_tb_{h}d") - 1.0)
                / (pl.col(f"_rank_raw_tb_{h}d").count().over("valid_time") - 1.0 + 1e-10)
            ).alias(rank_col)
        ).drop(f"_rank_raw_tb_{h}d")

    keep = ["ticker", "valid_time"]
    for h in horizons:
        keep.extend([f"tb_label_{h}d", f"tb_ret_{h}d", f"rank_tb_{h}d"])
    available = [c for c in keep if c in df.columns]
    return df.select(available)


def _compute_forward_return_targets(
    ohlcv_df: pl.DataFrame,
    horizons: list[int] | None = None,
    risk_adjusted: bool = True,
    vol_window: int = 21,
) -> pl.DataFrame:
    """Compute forward returns and their cross-sectional ranks."""
    if horizons is None:
        horizons = [1, 5]

    df = ohlcv_df.sort(["ticker", "valid_time"])

    def _compute_returns(ticker_df: pl.DataFrame) -> pl.DataFrame:
        c = pl.col("adj_close")
        exprs = []

        for h in horizons:
            fwd_price = c.shift(-h)
            raw_fwd_ret = fwd_price / c - 1.0

            max_gap = pl.duration(days=7 * h)
            valid_gap = (pl.col("valid_time").shift(-h) - pl.col("valid_time")) <= max_gap
            exprs.append(
                pl.when(valid_gap)
                .then(raw_fwd_ret)
                .otherwise(None)
                .alias(f"ret_fwd_{h}d")
            )

        if risk_adjusted:
            ticker_df = (
                ticker_df.sort("valid_time")
                .with_columns(
                    pl.col("adj_close").pct_change().alias("daily_ret")
                )
                .with_columns(
                    (pl.col("daily_ret") ** 2)
                    .rolling_mean(window_size=vol_window, min_samples=vol_window)
                    .alias("ret_sq_rolling"),
                    pl.col("daily_ret")
                    .shift(1)
                    .rolling_mean(window_size=vol_window, min_samples=vol_window)
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
                    (fwd_ret / (pl.col("rolling_vol") + 1e-8)).alias(f"_risk_adj_raw_{h}d")
                )

        cols_to_drop = ["daily_ret", "ret_sq_rolling", "ret_rolling", "rolling_var", "rolling_vol"]
        existing_cols = [col for col in cols_to_drop if col in ticker_df.columns]
        ticker_df = ticker_df.with_columns(exprs).drop(existing_cols)

        if risk_adjusted:
            winsorize_exprs = []
            raw_aliases = []
            for h in horizons:
                raw_alias = f"_risk_adj_raw_{h}d"
                if raw_alias not in ticker_df.columns:
                    continue
                col_data = ticker_df[raw_alias].drop_nulls()
                if len(col_data) < 10:
                    q01, q99 = -5.0, 5.0
                else:
                    q01 = float(col_data.quantile(0.01) or -5.0)
                    q99 = float(col_data.quantile(0.99) or 5.0)
                winsorize_exprs.append(
                    pl.col(raw_alias).clip(q01, q99).alias(f"risk_adj_fwd_{h}d")
                )
                raw_aliases.append(raw_alias)
            if winsorize_exprs:
                ticker_df = ticker_df.with_columns(winsorize_exprs).drop(raw_aliases)

        return ticker_df

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
