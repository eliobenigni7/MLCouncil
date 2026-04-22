"""Deterministic walk-forward diagnostics and lookahead-bias checks for model validation."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

try:
    import polars as pl
except ImportError:  # pragma: no cover
    pl = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class WalkForwardSplit:
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp


def build_walk_forward_splits(
    dates: Sequence[pd.Timestamp],
    train_window: int,
    test_window: int,
    step: int | None = None,
) -> list[WalkForwardSplit]:
    """Build deterministic rolling train/test windows over a date sequence."""
    if train_window <= 0 or test_window <= 0:
        raise ValueError("train_window and test_window must be positive")

    index = pd.DatetimeIndex(pd.to_datetime(list(dates)))
    if index.empty:
        return []

    step_size = step or test_window
    splits: list[WalkForwardSplit] = []
    start = 0
    while start + train_window + test_window <= len(index):
        train_slice = index[start : start + train_window]
        test_slice = index[start + train_window : start + train_window + test_window]
        splits.append(
            WalkForwardSplit(
                train_start=pd.Timestamp(train_slice[0]),
                train_end=pd.Timestamp(train_slice[-1]),
                test_start=pd.Timestamp(test_slice[0]),
                test_end=pd.Timestamp(test_slice[-1]),
            )
        )
        start += step_size
    return splits


def estimate_pbo(train_scores: Sequence[float], test_scores: Sequence[float]) -> float:
    """Return a deterministic PBO-style proxy from paired train/test scores.

    The proxy measures how often above-median in-sample strength is matched by
    below-median out-of-sample performance. It is intentionally simple and
    deterministic, designed for CI and promotion gating rather than for
    publication-grade statistical inference.
    """
    train = pd.Series(train_scores, dtype=float)
    test = pd.Series(test_scores, dtype=float)
    if train.empty or test.empty or len(train) != len(test):
        return 1.0

    train_median = float(train.median())
    test_median = float(test.median())
    return float(((train >= train_median) & (test <= test_median)).mean())


def summarize_walk_forward_metrics(metrics_df: pd.DataFrame) -> dict[str, float]:
    """Summarize deterministic out-of-sample diagnostics from walk-forward windows."""
    if metrics_df.empty:
        return {
            "walk_forward_window_count": 0,
            "oos_sharpe": 0.0,
            "oos_max_drawdown": 0.0,
            "oos_turnover": 0.0,
            "pbo": 1.0,
        }

    return {
        "walk_forward_window_count": int(len(metrics_df)),
        "oos_sharpe": float(metrics_df["test_sharpe"].mean()) if "test_sharpe" in metrics_df else 0.0,
        "oos_max_drawdown": float(metrics_df["test_max_drawdown"].mean()) if "test_max_drawdown" in metrics_df else 0.0,
        "oos_turnover": float(metrics_df["test_turnover"].mean()) if "test_turnover" in metrics_df else 0.0,
        "pbo": estimate_pbo(
            metrics_df["train_sharpe"] if "train_sharpe" in metrics_df else [],
            metrics_df["test_sharpe"] if "test_sharpe" in metrics_df else [],
        ),
    }


def build_purged_walk_forward_splits(
    dates: Sequence[pd.Timestamp],
    train_window: int,
    test_window: int,
    step: int | None = None,
    purge_period: int = 0,
    embargo_period: int = 0,
) -> list[WalkForwardSplit]:
    """Build rolling splits with explicit purge/embargo guards.

    ``purge_period`` removes the most recent dates from the training slice.
    ``embargo_period`` inserts a gap between train and test windows.
    """
    if train_window <= 0 or test_window <= 0:
        raise ValueError("train_window and test_window must be positive")
    if purge_period < 0 or embargo_period < 0:
        raise ValueError("purge_period and embargo_period must be non-negative")

    index = pd.DatetimeIndex(pd.to_datetime(list(dates))).sort_values().unique()
    if index.empty:
        return []

    step_size = step or test_window
    splits: list[WalkForwardSplit] = []
    start = 0

    while True:
        train_start_idx = start
        train_end_idx = start + train_window - 1
        purged_train_end_idx = train_end_idx - purge_period
        test_start_idx = train_end_idx + 1 + embargo_period
        test_end_idx = test_start_idx + test_window - 1

        if test_end_idx >= len(index):
            break

        if purged_train_end_idx >= train_start_idx:
            train_slice = index[train_start_idx : purged_train_end_idx + 1]
            test_slice = index[test_start_idx : test_end_idx + 1]
            if len(train_slice) > 0 and len(test_slice) > 0:
                splits.append(
                    WalkForwardSplit(
                        train_start=pd.Timestamp(train_slice[0]),
                        train_end=pd.Timestamp(train_slice[-1]),
                        test_start=pd.Timestamp(test_slice[0]),
                        test_end=pd.Timestamp(test_slice[-1]),
                    )
                )

        start += step_size

    return splits


def _normalize_cross_sectional_weights(signal_row: pd.Series) -> pd.Series:
    raw = signal_row.astype(float).fillna(0.0)
    centered = raw - raw.mean()
    gross = float(centered.abs().sum())
    if gross > 1e-12:
        return centered / gross

    gross_raw = float(raw.abs().sum())
    if gross_raw > 1e-12:
        return raw / gross_raw
    return raw * 0.0


def compute_strategy_returns(
    signals: pd.DataFrame,
    forward_returns: pd.DataFrame,
) -> pd.Series:
    """Compute a deterministic daily strategy return series."""
    if signals.empty or forward_returns.empty:
        return pd.Series(dtype=float, name="strategy_return")

    common_index = pd.DatetimeIndex(signals.index).intersection(pd.DatetimeIndex(forward_returns.index))
    common_cols = signals.columns.intersection(forward_returns.columns)
    if len(common_index) == 0 or len(common_cols) == 0:
        return pd.Series(dtype=float, name="strategy_return")

    sig = signals.loc[common_index, common_cols].astype(float)
    fwd = forward_returns.loc[common_index, common_cols].astype(float)

    rows: list[tuple[pd.Timestamp, float]] = []
    for d in common_index:
        weights = _normalize_cross_sectional_weights(sig.loc[d])
        daily_ret = float((weights * fwd.loc[d].fillna(0.0)).sum())
        rows.append((pd.Timestamp(d), daily_ret))

    if not rows:
        return pd.Series(dtype=float, name="strategy_return")

    result = pd.Series(
        [r for _, r in rows],
        index=pd.DatetimeIndex([d for d, _ in rows]),
        name="strategy_return",
        dtype=float,
    ).sort_index()
    return result


def estimate_turnover_from_signals(signals: pd.DataFrame) -> float:
    """Estimate one-way turnover from daily normalized signal weights."""
    if signals.empty or len(signals) < 2:
        return 0.0

    normalized = signals.astype(float).apply(_normalize_cross_sectional_weights, axis=1)
    if not isinstance(normalized, pd.DataFrame) or normalized.empty:
        return 0.0

    turnover = normalized.diff().abs().sum(axis=1) * 0.5
    if turnover.empty:
        return 0.0
    return float(turnover.iloc[1:].mean()) if len(turnover) > 1 else 0.0


def _annualized_sharpe(returns: pd.Series) -> float:
    clean = returns.dropna()
    if len(clean) < 2:
        return 0.0
    std = float(clean.std())
    if std < 1e-12:
        return 0.0
    return float(clean.mean() / std * np.sqrt(252))


def _max_drawdown_from_returns(returns: pd.Series) -> float:
    clean = returns.fillna(0.0)
    if clean.empty:
        return 0.0
    equity = (1.0 + clean).cumprod()
    peak = equity.cummax()
    drawdown = equity / peak - 1.0
    return float(drawdown.min())


def _cagr_from_returns(returns: pd.Series) -> float:
    clean = returns.fillna(0.0)
    if clean.empty:
        return 0.0
    years = len(clean) / 252.0
    if years <= 0:
        return 0.0
    final_equity = float((1.0 + clean).cumprod().iloc[-1])
    if final_equity <= 0:
        return -1.0
    return float(final_equity ** (1.0 / years) - 1.0)


def build_benchmark_suite(
    forward_returns: pd.DataFrame,
    momentum_lookback: int = 5,
    vol_lookback: int = 20,
    target_vol_annual: float = 0.15,
) -> dict[str, pd.Series]:
    """Build a compact benchmark suite from forward-return data."""
    if forward_returns.empty:
        return {}

    returns = forward_returns.astype(float).fillna(0.0).sort_index()
    index = returns.index

    equal_weight = returns.mean(axis=1).rename("equal_weight")

    # Momentum long-only: select top half by trailing lookback return.
    trailing = (
        (1.0 + returns)
        .rolling(momentum_lookback, min_periods=max(2, momentum_lookback // 2))
        .apply(np.prod, raw=True)
        - 1.0
    ).shift(1)
    momentum_rows: list[tuple[pd.Timestamp, float]] = []
    for d in index:
        score = trailing.loc[d].replace([np.inf, -np.inf], np.nan).dropna()
        if score.empty:
            momentum_rows.append((pd.Timestamp(d), 0.0))
            continue
        cutoff = float(score.quantile(0.5))
        selected = score[score >= cutoff].index.tolist()
        if not selected:
            selected = score.nlargest(max(1, len(score) // 2)).index.tolist()
        w = pd.Series(1.0 / len(selected), index=selected, dtype=float)
        r = returns.loc[d, selected].fillna(0.0)
        momentum_rows.append((pd.Timestamp(d), float((w * r).sum())))
    momentum_long_only = pd.Series(
        [v for _, v in momentum_rows],
        index=pd.DatetimeIndex([d for d, _ in momentum_rows]),
        name="momentum_long_only",
        dtype=float,
    )

    # Inverse-volatility heuristic benchmark.
    rolling_vol = returns.rolling(vol_lookback, min_periods=max(3, vol_lookback // 4)).std().shift(1)
    inv_vol_rows: list[tuple[pd.Timestamp, float]] = []
    for d in index:
        vol_row = rolling_vol.loc[d].replace(0.0, np.nan).replace([np.inf, -np.inf], np.nan).dropna()
        if vol_row.empty:
            inv_vol_rows.append((pd.Timestamp(d), float(equal_weight.loc[d])))
            continue
        w = (1.0 / vol_row).astype(float)
        w = w / w.sum()
        r = returns.loc[d, w.index].fillna(0.0)
        inv_vol_rows.append((pd.Timestamp(d), float((w * r).sum())))
    inverse_volatility = pd.Series(
        [v for _, v in inv_vol_rows],
        index=pd.DatetimeIndex([d for d, _ in inv_vol_rows]),
        name="inverse_volatility",
        dtype=float,
    )

    # Vol-target equal-weight benchmark.
    target_daily_vol = float(target_vol_annual) / np.sqrt(252.0)
    realized_vol = equal_weight.rolling(vol_lookback, min_periods=max(5, vol_lookback // 4)).std().shift(1)
    leverage = (target_daily_vol / (realized_vol + 1e-12)).clip(lower=0.0, upper=3.0).fillna(1.0)
    vol_target_equal_weight = (equal_weight * leverage).rename("vol_target_equal_weight")

    return {
        "equal_weight": equal_weight,
        "momentum_long_only": momentum_long_only.reindex(index, fill_value=0.0),
        "inverse_volatility": inverse_volatility.reindex(index, fill_value=0.0),
        "vol_target_equal_weight": vol_target_equal_weight.reindex(index, fill_value=0.0),
    }


def summarize_benchmark_comparison(
    strategy_returns: pd.Series,
    benchmark_returns: Mapping[str, pd.Series] | None,
) -> pd.DataFrame:
    """Compare OOS strategy returns against benchmark return series."""
    columns = [
        "benchmark",
        "strategy_sharpe",
        "benchmark_sharpe",
        "sharpe_delta",
        "strategy_cagr",
        "benchmark_cagr",
        "cagr_delta",
        "strategy_max_drawdown",
        "benchmark_max_drawdown",
    ]
    if strategy_returns.empty or not benchmark_returns:
        return pd.DataFrame(columns=columns)

    rows: list[dict[str, float | str]] = []
    strategy = strategy_returns.sort_index()
    for name, benchmark in benchmark_returns.items():
        bench = benchmark.sort_index().astype(float)
        common = strategy.index.intersection(bench.index)
        if len(common) < 2:
            continue

        strat_slice = strategy.loc[common]
        bench_slice = bench.loc[common]
        rows.append(
            {
                "benchmark": str(name),
                "strategy_sharpe": _annualized_sharpe(strat_slice),
                "benchmark_sharpe": _annualized_sharpe(bench_slice),
                "sharpe_delta": _annualized_sharpe(strat_slice) - _annualized_sharpe(bench_slice),
                "strategy_cagr": _cagr_from_returns(strat_slice),
                "benchmark_cagr": _cagr_from_returns(bench_slice),
                "cagr_delta": _cagr_from_returns(strat_slice) - _cagr_from_returns(bench_slice),
                "strategy_max_drawdown": _max_drawdown_from_returns(strat_slice),
                "benchmark_max_drawdown": _max_drawdown_from_returns(bench_slice),
            }
        )

    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(rows, columns=columns).sort_values("sharpe_delta", ascending=False).reset_index(drop=True)


def _combine_component_signals(
    component_signals: Mapping[str, pd.DataFrame],
) -> pd.DataFrame:
    combined: pd.DataFrame | None = None
    n_components = 0
    for frame in component_signals.values():
        if frame is None or frame.empty:
            continue
        current = frame.astype(float).sort_index()
        if combined is None:
            combined = current.copy()
        else:
            combined = combined.add(current, fill_value=0.0)
        n_components += 1

    if combined is None or n_components == 0:
        return pd.DataFrame()
    return combined / float(n_components)


def compute_ablation_contributions(
    *,
    component_signals: Mapping[str, pd.DataFrame],
    forward_returns: pd.DataFrame,
) -> pd.DataFrame:
    """Estimate standalone and marginal contributions for each component."""
    columns = [
        "component",
        "standalone_sharpe",
        "standalone_cagr",
        "standalone_max_drawdown",
        "marginal_sharpe_delta",
        "ensemble_sharpe",
    ]
    usable = {k: v for k, v in component_signals.items() if v is not None and not v.empty}
    if not usable or forward_returns.empty:
        return pd.DataFrame(columns=columns)

    ensemble_signal = _combine_component_signals(usable)
    ensemble_returns = compute_strategy_returns(ensemble_signal, forward_returns)
    ensemble_sharpe = _annualized_sharpe(ensemble_returns)

    rows: list[dict[str, float | str]] = []
    for name, signal in usable.items():
        standalone_returns = compute_strategy_returns(signal, forward_returns)
        without = {k: v for k, v in usable.items() if k != name}
        without_signal = _combine_component_signals(without)
        without_returns = compute_strategy_returns(without_signal, forward_returns) if not without_signal.empty else pd.Series(dtype=float)
        marginal_delta = ensemble_sharpe - _annualized_sharpe(without_returns)

        rows.append(
            {
                "component": str(name),
                "standalone_sharpe": _annualized_sharpe(standalone_returns),
                "standalone_cagr": _cagr_from_returns(standalone_returns),
                "standalone_max_drawdown": _max_drawdown_from_returns(standalone_returns),
                "marginal_sharpe_delta": float(marginal_delta),
                "ensemble_sharpe": float(ensemble_sharpe),
            }
        )

    return pd.DataFrame(rows, columns=columns).sort_values(
        "marginal_sharpe_delta",
        ascending=False,
    ).reset_index(drop=True)


def derive_regime_labels(
    market_returns: pd.Series,
    lookback: int = 20,
) -> pd.Series:
    """Derive coarse bull/bear/transition labels from market return/volatility."""
    series = market_returns.astype(float).fillna(0.0).sort_index()
    if series.empty:
        return pd.Series(dtype=object, name="regime")

    min_periods = max(3, lookback // 4)
    roll_ret = series.rolling(lookback, min_periods=min_periods).mean()
    roll_vol = series.rolling(lookback, min_periods=min_periods).std()
    vol_cutoff = float(roll_vol.median(skipna=True)) if not roll_vol.dropna().empty else 0.0

    labels = pd.Series("transition", index=series.index, dtype=object, name="regime")
    labels[(roll_ret > 0.0) & (roll_vol <= vol_cutoff)] = "bull"
    labels[(roll_ret < 0.0) & (roll_vol > vol_cutoff)] = "bear"
    return labels.ffill().fillna("transition")


def summarize_regime_performance(
    strategy_returns: pd.Series,
    regime_labels: pd.Series,
) -> pd.DataFrame:
    """Summarize strategy behavior by market regime."""
    columns = ["regime", "n_obs", "mean_return", "volatility", "sharpe", "max_drawdown", "cagr"]
    if strategy_returns.empty or regime_labels.empty:
        return pd.DataFrame(columns=columns)

    common = strategy_returns.index.intersection(regime_labels.index)
    if len(common) == 0:
        return pd.DataFrame(columns=columns)

    ret = strategy_returns.loc[common].astype(float)
    reg = regime_labels.loc[common].astype(str)
    rows: list[dict[str, float | str | int]] = []
    for regime, grp in ret.groupby(reg):
        rows.append(
            {
                "regime": str(regime),
                "n_obs": int(len(grp)),
                "mean_return": float(grp.mean()) if len(grp) else 0.0,
                "volatility": float(grp.std()) if len(grp) > 1 else 0.0,
                "sharpe": _annualized_sharpe(grp),
                "max_drawdown": _max_drawdown_from_returns(grp),
                "cagr": _cagr_from_returns(grp),
            }
        )

    all_regimes = sorted(pd.Index(regime_labels.dropna().astype(str).unique()).tolist())
    present = {str(row["regime"]) for row in rows}
    for regime in all_regimes:
        if regime in present:
            continue
        rows.append(
            {
                "regime": regime,
                "n_obs": 0,
                "mean_return": 0.0,
                "volatility": 0.0,
                "sharpe": 0.0,
                "max_drawdown": 0.0,
                "cagr": 0.0,
            }
        )

    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(rows, columns=columns).sort_values("n_obs", ascending=False).reset_index(drop=True)


def run_walk_forward_analysis(
    *,
    signals: pd.DataFrame,
    forward_returns: pd.DataFrame,
    train_window: int,
    test_window: int,
    step: int | None = None,
    purge_period: int = 0,
    embargo_period: int = 0,
    benchmark_returns: Mapping[str, pd.Series] | None = None,
    regime_labels: pd.Series | None = None,
    component_signals: Mapping[str, pd.DataFrame] | None = None,
) -> dict[str, pd.DataFrame | pd.Series | dict[str, float]]:
    """Run deterministic walk-forward diagnostics plus benchmark/regime reports."""
    splits = build_purged_walk_forward_splits(
        dates=signals.index,
        train_window=train_window,
        test_window=test_window,
        step=step,
        purge_period=purge_period,
        embargo_period=embargo_period,
    )

    rows: list[dict[str, float | int | str]] = []
    oos_chunks: list[pd.Series] = []
    for window_id, split in enumerate(splits):
        train_signals = signals.loc[split.train_start : split.train_end]
        train_returns = forward_returns.loc[split.train_start : split.train_end]
        test_signals = signals.loc[split.test_start : split.test_end]
        test_returns = forward_returns.loc[split.test_start : split.test_end]

        train_strategy = compute_strategy_returns(train_signals, train_returns)
        test_strategy = compute_strategy_returns(test_signals, test_returns)

        rows.append(
            {
                "window_id": int(window_id),
                "train_start": split.train_start.isoformat(),
                "train_end": split.train_end.isoformat(),
                "test_start": split.test_start.isoformat(),
                "test_end": split.test_end.isoformat(),
                "train_sharpe": _annualized_sharpe(train_strategy),
                "test_sharpe": _annualized_sharpe(test_strategy),
                "test_max_drawdown": _max_drawdown_from_returns(test_strategy),
                "test_turnover": estimate_turnover_from_signals(test_signals),
            }
        )
        if not test_strategy.empty:
            oos_chunks.append(test_strategy)

    window_metrics = pd.DataFrame(rows)
    summary = summarize_walk_forward_metrics(window_metrics)

    if oos_chunks:
        oos_returns = pd.concat(oos_chunks).sort_index()
        if oos_returns.index.has_duplicates:
            oos_returns = oos_returns.groupby(level=0).mean()
    else:
        oos_returns = pd.Series(dtype=float, name="strategy_return")

    if benchmark_returns:
        benchmark_map = dict(benchmark_returns)
    else:
        benchmark_map = build_benchmark_suite(forward_returns)
        if "SPY" in forward_returns.columns:
            benchmark_map["spy"] = forward_returns["SPY"].astype(float)

    benchmark_df = summarize_benchmark_comparison(oos_returns, benchmark_map)

    resolved_regimes = regime_labels
    if resolved_regimes is None:
        if "equal_weight" in benchmark_map:
            resolved_regimes = derive_regime_labels(benchmark_map["equal_weight"])
        elif not forward_returns.empty:
            resolved_regimes = derive_regime_labels(forward_returns.mean(axis=1).astype(float))
        else:
            resolved_regimes = pd.Series(dtype=object)

    regime_df = summarize_regime_performance(
        oos_returns,
        resolved_regimes if resolved_regimes is not None else pd.Series(dtype=object),
    )

    if not benchmark_df.empty and "equal_weight" in set(benchmark_df["benchmark"]):
        row = benchmark_df.loc[benchmark_df["benchmark"] == "equal_weight"].iloc[0]
        summary["equal_weight_sharpe_delta"] = float(row["sharpe_delta"])
        summary["equal_weight_cagr_delta"] = float(row["cagr_delta"])
    summary["regime_count"] = int(regime_df["regime"].nunique()) if not regime_df.empty else 0

    if component_signals and not oos_returns.empty:
        if oos_returns.index.empty:
            component_oos = {}
            fwd_oos = pd.DataFrame()
        else:
            start = oos_returns.index.min()
            end = oos_returns.index.max()
            component_oos = {
                name: frame.loc[start:end].copy()
                for name, frame in component_signals.items()
                if frame is not None and not frame.empty
            }
            fwd_oos = forward_returns.loc[start:end].copy()
        ablation_df = compute_ablation_contributions(
            component_signals=component_oos,
            forward_returns=fwd_oos,
        )
    else:
        ablation_df = pd.DataFrame(
            columns=[
                "component",
                "standalone_sharpe",
                "standalone_cagr",
                "standalone_max_drawdown",
                "marginal_sharpe_delta",
                "ensemble_sharpe",
            ]
        )

    if not ablation_df.empty:
        summary["ablation_component_count"] = int(len(ablation_df))
        summary["best_component_marginal_sharpe"] = float(
            ablation_df.iloc[0]["marginal_sharpe_delta"]
        )

    return {
        "window_metrics": window_metrics,
        "summary": summary,
        "oos_returns": oos_returns,
        "benchmark_comparison": benchmark_df,
        "regime_performance": regime_df,
        "ablation_analysis": ablation_df,
    }


# ---------------------------------------------------------------------------
# Lookahead-bias detection
# ---------------------------------------------------------------------------

_SUSPICIOUS_PATTERNS = {"forward", "future", "next_day", "lead", "tomorrow"}

IC_CEILING = 0.50  # Spearman IC > 0.50 is unrealistically high for daily returns


@dataclass(frozen=True)
class LookaheadWarning:
    """Single finding from a lookahead-bias check."""

    check: str
    column: str
    detail: str


def check_feature_name_leakage(feature_cols: Sequence[str]) -> list[LookaheadWarning]:
    """Flag feature names that suggest forward-looking information."""
    warnings: list[LookaheadWarning] = []
    for col in feature_cols:
        lower = col.lower()
        for pat in _SUSPICIOUS_PATTERNS:
            if pat in lower:
                warnings.append(
                    LookaheadWarning(
                        check="name_pattern",
                        column=col,
                        detail=f"Feature name contains '{pat}' — may encode future data.",
                    )
                )
                break
    return warnings


def check_feature_target_alignment(
    features_df: pd.DataFrame,
    target_col: str = "forward_return",
    date_col: str = "valid_time",
) -> list[LookaheadWarning]:
    """Verify that feature dates are strictly before target dates.

    When features and targets live in the same DataFrame, the target at
    row T should represent the return from T → T+1, meaning the feature
    row at T must only use information up to and including T's close.

    This check verifies that the target column is indeed shifted (i.e.
    it is not identical to the same-day return column, if one exists).
    """
    warnings: list[LookaheadWarning] = []
    if target_col not in features_df.columns or date_col not in features_df.columns:
        return warnings

    # Check: target should differ from any "return" feature by at least a shift
    return_cols = [c for c in features_df.columns if "return" in c.lower() and c != target_col]
    for rc in return_cols:
        if features_df[target_col].equals(features_df[rc]):
            warnings.append(
                LookaheadWarning(
                    check="target_alignment",
                    column=rc,
                    detail=(
                        f"Target '{target_col}' is identical to feature '{rc}' — "
                        "target may not be shifted forward, causing lookahead bias."
                    ),
                )
            )
    return warnings


def check_unrealistic_ic(
    features_df: pd.DataFrame,
    target_col: str = "forward_return",
    ic_ceiling: float = IC_CEILING,
) -> list[LookaheadWarning]:
    """Flag features with suspiciously high information coefficient.

    A Spearman IC > ``ic_ceiling`` (default 0.50) on daily returns is
    almost certainly data leakage rather than genuine alpha.
    """
    from scipy.stats import spearmanr

    warnings: list[LookaheadWarning] = []
    if target_col not in features_df.columns:
        return warnings

    target = features_df[target_col].dropna()
    numeric_cols = features_df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if col == target_col:
            continue
        common = features_df[[col, target_col]].dropna()
        if len(common) < 30:
            continue
        corr, _ = spearmanr(common[col], common[target_col])
        if abs(corr) > ic_ceiling:
            warnings.append(
                LookaheadWarning(
                    check="unrealistic_ic",
                    column=col,
                    detail=(
                        f"Spearman IC = {corr:.3f} exceeds ceiling {ic_ceiling:.2f} — "
                        "likely lookahead bias or data leakage."
                    ),
                )
            )
    return warnings


def validate_no_lookahead(
    features_df: pd.DataFrame | None = None,
    feature_cols: Sequence[str] | None = None,
    target_col: str = "forward_return",
    date_col: str = "valid_time",
    ic_ceiling: float = IC_CEILING,
    raise_on_error: bool = False,
) -> list[LookaheadWarning]:
    """Run all lookahead-bias checks and return combined warnings.

    Parameters
    ----------
    features_df:
        DataFrame with features and target.  When ``None``, only the
        name-pattern check runs (using *feature_cols*).
    feature_cols:
        Explicit list of feature column names.  Inferred from
        *features_df* when not provided.
    target_col:
        Name of the target (forward return) column.
    date_col:
        Name of the date column.
    ic_ceiling:
        Max plausible absolute Spearman IC before flagging.
    raise_on_error:
        If ``True``, raise ``ValueError`` when any warning is found.

    Returns
    -------
    list[LookaheadWarning]
    """
    all_warnings: list[LookaheadWarning] = []

    cols = list(feature_cols) if feature_cols else []
    if features_df is not None and not cols:
        cols = [c for c in features_df.columns if c not in {target_col, date_col, "ticker"}]

    # 1. Name-based check
    all_warnings.extend(check_feature_name_leakage(cols))

    if features_df is not None and not features_df.empty:
        # 2. Alignment check
        all_warnings.extend(
            check_feature_target_alignment(features_df, target_col, date_col)
        )
        # 3. IC ceiling check
        all_warnings.extend(
            check_unrealistic_ic(features_df, target_col, ic_ceiling)
        )

    for w in all_warnings:
        logger.warning("Lookahead bias check [%s] %s: %s", w.check, w.column, w.detail)

    if raise_on_error and all_warnings:
        msg = "; ".join(f"[{w.check}] {w.column}: {w.detail}" for w in all_warnings)
        raise ValueError(f"Lookahead bias detected: {msg}")

    return all_warnings
