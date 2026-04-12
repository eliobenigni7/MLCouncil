"""Deterministic walk-forward diagnostics and lookahead-bias checks for model validation."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

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
