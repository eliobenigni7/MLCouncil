"""Deterministic walk-forward diagnostics for model validation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import pandas as pd


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
