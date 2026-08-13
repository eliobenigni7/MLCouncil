"""Multi-period portfolio optimization scaffold (Gârleanu–Pedersen style).

Full dynamic programming / convex multi-period solver is deferred. This module
defines the interface for *target smoothing* under transaction costs when
rebalancing over a horizon H.

Reference: Gârleanu & Pedersen, "Dynamic Trading with Predictable Returns and
Transaction Costs", Journal of Finance (2013).

Enable experiments via ``MLCOUNCIL_MULTI_PERIOD_TC=true`` (pipeline integration
not wired; call :func:`smooth_target_weights` from backtests/scripts).
Canary status: shadow — target: P-1.1 — expiry: 2027-02-01 (promote via canary o retire)
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import pandas as pd


def multi_period_tc_enabled() -> bool:
    raw = os.getenv("MLCOUNCIL_MULTI_PERIOD_TC", "").strip().lower()
    return raw in ("true", "1", "yes")


@dataclass(frozen=True)
class MultiPeriodTCConfig:
    """Knobs for GP-style exponential smoothing (scaffold)."""

    horizon_days: int = 5
    risk_aversion: float = 1.0
    tc_lambda: float = 2.0  # scale aligned with council.portfolio tc_lambda
    smoothing: float = 0.5  # fraction of gap closed per day toward w*


def multi_period_config_from_env() -> MultiPeriodTCConfig:
    def _int(key: str, default: int) -> int:
        raw = os.getenv(key)
        if raw is None or not str(raw).strip():
            return default
        try:
            return max(1, int(raw))
        except ValueError:
            return default

    def _float(key: str, default: float) -> float:
        raw = os.getenv(key)
        if raw is None or not str(raw).strip():
            return default
        try:
            return float(raw)
        except ValueError:
            return default

    return MultiPeriodTCConfig(
        horizon_days=_int("MLCOUNCIL_MP_HORIZON_DAYS", 5),
        risk_aversion=_float("MLCOUNCIL_MP_RISK_AVERSION", 1.0),
        tc_lambda=_float("MLCOUNCIL_MP_TC_LAMBDA", 2.0),
        smoothing=_float("MLCOUNCIL_MP_SMOOTHING", 0.5),
    )


def smooth_target_weights(
    w_star: pd.Series,
    w_current: pd.Series,
    *,
    config: MultiPeriodTCConfig | None = None,
) -> pd.Series:
    """One-step Gârleanu–Pedersen-style smoothing toward optimal target ``w_star``.

    Implements a tractable approximation::

        w_trade = w_current + α (w* - w_current),   α = min(1, smoothing / H)

    rather than solving the full multi-period Bellman system. Production daily
    path is unchanged unless callers opt in explicitly.
    """
    cfg = config or multi_period_config_from_env()
    tickers = w_star.index.union(w_current.index)
    star = w_star.reindex(tickers).fillna(0.0).values
    curr = w_current.reindex(tickers).fillna(0.0).values
    h = max(1, int(cfg.horizon_days))
    alpha = float(np.clip(cfg.smoothing / h, 0.0, 1.0))
    traded = curr + alpha * (star - curr)
    total = float(traded.sum())
    if total > 1e-12:
        traded = traded / total
    return pd.Series(traded, index=tickers, name="target_weight")


def plan_multiperiod_rebalance(
    w_star: pd.Series,
    w_current: pd.Series,
    *,
    config: MultiPeriodTCConfig | None = None,
) -> list[pd.Series]:
    """Return a length-``horizon_days`` list of daily weight paths (scaffold).

    Each step applies :func:`smooth_target_weights` with the same ``w_star``;
    not a full stochastic control solution.
    """
    cfg = config or multi_period_config_from_env()
    path: list[pd.Series] = []
    w = w_current.copy()
    for _ in range(cfg.horizon_days):
        w = smooth_target_weights(w_star, w, config=cfg)
        path.append(w.copy())
    return path
