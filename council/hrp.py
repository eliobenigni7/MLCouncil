"""Hierarchical Risk Parity (López de Prado) weight construction.

Used as an optional *soft prior* blended with the CVXPY mean-variance solution
when ``MLCOUNCIL_HRP_SOFT_PRIOR=true``, or via ``MLCOUNCIL_PORTFOLIO_MODE=hrp_blend``.
Canary status: shadow — target: P-1.1 — expiry: 2027-02-01 (promote via canary o retire)
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform


def _cov_to_corr(cov: np.ndarray) -> np.ndarray:
    std = np.sqrt(np.diag(cov))
    std[std < 1e-12] = 1e-12
    corr = cov / np.outer(std, std)
    return np.clip(corr, -1.0, 1.0)


def _get_ivp(cov: np.ndarray) -> np.ndarray:
    ivp = 1.0 / np.maximum(np.diag(cov), 1e-12)
    return ivp / ivp.sum()


def _get_quasi_diag(link: np.ndarray) -> list[int]:
    """Return leaf permutation from scipy linkage matrix."""
    link = link.astype(int)
    sort_ix = [int(link[-1, 0]), int(link[-1, 1])]
    num_items = int(link[-1, 3])
    while max(sort_ix) >= num_items:
        sort_ix_old = sort_ix.copy()
        sort_ix = []
        for item in sort_ix_old:
            if item < num_items:
                sort_ix.append(item)
            else:
                j = item - num_items
                sort_ix.append(int(link[j, 0]))
                sort_ix.append(int(link[j, 1]))
    return sort_ix


def _get_cluster_var(cov: np.ndarray, cluster_items: list[int]) -> float:
    sub = cov[np.ix_(cluster_items, cluster_items)]
    w = _get_ivp(sub).reshape(-1, 1)
    return float((w.T @ sub @ w).squeeze())


def _recursive_bisection(cov: np.ndarray, sort_ix: list[int]) -> np.ndarray:
    w = np.ones(len(sort_ix), dtype=float)
    clusters = [sort_ix]
    while clusters:
        next_clusters: list[list[int]] = []
        for cluster in clusters:
            if len(cluster) <= 1:
                continue
            mid = len(cluster) // 2
            left = cluster[:mid]
            right = cluster[mid:]
            if not left or not right:
                continue
            var_left = _get_cluster_var(cov, left)
            var_right = _get_cluster_var(cov, right)
            alpha = 1.0 - var_left / (var_left + var_right + 1e-12)
            w[left] *= alpha
            w[right] *= 1.0 - alpha
            if len(left) > 1:
                next_clusters.append(left)
            if len(right) > 1:
                next_clusters.append(right)
        clusters = next_clusters
    return w


def hrp_weights_from_covariance(cov: pd.DataFrame) -> pd.Series:
    """Return long-only HRP weights summing to 1 from a covariance matrix."""
    tickers = list(cov.index)
    n = len(tickers)
    if n == 0:
        return pd.Series(dtype=float, name="target_weight")
    if n == 1:
        return pd.Series({tickers[0]: 1.0}, name="target_weight")

    cov_arr = cov.reindex(index=tickers, columns=tickers).fillna(0.0).values
    cov_arr = (cov_arr + cov_arr.T) / 2.0 + np.eye(n) * 1e-8

    corr = _cov_to_corr(cov_arr)
    dist = np.sqrt(np.maximum(0.0, 0.5 * (1.0 - corr)))
    np.fill_diagonal(dist, 0.0)
    condensed = squareform(dist, checks=False)
    link = linkage(condensed, method="single")
    sort_ix = _get_quasi_diag(link)

    if len(sort_ix) != n or len(set(sort_ix)) != n:
        raw = _get_ivp(cov_arr)
    else:
        raw = _recursive_bisection(cov_arr, sort_ix)

    raw = np.clip(raw, 0.0, None)
    total = float(raw.sum())
    if total < 1e-12:
        raw = _get_ivp(cov_arr)
    else:
        raw /= total
    return pd.Series(raw, index=tickers, name="target_weight")


def covariance_condition_number(cov: pd.DataFrame) -> float:
    arr = cov.fillna(0.0).values
    if arr.size == 0:
        return float("inf")
    n = arr.shape[0]
    return float(np.linalg.cond(arr + np.eye(n) * 1e-8))


def hrp_blend_weighting_mode() -> str:
    """``fixed`` (default) or ``ir`` (condition-number proxy for blend λ)."""
    raw = os.getenv("MLCOUNCIL_HRP_BLEND_WEIGHTING", "fixed").strip().lower()
    return raw if raw in ("fixed", "ir") else "fixed"


def resolve_hrp_blend_lambda(
    cov: pd.DataFrame,
    *,
    fixed_blend: float | None = None,
    weighting: str | None = None,
) -> float:
    """Blend weight λ on HRP: ``(1-λ)*CVXPY + λ*HRP``.

    *fixed* — ``MLCOUNCIL_HRP_BLEND`` (default 0.5 for ``hrp_blend`` portfolio mode).
    *ir* — raises λ when covariance is ill-conditioned (proxy for trusting HRP more).
    """
    mode = (weighting or hrp_blend_weighting_mode()).strip().lower()
    if fixed_blend is None:
        fixed_blend = float(os.getenv("MLCOUNCIL_HRP_BLEND", "0.5"))
    lam = float(np.clip(fixed_blend, 0.0, 1.0))
    if mode != "ir" or cov.shape[0] < 2:
        return lam

    ref = float(os.getenv("MLCOUNCIL_HRP_IR_COND_REF", "100"))
    cond = covariance_condition_number(cov)
    if not np.isfinite(cond) or ref <= 1.0:
        return lam

    ir_proxy = float(np.log1p(cond) / np.log1p(ref))
    ir_proxy = float(np.clip(ir_proxy, 0.15, 0.85))
    # Convex mix: respect configured floor/ceiling while letting IR scale λ
    return float(np.clip(0.5 * lam + 0.5 * ir_proxy, 0.15, 0.85))


def blend_cvxpy_with_hrp(
    cvxpy_weights: np.ndarray,
    cov: pd.DataFrame,
    tickers: list[str],
    *,
    blend_lambda: float | None = None,
) -> np.ndarray:
    """Linear blend of CVXPY weights with HRP, renormalized to sum 1."""
    n = len(tickers)
    if n < 2:
        return np.asarray(cvxpy_weights, dtype=float).reshape(-1)

    cov_df = cov.reindex(index=tickers, columns=tickers).fillna(0.0)
    lam = (
        float(np.clip(blend_lambda, 0.0, 1.0))
        if blend_lambda is not None
        else resolve_hrp_blend_lambda(cov_df)
    )
    hrp_w = hrp_weights_from_covariance(cov_df).reindex(tickers).fillna(0.0).values
    cvx = np.asarray(cvxpy_weights, dtype=float).reshape(-1)
    blended = (1.0 - lam) * cvx + lam * hrp_w
    total = float(blended.sum())
    if total > 1e-12:
        blended /= total
    else:
        blended = hrp_w
    return np.clip(blended, 0.0, None)
