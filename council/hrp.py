"""Hierarchical Risk Parity (López de Prado) weight construction.

Used as an optional *soft prior* blended with the CVXPY mean-variance solution
when ``MLCOUNCIL_HRP_SOFT_PRIOR=true``.
"""

from __future__ import annotations

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
