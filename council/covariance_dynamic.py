"""Dynamic conditional covariance estimators (T3.4 shadow).

``MLCOUNCIL_COVARIANCE_ESTIMATOR`` — ``ledoit`` (default), ``dcc``, or ``factor``
(factor scaffold Σ = B Σ_f B' + D; falls back to Ledoit-Wolf when factors missing).
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
from loguru import logger

_COVARIANCE_MODES = frozenset({"ledoit", "dcc", "factor"})


def covariance_estimator_mode() -> str:
    """``ledoit`` (default), ``dcc``, or ``factor``."""
    raw = os.getenv("MLCOUNCIL_COVARIANCE_ESTIMATOR", "ledoit").strip().lower()
    return raw if raw in _COVARIANCE_MODES else "ledoit"


class DCCEstimator:
    """DCC-GARCH scaffold: univariate GARCH(1,1) vols + EWMA correlation dynamics.

    Falls back to sample covariance + diagonal shrinkage when ``arch`` is absent
    or the return panel is too short.
    """

    def __init__(
        self,
        *,
        a: float = 0.03,
        b: float = 0.96,
        min_obs: int = 30,
    ) -> None:
        self.a = float(a)
        self.b = float(b)
        self.min_obs = int(min_obs)
        self._tickers: list[str] = []
        self._cov: pd.DataFrame | None = None

    def fit(self, returns: pd.DataFrame) -> "DCCEstimator":
        """Fit on wide daily returns (index=dates, columns=tickers)."""
        rets = returns.dropna(how="all").dropna(axis=1, how="all")
        if rets.shape[1] < 1:
            raise ValueError("returns must have at least one ticker column")

        self._tickers = list(rets.columns)
        if len(rets) < self.min_obs:
            logger.warning(
                f"DCC: only {len(rets)} rows (< {self.min_obs}); using sample cov fallback"
            )
            self._cov = self._sample_cov(rets)
            return self

        std_resid = self._garch_standardized_residuals(rets)
        if std_resid is None or std_resid.shape[0] < 5:
            self._cov = self._sample_cov(rets)
            return self

        q_bar = std_resid.cov(min_periods=5).values
        q_t = q_bar.copy()
        eps = std_resid.values
        for t in range(1, len(eps)):
            e = eps[t - 1 : t].T @ eps[t - 1 : t]
            q_t = (1.0 - self.a - self.b) * q_bar + self.a * e + self.b * q_t

        vols = self._univariate_vols(rets)
        d = np.diag(vols)
        sigma = d @ q_t @ d
        sigma = self._make_psd(sigma)
        self._cov = pd.DataFrame(sigma, index=self._tickers, columns=self._tickers)
        return self

    def cov(self) -> pd.DataFrame:
        if self._cov is None:
            raise RuntimeError("DCCEstimator not fitted")
        return self._cov.copy()

    def _garch_standardized_residuals(self, rets: pd.DataFrame) -> pd.DataFrame | None:
        try:
            from arch import arch_model
        except ImportError:
            logger.warning("arch not installed; DCC uses sample-residual correlation")
            z = rets - rets.mean()
            vol = rets.std().replace(0.0, np.nan).fillna(1e-6)
            return z.div(vol)
        cols = []
        for ticker in rets.columns:
            series = rets[ticker].dropna() * 100.0
            if len(series) < self.min_obs:
                continue
            try:
                am = arch_model(series, vol="Garch", p=1, q=1, rescale=False)
                res = am.fit(disp="off", show_warning=False)
                std = np.asarray(res.conditional_volatility, dtype=float)
                std[std < 1e-8] = 1e-8
                z = np.asarray(res.resid, dtype=float) / std
                cols.append(pd.Series(z, index=series.index, name=ticker))
            except Exception as exc:
                logger.debug(f"DCC GARCH failed for {ticker}: {exc}")
        if not cols:
            return None
        return pd.concat(cols, axis=1).dropna(how="any")

    def _univariate_vols(self, rets: pd.DataFrame) -> np.ndarray:
        vols = rets.std().replace(0.0, np.nan).fillna(1e-6).values
        return np.maximum(vols, 1e-6)

    @staticmethod
    def _sample_cov(rets: pd.DataFrame) -> pd.DataFrame:
        cov = rets.cov(min_periods=5)
        arr = cov.values
        arr = DCCEstimator._make_psd(arr)
        return pd.DataFrame(arr, index=cov.index, columns=cov.columns)

    @staticmethod
    def _make_psd(matrix: np.ndarray, eps: float = 1e-8) -> np.ndarray:
        sym = (matrix + matrix.T) / 2.0
        eigvals, eigvecs = np.linalg.eigh(sym)
        eigvals = np.maximum(eigvals, eps)
        return eigvecs @ np.diag(eigvals) @ eigvecs.T


class FactorCovarianceEstimator:
    """Factor-model covariance scaffold: Σ = B Σ_f B' + D.

    When factor returns or loadings are not supplied, falls back to Ledoit-Wolf
    on the asset return panel (same path as ``ledoit`` mode).
    """

    def __init__(self, *, min_obs: int = 30, n_factors: int = 3) -> None:
        self.min_obs = int(min_obs)
        self.n_factors = int(n_factors)
        self._tickers: list[str] = []
        self._cov: pd.DataFrame | None = None

    def fit(
        self,
        returns: pd.DataFrame,
        *,
        factor_returns: pd.DataFrame | None = None,
        loadings: pd.DataFrame | None = None,
    ) -> "FactorCovarianceEstimator":
        """Fit Σ = B Σ_f B' + diag(D) from wide asset returns."""
        rets = returns.dropna(how="all").dropna(axis=1, how="all")
        if rets.shape[1] < 1:
            raise ValueError("returns must have at least one ticker column")
        self._tickers = list(rets.columns)

        if len(rets) < self.min_obs:
            logger.warning(
                f"Factor cov: only {len(rets)} rows (< {self.min_obs}); LW fallback"
            )
            self._cov = _ledoit_wolf_cov(rets)
            return self

        B, sigma_f, residual_var = self._estimate_factor_structure(
            rets, factor_returns=factor_returns, loadings=loadings
        )
        if B is None or sigma_f is None:
            logger.warning("Factor cov: missing factors/loadings; LW fallback")
            self._cov = _ledoit_wolf_cov(rets)
            return self

        d = np.diag(np.maximum(residual_var, 1e-8))
        sigma = B @ sigma_f @ B.T + d
        sigma = DCCEstimator._make_psd(sigma)
        self._cov = pd.DataFrame(sigma, index=self._tickers, columns=self._tickers)
        return self

    def cov(self) -> pd.DataFrame:
        if self._cov is None:
            raise RuntimeError("FactorCovarianceEstimator not fitted")
        return self._cov.copy()

    def _estimate_factor_structure(
        self,
        rets: pd.DataFrame,
        *,
        factor_returns: pd.DataFrame | None,
        loadings: pd.DataFrame | None,
    ) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
        tickers = list(rets.columns)
        n = len(tickers)
        clean = rets.dropna()
        if clean.shape[0] < 5:
            return None, None, None

        if loadings is not None and factor_returns is not None:
            B = loadings.reindex(index=tickers).fillna(0.0).values
            f_rets = factor_returns.dropna(how="all")
            if f_rets.shape[0] < 5 or B.shape[1] != f_rets.shape[1]:
                return None, None, None
            sigma_f = f_rets.cov(min_periods=5).values
            # Align factor panel to asset return dates (inner join on index)
            common_idx = clean.index.intersection(f_rets.index)
            if len(common_idx) < 5:
                return None, None, None
            y = clean.loc[common_idx].values
            f = f_rets.loc[common_idx].values
            resid = y - f @ B.T
            residual_var = np.var(resid, axis=0, ddof=1)
            return B, sigma_f, residual_var

        # PCA factors from return panel when external factors not provided
        try:
            from sklearn.decomposition import PCA
        except ImportError:
            return None, None, None

        k = min(self.n_factors, n - 1, clean.shape[0] - 1)
        if k < 1:
            return None, None, None

        pca = PCA(n_components=k)
        scores = pca.fit_transform(clean.values)
        B = pca.components_.T  # n_assets × k
        sigma_f = np.cov(scores, rowvar=False)
        if sigma_f.ndim == 0:
            sigma_f = np.array([[float(sigma_f)]])
        fitted = scores @ pca.components_
        resid = clean.values - fitted
        residual_var = np.var(resid, axis=0, ddof=1)
        return B, sigma_f, residual_var


def _ledoit_wolf_cov(returns_wide: pd.DataFrame) -> pd.DataFrame:
    """Ledoit-Wolf shrinkage on a wide return panel."""
    rets = returns_wide.dropna(axis=1, how="all").dropna()
    if len(rets.columns) <= 1 or len(rets) < 5:
        arr = DCCEstimator._make_psd(rets.cov(min_periods=5).values)
        tickers = list(rets.columns) if len(rets.columns) else list(returns_wide.columns)
        return pd.DataFrame(arr, index=tickers, columns=tickers)

    from sklearn.covariance import LedoitWolf

    lw = LedoitWolf().fit(rets.values)
    arr = DCCEstimator._make_psd(lw.covariance_)
    return pd.DataFrame(arr, index=rets.columns, columns=rets.columns)


def compute_covariance_from_returns(
    returns_wide: pd.DataFrame,
    *,
    estimator: str | None = None,
) -> pd.DataFrame:
    """Build covariance matrix from a wide daily return panel."""
    mode = (estimator or covariance_estimator_mode()).strip().lower()
    if mode == "dcc":
        return DCCEstimator().fit(returns_wide).cov()
    if mode == "factor":
        return FactorCovarianceEstimator().fit(returns_wide).cov()

    cov_df = returns_wide.cov(min_periods=30)
    if len(returns_wide.columns) > 1 and len(returns_wide) >= 5:
        returns_clean = returns_wide.dropna(axis=1, how="all").dropna()
        if len(returns_clean) >= 5 and len(returns_clean.columns) > 1:
            cov_df = _ledoit_wolf_cov(returns_clean)
    return cov_df


def shrink_covariance_matrix(
    cov_df: pd.DataFrame,
    *,
    n_obs: int | None = None,
) -> pd.DataFrame:
    """Ledoit-Wolf shrinkage when only a sample covariance matrix is available.

    The Dagster path already shrinks via :func:`compute_covariance_from_returns`;
    this helper covers direct calls to ``PortfolioConstructor.optimize`` (tests,
    scripts) that pass a raw sample ``DataFrame.cov()``.
    """
    tickers = list(cov_df.index)
    n = len(tickers)
    if n <= 1:
        arr = DCCEstimator._make_psd(np.asarray(cov_df, dtype=float))
        return pd.DataFrame(arr, index=tickers, columns=tickers)

    sym = (np.asarray(cov_df, dtype=float) + np.asarray(cov_df, dtype=float).T) / 2.0
    effective_n = n_obs or int(os.getenv("MLCOUNCIL_COVARIANCE_WINDOW", "90"))

    try:
        from sklearn.covariance import LedoitWolf
    except ImportError:
        logger.debug("sklearn unavailable; symmetrizing covariance without LW shrinkage")
        arr = DCCEstimator._make_psd(sym)
        return pd.DataFrame(arr, index=tickers, columns=tickers)

    try:
        root = np.linalg.cholesky(DCCEstimator._make_psd(sym) + np.eye(n) * 1e-10)
    except np.linalg.LinAlgError:
        arr = DCCEstimator._make_psd(sym)
        return pd.DataFrame(arr, index=tickers, columns=tickers)

    k = max(int(effective_n), n + 2)
    rng = np.random.default_rng(0)
    panel = rng.standard_normal((k, n)) @ root.T
    shrunk = LedoitWolf().fit(panel).covariance_
    arr = DCCEstimator._make_psd(shrunk)
    return pd.DataFrame(arr, index=tickers, columns=tickers)
