"""Dynamic conditional covariance estimators (T3.4 shadow).

``MLCOUNCIL_COVARIANCE_ESTIMATOR=dcc`` enables DCC-GARCH scaffold;
default ``ledoit`` keeps Ledoit-Wolf in ``data/pipeline._compute_covariance``.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
from loguru import logger


def covariance_estimator_mode() -> str:
    """``ledoit`` (default) or ``dcc``."""
    raw = os.getenv("MLCOUNCIL_COVARIANCE_ESTIMATOR", "ledoit").strip().lower()
    return raw if raw in ("ledoit", "dcc") else "ledoit"


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


def compute_covariance_from_returns(
    returns_wide: pd.DataFrame,
    *,
    estimator: str | None = None,
) -> pd.DataFrame:
    """Build covariance matrix from a wide daily return panel."""
    mode = (estimator or covariance_estimator_mode()).strip().lower()
    if mode == "dcc":
        return DCCEstimator().fit(returns_wide).cov()

    cov_df = returns_wide.cov(min_periods=30)
    if len(returns_wide.columns) > 1 and len(returns_wide) >= 5:
        from sklearn.covariance import LedoitWolf

        returns_clean = returns_wide.dropna(axis=1, how="all").dropna()
        if len(returns_clean) >= 5 and len(returns_clean.columns) > 1:
            lw = LedoitWolf().fit(returns_clean.values)
            cov_df = pd.DataFrame(
                lw.covariance_,
                index=returns_clean.columns,
                columns=returns_clean.columns,
            )
    return cov_df
