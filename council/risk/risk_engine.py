"""Advanced Risk Management System for MLCouncil.

Provides institutional-grade risk management:
- Value at Risk (VaR) - Historical, Parametric, Monte Carlo
- Conditional VaR (CVaR / Expected Shortfall)
- Real-time position exposure tracking
- Sector and factor exposure limits
- Correlation stress testing
- Greeks approximation for equity portfolio

Usage:
    from council.risk.risk_engine import RiskEngine, RiskReport

    risk = RiskEngine()
    report = risk.compute_full_risk(positions, prices, returns)
    breaches = risk.check_limits(report)
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import norm as _norm

_ROOT = Path(__file__).parents[2]
RISK_DIR = _ROOT / "data" / "risk"
_DEFAULT_SECTOR_MAP_PATH = _ROOT / "config" / "sector_map.json"
RISK_DIR.mkdir(parents=True, exist_ok=True)
logger = logging.getLogger(__name__)


@dataclass
class Position:
    symbol: str
    quantity: float
    avg_price: float
    current_price: float
    sector: str = "Unknown"
    beta: float = 1.0

    @property
    def market_value(self) -> float:
        return self.quantity * self.current_price

    @property
    def cost_basis(self) -> float:
        return self.quantity * self.avg_price

    @property
    def unrealized_pnl(self) -> float:
        return self.market_value - self.cost_basis

    @property
    def unrealized_pnl_pct(self) -> float:
        return self.unrealized_pnl / self.cost_basis if self.cost_basis > 0 else 0.0


@dataclass
class VaRReport:
    var_1d: float
    var_5d: float
    var_10d: float
    cvar_1d: float
    cvar_5d: float
    cvar_10d: float
    var_method: str
    confidence_level: float
    portfolio_value: float

    def to_dict(self) -> dict:
        return {
            "var_1d": self.var_1d,
            "var_5d": self.var_5d,
            "var_10d": self.var_10d,
            "cvar_1d": self.cvar_1d,
            "cvar_5d": self.cvar_5d,
            "cvar_10d": self.cvar_10d,
            "var_method": self.var_method,
            "confidence_level": self.confidence_level,
            "portfolio_value": self.portfolio_value,
        }


@dataclass
class MonteCarloVaRResult:
    """Result of the multivariate Monte Carlo VaR simulation.

    ``var_*`` fields are the portfolio VaR; ``es_*`` fields are the expected
    shortfall (mean loss beyond the VaR quantile). ``cvar_*`` fields are
    aliases of ``es_*`` kept for backward compatibility. All values are
    positive losses; ``*_pct`` are fractions of the portfolio value and
    ``*_dollar`` are absolute dollar amounts.

    Iterating over the result yields ``(var_dollar, cvar_dollar)`` so legacy
    tuple unpacking keeps working.
    """

    var_pct: float
    var_dollar: float
    cvar_pct: float
    cvar_dollar: float
    es_pct: float = 0.0
    es_dollar: float = 0.0
    n_simulations: int = 0
    horizon: int = 1

    def __iter__(self):
        # Backward-compatible 2-tuple unpacking: (var_dollar, cvar_dollar).
        yield self.var_dollar
        yield self.cvar_dollar


@dataclass
class ExposureReport:
    total_market_value: float
    net_exposure: float
    gross_exposure: float
    sector_exposure: dict[str, float]
    factor_exposure: dict[str, float]
    concentration: dict[str, float]
    beta_exposure: float
    sector_weights: dict[str, float]

    def to_dict(self) -> dict:
        return {
            "total_market_value": self.total_market_value,
            "net_exposure": self.net_exposure,
            "gross_exposure": self.gross_exposure,
            "sector_exposure": self.sector_exposure,
            "factor_exposure": self.factor_exposure,
            "concentration": self.concentration,
            "beta_exposure": self.beta_exposure,
            "sector_weights": self.sector_weights,
        }


@dataclass
class RiskLimits:
    max_var_pct: float = 0.015
    max_cvar_pct: float = 0.025
    max_sector_exposure: float = 0.40  # 0.40 = 40% cap (relaxed from 35% to reduce infeasible optimizer fallbacks)
    max_single_position: float = 0.10
    max_crypto_position: float = 0.20
    max_net_exposure: float = 1.0
    max_gross_exposure: float = 2.0
    max_beta_exposure: float = 0.5
    max_correlation: float = 0.7
    min_diversification_ratio: float = 0.3


@dataclass
class RiskBreach:
    limit_name: str
    current_value: float
    limit_value: float
    severity: str
    message: str


@dataclass
class RiskReport:
    timestamp: datetime
    portfolio_value: float
    var: VaRReport
    exposure: ExposureReport
    pnl_today: float
    return_today: float
    volatility_1d: float
    volatility_20d: float
    sharpe_estimate: float
    max_drawdown_current: float
    breaches: list[RiskBreach] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp.isoformat(),
            "portfolio_value": self.portfolio_value,
            "var": self.var.to_dict(),
            "exposure": self.exposure.to_dict(),
            "pnl_today": self.pnl_today,
            "return_today": self.return_today,
            "volatility_1d": self.volatility_1d,
            "volatility_20d": self.volatility_20d,
            "sharpe_estimate": self.sharpe_estimate,
            "max_drawdown_current": self.max_drawdown_current,
            "breaches": [
                {
                    "limit_name": b.limit_name,
                    "current_value": b.current_value,
                    "limit_value": b.limit_value,
                    "severity": b.severity,
                    "message": b.message,
                }
                for b in self.breaches
            ],
        }


class RiskEngine:
    def __init__(
        self,
        limits: Optional[RiskLimits] = None,
        sector_map: Optional[dict[str, str]] = None,
        seed: int | None = None,
    ):
        self.limits = limits or RiskLimits()
        # Override with runtime env values if set
        import os
        if os.getenv("MLCOUNCIL_MAX_CRYPTO_POSITION_SIZE"):
            self.limits.max_crypto_position = float(os.getenv("MLCOUNCIL_MAX_CRYPTO_POSITION_SIZE"))
        if os.getenv("MLCOUNCIL_MAX_POSITION_SIZE"):
            self.limits.max_single_position = float(os.getenv("MLCOUNCIL_MAX_POSITION_SIZE"))
        self.sector_map = sector_map or load_sector_map()
        self.seed = seed
        self._returns_history: Optional[pd.DataFrame] = None
        self._equity_curve: Optional[pd.Series] = None
        self._peak_equity: float = 0
        self._warned_unknown_tickers: set[str] = set()
        self._correlation_scale: float = 1.0

    def _resolve_sector(self, position: Position) -> str:
        explicit_sector = (position.sector or "").strip()
        if explicit_sector and explicit_sector not in {"Unknown", "Other"}:
            return explicit_sector

        sector = self.sector_map.get(position.symbol)
        if sector:
            return sector

        if position.symbol not in self._warned_unknown_tickers:
            logger.warning(
                "Unknown sector mapping for ticker %s; defaulting to Other",
                position.symbol,
            )
            self._warned_unknown_tickers.add(position.symbol)
        return "Other"

    def compute_var_historical(
        self,
        returns: pd.Series,
        portfolio_value: float,
        confidence: float = 0.99,
        horizon: int = 1,
    ) -> tuple[float, float]:
        if len(returns) < 30:
            return 0.0, 0.0

        scaled_returns = returns * np.sqrt(horizon)
        var_pct = np.percentile(scaled_returns, (1 - confidence) * 100)
        cvar_pct = scaled_returns[scaled_returns <= var_pct].mean()

        var_dollar = abs(var_pct) * portfolio_value
        cvar_dollar = abs(cvar_pct) * portfolio_value if not np.isnan(cvar_pct) else var_dollar * 1.5

        return var_dollar, cvar_dollar

    def compute_var_parametric(
        self,
        returns: pd.Series,
        portfolio_value: float,
        confidence: float = 0.99,
        horizon: int = 1,
    ) -> tuple[float, float]:
        if len(returns) < 30:
            return 0.0, 0.0

        mu = returns.mean() * horizon
        sigma = returns.std() * np.sqrt(horizon)

        z_score = _norm.ppf(confidence)

        var_pct = mu - z_score * sigma
        # Closed-form parametric CVaR for a Gaussian: E[L | L > VaR]
        # = mu - sigma * phi(z) / (1 - confidence), where phi is the standard normal PDF.
        cvar_pct = mu - sigma * _norm.pdf(z_score) / (1 - confidence)

        var_dollar = abs(var_pct) * portfolio_value
        cvar_dollar = abs(cvar_pct) * portfolio_value

        return var_dollar, cvar_dollar

    def compute_var_monte_carlo(
        self,
        returns: pd.DataFrame,
        weights: dict[str, float],
        portfolio_value: float,
        n_simulations: int = 10000,
        confidence: float = 0.99,
        horizon: int = 1,
        seed: int | None = None,
        tail_dof: int | None = 50,
        stress_replay: bool = False,
    ) -> MonteCarloVaRResult:
        """Multivariate Monte Carlo VaR / ES with multi-step daily paths.

        Simulates ``n_simulations`` paths of ``horizon`` daily portfolio
        returns (see :meth:`simulate_daily_returns`) and takes the empirical
        quantiles of the pathwise-compounded P&L at the end of the horizon:

        - **Covariance**: Ledoit-Wolf shrinkage (sklearn), replacing the old
          ridge + eigenvalue-clipping regularizer.
        - **Multi-step**: daily paths are compounded pathwise, so vol
          clustering (GARCH(1,1) / DCC(1,1) daily covariance dynamics when
          available) breaks the naive ``sqrt(horizon)`` scaling.
        - **t-copula**: ``tail_dof`` controls the innovation tails. The
          default ``tail_dof=50`` is practically Gaussian; ``tail_dof=5`` is
          heavy-tailed; ``tail_dof=None`` (or ``>= 10_000``) is exactly
          Gaussian. Sampling is ``X = mu + sigma * sqrt((nu-2)/nu) * Z`` with
          ``Z ~ t_nu(0, R)``, ``R`` the Ledoit-Wolf correlation. The t-copula
          has lower-tail dependence ``lambda_L = 2 * t_{nu+1}(
          -sqrt((nu+1)(1-rho)/(1+rho)))`` (0 for the Gaussian copula, ~0.21
          for ``nu=5, rho=0.5``).
        - **Stress replay** (``stress_replay=True``): correlation stress
          ``Sigma* = D (rho* 11' + (1-rho*) I) D`` with ``rho* = 0.9`` plus a
          top-k eigenvalue shock ``(1 + s_k)`` with ``s_k = 0.5``, applied to
          every daily covariance matrix.

        Returns a :class:`MonteCarloVaRResult` with VaR and expected
        shortfall (ES = mean of the losses beyond the VaR quantile) both as
        fractions of the portfolio value and in dollars.
        """
        tickers = list(weights.keys())
        available_tickers = [t for t in tickers if t in returns.columns]

        if len(available_tickers) < 2:
            return MonteCarloVaRResult(0.0, 0.0, 0.0, 0.0)

        aligned = returns[available_tickers].copy()
        aligned = aligned.dropna(how="any")
        if len(aligned) < 2:
            return MonteCarloVaRResult(0.0, 0.0, 0.0, 0.0)

        effective_seed = self.seed if seed is None else seed
        w = np.array([weights.get(t, 0.0) for t in available_tickers], dtype=float)

        paths = self.simulate_daily_returns(
            aligned,
            n_simulations=n_simulations,
            horizon=horizon,
            seed=effective_seed,
            tail_dof=tail_dof,
            stress_replay=stress_replay,
        )
        # Pathwise compounding: product of (1 + daily return) per path - 1.
        compounded = np.prod(1.0 + paths, axis=1) - 1.0
        simulated_pnl = (compounded @ w) * portfolio_value

        var_neg = np.percentile(simulated_pnl, (1 - confidence) * 100)
        tail = simulated_pnl[simulated_pnl <= var_neg]
        es_neg = float(tail.mean()) if len(tail) else var_neg

        var_dollar = abs(float(var_neg))
        es_dollar = abs(es_neg)
        var_pct = var_dollar / portfolio_value if portfolio_value > 0 else 0.0
        es_pct = es_dollar / portfolio_value if portfolio_value > 0 else 0.0
        return MonteCarloVaRResult(
            var_pct=var_pct,
            var_dollar=var_dollar,
            cvar_pct=es_pct,
            cvar_dollar=es_dollar,
            es_pct=es_pct,
            es_dollar=es_dollar,
            n_simulations=n_simulations,
            horizon=horizon,
        )

    @staticmethod
    def simulate_daily_returns(
        returns: pd.DataFrame,
        n_simulations: int = 10000,
        horizon: int = 1,
        seed: int | None = None,
        tail_dof: int | None = 50,
        covariance: np.ndarray | None = None,
        mean: np.ndarray | None = None,
        stress_replay: bool = False,
    ) -> np.ndarray:
        """Simulate multi-step daily return paths ``(n_simulations, horizon, n_assets)``.

        Each path compounds ``horizon`` daily draws so portfolio risk can be
        read from the empirical distribution of the compounded P&L. The daily
        covariance is:

        1. the DCC(1,1) conditional covariance (GARCH(1,1) vols + EWMA
           correlation, as in ``council/risk/covariance_dynamic.py``) when
           ``arch`` is installed;
        2. otherwise a GARCH(1,1) vol forecast around the constant
           Ledoit-Wolf correlation: persistence ``phi = alpha + beta`` is the
           AR(1) coefficient of squared returns and ``omega = sigma_bar^2
           (1 - phi)``, seeded from the last 20 squared returns;
        3. otherwise the constant Ledoit-Wolf covariance.

        Innovations are drawn from a t-copula with ``tail_dof`` degrees of
        freedom: ``X = mu + sigma * sqrt((nu-2)/nu) * Z`` with
        ``Z ~ t_nu(0, R)``. ``tail_dof=None`` (or ``>= 10_000``) means
        ``nu = infinity``, i.e. the Gaussian copula.
        """
        n_assets = returns.shape[1]
        rng = np.random.default_rng(seed)
        if covariance is None:
            covariance = RiskEngine._ledoit_wolf_covariance(returns)
        if mean is None:
            mean = returns.mean().to_numpy(dtype=float)
        covariance = RiskEngine._make_psd(np.asarray(covariance, dtype=float))
        if covariance.shape[0] != n_assets:
            raise ValueError(f"covariance must be {n_assets}x{n_assets}, got {covariance.shape}")

        daily_cov = RiskEngine._dcc_daily_covariance_path(returns, horizon)
        if daily_cov is None:
            daily_cov = RiskEngine._garch_daily_covariance_path(returns, horizon, covariance)
        if daily_cov is None:
            daily_cov = np.repeat(covariance[None, :, :], horizon, axis=0)

        if stress_replay:
            for t in range(horizon):
                daily_cov[t] = RiskEngine._stress_replay_covariance(daily_cov[t])

        gaussian = tail_dof is None or tail_dof >= 10_000
        paths = np.empty((n_simulations, horizon, n_assets))
        for t in range(horizon):
            sigma_t = daily_cov[t]
            vol_t = np.sqrt(np.maximum(np.diag(sigma_t), 1e-12))
            corr_t = sigma_t / np.outer(vol_t, vol_t)
            chol = np.linalg.cholesky(
                RiskEngine._make_psd(corr_t) + np.eye(n_assets) * 1e-12
            )
            z = rng.standard_normal((n_simulations, n_assets)) @ chol.T
            if not gaussian:
                # t-copula: Z = Y * sqrt(nu / W) with W ~ chi2(nu), then
                # rescale to unit variance: X = mu + sigma * sqrt((nu-2)/nu) Z.
                nu = float(tail_dof)
                chi2 = rng.chisquare(nu, size=n_simulations)
                z = z * np.sqrt((nu - 2.0) / chi2)[:, None]
            paths[:, t, :] = mean[None, :] + z * vol_t[None, :]
        return paths

    @staticmethod
    def _ledoit_wolf_covariance(returns: pd.DataFrame) -> np.ndarray:
        """Ledoit-Wolf shrunk covariance from a wide return panel.

        Replaces the previous ridge + eigenvalue-clipping regularizer: LW
        shrinkage is asymptotically optimal under Frobenius loss and keeps
        the sample dependence structure while guaranteeing positive
        definiteness. Falls back to the (PSD-clipped) sample covariance when
        sklearn is unavailable.
        """
        values = returns.to_numpy(dtype=float)
        sample_cov = np.cov(values, rowvar=False)
        if returns.shape[1] <= 1 or len(returns) < 5:
            return RiskEngine._make_psd(sample_cov)
        try:
            from sklearn.covariance import LedoitWolf

            cov = LedoitWolf().fit(values).covariance_
        except Exception:
            cov = sample_cov
        return RiskEngine._make_psd(cov)

    @staticmethod
    def _make_psd(matrix: np.ndarray, eps: float = 1e-8) -> np.ndarray:
        """Symmetrize and clip eigenvalues to keep a matrix positive definite."""
        arr = np.asarray(matrix, dtype=float)
        if arr.ndim == 0:
            arr = arr.reshape(1, 1)
        sym = (arr + arr.T) / 2.0
        eigvals, eigvecs = np.linalg.eigh(sym)
        eigvals = np.maximum(eigvals, eps)
        return eigvecs @ np.diag(eigvals) @ eigvecs.T

    @staticmethod
    def _dcc_daily_covariance_path(
        returns: pd.DataFrame, horizon: int
    ) -> np.ndarray | None:
        """Daily conditional covariance matrices (``horizon x n x n``) via DCC(1,1).

        Replays the GARCH(1,1) + EWMA-correlation recursion used by
        ``council/covariance_dynamic.DCCEstimator`` (see
        ``docs/math-drilldown`` section 1.3, Step 1):

        .. math::

            sigma^2_{i,t} = omega_i + alpha_i e^2_{i,t-1} + beta_i sigma^2_{i,t-1}
            Q_t = (1-a-b) Qbar + a e_{t-1} e'_{t-1} + b Q_{t-1}
            R_t = diag(Q_t)^{-1/2} Q_t diag(Q_t)^{-1/2}
            Sigma_t = D_t R_t D_t

        Returns None when ``arch`` is unavailable or the fit fails; callers
        then fall back to the constant Ledoit-Wolf covariance with GARCH vol
        dynamics (multi-step compounding still applies).
        """
        try:
            from arch import arch_model
            from council.risk.covariance_dynamic import DCCEstimator
        except Exception:
            return None
        n = returns.shape[1]
        if n < 2 or len(returns) < 30:
            return None
        try:
            dcc = DCCEstimator()
            a, b = dcc.a, dcc.b
            z_cols = []
            sig2_cols = []
            garch_params = []
            scale = 100.0
            for ticker in returns.columns:
                series = returns[ticker].dropna() * scale
                am = arch_model(series, vol="Garch", p=1, q=1, rescale=False)
                res = am.fit(disp="off", show_warning=False)
                cv = np.maximum(
                    np.asarray(res.conditional_volatility, dtype=float) / scale,
                    1e-8,
                )
                resid = np.asarray(res.resid, dtype=float) / scale
                z_cols.append(pd.Series(resid / cv, index=series.index, name=ticker))
                sig2_cols.append(pd.Series(cv**2, index=series.index, name=ticker))
                params = np.asarray(res.params, dtype=float)
                garch_params.append((params[0] / scale**2, params[1], params[2]))
            zdf = pd.concat(z_cols, axis=1).dropna(how="any")
            if len(zdf) < 5:
                return None
            z = zdf.to_numpy(dtype=float)
            q_bar = np.cov(z, rowvar=False)
            q_t = q_bar.copy()
            for t in range(1, len(z)):
                e = np.outer(z[t - 1], z[t - 1])
                q_t = (1.0 - a - b) * q_bar + a * e + b * q_t
            sig2 = np.asarray([s.iloc[-1] for s in sig2_cols], dtype=float)
            params = np.asarray(garch_params, dtype=float)
            path = np.empty((horizon, n, n))
            for t in range(horizon):
                d_inv = np.diag(1.0 / np.sqrt(np.maximum(np.diag(q_t), 1e-12)))
                r_t = d_inv @ q_t @ d_inv
                d_t = np.diag(np.sqrt(np.maximum(sig2, 1e-12)))
                path[t] = d_t @ r_t @ d_t
                # Conditional forecasts: E[e_t e'_t] = R_t, E[e^2_t] = sigma^2_t.
                q_t = (1.0 - a - b) * q_bar + a * r_t + b * q_t
                sig2 = params[:, 0] + (params[:, 1] + params[:, 2]) * sig2
            return path
        except Exception as exc:
            logger.debug("DCC daily covariance path unavailable: %s", exc)
            return None

    @staticmethod
    def _garch_daily_covariance_path(
        returns: pd.DataFrame,
        horizon: int,
        covariance: np.ndarray,
    ) -> np.ndarray | None:
        """GARCH(1,1) daily vol forecasts around a constant correlation.

        Method-of-moments calibration (no ``arch`` dependency): persistence
        ``phi = alpha + beta`` is the AR(1) coefficient of the squared
        returns and ``omega = sigma_bar^2 (1 - phi)`` with ``sigma_bar^2``
        the sample variance. The current vol state is seeded with a
        persistence-weighted blend of the last 20 squared returns and the
        unconditional variance (``sig2_0 = phi * recent + (1-phi) *
        sigma_bar^2``), so iid data stays at the unconditional level while a
        recent vol cluster drives the forecast. The conditional forecast
        ``sigma^2_{t+1} = omega + phi * sigma^2_t`` then decays toward the
        unconditional level over the horizon (breaking ``sqrt(horizon)``
        scaling). Correlation is constant at the Ledoit-Wolf level:
        ``Sigma_t = D_t R_lw D_t``.
        """
        n = returns.shape[1]
        if n < 2 or len(returns) < 30:
            return None
        values = returns.to_numpy(dtype=float)
        r2 = values**2
        phis = []
        for j in range(n):
            prev, curr = r2[:-1, j], r2[1:, j]
            if prev.std() > 0 and curr.std() > 0:
                phi = float(np.corrcoef(prev, curr)[0, 1])
            else:
                phi = 0.0
            phis.append(float(np.clip(phi, 0.1, 0.99)))
        phi = np.asarray(phis, dtype=float)
        sigma_bar2 = np.maximum(values.var(axis=0, ddof=1), 1e-12)
        omega = sigma_bar2 * (1.0 - phi)
        recent = np.maximum(np.mean(r2[-20:], axis=0), 1e-12)
        sig2 = phi * recent + (1.0 - phi) * sigma_bar2
        std_lw = np.sqrt(np.maximum(np.diag(covariance), 1e-12))
        corr_lw = covariance / np.outer(std_lw, std_lw)
        path = np.empty((horizon, n, n))
        for t in range(horizon):
            d_t = np.diag(np.sqrt(sig2))
            path[t] = d_t @ corr_lw @ d_t
            sig2 = omega + phi * sig2
        return path

    @staticmethod
    def _stress_replay_covariance(
        cov: np.ndarray,
        rho_star: float = 0.9,
        shock: float = 0.5,
        top_k: int = 2,
    ) -> np.ndarray:
        """Apply the stress-replay shocks to a covariance matrix (drill-down 1.3, Step 4).

        (a) Correlation stress: ``Sigma* = D (rho* 11' + (1-rho*) I) D`` with
            ``rho* = 0.9`` — all pairwise correlations forced to 0.9, vols
            unchanged.
        (b) Eigenvalue stress: the top-k principal components of the stressed
            matrix are scaled by ``(1 + s_k)`` with ``s_k = 0.5``:
            ``Sigma* = V diag(lambda (1 + s)) V'``.
        """
        cov = RiskEngine._make_psd(np.asarray(cov, dtype=float))
        std = np.sqrt(np.maximum(np.diag(cov), 1e-12))
        n = cov.shape[0]
        corr_stress = rho_star * np.ones((n, n)) + (1.0 - rho_star) * np.eye(n)
        stressed = np.outer(std, std) * corr_stress
        eigvals, eigvecs = np.linalg.eigh(stressed)
        order = np.argsort(eigvals)[::-1]
        eigvals = eigvals[order]
        eigvecs = eigvecs[:, order]
        k = max(1, min(top_k, n))
        shocks = np.ones(n)
        shocks[:k] = 1.0 + shock
        stressed = eigvecs @ np.diag(eigvals * shocks) @ eigvecs.T
        return RiskEngine._make_psd(stressed)

    def compute_var(
        self,
        returns: pd.DataFrame,
        positions: list[Position],
        portfolio_value: float,
        method: str = "historical",
        confidence: float = 0.99,
        seed: int | None = None,
        *,
        tail_dof: int | None = None,
        stress_replay: bool = False,
    ) -> VaRReport:
        """Compute 1/5/10-day VaR and CVaR for a portfolio.

        ``method`` is one of ``historical`` (default), ``parametric``,
        ``generative`` or ``monte_carlo`` (catch-all). ``tail_dof`` and
        ``stress_replay`` only affect the Monte Carlo method: Student-t
        innovation tails (``None`` = Gaussian, the legacy default; 50 =
        practically Gaussian; 5 = heavy tails) and optional stress replay.
        """
        weights = {p.symbol: p.market_value / portfolio_value for p in positions}

        if method == "historical":
            portfolio_returns = self._compute_portfolio_returns(returns, weights)
            var_1d, cvar_1d = self.compute_var_historical(portfolio_returns, portfolio_value, confidence, 1)
            var_5d, cvar_5d = self.compute_var_historical(portfolio_returns, portfolio_value, confidence, 5)
            var_10d, cvar_10d = self.compute_var_historical(portfolio_returns, portfolio_value, confidence, 10)
        elif method == "parametric":
            portfolio_returns = self._compute_portfolio_returns(returns, weights)
            var_1d, cvar_1d = self.compute_var_parametric(portfolio_returns, portfolio_value, confidence, 1)
            var_5d, cvar_5d = self.compute_var_parametric(portfolio_returns, portfolio_value, confidence, 5)
            var_10d, cvar_10d = self.compute_var_parametric(portfolio_returns, portfolio_value, confidence, 10)
        elif method == "generative":
            from council.risk.generative_stress import GenerativeStressEngine

            tickers = [t for t in weights if t in returns.columns]
            wide = returns[tickers] if tickers else returns
            stress = GenerativeStressEngine(n_scenarios=10_000).sample_scenarios(wide)
            var_1d = abs(stress.var_95)
            # CVaR from the empirical tail of the already-sampled scenarios
            # (mean of the draws beyond the 95% VaR) instead of the Gaussian
            # ES/VaR ratio 1.25 — the simulation was decorative before
            # (drill-down finding #1).
            tail = stress.scenarios[stress.scenarios <= stress.var_95]
            cvar_1d = abs(float(tail.mean())) if len(tail) else var_1d * 1.25
            var_5d, cvar_5d = var_1d * np.sqrt(5), cvar_1d * np.sqrt(5)
            var_10d, cvar_10d = var_1d * np.sqrt(10), cvar_1d * np.sqrt(10)
        else:
            result_1d = self.compute_var_monte_carlo(
                returns,
                weights,
                portfolio_value,
                10000,
                confidence,
                1,
                seed=seed,
                tail_dof=tail_dof,
                stress_replay=stress_replay,
            )
            var_1d, cvar_1d = result_1d.var_dollar, result_1d.cvar_dollar
            result_5d = self.compute_var_monte_carlo(
                returns,
                weights,
                portfolio_value,
                10000,
                confidence,
                5,
                seed=seed,
                tail_dof=tail_dof,
                stress_replay=stress_replay,
            )
            var_5d, cvar_5d = result_5d.var_dollar, result_5d.cvar_dollar
            result_10d = self.compute_var_monte_carlo(
                returns,
                weights,
                portfolio_value,
                10000,
                confidence,
                10,
                seed=seed,
                tail_dof=tail_dof,
                stress_replay=stress_replay,
            )
            var_10d, cvar_10d = result_10d.var_dollar, result_10d.cvar_dollar

        return VaRReport(
            var_1d=var_1d,
            var_5d=var_5d,
            var_10d=var_10d,
            cvar_1d=cvar_1d,
            cvar_5d=cvar_5d,
            cvar_10d=cvar_10d,
            var_method=method,
            confidence_level=confidence,
            portfolio_value=portfolio_value,
        )

    def _compute_portfolio_returns(
        self,
        returns: pd.DataFrame,
        weights: dict[str, float],
    ) -> pd.Series:
        portfolio_ret = pd.Series(0.0, index=returns.index)
        for symbol, weight in weights.items():
            if symbol in returns.columns:
                portfolio_ret += returns[symbol] * weight
        return portfolio_ret

    def compute_exposure(
        self,
        positions: list[Position],
        portfolio_value: float,
    ) -> ExposureReport:
        if portfolio_value <= 0:
            portfolio_value = sum(p.market_value for p in positions)

        sector_values: dict[str, float] = {}
        long_value = 0.0
        short_value = 0.0
        total_long = 0.0
        total_short = 0.0

        for pos in positions:
            sector = self._resolve_sector(pos)
            if sector not in sector_values:
                sector_values[sector] = 0.0
            sector_values[sector] += pos.market_value

            if pos.market_value > 0:
                long_value += pos.market_value
            else:
                short_value += abs(pos.market_value)

            total_long += abs(pos.market_value)

        gross_exposure = (long_value + short_value) / portfolio_value
        net_exposure = (long_value - short_value) / portfolio_value

        sector_weights = {k: v / portfolio_value for k, v in sector_values.items()}

        concentration = {
            pos.symbol: pos.market_value / portfolio_value
            for pos in sorted(positions, key=lambda p: p.market_value, reverse=True)
        }

        beta_exposure = sum(pos.beta * (pos.market_value / portfolio_value) for pos in positions)

        return ExposureReport(
            total_market_value=portfolio_value,
            net_exposure=net_exposure,
            gross_exposure=gross_exposure,
            sector_exposure=sector_values,
            factor_exposure={"beta": beta_exposure},
            concentration=concentration,
            beta_exposure=beta_exposure,
            sector_weights=sector_weights,
        )

    def check_limits(self, report: RiskReport) -> list[RiskBreach]:
        breaches = []
        var_pct = report.var.var_1d / report.portfolio_value if report.portfolio_value > 0 else 0
        if var_pct > self.limits.max_var_pct:
            breaches.append(RiskBreach(
                limit_name="VaR Limit",
                current_value=var_pct,
                limit_value=self.limits.max_var_pct,
                severity="HIGH" if var_pct > self.limits.max_var_pct * 1.5 else "MEDIUM",
                message=f"1-day VaR ({var_pct:.2%}) exceeds limit ({self.limits.max_var_pct:.2%})",
            ))

        cvar_pct = report.var.cvar_1d / report.portfolio_value if report.portfolio_value > 0 else 0
        if cvar_pct > self.limits.max_cvar_pct:
            breaches.append(RiskBreach(
                limit_name="CVaR Limit",
                current_value=cvar_pct,
                limit_value=self.limits.max_cvar_pct,
                severity="HIGH" if cvar_pct > self.limits.max_cvar_pct * 1.5 else "MEDIUM",
                message=f"1-day CVaR ({cvar_pct:.2%}) exceeds limit ({self.limits.max_cvar_pct:.2%})",
            ))

        for sector, exposure in report.exposure.sector_weights.items():
            if exposure > self.limits.max_sector_exposure:
                breaches.append(RiskBreach(
                    limit_name="Sector Exposure",
                    current_value=exposure,
                    limit_value=self.limits.max_sector_exposure,
                    severity="HIGH" if exposure > self.limits.max_sector_exposure * 1.3 else "MEDIUM",
                    message=f"Sector {sector} exposure ({exposure:.2%}) exceeds limit ({self.limits.max_sector_exposure:.2%})",
                ))

        for symbol, exposure in report.exposure.concentration.items():
            from execution.alpaca_adapter import AlpacaLiveNode
            limit = self.limits.max_crypto_position if AlpacaLiveNode._is_crypto(symbol) else self.limits.max_single_position
            if exposure > limit:
                breaches.append(RiskBreach(
                    limit_name="Position Limit",
                    current_value=exposure,
                    limit_value=limit,
                    severity="HIGH",
                    message=f"Position {symbol} ({exposure:.2%}) exceeds limit ({limit:.2%})",
                ))

        if abs(report.exposure.net_exposure) > self.limits.max_net_exposure:
            breaches.append(RiskBreach(
                limit_name="Net Exposure",
                current_value=report.exposure.net_exposure,
                limit_value=self.limits.max_net_exposure,
                severity="HIGH",
                message=f"Net exposure ({report.exposure.net_exposure:.2%}) exceeds limit ({self.limits.max_net_exposure:.2%})",
            ))

        if report.exposure.gross_exposure > self.limits.max_gross_exposure:
            breaches.append(RiskBreach(
                limit_name="Gross Exposure",
                current_value=report.exposure.gross_exposure,
                limit_value=self.limits.max_gross_exposure,
                severity="MEDIUM",
                message=f"Gross exposure ({report.exposure.gross_exposure:.2%}) exceeds limit ({self.limits.max_gross_exposure:.2%})",
            ))

        if abs(report.exposure.beta_exposure) > self.limits.max_beta_exposure:
            breaches.append(RiskBreach(
                limit_name="Beta Exposure",
                current_value=report.exposure.beta_exposure,
                limit_value=self.limits.max_beta_exposure,
                severity="MEDIUM",
                message=f"Beta exposure ({report.exposure.beta_exposure:.2f}) exceeds limit ({self.limits.max_beta_exposure:.2f})",
            ))

        return breaches

    def detect_correlation_breakdown(
        self,
        cov: pd.DataFrame,
        threshold: float | None = None,
        max_correlated_pairs_ratio: float | None = None,
    ) -> float | None:
        """Detect correlation breakdown from covariance matrix.

        Computes the correlation matrix from the given covariance matrix.
        If the ratio of highly-correlated pairs (|correlation| > threshold)
        exceeds max_correlated_pairs_ratio, returns a scale factor (0.5)
        to reduce exposure. Returns None if no breakdown detected.

        Env overrides:
            MLCOUNCIL_CORRELATION_THRESHOLD (default 0.7)
            MLCOUNCIL_MAX_CORRELATED_PAIRS (default 0.4)
        """
        if cov.empty or cov.shape[0] < 2:
            return None

        if threshold is None:
            threshold = float(
                os.getenv("MLCOUNCIL_CORRELATION_THRESHOLD", "0.7")
            )
        if max_correlated_pairs_ratio is None:
            max_correlated_pairs_ratio = float(
                os.getenv("MLCOUNCIL_MAX_CORRELATED_PAIRS", "0.4")
            )

        n = cov.shape[0]
        # Compute correlation matrix from covariance
        std_diag = np.sqrt(np.diag(cov))
        # Guard against zero-variance assets
        if (std_diag <= 0).any():
            return None
        d_inv = np.diag(1.0 / std_diag)
        corr = d_inv @ cov.values @ d_inv

        # Extract upper triangle (excluding diagonal)
        i_upper = np.triu_indices(n, k=1)
        corr_vals = corr[i_upper]

        total_pairs = len(corr_vals)
        if total_pairs == 0:
            return None

        high_corr_count = int(np.sum(np.abs(corr_vals) > threshold))
        ratio = high_corr_count / total_pairs

        logger.debug(
            "Correlation breakdown check: %d/%d pairs > |%.2f| (ratio=%.3f, limit=%.3f)",
            high_corr_count,
            total_pairs,
            threshold,
            ratio,
            max_correlated_pairs_ratio,
        )

        if ratio > max_correlated_pairs_ratio:
            logger.warning(
                "Correlation breakdown detected: %.1f%% of pairs exceed |%.2f| "
                "(limit %.1f%%) — applying 0.5x scale factor",
                ratio * 100,
                threshold,
                max_correlated_pairs_ratio * 100,
            )
            return 0.5

        return None

    def check_limits_from_weights(
        self,
        weights: pd.Series,
        cov: pd.DataFrame,
        portfolio_value: float = 100_000.0,
    ) -> tuple[bool, list[str]]:
        """Lightweight pre-trade risk check using only weights and covariance.

        Returns (limits_ok, list_of_breach_descriptions).
        Suitable for use in the daily pipeline before orders are submitted.
        """
        sector_map = self.sector_map
        breaches: list[str] = []

        # Sector exposure check
        sector_weights: dict[str, float] = {}
        for ticker, w in weights.items():
            sector = sector_map.get(ticker, "Other")
            sector_weights[sector] = sector_weights.get(sector, 0.0) + abs(w)
        for sector, exposure in sector_weights.items():
            if exposure > self.limits.max_sector_exposure:
                breaches.append(
                    f"Sector Exposure: {sector} at {exposure:.2%} "
                    f"(limit {self.limits.max_sector_exposure:.2%})"
                )

        # Single position concentration check
        for ticker, w in weights.items():
            limit = (
                self.limits.max_crypto_position
                if ticker in {"BTCUSD", "ETHUSD"}
                else self.limits.max_single_position
            )
            if abs(w) > limit:
                breaches.append(
                    f"Position Limit: {ticker} at {abs(w):.2%} (limit {limit:.2%})"
                )

        # VaR approximation from covariance
        if len(weights) > 1 and not cov.empty:
            w_arr = weights.reindex(cov.columns).fillna(0.0).values
            cov_aligned = cov.reindex(
                index=weights.index, columns=weights.index
            ).fillna(0.0)
            port_var = float(w_arr @ cov_aligned.values @ w_arr)
            if port_var > 0:
                daily_vol = np.sqrt(port_var)
                # Approximate 1-day 99% VaR (parametric, Gaussian)
                var_pct = 2.326 * daily_vol  # z = 2.326 for 99%
                if var_pct > self.limits.max_var_pct:
                    breaches.append(
                        f"VaR Limit: {var_pct:.2%} (limit {self.limits.max_var_pct:.2%})"
                    )

        # Correlation breakdown check
        scale = self.detect_correlation_breakdown(cov)
        if scale is not None:
            self._correlation_scale = scale
            breaches.append(
                f"Correlation Breakdown: {scale:.0%} scale factor applied "
                f"(excessive cross-asset correlation detected)"
            )
        else:
            self._correlation_scale = 1.0

        return (len(breaches) == 0, breaches)

    @property
    def correlation_scale_factor(self) -> float:
        """Return the current correlation scale factor.

        Returns 1.0 normally (no breakdown). If a correlation breakdown
        has been detected via check_limits_from_weights, returns the
        reduction factor (e.g. 0.5).
        """
        return self._correlation_scale

    def compute_full_risk(
        self,
        positions: list[Position],
        returns: pd.DataFrame,
        portfolio_value: float,
        equity_curve: Optional[pd.Series] = None,
        var_method: str = "historical",
        seed: int | None = None,
        *,
        tail_dof: int | None = None,
        stress_replay: bool = False,
    ) -> RiskReport:
        portfolio_value = portfolio_value or sum(p.market_value for p in positions)

        var_report = self.compute_var(
            returns,
            positions,
            portfolio_value,
            method=var_method,
            seed=seed,
            tail_dof=tail_dof,
            stress_replay=stress_replay,
        )
        exposure_report = self.compute_exposure(positions, portfolio_value)

        today_return = 0.0
        pnl_today = 0.0
        if equity_curve is not None and len(equity_curve) >= 2:
            today_return = equity_curve.pct_change().iloc[-1]
            pnl_today = portfolio_value * today_return

        vol_1d = returns.iloc[-1].std() if len(returns) > 0 else 0.0
        vol_20d = returns.tail(20).std().mean() if len(returns) >= 20 else vol_1d

        sharpe = 0.0
        if vol_20d > 0:
            sharpe = (returns.mean().mean() * 252) / (vol_20d * np.sqrt(252))

        peak = max(self._peak_equity, portfolio_value)
        self._peak_equity = peak
        current_dd = (peak - portfolio_value) / peak if peak > 0 else 0

        report = RiskReport(
            timestamp=datetime.now(timezone.utc),
            portfolio_value=portfolio_value,
            var=var_report,
            exposure=exposure_report,
            pnl_today=pnl_today,
            return_today=today_return,
            volatility_1d=vol_1d,
            volatility_20d=vol_20d,
            sharpe_estimate=sharpe,
            max_drawdown_current=current_dd,
        )

        report.breaches = self.check_limits(report)
        return report

    def save_report(self, report: RiskReport, date: Optional[str] = None) -> Path:
        date = date or datetime.now().strftime("%Y-%m-%d")
        report_file = RISK_DIR / f"risk_report_{date}.json"
        report_file.write_text(json.dumps(report.to_dict(), indent=2))
        return report_file

    def load_report(self, date: str) -> Optional[RiskReport]:
        report_file = RISK_DIR / f"risk_report_{date}.json"
        if not report_file.exists():
            return None

        try:
            data = json.loads(report_file.read_text())
            var_data = data["var"]
            var = VaRReport(**var_data)
            exp_data = data["exposure"]
            exposure = ExposureReport(**exp_data)
            return RiskReport(
                timestamp=datetime.fromisoformat(data["timestamp"]),
                portfolio_value=data["portfolio_value"],
                var=var,
                exposure=exposure,
                pnl_today=data["pnl_today"],
                return_today=data["return_today"],
                volatility_1d=data["volatility_1d"],
                volatility_20d=data["volatility_20d"],
                sharpe_estimate=data["sharpe_estimate"],
                max_drawdown_current=data["max_drawdown_current"],
            )
        except Exception:
            return None


def create_positions_from_broker(positions_df: pd.DataFrame) -> list[Position]:
    sector_map = load_sector_map()
    positions = []
    for _, row in positions_df.iterrows():
        symbol = row["symbol"]
        positions.append(Position(
            symbol=symbol,
            quantity=int(row["qty"]),
            avg_price=float(row["avg_price"]),
            current_price=float(row.get("current_price", row["avg_price"])),
            sector=sector_map.get(symbol, "Other"),
            beta=1.0,
        ))
    return positions


def load_sector_map(path: Path | None = None) -> dict[str, str]:
    sector_map_path = path or _DEFAULT_SECTOR_MAP_PATH
    if sector_map_path.exists():
        with sector_map_path.open() as handle:
            data = json.load(handle)
        return {str(symbol): str(sector) for symbol, sector in data.items()}

    from data.features.sector_exposure import SECTOR_MAP

    return dict(SECTOR_MAP)
