"""Portfolio construction: convert alpha signals into optimised weights.

Optimisation problem (cvxpy)
----------------------------
    maximize   (alpha ⊙ multipliers)' w - tc_penalty * turnover
    subject to
        sum(w)           == 1            budget constraint
        w                >= 0            long-only (PoC)
        w                <= max_position concentration limit
        |w - w_curr|_1   <= max_turnover one-way turnover cap
        w' Σ w           <= σ_daily²    portfolio vol cap
        sector[w]        <= sector_cap   sector exposure cap

Post-processing
---------------
After solving, weights below ``min_position`` are zeroed and the remainder
is renormalized so the budget constraint is still satisfied exactly.

Transaction Cost Model
----------------------
TC = sum(|dw_i|) * (commission_bps + slippage_bps) / 10000 * portfolio_value
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger

from data.features.sector_exposure import (
    SECTOR_MAP,
    UNIQUE_SECTORS,
    compute_sector_exposures,
    compute_beta_vector,
)


class PortfolioConstructor:
    """Convert alpha signals and conformal multipliers into portfolio weights.

    Attributes
    ----------
    max_position : float
        Maximum weight for any single ticker (default 10 %).
    min_position : float
        Post-processing floor: weights below this value are zeroed (default 1 %).
    max_turnover : float
        Maximum one-way portfolio turnover per day (default 30 %).
    long_only : bool
        Enforce non-negative weights (PoC: True).
    max_vol_ann : float
        Annualised portfolio volatility cap (default 20 %).
    sector_cap : float
        Maximum weight per sector (default 25 %).
    beta_neutral : bool
        If True, enforce beta neutrality (default False).
    max_beta_exposure : float
        Maximum absolute portfolio beta if beta_neutral (default 0.3).
    commission_bps : float
        Commission in basis points (default 1.0 bps).
    slippage_bps : float
        Slippage in basis points (default 3.0 bps).
    tc_lambda : float
        Transaction cost penalty weight (default 1.0).
    """

    def __init__(self) -> None:
        self.max_position: float = 0.10
        self.min_position: float = 0.01
        self.max_turnover: float = 0.30
        self.long_only: bool = True
        self.max_vol_ann: float = 0.20
        self.sector_cap: float = 0.25
        self.beta_neutral: bool = False
        self.max_beta_exposure: float = 0.30
        self.commission_bps: float = 1.0
        self.slippage_bps: float = 3.0
        self.tc_lambda: float = 1.0

    def compute_transaction_cost(
        self,
        w_old: np.ndarray,
        w_new: np.ndarray,
        portfolio_value: float = 1.0,
    ) -> float:
        turnover = np.abs(w_new - w_old).sum() / 2
        cost_bps = self.commission_bps + self.slippage_bps
        tc_cost = turnover * cost_bps / 10000 * portfolio_value
        return tc_cost

    def optimize(
        self,
        alpha_signals: pd.Series,
        position_multipliers: pd.Series,
        current_weights: pd.Series,
        returns_covariance: pd.DataFrame,
        market_returns: pd.Series = None,
        prices: pd.Series = None,
    ) -> pd.Series:
        """Solve the constrained mean-variance optimisation.

        Parameters
        ----------
        alpha_signals:
            z-scored council signals indexed by ticker.
        position_multipliers:
            Conformal multipliers in [0.2, 2.0] indexed by ticker.
        current_weights:
            Current portfolio weights (sum to 1, may be empty for day 1).
        returns_covariance:
            Daily returns covariance matrix (ticker × ticker).
        market_returns:
            Series of market returns for beta calculation (optional).
        prices:
            Current prices for TC estimation (optional).

        Returns
        -------
        pd.Series(index=ticker, values=target_weight) summing to 1.0.
        """
        tickers = alpha_signals.index.tolist()
        n = len(tickers)

        try:
            import cvxpy as cp
        except ModuleNotFoundError:
            logger.warning(
                "cvxpy not installed. Returning equal-weight fallback."
            )
            return pd.Series(np.ones(n) / n, index=tickers, name="target_weight")

        mults = position_multipliers.reindex(alpha_signals.index).fillna(1.0)
        effective_alpha = (alpha_signals * mults).reindex(tickers).fillna(0.0).values

        w_curr = current_weights.reindex(tickers).fillna(0.0).values

        cov_raw = (
            returns_covariance
            .reindex(index=tickers, columns=tickers)
            .fillna(0.0)
            .values
        )
        cov = (cov_raw + cov_raw.T) / 2 + np.eye(n) * 1e-6

        max_vol_daily = self.max_vol_ann / np.sqrt(252)

        w = cp.Variable(n, name="weights")

        alpha_objective = effective_alpha @ w

        turnover = cp.norm1(w - w_curr) / 2
        tc_cost = turnover * (self.commission_bps + self.slippage_bps) / 10000
        objective = cp.Maximize(alpha_objective - self.tc_lambda * tc_cost)

        constraints: list = [
            cp.sum(w) == 1.0,
            w <= self.max_position,
            cp.norm1(w - w_curr) <= self.max_turnover,
            cp.quad_form(w, cov) <= max_vol_daily ** 2,
        ]
        if self.long_only:
            constraints.append(w >= 0.0)

        if self.sector_cap < 1.0:
            for sector in UNIQUE_SECTORS:
                sector_tickers = [t for t in tickers if SECTOR_MAP.get(t) == sector]
                if sector_tickers:
                    sector_indices = [tickers.index(t) for t in sector_tickers]
                    sector_exposure = cp.sum(w[sector_indices])
                    constraints.append(sector_exposure <= self.sector_cap)

        if self.beta_neutral and market_returns is not None:
            beta_vec = compute_beta_vector(
                returns_covariance[[t for t in tickers if t in returns_covariance.columns]],
                market_returns,
            ).reindex(tickers).fillna(1.0).values
            portfolio_beta = w @ beta_vec
            constraints.append(cp.abs(portfolio_beta) <= self.max_beta_exposure)

        prob = cp.Problem(objective, constraints)

        solved = False
        for solver in (None, cp.SCS):
            try:
                if solver is None:
                    prob.solve(verbose=False)
                else:
                    prob.solve(solver=solver, verbose=False)
                if prob.status in {"optimal", "optimal_inaccurate"} and w.value is not None:
                    solved = True
                    break
            except Exception as exc:
                logger.debug(f"Solver {solver} failed: {exc}")

        if not solved or w.value is None:
            logger.warning(
                f"Portfolio optimisation failed (status={prob.status!r}). "
                "Returning equal-weight fallback."
            )
            return pd.Series(np.ones(n) / n, index=tickers, name="target_weight")

        weights = np.clip(w.value, 0.0, None)

        weights[weights < self.min_position] = 0.0
        total = weights.sum()
        if total < 1e-9:
            weights = np.ones(n) / n
        else:
            weights /= total

        return pd.Series(weights, index=tickers, name="target_weight")

    def compute_orders(
        self,
        target_weights: pd.Series,
        current_weights: pd.Series,
        portfolio_value: float,
    ) -> pd.DataFrame:
        """Convert weight deltas into a trade list.

        Parameters
        ----------
        target_weights:
            Target portfolio weights (sum to 1).
        current_weights:
            Current portfolio weights (sum to 1).
        portfolio_value:
            Total portfolio value in USD.

        Returns
        -------
        pd.DataFrame with columns: ticker, direction, quantity, target_weight.
        ``quantity`` is in USD.  Trades smaller than $1 are omitted.
        """
        all_tickers = target_weights.index.union(current_weights.index)
        target = target_weights.reindex(all_tickers).fillna(0.0)
        current = current_weights.reindex(all_tickers).fillna(0.0)
        delta = target - current

        rows = []
        for ticker in all_tickers:
            dw = float(delta[ticker])
            usd = abs(dw) * portfolio_value
            if usd < 1.0:
                continue
            rows.append(
                {
                    "ticker": ticker,
                    "direction": "buy" if dw > 0 else "sell",
                    "quantity": round(usd, 2),
                    "target_weight": float(target[ticker]),
                }
            )

        if not rows:
            return pd.DataFrame(
                columns=["ticker", "direction", "quantity", "target_weight"]
            )
        return pd.DataFrame(
            rows, columns=["ticker", "direction", "quantity", "target_weight"]
        )
