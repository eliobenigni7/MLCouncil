"""Portfolio construction: convert alpha signals into optimised weights.

Optimisation problem (cvxpy)
----------------------------
    maximize   (alpha ⊙ multipliers)' w
    subject to
        sum(w)           == 1            budget constraint
        w                >= 0            long-only (PoC)
        w                <= max_position concentration limit
        |w - w_curr|_1   <= max_turnover one-way turnover cap
        w' Σ w           <= σ_daily²    portfolio vol cap

Post-processing
---------------
After solving, weights below ``min_position`` are zeroed and the remainder
is renormalized so the budget constraint is still satisfied exactly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger


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
    """

    def __init__(self) -> None:
        self.max_position: float = 0.10
        self.min_position: float = 0.01
        self.max_turnover: float = 0.30
        self.long_only: bool = True
        self.max_vol_ann: float = 0.20

    # ------------------------------------------------------------------
    # optimize
    # ------------------------------------------------------------------

    def optimize(
        self,
        alpha_signals: pd.Series,
        position_multipliers: pd.Series,
        current_weights: pd.Series,
        returns_covariance: pd.DataFrame,
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

        Returns
        -------
        pd.Series(index=ticker, values=target_weight) summing to 1.0.
        """
        import cvxpy as cp

        tickers = alpha_signals.index.tolist()
        n = len(tickers)

        # Scale alpha by conformal position multipliers
        mults = position_multipliers.reindex(alpha_signals.index).fillna(1.0)
        effective_alpha = (alpha_signals * mults).reindex(tickers).fillna(0.0).values

        # Align current weights (zero for tickers not yet held)
        w_curr = current_weights.reindex(tickers).fillna(0.0).values

        # Covariance — symmetrize and add small diagonal for numerical stability
        cov_raw = (
            returns_covariance
            .reindex(index=tickers, columns=tickers)
            .fillna(0.0)
            .values
        )
        cov = (cov_raw + cov_raw.T) / 2 + np.eye(n) * 1e-6

        # Daily volatility cap
        max_vol_daily = self.max_vol_ann / np.sqrt(252)

        w = cp.Variable(n, name="weights")

        objective = cp.Maximize(effective_alpha @ w)

        constraints: list = [
            cp.sum(w) == 1.0,
            w <= self.max_position,
            cp.norm1(w - w_curr) <= self.max_turnover,
            cp.quad_form(w, cov) <= max_vol_daily ** 2,
        ]
        if self.long_only:
            constraints.append(w >= 0.0)

        prob = cp.Problem(objective, constraints)

        # Try solvers in order of preference
        solved = False
        for solver in (None, cp.SCS):          # None → cvxpy auto-selects
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

        # Post-process: drop micro-positions below min_position
        weights[weights < self.min_position] = 0.0
        total = weights.sum()
        if total < 1e-9:
            weights = np.ones(n) / n
        else:
            weights /= total

        return pd.Series(weights, index=tickers, name="target_weight")

    # ------------------------------------------------------------------
    # compute_orders
    # ------------------------------------------------------------------

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
