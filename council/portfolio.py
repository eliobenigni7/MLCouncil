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

from council.transaction_costs import TransactionCostModel
from data.features.sector_exposure import (
    SECTOR_MAP,
    UNIQUE_SECTORS,
    compute_effective_sector_cap,
    compute_sector_exposures,
    compute_beta_vector,
    get_ticker_sector,
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
        import os
        self.max_position: float = float(os.getenv("MLCOUNCIL_MAX_POSITION_SIZE", "0.10"))
        self.min_position: float = 0.01
        self.max_turnover: float = float(os.getenv("MLCOUNCIL_MAX_TURNOVER", "0.30"))
        self.long_only: bool = True
        self.max_vol_ann: float = 0.20
        self.sector_cap: float = 0.25
        # Beta constraint enabled by default: the council generates pure
        # cross-sectional alpha signals (z-scored, regime-agnostic), so the
        # portfolio should not carry unintended systematic market exposure.
        # The constraint caps |portfolio_beta| <= max_beta_exposure when
        # market_returns is provided to optimize(); no-op otherwise.
        self.beta_neutral: bool = True
        self.max_beta_exposure: float = 0.30
        self.commission_bps: float = 1.0
        self.slippage_bps: float = 3.0
        self.tc_lambda: float = 1.0
        # Crypto settings
        self.crypto_enabled: bool = os.getenv("MLCOUNCIL_CRYPTO_ENABLED", "false").lower() == "true"
        self.max_crypto_position: float = float(os.getenv("MLCOUNCIL_MAX_CRYPTO_POSITION_SIZE", "0.20"))
        self.max_crypto_turnover: float = float(os.getenv("MLCOUNCIL_MAX_CRYPTO_TURNOVER", "0.40"))
        self.cost_model = TransactionCostModel(
            commission_bps=self.commission_bps,
            slippage_bps=self.slippage_bps,
        )

    def compute_transaction_cost(
        self,
        w_old: np.ndarray,
        w_new: np.ndarray,
        portfolio_value: float = 1.0,
    ) -> float:
        self.cost_model = TransactionCostModel(
            commission_bps=self.commission_bps,
            slippage_bps=self.slippage_bps,
        )
        return self.cost_model.estimate_cost_from_weights(
            w_old,
            w_new,
            portfolio_value=portfolio_value,
        )

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
        effective_sector_cap = compute_effective_sector_cap(
            tickers,
            base_sector_cap=self.sector_cap,
            max_position=self.max_position,
        )

        w = cp.Variable(n, name="weights")

        alpha_objective = effective_alpha @ w

        turnover = cp.norm1(w - w_curr) / 2
        self.cost_model = TransactionCostModel(
            commission_bps=self.commission_bps,
            slippage_bps=self.slippage_bps,
        )
        tc_cost = turnover * self.cost_model.total_cost_bps / 10000
        objective = cp.Maximize(alpha_objective - self.tc_lambda * tc_cost)

        constraints: list = [
            cp.sum(w) == 1.0,
            w <= self.max_position,
            cp.quad_form(w, cov) <= max_vol_daily ** 2,
        ]
        if np.abs(w_curr).sum() > 1e-9:
            constraints.append(cp.norm1(w - w_curr) <= self.max_turnover)
        if self.long_only:
            constraints.append(w >= 0.0)

        if effective_sector_cap < 1.0:
            for sector in UNIQUE_SECTORS:
                sector_tickers = [t for t in tickers if SECTOR_MAP.get(t) == sector]
                if sector_tickers:
                    sector_indices = [tickers.index(t) for t in sector_tickers]
                    sector_exposure = cp.sum(w[sector_indices])
                    constraints.append(sector_exposure <= effective_sector_cap)

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
                "Returning sector-aware fallback."
            )
            return self._feasible_fallback_weights(
                alpha_signals=alpha_signals.reindex(tickers).fillna(0.0),
                sector_cap=effective_sector_cap,
            )

        weights = np.clip(w.value, 0.0, None)

        weights[weights < self.min_position] = 0.0
        total = weights.sum()
        if total < 1e-9:
            weights = np.ones(n) / n
        else:
            weights /= total

        # Re-normalising after zeroing small positions can push surviving
        # weights above max_position, violating the CVXPY constraint.
        # Example: 10 positions at 10% each; zero one → re-norm gives 11.1%.
        # Fix: clip again and re-normalise a second time.
        if weights.max() > self.max_position + 1e-9:
            weights = np.clip(weights, 0.0, self.max_position)
            total = weights.sum()
            if total > 1e-9:
                weights /= total

        return pd.Series(weights, index=tickers, name="target_weight")

    def optimize_with_crypto(
        self,
        alpha_signals: pd.Series,
        position_multipliers: pd.Series,
        current_weights: pd.Series,
        returns_covariance: pd.DataFrame,
        market_returns: pd.Series = None,
        prices: pd.Series = None,
    ) -> pd.Series:
        """Optimize portfolio handling equity and crypto separately.

        Crypto is optimized with its own limit settings (max_crypto_position,
        max_crypto_turnover) if MLCOUNCIL_CRYPTO_ENABLED=true. Otherwise crypto
        tickers are included in the standard optimization.
        """
        from execution.alpaca_adapter import AlpacaLiveNode

        crypto_tickers = [t for t in alpha_signals.index if AlpacaLiveNode._is_crypto(t)]
        equity_tickers = [t for t in alpha_signals.index if t not in crypto_tickers]

        results = {}

        # Equity optimization
        if equity_tickers:
            eq_signals = alpha_signals[equity_tickers]
            eq_mults = position_multipliers.reindex(eq_signals.index)
            eq_curr = current_weights.reindex(eq_signals.index)
            eq_cov = returns_covariance.reindex(index=equity_tickers, columns=equity_tickers)
            eq_market = market_returns if market_returns is not None else None
            eq_prices = prices.reindex(eq_signals.index) if prices is not None else None
            results = self.optimize(
                eq_signals, eq_mults, eq_curr, eq_cov, eq_market, eq_prices
            ).to_dict()

        # Crypto optimization
        if crypto_tickers and self.crypto_enabled:
            saved_max_pos = self.max_position
            saved_max_turn = self.max_turnover
            self.max_position = self.max_crypto_position
            self.max_turnover = self.max_crypto_turnover

            cr_signals = alpha_signals[crypto_tickers]
            cr_mults = position_multipliers.reindex(cr_signals.index)
            cr_curr = current_weights.reindex(cr_signals.index)
            cr_cov = returns_covariance.reindex(index=crypto_tickers, columns=crypto_tickers)
            cr_market = market_returns
            cr_prices = prices.reindex(cr_signals.index) if prices is not None else None
            cr_result = self.optimize(
                cr_signals, cr_mults, cr_curr, cr_cov, cr_market, cr_prices
            ).to_dict()

            self.max_position = saved_max_pos
            self.max_turnover = saved_max_turn
            results.update(cr_result)
        elif crypto_tickers:
            # Crypto disabled — include in equity optimization with standard limits
            cr_signals = alpha_signals[crypto_tickers]
            cr_mults = position_multipliers.reindex(cr_signals.index)
            cr_curr = current_weights.reindex(cr_signals.index)
            cr_cov = returns_covariance.reindex(index=crypto_tickers, columns=crypto_tickers)
            cr_result = self.optimize(
                cr_signals, cr_mults, cr_curr, cr_cov
            ).to_dict()
            results.update(cr_result)

        return pd.Series(results, name="target_weight")

    def _feasible_fallback_weights(
        self,
        *,
        alpha_signals: pd.Series,
        sector_cap: float,
    ) -> pd.Series:
        tickers = alpha_signals.sort_values(ascending=False).index.tolist()
        weights = pd.Series(0.0, index=tickers, dtype=float, name="target_weight")
        sector_weights: dict[str, float] = {}
        remaining = 1.0

        while remaining > 1e-9:
            progress = False
            for ticker in tickers:
                current_weight = float(weights[ticker])
                ticker_room = self.max_position - current_weight
                if ticker_room <= 1e-9:
                    continue

                sector = get_ticker_sector(ticker)
                sector_room = sector_cap - sector_weights.get(sector, 0.0)
                if sector_room <= 1e-9:
                    continue

                increment = min(ticker_room, sector_room, remaining)
                if increment <= 1e-9:
                    continue

                weights[ticker] = current_weight + increment
                sector_weights[sector] = sector_weights.get(sector, 0.0) + increment
                remaining -= increment
                progress = True
                if remaining <= 1e-9:
                    break

            if not progress:
                break

        if remaining > 1e-6:
            logger.warning(
                "Sector-aware fallback could not deploy the full budget. "
                f"Remaining cash fraction: {remaining:.2%}"
            )

        total = weights.sum()
        if total <= 1e-9:
            return pd.Series(np.ones(len(tickers)) / len(tickers), index=tickers, name="target_weight")

        return (weights / total).rename("target_weight")

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
