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

from council.transaction_costs import (
    TransactionCostModel,
    get_default_commission_bps,
    get_default_slippage_bps,
)
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
        Commission in basis points (default from MLCOUNCIL_COMMISSION_BPS, 0.0 bps).
    slippage_bps : float
        Slippage in basis points (default from MLCOUNCIL_SLIPPAGE_BPS, 3.0 bps).
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
        # Defaults come from runtime env for parity with backtests.
        self.commission_bps: float = get_default_commission_bps()
        self.slippage_bps: float = get_default_slippage_bps()
        self.tc_lambda: float = 1.0
        # Minimum absolute z-score to enter a position (filters noise).
        self.min_signal_strength: float = float(os.getenv("MLCOUNCIL_MIN_SIGNAL_STRENGTH", "0.20"))
        # Drawdown circuit breaker: scale exposure when portfolio loses too much.
        self.max_drawdown_threshold: float = float(os.getenv("MLCOUNCIL_MAX_DRAWDOWN_PCT", "0.07"))
        # Crypto settings
        self.crypto_enabled: bool = os.getenv("MLCOUNCIL_CRYPTO_ENABLED", "false").lower() == "true"
        self.max_crypto_position: float = float(os.getenv("MLCOUNCIL_MAX_CRYPTO_POSITION_SIZE", "0.20"))
        self.max_crypto_turnover: float = float(os.getenv("MLCOUNCIL_MAX_CRYPTO_TURNOVER", "0.40"))
        self.cost_model = TransactionCostModel(
            commission_bps=self.commission_bps,
            slippage_bps=self.slippage_bps,
        )

    def _get_portfolio_tier(self, portfolio_value: float) -> dict:
        """Return size-adaptive constraints based on portfolio value.

        Small portfolios concentrate into the top signals to reduce execution
        friction and improve signal-to-noise ratio.  Large portfolios use the
        full universe with standard constraints.

        Returns dict with keys:
            n_positions   – number of tickers to include (None = full universe)
            max_position  – maximum weight per ticker
            min_position  – minimum weight floor (post-processing)
            max_turnover  – one-way daily turnover cap
            min_trade_usd – minimum USD value per order
        """
        if portfolio_value < 5_000:
            return {
                "n_positions": 3,
                "max_position": 0.45,
                "min_position": 0.05,
                "max_turnover": 0.50,
                "min_trade_usd": 50.0,
                "budget_fraction": 1.0,
            }
        elif portfolio_value < 25_000:
            return {
                "n_positions": 5,
                "max_position": 0.25,
                "min_position": 0.03,
                "max_turnover": 0.40,
                "min_trade_usd": 25.0,
                "budget_fraction": 1.0,
            }
        elif portfolio_value < 100_000:
            return {
                "n_positions": 10,
                "max_position": 0.15,
                "min_position": 0.02,
                "max_turnover": 0.35,
                "min_trade_usd": 10.0,
                "budget_fraction": 1.0,
            }
        else:
            # Reserve ~15% budget for intraday moves by limiting positions
            # and keeping some headroom below the position cap.
            return {
                "n_positions": 12,
                "max_position": max(self.max_position, 0.13),
                "min_position": self.min_position,
                "max_turnover": self.max_turnover,
                "min_trade_usd": max(1.0, portfolio_value * 0.0005),
                "budget_fraction": 0.85,
            }

    @staticmethod
    def _get_budget_fraction(tier: dict) -> float:
        fraction = float(tier.get("budget_fraction", 1.0))
        if fraction < 0.0:
            return 0.0
        if fraction > 1.0:
            return 1.0
        return fraction

    def apply_drawdown_scale(
        self,
        target_weights: pd.Series,
        portfolio_return_5d: float,
    ) -> pd.Series:
        """Scale down positions when recent drawdown exceeds threshold.

        If the 5-day portfolio return is worse than -max_drawdown_threshold,
        exposure is scaled proportionally so that a portfolio at twice the
        threshold receives half the normal position sizes.

        Parameters
        ----------
        target_weights:
            Optimised target weights before circuit-breaker adjustment.
        portfolio_return_5d:
            Realised 5-day portfolio return (negative = loss).

        Returns
        -------
        Scaled weights (no re-normalisation).
        """
        if portfolio_return_5d >= -self.max_drawdown_threshold:
            return target_weights

        # Linear scale: at threshold → scale=1; at 2× threshold → scale=0.5.
        excess = abs(portfolio_return_5d) - self.max_drawdown_threshold
        scale = max(0.25, 1.0 - excess / self.max_drawdown_threshold)
        logger.warning(
            f"Drawdown circuit breaker: 5d return={portfolio_return_5d:.2%}, "
            f"scaling exposure to {scale:.0%}"
        )
        return (target_weights * scale).rename("target_weight")

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

    @staticmethod
    def _project_to_capped_simplex(
        values: np.ndarray,
        budget_fraction: float,
        upper_bounds: np.ndarray,
    ) -> np.ndarray:
        """Project weights onto a capped simplex with exact budget and caps."""
        v = np.asarray(values, dtype=float).copy()
        u = np.asarray(upper_bounds, dtype=float).copy()
        if v.size == 0:
            return v
        v = np.where(np.isfinite(v), v, 0.0)
        u = np.where(np.isfinite(u), np.maximum(u, 0.0), 0.0)
        budget = float(np.clip(budget_fraction, 0.0, float(u.sum())))
        if budget <= 1e-12:
            return np.zeros_like(v)
        if float(u.sum()) <= budget + 1e-12:
            return u

        def clipped_sum(tau: float) -> float:
            return float(np.clip(v - tau, 0.0, u).sum())

        lo = float(np.min(v - u)) - 1.0
        hi = float(np.max(v)) + 1.0
        while clipped_sum(lo) < budget:
            lo -= max(1.0, abs(lo) + 1.0)
        while clipped_sum(hi) > budget:
            hi += max(1.0, abs(hi) + 1.0)

        for _ in range(100):
            mid = (lo + hi) / 2.0
            if clipped_sum(mid) > budget:
                lo = mid
            else:
                hi = mid

        return np.clip(v - hi, 0.0, u)


    def optimize(
        self,
        alpha_signals: pd.Series,
        position_multipliers: pd.Series,
        current_weights: pd.Series,
        returns_covariance: pd.DataFrame,
        market_returns: pd.Series = None,
        prices: pd.Series = None,
        portfolio_value: float = 100_000,
        days_since_last_rebalance: int = 999,
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
        portfolio_value:
            Total portfolio value in USD.  Used to select the sizing tier.
        days_since_last_rebalance:
            Trading days elapsed since the last rebalance.  Small portfolios
            require a minimum cool-down to avoid excessive turnover costs.

        Returns
        -------
        pd.Series(index=ticker, values=target_weight) summing to 1.0.
        """
        tier = self._get_portfolio_tier(portfolio_value)

        # Small-portfolio rebalance cool-down: skip if we rebalanced recently
        # and the expected turnover wouldn't justify the slippage cost.
        min_rebalance_days = 1 if tier["n_positions"] is None else (
            5 if portfolio_value < 25_000 else 3
        )
        if days_since_last_rebalance < min_rebalance_days and current_weights.abs().sum() > 1e-9:
            logger.debug(
                f"Skipping rebalance: only {days_since_last_rebalance}d since last "
                f"(min={min_rebalance_days}d for portfolio_value=${portfolio_value:,.0f})"
            )
            return current_weights.reindex(alpha_signals.index).fillna(0.0).rename("target_weight")

        # Gate: drop signals below minimum z-score strength (noise filter).
        alpha_signals = alpha_signals.copy()
        alpha_signals[alpha_signals.abs() < self.min_signal_strength] = 0.0

        # Tier: pre-filter to top-N tickers by |z-score| for small portfolios.
        if tier["n_positions"] is not None:
            top_n = min(tier["n_positions"], (alpha_signals != 0.0).sum())
            top_n = max(top_n, 1)
            top_tickers = alpha_signals.abs().nlargest(top_n).index
            alpha_signals = alpha_signals.reindex(top_tickers).fillna(0.0)

        effective_max_position = tier["max_position"]
        effective_min_position = tier["min_position"]
        effective_max_turnover = tier["max_turnover"]

        tickers = alpha_signals.index.tolist()
        n = len(tickers)

        budget_fraction = self._get_budget_fraction(tier)
        if n > 0:
            # Ensure the optimization remains feasible on narrow universes.
            # When only a few tickers survive filtering, strict max_position
            # plus reserved cash can make the budget impossible to deploy.
            min_required_position = budget_fraction / n
            if min_required_position > effective_max_position + 1e-9:
                if n < 12:
                    budget_fraction = 1.0
                    min_required_position = budget_fraction / n
                effective_max_position = min(
                    1.0,
                    max(effective_max_position, min_required_position),
                )

        effective_sector_cap = compute_effective_sector_cap(
            tickers,
            base_sector_cap=self.sector_cap,
            max_position=effective_max_position,
        )

        try:
            import cvxpy as cp
        except ModuleNotFoundError:
            logger.warning(
                "cvxpy not installed. Returning sector-aware fallback."
            )
            current_fallback = self._fallback_from_current_weights(
                tickers=tickers,
                current_weights=current_weights,
                budget_fraction=budget_fraction,
                max_position=effective_max_position,
                sector_cap=effective_sector_cap,
                max_turnover=effective_max_turnover,
            )
            if current_fallback is not None:
                return current_fallback
            return self._feasible_fallback_weights(
                alpha_signals=alpha_signals.reindex(tickers).fillna(0.0),
                sector_cap=effective_sector_cap,
                budget_fraction=budget_fraction,
                max_position=effective_max_position,
            )

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
        self.cost_model = TransactionCostModel(
            commission_bps=self.commission_bps,
            slippage_bps=self.slippage_bps,
        )
        tc_cost = turnover * self.cost_model.total_cost_bps / 10000
        objective = cp.Maximize(alpha_objective - self.tc_lambda * tc_cost)

        constraints: list = [
            cp.sum(w) == budget_fraction,
            w <= effective_max_position,
            cp.quad_form(w, cov) <= max_vol_daily ** 2,
        ]
        if np.abs(w_curr).sum() > 1e-9:
            constraints.append(cp.norm1(w - w_curr) <= effective_max_turnover)
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
                budget_fraction=budget_fraction,
                max_position=effective_max_position,
            )

        weights = np.clip(w.value, 0.0, None)
        weights = self._project_to_capped_simplex(
            weights,
            budget_fraction=budget_fraction,
            upper_bounds=np.full(n, effective_max_position, dtype=float),
        )

        return pd.Series(weights, index=tickers, name="target_weight")

    def optimize_with_crypto(
        self,
        alpha_signals: pd.Series,
        position_multipliers: pd.Series,
        current_weights: pd.Series,
        returns_covariance: pd.DataFrame,
        market_returns: pd.Series = None,
        prices: pd.Series = None,
        portfolio_value: float = 100_000,
    ) -> pd.Series:
        """Optimize portfolio handling equity and crypto separately.

        Crypto is optimized with its own limit settings (max_crypto_position,
        max_crypto_turnover) if MLCOUNCIL_CRYPTO_ENABLED=true. Otherwise crypto
        tickers are included in the standard optimization.
        """
        from execution.alpaca_adapter import AlpacaLiveNode

        crypto_tickers = [t for t in alpha_signals.index if AlpacaLiveNode._is_crypto(t)]
        equity_tickers = [t for t in alpha_signals.index if t not in crypto_tickers]
        overall_budget_fraction = self._get_budget_fraction(
            self._get_portfolio_tier(portfolio_value)
        )

        results: dict[str, float] = {}

        def _signal_share(tickers: list[str]) -> float:
            if not tickers:
                return 0.0
            total_abs = float(alpha_signals.abs().sum())
            if total_abs <= 1e-12:
                return float(len(tickers)) / float(len(alpha_signals)) if len(alpha_signals) else 0.0
            return float(alpha_signals.reindex(tickers).abs().sum()) / total_abs

        # Equity optimization
        if equity_tickers:
            eq_signals = alpha_signals[equity_tickers]
            eq_mults = position_multipliers.reindex(eq_signals.index)
            eq_curr = current_weights.reindex(eq_signals.index)
            eq_cov = returns_covariance.reindex(index=equity_tickers, columns=equity_tickers)
            eq_market = market_returns if market_returns is not None else None
            eq_prices = prices.reindex(eq_signals.index) if prices is not None else None
            eq_result = self.optimize(
                eq_signals,
                eq_mults,
                eq_curr,
                eq_cov,
                eq_market,
                eq_prices,
                portfolio_value=portfolio_value,
            ).to_dict()
            eq_share = _signal_share(equity_tickers)
            results.update({k: float(v) * eq_share for k, v in eq_result.items()})

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
                cr_signals,
                cr_mults,
                cr_curr,
                cr_cov,
                cr_market,
                cr_prices,
            ).to_dict()

            self.max_position = saved_max_pos
            self.max_turnover = saved_max_turn
            cr_share = _signal_share(crypto_tickers)
            results.update({k: float(v) * cr_share for k, v in cr_result.items()})
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

        combined = pd.Series(results, dtype=float, name="target_weight")
        total = float(combined.sum())
        if total > 1e-9:
            # Keep the overall budget as much as possible, but ensure the final
            # combined portfolio respects per-asset caps. This prevents a narrow
            # crypto universe from concentrating the whole budget into BTC/USD.
            upper_bounds = np.array(
                [
                    self.max_crypto_position if AlpacaLiveNode._is_crypto(ticker) else self.max_position
                    for ticker in combined.index
                ],
                dtype=float,
            )
            feasible_budget = min(total, float(upper_bounds.sum()))
            combined = pd.Series(
                self._project_to_capped_simplex(
                    combined.clip(lower=0.0).values,
                    budget_fraction=feasible_budget,
                    upper_bounds=upper_bounds,
                ),
                index=combined.index,
                name="target_weight",
            )
        return combined.sort_index()

    def _feasible_fallback_weights(
        self,
        *,
        alpha_signals: pd.Series,
        sector_cap: float,
        budget_fraction: float = 1.0,
        max_position: float | None = None,
    ) -> pd.Series:
        tickers = alpha_signals.sort_values(ascending=False).index.tolist()
        weights = pd.Series(0.0, index=tickers, dtype=float, name="target_weight")
        sector_weights: dict[str, float] = {}
        remaining = budget_fraction
        position_cap = self.max_position if max_position is None else max_position

        while remaining > 1e-9:
            progress = False
            for ticker in tickers:
                current_weight = float(weights[ticker])
                ticker_room = position_cap - current_weight
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
            fallback = np.ones(len(tickers)) / len(tickers)
            return pd.Series(fallback * budget_fraction, index=tickers, name="target_weight")

        return weights.rename("target_weight")

    def _fallback_from_current_weights(
        self,
        *,
        tickers: list[str],
        current_weights: pd.Series,
        budget_fraction: float,
        max_position: float,
        sector_cap: float,
        max_turnover: float,
    ) -> pd.Series | None:
        aligned = current_weights.reindex(tickers).fillna(0.0).astype(float)
        if self.long_only:
            aligned = aligned.clip(lower=0.0)

        total = float(aligned.sum())
        if total <= 1e-9:
            return None

        weights = aligned / total * budget_fraction
        if float(weights.max()) > max_position + 1e-9:
            weights = weights.clip(upper=max_position)
            clipped_total = float(weights.sum())
            if clipped_total <= 1e-9:
                return None
            weights = weights / clipped_total * budget_fraction

        if sector_cap < 1.0:
            exposures = compute_sector_exposures(weights)
            if bool((exposures > sector_cap + 1e-9).any()):
                return None

        turnover = float((weights - aligned).abs().sum())
        if turnover > max_turnover and turnover > 1e-9:
            blend = max(0.0, min(1.0, max_turnover / turnover))
            weights = aligned + (weights - aligned) * blend

        return weights.rename("target_weight")

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

        tier = self._get_portfolio_tier(portfolio_value)
        min_trade_usd = tier["min_trade_usd"]

        rows = []
        for ticker in all_tickers:
            dw = float(delta[ticker])
            usd = abs(dw) * portfolio_value
            if usd < min_trade_usd:
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
