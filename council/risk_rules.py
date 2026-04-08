"""Position risk rules and portfolio drawdown protection.

Features:
- Stop-loss per position (hard stop at N% loss)
- Trailing stop (moves up as position gains)
- Time-based exit (max holding period)
- Portfolio-level drawdown protection
- Bear regime de-risking

Usage:
    from council.risk_rules import PositionRiskRules, DrawdownProtection

    risk = PositionRiskRules()
    exits = risk.compute_exits(positions, prices)

    protection = DrawdownProtection()
    cash_fraction, reason = protection.should_de-risk(
        portfolio_value=95000,
        peak_value=100000,
        regime="bear"
    )
"""

from __future__ import annotations

import pandas as pd
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional


@dataclass
class PositionRiskResult:
    symbol: str
    reason: str
    suggested_action: str
    urgency: str


@dataclass
class ExitSignal:
    symbol: str
    reason: str
    current_return: float
    stop_price: Optional[float] = None


class PositionRiskRules:
    def __init__(
        self,
        stop_loss_pct: float = 0.05,
        trailing_stop_pct: float = 0.10,
        max_holding_days: int = 20,
        profit_take_pct: float = 0.20,
    ):
        self.stop_loss_pct = stop_loss_pct
        self.trailing_stop_pct = trailing_stop_pct
        self.max_holding_days = max_holding_days
        self.profit_take_pct = profit_take_pct

    def compute_exits(
        self,
        positions: pd.DataFrame,
        prices: pd.Series,
        reference_date: Optional[datetime] = None,
    ) -> list[ExitSignal]:
        """Compute exit signals for all positions.

        Parameters
        ----------
        reference_date:
            The date to use as "now" for time-based exits. Pass an explicit
            value in backtests to avoid non-deterministic behaviour. Defaults
            to ``datetime.now()`` when not provided (live trading only).
        """
        if reference_date is None:
            reference_date = datetime.now()
        exits = []
        for _, pos in positions.iterrows():
            symbol = pos["symbol"]
            if symbol not in prices.index:
                continue

            entry_price = pos["entry_price"]
            current_price = prices[symbol]
            # Default entry_date = reference_date - max_holding_days so that a
            # position with no recorded entry immediately triggers a time exit.
            entry_date = pos.get("entry_date", reference_date - timedelta(days=self.max_holding_days))

            ret = (current_price - entry_price) / entry_price

            if ret <= -self.stop_loss_pct:
                exits.append(ExitSignal(
                    symbol=symbol,
                    reason="stop_loss",
                    current_return=ret,
                    stop_price=entry_price * (1 - self.stop_loss_pct),
                ))
                continue

            peak_price = pos.get("peak_price", entry_price)
            if peak_price < current_price:
                peak_price = current_price

            trailing_stop = peak_price * (1 - self.trailing_stop_pct)
            if current_price < trailing_stop and ret > 0:
                exits.append(ExitSignal(
                    symbol=symbol,
                    reason="trailing_stop",
                    current_return=ret,
                    stop_price=trailing_stop,
                ))
                continue

            days_held = (reference_date - entry_date).days
            if days_held >= self.max_holding_days and ret > 0:
                exits.append(ExitSignal(
                    symbol=symbol,
                    reason="time_limit",
                    current_return=ret,
                ))
                continue

            if ret >= self.profit_take_pct:
                exits.append(ExitSignal(
                    symbol=symbol,
                    reason="profit_take",
                    current_return=ret,
                ))

        return exits

    def should_add_to_position(
        self,
        symbol: str,
        current_position_value: float,
        proposed_addition: float,
        entry_price: float,
        current_price: float,
        max_position_value: float,
    ) -> tuple[bool, str]:
        new_position_value = current_position_value + proposed_addition

        if new_position_value > max_position_value:
            return False, f"Would exceed max position ${max_position_value}"

        ret = (current_price - entry_price) / entry_price
        if ret < -0.03:
            return False, "Position is underwater >3%, do not add"

        if current_position_value / max_position_value > 0.8:
            return False, "Already at >80% of max position"

        return True, "OK"


class DrawdownProtection:
    def __init__(
        self,
        max_drawdown: float = 0.15,
        reduce_exposure_at: float = 0.10,
        min_cash_fraction: float = 0.20,
        bear_regime_cash: float = 0.20,
    ):
        self.max_drawdown = max_drawdown
        self.reduce_exposure_at = reduce_exposure_at
        self.min_cash_fraction = min_cash_fraction
        self.bear_regime_cash = bear_regime_cash

    def should_de_risk(
        self,
        portfolio_value: float,
        peak_value: float,
        regime: Optional[str] = None,
    ) -> tuple[float, str]:
        drawdown = (peak_value - portfolio_value) / peak_value if peak_value > 0 else 0

        # Hard limits take priority over soft regime signals.
        if drawdown > self.max_drawdown:
            return 0.50, "max_drawdown_breach"

        if drawdown > self.reduce_exposure_at:
            ratio = (drawdown - self.reduce_exposure_at) / (self.max_drawdown - self.reduce_exposure_at)
            cash_fraction = self.min_cash_fraction + ratio * (0.50 - self.min_cash_fraction)
            return min(cash_fraction, 0.50), "drawdown_warning"

        if regime == "bear":
            return self.bear_regime_cash, "bear_regime"

        return 0.0, "no_action"

    def compute_target_cash(
        self,
        current_cash: float,
        portfolio_value: float,
        peak_value: float,
        regime: Optional[str] = None,
    ) -> tuple[float, str]:
        target_cash_fraction, reason = self.should_de_risk(portfolio_value, peak_value, regime)
        target_cash = portfolio_value * target_cash_fraction
        return target_cash, reason

    def should_rebalance(
        self,
        current_weights: pd.Series,
        target_weights: pd.Series,
        threshold: float = 0.02,
    ) -> bool:
        diff = (current_weights - target_weights).abs()
        return diff.max() > threshold


class PortfolioRiskMonitor:
    def __init__(
        self,
        position_rules: Optional[PositionRiskRules] = None,
        drawdown_protection: Optional[DrawdownProtection] = None,
    ):
        self.position_rules = position_rules or PositionRiskRules()
        self.drawdown_protection = drawdown_protection or DrawdownProtection()
        self._peak_value = None
        self._peak_date = None

    def update_peak(self, portfolio_value: float) -> None:
        if self._peak_value is None or portfolio_value > self._peak_value:
            self._peak_value = portfolio_value
            self._peak_date = datetime.now()

    def get_risk_status(
        self,
        portfolio_value: float,
        current_positions: pd.DataFrame,
        prices: pd.Series,
        regime: Optional[str] = None,
    ) -> dict:
        self.update_peak(portfolio_value)

        exits = self.position_rules.compute_exits(current_positions, prices)
        cash_fraction, reason = self.drawdown_protection.should_de_risk(
            portfolio_value, self._peak_value, regime
        )

        return {
            "peak_value": self._peak_value,
            "current_value": portfolio_value,
            "drawdown": (self._peak_value - portfolio_value) / self._peak_value if self._peak_value else 0,
            "regime": regime,
            "target_cash_fraction": cash_fraction,
            "risk_reason": reason,
            "exit_signals": [
                {"symbol": e.symbol, "reason": e.reason, "return": e.current_return}
                for e in exits
            ],
            "has_exit_signals": len(exits) > 0,
            "should_de_risk": cash_fraction > 0,
        }
