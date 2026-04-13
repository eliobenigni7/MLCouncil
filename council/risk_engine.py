"""Advanced Risk Management System for MLCouncil.

Provides institutional-grade risk management:
- Value at Risk (VaR) - Historical, Parametric, Monte Carlo
- Conditional VaR (CVaR / Expected Shortfall)
- Real-time position exposure tracking
- Sector and factor exposure limits
- Correlation stress testing
- Greeks approximation for equity portfolio

Usage:
    from council.risk_engine import RiskEngine, RiskReport

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

_ROOT = Path(__file__).parents[1]
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
    max_var_pct: float = 0.02
    max_cvar_pct: float = 0.035
    max_sector_exposure: float = 0.25
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
    ) -> tuple[float, float]:
        tickers = list(weights.keys())
        available_tickers = [t for t in tickers if t in returns.columns]

        if len(available_tickers) < 2:
            return 0.0, 0.0

        mean_returns = returns[available_tickers].mean()
        cov_matrix = returns[available_tickers].cov()

        w = np.array([weights.get(t, 0) for t in available_tickers])
        mu = mean_returns.values @ w
        sigma = np.sqrt(w @ cov_matrix.values @ w)

        rng = np.random.default_rng(self.seed if seed is None else seed)
        simulated_returns = rng.normal(mu * horizon, sigma * np.sqrt(horizon), n_simulations)
        simulated_pnl = simulated_returns * portfolio_value

        var_pct = np.percentile(simulated_pnl, (1 - confidence) * 100)
        cvar_pct = simulated_pnl[simulated_pnl <= var_pct].mean()

        return abs(var_pct), abs(cvar_pct) if not np.isnan(cvar_pct) else abs(var_pct) * 1.5

    def compute_var(
        self,
        returns: pd.DataFrame,
        positions: list[Position],
        portfolio_value: float,
        method: str = "historical",
        confidence: float = 0.99,
        seed: int | None = None,
    ) -> VaRReport:
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
        else:
            var_1d, cvar_1d = self.compute_var_monte_carlo(
                returns,
                weights,
                portfolio_value,
                10000,
                confidence,
                1,
                seed=seed,
            )
            var_5d, cvar_5d = self.compute_var_monte_carlo(
                returns,
                weights,
                portfolio_value,
                10000,
                confidence,
                5,
                seed=seed,
            )
            var_10d, cvar_10d = self.compute_var_monte_carlo(
                returns,
                weights,
                portfolio_value,
                10000,
                confidence,
                10,
                seed=seed,
            )

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

    def compute_full_risk(
        self,
        positions: list[Position],
        returns: pd.DataFrame,
        portfolio_value: float,
        equity_curve: Optional[pd.Series] = None,
        var_method: str = "historical",
        seed: int | None = None,
    ) -> RiskReport:
        portfolio_value = portfolio_value or sum(p.market_value for p in positions)

        var_report = self.compute_var(
            returns,
            positions,
            portfolio_value,
            method=var_method,
            seed=seed,
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
