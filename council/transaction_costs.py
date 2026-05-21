"""Shared transaction cost model for portfolio construction and backtests."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from council.cost_calibration import (
    DEFAULT_CALIBRATION_PATH,
    CalibrationArtifact,
    CalibrationHashError,
    TIER_BY_TICKER,
    load_calibration,
    ticker_tier,
)

logger = logging.getLogger(__name__)

_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_COMMISSION_BPS = 1.0
DEFAULT_SLIPPAGE_BPS = 3.0
DEFAULT_CONFIDENCE_FLOOR = 30


def _read_bps_env(key: str, default: float) -> float:
    raw = os.getenv(key)
    if raw is None or not raw.strip():
        return float(default)
    try:
        return float(raw)
    except ValueError:
        return float(default)


def get_default_commission_bps() -> float:
    return _read_bps_env("MLCOUNCIL_COMMISSION_BPS", DEFAULT_COMMISSION_BPS)


def get_default_slippage_bps() -> float:
    return _read_bps_env("MLCOUNCIL_SLIPPAGE_BPS", DEFAULT_SLIPPAGE_BPS)


def get_confidence_floor() -> int:
    raw = os.getenv("MLCOUNCIL_COST_CALIBRATION_CONFIDENCE_FLOOR")
    if raw is None or not raw.strip():
        return DEFAULT_CONFIDENCE_FLOOR
    try:
        return max(1, int(raw))
    except ValueError:
        return DEFAULT_CONFIDENCE_FLOOR


def get_calibration_path() -> Optional[Path]:
    """Return calibration JSON path, or None when calibration is disabled."""
    raw = os.getenv("MLCOUNCIL_COST_CALIBRATION_PATH")
    if raw is not None:
        if not raw.strip():
            return None
        return Path(raw)
    return DEFAULT_CALIBRATION_PATH


def get_active_calibration_version(path: Optional[Path] = None) -> str:
    """SHA-256 of the active calibration artifact, or empty when static-only."""
    calib_path = path if path is not None else get_calibration_path()
    if calib_path is None or not calib_path.exists():
        return ""
    try:
        artifact = load_calibration(calib_path)
        return artifact.version
    except CalibrationHashError as exc:
        logger.warning("Cost calibration manifest invalid: %s", exc)
        return ""


def resolve_slippage_bps(
    ticker: str,
    *,
    artifact: Optional[CalibrationArtifact] = None,
    confidence_floor: Optional[int] = None,
) -> float:
    """Blend static lookup slippage with calibrated kappa when sample allows."""
    lookup = estimate_slippage_bps(ticker)
    if artifact is None:
        return lookup

    floor = confidence_floor if confidence_floor is not None else get_confidence_floor()
    tier = ticker_tier(ticker)

    kappa = artifact.kappa_by_ticker.get(ticker)
    n = artifact.fill_count_by_ticker.get(ticker, 0)
    if kappa is None:
        kappa = artifact.kappa_by_tier.get(tier)
        n = artifact.fill_count_by_tier.get(tier, 0)

    if kappa is None or n <= 0:
        return lookup

    alpha = min(1.0, float(n) / float(floor))
    return float((1.0 - alpha) * lookup + alpha * kappa)


def build_slippage_bps_by_ticker(
    artifact: CalibrationArtifact,
    tickers: Optional[set[str]] = None,
) -> dict[str, float]:
    """Build per-ticker blended slippage for all known universe tickers."""
    universe = tickers if tickers is not None else set(TIER_BY_TICKER.keys())
    return {t: resolve_slippage_bps(t, artifact=artifact) for t in universe}


def estimate_slippage_bps(ticker: str, dollar_volume: float | None = None) -> float:
    """Estimate slippage in basis points from a static per-ticker liquidity table.

    This is a configurable heuristic lookup, not a calibrated Almgren-Chriss
    optimal-execution solver. An optional volume multiplier nudges illiquid names
    higher when ``dollar_volume`` is provided.
    """
    ILLIQUIDITY_MAP = {
        # Mega-cap — tight spreads ($50B+ market cap, >$5B daily volume)
        "AAPL": 2.0, "MSFT": 2.0, "GOOGL": 2.5, "AMZN": 2.5,
        "META": 2.5, "NVDA": 2.0, "TSLA": 3.0, "JPM": 3.0,
        "V": 3.0, "MA": 3.0, "LLY": 3.0, "UNH": 3.5,
        "JNJ": 3.0, "WMT": 3.0, "PG": 3.0, "KO": 3.0, "PEP": 3.0,
        "XOM": 3.5, "CVX": 3.5, "HD": 3.0, "NKE": 4.0, "CAT": 4.0,
        "BA": 5.0, "HON": 3.5, "UNP": 4.0, "NEE": 4.0, "DUK": 4.0,
        "GS": 4.0, "BAC": 3.5, "LIN": 4.0, "APD": 5.0, "DIS": 4.0,
        "TMUS": 4.0, "AMT": 5.0, "PLD": 5.0,
        # Large-cap
        "UBER": 4.0, "PLTR": 5.0, "CRWD": 5.0, "DDOG": 5.0,
        "SHOP": 5.0, "MRK": 3.5, "ABT": 4.0, "PFE": 4.0, "COP": 4.0,
        # Mid-cap — wider spreads
        "ETSY": 8.0, "FVRR": 12.0, "ROKU": 8.0, "DOCU": 10.0,
        "ABNB": 6.0, "NET": 7.0, "SQ": 6.0, "SNOW": 7.0,
        # Crypto — 24/7 high liquidity
        "BTCUSD": 2.0, "ETHUSD": 2.5,
    }
    base = ILLIQUIDITY_MAP.get(ticker, 5.0)  # default 5 bps

    if dollar_volume is not None and dollar_volume > 0:
        reference_volume = 1e9
        volume_factor = max(0.5, min(2.0, (reference_volume / dollar_volume) ** 0.3))
        base *= volume_factor

    return base


def _load_calibration_safe(path: Path) -> Optional[CalibrationArtifact]:
    if not path.exists():
        return None
    try:
        return load_calibration(path)
    except CalibrationHashError as exc:
        logger.warning(
            "Cost calibration failed verification at %s: %s — using static lookup",
            path,
            exc,
        )
        return None


@dataclass(frozen=True)
class TransactionCostModel:
    """Estimate transaction costs from either weights or traded notional."""

    commission_bps: float = DEFAULT_COMMISSION_BPS
    slippage_bps: float = DEFAULT_SLIPPAGE_BPS
    slippage_bps_by_ticker: Optional[dict[str, float]] = field(default=None)
    calibration_version: str = ""

    @classmethod
    def from_env(cls, *, use_calibration: bool = True) -> "TransactionCostModel":
        """Build cost model from env defaults, optionally blending calibration."""
        commission = get_default_commission_bps()
        default_slippage = get_default_slippage_bps()

        if not use_calibration:
            return cls(
                commission_bps=commission,
                slippage_bps=default_slippage,
                slippage_bps_by_ticker=None,
                calibration_version="",
            )

        calib_path = get_calibration_path()
        if calib_path is None:
            return cls(
                commission_bps=commission,
                slippage_bps=default_slippage,
                slippage_bps_by_ticker=None,
                calibration_version="",
            )

        artifact = _load_calibration_safe(calib_path)
        if artifact is None:
            return cls(
                commission_bps=commission,
                slippage_bps=default_slippage,
                slippage_bps_by_ticker=None,
                calibration_version="",
            )

        by_ticker = build_slippage_bps_by_ticker(artifact)
        return cls(
            commission_bps=commission,
            slippage_bps=default_slippage,
            slippage_bps_by_ticker=by_ticker,
            calibration_version=artifact.version,
        )

    @classmethod
    def static_lookup(cls) -> "TransactionCostModel":
        """Force static per-ticker lookup (no calibration blend)."""
        return cls.from_env(use_calibration=False)

    def slippage_bps_for(self, ticker: str) -> float:
        if self.slippage_bps_by_ticker is not None:
            return float(
                self.slippage_bps_by_ticker.get(ticker, estimate_slippage_bps(ticker))
            )
        return estimate_slippage_bps(ticker)

    def total_cost_bps_for(self, ticker: str) -> float:
        return float(self.commission_bps + self.slippage_bps_for(ticker))

    @property
    def total_cost_bps(self) -> float:
        return float(self.commission_bps + self.slippage_bps)

    def weighted_slippage_bps(
        self,
        w_old: np.ndarray,
        w_new: np.ndarray,
        tickers: list[str] | np.ndarray,
    ) -> float:
        """Turnover-weighted average slippage bps across tickers."""
        w_old_arr = np.asarray(w_old, dtype=float)
        w_new_arr = np.asarray(w_new, dtype=float)
        deltas = np.abs(w_new_arr - w_old_arr)
        total_delta = float(deltas.sum())
        if total_delta < 1e-12:
            return self.slippage_bps

        if self.slippage_bps_by_ticker is None:
            return self.slippage_bps

        weighted = 0.0
        for i, t in enumerate(tickers):
            if deltas[i] < 1e-12:
                continue
            weighted += deltas[i] * self.slippage_bps_for(str(t))
        return float(weighted / total_delta)

    def estimate_turnover(self, w_old: np.ndarray, w_new: np.ndarray) -> float:
        w_old_arr = np.asarray(w_old, dtype=float)
        w_new_arr = np.asarray(w_new, dtype=float)
        return float(np.abs(w_new_arr - w_old_arr).sum() / 2.0)

    def estimate_cost_from_weight_deltas(
        self,
        w_old: np.ndarray,
        w_new: np.ndarray,
        tickers: list[str] | np.ndarray,
        *,
        portfolio_value: float = 1.0,
    ) -> float:
        """Per-ticker cost: sum(|dw_i|) * (commission + slippage_i) / 10_000 * PV."""
        w_old_arr = np.asarray(w_old, dtype=float)
        w_new_arr = np.asarray(w_new, dtype=float)
        deltas = np.abs(w_new_arr - w_old_arr)
        if deltas.sum() < 1e-12:
            return 0.0

        cost_frac = 0.0
        for i, t in enumerate(tickers):
            if deltas[i] < 1e-12:
                continue
            bps = self.total_cost_bps_for(str(t))
            cost_frac += deltas[i] * bps / 10_000.0
        return float(cost_frac * float(portfolio_value))

    def estimate_cost_from_turnover(
        self,
        turnover: float,
        *,
        portfolio_value: float = 1.0,
        tickers: Optional[list[str] | np.ndarray] = None,
        w_old: Optional[np.ndarray] = None,
        w_new: Optional[np.ndarray] = None,
    ) -> float:
        if (
            self.slippage_bps_by_ticker is not None
            and tickers is not None
            and w_old is not None
            and w_new is not None
        ):
            return self.estimate_cost_from_weight_deltas(
                w_old, w_new, tickers, portfolio_value=portfolio_value
            )
        return float(float(turnover) * self.total_cost_bps / 10_000.0 * float(portfolio_value))

    def estimate_cost_from_weights(
        self,
        w_old: np.ndarray,
        w_new: np.ndarray,
        *,
        portfolio_value: float = 1.0,
        tickers: Optional[list[str] | np.ndarray] = None,
    ) -> float:
        if self.slippage_bps_by_ticker is not None and tickers is not None:
            return self.estimate_cost_from_weight_deltas(
                w_old, w_new, tickers, portfolio_value=portfolio_value
            )
        turnover = self.estimate_turnover(w_old, w_new)
        return self.estimate_cost_from_turnover(turnover, portfolio_value=portfolio_value)

    def estimate_cost_from_notional(self, traded_notional: float) -> float:
        return float(float(traded_notional) * self.total_cost_bps / 10_000.0)
