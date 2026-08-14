"""Risk management: risk engine, position rules, dynamic covariance, generative stress,
causal drift, TDA early warning and streaming drift detectors (ADWIN / DDM)."""

from .covariance_dynamic import (
    DCCEstimator,
    FactorCovarianceEstimator,
    compute_covariance_from_returns,
    shrink_covariance_matrix,
)
from .risk_engine import (
    ExposureReport,
    Position,
    RiskBreach,
    RiskEngine,
    RiskLimits,
    RiskReport,
    VaRReport,
)
from .risk_rules import DrawdownProtection, ExitSignal, PortfolioRiskMonitor, PositionRiskRules

__all__ = [
    "DCCEstimator",
    "FactorCovarianceEstimator",
    "compute_covariance_from_returns",
    "shrink_covariance_matrix",
    "ExposureReport",
    "Position",
    "RiskBreach",
    "RiskEngine",
    "RiskLimits",
    "RiskReport",
    "VaRReport",
    "DrawdownProtection",
    "ExitSignal",
    "PortfolioRiskMonitor",
    "PositionRiskRules",
]
