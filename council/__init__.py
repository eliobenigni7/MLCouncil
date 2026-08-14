"""Council package: ensemble aggregation, conformal position sizing, portfolio optimisation,
and model monitoring / drift detection."""

from .aggregation.aggregator import CouncilAggregator, OrthogonalityMonitor
from .monitoring.alerts import AlertDispatcher, AlertResult, Severity, load_current_alerts
from .sizing.conformal import ConformalPositionSizer
from .sizing.cqr import CQRPositionSizer, StackingMetaLearner, get_position_sizer
from .sizing.fractional_kelly import FractionalKellySizer
from .risk.covariance_dynamic import (
    DCCEstimator,
    FactorCovarianceEstimator,
    compute_covariance_from_returns,
    shrink_covariance_matrix,
)
from .monitoring.evidently_reports import generate_drift_report, generate_model_performance_report
from .aggregation.moe_gating import MoEGatingNetwork, aggregator_mode as moe_aggregator_mode
from .monitoring.monitor import CouncilMonitor
from .portfolio.portfolio import PortfolioConstructor
from .portfolio.portfolio_diff import (
    DifferentiablePortfolioConstructor,
    HRPBlendPortfolioConstructor,
    get_portfolio_constructor,
)
from .portfolio.portfolio_multiperiod import (
    MultiPeriodTCConfig,
    multi_period_tc_enabled,
    smooth_target_weights,
)
from .risk.risk_rules import (
    PositionRiskRules,
    DrawdownProtection,
    PortfolioRiskMonitor,
    ExitSignal,
)
from .risk.risk_engine import (
    RiskEngine,
    RiskReport,
    RiskLimits,
    RiskBreach,
    VaRReport,
    ExposureReport,
    Position,
)
from .mlflow_utils import (
    MLflowTracker,
    build_run_tags,
    get_tracker,
    log_ic_metrics,
    log_signal_metrics,
    validate_promotion_gate,
)

__all__ = [
    "CouncilAggregator",
    "OrthogonalityMonitor",
    "ConformalPositionSizer",
    "CQRPositionSizer",
    "FractionalKellySizer",
    "StackingMetaLearner",
    "get_position_sizer",
    "MoEGatingNetwork",
    "moe_aggregator_mode",
    "DCCEstimator",
    "FactorCovarianceEstimator",
    "compute_covariance_from_returns",
    "shrink_covariance_matrix",
    "PortfolioConstructor",
    "DifferentiablePortfolioConstructor",
    "HRPBlendPortfolioConstructor",
    "get_portfolio_constructor",
    "MultiPeriodTCConfig",
    "multi_period_tc_enabled",
    "smooth_target_weights",
    "CouncilMonitor",
    "AlertResult",
    "AlertDispatcher",
    "Severity",
    "load_current_alerts",
    "generate_drift_report",
    "generate_model_performance_report",
    "PositionRiskRules",
    "DrawdownProtection",
    "PortfolioRiskMonitor",
    "ExitSignal",
    "RiskEngine",
    "RiskReport",
    "RiskLimits",
    "RiskBreach",
    "VaRReport",
    "ExposureReport",
    "Position",
    "MLflowTracker",
    "build_run_tags",
    "get_tracker",
    "log_ic_metrics",
    "log_signal_metrics",
    "validate_promotion_gate",
]
