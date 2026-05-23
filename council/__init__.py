"""Council package: ensemble aggregation, conformal position sizing, portfolio optimisation,
and model monitoring / drift detection."""

from .aggregator import CouncilAggregator, OrthogonalityMonitor
from .alerts import AlertDispatcher, AlertResult, Severity, load_current_alerts
from .conformal import ConformalPositionSizer
from .cqr import CQRPositionSizer, StackingMetaLearner, get_position_sizer
from .fractional_kelly import FractionalKellySizer
from .covariance_dynamic import (
    DCCEstimator,
    FactorCovarianceEstimator,
    compute_covariance_from_returns,
    shrink_covariance_matrix,
)
from .evidently_reports import generate_drift_report, generate_model_performance_report
from .moe_gating import MoEGatingNetwork, aggregator_mode as moe_aggregator_mode
from .monitor import CouncilMonitor
from .portfolio import PortfolioConstructor
from .portfolio_diff import (
    DifferentiablePortfolioConstructor,
    HRPBlendPortfolioConstructor,
    get_portfolio_constructor,
)
from .portfolio_multiperiod import (
    MultiPeriodTCConfig,
    multi_period_tc_enabled,
    smooth_target_weights,
)
from .risk_rules import (
    PositionRiskRules,
    DrawdownProtection,
    PortfolioRiskMonitor,
    ExitSignal,
)
from .risk_engine import (
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
