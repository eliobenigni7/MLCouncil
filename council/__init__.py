"""Council package: ensemble aggregation, conformal position sizing, portfolio optimisation,
and model monitoring / drift detection."""

from .aggregator import CouncilAggregator, OrthogonalityMonitor
from .alerts import AlertDispatcher, AlertResult, Severity, load_current_alerts
from .conformal import ConformalPositionSizer
from .evidently_reports import generate_drift_report, generate_model_performance_report
from .monitor import CouncilMonitor
from .portfolio import PortfolioConstructor
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
    "PortfolioConstructor",
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
