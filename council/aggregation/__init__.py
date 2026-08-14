"""Signal aggregation: weighted ensemble (CouncilAggregator) and MoE gating shadow (T3.1)."""

from .aggregator import CouncilAggregator, OrthogonalityMonitor
from .moe_gating import MoEGatingNetwork, aggregator_mode

__all__ = [
    "CouncilAggregator",
    "OrthogonalityMonitor",
    "MoEGatingNetwork",
    "aggregator_mode",
]
