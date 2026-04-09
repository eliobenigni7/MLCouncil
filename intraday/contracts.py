from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class FeatureSnapshot:
    as_of: str
    features: dict[str, dict[str, float]]
    source: str
    version: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class AgentDecisionTrace:
    agent_name: str
    summary: str
    confidence: float
    sentiment: dict[str, float]
    rationale: list[str]
    prompt_version: str
    model_version: str
    ticker_scores: dict[str, float] = field(default_factory=dict)
    provider: str = "local"
    request_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ExecutionIntent:
    ticker: str
    side: str
    confidence: float
    target_weight: float
    quantity_notional: float
    rationale: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class IntradayDecision:
    decision_id: str
    as_of: str
    market_session: str
    schedule_minutes: int
    universe: list[str]
    market_snapshot_version: str
    feature_snapshot: FeatureSnapshot
    agent_trace: AgentDecisionTrace
    execution_intents: list[ExecutionIntent] = field(default_factory=list)
    data_snapshot_version: str = "unknown"
    strategy_version: str = "intraday-v1"

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["feature_snapshot"] = self.feature_snapshot.to_dict()
        payload["agent_trace"] = self.agent_trace.to_dict()
        payload["execution_intents"] = [intent.to_dict() for intent in self.execution_intents]
        return payload


@dataclass(slots=True)
class ModelPromotionRecord:
    model_name: str
    model_version: str
    promoted_at: str
    stage: str
    metrics: dict[str, float]
    tags: dict[str, str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
