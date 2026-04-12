from __future__ import annotations

from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from intraday.contracts import AgentDecisionTrace, ExecutionIntent, FeatureSnapshot


def test_market_calendar_handles_holiday_and_half_day_close():
    from intraday.market import USMarketCalendar

    calendar = USMarketCalendar()

    independence_observed = datetime(2026, 7, 3, 15, 0, tzinfo=ZoneInfo("America/New_York"))
    half_day_open = datetime(2026, 11, 27, 12, 30, tzinfo=ZoneInfo("America/New_York"))
    half_day_closed = datetime(2026, 11, 27, 14, 0, tzinfo=ZoneInfo("America/New_York"))

    assert calendar.is_market_open(independence_observed) is False
    assert calendar.is_market_open(half_day_open) is True
    assert calendar.is_market_open(half_day_closed) is False


def test_intraday_supervisor_run_cycle_is_idempotent_for_same_slot(tmp_path: Path):
    from intraday.market_data import MarketSnapshot
    from intraday.supervisor import IntradaySupervisor

    class FakeAdapter:
        def __init__(self):
            self.calls = 0

        def get_market_snapshot(self, *, as_of: datetime, universe: list[str]):
            self.calls += 1
            return MarketSnapshot(
                as_of=as_of,
                universe=universe,
                bars={
                    "AAPL": {"close": 201.0, "return_15m": 0.012},
                    "MSFT": {"close": 412.0, "return_15m": -0.004},
                },
                source="fake-feed",
                session="regular",
            )

        def get_news_snapshot(self, *, as_of: datetime, universe: list[str]):
            return [
                {
                    "ticker": "AAPL",
                    "headline": "Apple demand remains resilient",
                    "sentiment_score": 0.7,
                    "published_at": as_of.isoformat(),
                    "source": "fake-news",
                }
            ]

    class FakeAgent:
        def synthesize(
            self,
            *,
            market_snapshot,
            feature_snapshot,
            news_items,
        ) -> AgentDecisionTrace:
            del market_snapshot, feature_snapshot
            return AgentDecisionTrace(
                agent_name="rule-based-agent",
                summary="Momentum and sentiment favor AAPL over MSFT.",
                confidence=0.68,
                sentiment={
                    "AAPL": 0.7,
                    "MSFT": -0.1,
                },
                rationale=[
                    "AAPL has positive 15-minute momentum.",
                    f"News items considered: {len(news_items)}",
                ],
                prompt_version="fallback-v1",
                model_version="rule-based-v1",
            )

    now = datetime(2026, 4, 9, 10, 30, tzinfo=ZoneInfo("America/New_York"))
    storage_dir = tmp_path / "intraday"
    supervisor = IntradaySupervisor(
        market_data_adapter=FakeAdapter(),
        agent_orchestrator=FakeAgent(),
        storage_dir=storage_dir,
        universe=["AAPL", "MSFT"],
    )

    first = supervisor.run_cycle(now=now)
    second = supervisor.run_cycle(now=now)

    assert first.decision_id == second.decision_id
    assert len(supervisor.list_decisions(limit=10)) == 1
    assert supervisor.state["last_completed_slot"] == "2026-04-09T10:30:00-04:00"


def test_intraday_supervisor_builds_execution_intents(tmp_path: Path):
    from intraday.market_data import MarketSnapshot
    from intraday.supervisor import IntradaySupervisor

    class FakeAdapter:
        def get_market_snapshot(self, *, as_of: datetime, universe: list[str]):
            return MarketSnapshot(
                as_of=as_of,
                universe=universe,
                bars={
                    "AAPL": {"close": 201.0, "return_15m": 0.02},
                    "MSFT": {"close": 412.0, "return_15m": -0.01},
                },
                source="fake-feed",
                session="regular",
            )

        def get_news_snapshot(self, *, as_of: datetime, universe: list[str]):
            return []

    class FakeAgent:
        def synthesize(
            self,
            *,
            market_snapshot,
            feature_snapshot: FeatureSnapshot,
            news_items,
        ) -> AgentDecisionTrace:
            del market_snapshot, feature_snapshot, news_items
            return AgentDecisionTrace(
                agent_name="rule-based-agent",
                summary="Buy AAPL and trim MSFT.",
                confidence=0.82,
                sentiment={"AAPL": 0.5, "MSFT": -0.2},
                rationale=["AAPL stronger than MSFT."],
                prompt_version="fallback-v1",
                model_version="rule-based-v1",
            )

    supervisor = IntradaySupervisor(
        market_data_adapter=FakeAdapter(),
        agent_orchestrator=FakeAgent(),
        storage_dir=tmp_path / "intraday",
        universe=["AAPL", "MSFT"],
    )

    decision = supervisor.run_cycle(
        now=datetime(2026, 4, 9, 11, 15, tzinfo=ZoneInfo("America/New_York"))
    )

    assert decision.execution_intents
    assert all(isinstance(intent, ExecutionIntent) for intent in decision.execution_intents)
    assert decision.agent_trace.confidence == 0.82


def test_intraday_supervisor_uses_agent_ticker_scores_to_rank_execution_intents(tmp_path: Path):
    from intraday.market_data import MarketSnapshot
    from intraday.supervisor import IntradaySupervisor

    class FakeAdapter:
        def get_market_snapshot(self, *, as_of: datetime, universe: list[str]):
            return MarketSnapshot(
                as_of=as_of,
                universe=universe,
                bars={
                    "AAPL": {"close": 201.0, "return_15m": 0.03, "day_change_pct": 0.01},
                    "MSFT": {"close": 412.0, "return_15m": 0.01, "day_change_pct": 0.02},
                },
                source="polygon",
                session="regular",
            )

        def get_news_snapshot(self, *, as_of: datetime, universe: list[str]):
            return []

    class FakeAgent:
        def synthesize(self, *, market_snapshot, feature_snapshot: FeatureSnapshot, news_items):
            del market_snapshot, feature_snapshot, news_items
            return AgentDecisionTrace(
                agent_name="openai-intraday-agent",
                summary="MSFT gets higher conviction despite lower raw momentum.",
                confidence=0.88,
                sentiment={"AAPL": 0.1, "MSFT": 0.5},
                ticker_scores={"AAPL": 0.1, "MSFT": 0.9},
                rationale=["MSFT has stronger qualitative setup."],
                prompt_version="openai-intraday-v1",
                model_version="gpt-4o-mini",
            )

    supervisor = IntradaySupervisor(
        market_data_adapter=FakeAdapter(),
        agent_orchestrator=FakeAgent(),
        storage_dir=tmp_path / "intraday",
        universe=["AAPL", "MSFT"],
    )

    decision = supervisor.run_cycle(
        now=datetime(2026, 4, 9, 11, 30, tzinfo=ZoneInfo("America/New_York"))
    )

    assert decision.execution_intents[0].ticker == "MSFT"


def test_intraday_supervisor_list_decisions_orders_by_as_of(tmp_path: Path):
    from intraday.market_data import MarketSnapshot
    from intraday.supervisor import IntradaySupervisor

    class FakeAdapter:
        def get_market_snapshot(self, *, as_of: datetime, universe: list[str]):
            return MarketSnapshot(
                as_of=as_of,
                universe=universe,
                bars={"AAPL": {"close": 201.0, "return_15m": 0.02}},
                source="fake-feed",
                session="regular",
            )

        def get_news_snapshot(self, *, as_of: datetime, universe: list[str]):
            return []

    class FakeAgent:
        def synthesize(self, *, market_snapshot, feature_snapshot, news_items):
            del market_snapshot, feature_snapshot, news_items
            return AgentDecisionTrace(
                agent_name="rule-based-agent",
                summary="Buy AAPL.",
                confidence=0.82,
                sentiment={"AAPL": 0.5},
                rationale=["AAPL stronger than baseline."],
                prompt_version="fallback-v1",
                model_version="rule-based-v1",
            )

    supervisor = IntradaySupervisor(
        market_data_adapter=FakeAdapter(),
        agent_orchestrator=FakeAgent(),
        storage_dir=tmp_path / "intraday",
        universe=["AAPL"],
    )

    later = supervisor.run_cycle(
        now=datetime(2026, 4, 9, 11, 45, tzinfo=ZoneInfo("America/New_York"))
    )
    earlier = supervisor.run_cycle(
        now=datetime(2026, 4, 9, 11, 30, tzinfo=ZoneInfo("America/New_York"))
    )

    decisions = supervisor.list_decisions(limit=2)

    assert decisions[0]["as_of"] == later.as_of
    assert decisions[1]["as_of"] == earlier.as_of


def test_intraday_supervisor_list_decisions_skips_unreadable_day_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    from intraday.supervisor import IntradaySupervisor

    decisions_dir = tmp_path / "intraday" / "decisions"
    unreadable_day = decisions_dir / "2026-04-09"
    readable_day = decisions_dir / "2026-04-08"
    unreadable_day.mkdir(parents=True)
    readable_day.mkdir(parents=True)
    (readable_day / "decision.json").write_text(
        '{"decision_id": "stable", "as_of": "2026-04-08T15:30:00-04:00"}'
    )

    original_iterdir = Path.iterdir

    def patched_iterdir(self: Path):
        if self == unreadable_day:
            raise OSError(12, "Cannot allocate memory", str(self))
        return original_iterdir(self)

    monkeypatch.setattr(Path, "iterdir", patched_iterdir)

    supervisor = IntradaySupervisor(
        market_data_adapter=object(),
        storage_dir=tmp_path / "intraday",
    )

    assert supervisor.list_decisions(limit=10) == [
        {"decision_id": "stable", "as_of": "2026-04-08T15:30:00-04:00"}
    ]


def test_intraday_supervisor_list_decisions_skips_unreadable_directory(tmp_path: Path, monkeypatch):
    from intraday.supervisor import IntradaySupervisor

    class FakeAdapter:
        def get_market_snapshot(self, *, as_of: datetime, universe: list[str]):
            raise AssertionError("market snapshot should not be requested")

        def get_news_snapshot(self, *, as_of: datetime, universe: list[str]):
            raise AssertionError("news snapshot should not be requested")

    supervisor = IntradaySupervisor(
        market_data_adapter=FakeAdapter(),
        storage_dir=tmp_path / "intraday",
        universe=["AAPL"],
    )

    decisions_dir = supervisor.storage_dir / "decisions"
    healthy_dir = decisions_dir / "2026-04-09"
    healthy_dir.mkdir(parents=True, exist_ok=True)
    (healthy_dir / "good.json").write_text(
        '{"decision_id": "good", "as_of": "2026-04-09T10:30:00-04:00"}'
    )

    broken_dir = decisions_dir / "2026-04-10"
    broken_dir.mkdir(parents=True, exist_ok=True)

    original_iterdir = Path.iterdir

    def flaky_iterdir(self: Path):
        if self == decisions_dir:
            raise OSError("Cannot allocate memory")
        return original_iterdir(self)

    monkeypatch.setattr(Path, "iterdir", flaky_iterdir)

    payloads = supervisor.list_decisions(limit=5)

    assert payloads == []


def test_intraday_supervisor_runs_crypto_universe_outside_equity_hours(tmp_path: Path):
    from intraday.market import USMarketCalendar
    from intraday.supervisor import IntradaySupervisor

    supervisor = IntradaySupervisor(
        market_data_adapter=object(),
        storage_dir=tmp_path / "intraday",
        universe=["AAPL", "BTCUSD"],
        calendar=USMarketCalendar(),
    )

    sunday = datetime(2026, 4, 12, 3, 0, tzinfo=ZoneInfo("America/New_York"))

    assert supervisor._should_run_cycle(sunday) is True
    assert supervisor._resolve_market_session(sunday) == "crypto"


def test_intraday_supervisor_keeps_equity_only_schedule_closed_outside_market_hours(tmp_path: Path):
    from intraday.market import USMarketCalendar
    from intraday.supervisor import IntradaySupervisor

    supervisor = IntradaySupervisor(
        market_data_adapter=object(),
        storage_dir=tmp_path / "intraday",
        universe=["AAPL", "MSFT"],
        calendar=USMarketCalendar(),
    )

    sunday = datetime(2026, 4, 12, 3, 0, tzinfo=ZoneInfo("America/New_York"))

    assert supervisor._should_run_cycle(sunday) is False
    assert supervisor._resolve_market_session(sunday) == "closed"
