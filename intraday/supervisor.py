from __future__ import annotations

import hashlib
import json
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from intraday.agent import FallbackIntradayAgent
from intraday.contracts import (
    AgentDecisionTrace,
    ExecutionIntent,
    FeatureSnapshot,
    IntradayDecision,
)
from intraday.market import USMarketCalendar
from intraday.market_data import MarketDataAdapter, MarketSnapshot


_NY_TZ = ZoneInfo("America/New_York")


class IntradaySupervisor:
    def __init__(
        self,
        *,
        market_data_adapter: MarketDataAdapter,
        agent_orchestrator=None,
        storage_dir: str | Path = "data/intraday",
        universe: list[str] | None = None,
        schedule_minutes: int = 15,
        calendar: USMarketCalendar | None = None,
    ) -> None:
        self.market_data_adapter = market_data_adapter
        self.agent_orchestrator = agent_orchestrator or FallbackIntradayAgent()
        self.storage_dir = Path(storage_dir)
        self.schedule_minutes = schedule_minutes
        self.universe = list(universe or ["AAPL", "MSFT", "NVDA", "AMZN", "META"])
        self.calendar = calendar or USMarketCalendar()
        self._has_crypto_universe = any(self._is_crypto_ticker(ticker) for ticker in self.universe)
        self._lock = threading.Lock()
        self._worker: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._state_path = self.storage_dir / "supervisor_state.json"
        self.state = self._load_state()

    def start(self) -> dict[str, Any]:
        with self._lock:
            self.state["running"] = True
            self.state["paused"] = False
            self._persist_state()
            if self._worker is None or not self._worker.is_alive():
                self._stop_event.clear()
                self._worker = threading.Thread(
                    target=self._run_loop,
                    daemon=True,
                    name="mlcouncil-intraday-supervisor",
                )
                self._worker.start()
        return self.get_status()

    def pause(self) -> dict[str, Any]:
        with self._lock:
            self.state["paused"] = True
            self._persist_state()
        return self.get_status()

    def resume(self) -> dict[str, Any]:
        with self._lock:
            self.state["running"] = True
            self.state["paused"] = False
            self._persist_state()
        return self.get_status()

    def stop(self) -> dict[str, Any]:
        with self._lock:
            self.state["running"] = False
            self.state["paused"] = False
            self._persist_state()
            self._stop_event.set()
        return self.get_status()

    def get_status(self) -> dict[str, Any]:
        latest = self.get_latest_decision()
        return {
            "running": bool(self.state.get("running", False)),
            "paused": bool(self.state.get("paused", False)),
            "schedule_minutes": self.schedule_minutes,
            "market_session": str(self.state.get("market_session", "closed")),
            "last_completed_slot": self.state.get("last_completed_slot"),
            "latest_decision_id": latest.get("decision_id") if latest else None,
        }

    def get_health(self) -> dict[str, Any]:
        status = self.get_status()
        state = "running" if status["running"] and not status["paused"] else "idle"
        if status["paused"]:
            state = "paused"
        return {
            "status": state,
            "last_completed_slot": status["last_completed_slot"],
            "latest_decision_id": status["latest_decision_id"],
        }

    def run_cycle(self, *, now: datetime | None = None) -> IntradayDecision:
        moment = self._normalize_now(now)
        slot = self.calendar.slot_start(moment, self.schedule_minutes)
        slot_key = slot.isoformat()
        existing = self._load_decision_by_slot(slot_key)
        if existing is not None and self._is_current_payload(existing):
            self.state["last_completed_slot"] = slot_key
            self._persist_state()
            return self._decision_from_payload(existing)

        market_snapshot = self.market_data_adapter.get_market_snapshot(
            as_of=slot,
            universe=self.universe,
        )
        news_items = self.market_data_adapter.get_news_snapshot(
            as_of=slot,
            universe=self.universe,
        )
        feature_snapshot = self._build_feature_snapshot(slot, market_snapshot, news_items)
        agent_trace = self.agent_orchestrator.synthesize(
            market_snapshot=market_snapshot,
            feature_snapshot=feature_snapshot,
            news_items=news_items,
        )
        decision = self._build_decision(
            slot=slot,
            market_snapshot=market_snapshot,
            feature_snapshot=feature_snapshot,
            agent_trace=agent_trace,
        )
        self._persist_decision(decision, market_snapshot)
        self.state["market_session"] = market_snapshot.session
        self.state["last_completed_slot"] = slot_key
        self._persist_state()
        return decision

    def _is_current_payload(self, payload: dict[str, Any]) -> bool:
        feature_snapshot = payload.get("feature_snapshot")
        if not isinstance(feature_snapshot, dict):
            return False

        source = str(feature_snapshot.get("source", ""))
        features = feature_snapshot.get("features", {})
        if not isinstance(features, dict):
            return False

        adapter_name = self.market_data_adapter.__class__.__name__
        if adapter_name == "PolygonMarketDataAdapter":
            if source != "polygon-prev+alpaca-fallback":
                return False
            for values in features.values():
                if not isinstance(values, dict):
                    return False
                if "previous_day_volume" not in values:
                    return False

        return True

    def list_decisions(self, *, limit: int = 20) -> list[dict[str, Any]]:
        decisions_dir = self.storage_dir / "decisions"
        if not decisions_dir.exists():
            return []
        payloads = self._load_recent_decision_payloads(decisions_dir, limit=limit)
        payloads.sort(key=lambda item: str(item.get("as_of", "")), reverse=True)
        if len(payloads) > limit:
            return payloads[:limit]
        return payloads

    def get_latest_decision(self) -> dict[str, Any] | None:
        decisions = self.list_decisions(limit=1)
        return decisions[0] if decisions else None

    def explain_decision(self, decision_id: str) -> dict[str, Any]:
        payload = self._load_decision_by_id(decision_id)
        if payload is None:
            raise KeyError(decision_id)
        agent_trace = payload.get("agent_trace", {})
        return {
            "decision_id": decision_id,
            "summary": agent_trace.get("summary"),
            "rationale": agent_trace.get("rationale", []),
            "execution_intents": payload.get("execution_intents", []),
            "market_session": payload.get("market_session"),
            "as_of": payload.get("as_of"),
        }

    def _run_loop(self) -> None:
        while not self._stop_event.is_set():
            if self.state.get("running") and not self.state.get("paused"):
                now = self._normalize_now(None)
                self.state["market_session"] = self._resolve_market_session(now)
                self._persist_state()
                if self._should_run_cycle(now):
                    self.run_cycle(now=now)
            time.sleep(1.0)

    def _build_feature_snapshot(
        self,
        slot: datetime,
        market_snapshot: MarketSnapshot,
        news_items: list[dict[str, Any]],
    ) -> FeatureSnapshot:
        sentiment_by_ticker: dict[str, list[float]] = {}
        headline_count: dict[str, int] = {}
        for item in news_items:
            ticker = str(item.get("ticker", "")).upper()
            if not ticker:
                continue
            headline_count[ticker] = headline_count.get(ticker, 0) + 1
            sentiment_by_ticker.setdefault(ticker, []).append(
                float(item.get("sentiment_score", 0.0) or 0.0)
            )

        features: dict[str, dict[str, float]] = {}
        for ticker, bar in market_snapshot.bars.items():
            news_scores = sentiment_by_ticker.get(ticker, [])
            base_features = {
                key: float(value)
                for key, value in bar.items()
                if isinstance(value, (int, float))
            }
            base_features["momentum_15m"] = float(bar.get("return_15m", 0.0) or 0.0)
            base_features["sentiment_score"] = (
                sum(news_scores) / len(news_scores) if news_scores else 0.0
            )
            base_features["headline_count"] = float(headline_count.get(ticker, 0))
            features[ticker] = base_features

        return FeatureSnapshot(
            as_of=slot.isoformat(),
            features=features,
            source=market_snapshot.source,
            version=f"intraday-features-{slot.strftime('%Y%m%d%H%M')}",
        )

    def _build_decision(
        self,
        *,
        slot: datetime,
        market_snapshot: MarketSnapshot,
        feature_snapshot: FeatureSnapshot,
        agent_trace: AgentDecisionTrace,
    ) -> IntradayDecision:
        intents = self._build_execution_intents(feature_snapshot, agent_trace)
        slot_key = slot.isoformat()
        decision_id = hashlib.sha256(
            "|".join([slot_key, ",".join(self.universe), feature_snapshot.version]).encode("utf-8")
        ).hexdigest()[:16]
        return IntradayDecision(
            decision_id=decision_id,
            as_of=slot_key,
            market_session=market_snapshot.session,
            schedule_minutes=self.schedule_minutes,
            universe=self.universe,
            market_snapshot_version=f"{market_snapshot.source}-{slot.strftime('%Y%m%d%H%M')}",
            feature_snapshot=feature_snapshot,
            agent_trace=agent_trace,
            execution_intents=intents,
            data_snapshot_version=f"{market_snapshot.source}-{slot.strftime('%Y%m%d%H%M')}",
        )

    def _build_execution_intents(
        self,
        feature_snapshot: FeatureSnapshot,
        agent_trace: AgentDecisionTrace,
    ) -> list[ExecutionIntent]:
        scored: list[tuple[str, float, float]] = []
        for ticker, feature in feature_snapshot.features.items():
            momentum = float(feature.get("momentum_15m", 0.0))
            sentiment = float(agent_trace.sentiment.get(ticker, 0.0))
            ticker_bias = float(agent_trace.ticker_scores.get(ticker, 0.0))
            day_change = float(feature.get("day_change_pct", 0.0))
            vwap_distance = float(feature.get("distance_from_vwap", 0.0))
            score = momentum + (0.25 * sentiment) + (0.60 * ticker_bias) + (0.10 * day_change) + (0.05 * vwap_distance)
            scored.append((ticker, score, float(feature.get("close", 0.0))))

        scored.sort(key=lambda item: item[1], reverse=True)
        intents: list[ExecutionIntent] = []
        for ticker, score, close in scored[:2]:
            side = "buy" if score >= 0 else "sell"
            target_weight = round(min(max(abs(score) * 2.5, 0.0), 0.10), 4)
            if target_weight <= 0:
                continue
            intents.append(
                ExecutionIntent(
                    ticker=ticker,
                    side=side,
                    confidence=round(min(max(agent_trace.confidence, 0.0), 1.0), 4),
                    target_weight=target_weight,
                    quantity_notional=round(target_weight * 100_000.0, 2),
                    rationale=f"{ticker} score={score:.4f} close={close:.2f}",
                )
            )
        return intents

    def _persist_decision(self, decision: IntradayDecision, market_snapshot: MarketSnapshot) -> None:
        decision_dir = self.storage_dir / "decisions" / decision.as_of[:10]
        decision_dir.mkdir(parents=True, exist_ok=True)
        payload = decision.to_dict()
        payload["market_snapshot"] = market_snapshot.to_dict()
        path = decision_dir / f"{decision.decision_id}.json"
        path.write_text(json.dumps(payload, indent=2))

    def _load_recent_decision_payloads(
        self,
        decisions_dir: Path,
        *,
        limit: int,
    ) -> list[dict[str, Any]]:
        payloads: list[dict[str, Any]] = []
        try:
            dated_dirs = sorted(
                (path for path in decisions_dir.iterdir() if path.is_dir()),
                key=lambda path: path.name,
                reverse=True,
            )
        except OSError:
            return payloads

        for dated_dir in dated_dirs:
            try:
                files = sorted(
                    (path for path in dated_dir.iterdir() if path.is_file() and path.suffix == ".json"),
                    key=lambda path: path.name,
                    reverse=True,
                )
            except OSError:
                continue
            for file_path in files:
                try:
                    payloads.append(json.loads(file_path.read_text()))
                except (OSError, json.JSONDecodeError):
                    continue
            if len(payloads) >= limit:
                return payloads
        return payloads

    def _load_decision_by_slot(self, slot_key: str) -> dict[str, Any] | None:
        for payload in self.list_decisions(limit=500):
            if payload.get("as_of") == slot_key:
                return payload
        return None

    def _load_decision_by_id(self, decision_id: str) -> dict[str, Any] | None:
        for payload in self.list_decisions(limit=500):
            if payload.get("decision_id") == decision_id:
                return payload
        return None

    def _decision_from_payload(self, payload: dict[str, Any]) -> IntradayDecision:
        feature_snapshot = FeatureSnapshot(**payload["feature_snapshot"])
        agent_trace = AgentDecisionTrace(**payload["agent_trace"])
        intents = [ExecutionIntent(**item) for item in payload.get("execution_intents", [])]
        return IntradayDecision(
            decision_id=payload["decision_id"],
            as_of=payload["as_of"],
            market_session=payload["market_session"],
            schedule_minutes=int(payload["schedule_minutes"]),
            universe=list(payload.get("universe", [])),
            market_snapshot_version=payload["market_snapshot_version"],
            feature_snapshot=feature_snapshot,
            agent_trace=agent_trace,
            execution_intents=intents,
            data_snapshot_version=payload.get("data_snapshot_version", "unknown"),
            strategy_version=payload.get("strategy_version", "intraday-v1"),
        )

    def _load_state(self) -> dict[str, Any]:
        if not self._state_path.exists():
            return {
                "running": False,
                "paused": False,
                "market_session": "closed",
                "last_completed_slot": None,
            }
        try:
            return json.loads(self._state_path.read_text())
        except (OSError, json.JSONDecodeError):
            return {
                "running": False,
                "paused": False,
                "market_session": "closed",
                "last_completed_slot": None,
            }

    def _persist_state(self) -> None:
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self._state_path.write_text(json.dumps(self.state, indent=2))

    def _resolve_market_session(self, now: datetime) -> str:
        window = self.calendar.trading_window(now)
        if window is None:
            return "crypto" if self._has_crypto_universe else "closed"
        if self.calendar.is_market_open(now):
            return window.session
        if self._has_crypto_universe:
            return "crypto"
        if now.astimezone(_NY_TZ) < window.opens_at:
            return "pre_market"
        return "post_market"

    def _should_run_cycle(self, now: datetime) -> bool:
        if self._has_crypto_universe:
            return True
        return self.calendar.is_market_open(now)

    @staticmethod
    def _is_crypto_ticker(ticker: str) -> bool:
        upper = str(ticker).strip().upper()
        if not upper:
            return False
        normalized = upper.replace("/", "").replace("-", "")
        base_currency, quote_currency = normalized[:-3], normalized[-3:]
        return (
            len(normalized) >= 6
            and quote_currency == "USD"
            and 2 <= len(base_currency) <= 10
            and base_currency.isalpha()
        )

    def _normalize_now(self, now: datetime | None) -> datetime:
        if now is None:
            return datetime.now(tz=_NY_TZ)
        if now.tzinfo is None:
            return now.replace(tzinfo=_NY_TZ)
        return now.astimezone(_NY_TZ)
