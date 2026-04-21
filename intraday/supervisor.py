from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import yaml

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
_ROOT = Path(__file__).resolve().parents[1]


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
        executor=None,
    ) -> None:
        self.market_data_adapter = market_data_adapter
        self.agent_orchestrator = agent_orchestrator or FallbackIntradayAgent()
        self.storage_dir = Path(storage_dir)
        self.schedule_minutes = schedule_minutes
        self.min_valid_close_ratio = float(
            os.getenv("MLCOUNCIL_INTRADAY_MIN_VALID_CLOSE_RATIO", "0.70")
        )
        self.min_informative_ratio = float(
            os.getenv("MLCOUNCIL_INTRADAY_MIN_INFORMATIVE_RATIO", "0.50")
        )
        raw_universe = list(universe or ["AAPL", "MSFT", "NVDA", "AMZN", "META"])
        self.universe = raw_universe
        self.calendar = calendar or USMarketCalendar()
        self.crypto_enabled = (
            os.getenv("MLCOUNCIL_CRYPTO_ENABLED", "false").strip().lower() == "true"
        )
        self.equity_universe = [
            ticker for ticker in raw_universe if not self._is_crypto_ticker(ticker)
        ]
        inline_crypto_universe = [
            ticker for ticker in raw_universe if self._is_crypto_ticker(ticker)
        ]
        configured_crypto_universe = self._load_crypto_universe()
        if self.crypto_enabled:
            self.crypto_universe = configured_crypto_universe or inline_crypto_universe
        else:
            self.crypto_universe = []
        self._has_crypto_universe = bool(self.crypto_universe)
        self.executor = executor
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
        active_universe = self._select_universe(moment)
        resolved_session = self._resolve_market_session(moment)
        slot = self.calendar.slot_start(moment, self.schedule_minutes)
        slot_key = slot.isoformat()
        existing = self._load_decision_by_slot(slot_key)
        if (
            existing is not None
            and list(existing.get("universe", [])) == active_universe
            and self._is_current_payload(existing)
        ):
            self.state["last_completed_slot"] = slot_key
            self._persist_state()
            return self._decision_from_payload(existing)

        market_snapshot = self.market_data_adapter.get_market_snapshot(
            as_of=slot,
            universe=active_universe,
        )
        news_items = self.market_data_adapter.get_news_snapshot(
            as_of=slot,
            universe=active_universe,
        )
        feature_snapshot = self._build_feature_snapshot(
            slot, market_snapshot, news_items
        )
        data_ready, data_reason = self._is_market_snapshot_ready(
            feature_snapshot=feature_snapshot,
            expected_tickers=active_universe,
        )
        if not data_ready:
            agent_trace = AgentDecisionTrace(
                agent_name="supervisor-data-guard",
                summary=f"data_not_ready: {data_reason}",
                confidence=0.0,
                sentiment={},
                rationale=[
                    f"Snapshot source={feature_snapshot.source}",
                    data_reason,
                ],
                prompt_version="data-guard-v1",
                model_version="data-guard-v1",
                provider="supervisor",
            )
            decision = self._build_decision(
                slot=slot,
                universe=active_universe,
                market_session=resolved_session,
                market_snapshot_source=market_snapshot.source,
                feature_snapshot=feature_snapshot,
                agent_trace=agent_trace,
                decision_state="data_not_ready",
            )
            self._persist_decision(decision, market_snapshot)
            self.state["market_session"] = resolved_session
            self.state["last_completed_slot"] = slot_key
            self._persist_state()
            return decision
        agent_trace = self.agent_orchestrator.synthesize(
            market_snapshot=market_snapshot,
            feature_snapshot=feature_snapshot,
            news_items=news_items,
        )
        decision = self._build_decision(
            slot=slot,
            universe=active_universe,
            market_session=resolved_session,
            market_snapshot_source=market_snapshot.source,
            feature_snapshot=feature_snapshot,
            agent_trace=agent_trace,
            decision_state="ready",
        )
        self._persist_decision(decision, market_snapshot)
        self.state["market_session"] = resolved_session
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
                    decision = self.run_cycle(now=now)
                    if (
                        decision.decision_state == "ready"
                        and decision.execution_intents
                        and not self._is_decision_executed(decision.decision_id)
                    ):
                        self._execute_decision(decision)
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
        universe: list[str],
        market_session: str,
        market_snapshot_source: str,
        feature_snapshot: FeatureSnapshot,
        agent_trace: AgentDecisionTrace,
        decision_state: str = "ready",
    ) -> IntradayDecision:
        intents = self._build_execution_intents(feature_snapshot, agent_trace)
        slot_key = slot.isoformat()
        decision_id = hashlib.sha256(
            "|".join([slot_key, ",".join(universe), feature_snapshot.version]).encode(
                "utf-8"
            )
        ).hexdigest()[:16]
        return IntradayDecision(
            decision_id=decision_id,
            as_of=slot_key,
            market_session=market_session,
            schedule_minutes=self.schedule_minutes,
            universe=universe,
            market_snapshot_version=f"{market_snapshot_source}-{slot.strftime('%Y%m%d%H%M')}",
            feature_snapshot=feature_snapshot,
            agent_trace=agent_trace,
            execution_intents=intents,
            data_snapshot_version=f"{market_snapshot_source}-{slot.strftime('%Y%m%d%H%M')}",
            decision_state=decision_state,
        )

    def _build_execution_intents(
        self,
        feature_snapshot: FeatureSnapshot,
        agent_trace: AgentDecisionTrace,
        portfolio_value: float = 100_000.0,
    ) -> list[ExecutionIntent]:
        scored: list[tuple[str, float, float]] = []
        for ticker, feature in feature_snapshot.features.items():
            momentum = float(feature.get("momentum_15m", 0.0))
            sentiment = float(agent_trace.sentiment.get(ticker, 0.0))
            ticker_bias = float(agent_trace.ticker_scores.get(ticker, 0.0))
            day_change = float(feature.get("day_change_pct", 0.0))
            vwap_distance = float(feature.get("distance_from_vwap", 0.0))
            score = (
                momentum
                + (0.25 * sentiment)
                + (0.60 * ticker_bias)
                + (0.10 * day_change)
                + (0.05 * vwap_distance)
            )
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

    def _persist_decision(
        self, decision: IntradayDecision, market_snapshot: MarketSnapshot
    ) -> None:
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
                    (
                        path
                        for path in dated_dir.iterdir()
                        if path.is_file() and path.suffix == ".json"
                    ),
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
        intents = [
            ExecutionIntent(**item) for item in payload.get("execution_intents", [])
        ]
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
            decision_state=payload.get("decision_state", "ready"),
        )

    def _is_market_snapshot_ready(
        self,
        *,
        feature_snapshot: FeatureSnapshot,
        expected_tickers: list[str],
    ) -> tuple[bool, str]:
        features = feature_snapshot.features
        if not expected_tickers:
            return False, "empty intraday universe"
        if not features:
            return False, "empty feature snapshot"

        valid_close = 0
        informative = 0
        expected_count = len(expected_tickers)
        for ticker in expected_tickers:
            row = features.get(ticker, {})
            close = float(row.get("close", 0.0) or 0.0)
            if close > 0.0:
                valid_close += 1
            signals = (
                abs(float(row.get("return_15m", 0.0) or 0.0)),
                abs(float(row.get("day_change_pct", 0.0) or 0.0)),
                abs(float(row.get("intraday_range_pct", 0.0) or 0.0)),
                abs(float(row.get("distance_from_vwap", 0.0) or 0.0)),
                abs(float(row.get("spread_bps", 0.0) or 0.0)),
                float(row.get("minute_volume", 0.0) or 0.0),
                float(row.get("previous_day_volume", 0.0) or 0.0),
            )
            if any(value > 0.0 for value in signals):
                informative += 1

        close_ratio = valid_close / expected_count
        informative_ratio = informative / expected_count
        if close_ratio < self.min_valid_close_ratio:
            return (
                False,
                f"valid_close_ratio={close_ratio:.2f} below threshold={self.min_valid_close_ratio:.2f}",
            )
        if informative_ratio < self.min_informative_ratio:
            return (
                False,
                f"informative_ratio={informative_ratio:.2f} below threshold={self.min_informative_ratio:.2f}",
            )
        return True, "ok"

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
        if self.calendar.is_market_open(now):
            return window.session if window is not None else "regular"
        if self._has_crypto_universe and self.calendar.is_crypto_market_open(now):
            return "crypto"
        if window is None:
            return "closed"
        if now.astimezone(_NY_TZ) < window.opens_at:
            return "pre_market"
        return "post_market"

    def _should_run_cycle(self, now: datetime) -> bool:
        if self.calendar.is_market_open(now):
            return bool(self.equity_universe)
        return self._has_crypto_universe and self.calendar.is_crypto_market_open(now)

    def _select_universe(self, now: datetime) -> list[str]:
        if self.calendar.is_market_open(now):
            return list(self.equity_universe)
        if self._has_crypto_universe and self.calendar.is_crypto_market_open(now):
            return list(self.crypto_universe)
        return list(self.equity_universe)

    def _load_crypto_universe(self) -> list[str]:
        config_path = _ROOT / "config" / "universe.yaml"
        try:
            with open(config_path) as f:
                cfg = yaml.safe_load(f) or {}
        except OSError:
            return []

        crypto_cfg = cfg.get("crypto_universe", {})
        if not isinstance(crypto_cfg, dict):
            return []

        tickers: list[str] = []
        seen: set[str] = set()
        for bucket_values in crypto_cfg.values():
            if not isinstance(bucket_values, list):
                continue
            for ticker in bucket_values:
                normalized = str(ticker).strip().upper()
                if not normalized or normalized in seen:
                    continue
                seen.add(normalized)
                tickers.append(normalized)
        return tickers

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

    def _execute_decision(self, decision: IntradayDecision) -> None:
        if self.executor is None:
            return
        execution_status = "attempted"
        try:
            result = self.executor(decision.to_dict())
            if isinstance(result, dict):
                if result.get("error"):
                    execution_status = "blocked"
                else:
                    execution_status = "degraded" if result.get("orders_rejected") else "success"
        except Exception as e:
            import traceback
            print(f"[INTRADAY] Execution failed for {decision.decision_id}: {e}")
            print(traceback.format_exc())
            execution_status = "failed"
        self._mark_decision_execution_status(decision, execution_status)

    def _is_decision_executed(self, decision_id: str) -> bool:
        payload = self._load_decision_by_id(decision_id)
        if payload is None:
            return False
        execution_status = str(payload.get("execution_status", "")).lower()
        return execution_status in {"attempted", "success", "degraded", "blocked", "submitted"}

    def _mark_decision_execution_status(
        self,
        decision: IntradayDecision,
        execution_status: str,
    ) -> None:
        decision_path = self.storage_dir / "decisions" / decision.as_of[:10] / f"{decision.decision_id}.json"
        if not decision_path.exists():
            return
        try:
            payload = json.loads(decision_path.read_text())
        except (OSError, json.JSONDecodeError):
            return
        payload["execution_status"] = execution_status
        try:
            decision_path.write_text(json.dumps(payload, indent=2))
        except OSError:
            return
