from __future__ import annotations

import json
import os
from typing import Any

import httpx

from intraday.contracts import AgentDecisionTrace, FeatureSnapshot
from intraday.market_data import MarketSnapshot


class FallbackIntradayAgent:
    """Deterministic agent used when no external LLM orchestration is configured."""

    def synthesize(
        self,
        *,
        market_snapshot: MarketSnapshot,
        feature_snapshot: FeatureSnapshot,
        news_items: list[dict],
    ) -> AgentDecisionTrace:
        sentiment: dict[str, float] = {}
        by_ticker: dict[str, list[float]] = {}
        for item in news_items:
            ticker = str(item.get("ticker", "")).upper()
            if not ticker:
                continue
            by_ticker.setdefault(ticker, []).append(float(item.get("sentiment_score", 0.0) or 0.0))

        rationale: list[str] = []
        ticker_scores: dict[str, float] = {}
        for ticker in market_snapshot.universe:
            news_scores = by_ticker.get(ticker, [])
            sentiment[ticker] = sum(news_scores) / len(news_scores) if news_scores else 0.0
            feature = feature_snapshot.features.get(ticker, {})
            momentum = float(feature.get("momentum_15m", 0.0))
            day_change = float(feature.get("day_change_pct", 0.0))
            ticker_scores[ticker] = momentum + (0.5 * sentiment[ticker]) + (0.25 * day_change)
            rationale.append(
                f"{ticker}: momentum_15m={momentum:.4f}, sentiment={sentiment[ticker]:.4f}"
            )

        confidence = min(0.9, 0.45 + (sum(abs(v) for v in sentiment.values()) / max(len(sentiment), 1)))
        return AgentDecisionTrace(
            agent_name="fallback-intraday-agent",
            summary="Fallback agent synthesized intraday momentum and sentiment signals.",
            confidence=confidence,
            sentiment=sentiment,
            ticker_scores=ticker_scores,
            rationale=rationale,
            prompt_version="fallback-v1",
            model_version="rule-based-v1",
            provider="fallback",
        )


class OpenAIIntradayAgent:
    """Primary intraday orchestration agent backed by OpenAI with fallback."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        client: httpx.Client | None = None,
        base_url: str = "https://api.openai.com",
        model: str | None = None,
        timeout: float = 30.0,
        fallback_agent: FallbackIntradayAgent | None = None,
    ) -> None:
        resolved_key = api_key or os.getenv("OPENAI_API_KEY")
        if not resolved_key:
            raise EnvironmentError("OPENAI_API_KEY is required for OpenAIIntradayAgent")

        self.api_key = resolved_key
        self.base_url = base_url.rstrip("/")
        self.model = model or os.getenv("MLCOUNCIL_OPENAI_INTRADAY_MODEL", "gpt-4o-mini")
        self._client = client or httpx.Client(base_url=self.base_url, timeout=timeout)
        self._fallback_agent = fallback_agent or FallbackIntradayAgent()

    def synthesize(
        self,
        *,
        market_snapshot: MarketSnapshot,
        feature_snapshot: FeatureSnapshot,
        news_items: list[dict],
    ) -> AgentDecisionTrace:
        payload = self._build_payload(
            market_snapshot=market_snapshot,
            feature_snapshot=feature_snapshot,
            news_items=news_items,
        )
        try:
            response = self._client.post(
                "/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
            )
            response.raise_for_status()
            body = response.json()
            content = self._extract_content(body)
            parsed = json.loads(content)
            return AgentDecisionTrace(
                agent_name="openai-intraday-agent",
                summary=str(parsed["summary"]),
                confidence=float(parsed["confidence"]),
                sentiment={k: float(v) for k, v in parsed.get("sentiment", {}).items()},
                ticker_scores={k: float(v) for k, v in parsed.get("ticker_scores", {}).items()},
                rationale=[str(item) for item in parsed.get("rationale", [])],
                prompt_version="openai-intraday-v1",
                model_version=self.model,
                provider="openai",
                request_id=body.get("id"),
            )
        except Exception:
            return self._fallback_agent.synthesize(
                market_snapshot=market_snapshot,
                feature_snapshot=feature_snapshot,
                news_items=news_items,
            )

    def _build_payload(
        self,
        *,
        market_snapshot: MarketSnapshot,
        feature_snapshot: FeatureSnapshot,
        news_items: list[dict],
    ) -> dict[str, Any]:
        prompt = {
            "market_snapshot": market_snapshot.to_dict(),
            "feature_snapshot": feature_snapshot.to_dict(),
            "news_items": news_items[:20],
        }
        return {
            "model": self.model,
            "temperature": 0.2,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are an intraday US equities orchestrator. "
                        "Return only JSON following the schema and infer ticker-level conviction."
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(prompt),
                },
            ],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "intraday_agent_trace",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "summary": {"type": "string"},
                            "confidence": {"type": "number"},
                            "sentiment": {
                                "type": "object",
                                "additionalProperties": {"type": "number"},
                            },
                            "ticker_scores": {
                                "type": "object",
                                "additionalProperties": {"type": "number"},
                            },
                            "rationale": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                        },
                        "required": [
                            "summary",
                            "confidence",
                            "sentiment",
                            "ticker_scores",
                            "rationale",
                        ],
                        "additionalProperties": False,
                    },
                    "strict": True,
                },
            },
        }

    @staticmethod
    def _extract_content(body: dict[str, Any]) -> str:
        choices = body.get("choices", [])
        if not choices:
            raise ValueError("OpenAI response missing choices")
        message = choices[0].get("message", {})
        content = message.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            text_parts = []
            for part in content:
                if isinstance(part, dict) and "text" in part:
                    text_parts.append(str(part["text"]))
            return "".join(text_parts)
        raise ValueError("Unsupported OpenAI response content format")
