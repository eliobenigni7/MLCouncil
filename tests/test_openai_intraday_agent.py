from __future__ import annotations

import json
from datetime import datetime

import httpx
import pytest

from intraday.contracts import FeatureSnapshot
from intraday.market_data import MarketSnapshot


def test_openai_intraday_agent_parses_structured_response(monkeypatch):
    from intraday.agent import OpenAIIntradayAgent

    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    captured_request: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured_request["json"] = json.loads(request.content.decode("utf-8"))
        return httpx.Response(
            200,
            json={
                "id": "resp_123",
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "summary": "AAPL strongest intraday long candidate.",
                                    "confidence": 0.84,
                                    "sentiment": {"AAPL": 0.7, "MSFT": -0.2},
                                    "ticker_scores": {"AAPL": 0.9, "MSFT": -0.4},
                                    "rationale": [
                                        "AAPL shows positive short-term momentum.",
                                        "MSFT headlines skewed negative.",
                                    ],
                                }
                            )
                        }
                    }
                ],
            },
        )

    client = httpx.Client(
        transport=httpx.MockTransport(handler),
        base_url="https://api.openai.com",
    )
    agent = OpenAIIntradayAgent(
        api_key="sk-test",
        client=client,
        base_url="https://api.openai.com",
        model="gpt-4o-mini",
    )

    trace = agent.synthesize(
        market_snapshot=MarketSnapshot(
            as_of=datetime(2026, 4, 9, 14, 45),
            universe=["AAPL", "MSFT"],
            bars={
                "AAPL": {"close": 201.0, "return_15m": 0.02},
                "MSFT": {"close": 412.0, "return_15m": -0.01},
            },
            source="polygon",
            session="regular",
        ),
        feature_snapshot=FeatureSnapshot(
            as_of="2026-04-09T14:45:00Z",
            features={
                "AAPL": {"momentum_15m": 0.02, "day_change_pct": 0.01},
                "MSFT": {"momentum_15m": -0.01, "day_change_pct": -0.02},
            },
            source="polygon",
            version="intraday-v2",
        ),
        news_items=[
            {"ticker": "AAPL", "headline": "Apple demand still strong"},
            {"ticker": "MSFT", "headline": "Microsoft sees weaker cloud checks"},
        ],
    )

    assert trace.agent_name == "openai-intraday-agent"
    assert trace.prompt_version == "openai-intraday-v1"
    assert trace.model_version == "gpt-4o-mini"
    assert trace.confidence == pytest.approx(0.84)
    assert trace.sentiment["AAPL"] == pytest.approx(0.7)
    assert trace.ticker_scores["MSFT"] == pytest.approx(-0.4)
    assert captured_request["json"]["response_format"]["type"] == "json_schema"


def test_openai_intraday_agent_requires_api_key(monkeypatch):
    from intraday.agent import OpenAIIntradayAgent

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with pytest.raises(EnvironmentError, match="OPENAI_API_KEY"):
        OpenAIIntradayAgent()
