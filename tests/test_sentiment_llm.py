"""Tests for models/sentiment_llm.py — no GPU, no model downloads."""

from __future__ import annotations

import pandas as pd
import pytest

from data.store.vector_store import VectorStore
from models.sentiment_llm import (
    LLMSentimentScorer,
    apply_hallucination_guard,
    llm_sentiment_shadow_enabled,
    log_shadow_scores,
    parse_sentiment_score,
    run_shadow_comparison,
)


def test_parse_sentiment_score_kv():
    assert parse_sentiment_score("sentiment=0.42") == pytest.approx(0.42)


def test_parse_sentiment_score_json():
    assert parse_sentiment_score('{"sentiment_score": -0.25}') == pytest.approx(-0.25)


def test_parse_sentiment_score_clamps():
    assert parse_sentiment_score("sentiment=2.5") == 1.0
    assert parse_sentiment_score("sentiment=-3") == -1.0


def test_hallucination_guard_unparseable():
    score, ok = apply_hallucination_guard(None)
    assert score == 0.0
    assert ok is False


def test_hallucination_guard_parseable():
    score, ok = apply_hallucination_guard(0.6)
    assert score == 0.6
    assert ok is True


def test_scorer_uses_injected_llm():
    def fake_llm(prompt: str) -> str:
        return "sentiment=0.75"

    scorer = LLMSentimentScorer(llm_fn=fake_llm)
    result = scorer.score_text("Company beats earnings", ticker="AAPL")
    assert result.score == pytest.approx(0.75)
    assert result.parseable is True
    assert result.backend == "injected"


def test_scorer_hallucination_guard_zero():
    scorer = LLMSentimentScorer(llm_fn=lambda _: "not a number at all")
    result = scorer.score_text("random headline")
    assert result.score == 0.0
    assert result.parseable is False


def test_scorer_keyword_mock_backend(monkeypatch):
    monkeypatch.setenv("MLCOUNCIL_LLM_SENTIMENT_MOCK", "true")
    scorer = LLMSentimentScorer()
    pos = scorer.score_text("Company beats earnings and strong growth profit")
    neg = scorer.score_text("Company misses earnings loss decline warning")
    assert pos.score > 0
    assert neg.score < 0


def test_rag_retrieval_in_prompt():
    store = VectorStore(collection_name="test", force_mock=True)
    store.upsert_passages(
        "AAPL",
        "10-K",
        "2024-01-01",
        ["Apple revenue growth exceeded expectations"],
    )

    captured: list[str] = []

    def capture(prompt: str) -> str:
        captured.append(prompt)
        return "sentiment=0.1"

    scorer = LLMSentimentScorer(vector_store=store, llm_fn=capture)
    scorer.score_text("Apple revenue growth", ticker="AAPL")
    assert "SEC filings" in captured[0]
    assert "revenue growth" in captured[0]


def test_predict_shadow_z_score():
    responses = iter(["sentiment=0.8", "sentiment=-0.2"])

    def varied(_: str) -> str:
        return next(responses)

    scorer = LLMSentimentScorer(llm_fn=varied)
    df = pd.DataFrame(
        {
            "ticker": ["A", "B"],
            "title": ["good beat growth", "bad miss loss"],
        }
    )
    out = scorer.predict_shadow(df)
    assert len(out) == 2
    assert abs(out.mean()) < 1e-9


def test_log_shadow_scores(tmp_path):
    fin = pd.Series({"AAPL": 0.2, "MSFT": -0.1})
    llm = pd.Series({"AAPL": 0.3, "MSFT": 0.0})
    path = log_shadow_scores("2024-06-01", fin, llm, output_dir=tmp_path)
    assert path.exists()
    df = pd.read_parquet(path)
    assert set(df.columns) >= {"ticker", "finbert_score", "llm_score"}


def test_run_shadow_comparison(tmp_path, monkeypatch):
    monkeypatch.setenv("MLCOUNCIL_LLM_SENTIMENT_MOCK", "true")
    monkeypatch.setattr("models.sentiment_llm._SHADOW_DIR", tmp_path)
    news = pd.DataFrame({"ticker": ["AAPL"], "title": ["beats earnings"]})
    fin = pd.Series({"AAPL": 0.1})
    report = run_shadow_comparison(
        news,
        fin,
        partition_date="2024-06-01",
        scorer=LLMSentimentScorer(llm_fn=lambda _: "sentiment=0.2"),
    )
    assert "shadow_path" in report
    assert (tmp_path / "2024-06-01.parquet").exists()


def test_shadow_env_flag(monkeypatch):
    monkeypatch.delenv("MLCOUNCIL_LLM_SENTIMENT_SHADOW", raising=False)
    assert llm_sentiment_shadow_enabled() is False
    monkeypatch.setenv("MLCOUNCIL_LLM_SENTIMENT_SHADOW", "true")
    assert llm_sentiment_shadow_enabled() is True


def test_vector_store_mock_backend():
    store = VectorStore(force_mock=True)
    n = store.upsert_passages("MSFT", "10-Q", "2024-03-01", ["Microsoft cloud revenue"])
    assert n == 1
    hits = store.retrieve("Microsoft cloud", top_k=2, ticker="MSFT")
    assert len(hits) >= 1
    assert hits[0].score > 0
