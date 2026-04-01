"""Tests for models/sentiment.py and data/ingest/news_processor.py.

Test strategy
-------------
All tests that touch SentimentModel use a pytest fixture that:
  * bypasses transformer download by setting _pipeline to a sentinel (non-None)
  * injects an in-memory SentimentCache (db_path=":memory:")
  * mocks _run_pipeline via unittest.mock.patch.object

This makes every test deterministic and fast (<1 s total).

The actual FinBERT model is never downloaded during the test suite.
Mark tests with @pytest.mark.integration and opt-in with
  pytest -m integration
to run against the real model.
"""

from __future__ import annotations

from datetime import date
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import polars as pl
import pytest

from data.ingest.news_processor import (
    SentimentCache,
    assign_source_weight,
    clean_headline,
    deduplicate_news,
)
from models.sentiment import SentimentModel


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

def _make_model(decay: float = 0.7) -> SentimentModel:
    """Return a SentimentModel with a mocked pipeline and in-memory cache."""
    cache = SentimentCache(db_path=":memory:")
    model = SentimentModel.__new__(SentimentModel)
    model._model_name = "ProsusAI/finbert"
    model._fallback_model = "yiyanghkust/finbert-tone"
    model._recency_decay = decay
    model._decay = decay
    model._batch_size = 32
    model._max_length = 512
    model._params = {}
    model._pipeline = True      # truthy → _load_pipeline() is a no-op
    model._cache = cache
    model._last_trained = None
    model._n_features = None
    return model


# FinBERT-style output: list of lists of {"label": ..., "score": ...}
def _finbert_output(pos: float, neg: float) -> list[list[dict]]:
    neutral = max(0.0, 1.0 - pos - neg)
    return [[
        {"label": "positive", "score": pos},
        {"label": "negative", "score": neg},
        {"label": "neutral",  "score": neutral},
    ]]


# ---------------------------------------------------------------------------
# 1. test_score_positive
# ---------------------------------------------------------------------------

def test_score_positive():
    """A clearly bullish headline should produce score > 0."""
    model = _make_model()

    with patch.object(model, "_run_pipeline") as mock_pipe:
        # Simulate FinBERT returning high positive probability
        mock_pipe.return_value = [0.82]   # pos=0.88 neg=0.06 → 0.82
        scores = model.score_headlines(["Company beats earnings expectations"])

    assert len(scores) == 1
    assert scores[0] > 0, f"Expected positive score, got {scores[0]}"


# ---------------------------------------------------------------------------
# 2. test_score_negative
# ---------------------------------------------------------------------------

def test_score_negative():
    """A clearly bearish headline should produce score < 0."""
    model = _make_model()

    with patch.object(model, "_run_pipeline") as mock_pipe:
        mock_pipe.return_value = [-0.75]  # neg >> pos
        scores = model.score_headlines(
            ["Company misses revenue forecast, stock falls"]
        )

    assert len(scores) == 1
    assert scores[0] < 0, f"Expected negative score, got {scores[0]}"


# ---------------------------------------------------------------------------
# 3. test_score_neutral
# ---------------------------------------------------------------------------

def test_score_neutral():
    """A neutral headline should produce a score close to 0."""
    model = _make_model()

    with patch.object(model, "_run_pipeline") as mock_pipe:
        # pos=0.25, neg=0.20 → score = 0.05
        mock_pipe.return_value = [0.05]
        scores = model.score_headlines(["Company announces Q3 earnings date"])

    assert len(scores) == 1
    assert abs(scores[0]) < 0.5, (
        f"Expected near-neutral score, got {scores[0]}"
    )


# ---------------------------------------------------------------------------
# 4. test_aggregation_recency
# ---------------------------------------------------------------------------

def test_aggregation_recency():
    """More recent news should carry more weight than older news.

    Setup
    -----
    * Two tickers (A, B) each have one positive headline.
    * Ticker A's headline is from day -1  (weight = gamma^0 = 1.0).
    * Ticker B's headline is from day -3  (weight = gamma^2 ≈ 0.49).
    * Both have source_weight = 1.0.

    Expected: score(A) > score(B) when the raw sentiment is equal,
    because _aggregate_scores normalises by total weight, so the
    ticker with fresher news has a higher effective signal when other
    news would dilute the older ticker — but the real check is that if
    we add a *neutral* headline from day -1 for ticker B, A comes out
    higher.

    Simpler direct check: give A only a positive recent item; give B
    only a positive old item.  With equal raw scores the weighted
    average is the same (only one item each), so instead we pair each
    positive headline with a negative one of different recency:

    * A: positive day-1 (w=1.0) + negative day-3 (w=0.49) → agg > 0
    * B: negative day-1 (w=1.0) + positive day-3 (w=0.49) → agg < 0
    """
    model = _make_model(decay=0.7)

    today = date(2024, 3, 1)
    day_minus_2 = date(2024, 2, 28)

    ticker_news = {
        "GOOD": [
            ("positive headline", today,       1.0),   # recent
            ("negative headline", day_minus_2, 1.0),   # older
        ],
        "BAD": [
            ("negative headline", today,       1.0),   # recent
            ("positive headline", day_minus_2, 1.0),   # older
        ],
    }

    with patch.object(model, "_run_pipeline") as mock_pipe:
        # score_headlines is called per-ticker
        # GOOD: ["positive headline", "negative headline"] → [+0.8, -0.8]
        # BAD:  ["negative headline", "positive headline"] → [-0.8, +0.8]
        mock_pipe.side_effect = lambda texts: [
            0.8 if "positive" in t else -0.8 for t in texts
        ]
        agg = model._aggregate_scores(ticker_news)

    assert "GOOD" in agg and "BAD" in agg
    assert agg["GOOD"] > 0, f"GOOD ticker should be positive, got {agg['GOOD']}"
    assert agg["BAD"] < 0, f"BAD ticker should be negative, got {agg['BAD']}"
    assert agg["GOOD"] > agg["BAD"]


# ---------------------------------------------------------------------------
# 5. test_cross_sectional_zscore
# ---------------------------------------------------------------------------

def test_cross_sectional_zscore():
    """predict() output should be cross-sectionally z-scored (mean≈0, std≈1)."""
    n_tickers = 10
    tickers = [f"T{i:02d}" for i in range(n_tickers)]
    today = date(2024, 3, 1)

    # Build a grouped features DataFrame: one row per ticker with one headline
    features = pl.DataFrame({
        "ticker":     tickers,
        "valid_time": [today] * n_tickers,
        "headlines":  [[f"headline for {t}"] for t in tickers],
        "source":     ["Reuters"] * n_tickers,
    })

    model = _make_model()

    # Assign raw scores that vary across tickers
    raw_score_map = {f"headline for {t}": float(i) * 0.1 for i, t in enumerate(tickers)}

    with patch.object(model, "_run_pipeline") as mock_pipe:
        mock_pipe.side_effect = lambda texts: [raw_score_map.get(t, 0.0) for t in texts]
        result = model.predict(features)

    assert isinstance(result, pd.Series)
    assert set(result.index) == set(tickers)
    assert abs(result.mean()) < 1e-9, f"Mean should be ~0, got {result.mean()}"
    assert abs(result.std() - 1.0) < 1e-9, f"Std should be ~1, got {result.std()}"


# ---------------------------------------------------------------------------
# 6. test_cache_hit
# ---------------------------------------------------------------------------

def test_cache_hit():
    """Scoring the same headline twice should call _run_pipeline only once."""
    model = _make_model()

    headline = "Company reports record quarterly profit"

    with patch.object(model, "_run_pipeline") as mock_pipe:
        mock_pipe.return_value = [0.72]

        scores_first = model.score_headlines([headline])
        scores_second = model.score_headlines([headline])

    # Pipeline called exactly once despite two calls to score_headlines
    mock_pipe.assert_called_once()
    assert scores_first == scores_second, (
        "Cached score should match original score"
    )


# ---------------------------------------------------------------------------
# news_processor unit tests
# ---------------------------------------------------------------------------

def test_clean_headline_removes_html():
    assert "<b>" not in clean_headline("<b>Stock surges</b> after earnings")


def test_clean_headline_removes_ticker_mentions():
    result = clean_headline("[AAPL] $MSFT reports strong growth")
    assert "[AAPL]" not in result
    assert "$MSFT" not in result
    assert "reports strong growth" in result


def test_clean_headline_truncates_to_512():
    long_text = "word " * 200   # 1000 chars
    assert len(clean_headline(long_text)) <= 512


def test_assign_source_weight_tiers():
    assert assign_source_weight("Reuters") == 1.0
    assert assign_source_weight("Bloomberg") == 1.0
    assert assign_source_weight("WSJ") == 1.0
    assert assign_source_weight("Benzinga") == 0.7
    assert assign_source_weight("Seeking Alpha") == 0.7
    assert assign_source_weight("Some Random Blog") == 0.5
    assert assign_source_weight("") == 0.5


def test_deduplicate_news_url():
    df = pl.DataFrame({
        "ticker":     ["AAPL", "AAPL"],
        "valid_time": [date(2024, 1, 1), date(2024, 1, 1)],
        "title":      ["Apple reports profit", "Different title"],
        "url":        ["http://example.com/1", "http://example.com/1"],
        "source":     ["Reuters", "Bloomberg"],
    })
    deduped = deduplicate_news(df)
    assert len(deduped) == 1


def test_deduplicate_news_fuzzy_title():
    df = pl.DataFrame({
        "ticker":     ["AAPL", "AAPL"],
        "valid_time": [date(2024, 1, 1), date(2024, 1, 2)],
        "title":      ["Apple reports profit!", "Apple reports profit"],
        "url":        ["http://example.com/1", "http://example.com/2"],
        "source":     ["Reuters", "Bloomberg"],
    })
    deduped = deduplicate_news(df)
    # The two titles normalise to the same hash → one row survives
    assert len(deduped) == 1


def test_sentiment_cache_miss_and_hit():
    cache = SentimentCache(db_path=":memory:")
    assert cache.get("never seen this") is None

    cache.set("test headline", 0.55)
    result = cache.get("test headline")
    assert result is not None
    assert abs(result - 0.55) < 1e-9


def test_sentiment_cache_overwrite():
    cache = SentimentCache(db_path=":memory:")
    cache.set("headline", 0.3)
    cache.set("headline", 0.9)   # overwrite
    assert abs(cache.get("headline") - 0.9) < 1e-9
