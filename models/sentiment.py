"""FinBERT sentiment alpha model.

Architecture
------------
* Loads ``ProsusAI/finbert`` (or ``yiyanghkust/finbert-tone`` as a lighter
  fallback) via the HuggingFace Transformers pipeline.
* ``score_headlines()`` converts the three-class output into a scalar in
  [-1, +1]: ``score = P(positive) − P(negative)``.
* Scores are persisted to a SQLite cache so identical headlines are never
  re-scored.
* ``predict()`` groups news by ticker, applies recency decay + source
  weighting, then cross-sectionally z-scores the result.
* ``fit()`` calibrates the recency decay factor (``gamma``) to maximise
  Spearman IC against forward returns on historical data.  The transformer
  weights themselves are never fine-tuned.

CPU throughput note
-------------------
FinBERT on a modern CPU processes ~100 headlines/s.  For a 50-stock universe
with 3 headlines/stock/day (150 total), prediction takes < 2 s on first run;
subsequent runs hit the cache and are near-instant.
"""

from __future__ import annotations

from datetime import date, datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import polars as pl
import yaml

from .base import BaseModel

_ROOT = Path(__file__).parents[1]

# Columns that are not feature inputs
_META_COLS = {"ticker", "valid_time", "date", "transaction_time"}


class SentimentModel(BaseModel):
    """FinBERT-based sentiment alpha model.

    Parameters
    ----------
    config_path:
        Path to ``config/models.yaml``.  The ``sentiment`` key is read for
        model name, batch size, decay prior, etc.
    cache:
        Optional pre-built ``SentimentCache`` instance.  When ``None`` the
        model lazily creates one at ``data/cache/sentiment.db`` on first use.
        Pass ``SentimentCache(":memory:")`` in tests to avoid disk I/O.
    """

    name = "sentiment"

    def __init__(
        self,
        config_path: str = "config/models.yaml",
        cache=None,
    ) -> None:
        with open(config_path) as f:
            cfg = yaml.safe_load(f)

        sent_cfg: dict = cfg.get("sentiment", {})
        self._model_name: str = sent_cfg.get("model_name", "ProsusAI/finbert")
        self._fallback_model: str = sent_cfg.get(
            "fallback_model", "yiyanghkust/finbert-tone"
        )
        self._recency_decay: float = float(sent_cfg.get("recency_decay", 0.7))
        self._batch_size: int = int(sent_cfg.get("batch_size", 32))
        self._max_length: int = int(sent_cfg.get("max_length", 512))
        self._params: dict = sent_cfg

        # Runtime state
        self._pipeline = None          # HuggingFace pipeline (lazy-loaded)
        self._decay: float = self._recency_decay  # calibrated by fit()
        self._cache = cache            # SentimentCache (lazy-created if None)
        self._last_trained: str | None = None
        self._n_features: int | None = None

    # ------------------------------------------------------------------
    # Pipeline loading
    # ------------------------------------------------------------------

    def _load_pipeline(self) -> None:
        """Lazy-load the HuggingFace FinBERT pipeline (first call only)."""
        if self._pipeline is not None:
            return

        import torch
        from transformers import pipeline

        device = 0 if torch.cuda.is_available() else -1

        try:
            self._pipeline = pipeline(
                "text-classification",
                model=self._model_name,
                device=device,
                top_k=None,
                truncation=True,
                max_length=self._max_length,
            )
        except Exception:
            # Fall back to the lighter model (faster download, slightly lower accuracy)
            self._pipeline = pipeline(
                "text-classification",
                model=self._fallback_model,
                device=device,
                top_k=None,
                truncation=True,
                max_length=self._max_length,
            )

    def _get_cache(self):
        """Return the sentiment cache, creating it on first access."""
        if self._cache is None:
            from data.ingest.news_processor import SentimentCache
            self._cache = SentimentCache()
        return self._cache

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def _run_pipeline(self, texts: list[str]) -> list[float]:
        """Call the HuggingFace pipeline and convert to [-1, +1] floats.

        Separated from ``score_headlines`` so tests can mock this method
        without loading any transformer weights.

        Parameters
        ----------
        texts:
            Pre-truncated headline strings (already cleaned).

        Returns
        -------
        list[float] of the same length as *texts*.
        """
        raw = self._pipeline(texts)
        scores: list[float] = []
        for result in raw:
            # result: [{"label": "positive", "score": 0.9}, ...]
            label_scores = {r["label"].lower(): r["score"] for r in result}
            pos = label_scores.get("positive", 0.0)
            neg = label_scores.get("negative", 0.0)
            scores.append(float(pos - neg))
        return scores

    def score_headlines(self, headlines: list[str]) -> list[float]:
        """Score a list of financial headlines.

        Checks the persistent cache first; only calls the transformer for
        cache misses, then stores the new scores.

        Parameters
        ----------
        headlines:
            Raw or pre-cleaned headline strings.

        Returns
        -------
        list[float] in [-1, +1] — same order as input.
        ``-1`` = strongly negative, ``0`` = neutral, ``+1`` = strongly positive.
        """
        if not headlines:
            return []

        self._load_pipeline()
        cache = self._get_cache()

        output: list[Optional[float]] = [None] * len(headlines)
        miss_idx: list[int] = []
        miss_texts: list[str] = []

        for i, h in enumerate(headlines):
            cached = cache.get(h)
            if cached is not None:
                output[i] = cached
            else:
                miss_idx.append(i)
                miss_texts.append(h[:self._max_length])

        if miss_texts:
            raw_scores: list[float] = []
            for j in range(0, len(miss_texts), self._batch_size):
                batch = miss_texts[j : j + self._batch_size]
                raw_scores.extend(self._run_pipeline(batch))

            for list_pos, (orig_idx, score) in enumerate(
                zip(miss_idx, raw_scores)
            ):
                output[orig_idx] = score
                cache.set(headlines[orig_idx], score)

        return output  # type: ignore[return-value]  # all slots filled above

    # ------------------------------------------------------------------
    # fit — calibrate aggregation weights
    # ------------------------------------------------------------------

    def fit(self, features: pl.DataFrame, targets: pd.Series) -> None:
        """Calibrate the recency decay factor on historical data.

        Uses ``scipy.optimize.minimize_scalar`` (bounded Brent method) to
        find the ``gamma`` in [0.30, 0.95] that maximises |Spearman IC|
        between aggregated sentiment scores and cross-sectional forward
        returns.

        The transformer weights are **not** modified.

        Parameters
        ----------
        features:
            pl.DataFrame with columns ``ticker``, ``valid_time`` (or
            ``date``), and ``headlines`` (``List[Utf8]``) or ``title``
            (``Utf8``).  Optionally a ``source`` column.
        targets:
            pd.Series of cross-sectional rank percentiles.  Index may be a
            ``(ticker, valid_time)`` MultiIndex or a flat index aligned
            row-by-row with *features*.
        """
        from scipy.optimize import minimize_scalar
        from scipy.stats import spearmanr

        date_col = "valid_time" if "valid_time" in features.columns else "date"
        dates_sorted = sorted(features[date_col].unique().to_list())
        ref_date = dates_sorted[-1]

        ticker_news = self._build_ticker_news(features, date_col)

        # Pre-score every unique headline once (uses cache)
        all_headlines = list(
            {h for news in ticker_news.values() for h, _, _ in news}
        )
        if not all_headlines:
            self._last_trained = datetime.now().isoformat()
            return
        raw_scores_list = self.score_headlines(all_headlines)
        scores_lookup: dict[str, float] = dict(zip(all_headlines, raw_scores_list))

        # Build pre-scored structure: {ticker: [(h, pub_date, sw, score), ...]}
        pre_scored: dict[str, list[tuple]] = {
            ticker: [
                (h, pub_date, sw, scores_lookup.get(h, 0.0))
                for h, pub_date, sw in news
            ]
            for ticker, news in ticker_news.items()
        }

        # Align targets to the reference date
        if isinstance(targets.index, pd.MultiIndex):
            level_vals = targets.index.get_level_values(1)
            # normalise both sides to Python date
            ref_as_date = ref_date if isinstance(ref_date, date) else pd.Timestamp(ref_date).date()
            try:
                norm_level = pd.Series(level_vals).apply(
                    lambda x: x.date() if hasattr(x, "date") and callable(x.date) else x
                )
                mask = norm_level == ref_as_date
                tgt_slice = targets.iloc[mask.values]
                tgt_slice.index = targets.index.get_level_values(0)[mask.values]
            except Exception:
                tgt_slice = pd.Series(dtype=float)
        else:
            tgt_slice = targets

        def _neg_ic(decay: float) -> float:
            agg: dict[str, float] = {}
            for ticker, items in pre_scored.items():
                if not items:
                    continue
                dates = [it[1] for it in items if it[1] is not None]
                if not dates:
                    continue
                ref = max(dates)
                ws = tw = 0.0
                for _, pub_date, sw, score in items:
                    days = (
                        max(0, (_to_date(ref) - _to_date(pub_date)).days)
                        if pub_date is not None
                        else 0
                    )
                    w = (decay ** days) * sw
                    ws += score * w
                    tw += w
                if tw > 0:
                    agg[ticker] = ws / tw

            tickers = list(agg.keys())
            if len(tickers) < 5:
                return 0.0
            s_vals = np.array([agg[t] for t in tickers])
            t_vals = np.array([
                float(tgt_slice.get(t, np.nan)) if hasattr(tgt_slice, "get")
                else np.nan
                for t in tickers
            ])
            mask = ~np.isnan(t_vals)
            s_vals, t_vals = s_vals[mask], t_vals[mask]
            if len(s_vals) < 5:
                return 0.0
            ic, _ = spearmanr(s_vals, t_vals)
            return -abs(float(ic)) if not np.isnan(ic) else 0.0

        result = minimize_scalar(_neg_ic, bounds=(0.30, 0.95), method="bounded")
        self._decay = float(result.x)
        self._last_trained = datetime.now().isoformat()
        self._n_features = len(features.columns)

    # ------------------------------------------------------------------
    # predict
    # ------------------------------------------------------------------

    def predict(self, features: pl.DataFrame) -> pd.Series:
        """Generate cross-sectional sentiment alpha scores.

        Parameters
        ----------
        features:
            pl.DataFrame with at minimum: ``ticker``, ``valid_time`` (or
            ``date``), and either:

            * ``headlines``: ``List[Utf8]`` column — one list of headline
              strings per ``(ticker, date)`` row (grouped format), **or**
            * ``title``: ``Utf8`` column — one headline per row (raw news
              format as produced by ``download_news``).

            An optional ``source`` column is used for source weighting.

        Returns
        -------
        pd.Series
            index = ticker (str), values = cross-sectional z-scored sentiment.
            Positive = bullish signal, negative = bearish signal.
            Tickers with no news receive score 0.0 (neutral) before z-scoring.
        """
        date_col = "valid_time" if "valid_time" in features.columns else "date"
        all_tickers = features["ticker"].unique().to_list()

        ticker_news = self._build_ticker_news(features, date_col)
        agg = self._aggregate_scores(ticker_news)

        raw = pd.Series({t: agg.get(t, np.nan) for t in all_tickers})

        # Z-score only over tickers that have actual news scores.
        # Including no-news tickers (filled to 0.0) in the z-score distribution
        # biases the mean and makes those tickers look artificially bearish on
        # days when the rest of the universe has positive sentiment.
        has_signal = raw.notna()
        if has_signal.any():
            signal = raw[has_signal]
            std = signal.std()
            if std > 1e-9:
                normalized = (signal - signal.mean()) / std
            else:
                # All scored tickers have the same sentiment — flat day.
                normalized = pd.Series(0.0, index=signal.index)
            # No-news tickers receive 0.0 (neutral) without distorting the distribution.
            result = normalized.reindex(raw.index).fillna(0.0)
        else:
            # No news at all for this date — return neutral for every ticker.
            result = pd.Series(0.0, index=raw.index)

        result.index.name = "ticker"
        return result

    # ------------------------------------------------------------------
    # Aggregation
    # ------------------------------------------------------------------

    def _aggregate_scores(
        self,
        ticker_news: dict[str, list[tuple[str, date, float]]],
    ) -> dict[str, float]:
        """Aggregate per-ticker headlines into a single sentiment score.

        Calls ``score_headlines`` (with cache) for each ticker's headlines,
        then applies recency decay and source weighting.

        Parameters
        ----------
        ticker_news:
            ``{ticker: [(headline, pub_date, source_weight), ...]}``

            * *headline*: cleaned text string.
            * *pub_date*: publication date (``date`` or ``datetime``).
              Used to compute days-ago relative to the most recent item.
            * *source_weight*: credibility multiplier from
              ``assign_source_weight()``.

        Returns
        -------
        dict mapping ticker → weighted-average sentiment score.
        Tickers with no scoreable headlines are absent from the result.
        """
        result: dict[str, float] = {}

        for ticker, news_list in ticker_news.items():
            if not news_list:
                continue

            headlines = [item[0] for item in news_list]
            scores = self.score_headlines(headlines)

            # Reference date = most recent publication date in this ticker's news
            dates = [item[1] for item in news_list if item[1] is not None]
            ref = max(dates, key=_to_date) if dates else None

            ws = tw = 0.0
            for (headline, pub_date, sw), score in zip(news_list, scores):
                if ref is not None and pub_date is not None:
                    try:
                        days = max(0, (_to_date(ref) - _to_date(pub_date)).days)
                    except Exception:
                        days = 0
                else:
                    days = 0

                w = (self._decay ** days) * sw
                ws += score * w
                tw += w

            if tw > 0:
                result[ticker] = ws / tw

        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_ticker_news(
        self,
        features: pl.DataFrame,
        date_col: str,
    ) -> dict[str, list[tuple[str, date, float]]]:
        """Extract a ``{ticker: [(headline, pub_date, source_weight)]}`` dict.

        Handles both the grouped format (``headlines: List[Utf8]``) and the
        raw single-headline format (``title: Utf8``).
        """
        from data.ingest.news_processor import assign_source_weight, clean_headline

        has_headlines = "headlines" in features.columns
        has_title = "title" in features.columns
        has_source = "source" in features.columns

        ticker_news: dict[str, list] = {}

        for row in features.to_dicts():
            ticker: str = row["ticker"]
            pub_date = row.get(date_col)
            if pub_date is not None and hasattr(pub_date, "date") and callable(pub_date.date):
                pub_date = pub_date.date()

            source: str = (row.get("source") or "") if has_source else ""
            sw = assign_source_weight(source)

            if has_headlines:
                raw_headlines = row.get("headlines") or []
            elif has_title:
                t = row.get("title") or ""
                raw_headlines = [t] if t else []
            else:
                raw_headlines = []

            if ticker not in ticker_news:
                ticker_news[ticker] = []

            for h in raw_headlines:
                cleaned = clean_headline(str(h))
                if cleaned:
                    ticker_news[ticker].append((cleaned, pub_date, sw))

        return ticker_news


# ---------------------------------------------------------------------------
# Internal utility
# ---------------------------------------------------------------------------

def _to_date(d) -> date:
    """Coerce a date / datetime / Timestamp to a ``datetime.date``."""
    if isinstance(d, date) and not isinstance(d, datetime):
        return d
    if isinstance(d, datetime):
        return d.date()
    try:
        return pd.Timestamp(d).date()
    except Exception:
        return date.min
