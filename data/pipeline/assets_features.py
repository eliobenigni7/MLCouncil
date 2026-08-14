"""Layer 2 — Features: alpha158_features, sentiment_features + contract check.

Parte del package data/pipeline (ex data/pipeline.py). Gli asset accedono alle
costanti/helper condivisi via ``_shared.NOME`` (vedi data/pipeline/_shared.py).
"""

import numpy as np
import pandas as pd
import polars as pl
import dagster as dg
from dagster import AssetExecutionContext
from datetime import date as date_type

from observability.tracing import trace_span

from . import _shared
from .assets_ingest import raw_ohlcv, raw_macro, raw_news
from ._shared import (
    _DAILY_PARTITIONS,
    _RETRY,
    _EXCLUDE_COLS,
    _MIN_ALPHA_FEATURES,
    _record_asset_metadata,
    _contract_check_result,
)


# ===========================================================================
# LAYER 2 — FEATURES
# ===========================================================================

@dg.asset(
    partitions_def=_DAILY_PARTITIONS,
    retry_policy=_RETRY,
    description="Feature tecniche + macro look-ahead safe per il modello tecnico.",
    deps=[raw_ohlcv, raw_macro],
)
def alpha158_features(
    context: AssetExecutionContext,
) -> pl.DataFrame:
    """Calcola le feature Alpha158 sull'OHLCV storico + contesto macro.

    Per le rolling window (fino a 252 giorni) carica l'intera storia
    disponibile su disco, poi restituisce solo le righe del giorno corrente.
    """
    partition_date = context.partition_key
    today = date_type.fromisoformat(partition_date)

    with trace_span(
        "mlcouncil.features.alpha158_features",
        layer="features",
        asset="alpha158_features",
        partition_date=partition_date,
    ):
        return _run_alpha158_features(context, partition_date, today)


def _run_alpha158_features(
    context: AssetExecutionContext,
    partition_date: str,
    today: date_type,
) -> pl.DataFrame:
    from data.features.alpha158 import compute_alpha158

    # Alpha158 richiede la storia completa per le rolling window
    all_ohlcv = _shared._load_all_ohlcv()
    if all_ohlcv.is_empty():
        raise ValueError("Nessun dato OHLCV disponibile per Alpha158")

    macro_ctx = _shared._load_macro_context_from_disk()
    if macro_ctx.is_empty():
        macro_ctx = None
    features = compute_alpha158(all_ohlcv, macro_df=macro_ctx)

    # Filtra al giorno corrente
    day_feat = features.filter(pl.col("valid_time") == today)
    if day_feat.is_empty():
        raise ValueError(f"Nessuna feature calcolata per {partition_date}")

    # Quality checks
    non_meta_cols = [c for c in day_feat.columns if c not in _EXCLUDE_COLS]
    assert len(non_meta_cols) >= _MIN_ALPHA_FEATURES, (
        f"Solo {len(non_meta_cols)} feature, attese almeno {_MIN_ALPHA_FEATURES}"
    )
    float_cols = [
        c for c in non_meta_cols
        if day_feat[c].dtype in (pl.Float32, pl.Float64)
    ]
    if float_cols:
        nan_sum = (
            day_feat.select([pl.col(c).is_nan().sum() for c in float_cols])
            .to_pandas().sum().sum()
        )
        assert nan_sum == 0, f"NaN nelle feature: {nan_sum}"

    context.log.info(
        f"alpha158_features [{partition_date}]: "
        f"{day_feat.shape[0]} righe × {len(non_meta_cols)} feature"
    )
    _record_asset_metadata(context, "alpha158_features", day_feat, partition_date)
    return day_feat


@dg.asset(
    partitions_def=_DAILY_PARTITIONS,
    retry_policy=_RETRY,
    description="Feature di sentiment per ticker (FinBERT su titoli di news).",
    deps=[raw_news],
)
def sentiment_features(
    context: AssetExecutionContext,
) -> pl.DataFrame:
    """Aggrega i punteggi di sentiment FinBERT per ticker.

    Se FinBERT non è disponibile (es. assenza GPU/PyTorch) restituisce
    un DataFrame con sentiment_score = 0.0 per tutti i ticker.
    """
    partition_date = context.partition_key
    today = date_type.fromisoformat(partition_date)

    _empty = pl.DataFrame({
        "ticker":                  pl.Series([], dtype=pl.Utf8),
        "valid_time":              pl.Series([], dtype=pl.Date),
        "sentiment_score":         pl.Series([], dtype=pl.Float64),
        "sentiment_headline_count": pl.Series([], dtype=pl.Int64),
        "sentiment_fallback_count": pl.Series([], dtype=pl.Int64),
    })

    news_path = _shared._DATA_DIR / "raw" / "news" / f"{partition_date}.parquet"
    raw_news = pl.read_parquet(news_path) if news_path.exists() else _empty

    if raw_news.is_empty():
        context.log.warning(
            f"sentiment_features [{partition_date}]: nessuna news disponibile"
        )
        _record_asset_metadata(context, "sentiment_features", _empty, partition_date)
        return _empty

    try:
        from models.sentiment import SentimentModel
    except ImportError as exc:
        context.log.warning(
            f"sentiment_features [{partition_date}]: "
            f"FinBERT non disponibile ({exc}), fallback a 0.0"
        )
        tickers = raw_news["ticker"].unique().to_list()
        fallback = pl.DataFrame({
            "ticker":                  tickers,
            "valid_time":              [today] * len(tickers),
            "sentiment_score":         [0.0] * len(tickers),
            "sentiment_headline_count": [0] * len(tickers),
            "sentiment_fallback_count": [0] * len(tickers),
        })
        _record_asset_metadata(context, "sentiment_features", fallback, partition_date)
        return fallback

    model = SentimentModel()
    date_col = "valid_time" if "valid_time" in raw_news.columns else "date"
    ticker_news = model._build_ticker_news(raw_news, date_col)

    all_headlines = [
        item[0]
        for items in ticker_news.values()
        for item in items
        if item[0]
    ]
    try:
        scored = model.score_headlines(all_headlines)
        headline_scores = dict(zip(all_headlines, scored))
    except Exception:
        headline_scores = {h: 0.0 for h in all_headlines}

    ticker_scores, agg_meta = model.aggregate_scored_headlines(
        ticker_news, headline_scores
    )

    records: list[dict] = []
    for ticker, score in ticker_scores.items():
        n_headlines = len(ticker_news.get(ticker, []))
        n_fallback = sum(1 for _, _, sw in ticker_news.get(ticker, []) if sw == 0.5)
        records.append({
            "ticker":                  ticker,
            "valid_time":              today,
            "sentiment_score":         float(score),
            "sentiment_headline_count": n_headlines,
            "sentiment_fallback_count": n_fallback,
        })

    if not records:
        _record_asset_metadata(context, "sentiment_features", _empty, partition_date)
        return _empty

    df = pl.DataFrame(records)

    try:
        from models.sentiment_llm import LLMSentimentScorer, llm_sentiment_shadow_enabled, log_shadow_scores

        if llm_sentiment_shadow_enabled():
            scorer = LLMSentimentScorer()
            finbert_s = pd.Series(ticker_scores, name="finbert")
            llm_scores: dict[str, float] = {}
            for ticker, items in ticker_news.items():
                texts = [item[0] for item in items if item and item[0]]
                if not texts:
                    llm_scores[ticker] = 0.0
                    continue
                llm_scores[ticker] = float(np.mean([scorer.score_text(t) for t in texts[:5]]))
            log_shadow_scores(partition_date, finbert_s, pd.Series(llm_scores, name="llm"))
            context.log.info(
                f"sentiment_features [{partition_date}]: LLM shadow logged "
                f"({len(llm_scores)} tickers)"
            )
    except Exception as exc:
        context.log.debug(f"sentiment_features [{partition_date}]: LLM shadow skip ({exc})")

    context.log.info(
        f"sentiment_features [{partition_date}]: "
        f"{len(records)} ticker con sentiment | "
        f"headlines={agg_meta.get('headline_count', 0)} "
        f"fallback_sources={agg_meta.get('fallback_count', 0)}"
    )
    _record_asset_metadata(context, "sentiment_features", df, partition_date)
    return df


@dg.asset_check(
    asset=alpha158_features,
    name="alpha158_features_contract",
    blocking=True,
    partitions_def=_DAILY_PARTITIONS,
)
def alpha158_features_contract(alpha158_features: pl.DataFrame) -> dg.AssetCheckResult:
    partition_date = None
    if not alpha158_features.is_empty() and "valid_time" in alpha158_features.columns:
        partition_date = str(alpha158_features["valid_time"].max())
    return _contract_check_result("alpha158_features", alpha158_features, partition_date)


@dg.asset_check(
    asset=sentiment_features,
    name="sentiment_features_contract",
    blocking=True,
    partitions_def=_DAILY_PARTITIONS,
)
def sentiment_features_contract(sentiment_features: pl.DataFrame) -> dg.AssetCheckResult:
    partition_date = None
    if not sentiment_features.is_empty() and "valid_time" in sentiment_features.columns:
        partition_date = str(sentiment_features["valid_time"].max())
    return _contract_check_result("sentiment_features", sentiment_features, partition_date)
