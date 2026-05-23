"""Dagster pipeline: orchestrazione giornaliera dell'intera catena MLCouncil.

Layer 1 (Ingest)  →  raw_ohlcv, raw_news, raw_macro
Layer 2 (Features) → alpha158_features, sentiment_features
Layer 3 (Signals)  → lgbm_signals, sentiment_signals, current_regime
Layer 4 (Council)  → council_signal, portfolio_weights, daily_orders

Schedule: 21:30 ET ogni giorno lavorativo (lun-ven).
Ogni asset è configurato con RetryPolicy(max_retries=2).

Per avviare manualmente:
    dagster job execute -j daily_pipeline

Per avviare il server:
    dagster dev -f data/pipeline.py
"""

import hashlib
import os
import pickle
import sys
from datetime import date as date_type, timedelta
import pytz
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import polars as pl
import yaml
import dagster as dg
from dagster import AssetExecutionContext, RunFailureSensorContext

from council.artifacts import write_artifact_manifest
from data.contracts import LINEAGE_COLUMNS, validate_asset_contract, version_payload
from data.lineage import (
    attach_lineage,
    build_feature_lineage,
    build_pipeline_run_id,
    checkpoint_version,
    dataframe_lineage_columns,
    extract_lineage,
    lineage_artifact_payload,
    merge_lineage,
    merge_versions,
)
from observability.tracing import init_tracing, trace_span

init_tracing(service_name="mlcouncil-dagster")

try:
    from council.production_config import apply_manifest_to_environ

    apply_manifest_to_environ()
except Exception:
    pass

# ---------------------------------------------------------------------------
# Path bootstrap — consente import relativi da qualsiasi working directory
# ---------------------------------------------------------------------------

_ROOT = Path(__file__).parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

_DATA_DIR       = _ROOT / "data" / "raw"
_ORDERS_DIR     = _ROOT / "data" / "orders"
_RESULTS_DIR    = _ROOT / "data" / "results"
_CHECKPOINTS    = _ROOT / "models" / "checkpoints"
_EXCLUDE_COLS   = {"ticker", "valid_time", "transaction_time"}
_MIN_ALPHA_FEATURES = 50
_DEFAULT_PORTFOLIO_VALUE = 100_000.0


class LivePortfolioSnapshotError(RuntimeError):
    """Errore pipeline per snapshot live Alpaca non disponibile o non valido."""


# ---------------------------------------------------------------------------
# Shared config
# ---------------------------------------------------------------------------

_DAILY_PARTITIONS = dg.DailyPartitionsDefinition(start_date="2018-01-01")
_RETRY            = dg.RetryPolicy(max_retries=2, delay=30)


def _safe_pickle_load(path: Path):
    """Carica un checkpoint pickle solo con sidecar SHA-256 obbligatorio."""
    from council.pickle_security import trusted_pickle_load

    return trusted_pickle_load(path, require_hash=True)


def _load_universe(include_crypto: bool = True) -> list[str]:
    """Carica la lista dei ticker da config/universe.yaml.

    Supporta sia il formato legacy con `universe.tickers` sia il formato
    bucketed corrente (`large_cap`, `mid_cap`, ...), ignorando la sezione
    `settings`. Include anche `crypto_universe` se presente e include_crypto=True.
    """
    with open(_ROOT / "config" / "universe.yaml") as f:
        cfg = yaml.safe_load(f)

    tickers: list[str] = []
    seen: set[str] = set()

    # Equity universe
    universe_cfg = cfg.get("universe", {})
    if isinstance(universe_cfg.get("tickers"), list):
        equity_tickers = universe_cfg["tickers"]
    else:
        equity_tickers = []
        for bucket_name, bucket_values in universe_cfg.items():
            if bucket_name == "settings" or not isinstance(bucket_values, list):
                continue
            equity_tickers.extend(bucket_values)

    for ticker in equity_tickers:
        if ticker not in seen:
            seen.add(ticker)
            tickers.append(ticker)

    # Crypto universe
    if include_crypto:
        crypto_cfg = cfg.get("crypto_universe", {})
        if isinstance(crypto_cfg, dict):
            for bucket_values in crypto_cfg.values():
                if not isinstance(bucket_values, list):
                    continue
                for ticker in bucket_values:
                    if ticker not in seen:
                        seen.add(ticker)
                        tickers.append(ticker)
        elif isinstance(crypto_cfg, list):
            # Flat list format
            for ticker in crypto_cfg:
                if ticker not in seen:
                    seen.add(ticker)
                    tickers.append(ticker)

    return tickers


def load_universe_as_of(
    as_of_date: str | date_type | None = None,
    include_crypto: bool = True,
) -> list[str]:
    """Return only tickers that were universe members on *as_of_date*.

    Uses ``config/universe_history.yaml`` which records ``added`` /
    ``removed`` dates per ticker.  Falls back to :func:`_load_universe`
    (full current universe) when the history file is missing or
    *as_of_date* is ``None``.

    Parameters
    ----------
    as_of_date:
        ISO-8601 date string or ``datetime.date``.  ``None`` → current
        universe (no survivorship filtering).
    include_crypto:
        Whether to include crypto tickers (BTCUSD, ETHUSD …).
    """
    if as_of_date is None:
        return _load_universe(include_crypto=include_crypto)

    if isinstance(as_of_date, str):
        as_of_date = date_type.fromisoformat(as_of_date)

    history_path = _ROOT / "config" / "universe_history.yaml"
    if not history_path.exists():
        return _load_universe(include_crypto=include_crypto)

    with open(history_path) as f:
        history = yaml.safe_load(f) or {}

    membership = history.get("membership", {})
    if not membership:
        return _load_universe(include_crypto=include_crypto)

    # Also load the current universe to know which tickers are equity vs crypto
    with open(_ROOT / "config" / "universe.yaml") as f:
        cfg = yaml.safe_load(f)
    crypto_tickers: set[str] = set()
    crypto_cfg = cfg.get("crypto_universe", {})
    if isinstance(crypto_cfg, dict):
        for bucket_values in crypto_cfg.values():
            if isinstance(bucket_values, list):
                crypto_tickers.update(bucket_values)
    elif isinstance(crypto_cfg, list):
        crypto_tickers.update(crypto_cfg)

    tickers: list[str] = []
    for ticker, periods in membership.items():
        if not include_crypto and ticker in crypto_tickers:
            continue
        added = date_type.fromisoformat(str(periods.get("added", "2018-01-01")))
        removed_raw = periods.get("removed")
        removed = date_type.fromisoformat(str(removed_raw)) if removed_raw else None

        if as_of_date >= added and (removed is None or as_of_date < removed):
            tickers.append(ticker)

    return tickers


def _normalize_df(df: pl.DataFrame) -> pl.DataFrame:
    """Normalize datetime columns to UTC timezone for Polars 1.x strict concat.
    Also cast Datetime to Date for compatibility with existing parquet files."""
    if df.is_empty():
        return df
    # Cast Datetime -> Date (UTC midnight) for compatibility
    for c in df.columns:
        if df[c].dtype == pl.Datetime:
            df = df.with_columns(
                pl.col(c).dt.replace_time_zone("UTC").dt.convert_time_zone("UTC").cast(pl.Date)
            )
    return df


def _load_all_ohlcv(extra: pl.DataFrame | None = None) -> pl.DataFrame:
    """Legge tutti i parquet OHLCV storici, con eventuale append di `extra`."""
    ohlcv_dir = _DATA_DIR / "ohlcv"
    frames: list[pl.DataFrame] = []
    if ohlcv_dir.exists():
        for ticker_dir in sorted(ohlcv_dir.iterdir()):
            if not ticker_dir.is_dir():
                continue
            for pq in sorted(ticker_dir.glob("*.parquet")):
                try:
                    frames.append(_normalize_df(pl.read_parquet(pq)))
                except Exception:
                    pass
    if extra is not None and not extra.is_empty():
        frames.append(_normalize_df(extra))
    if not frames:
        return pl.DataFrame()
    return (
        pl.concat(frames)
        .unique(["ticker", "valid_time"])
        .sort(["ticker", "valid_time"])
    )


def _load_partitioned_parquet(subdir: str, partition_date: str) -> pl.DataFrame:
    """Load a partitioned parquet written under data/raw/<subdir>/<date>.parquet."""
    path = _DATA_DIR / "raw" / subdir / f"{partition_date}.parquet"
    if not path.exists():
        return pl.DataFrame()
    try:
        return _normalize_df(pl.read_parquet(path))
    except Exception:
        return pl.DataFrame()


def _load_macro_context_from_disk() -> pl.DataFrame:
    """Load the macro context parquet files saved by download_macro."""
    macro_dir = _DATA_DIR / "macro"

    def _path(name: str) -> str | None:
        p = macro_dir / f"{name}.parquet"
        return str(p) if p.exists() else None

    from data.features.alpha158 import build_macro_context

    return build_macro_context(
        vix_path=_path("vix"),
        treasuries_path=_path("treasuries"),
        sp500_path=_path("sp500"),
    )


def _record_asset_metadata(
    context: AssetExecutionContext,
    asset_name: str,
    payload,
    partition_date: str,
    lineage: dict[str, str] | None = None,
) -> dict[str, object]:
    contract_summary = validate_asset_contract(asset_name, payload, partition_date)
    metadata: dict[str, object] = {
        "asset_name": asset_name,
        "partition_date": partition_date,
        "row_count": contract_summary["row_count"],
        "column_count": contract_summary["column_count"],
        "payload_version": version_payload(asset_name, payload, partition_date),
    }
    if lineage:
        metadata.update(lineage_artifact_payload(lineage))
    context.add_output_metadata(metadata)
    return metadata


def _contract_check_result(asset_name: str, payload, partition_date: str | None = None) -> dg.AssetCheckResult:
    try:
        summary = validate_asset_contract(asset_name, payload, partition_date)
    except Exception as exc:
        return dg.AssetCheckResult(
            passed=False,
            metadata={
                "asset_name": asset_name,
                "error": str(exc),
                "partition_date": partition_date or "n/a",
            },
        )

    return dg.AssetCheckResult(
        passed=True,
        metadata={
            "asset_name": asset_name,
            "row_count": summary["row_count"],
            "column_count": summary["column_count"],
            "partition_date": partition_date or "n/a",
        },
    )


def _load_live_portfolio_snapshot(
    target_tickers: list[str] | None = None,
) -> tuple[pd.Series, float]:
    zero_weights = pd.Series(dtype=float, name="current_weight")
    if target_tickers is not None:
        zero_weights = pd.Series(
            0.0, index=target_tickers, dtype=float, name="current_weight"
        )
    try:
        from execution.alpaca_adapter import AlpacaConfig, AlpacaLiveNode

        node = AlpacaLiveNode(AlpacaConfig.from_env())
        account = node.get_account_info()
        portfolio_value = float(account.get("portfolio_value", 0.0) or 0.0)
        if not np.isfinite(portfolio_value) or portfolio_value <= 0:
            raise LivePortfolioSnapshotError(
                f"live portfolio snapshot: invalid portfolio value {portfolio_value!r}"
            )

        positions_df = node.get_all_positions(strict=True)
        if positions_df.empty:
            return zero_weights, portfolio_value

        required_cols = {"symbol", "current_value"}
        missing_cols = sorted(required_cols - set(positions_df.columns))
        if missing_cols:
            raise LivePortfolioSnapshotError(
                "live portfolio snapshot: malformed positions payload "
                f"(missing columns: {', '.join(missing_cols)})"
            )

        if positions_df["symbol"].isna().any():
            raise LivePortfolioSnapshotError(
                "live portfolio snapshot: malformed positions payload (null symbols)"
            )

        current_values = pd.to_numeric(
            positions_df["current_value"], errors="coerce"
        ).astype(float)
        if current_values.isna().any() or not np.isfinite(current_values).all():
            raise LivePortfolioSnapshotError(
                "live portfolio snapshot: malformed positions payload "
                "(invalid current_value)"
            )

        normalized_positions = positions_df.assign(current_value=current_values)
        if normalized_positions["symbol"].duplicated().any():
            # Alpaca puo' restituire lo stesso simbolo da sorgenti multiple
            # (es. TradingClient + endpoint crypto). Manteniamo una sola riga
            # per ticker per evitare di contare due volte la stessa esposizione.
            normalized_positions = normalized_positions.drop_duplicates(
                subset=["symbol"], keep="last"
            )

        current_weights = (
            normalized_positions
            .set_index("symbol")["current_value"]
            .astype(float)
            .div(portfolio_value)
            .rename("current_weight")
        )
        if target_tickers is None:
            return current_weights.sort_index(), portfolio_value
        return current_weights.reindex(target_tickers).fillna(0.0), portfolio_value
    except LivePortfolioSnapshotError:
        raise
    except Exception as exc:
        raise LivePortfolioSnapshotError(
            f"live portfolio snapshot unavailable: {exc}"
        ) from exc


# ===========================================================================
# LAYER 1 — INGEST
# ===========================================================================

@dg.asset(
    partitions_def=_DAILY_PARTITIONS,
    retry_policy=_RETRY,
    description="OHLCV giornaliero per tutto l'universo (yfinance, schema bi-temporale).",
)
def raw_ohlcv(context: AssetExecutionContext) -> None:
    """Scarica e salva i dati OHLCV per la data di partizione."""
    from data.ingest.market_data import download_daily

    partition_date = context.partition_key
    with trace_span(
        "mlcouncil.ingest.raw_ohlcv",
        layer="ingest",
        asset="raw_ohlcv",
        partition_date=partition_date,
    ):
        tickers = _load_universe()

        df = download_daily(tickers=tickers, date=partition_date, data_dir=_DATA_DIR)

        # Quality checks
        assert df.shape[0] > 0, "Nessun dato scaricato"
        assert "valid_time" in df.columns, "Campo bi-temporale mancante"
        if df["close"].dtype in (pl.Float32, pl.Float64):
            nan_close = df["close"].is_nan().sum()
            assert nan_close == 0, f"NaN nei prezzi di chiusura: {nan_close}"

        context.log.info(
            f"raw_ohlcv [{partition_date}]: {df.shape[0]} righe, "
            f"{df['ticker'].n_unique()} ticker"
        )
        _record_asset_metadata(context, "raw_ohlcv", df, partition_date)


@dg.asset(
    partitions_def=_DAILY_PARTITIONS,
    retry_policy=_RETRY,
    description="Headline di notizie finanziarie dal feed RSS Yahoo Finance.",
)
def raw_news(context: AssetExecutionContext) -> None:
    """Scarica le notizie per la data di partizione."""
    from data.ingest.news import download_news

    partition_date = context.partition_key
    with trace_span(
        "mlcouncil.ingest.raw_news",
        layer="ingest",
        asset="raw_news",
        partition_date=partition_date,
    ):
        tickers = _load_universe()
        df = download_news(tickers=tickers, date=partition_date, data_dir=_DATA_DIR)
        context.log.info(f"raw_news [{partition_date}]: {df.shape[0]} headline")
        _record_asset_metadata(context, "raw_news", df, partition_date)


@dg.asset(
    partitions_def=_DAILY_PARTITIONS,
    retry_policy=_RETRY,
    description="Dati macro (VIX, Treasury spread, S&P500) da FRED.",
)
def raw_macro(context: AssetExecutionContext) -> None:
    """Scarica e normalizza il contesto macro fino alla data di partizione."""
    from data.ingest.macro import download_macro
    from data.features.alpha158 import build_macro_context

    partition_date = context.partition_key

    with trace_span(
        "mlcouncil.ingest.raw_macro",
        layer="ingest",
        asset="raw_macro",
        partition_date=partition_date,
    ):
        download_macro(end=partition_date, data_dir=_DATA_DIR)

    macro_dir = _DATA_DIR / "macro"

    def _path(name: str) -> str | None:
        p = macro_dir / f"{name}.parquet"
        return str(p) if p.exists() else None

    macro = build_macro_context(
        vix_path=_path("vix"),
        treasuries_path=_path("treasuries"),
        sp500_path=_path("sp500"),
    )

    # Filtra fino alla data di partizione
    today = date_type.fromisoformat(partition_date)
    macro = macro.filter(pl.col("valid_time") <= today)

    context.log.info(f"raw_macro [{partition_date}]: {macro.shape[0]} righe macro")
    _record_asset_metadata(context, "raw_macro", macro, partition_date)


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
    all_ohlcv = _load_all_ohlcv()
    if all_ohlcv.is_empty():
        raise ValueError("Nessun dato OHLCV disponibile per Alpha158")

    macro_ctx = _load_macro_context_from_disk()
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

    news_path = _DATA_DIR / "raw" / "news" / f"{partition_date}.parquet"
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


# ===========================================================================
# LAYER 3 — MODEL SIGNALS
# ===========================================================================

@dg.asset(
    partitions_def=_DAILY_PARTITIONS,
    retry_policy=_RETRY,
    description="Segnali alpha dal modello LightGBM (CPCV-trained).",
)
def lgbm_signals(
    context: AssetExecutionContext,
    alpha158_features: pl.DataFrame,
) -> pd.Series:
    """Carica il checkpoint LightGBM e genera segnali cross-sezionali."""
    partition_date = context.partition_key
    with trace_span(
        "mlcouncil.signals.lgbm_signals",
        layer="signals",
        asset="lgbm_signals",
        partition_date=partition_date,
    ):
        return _run_lgbm_signals(context, alpha158_features, partition_date)


def _build_online_refit_history(
    partition_date: str,
    *,
    lookback_days: int = 60,
) -> tuple[pl.DataFrame, pd.Series, pl.DataFrame]:
    """Feature + target history for incremental refit (full OHLCV window)."""
    from data.features.alpha158 import compute_alpha158
    from data.features.target import compute_targets
    from models.online import build_targets_series, filter_features_from_date

    today = date_type.fromisoformat(partition_date)
    all_ohlcv = _load_all_ohlcv()
    if all_ohlcv.is_empty():
        return pl.DataFrame(), pd.Series(dtype=float), all_ohlcv

    ohlcv = filter_features_from_date(
        all_ohlcv,
        as_of=today,
        lookback_days=lookback_days,
    )
    macro_ctx = _load_macro_context_from_disk()
    if macro_ctx.is_empty():
        macro_ctx = None
    features = compute_alpha158(ohlcv, macro_df=macro_ctx)
    targets_pl = compute_targets(ohlcv, horizons=[1], risk_adjusted=False)
    targets = build_targets_series(targets_pl, horizon_col="rank_fwd_1d")
    return features, targets, ohlcv


def _run_lgbm_signals(
    context: AssetExecutionContext,
    alpha158_features: pl.DataFrame,
    partition_date: str,
) -> pd.Series:
    checkpoint = _CHECKPOINTS / "lgbm_latest.pkl"
    tickers = alpha158_features["ticker"].unique().to_list()

    try:
        from models.technical import TechnicalModel
    except ModuleNotFoundError as exc:
        context.log.warning(
            f"lgbm_signals [{partition_date}]: dipendenza modello mancante "
            f"({exc}) - fallback a 0.0"
        )
        fallback = pd.Series(0.0, index=tickers, name="lgbm_signal")
        lineage = build_feature_lineage(
            asset_name="alpha158_features",
            payload=alpha158_features,
            data_payload=alpha158_features.select(["ticker", "valid_time"]),
            context=context,
            partition_date=partition_date,
            model_version="lgbm-missing-dependency",
        )
        context.add_output_metadata(
            lineage_artifact_payload(lineage, signal_count=len(fallback), fallback="missing_dependency")
        )
        return attach_lineage(fallback, **lineage)

    model = TechnicalModel()
    online_meta: dict | None = None
    if checkpoint.exists():
        model.load(str(checkpoint))
        context.log.info(
            f"lgbm_signals [{partition_date}]: checkpoint caricato da {checkpoint}"
        )
        try:
            from models.online import online_learning_enabled, run_daily_incremental_update

            if online_learning_enabled():
                feat_hist, targets, ohlcv = _build_online_refit_history(partition_date)
                if not feat_hist.is_empty() and len(targets) > 0:
                    model, online_result = run_daily_incremental_update(
                        model,
                        checkpoint,
                        features_history=feat_hist,
                        targets=targets,
                        ohlcv=ohlcv,
                    )
                    online_meta = {
                        "accepted": online_result.accepted,
                        "ic_baseline": online_result.ic_baseline,
                        "ic_today": online_result.ic_today,
                        "drift_detected": online_result.drift_detected,
                        "message": online_result.message,
                    }
                    context.log.info(
                        f"lgbm_signals [{partition_date}]: online learning — "
                        f"{online_result.message}"
                    )
                    if online_result.drift_detected:
                        context.log.warning(
                            f"lgbm_signals [{partition_date}]: ADWIN drift su returns "
                            "60d — schedulare walk-forward retrain"
                        )
        except Exception as exc:
            context.log.warning(
                f"lgbm_signals [{partition_date}]: online learning fallito ({exc}), "
                "checkpoint champion invariato"
            )
    else:
        context.log.warning(
            f"lgbm_signals [{partition_date}]: checkpoint non trovato, "
            "segnali impostati a 0.0"
        )
        fallback = pd.Series(0.0, index=tickers, name="lgbm_signal")
        lineage = build_feature_lineage(
            asset_name="alpha158_features",
            payload=alpha158_features,
            data_payload=alpha158_features.select(["ticker", "valid_time"]),
            context=context,
            partition_date=partition_date,
            model_version="lgbm-no-checkpoint",
        )
        context.add_output_metadata(
            lineage_artifact_payload(lineage, signal_count=len(fallback), fallback="no_checkpoint")
        )
        return attach_lineage(fallback, **lineage)

    regime_hist = None
    try:
        from models.regime_features import load_regime_history, regime_features_enabled

        if regime_features_enabled():
            regime_hist = load_regime_history(_RESULTS_DIR / "regime_history.parquet")
    except Exception as exc:
        context.log.warning(
            f"lgbm_signals [{partition_date}]: regime features unavailable ({exc})"
        )

    signals = model.predict(alpha158_features, regime_history=regime_hist).rename(
        "lgbm_signal"
    )

    meta_stats = None
    try:
        from models.meta_label import apply_meta_label_gate, meta_label_enabled

        if meta_label_enabled():
            signals, meta_stats = apply_meta_label_gate(
                signals,
                alpha158_features,
                checkpoint=_CHECKPOINTS / "meta_label_latest.pkl",
            )
            if meta_stats is not None:
                context.log.info(
                    f"lgbm_signals [{partition_date}]: meta-label "
                    f"filtered={meta_stats.filtered_fraction:.1%} "
                    f"shadow={meta_stats.shadow}"
                )
    except Exception as exc:
        context.log.warning(
            f"lgbm_signals [{partition_date}]: meta-label gate failed ({exc})"
        )

    lineage = build_feature_lineage(
        asset_name="alpha158_features",
        payload=alpha158_features,
        data_payload=alpha158_features.select(["ticker", "valid_time"]),
        context=context,
        partition_date=partition_date,
        model_version=checkpoint_version(checkpoint, "lgbm-no-checkpoint"),
    )
    signals = attach_lineage(signals, **lineage)
    context.log.info(
        f"lgbm_signals [{partition_date}]: segnali per {len(signals)} ticker"
    )
    meta = lineage_artifact_payload(lineage, signal_count=len(signals))
    if online_meta:
        meta = {**meta, "online_learning": online_meta}
    if meta_stats is not None:
        meta = {**meta, "meta_label": meta_stats.to_dict()}
    context.add_output_metadata(meta)
    return signals


@dg.asset(
    partitions_def=_DAILY_PARTITIONS,
    retry_policy=_RETRY,
    description="Segnali alpha dal modello sentiment FinBERT (z-score cross-sezionale).",
)
def sentiment_signals(
    context: AssetExecutionContext,
    sentiment_features: pl.DataFrame,
) -> pd.Series:
    """Normalizza i punteggi di sentiment in z-score cross-sezionale."""
    partition_date = context.partition_key

    if sentiment_features.is_empty() or "sentiment_score" not in sentiment_features.columns:
        context.log.warning(
            f"sentiment_signals [{partition_date}]: nessun dato sentiment"
        )
        empty = pd.Series(dtype=float, name="sentiment_signal")
        lineage = build_feature_lineage(
            asset_name="sentiment_features",
            payload=sentiment_features.to_pandas() if not sentiment_features.is_empty() else pd.DataFrame(columns=["ticker", "valid_time", "sentiment_score"]),
            data_payload=sentiment_features.to_pandas() if not sentiment_features.is_empty() else pd.DataFrame(columns=["ticker", "valid_time"]),
            context=context,
            partition_date=partition_date,
            model_version="sentiment-derived",
        )
        context.add_output_metadata(
            lineage_artifact_payload(lineage, signal_count=0, fallback="no_sentiment_features")
        )
        return attach_lineage(empty, **lineage)

    sent = (
        sentiment_features
        .to_pandas()
        .set_index("ticker")["sentiment_score"]
    )

    # Z-score cross-sezionale
    std = sent.std()
    if std > 1e-9:
        sent = (sent - sent.mean()) / std

    sent = sent.rename("sentiment_signal")
    lineage = build_feature_lineage(
        asset_name="sentiment_features",
        payload=sentiment_features,
        data_payload=sentiment_features.select(["ticker", "valid_time"]),
        context=context,
        partition_date=partition_date,
        model_version="sentiment-derived",
    )
    sent = attach_lineage(sent, **lineage)
    context.log.info(
        f"sentiment_signals [{partition_date}]: {len(sent)} ticker"
    )
    context.add_output_metadata(lineage_artifact_payload(lineage, signal_count=len(sent)))
    return sent


@dg.asset(
    partitions_def=_DAILY_PARTITIONS,
    retry_policy=_RETRY,
    description="Regime di mercato corrente: 'bull', 'bear', o 'transition' (HMM).",
    deps=[raw_macro],
)
def current_regime(
    context: AssetExecutionContext,
) -> str:
    """Rileva il regime di mercato con il modello HMM."""
    partition_date = context.partition_key
    checkpoint = _CHECKPOINTS / "hmm_latest.pkl"

    raw_macro = _load_macro_context_from_disk()

    try:
        from models.regime import RegimeModel
    except ModuleNotFoundError as exc:
        context.log.warning(
            f"current_regime [{partition_date}]: dipendenza HMM mancante "
            f"({exc}) - fallback a 'transition'"
        )
        return "transition"

    if checkpoint.exists():
        regime_model = _safe_pickle_load(checkpoint)
        context.log.info(
            f"current_regime [{partition_date}]: HMM caricato da {checkpoint}"
        )
    else:
        #_checkpoint_intentionally_left_out — HMM must be trained by train_hmm_job
        context.log.error(
            f"current_regime [{partition_date}]: "
            f"checkpoint {checkpoint} non trovato. "
            "L'HMM non può allenarsi inline (richiede storico completo). "
            "Lancia 'train_hmm_job' per generare il checkpoint."
        )
        raise FileNotFoundError(
            f"HMM checkpoint mancante: {checkpoint}. "
            "Esegui train_hmm_job per allenare e salvare il modello HMM."
        )

    try:
        regime = regime_model.predict_regime(raw_macro)
    except Exception as exc:
        context.log.warning(
            f"HMM predict fallito: {exc} — fallback a 'transition'"
        )
        regime = "transition"

    context.log.info(f"current_regime [{partition_date}]: {regime}")
    context.add_output_metadata(
        lineage_artifact_payload(
            {
                "pipeline_run_id": build_pipeline_run_id(context, partition_date),
                "data_version": version_payload("raw_macro", raw_macro, partition_date),
                "feature_version": version_payload("raw_macro-context", raw_macro, partition_date),
                "model_version": checkpoint_version(checkpoint, "hmm-inline"),
            },
            regime=regime,
        )
    )
    return regime


# ============================================================================
# LAYER 3b — HMM TRAINING (separate job, full-history training)
# ============================================================================

_CHECKPOINTS.mkdir(parents=True, exist_ok=True)


@dg.asset(
    name="train_hmm",
    description=(
        "Allena l'HMM su tutto lo storico macro e salva il checkpoint. "
        "Job separato (non daily) — schedule: domenicale 23:00 ET."
    ),
)
def train_hmm(context: AssetExecutionContext) -> dict:
    """Allena RegimeModel su tutto lo storico macro e salva checkpoint + regime_history.

    Questo asset è UNPARTITIONED — gira una volta per generare il checkpoint
    che poi ``current_regime`` consuma ad ogni run giornaliero.
    """
    from data.ingest.macro import download_macro
    from data.features.alpha158 import build_macro_context

    # Carica tutto lo storico macro (no partition filter)
    today = date_type.today().isoformat()
    download_macro(end=today, data_dir=_DATA_DIR)

    macro_dir = _DATA_DIR / "macro"

    def _path(name: str) -> str | None:
        p = macro_dir / f"{name}.parquet"
        return str(p) if p.exists() else None

    macro = build_macro_context(
        vix_path=_path("vix"),
        treasuries_path=_path("treasuries"),
        sp500_path=_path("sp500"),
    )

    if macro.is_empty():
        raise RuntimeError(
            "train_hmm: nessun dato macro disponibile. "
            "Impossibile allenare l'HMM."
        )

    context.log.info(f"train_hmm: allenamento su {macro.shape[0]} osservazioni macro")

    from models.regime import RegimeModel
    regime_model = RegimeModel()
    regime_model.fit(macro)

    # Salva checkpoint con hash
    checkpoint_path = _CHECKPOINTS / "hmm_latest.pkl"
    regime_model.save(str(checkpoint_path))
    write_artifact_manifest(
        checkpoint_path,
        artifact_type="model_checkpoint",
        metadata={
            "model_name": "hmm",
            "n_observations": int(macro.shape[0]),
        },
    )
    context.log.info(f"train_hmm: checkpoint salvato in {checkpoint_path}")

    # Genera regime history per la dashboard
    try:
        history_df = regime_model.get_regime_history(macro)
        history_path = _RESULTS_DIR / "regime_history.parquet"
        _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        history_df.to_parquet(history_path, index=False)
        write_artifact_manifest(
            history_path,
            artifact_type="regime_history",
            metadata={"row_count": int(len(history_df))},
        )
        context.log.info(
            f"train_hmm: regime_history salvato in {history_path} "
            f"({len(history_df)} righe)"
        )
    except Exception as exc:
        context.log.warning(f"train_hmm: errore generando regime_history: {exc}")

    probs = regime_model.predict_probabilities(macro)
    current = regime_model.predict_regime(macro)
    context.log.info(
        f"train_hmm: regime corrente = {current.upper()} — "
        f"probabilità: {', '.join(f'{k}={v:.1%}' for k,v in probs.items())}"
    )

    return {
        "checkpoint": str(checkpoint_path),
        "regime": current,
        "probabilities": probs,
        "n_observations": macro.shape[0],
        "last_trained": regime_model._last_trained,
    }


train_hmm_job = dg.define_asset_job(
    name="train_hmm_job",
    selection=dg.AssetSelection.assets(train_hmm),
    description=(
        "Job HMM: allena RegimeModel su storico macro completo, "
        "salva checkpoint e regime_history. Schedule: domenicale 23:00 ET."
    ),
)


def _sunday_tags(context: "dg.ScheduleEvaluationContext") -> dict[str, str]:
    """Tags per domenica — processa la settimana appena conclusa."""
    et = pytz.timezone("America/New_York")
    scheduled = context.scheduled_execution_time
    if scheduled is None:
        partition_date = date_type.today().isoformat()
    else:
        if scheduled.tzinfo is None:
            scheduled = pytz.UTC.localize(scheduled)
        et_time = scheduled.astimezone(et)
        partition_date = et_time.date().isoformat()
    return {"dagster/partition": partition_date}


@dg.schedule(
    cron_schedule="0 23 * * 0",       # 23:00 ET ogni domenica
    execution_timezone="America/New_York",
    job=train_hmm_job,
)
def hmm_schedule(context: "dg.ScheduleEvaluationContext"):
    """Schedule HMM domenicale: processa la settimana appena conclusa."""
    et = pytz.timezone("America/New_York")
    scheduled = context.scheduled_execution_time
    if scheduled is None:
        partition_date = date_type.today().isoformat()
    else:
        if scheduled.tzinfo is None:
            scheduled = pytz.UTC.localize(scheduled)
        et_time = scheduled.astimezone(et)
        partition_date = et_time.date().isoformat()
    return dg.RunRequest(
        partition_key=partition_date,
        tags={"mlcouncil/partition": partition_date},
    )


# ============================================================================
# LAYER 4 — COUNCIL
# ============================================================================

@dg.asset(
    partitions_def=_DAILY_PARTITIONS,
    retry_policy=_RETRY,
    description="Segnale council aggregato (ensemble pesato per regime).",
)
def council_signal(
    context: AssetExecutionContext,
    lgbm_signals: pd.Series,
    sentiment_signals: pd.Series,
    current_regime: str,
) -> pd.Series:
    """Aggrega i segnali dei modelli con il CouncilAggregator."""
    partition_date = context.partition_key
    today = date_type.fromisoformat(partition_date)

    with trace_span(
        "mlcouncil.council.council_signal",
        layer="council",
        asset="council_signal",
        partition_date=partition_date,
    ):
        return _run_council_signal(
            context,
            lgbm_signals,
            sentiment_signals,
            current_regime,
            partition_date,
            today,
        )


def _run_council_signal(
    context: AssetExecutionContext,
    lgbm_signals: pd.Series,
    sentiment_signals: pd.Series,
    current_regime: str,
    partition_date: str,
    today: date_type,
) -> pd.Series:
    from council.aggregator import CouncilAggregator

    aggregator = CouncilAggregator(
        config_path=str(_ROOT / "config" / "regime_weights.yaml")
    )

    signals: dict[str, pd.Series] = {}
    if not lgbm_signals.empty:
        signals["lgbm"] = lgbm_signals
    if not sentiment_signals.empty:
        signals["sentiment"] = sentiment_signals

    if not signals:
        context.log.warning(
            f"council_signal [{partition_date}]: nessun segnale attivo"
        )
        return pd.Series(dtype=float, name="council_signal")

    from council.frontier import (
        apply_stacked_council_override,
        enrich_council_experts,
        load_regime_context,
    )

    tickers = sorted({t for s in signals.values() for t in s.index})
    signals = enrich_council_experts(
        signals, tickers=tickers, partition_date=partition_date
    )
    raw_macro = _load_macro_context_from_disk()
    regime_embedding, regime_centroids = load_regime_context(raw_macro, current_regime)

    from council.moe_gating import log_moe_shadow, moe_enabled

    if moe_enabled() and len(signals) >= 2:
        import os as _os

        _prev_mode = _os.environ.get("MLCOUNCIL_AGGREGATOR_MODE")
        try:
            _os.environ["MLCOUNCIL_AGGREGATOR_MODE"] = "linear"
            linear_signal = aggregator.aggregate(
                signals,
                regime=current_regime,
                date=today,
                regime_embedding=regime_embedding,
                regime_centroids=regime_centroids,
            )
            _os.environ["MLCOUNCIL_AGGREGATOR_MODE"] = "moe"
            moe_signal = aggregator.aggregate(
                signals,
                regime=current_regime,
                date=today,
                regime_embedding=regime_embedding,
                regime_centroids=regime_centroids,
            )
            log_entry = aggregator._weights_log.get(today, {})
            log_moe_shadow(
                partition_date,
                linear_signal=linear_signal,
                moe_signal=moe_signal,
                gate_weights=log_entry.get("moe_gate"),
                expert_order=list(signals.keys()),
                effective_weights=log_entry.get("weights"),
            )
            combined = moe_signal
        finally:
            if _prev_mode is None:
                _os.environ.pop("MLCOUNCIL_AGGREGATOR_MODE", None)
            else:
                _os.environ["MLCOUNCIL_AGGREGATOR_MODE"] = _prev_mode
    else:
        combined = aggregator.aggregate(
            signals,
            regime=current_regime,
            date=today,
            regime_embedding=regime_embedding,
            regime_centroids=regime_centroids,
        )

    combined = combined.rename("council_signal")
    combined = apply_stacked_council_override(combined, signals, partition_date)

    from models.options_sentiment import options_sentiment_enabled, run_shadow_batch

    if options_sentiment_enabled():
        try:
            opt_report = run_shadow_batch(tickers, partition_date=partition_date)
            context.log.info(
                f"council_signal [{partition_date}]: options sentiment shadow "
                f"status={opt_report.get('status')}"
            )
        except Exception as exc:
            context.log.warning(
                f"council_signal [{partition_date}]: options sentiment shadow failed ({exc})"
            )

    from council.cqr import (
        DEFAULT_STACKING_CHECKPOINT,
        StackingMetaLearner,
        log_stacking_shadow,
        stacking_shadow_enabled,
    )

    if stacking_shadow_enabled() and len(signals) >= 2:
        base_df = pd.DataFrame({m: s for m, s in signals.items()}).fillna(0.0)
        if DEFAULT_STACKING_CHECKPOINT.exists():
            try:
                meta = StackingMetaLearner.load(DEFAULT_STACKING_CHECKPOINT)
                stacked = meta.predict(base_df)
            except Exception as exc:
                context.log.warning(
                    f"council_signal [{partition_date}]: stacking shadow failed ({exc})"
                )
                stacked = base_df.mean(axis=1)
        else:
            stacked = base_df.mean(axis=1)
        log_stacking_shadow(partition_date, combined, stacked.rename("stacked_signal"))

    hmm_version = checkpoint_version(_CHECKPOINTS / "hmm_latest.pkl", "hmm-inline")
    lineage = merge_lineage(
        lgbm_signals,
        sentiment_signals,
        context=context,
        partition_date=partition_date,
        model_version=hmm_version,
    )
    combined = attach_lineage(combined, **lineage)
    signal_payload = pd.DataFrame(
        {
            "ticker": list(combined.index),
            "council_signal": combined.values,
        }
    )
    for key, values in dataframe_lineage_columns(lineage, len(signal_payload)).items():
        signal_payload[key] = values
    _record_asset_metadata(context, "council_signal", signal_payload, partition_date, lineage)
    context.log.info(
        f"council_signal [{partition_date}]: {len(combined)} ticker | "
        f"regime={current_regime}"
    )
    context.add_output_metadata(lineage_artifact_payload(lineage, signal_count=len(combined), regime=current_regime))
    return combined


@dg.asset(
    partitions_def=_DAILY_PARTITIONS,
    retry_policy=_RETRY,
    description="Salva lo stato dell'aggregator e la attribution corrente in data/results/.",
)
def save_council_results(
    context: AssetExecutionContext,
    lgbm_signals: pd.Series,
    sentiment_signals: pd.Series,
    current_regime: str,
    council_signal: pd.Series,
) -> None:
    """Serializza CouncilAggregator e attribution parquet in data/results/.

    Scrive:
    - data/results/aggregator.pkl       → stato completo CouncilAggregator
    - data/results/attribution.parquet → DataFrame con pesi, IC, Sharpe per ogni modello

    Lo step è idempotente: se i file esistono già li sovrascrive con i dati più recenti.
    """
    from council.aggregator import CouncilAggregator

    partition_date = context.partition_key
    today = date_type.fromisoformat(partition_date)

    # Crea e popola l'aggregator con i segnali disponibili
    aggregator = CouncilAggregator(
        config_path=str(_ROOT / "config" / "regime_weights.yaml")
    )

    signals: dict[str, pd.Series] = {}
    if not lgbm_signals.empty:
        signals["lgbm"] = lgbm_signals
    if not sentiment_signals.empty:
        signals["sentiment"] = sentiment_signals

    if signals:
        # Esegue aggregate per popolare _weights_log sull'ultimo giorno
        aggregator.aggregate(signals, regime=current_regime, date=today)

    # Salva stato aggregator
    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    aggregator_path = _RESULTS_DIR / "aggregator.pkl"
    aggregator.save(str(aggregator_path))
    write_artifact_manifest(
        aggregator_path,
        artifact_type="aggregator_state",
        metadata={"partition_date": partition_date},
    )

    # Salva attribution parquet
    if not aggregator._weights_log:
        context.log.warning(
            f"save_council_results [{partition_date}]: weights_log vuoto, "
            "skipping attribution.parquet"
        )
    else:
        attr_rows = []
        for log_date, log_entry in aggregator._weights_log.items():
            weights_used = log_entry.get("weights", {})
            contributions = log_entry.get("contributions", {})
            ic_by_model = aggregator._ic_by_date
            for model_name in weights_used:
                ic_entries = sorted(ic_by_model.get(model_name, {}).items())
                recent_30 = [v for _, v in ic_entries[-30:]]
                recent_60 = [v for _, v in ic_entries[-60:]]
                ic_30d = float(np.mean(recent_30)) if len(recent_30) >= 1 else float("nan")
                sharpe_60d = (
                    float(np.mean(recent_60) / (np.std(recent_60) + 1e-9) * np.sqrt(252))
                    if len(recent_60) >= 2
                    else float("nan")
                )
                row = {
                    "date": pd.Timestamp(log_date),
                    "model_name": model_name,
                    "weight": weights_used.get(model_name, float("nan")),
                    "ic_rolling_30d": ic_30d,
                    "sharpe_rolling_60d": sharpe_60d,
                    "pnl_contribution": contributions.get(model_name, float("nan")),
                }
                if "weight_sum" in log_entry:
                    row["effective_weight_sum"] = log_entry.get("weight_sum")
                attr_rows.append(row)

        if attr_rows:
            attr_columns = [
                "date", "model_name", "weight",
                "ic_rolling_30d", "sharpe_rolling_60d", "pnl_contribution",
            ]
            if attr_rows and "effective_weight_sum" in attr_rows[0]:
                attr_columns.append("effective_weight_sum")
            attr_df = pd.DataFrame(attr_rows, columns=attr_columns)
            attr_df.to_parquet(_RESULTS_DIR / "attribution.parquet", index=False)
            write_artifact_manifest(
                _RESULTS_DIR / "attribution.parquet",
                artifact_type="model_attribution",
                metadata={
                    "partition_date": partition_date,
                    "row_count": int(len(attr_df)),
                },
            )
            context.log.info(
                f"save_council_results [{partition_date}]: "
                f"attribution.parquet scritto ({len(attr_df)} righe)"
            )

    context.log.info(f"save_council_results [{partition_date}]: completato")


@dg.asset(
    partitions_def=_DAILY_PARTITIONS,
    retry_policy=_RETRY,
    description="Salva regime corrente e storia regimi in data/results/.",
    deps=[raw_macro],
)
def save_regime_results(
    context: AssetExecutionContext,
    current_regime: str,
) -> None:
    """Scrive current_regime.json e regime_history.parquet in data/results/.

    Scrive:
    - data/results/current_regime.json    → regime attuale con probabilità
    - data/results/regime_history.parquet → storia completa regimi con probabilità
    """
    partition_date = context.partition_key

    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    raw_macro = _load_macro_context_from_disk()

    # ------------------------------------------------------------------
    # current_regime.json
    # ------------------------------------------------------------------
    probs: dict[str, float] = {"bull": 0.0, "bear": 0.0, "transition": 0.0}
    if not raw_macro.is_empty():
        try:
            from models.regime import RegimeModel
            checkpoint = _CHECKPOINTS / "hmm_latest.pkl"
            if checkpoint.exists():
                regime_model: RegimeModel = _safe_pickle_load(checkpoint)
                prob_dict = regime_model.predict_probabilities(raw_macro)
                for key in probs:
                    if key in prob_dict:
                        probs[key] = float(prob_dict[key])
        except Exception as exc:
            context.log.warning(
                f"save_regime_results [{partition_date}]: "
                f"probabilities unavailable ({exc})"
            )

    regime_payload = {
        "regime": current_regime,
        "bull": probs["bull"],
        "bear": probs["bear"],
        "transition": probs["transition"],
    }
    import json
    with open(_RESULTS_DIR / "current_regime.json", "w") as f:
        json.dump(regime_payload, f, indent=2, default=str)
    write_artifact_manifest(
        _RESULTS_DIR / "current_regime.json",
        artifact_type="regime_snapshot",
        metadata={"partition_date": partition_date},
    )
    context.log.info(
        f"save_regime_results [{partition_date}]: "
        f"current_regime.json written (regime={current_regime})"
    )

    # ------------------------------------------------------------------
    # regime_history.parquet
    # ------------------------------------------------------------------
    if not raw_macro.is_empty():
        try:
            from models.regime import RegimeModel
            checkpoint = _CHECKPOINTS / "hmm_latest.pkl"
            if checkpoint.exists():
                regime_model: RegimeModel = _safe_pickle_load(checkpoint)
                hist_df = regime_model.get_regime_history(raw_macro)
                if "valid_time" in hist_df.columns:
                    hist_df = hist_df.rename(columns={"valid_time": "date"})
                if "date" in hist_df.columns:
                    hist_df["date"] = pd.to_datetime(hist_df["date"])
                hist_df.to_parquet(_RESULTS_DIR / "regime_history.parquet", index=False)
                write_artifact_manifest(
                    _RESULTS_DIR / "regime_history.parquet",
                    artifact_type="regime_history",
                    metadata={
                        "partition_date": partition_date,
                        "row_count": int(len(hist_df)),
                    },
                )
                context.log.info(
                    f"save_regime_results [{partition_date}]: "
                    f"regime_history.parquet written ({len(hist_df)} righe)"
                )
        except Exception as exc:
            context.log.warning(
                f"save_regime_results [{partition_date}]: "
                f"regime_history unavailable ({exc})"
            )


@dg.asset(
    partitions_def=_DAILY_PARTITIONS,
    retry_policy=_RETRY,
    description="Pesi di portafoglio ottimizzati (mean-variance cvxpy).",
)
def portfolio_weights(
    context: AssetExecutionContext,
    council_signal: pd.Series,
    alpha158_features: pl.DataFrame,
) -> pd.Series:
    """Ottimizza il portafoglio con conformal sizing e covariance storica.

    Se il conformal sizer non è disponibile usa moltiplicatori unitari.
    La matrice di covarianza è calcolata sulle ultime 90 sessioni disponibili.
    """
    partition_date = context.partition_key

    with trace_span(
        "mlcouncil.council.portfolio_weights",
        layer="council",
        asset="portfolio_weights",
        partition_date=partition_date,
    ):
        return _run_portfolio_weights(
            context, council_signal, alpha158_features, partition_date
        )


def _run_portfolio_weights(
    context: AssetExecutionContext,
    council_signal: pd.Series,
    alpha158_features: pl.DataFrame,
    partition_date: str,
) -> pd.Series:
    from council.cqr import get_position_sizer, position_sizer_checkpoint_name, position_sizing_mode
    from council.portfolio_diff import get_portfolio_constructor

    if council_signal.empty:
        context.log.warning(
            f"portfolio_weights [{partition_date}]: nessun segnale ricevuto"
        )
        empty = pd.Series(dtype=float, name="target_weight")
        lineage = extract_lineage(council_signal)
        empty = attach_lineage(empty, **lineage)
        empty_payload = pd.DataFrame(
            columns=["ticker", "target_weight", *LINEAGE_COLUMNS]
        )
        _record_asset_metadata(
            context,
            "portfolio_weights",
            empty_payload,
            partition_date,
            lineage,
        )
        context.add_output_metadata(lineage_artifact_payload(lineage, position_count=0))
        return empty

    tickers = council_signal.index.tolist()

    # Matrice di covarianza su ultime 90 sessioni
    cov_df = _compute_covariance(tickers)
    cov_tickers = [t for t in tickers if t in cov_df.columns]
    if not cov_tickers:
        cov_tickers = tickers
        n = len(tickers)
        cov_df = pd.DataFrame(
            np.eye(n) * 0.0001, index=tickers, columns=tickers
        )

    signal_aligned = council_signal.reindex(cov_tickers).fillna(0.0)
    cov = cov_df.reindex(index=cov_tickers, columns=cov_tickers).fillna(0.0)

    # Market returns for beta neutrality
    market_returns = _load_market_returns()

    # Position sizing (conformal default, CQR when MLCOUNCIL_POSITION_SIZING=cqr, kelly when MLCOUNCIL_POSITION_SIZING=kelly)
    sizing_mode = position_sizing_mode()
    if sizing_mode == "kelly":
        from council.fractional_kelly import FractionalKellySizer

        sizer = FractionalKellySizer()
        context.log.info(
            f"portfolio_weights [{partition_date}]: "
            "FractionalKellySizer istanziato direttamente"
        )
        # Kelly sizer non usa features, passa None
        multipliers = sizer.compute_position_multipliers(signal_aligned)
    else:
        sizer_checkpoint = _CHECKPOINTS / position_sizer_checkpoint_name()
        if sizer_checkpoint.exists():
            sizer = _safe_pickle_load(sizer_checkpoint)
            context.log.info(
                f"portfolio_weights [{partition_date}]: "
                f"position sizer caricato da {sizer_checkpoint}"
            )
            # Use real Alpha158 features for interval width
            n = len(cov_tickers)
            feat_df = alpha158_features.filter(pl.col("ticker").is_in(cov_tickers))
            feat_cols = [c for c in feat_df.columns if c not in _EXCLUDE_COLS]
            if (
                len(feat_df) == n
                and len(feat_cols) >= (sizer._n_features or 0)
                and sizer._n_features is not None
            ):
                X_real = feat_df.select(feat_cols[:sizer._n_features]).to_numpy()
                multipliers = sizer.compute_position_multipliers(signal_aligned, X_real)
            else:
                # Fallback: fewer tickers in features than sizer expects
                X_dummy = np.zeros((n, sizer._n_features or 1))
                context.log.warning(
                    f"portfolio_weights [{partition_date}]: "
                    f"feature/ticker mismatch ({len(feat_df)} vs {n} tickers, "
                    f"{len(feat_cols)} vs {sizer._n_features} features) — "
                    f"using dummy features for conformal sizing"
                )
                multipliers = sizer.compute_position_multipliers(signal_aligned, X_dummy)
        else:
            context.log.warning(
                f"portfolio_weights [{partition_date}]: "
                "position sizer non trovato — multipliers=1.0"
            )
            multipliers = pd.Series(1.0, index=cov_tickers, name="multiplier")

    # Pesi correnti: portafoglio live se disponibile, altrimenti bootstrap da zero.
    current_w, portfolio_value = _load_live_portfolio_snapshot(cov_tickers)

    constructor = get_portfolio_constructor()
    optimize_with_crypto = getattr(constructor, "optimize_with_crypto", None)
    has_crypto = any(_pipeline_crypto_check(ticker) for ticker in cov_tickers)
    if callable(optimize_with_crypto) and has_crypto:
        weights = optimize_with_crypto(
            alpha_signals=signal_aligned,
            position_multipliers=multipliers,
            current_weights=current_w,
            returns_covariance=cov,
            market_returns=market_returns,
            portfolio_value=portfolio_value,
        )
    else:
        weights = constructor.optimize(
            alpha_signals=signal_aligned,
            position_multipliers=multipliers,
            current_weights=current_w,
            returns_covariance=cov,
            market_returns=market_returns,
            portfolio_value=portfolio_value,
        )

    # ── Pre-trade risk check ──────────────────────────────────────────
    from council.risk_engine import RiskEngine
    risk = RiskEngine()
    limits_ok, breaches = risk.check_limits_from_weights(weights, cov)
    if not limits_ok:
        context.log.warning(
            f"portfolio_weights [{partition_date}]: "
            f"risk limits breached: {breaches} — scaling down positions"
        )
        # Scale all weights proportionally until limits are met
        for breach in breaches:
            if "sector" in str(breach).lower():
                # Reduce overweight sectors
                from data.features.sector_exposure import compute_sector_exposures, get_ticker_sector
                sector_exposures = compute_sector_exposures(weights)
                for sector, exposure in sector_exposures.items():
                    if exposure > 0.35:
                        scale = 0.35 / exposure
                        for t in weights.index:
                            if get_ticker_sector(t) == sector:
                                weights[t] *= scale
            elif "var" in str(breach).lower():
                weights *= 0.5  # Halve all positions if VaR breach
    # Re-normalize weights
    if abs(weights.sum()) > 1e-9:
        weights = weights / weights.abs().sum() * min(weights.abs().sum(), 1.0)

    weights = attach_lineage(weights.rename("target_weight"), **extract_lineage(council_signal))
    weights_lineage = extract_lineage(weights)
    weights_payload = pd.DataFrame(
        {
            "ticker": list(weights.index),
            "target_weight": weights.values,
        }
    )
    for key, values in dataframe_lineage_columns(weights_lineage, len(weights_payload)).items():
        weights_payload[key] = values
    _record_asset_metadata(
        context,
        "portfolio_weights",
        weights_payload,
        partition_date,
        weights_lineage,
    )
    context.log.info(
        f"portfolio_weights [{partition_date}]: {len(weights)} posizioni | "
        f"top3={weights.nlargest(3).round(3).to_dict()}"
    )
    context.add_output_metadata(
        lineage_artifact_payload(weights_lineage, position_count=len(weights))
    )
    return weights


def _pipeline_crypto_check(ticker: str) -> bool:
    from execution.alpaca_adapter import AlpacaLiveNode

    return AlpacaLiveNode._is_crypto(ticker)


@dg.asset(
    partitions_def=_DAILY_PARTITIONS,
    retry_policy=_RETRY,
    description="Lista ordini giornalieri (buy/sell) salvata in data/orders/{date}.parquet.",
)
def daily_orders(
    context: AssetExecutionContext,
    portfolio_weights: pd.Series,
) -> pd.DataFrame:
    """Genera e persiste la lista ordini dal delta di pesi target."""
    partition_date = context.partition_key

    with trace_span(
        "mlcouncil.council.daily_orders",
        layer="council",
        asset="daily_orders",
        partition_date=partition_date,
    ):
        return _run_daily_orders(context, portfolio_weights, partition_date)


def _run_daily_orders(
    context: AssetExecutionContext,
    portfolio_weights: pd.Series,
    partition_date: str,
) -> pd.DataFrame:
    from council.portfolio import PortfolioConstructor

    _ORDERS_DIR.mkdir(parents=True, exist_ok=True)

    lineage = extract_lineage(portfolio_weights)
    if not lineage:
        lineage = {
            "pipeline_run_id": build_pipeline_run_id(context, partition_date),
            "data_version": "unknown",
            "feature_version": "unknown",
            "model_version": "unknown",
        }

    from council.transaction_costs import get_active_calibration_version

    cost_calib_version = get_active_calibration_version()

    if portfolio_weights.empty:
        context.log.warning(
            f"daily_orders [{partition_date}]: nessun peso → nessun ordine"
        )
        empty_cols = [
            "ticker",
            "direction",
            "quantity",
            "target_weight",
            "cost_calibration_version",
            *dataframe_lineage_columns(lineage, 0).keys(),
        ]
        empty_orders = pd.DataFrame(columns=empty_cols)
        empty_path = _ORDERS_DIR / f"{partition_date}.parquet"
        empty_orders.to_parquet(empty_path, index=False)
        if empty_path.exists():
            write_artifact_manifest(
                empty_path,
                artifact_type="daily_orders",
                lineage=lineage,
                metadata={"partition_date": partition_date, "row_count": 0},
            )
        _record_asset_metadata(context, "daily_orders", empty_orders, partition_date, lineage)
        return empty_orders

    current_w, portfolio_value = _load_live_portfolio_snapshot()

    constructor = PortfolioConstructor()
    orders = constructor.compute_orders(
        target_weights=portfolio_weights,
        current_weights=current_w,
        portfolio_value=portfolio_value,
    )
    if orders.empty:
        orders = pd.DataFrame(
            columns=["ticker", "direction", "quantity", "target_weight", "cost_calibration_version"]
        )

    if len(orders) > 0:
        orders["cost_calibration_version"] = cost_calib_version

    for key, values in dataframe_lineage_columns(lineage, len(orders)).items():
        orders[key] = values

    out_path = _ORDERS_DIR / f"{partition_date}.parquet"
    _record_asset_metadata(context, "daily_orders", orders, partition_date, lineage)
    orders.to_parquet(out_path, index=False)
    if out_path.exists():
        write_artifact_manifest(
            out_path,
            artifact_type="daily_orders",
            lineage=lineage,
            metadata={"partition_date": partition_date, "row_count": int(len(orders))},
        )
    if not orders.empty:
        context.log.info(
            f"daily_orders [{partition_date}]: "
            f"{len(orders)} ordini → {out_path}"
        )
    else:
        context.log.info(
            f"daily_orders [{partition_date}]: nessun ordine (portafoglio ottimale)"
        )

    return orders


# ===========================================================================
# LAYER 4b — COST CALIBRATION (nightly job, unpartitioned)
# ===========================================================================

def _lineage_from_daily_orders(daily_orders: pd.DataFrame) -> tuple[str, str]:
    """Extract pipeline_run_id and cost_calibration_version from orders lineage."""
    if daily_orders is None or daily_orders.empty:
        return "", ""
    row = daily_orders.iloc[0]
    return (
        str(row.get("pipeline_run_id", "") or ""),
        str(row.get("cost_calibration_version", "") or ""),
    )


@dg.asset(
    ins={"daily_orders": dg.AssetIn(partition_mapping=dg.LastPartitionMapping())},
    retry_policy=_RETRY,
    description=(
        "Nightly self-calibrating transaction cost artifact (ADR-0003 Stage B). "
        "Reads data/operations/fills/*.parquet and writes "
        "data/operations/cost_calibration.json + .manifest sidecar. "
        "Joins pipeline_run_id from the latest materialized daily_orders partition."
    ),
)
def cost_calibration_artifact(
    context: AssetExecutionContext,
    daily_orders: pd.DataFrame,
) -> dict:
    """Build kappa_slippage_bps per ticker/tier from realised fills.

    Unpartitioned: the calibrator consumes a rolling window of the entire
    fill log, partitioned upstream by month. Returns a summary dict for
    Dagster metadata; the durable artifact is the on-disk JSON + manifest.
    """
    from council.cost_calibration import (
        DEFAULT_CALIBRATION_PATH,
        DEFAULT_FILLS_DIR,
        run_calibration_job,
    )
    from runtime_env import get_config_hash

    orders_run_id, _orders_calib_ver = _lineage_from_daily_orders(daily_orders)
    pipeline_run_id = orders_run_id or getattr(context, "run_id", "") or ""
    config_hash = get_config_hash()

    if orders_run_id:
        context.log.info(
            f"cost_calibration_artifact: lineage pipeline_run_id={orders_run_id} "
            f"from daily_orders"
        )

    artifact = run_calibration_job(
        fills_dir=DEFAULT_FILLS_DIR,
        out_path=DEFAULT_CALIBRATION_PATH,
        pipeline_run_id=pipeline_run_id,
        config_hash=config_hash,
    )

    if artifact is None:
        context.log.warning(
            "cost_calibration_artifact: no fills available — skipping write. "
            "TransactionCostModel will continue using static lookup."
        )
        return {
            "status": "skipped_no_fills",
            "fills_dir": str(DEFAULT_FILLS_DIR),
        }

    context.log.info(
        f"cost_calibration_artifact: {artifact.fill_sample_count} fills → "
        f"{len(artifact.kappa_by_ticker)} tickers, {len(artifact.kappa_by_tier)} tiers "
        f"(version={artifact.version[:12]}…)"
    )
    return {
        "status": "ok",
        "fill_sample_count": artifact.fill_sample_count,
        "kappa_by_ticker": artifact.kappa_by_ticker,
        "kappa_by_tier": artifact.kappa_by_tier,
        "version": artifact.version,
        "pipeline_run_id": pipeline_run_id,
    }


@dg.asset(
    ins={
        "calibration_summary": dg.AssetIn("cost_calibration_artifact"),
        "daily_orders": dg.AssetIn(partition_mapping=dg.LastPartitionMapping()),
    },
    retry_policy=_RETRY,
    description=(
        "Post-calibration promotion gate: A/B static vs calibrated costs on cached "
        "strategy weights; auto-writes config/runtime_override.env on failure."
    ),
)
def cost_calibration_gate(
    context: AssetExecutionContext,
    calibration_summary: dict,
    daily_orders: pd.DataFrame,
) -> dict:
    from council.cost_calibration_gate import run_cost_calibration_promotion_gate

    report = run_cost_calibration_promotion_gate(
        calibration_summary=calibration_summary,
        daily_orders=daily_orders,
    )
    context.log.info(
        f"cost_calibration_gate: status={report.get('status')} "
        f"passed={report.get('promotion_passed')} reverted={report.get('reverted')}"
    )
    if report.get("reasons"):
        for reason in report["reasons"]:
            context.log.warning(f"cost_calibration_gate: {reason}")
    return report


@dg.asset(
    retry_policy=_RETRY,
    description="Weekly TDA topology stress signal (T4.5 shadow).",
)
def tda_warning_signal(context: AssetExecutionContext) -> dict:
    """Compute rolling beta1 proxy on multivariate returns; log alert metadata."""
    from council.tda_warning import PersistentHomologyAnalyser, tda_warning_enabled

    if not tda_warning_enabled():
        return {"status": "disabled"}

    tickers = _load_universe()[:12]
    ohlcv_dir = _DATA_DIR / "ohlcv"
    frames: list[pl.DataFrame] = []
    for ticker in tickers:
        ticker_dir = ohlcv_dir / ticker
        if ticker_dir.exists():
            for pq in sorted(ticker_dir.glob("*.parquet")):
                try:
                    frames.append(_normalize_df(pl.read_parquet(pq)))
                except Exception:
                    pass
    if not frames:
        return {"status": "skipped_no_returns"}
    ohlcv = (
        pl.concat(frames)
        .sort(["ticker", "valid_time", "transaction_time"])
        .unique(["ticker", "valid_time"], keep="last")
        .sort(["ticker", "valid_time"])
    )
    returns_wide = (
        ohlcv.select(["ticker", "valid_time", "adj_close"])
        .with_columns(
            (pl.col("adj_close") / pl.col("adj_close").shift(1) - 1)
            .over("ticker")
            .alias("ret_1d")
        )
        .filter(pl.col("ret_1d").is_not_null())
        .pivot(values="ret_1d", index="valid_time", on="ticker")
        .to_pandas()
        .set_index("valid_time")
        .tail(90)
    )
    if returns_wide.empty:
        return {"status": "skipped_no_returns"}
    analyser = PersistentHomologyAnalyser()
    result = analyser.analyse(returns_wide)
    out_path = _RESULTS_DIR / "tda_warning_latest.json"
    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    import json

    payload = result.to_dict()
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    context.log.info(f"tda_warning_signal: {payload}")
    if result.is_alert:
        context.log.warning(
            f"tda_warning_signal: beta1_proxy={result.beta1_proxy:.3f} "
            f">= {result.threshold}"
        )
    return payload


tda_warning_job = dg.define_asset_job(
    name="tda_warning_job",
    selection=dg.AssetSelection.assets(tda_warning_signal),
    description="Weekly TDA early-warning check.",
)


@dg.schedule(
    cron_schedule="0 6 * * 1",
    execution_timezone="UTC",
    job=tda_warning_job,
)
def tda_warning_schedule(context: "dg.ScheduleEvaluationContext"):
    return dg.RunRequest(tags={"mlcouncil/job": "tda_warning"})


@dg.asset(
    retry_policy=_RETRY,
    description=(
        "Weekly alpha model promotion gate (T1.1). Evaluates shadow challengers vs "
        "champion walk-forward metrics. Production promotion requires "
        "scripts/promote_model.py after 3 consecutive passes."
    ),
)
def model_promotion_gate(context: AssetExecutionContext) -> dict:
    """Run walk-forward gate for production alpha models (shadow only)."""
    from council.walkforward_promotion_gate import SUPPORTED_MODELS, run_model_promotion_gate

    auto_promote = os.getenv("MLCOUNCIL_AUTO_PROMOTE_MODELS", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    reports: dict[str, dict] = {}
    for model in sorted(SUPPORTED_MODELS):
        report = run_model_promotion_gate(model, dry_run=False)
        reports[model] = report
        context.log.info(
            f"model_promotion_gate [{model}]: status={report.get('status')} "
            f"passed={report.get('promotion_passed')} "
            f"eligible={report.get('auto_promote_eligible')}"
        )
        if auto_promote and report.get("auto_promote_eligible"):
            try:
                from council.walkforward_promotion_gate import promote_model_to_production

                promote_model_to_production(model, force=False)
                context.log.info(f"model_promotion_gate: auto-promoted {model}")
            except Exception as exc:
                context.log.warning(f"model_promotion_gate: auto-promote {model} failed: {exc}")

    return {"models": reports, "auto_promote": auto_promote}


cost_calibration_job = dg.define_asset_job(
    name="cost_calibration_job",
    selection=dg.AssetSelection.assets(
        cost_calibration_artifact,
        cost_calibration_gate,
    ),
    description=(
        "Nightly cost-calibration job: rebuilds kappa, runs promotion gate, "
        "reverts to static lookup on failure."
    ),
)


@dg.schedule(
    cron_schedule="0 23 * * *",  # 23:00 ET every day
    execution_timezone="America/New_York",
    job=cost_calibration_job,
)
def cost_calibration_schedule(context: "dg.ScheduleEvaluationContext"):
    """Nightly recalibration at 23:00 ET after market close + paper trade settlement."""
    return dg.RunRequest(tags={"mlcouncil/job": "cost_calibration"})


walkforward_promotion_job = dg.define_asset_job(
    name="walkforward_promotion_job",
    selection=dg.AssetSelection.assets(model_promotion_gate),
    description="Weekly walk-forward champion/challenger gate (alpha models).",
)


@dg.schedule(
    cron_schedule="0 2 * * 1",
    execution_timezone="UTC",
    job=walkforward_promotion_job,
)
def walkforward_promotion_schedule(context: "dg.ScheduleEvaluationContext"):
    """Monday 02:00 UTC — aligns with .github/workflows/walk-forward-ci.yml."""
    return dg.RunRequest(tags={"mlcouncil/job": "walkforward_promotion"})


# ===========================================================================
# HELPERS (non-asset)
# ===========================================================================

def _compute_covariance(tickers: list[str]) -> pd.DataFrame:
    """Carica OHLCV da disco e calcola la matrice di covarianza (ultime 90 sessioni)."""
    ohlcv_dir = _DATA_DIR / "ohlcv"
    frames: list[pl.DataFrame] = []

    for ticker in tickers:
        ticker_dir = ohlcv_dir / ticker
        if ticker_dir.exists():
            for pq in sorted(ticker_dir.glob("*.parquet")):
                try:
                    frames.append(_normalize_df(pl.read_parquet(pq)))
                except Exception:
                    pass

    if not frames:
        n = len(tickers)
        return pd.DataFrame(np.eye(n) * 0.0001, index=tickers, columns=tickers)

    ohlcv = (
        pl.concat(frames)
        .sort(["ticker", "valid_time", "transaction_time"])
        .unique(["ticker", "valid_time"], keep="last")
        .sort(["ticker", "valid_time"])
    )
    # drop_nulls before pivot would discard every row where *any* ticker has a
    # missing return (e.g. halts, sparse mid-caps).  Instead, compute returns
    # per ticker (nulls only at each ticker's first row) then pivot and use
    # pairwise covariance so tickers with partial overlap still contribute.
    returns_wide = (
        ohlcv
        .select(["ticker", "valid_time", "adj_close"])
        .with_columns(
            (pl.col("adj_close") / pl.col("adj_close").shift(1) - 1)
            .over("ticker")
            .alias("ret_1d")
        )
        .filter(pl.col("ret_1d").is_not_null())
        .pivot(values="ret_1d", index="valid_time", on="ticker")
        .to_pandas()
        .set_index("valid_time")
        .tail(90)
    )
    from council.covariance_dynamic import compute_covariance_from_returns

    return compute_covariance_from_returns(returns_wide)


def _load_market_returns() -> pd.Series | None:
    """Carica i ritorni di mercato (SPY o S&P 500) per beta neutrality."""
    spy_path = _DATA_DIR / "ohlcv" / "SPY"
    if not spy_path.exists():
        # Try alternative location
        spy_path = _DATA_DIR / "raw" / "ohlcv" / "SPY"
    if not spy_path.exists():
        return None
    try:
        all_files = sorted(spy_path.glob("*.parquet"))
        if not all_files:
            return None
        df = pd.concat([pd.read_parquet(f) for f in all_files], ignore_index=True)
        if "adj_close" in df.columns and "ticker" in df.columns:
            spy = df[df["ticker"] == "SPY"].sort_values("valid_time")
            spy_returns = spy["adj_close"].pct_change().dropna()
            spy_returns.index = spy["valid_time"].iloc[1:].values
            return spy_returns
    except Exception:
        pass
    return None


# ===========================================================================
# JOB, SCHEDULE, SENSOR
# ===========================================================================

daily_job = dg.define_asset_job(
    name="daily_pipeline",
    selection=dg.AssetSelection.all() - dg.AssetSelection.assets(train_hmm),
    partitions_def=_DAILY_PARTITIONS,
    description=(
        "Pipeline giornaliera MLCouncil: ingest \u2192 features \u2192 signals "
        "\u2192 council \u2192 orders"
    ),
)


def _prev_trading_day(
    scheduled: date_type | None,
) -> str:
    """Calcola la partition (giorno precedente del mercato) per lo schedule."""
    et = pytz.timezone("America/New_York")
    if scheduled is None:
        partition_date = date_type.today() - timedelta(days=1)
    else:
        if scheduled.tzinfo is None:
            scheduled = pytz.UTC.localize(scheduled)
        et_time = scheduled.astimezone(et)
        partition_date = et_time.date() - timedelta(days=1)
        if partition_date.strftime("%a") == "Sat":
            partition_date -= timedelta(days=1)
        elif partition_date.strftime("%a") == "Sun":
            partition_date -= timedelta(days=2)
    return partition_date.strftime("%Y-%m-%d")


@dg.schedule(
    cron_schedule="30 21 * * 1-5",   # 21:30 ET, lun-ven
    execution_timezone="America/New_York",
    job=daily_job,
)
def daily_schedule(context: "dg.ScheduleEvaluationContext"):
    """Schedule giornaliera: processa i dati del giorno di mercato precedente."""
    partition_key = _prev_trading_day(context.scheduled_execution_time)
    return dg.RunRequest(
        partition_key=partition_key,
        tags={"mlcouncil/partition": partition_key},
    )


@dg.run_failure_sensor(
    monitored_jobs=[daily_job],
    minimum_interval_seconds=60,
    description="Logga i fallimenti del daily_pipeline e segnala il run_id.",
)
def failure_sensor(context: RunFailureSensorContext) -> dg.SkipReason | None:
    """Monitora i fallimenti del daily_pipeline.

    In produzione estendere con notifica email/Slack tramite
    dagster.make_email_on_run_failure_sensor() o webhook custom.
    """
    failed_run = context.dagster_run
    error = context.failure_event.message if context.failure_event else "N/A"

    context.log.error(
        f"[failure_sensor] Run {failed_run.run_id!r} FALLITO.\n"
        f"  Job       : {failed_run.job_name}\n"
        f"  Partizione: {failed_run.tags.get('mlcouncil/partition', 'N/A')}\n"
        f"  Errore    : {error}\n"
        f"  Re-run    : dagster job execute -j daily_pipeline "
        f"--partition {failed_run.tags.get('mlcouncil/partition', '')}"
    )
    # Restituisce None → il sensore ha processato l'evento (non skippa)
    return None


# ===========================================================================
# DEFINITIONS
# ===========================================================================

_ALL_ASSETS = [
    raw_ohlcv,
    raw_news,
    raw_macro,
    alpha158_features,
    sentiment_features,
    lgbm_signals,
    sentiment_signals,
    current_regime,
    council_signal,
    save_council_results,
    save_regime_results,
    portfolio_weights,
    daily_orders,
    train_hmm,
    cost_calibration_artifact,
    cost_calibration_gate,
    model_promotion_gate,
    tda_warning_signal,
    # train_hmm + cost_calibration_* + model_promotion_gate sono unpartitioned:
    # schedule dedicate (train_hmm_job, cost_calibration_job, walkforward_promotion_job).
]


@dg.asset_check(
    asset=raw_ohlcv,
    name="raw_ohlcv_contract",
    blocking=True,
    partitions_def=_DAILY_PARTITIONS,
)
def raw_ohlcv_contract(raw_ohlcv: pl.DataFrame) -> dg.AssetCheckResult:
    partition_date = None
    if not raw_ohlcv.is_empty() and "valid_time" in raw_ohlcv.columns:
        partition_date = str(raw_ohlcv["valid_time"].max())
    return _contract_check_result("raw_ohlcv", raw_ohlcv, partition_date)


@dg.asset_check(
    asset=raw_news,
    name="raw_news_contract",
    blocking=True,
    partitions_def=_DAILY_PARTITIONS,
)
def raw_news_contract(raw_news: pl.DataFrame) -> dg.AssetCheckResult:
    partition_date = None
    if not raw_news.is_empty() and "valid_time" in raw_news.columns:
        partition_date = str(raw_news["valid_time"].max())
    return _contract_check_result("raw_news", raw_news, partition_date)


@dg.asset_check(
    asset=raw_macro,
    name="raw_macro_contract",
    blocking=True,
    partitions_def=_DAILY_PARTITIONS,
)
def raw_macro_contract(raw_macro: pl.DataFrame) -> dg.AssetCheckResult:
    partition_date = None
    if not raw_macro.is_empty() and "valid_time" in raw_macro.columns:
        partition_date = str(raw_macro["valid_time"].max())
    return _contract_check_result("raw_macro", raw_macro, partition_date)


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


@dg.asset_check(
    asset=council_signal,
    name="council_signal_contract",
    blocking=True,
    partitions_def=_DAILY_PARTITIONS,
)
def council_signal_contract(council_signal: pd.Series) -> dg.AssetCheckResult:
    lineage = extract_lineage(council_signal)
    if council_signal.empty:
        payload = pd.DataFrame(columns=["ticker", "council_signal", *LINEAGE_COLUMNS])
    else:
        payload = pd.DataFrame(
            {
                "ticker": list(council_signal.index),
                "council_signal": council_signal.values,
            }
        )
        for key, values in dataframe_lineage_columns(lineage, len(payload)).items():
            payload[key] = values
    return _contract_check_result("council_signal", payload)


@dg.asset_check(
    asset=portfolio_weights,
    name="portfolio_weights_contract",
    blocking=True,
    partitions_def=_DAILY_PARTITIONS,
)
def portfolio_weights_contract(portfolio_weights: pd.Series) -> dg.AssetCheckResult:
    lineage = extract_lineage(portfolio_weights)
    if portfolio_weights.empty:
        payload = pd.DataFrame(columns=["ticker", "target_weight", *LINEAGE_COLUMNS])
    else:
        payload = pd.DataFrame(
            {
                "ticker": list(portfolio_weights.index),
                "target_weight": portfolio_weights.values,
            }
        )
        for key, values in dataframe_lineage_columns(lineage, len(payload)).items():
            payload[key] = values
    return _contract_check_result("portfolio_weights", payload)


@dg.asset_check(
    asset=daily_orders,
    name="daily_orders_contract",
    blocking=True,
    partitions_def=_DAILY_PARTITIONS,
)
def daily_orders_contract(daily_orders: pd.DataFrame) -> dg.AssetCheckResult:
    partition_date = None
    if not daily_orders.empty and "ticker" in daily_orders.columns:
        partition_date = "n/a"
    return _contract_check_result("daily_orders", daily_orders, partition_date)

defs = dg.Definitions(
    assets=_ALL_ASSETS,
    asset_checks=[
        alpha158_features_contract,
        sentiment_features_contract,
        council_signal_contract,
        portfolio_weights_contract,
        daily_orders_contract,
    ],
    jobs=[
        daily_job,
        train_hmm_job,
        cost_calibration_job,
        walkforward_promotion_job,
        tda_warning_job,
    ],
    schedules=[
        daily_schedule,
        hmm_schedule,
        cost_calibration_schedule,
        walkforward_promotion_schedule,
        tda_warning_schedule,
    ],
    sensors=[failure_sensor],
)
