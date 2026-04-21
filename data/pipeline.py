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

from data.contracts import validate_asset_contract, version_payload
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
    """Load a pickle checkpoint only after verifying its SHA256 hash sidecar.

    Raises ValueError if a .hash sidecar exists and the digest does not match,
    preventing execution of tampered checkpoint files.
    """
    hash_path = path.with_suffix(path.suffix + ".hash")
    if hash_path.exists():
        expected = hash_path.read_text().strip()
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(
                f"Checkpoint hash mismatch for {path}: "
                f"expected {expected}, got {actual}. "
                "File may be corrupted or tampered with."
            )
    with open(path, "rb") as fh:
        return pickle.load(fh)


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
    macro_dir = _DATA_DIR / "raw" / "macro"

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
    from data.features.alpha158 import compute_alpha158

    partition_date = context.partition_key
    today = date_type.fromisoformat(partition_date)

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
        "ticker":          pl.Series([], dtype=pl.Utf8),
        "valid_time":      pl.Series([], dtype=pl.Date),
        "sentiment_score": pl.Series([], dtype=pl.Float64),
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
        from data.ingest.news_processor import clean_headline
    except ImportError as exc:
        context.log.warning(
            f"sentiment_features [{partition_date}]: "
            f"FinBERT non disponibile ({exc}), fallback a 0.0"
        )
        tickers = raw_news["ticker"].unique().to_list()
        fallback = pl.DataFrame({
            "ticker":          tickers,
            "valid_time":      [today] * len(tickers),
            "sentiment_score": [0.0] * len(tickers),
        })
        _record_asset_metadata(context, "sentiment_features", fallback, partition_date)
        return fallback

    news_pd = raw_news.to_pandas()
    news_pd["clean_title"] = news_pd["title"].apply(clean_headline)
    model = SentimentModel()

    # Batch all headlines across tickers into a single forward pass so that
    # the transformer's cache and GPU batching are used efficiently.
    all_headlines: list[str] = news_pd["clean_title"].tolist()
    all_tickers:   list[str] = news_pd["ticker"].tolist()
    try:
        all_scores = model.score_headlines(all_headlines)
    except Exception:
        all_scores = [0.0] * len(all_headlines)

    # Aggregate per-ticker: simple mean of headline scores.
    from collections import defaultdict
    ticker_scores: dict[str, list[float]] = defaultdict(list)
    for ticker, score in zip(all_tickers, all_scores):
        ticker_scores[ticker].append(score)

    records: list[dict] = []
    for ticker, scores in ticker_scores.items():
        records.append({
            "ticker":          ticker,
            "valid_time":      today,
            "sentiment_score": float(sum(scores) / len(scores)) if scores else 0.0,
        })

    if not records:
        _record_asset_metadata(context, "sentiment_features", _empty, partition_date)
        return _empty

    df = pl.DataFrame(records)
    context.log.info(
        f"sentiment_features [{partition_date}]: "
        f"{len(records)} ticker con sentiment"
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
    if checkpoint.exists():
        model.load(str(checkpoint))
        context.log.info(
            f"lgbm_signals [{partition_date}]: checkpoint caricato da {checkpoint}"
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

    signals = model.predict(alpha158_features).rename("lgbm_signal")
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
    context.add_output_metadata(lineage_artifact_payload(lineage, signal_count=len(signals)))
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
    context.log.info(f"train_hmm: checkpoint salvato in {checkpoint_path}")

    # Genera regime history per la dashboard
    try:
        history_df = regime_model.get_regime_history(macro)
        history_path = _RESULTS_DIR / "regime_history.parquet"
        _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        history_df.to_parquet(history_path, index=False)
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


hmm_schedule = dg.ScheduleDefinition(
    job=train_hmm_job,
    cron_schedule="0 23 * * 0",       # 23:00 ET ogni domenica
    execution_timezone="America/New_York",
    tags_fn=_sunday_tags,
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
    from council.aggregator import CouncilAggregator

    partition_date = context.partition_key
    today = date_type.fromisoformat(partition_date)

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

    combined = aggregator.aggregate(signals, regime=current_regime, date=today).rename("council_signal")
    hmm_version = checkpoint_version(_CHECKPOINTS / "hmm_latest.pkl", "hmm-inline")
    lineage = merge_lineage(
        lgbm_signals,
        sentiment_signals,
        context=context,
        partition_date=partition_date,
        model_version=hmm_version,
    )
    combined = attach_lineage(combined, **lineage)
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
    aggregator.save(str(_RESULTS_DIR / "aggregator.pkl"))

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
                attr_rows.append({
                    "date": pd.Timestamp(log_date),
                    "model_name": model_name,
                    "weight": weights_used.get(model_name, float("nan")),
                    "ic_rolling_30d": ic_30d,
                    "sharpe_rolling_60d": sharpe_60d,
                    "pnl_contribution": contributions.get(model_name, float("nan")),
                })

        if attr_rows:
            attr_df = pd.DataFrame(attr_rows, columns=[
                "date", "model_name", "weight",
                "ic_rolling_30d", "sharpe_rolling_60d", "pnl_contribution",
            ])
            attr_df.to_parquet(_RESULTS_DIR / "attribution.parquet", index=False)
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
                import pickle as pickle_mod
                with open(checkpoint, "rb") as f:
                    regime_model: RegimeModel = pickle_mod.load(f)
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
                import pickle as pickle_mod
                with open(checkpoint, "rb") as f:
                    regime_model: RegimeModel = pickle_mod.load(f)
                hist_df = regime_model.get_regime_history(raw_macro)
                if "valid_time" in hist_df.columns:
                    hist_df = hist_df.rename(columns={"valid_time": "date"})
                if "date" in hist_df.columns:
                    hist_df["date"] = pd.to_datetime(hist_df["date"])
                hist_df.to_parquet(_RESULTS_DIR / "regime_history.parquet", index=False)
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
) -> pd.Series:
    """Ottimizza il portafoglio con conformal sizing e covariance storica.

    Se il conformal sizer non è disponibile usa moltiplicatori unitari.
    La matrice di covarianza è calcolata sulle ultime 90 sessioni disponibili.
    """
    from council.portfolio import PortfolioConstructor
    from council.conformal import ConformalPositionSizer

    partition_date = context.partition_key

    if council_signal.empty:
        context.log.warning(
            f"portfolio_weights [{partition_date}]: nessun segnale ricevuto"
        )
        empty = pd.Series(dtype=float, name="target_weight")
        lineage = extract_lineage(council_signal)
        context.add_output_metadata(lineage_artifact_payload(lineage, position_count=0))
        return attach_lineage(empty, **lineage)

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

    # Conformal position sizing
    sizer_checkpoint = _CHECKPOINTS / "conformal_sizer.pkl"
    if sizer_checkpoint.exists():
        sizer = _safe_pickle_load(sizer_checkpoint)
        context.log.info(
            f"portfolio_weights [{partition_date}]: "
            f"conformal sizer caricato da {sizer_checkpoint}"
        )
        n = len(cov_tickers)
        X_dummy = np.zeros((n, sizer._n_features or 1))
        multipliers = sizer.compute_position_multipliers(signal_aligned, X_dummy)
    else:
        context.log.warning(
            f"portfolio_weights [{partition_date}]: "
            "conformal sizer non trovato — multipliers=1.0"
        )
        multipliers = pd.Series(1.0, index=cov_tickers, name="multiplier")

    # Pesi correnti: portafoglio live se disponibile, altrimenti bootstrap da zero.
    current_w, portfolio_value = _load_live_portfolio_snapshot(cov_tickers)

    constructor = PortfolioConstructor()
    optimize_with_crypto = getattr(constructor, "optimize_with_crypto", None)
    if callable(optimize_with_crypto):
        weights = optimize_with_crypto(
            alpha_signals=signal_aligned,
            position_multipliers=multipliers,
            current_weights=current_w,
            returns_covariance=cov,
            portfolio_value=portfolio_value,
        )
    else:
        weights = constructor.optimize(
            alpha_signals=signal_aligned,
            position_multipliers=multipliers,
            current_weights=current_w,
            returns_covariance=cov,
            portfolio_value=portfolio_value,
        )

    weights = attach_lineage(weights.rename("target_weight"), **extract_lineage(council_signal))
    context.log.info(
        f"portfolio_weights [{partition_date}]: {len(weights)} posizioni | "
        f"top3={weights.nlargest(3).round(3).to_dict()}"
    )
    context.add_output_metadata(
        lineage_artifact_payload(extract_lineage(weights), position_count=len(weights))
    )
    return weights


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
    from council.portfolio import PortfolioConstructor

    partition_date = context.partition_key

    _ORDERS_DIR.mkdir(parents=True, exist_ok=True)

    lineage = extract_lineage(portfolio_weights)
    if not lineage:
        lineage = {
            "pipeline_run_id": build_pipeline_run_id(context, partition_date),
            "data_version": "unknown",
            "feature_version": "unknown",
            "model_version": "unknown",
        }

    if portfolio_weights.empty:
        context.log.warning(
            f"daily_orders [{partition_date}]: nessun peso → nessun ordine"
        )
        empty_orders = pd.DataFrame(
            columns=["ticker", "direction", "quantity", "target_weight", *dataframe_lineage_columns(lineage, 0).keys()]
        )
        empty_orders.to_parquet(_ORDERS_DIR / f"{partition_date}.parquet", index=False)
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
        orders = pd.DataFrame(columns=["ticker", "direction", "quantity", "target_weight"])

    for key, values in dataframe_lineage_columns(lineage, len(orders)).items():
        orders[key] = values

    out_path = _ORDERS_DIR / f"{partition_date}.parquet"
    _record_asset_metadata(context, "daily_orders", orders, partition_date, lineage)
    orders.to_parquet(out_path, index=False)
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
    return returns_wide.cov(min_periods=30)


# ===========================================================================
# JOB, SCHEDULE, SENSOR
# ===========================================================================

daily_job = dg.define_asset_job(
    name="daily_pipeline",
    selection=dg.AssetSelection.all(),
    partitions_def=_DAILY_PARTITIONS,
    description=(
        "Pipeline giornaliera MLCouncil: ingest → features → signals "
        "→ council → orders"
    ),
)

def _daily_partition_tags(context: "dg.ScheduleEvaluationContext") -> dict[str, str]:
    """Ritorna i tags per la partition del job daily_pipeline.

    Per uno schedule che gira alle 21:30 ET lun-ven, la partition da processare
    è il giorno di mercato precedente (ieri). Se ieri era weekend, usa venerdi.
    """
    et = pytz.timezone("America/New_York")
    scheduled = context.scheduled_execution_time
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
    return {"dagster/partition": partition_date.strftime("%Y-%m-%d")}


daily_schedule = dg.ScheduleDefinition(
    job=daily_job,
    cron_schedule="30 21 * * 1-5",   # 21:30 ET, lun-ven
    execution_timezone="America/New_York",
    tags_fn=_daily_partition_tags,
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
        f"  Partizione: {failed_run.tags.get('dagster/partition', 'N/A')}\n"
        f"  Errore    : {error}\n"
        f"  Re-run    : dagster job execute -j daily_pipeline "
        f"--partition {failed_run.tags.get('dagster/partition', '')}"
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
    # train_hmm è escluso da daily_pipeline perché è unpartitioned
    # (schedule: domenicale 23:00 ET tramite train_hmm_job)
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
        daily_orders_contract,
    ],
    jobs=[daily_job, train_hmm_job],
    schedules=[daily_schedule, hmm_schedule],
    sensors=[failure_sensor],
)
