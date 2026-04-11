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
from datetime import date as date_type
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


def _load_universe() -> list[str]:
    """Carica la lista dei ticker da config/universe.yaml.

    Supporta sia il formato legacy con `universe.tickers` sia il formato
    bucketed corrente (`large_cap`, `mid_cap`, ...), ignorando la sezione
    `settings`.
    """
    with open(_ROOT / "config" / "universe.yaml") as f:
        cfg = yaml.safe_load(f)
    universe_cfg = cfg.get("universe", {})

    if isinstance(universe_cfg.get("tickers"), list):
        tickers = universe_cfg["tickers"]
    else:
        tickers = []
        for bucket_name, bucket_values in universe_cfg.items():
            if bucket_name == "settings" or not isinstance(bucket_values, list):
                continue
            tickers.extend(bucket_values)

    deduped: list[str] = []
    seen: set[str] = set()
    for ticker in tickers:
        if ticker not in seen:
            seen.add(ticker)
            deduped.append(ticker)
    return deduped


def _normalize_df(df: pl.DataFrame) -> pl.DataFrame:
    """Normalize datetime columns to UTC timezone for Polars 1.x strict concat."""
    if df.is_empty():
        return df
    tz_cols = [c for c in df.columns if df[c].dtype == pl.Datetime]
    if not tz_cols:
        return df
    return df.with_columns(
        pl.col(c).dt.replace_time_zone("UTC").dt.replace_time_zone(None) for c in tz_cols
    )


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

        current_weights = (
            positions_df
            .assign(current_value=current_values)
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
def raw_ohlcv(context: AssetExecutionContext) -> pl.DataFrame:
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
    return df


@dg.asset(
    partitions_def=_DAILY_PARTITIONS,
    retry_policy=_RETRY,
    description="Headline di notizie finanziarie dal feed RSS Yahoo Finance.",
)
def raw_news(context: AssetExecutionContext) -> pl.DataFrame:
    """Scarica le notizie per la data di partizione."""
    from data.ingest.news import download_news

    partition_date = context.partition_key
    tickers = _load_universe()

    df = download_news(tickers=tickers, date=partition_date, data_dir=_DATA_DIR)
    context.log.info(f"raw_news [{partition_date}]: {df.shape[0]} headline")
    _record_asset_metadata(context, "raw_news", df, partition_date)
    return df


@dg.asset(
    partitions_def=_DAILY_PARTITIONS,
    retry_policy=_RETRY,
    description="Dati macro (VIX, Treasury spread, S&P500) da FRED.",
)
def raw_macro(context: AssetExecutionContext) -> pl.DataFrame:
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
    return macro


# ===========================================================================
# LAYER 2 — FEATURES
# ===========================================================================

@dg.asset(
    partitions_def=_DAILY_PARTITIONS,
    retry_policy=_RETRY,
    description="Feature tecniche + macro look-ahead safe per il modello tecnico.",
)
def alpha158_features(
    context: AssetExecutionContext,
    raw_ohlcv: pl.DataFrame,
    raw_macro: pl.DataFrame,
) -> pl.DataFrame:
    """Calcola le feature Alpha158 sull'OHLCV storico + contesto macro.

    Per le rolling window (fino a 252 giorni) carica l'intera storia
    disponibile su disco, poi restituisce solo le righe del giorno corrente.
    """
    from data.features.alpha158 import compute_alpha158

    partition_date = context.partition_key
    today = date_type.fromisoformat(partition_date)

    # Alpha158 richiede la storia completa per le rolling window
    all_ohlcv = _load_all_ohlcv(extra=raw_ohlcv)
    if all_ohlcv.is_empty():
        raise ValueError("Nessun dato OHLCV disponibile per Alpha158")

    macro_ctx = raw_macro if not raw_macro.is_empty() else None
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
)
def sentiment_features(
    context: AssetExecutionContext,
    raw_news: pl.DataFrame,
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
)
def current_regime(
    context: AssetExecutionContext,
    raw_macro: pl.DataFrame,
) -> str:
    """Rileva il regime di mercato con il modello HMM."""
    partition_date = context.partition_key
    checkpoint = _CHECKPOINTS / "hmm_latest.pkl"

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
        context.log.warning(
            f"current_regime [{partition_date}]: "
            "checkpoint non trovato — training inline"
        )
        if raw_macro.is_empty():
            return "transition"
        regime_model = RegimeModel()
        try:
            regime_model.fit(raw_macro)
        except Exception as exc:
            context.log.warning(
                f"HMM fit fallito: {exc} — fallback a 'transition'"
            )
            return "transition"

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


# ===========================================================================
# LAYER 4 — COUNCIL
# ===========================================================================

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
    current_w, _ = _load_live_portfolio_snapshot(cov_tickers)

    constructor = PortfolioConstructor()
    weights = constructor.optimize(
        alpha_signals=signal_aligned,
        position_multipliers=multipliers,
        current_weights=current_w,
        returns_covariance=cov,
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

    ohlcv = pl.concat(frames).sort(["ticker", "valid_time"])
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

daily_schedule = dg.ScheduleDefinition(
    job=daily_job,
    cron_schedule="30 21 * * 1-5",   # 21:30 ET, lun-ven
    execution_timezone="America/New_York",
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
    portfolio_weights,
    daily_orders,
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
        raw_ohlcv_contract,
        raw_news_contract,
        raw_macro_contract,
        alpha158_features_contract,
        sentiment_features_contract,
        daily_orders_contract,
    ],
    jobs=[daily_job],
    schedules=[daily_schedule],
    sensors=[failure_sensor],
)
