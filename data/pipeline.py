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
                    frames.append(pl.read_parquet(pq))
                except Exception:
                    pass
    if extra is not None and not extra.is_empty():
        frames.append(extra)
    if not frames:
        return pl.DataFrame()
    return (
        pl.concat(frames)
        .unique(["ticker", "valid_time"])
        .sort(["ticker", "valid_time"])
    )


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
        return pl.DataFrame({
            "ticker":          tickers,
            "valid_time":      [today] * len(tickers),
            "sentiment_score": [0.0] * len(tickers),
        })

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
        return _empty

    df = pl.DataFrame(records)
    context.log.info(
        f"sentiment_features [{partition_date}]: "
        f"{len(records)} ticker con sentiment"
    )
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
        return pd.Series(0.0, index=tickers, name="lgbm_signal")

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
        return pd.Series(0.0, index=tickers, name="lgbm_signal")

    signals = model.predict(alpha158_features)
    context.log.info(
        f"lgbm_signals [{partition_date}]: segnali per {len(signals)} ticker"
    )
    return signals.rename("lgbm_signal")


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
        return pd.Series(dtype=float, name="sentiment_signal")

    sent = (
        sentiment_features
        .to_pandas()
        .set_index("ticker")["sentiment_score"]
    )

    # Z-score cross-sezionale
    std = sent.std()
    if std > 1e-9:
        sent = (sent - sent.mean()) / std

    context.log.info(
        f"sentiment_signals [{partition_date}]: {len(sent)} ticker"
    )
    return sent.rename("sentiment_signal")


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

    combined = aggregator.aggregate(signals, regime=current_regime, date=today)
    context.log.info(
        f"council_signal [{partition_date}]: {len(combined)} ticker | "
        f"regime={current_regime}"
    )
    return combined.rename("council_signal")


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
        return pd.Series(dtype=float, name="target_weight")

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

    # Pesi correnti: equal-weight (giorno 1 / baseline)
    n = len(cov_tickers)
    current_w = pd.Series(np.ones(n) / n, index=cov_tickers)

    constructor = PortfolioConstructor()
    weights = constructor.optimize(
        alpha_signals=signal_aligned,
        position_multipliers=multipliers,
        current_weights=current_w,
        returns_covariance=cov,
    )

    context.log.info(
        f"portfolio_weights [{partition_date}]: {len(weights)} posizioni | "
        f"top3={weights.nlargest(3).round(3).to_dict()}"
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

    if portfolio_weights.empty:
        context.log.warning(
            f"daily_orders [{partition_date}]: nessun peso → nessun ordine"
        )
        empty_orders = pd.DataFrame(
            columns=["ticker", "direction", "quantity", "target_weight"]
        )
        empty_orders.to_parquet(_ORDERS_DIR / f"{partition_date}.parquet", index=False)
        return empty_orders

    PORTFOLIO_VALUE = 1_000_000.0
    n = len(portfolio_weights)
    current_w = pd.Series(
        np.ones(n) / n, index=portfolio_weights.index
    )

    constructor = PortfolioConstructor()
    orders = constructor.compute_orders(
        target_weights=portfolio_weights,
        current_weights=current_w,
        portfolio_value=PORTFOLIO_VALUE,
    )

    out_path = _ORDERS_DIR / f"{partition_date}.parquet"
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
                    frames.append(pl.read_parquet(pq))
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

defs = dg.Definitions(
    assets=_ALL_ASSETS,
    jobs=[daily_job],
    schedules=[daily_schedule],
    sensors=[failure_sensor],
)
