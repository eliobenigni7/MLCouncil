"""Layer 3 — Signals & Council: lgbm_signals, sentiment_signals, current_regime,
train_hmm, council_signal, save_council_results, save_regime_results.

Parte del package data/pipeline (ex data/pipeline.py). Gli asset accedono alle
costanti/helper condivisi via ``_shared.NOME`` (vedi data/pipeline/_shared.py).
"""

import pytz
import numpy as np
import pandas as pd
import polars as pl
import dagster as dg
from dagster import AssetExecutionContext
from datetime import date as date_type

from council.artifacts import write_artifact_manifest
from data.contracts import LINEAGE_COLUMNS, version_payload
from data.lineage import (
    attach_lineage,
    build_feature_lineage,
    build_pipeline_run_id,
    checkpoint_version,
    dataframe_lineage_columns,
    extract_lineage,
    lineage_artifact_payload,
    merge_lineage,
)
from observability.tracing import trace_span

from . import _shared
from .assets_ingest import raw_macro
from ._shared import (
    _DAILY_PARTITIONS,
    _RETRY,
    _ROOT,
    _record_asset_metadata,
    _contract_check_result,
)


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
    all_ohlcv = _shared._load_all_ohlcv()
    if all_ohlcv.is_empty():
        return pl.DataFrame(), pd.Series(dtype=float), all_ohlcv

    ohlcv = filter_features_from_date(
        all_ohlcv,
        as_of=today,
        lookback_days=lookback_days,
    )
    macro_ctx = _shared._load_macro_context_from_disk()
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
    checkpoint = _shared._CHECKPOINTS / "lgbm_latest.pkl"
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
            regime_hist = load_regime_history(_shared._RESULTS_DIR / "regime_history.parquet")
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
                checkpoint=_shared._CHECKPOINTS / "meta_label_latest.pkl",
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
    checkpoint = _shared._CHECKPOINTS / "hmm_latest.pkl"

    raw_macro = _shared._load_macro_context_from_disk()

    try:
        from models.regime import RegimeModel
    except ModuleNotFoundError as exc:
        context.log.warning(
            f"current_regime [{partition_date}]: dipendenza HMM mancante "
            f"({exc}) - fallback a 'transition'"
        )
        return "transition"

    if checkpoint.exists():
        regime_model = _shared._safe_pickle_load(checkpoint)
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

_shared._CHECKPOINTS.mkdir(parents=True, exist_ok=True)


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
    download_macro(end=today, data_dir=_shared._DATA_DIR)

    macro_dir = _shared._DATA_DIR / "macro"

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
    checkpoint_path = _shared._CHECKPOINTS / "hmm_latest.pkl"
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
        history_path = _shared._RESULTS_DIR / "regime_history.parquet"
        _shared._RESULTS_DIR.mkdir(parents=True, exist_ok=True)
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
    from council.aggregation.aggregator import CouncilAggregator

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
    raw_macro = _shared._load_macro_context_from_disk()
    regime_embedding, regime_centroids = load_regime_context(raw_macro, current_regime)

    from council.aggregation.moe_gating import log_moe_shadow, moe_enabled

    if moe_enabled() and len(signals) >= 2:
        # Shadow MoE: iniettiamo il modo di aggregazione come parametro,
        # senza mutare l'ambiente globale (niente set-env con switch-and-restore).
        linear_signal = aggregator.aggregate(
            signals,
            regime=current_regime,
            date=today,
            regime_embedding=regime_embedding,
            regime_centroids=regime_centroids,
            aggregator_mode_override="linear",
        )
        moe_signal = aggregator.aggregate(
            signals,
            regime=current_regime,
            date=today,
            regime_embedding=regime_embedding,
            regime_centroids=regime_centroids,
            aggregator_mode_override="moe",
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

    from council.sizing.cqr import (
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

    hmm_version = checkpoint_version(_shared._CHECKPOINTS / "hmm_latest.pkl", "hmm-inline")
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
    from council.aggregation.aggregator import CouncilAggregator

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
    _shared._RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    aggregator_path = _shared._RESULTS_DIR / "aggregator.pkl"
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
            attr_df.to_parquet(_shared._RESULTS_DIR / "attribution.parquet", index=False)
            write_artifact_manifest(
                _shared._RESULTS_DIR / "attribution.parquet",
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

    _shared._RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    raw_macro = _shared._load_macro_context_from_disk()

    # ------------------------------------------------------------------
    # current_regime.json
    # ------------------------------------------------------------------
    probs: dict[str, float] = {"bull": 0.0, "bear": 0.0, "transition": 0.0}
    if not raw_macro.is_empty():
        try:
            from models.regime import RegimeModel
            checkpoint = _shared._CHECKPOINTS / "hmm_latest.pkl"
            if checkpoint.exists():
                regime_model: RegimeModel = _shared._safe_pickle_load(checkpoint)
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
    with open(_shared._RESULTS_DIR / "current_regime.json", "w") as f:
        json.dump(regime_payload, f, indent=2, default=str)
    write_artifact_manifest(
        _shared._RESULTS_DIR / "current_regime.json",
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
            checkpoint = _shared._CHECKPOINTS / "hmm_latest.pkl"
            if checkpoint.exists():
                regime_model: RegimeModel = _shared._safe_pickle_load(checkpoint)
                hist_df = regime_model.get_regime_history(raw_macro)
                if "valid_time" in hist_df.columns:
                    hist_df = hist_df.rename(columns={"valid_time": "date"})
                if "date" in hist_df.columns:
                    hist_df["date"] = pd.to_datetime(hist_df["date"])
                hist_df.to_parquet(_shared._RESULTS_DIR / "regime_history.parquet", index=False)
                write_artifact_manifest(
                    _shared._RESULTS_DIR / "regime_history.parquet",
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
