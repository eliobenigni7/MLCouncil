"""Job, schedule e sensor della pipeline MLCouncil + elenco asset completo.

Parte del package data/pipeline (ex data/pipeline.py).
"""

import pytz
import dagster as dg
from dagster import RunFailureSensorContext
from datetime import date as date_type, timedelta

from ._shared import _DAILY_PARTITIONS
from .assets_ingest import raw_ohlcv, raw_news, raw_macro
from .assets_features import alpha158_features, sentiment_features
from .assets_signals import (
    lgbm_signals,
    sentiment_signals,
    current_regime,
    council_signal,
    save_council_results,
    save_regime_results,
    train_hmm,
)
from .assets_portfolio import (
    portfolio_weights,
    daily_orders,
    cost_calibration_artifact,
    cost_calibration_gate,
)
from .assets_monitoring import (
    canary_health,
    tda_warning_signal,
    causal_drift_check,
    model_promotion_gate,
)


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


causal_drift_job = dg.define_asset_job(
    name="causal_drift_job",
    selection=dg.AssetSelection.assets(causal_drift_check),
    description="Weekly causal graph drift check.",
)


@dg.schedule(
    cron_schedule="0 2 * * 1",
    execution_timezone="UTC",
    job=causal_drift_job,
)
def causal_drift_schedule(context: "dg.ScheduleEvaluationContext"):
    return dg.RunRequest(tags={"mlcouncil/job": "causal_drift"})


monitored_jobs = [daily_job, tda_warning_job, causal_drift_job]


@dg.run_failure_sensor(
    monitored_jobs=monitored_jobs,
    minimum_interval_seconds=60,
    description="Logga i fallimenti del daily_pipeline e dei job settimanali e segnala il run_id.",
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
    canary_health,
    train_hmm,
    cost_calibration_artifact,
    cost_calibration_gate,
    model_promotion_gate,
    tda_warning_signal,
    causal_drift_check,
    # train_hmm + cost_calibration_* + model_promotion_gate sono unpartitioned:
    # schedule dedicate (train_hmm_job, cost_calibration_job, walkforward_promotion_job).
]
