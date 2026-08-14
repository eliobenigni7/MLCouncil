"""Dagster pipeline package: orchestrazione giornaliera dell'intera catena MLCouncil.

Layer 1 (Ingest)  →  raw_ohlcv, raw_news, raw_macro
Layer 2 (Features) → alpha158_features, sentiment_features
Layer 3 (Signals)  → lgbm_signals, sentiment_signals, current_regime
Layer 4 (Council)  → council_signal, portfolio_weights, daily_orders

Schedule: 21:30 ET ogni giorno lavorativo (lun-ven).
Ogni asset è configurato con RetryPolicy(max_retries=2).

Questo modulo assembla i layer definiti in ``data/pipeline/assets_*.py`` e
``data/pipeline/jobs.py`` e costruisce le ``defs`` Dagster. Il file
``data/pipeline.py`` resta l'entry point compatibile (per ``dagster dev -f
data/pipeline.py`` e per i test che lo caricano standalone): ri-esporta tutto
da qui e allinea il proprio namespace a ``_shared`` (vedi data/pipeline/_shared.py).
"""

import sys as _sys

import dagster as dg

from observability.tracing import init_tracing

init_tracing(service_name="mlcouncil-dagster")

try:
    from council.production_config import apply_manifest_to_environ

    apply_manifest_to_environ()
except Exception:
    pass

from . import _shared
from ._shared import (
    _ROOT,
    _DATA_DIR,
    _ORDERS_DIR,
    _RESULTS_DIR,
    _CHECKPOINTS,
    _EXCLUDE_COLS,
    _MIN_ALPHA_FEATURES,
    _DEFAULT_PORTFOLIO_VALUE,
    LivePortfolioSnapshotError,
    _DAILY_PARTITIONS,
    _RETRY,
    _safe_pickle_load,
    _load_universe,
    load_universe_as_of,
    _normalize_df,
    _load_all_ohlcv,
    _load_partitioned_parquet,
    _load_macro_context_from_disk,
    _record_asset_metadata,
    _contract_check_result,
    _load_live_portfolio_snapshot,
    _load_returns_wide,
    _compute_covariance,
    _load_market_returns,
)
from .assets_ingest import (
    raw_ohlcv,
    raw_news,
    raw_macro,
    raw_ohlcv_contract,
    raw_news_contract,
    raw_macro_contract,
)
from .assets_features import (
    alpha158_features,
    sentiment_features,
    alpha158_features_contract,
    sentiment_features_contract,
    _run_alpha158_features,
)
from .assets_signals import (
    lgbm_signals,
    sentiment_signals,
    current_regime,
    train_hmm,
    train_hmm_job,
    hmm_schedule,
    council_signal,
    save_council_results,
    save_regime_results,
    council_signal_contract,
    _build_online_refit_history,
    _run_lgbm_signals,
    _run_council_signal,
    _sunday_tags,
)
from .assets_portfolio import (
    portfolio_weights,
    daily_orders,
    cost_calibration_artifact,
    cost_calibration_gate,
    cost_calibration_job,
    cost_calibration_schedule,
    portfolio_weights_contract,
    daily_orders_contract,
    _run_portfolio_weights,
    _run_daily_orders,
    _pipeline_crypto_check,
    _lineage_from_daily_orders,
)
from .assets_monitoring import (
    canary_health,
    tda_warning_signal,
    causal_drift_check,
    model_promotion_gate,
    walkforward_promotion_job,
    walkforward_promotion_schedule,
    _build_canary_metrics,
    _CANARY_METRIC_PROXIES,
)
from .jobs import (
    daily_job,
    daily_schedule,
    tda_warning_job,
    tda_warning_schedule,
    causal_drift_job,
    causal_drift_schedule,
    monitored_jobs,
    failure_sensor,
    _ALL_ASSETS,
    _prev_trading_day,
)

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
        causal_drift_job,
    ],
    schedules=[
        daily_schedule,
        hmm_schedule,
        cost_calibration_schedule,
        walkforward_promotion_schedule,
        tda_warning_schedule,
        causal_drift_schedule,
    ],
    sensors=[failure_sensor],
)


# ---------------------------------------------------------------------------
# Namespace condiviso con _shared (vedi data/pipeline/_shared.py)
# ---------------------------------------------------------------------------
# I test (tests/test_pipeline.py, tests/test_canary.py) caricano data/pipeline.py
# come modulo standalone e patchano costanti/helper (es. _DATA_DIR,
# _safe_pickle_load) sul modulo entry point. Adottando _SharedNamespaceModule,
# ogni setattr su questo package (o sull'entry point) viene inoltrato anche a
# _shared.__dict__, lo stato effettivamente letto dagli asset (che accedono ai
# nomi come _shared.NOME).
_sys.modules[__name__].__class__ = _shared._SharedNamespaceModule
