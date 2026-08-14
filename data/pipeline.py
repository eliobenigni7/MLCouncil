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

Entry point compatibile del package ``data/pipeline`` (ex monolite ~2700 righe,
ora suddiviso in data/pipeline/assets_*.py + data/pipeline/jobs.py): ri-esporta
tutto dal package e allinea il proprio namespace a ``data.pipeline._shared``,
così le patch dei test (monkeypatch.setattr su questo modulo) raggiungono le
costanti/helper usati dagli asset (vedi data/pipeline/_shared.py).
"""

import sys as _sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Path bootstrap — consente import relativi da qualsiasi working directory
# ---------------------------------------------------------------------------

_ROOT = Path(__file__).parents[1]
if str(_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_ROOT))

import data.pipeline as _impl  # noqa: E402
import data.pipeline._shared as _shared  # noqa: E402

# ---------------------------------------------------------------------------
# Ri-esportazione completa dal package data.pipeline
# ---------------------------------------------------------------------------

# Constants e helper condivisi
load_universe_as_of = _impl.load_universe_as_of
_load_universe = _impl._load_universe
_safe_pickle_load = _impl._safe_pickle_load
_normalize_df = _impl._normalize_df
_load_all_ohlcv = _impl._load_all_ohlcv
_load_partitioned_parquet = _impl._load_partitioned_parquet
_load_macro_context_from_disk = _impl._load_macro_context_from_disk
_record_asset_metadata = _impl._record_asset_metadata
_contract_check_result = _impl._contract_check_result
_load_live_portfolio_snapshot = _impl._load_live_portfolio_snapshot
_load_returns_wide = _impl._load_returns_wide
_compute_covariance = _impl._compute_covariance
_load_market_returns = _impl._load_market_returns
LivePortfolioSnapshotError = _impl.LivePortfolioSnapshotError
_DATA_DIR = _impl._DATA_DIR
_ORDERS_DIR = _impl._ORDERS_DIR
_RESULTS_DIR = _impl._RESULTS_DIR
_CHECKPOINTS = _impl._CHECKPOINTS
_EXCLUDE_COLS = _impl._EXCLUDE_COLS
_MIN_ALPHA_FEATURES = _impl._MIN_ALPHA_FEATURES
_DEFAULT_PORTFOLIO_VALUE = _impl._DEFAULT_PORTFOLIO_VALUE
_DAILY_PARTITIONS = _impl._DAILY_PARTITIONS
_RETRY = _impl._RETRY

# Asset
raw_ohlcv = _impl.raw_ohlcv
raw_news = _impl.raw_news
raw_macro = _impl.raw_macro
alpha158_features = _impl.alpha158_features
sentiment_features = _impl.sentiment_features
lgbm_signals = _impl.lgbm_signals
sentiment_signals = _impl.sentiment_signals
current_regime = _impl.current_regime
council_signal = _impl.council_signal
save_council_results = _impl.save_council_results
save_regime_results = _impl.save_regime_results
portfolio_weights = _impl.portfolio_weights
daily_orders = _impl.daily_orders
canary_health = _impl.canary_health
train_hmm = _impl.train_hmm
cost_calibration_artifact = _impl.cost_calibration_artifact
cost_calibration_gate = _impl.cost_calibration_gate
model_promotion_gate = _impl.model_promotion_gate
tda_warning_signal = _impl.tda_warning_signal
causal_drift_check = _impl.causal_drift_check

# Asset checks
raw_ohlcv_contract = _impl.raw_ohlcv_contract
raw_news_contract = _impl.raw_news_contract
raw_macro_contract = _impl.raw_macro_contract
alpha158_features_contract = _impl.alpha158_features_contract
sentiment_features_contract = _impl.sentiment_features_contract
council_signal_contract = _impl.council_signal_contract
portfolio_weights_contract = _impl.portfolio_weights_contract
daily_orders_contract = _impl.daily_orders_contract

# Job, schedule, sensor
daily_job = _impl.daily_job
daily_schedule = _impl.daily_schedule
train_hmm_job = _impl.train_hmm_job
hmm_schedule = _impl.hmm_schedule
cost_calibration_job = _impl.cost_calibration_job
cost_calibration_schedule = _impl.cost_calibration_schedule
walkforward_promotion_job = _impl.walkforward_promotion_job
walkforward_promotion_schedule = _impl.walkforward_promotion_schedule
tda_warning_job = _impl.tda_warning_job
tda_warning_schedule = _impl.tda_warning_schedule
causal_drift_job = _impl.causal_drift_job
causal_drift_schedule = _impl.causal_drift_schedule
monitored_jobs = _impl.monitored_jobs
failure_sensor = _impl.failure_sensor

# Definitions
defs = _impl.defs
_ALL_ASSETS = _impl._ALL_ASSETS

# Helper privati (ri-esportati per compatibilità con test/scripts)
_run_alpha158_features = _impl._run_alpha158_features
_build_online_refit_history = _impl._build_online_refit_history
_run_lgbm_signals = _impl._run_lgbm_signals
_run_council_signal = _impl._run_council_signal
_sunday_tags = _impl._sunday_tags
_run_portfolio_weights = _impl._run_portfolio_weights
_run_daily_orders = _impl._run_daily_orders
_pipeline_crypto_check = _impl._pipeline_crypto_check
_lineage_from_daily_orders = _impl._lineage_from_daily_orders
_build_canary_metrics = _impl._build_canary_metrics
_CANARY_METRIC_PROXIES = _impl._CANARY_METRIC_PROXIES
_prev_trading_day = _impl._prev_trading_day

# ---------------------------------------------------------------------------
# Namespace condiviso con _shared (vedi data/pipeline/_shared.py)
# ---------------------------------------------------------------------------
# Le patch dei test (monkeypatch.setattr / patch.object su questo modulo,
# es. _DATA_DIR, _safe_pickle_load) vengono inoltrate anche a _shared.__dict__,
# lo stato effettivamente letto dagli asset del package.
_sys.modules[__name__].__class__ = _shared._SharedNamespaceModule
