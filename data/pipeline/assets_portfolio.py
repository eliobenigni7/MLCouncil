"""Layer 4 — Portfolio: portfolio_weights, daily_orders + cost calibration (nightly).

Parte del package data/pipeline (ex data/pipeline.py). Gli asset accedono alle
costanti/helper condivisi via ``_shared.NOME`` (vedi data/pipeline/_shared.py).
"""

import numpy as np
import pandas as pd
import polars as pl
import dagster as dg
from dagster import AssetExecutionContext

from council.artifacts import write_artifact_manifest
from data.contracts import LINEAGE_COLUMNS
from data.lineage import (
    attach_lineage,
    build_pipeline_run_id,
    dataframe_lineage_columns,
    extract_lineage,
    lineage_artifact_payload,
)
from observability.tracing import trace_span

from . import _shared
from ._shared import (
    _DAILY_PARTITIONS,
    _RETRY,
    _ROOT,
    _EXCLUDE_COLS,
    _record_asset_metadata,
    _contract_check_result,
    _load_market_returns,
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
    from council.sizing.cqr import get_position_sizer, position_sizer_checkpoint_name, position_sizing_mode
    from council.portfolio.portfolio_diff import get_portfolio_constructor

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
    cov_df = _shared._compute_covariance(tickers)
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
        from council.sizing.fractional_kelly import FractionalKellySizer

        sizer = FractionalKellySizer()
        context.log.info(
            f"portfolio_weights [{partition_date}]: "
            "FractionalKellySizer istanziato direttamente"
        )
        # Kelly sizer non usa features, passa None
        multipliers = sizer.compute_position_multipliers(signal_aligned)
    else:
        sizer_checkpoint = _shared._CHECKPOINTS / position_sizer_checkpoint_name()
        if sizer_checkpoint.exists():
            sizer = _shared._safe_pickle_load(sizer_checkpoint)
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
    current_w, portfolio_value = _shared._load_live_portfolio_snapshot(cov_tickers)

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
    from council.risk.risk_engine import RiskEngine
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
    from council.portfolio.portfolio import PortfolioConstructor

    _shared._ORDERS_DIR.mkdir(parents=True, exist_ok=True)

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
        empty_path = _shared._ORDERS_DIR / f"{partition_date}.parquet"
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

    current_w, portfolio_value = _shared._load_live_portfolio_snapshot()

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

    out_path = _shared._ORDERS_DIR / f"{partition_date}.parquet"
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
