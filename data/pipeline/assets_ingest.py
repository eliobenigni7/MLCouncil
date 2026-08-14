"""Layer 1 — Ingest: raw_ohlcv, raw_news, raw_macro + contract check.

Parte del package data/pipeline (ex data/pipeline.py). Gli asset accedono alle
costanti/helper condivisi via ``_shared.NOME`` (vedi data/pipeline/_shared.py).
"""

import polars as pl
import dagster as dg
from dagster import AssetExecutionContext
from datetime import date as date_type

from observability.tracing import trace_span

from . import _shared
from ._shared import (
    _DAILY_PARTITIONS,
    _RETRY,
    _load_universe,
    _record_asset_metadata,
    _contract_check_result,
)


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

    # Canary (F-0.4): applica le feature attive come policy di run, PRIMA che
    # gli asset che leggono i flag eseguano. raw_ohlcv è la radice del grafo:
    # ogni consumatore di flag canary (lgbm_signals, council_signal,
    # portfolio_weights, daily_orders) è transitivamente a valle, quindi con
    # esecuzione in-process i flag sono attivi al primo lettore. Il revert
    # disabilita la feature nello stato persistito (effettivo dalla run
    # successiva): niente mutazione mid-run (vedi council/canary.py).
    # Nota: con executor multiprocesso applicare i flag anche a livello di
    # import del modulo (come apply_manifest_to_environ).
    try:
        from council.canary import apply_canary_features

        applied = apply_canary_features()
        if applied:
            context.log.info(
                f"raw_ohlcv [{partition_date}]: canary attive ({', '.join(applied)})"
            )
    except Exception as exc:  # noqa: BLE001
        context.log.warning(
            f"raw_ohlcv [{partition_date}]: canary apply failed ({exc})"
        )

    with trace_span(
        "mlcouncil.ingest.raw_ohlcv",
        layer="ingest",
        asset="raw_ohlcv",
        partition_date=partition_date,
    ):
        tickers = _load_universe()

        df = download_daily(tickers=tickers, date=partition_date, data_dir=_shared._DATA_DIR)

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
        df = download_news(tickers=tickers, date=partition_date, data_dir=_shared._DATA_DIR)
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
        download_macro(end=partition_date, data_dir=_shared._DATA_DIR)

    macro_dir = _shared._DATA_DIR / "macro"

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
