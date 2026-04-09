"""Fundamentals ingestion (balance sheet, income statement, cash flow, earnings) via yfinance."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Optional
from zoneinfo import ZoneInfo

import polars as pl
import yfinance as yf
import yaml
from loguru import logger

_ROOT = Path(__file__).parents[2]


def _load_config() -> dict:
    with open(_ROOT / "config" / "universe.yaml") as f:
        return yaml.safe_load(f)


def _config_settings(cfg: dict) -> dict:
    return cfg.get("settings") or cfg.get("universe", {}).get("settings", {})


def _config_tickers(cfg: dict) -> list[str]:
    universe_cfg = cfg.get("universe", {})
    if isinstance(universe_cfg.get("tickers"), list):
        return universe_cfg["tickers"]
    tickers: list[str] = []
    for bucket_name, bucket_values in universe_cfg.items():
        if bucket_name == "settings" or not isinstance(bucket_values, list):
            continue
        tickers.extend(bucket_values)
    return tickers


def _transaction_time(cfg: dict) -> datetime:
    settings = _config_settings(cfg)
    tz = ZoneInfo(settings["transaction_timezone"])
    local_dt = datetime.now(tz).replace(
        hour=settings["transaction_time_hour"],
        minute=settings["transaction_time_minute"],
        second=0,
        microsecond=0,
    )
    return local_dt.astimezone(ZoneInfo("UTC"))


def _wide_to_long(ticker: str, pd_df, statement: str, tx_time: datetime) -> Optional[pl.DataFrame]:
    """Convert a yfinance wide financial statement (cols=dates) to long Polars DF."""
    if pd_df is None or pd_df.empty:
        return None

    try:
        df = pd_df.T.reset_index()
        df.columns = ["valid_time"] + list(pd_df.index)
        df.columns = [
            c if c == "valid_time" else str(c).lower().replace(" ", "_").replace("/", "_")
            for c in df.columns
        ]
        pl_df = pl.from_pandas(df)

        if pl_df["valid_time"].dtype != pl.Date:
            pl_df = pl_df.with_columns(pl.col("valid_time").cast(pl.Date))

        pl_df = pl_df.with_columns(
            [
                pl.lit(ticker).cast(pl.Utf8).alias("ticker"),
                pl.lit(statement).cast(pl.Utf8).alias("statement"),
                pl.lit(tx_time).cast(pl.Datetime("us", "UTC")).alias("transaction_time"),
            ]
        )

        # Move key columns to front
        front = ["ticker", "statement", "valid_time", "transaction_time"]
        rest = [c for c in pl_df.columns if c not in front]
        pl_df = pl_df.select(front + rest)

        # Cast all numeric cols to Float64
        pl_df = pl_df.with_columns(
            [
                pl.col(c).cast(pl.Float64, strict=False)
                for c in rest
                if pl_df[c].dtype not in (pl.Utf8, pl.Date, pl.Datetime)
            ]
        )
        return pl_df

    except Exception as exc:
        logger.warning(f"[{ticker}] error converting {statement}: {exc}")
        return None


def _earnings_to_df(ticker: str, yf_ticker: yf.Ticker, tx_time: datetime) -> Optional[pl.DataFrame]:
    """Extract earnings calendar (EPS actual vs estimate + announcement date)."""
    try:
        # earnings_dates is a DataFrame with DatetimeIndex
        ed = yf_ticker.earnings_dates
        if ed is None or ed.empty:
            return None

        ed = ed.reset_index()
        col_map = {}
        for c in ed.columns:
            key = str(c).lower().replace(" ", "_")
            col_map[c] = key
        ed = ed.rename(columns=col_map)

        pl_df = pl.from_pandas(ed)

        # Normalise the date column (may be 'earnings_date' or similar)
        date_col = next((c for c in pl_df.columns if "date" in c.lower()), None)
        if date_col and date_col != "valid_time":
            pl_df = pl_df.rename({date_col: "valid_time"})

        if "valid_time" in pl_df.columns and pl_df["valid_time"].dtype != pl.Date:
            pl_df = pl_df.with_columns(pl.col("valid_time").cast(pl.Date, strict=False))

        pl_df = pl_df.with_columns(
            [
                pl.lit(ticker).cast(pl.Utf8).alias("ticker"),
                pl.lit(tx_time).cast(pl.Datetime("us", "UTC")).alias("transaction_time"),
            ]
        )
        return pl_df

    except Exception as exc:
        logger.warning(f"[{ticker}] earnings_dates error: {exc}")
        return None


def download_fundamentals(
    tickers: Optional[list[str]] = None,
    data_dir: Optional[Path] = None,
) -> None:
    """Download and save balance sheet, income statement, cash flow, and earnings.

    Args:
        tickers: Override list; defaults to config/universe.yaml.
        data_dir: Override output root; defaults to config data_dir.
    """
    cfg = _load_config()
    settings = _config_settings(cfg)
    tickers = tickers or _config_tickers(cfg)
    data_dir = data_dir or (_ROOT / settings["data_dir"])
    tx_time = _transaction_time(cfg)

    logger.info(f"Downloading fundamentals for {len(tickers)} tickers")

    for ticker in tickers:
        out_dir = data_dir / "fundamentals" / ticker
        out_dir.mkdir(parents=True, exist_ok=True)

        try:
            t = yf.Ticker(ticker)
        except Exception as exc:
            logger.warning(f"[{ticker}] Ticker init error: {exc}")
            continue

        statements = {
            "balance_sheet": t.balance_sheet,
            "income_statement": t.financials,
            "cash_flow": t.cashflow,
        }

        for name, raw in statements.items():
            pl_df = _wide_to_long(ticker, raw, name, tx_time)
            if pl_df is not None and not pl_df.is_empty():
                out_path = out_dir / f"{name}.parquet"
                pl_df.write_parquet(out_path)
                logger.info(f"[{ticker}] {name}: {len(pl_df)} rows → {out_path}")
            else:
                logger.warning(f"[{ticker}] {name}: no data")

        # Earnings calendar
        earnings_df = _earnings_to_df(ticker, t, tx_time)
        if earnings_df is not None and not earnings_df.is_empty():
            out_path = out_dir / "earnings_calendar.parquet"
            earnings_df.write_parquet(out_path)
            logger.info(f"[{ticker}] earnings_calendar: {len(earnings_df)} rows → {out_path}")
        else:
            logger.warning(f"[{ticker}] earnings_calendar: no data")
