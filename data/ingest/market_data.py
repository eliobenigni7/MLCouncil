"""OHLCV market data ingestion via yfinance with bi-temporal Parquet storage."""

from __future__ import annotations

import io
from datetime import datetime, time
from pathlib import Path
from typing import Optional
from zoneinfo import ZoneInfo

import polars as pl
import yfinance as yf
import yaml
from loguru import logger

_ROOT = Path(__file__).parents[2]

_OHLCV_SCHEMA = {
    "ticker": pl.Utf8,
    "valid_time": pl.Date,
    "transaction_time": pl.Datetime("us", "UTC"),
    "open": pl.Float64,
    "high": pl.Float64,
    "low": pl.Float64,
    "close": pl.Float64,
    "adj_close": pl.Float64,
    "volume": pl.Int64,
}


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


def _config_crypto_tickers(cfg: dict) -> list[str]:
    """Return list of crypto tickers from universe.yaml crypto_universe section."""
    crypto_cfg = cfg.get("crypto_universe", {})
    tickers: list[str] = []
    for bucket_name, bucket_values in crypto_cfg.items():
        if not isinstance(bucket_values, list):
            continue
        tickers.extend(bucket_values)
    return tickers


# Map internal ticker (BTCUSD) -> yfinance ticker (BTC-USD)
def _to_yfinance_ticker(ticker: str) -> str:
    if ticker.upper() in ("BTCUSD", "BTC-USD", "BTC/USD"):
        return "BTC-USD"
    if ticker.upper() in ("ETHUSD", "ETH-USD", "ETH/USD"):
        return "ETH-USD"
    return ticker


def _transaction_time(cfg: dict) -> datetime:
    """Return today at the configured post-market time in UTC."""
    settings = _config_settings(cfg)
    tz = ZoneInfo(settings["transaction_timezone"])
    local_dt = datetime.now(tz).replace(
        hour=settings["transaction_time_hour"],
        minute=settings["transaction_time_minute"],
        second=0,
        microsecond=0,
    )
    return local_dt.astimezone(ZoneInfo("UTC"))


def _download_ticker(
    ticker: str,
    start: str,
    end: str,
    tx_time: datetime,
    forward_fill_limit: int,
) -> Optional[pl.DataFrame]:
    """Download OHLCV for a single ticker and return a Polars DataFrame."""
    # Use yfinance-compatible ticker (e.g. BTC-USD instead of BTCUSD)
    yf_ticker = _to_yfinance_ticker(ticker)
    try:
        raw = yf.download(
            yf_ticker,
            start=start,
            end=end,
            auto_adjust=False,
            progress=False,
            actions=False,
        )
    except Exception as exc:
        logger.warning(f"[{ticker}] download error: {exc}")
        return None

    if raw is None or raw.empty:
        logger.warning(f"[{ticker}] no data returned for {start}..{end}")
        return None

    # Flatten MultiIndex columns that yfinance can produce
    if isinstance(raw.columns, __import__("pandas").MultiIndex):
        raw.columns = [col[0] for col in raw.columns]

    raw = raw.rename(columns={"Adj Close": "adj_close"})
    raw.columns = [c.lower() for c in raw.columns]

    # Reset index so Date becomes a column
    raw = raw.reset_index()
    raw = raw.rename(columns={"date": "valid_time", "Date": "valid_time"})

    df = pl.from_pandas(raw)

    # Normalise column name casing (yfinance sometimes varies)
    col_map = {c: c.lower() for c in df.columns}
    col_map["adj close"] = "adj_close"
    df = df.rename({k: v for k, v in col_map.items() if k in df.columns})

    # Ensure valid_time is pl.Date
    if df["valid_time"].dtype != pl.Date:
        df = df.with_columns(pl.col("valid_time").cast(pl.Date))

    # Forward fill price columns (max 1 day) to handle minor yfinance gaps.
    # Limit is intentionally capped at 1 regardless of config: larger limits
    # propagate stale halt-day prices, creating fake returns = 0 and inflating
    # backtest IC by 10-30% on halted tickers.
    price_cols = ["open", "high", "low", "close", "adj_close"]
    df = df.with_columns(
        [pl.col(c).forward_fill(limit=1) for c in price_cols if c in df.columns]
    )

    # Drop rows where close is still null after forward fill.
    df = df.filter(pl.col("close").is_not_null())

    # Drop stale halt rows: same close as the previous row with zero volume.
    # These rows carry no real price discovery and produce artificial 0% returns
    # that corrupt momentum signals and IC measurements.
    if "volume" in df.columns:
        df = df.with_columns(
            pl.col("close").shift(1).alias("_prev_close")
        ).filter(
            ~(
                (pl.col("close") == pl.col("_prev_close"))
                & (pl.col("volume") == 0)
            )
        ).drop("_prev_close")

    if df.is_empty():
        logger.warning(f"[{ticker}] empty after null filtering")
        return None

    # Add bi-temporal and ticker columns
    df = df.with_columns(
        [
            pl.lit(ticker).alias("ticker"),
            pl.lit(tx_time).alias("transaction_time"),
        ]
    )

    # Cast to canonical schema
    df = df.with_columns(
        [
            pl.col("ticker").cast(pl.Utf8),
            pl.col("valid_time").cast(pl.Date),
            pl.col("transaction_time").cast(pl.Datetime("us", "UTC")),
            pl.col("open").cast(pl.Float64),
            pl.col("high").cast(pl.Float64),
            pl.col("low").cast(pl.Float64),
            pl.col("close").cast(pl.Float64),
            pl.col("adj_close").cast(pl.Float64),
            pl.col("volume").cast(pl.Int64),
        ]
    )

    return df.select(list(_OHLCV_SCHEMA.keys()))


def _save_by_year(df: pl.DataFrame, data_dir: Path) -> None:
    """Override writer to preserve prior yearly history across daily ingests."""
    ticker = df["ticker"][0]
    df = df.with_columns(pl.col("valid_time").dt.year().alias("_year"))
    for year, group in df.group_by("_year"):
        year_val = year[0] if isinstance(year, tuple) else year
        out_dir = data_dir / "ohlcv" / ticker
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{year_val}.parquet"
        to_store = group.drop("_year")
        if out_path.exists():
            existing = pl.read_parquet(out_path)
            to_store = (
                pl.concat([existing, to_store], how="vertical_relaxed")
                .sort(["ticker", "valid_time", "transaction_time"])
                .unique(["ticker", "valid_time"], keep="last")
            )
        to_store.write_parquet(out_path)
        logger.info(f"[{ticker}] saved {len(to_store)} rows -> {out_path}")


def download_universe(
    tickers: Optional[list[str]] = None,
    start: str = "2010-01-01",
    end: Optional[str] = None,
    data_dir: Optional[Path] = None,
) -> None:
    """Download full OHLCV history for the universe.

    Args:
        tickers: Override list; defaults to config/universe.yaml.
        start: ISO date string (inclusive).
        end: ISO date string (exclusive). Defaults to today.
        data_dir: Override output root; defaults to config data_dir.
    """
    cfg = _load_config()
    settings = _config_settings(cfg)
    tickers = tickers or _config_tickers(cfg)
    end = end or datetime.today().strftime("%Y-%m-%d")
    data_dir = data_dir or (_ROOT / settings["data_dir"])
    tx_time = _transaction_time(cfg)
    ff_limit = settings["forward_fill_max_days"]

    logger.info(f"Downloading universe: {len(tickers)} tickers {start}..{end}")
    for ticker in tickers:
        df = _download_ticker(ticker, start, end, tx_time, ff_limit)
        if df is not None:
            _save_by_year(df, data_dir)


def download_daily(
    tickers: Optional[list[str]] = None,
    date: Optional[str] = None,
    data_dir: Optional[Path] = None,
) -> pl.DataFrame:
    """Download OHLCV for a single trading day and save Parquet.

    Args:
        tickers: Override list; defaults to config/universe.yaml.
        date: ISO date string (e.g. "2024-01-15"). Defaults to today.
        data_dir: Override output root; defaults to config data_dir.

    Returns:
        Combined Polars DataFrame for all tickers (may be empty on holidays).
    """
    import pandas as pd

    cfg = _load_config()
    settings = _config_settings(cfg)
    tickers = tickers or _config_tickers(cfg)
    date = date or datetime.today().strftime("%Y-%m-%d")
    data_dir = data_dir or (_ROOT / settings["data_dir"])
    tx_time = _transaction_time(cfg)
    ff_limit = settings["forward_fill_max_days"]

    # yfinance end is exclusive — add one calendar day
    end = (pd.Timestamp(date) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

    logger.info(f"Downloading daily snapshot for {date}: {len(tickers)} tickers")
    frames: list[pl.DataFrame] = []
    for ticker in tickers:
        df = _download_ticker(ticker, date, end, tx_time, ff_limit)
        if df is not None:
            _save_by_year(df, data_dir)
            frames.append(df)

    if not frames:
        logger.warning(f"No data returned for {date} (market closed?)")
        return pl.DataFrame(schema=_OHLCV_SCHEMA)

    return pl.concat(frames)


def download_crypto_daily(
    date: Optional[str] = None,
    data_dir: Optional[Path] = None,
) -> pl.DataFrame:
    """Download OHLCV for crypto assets (BTC-USD, ETH-USD) for a single day.

    Downloads to data_dir/crypto/ rather than the equity ohlcv path,
    so crypto and equity data are stored separately.
    """
    import pandas as pd

    cfg = _load_config()
    settings = _config_settings(cfg)
    tickers = _config_crypto_tickers(cfg)

    if not tickers:
        logger.info("No crypto tickers configured in universe.yaml")
        return pl.DataFrame(schema=_OHLCV_SCHEMA)

    date = date or datetime.today().strftime("%Y-%m-%d")
    data_dir = data_dir or (_ROOT / settings.get("data_dir", "data/raw"))
    tx_time = _transaction_time(cfg)
    ff_limit = settings["forward_fill_max_days"]

    end = (pd.Timestamp(date) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

    logger.info(f"Downloading crypto daily for {date}: {tickers}")
    frames: list[pl.DataFrame] = []

    for ticker in tickers:
        df = _download_ticker(ticker, date, end, tx_time, ff_limit)
        if df is not None:
            # Save to crypto subdirectory
            ticker_df = df.with_columns(pl.col("ticker"))
            ticker_df = ticker_df.with_columns(
                pl.lit("crypto").alias("_asset_class")
            )
            out_dir = data_dir / "crypto" / ticker
            out_dir.mkdir(parents=True, exist_ok=True)
            year = pd.Timestamp(date).year
            out_path = out_dir / f"{year}.parquet"
            if out_path.exists():
                existing = pl.read_parquet(out_path)
                ticker_df = (
                    pl.concat([existing, ticker_df], how="vertical_relaxed")
                    .sort(["ticker", "valid_time", "transaction_time"])
                    .unique(["ticker", "valid_time"], keep="last")
                )
            ticker_df.drop(["_asset_class"]).write_parquet(out_path)
            logger.info(f"[{ticker}] saved {len(ticker_df)} rows -> {out_path}")
            frames.append(df)

    if not frames:
        logger.warning(f"No crypto data returned for {date}")
        return pl.DataFrame(schema=_OHLCV_SCHEMA)

    return pl.concat(frames)
