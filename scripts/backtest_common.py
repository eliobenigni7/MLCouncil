"""Shared helpers for MLCouncil backtest runners.

These utilities are intentionally lightweight and stable so that the
canonical one-year runner and the experimental challenger scripts do not
copy-paste the same data-loading logic.
"""
from __future__ import annotations

from pathlib import Path
from collections.abc import Sequence

import pandas as pd
import polars as pl
import yaml

ROOT = Path(__file__).resolve().parents[1]
EQUITY_UNIVERSE = {"AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA"}


def compute_proxy_sentiment(
    ohlcv: pl.DataFrame,
    as_of_date,
    tickers: Sequence[str],
    short_window: int = 5,
    long_window: int = 20,
) -> pd.Series:
    """Proxy sentiment from short-term vs medium-term momentum gap."""
    try:
        prices = (
            ohlcv.filter(pl.col("valid_time") <= pl.lit(as_of_date))
            .sort(["ticker", "valid_time"])
            .select(["ticker", "valid_time", "adj_close"])
        )
        if prices.is_empty():
            return pd.Series(0.0, index=tickers)

        price_pd = prices.to_pandas().pivot(index="valid_time", columns="ticker", values="adj_close").sort_index()
        price_pd.index = pd.to_datetime(price_pd.index)

        if len(price_pd) < long_window:
            return pd.Series(0.0, index=tickers)

        mom_short = price_pd.pct_change(short_window).iloc[-1]
        mom_long = price_pd.pct_change(long_window).iloc[-1]
        gap = mom_short - mom_long

        result = gap.reindex(tickers).fillna(0.0)
        std = result.std()
        if std > 1e-9:
            result = (result - result.mean()) / std
        return result.fillna(0.0)
    except Exception:
        return pd.Series(0.0, index=tickers)


def load_universe() -> set[str]:
    """Load the canonical research universe from config/universe.yaml."""
    with open(ROOT / "config" / "universe.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    tickers: set[str] = set()
    for bucket in ("large_cap", "mid_cap"):
        tickers.update(str(t).upper() for t in cfg.get("universe", {}).get(bucket, []) or [])
    return tickers or set(EQUITY_UNIVERSE)


def load_ohlcv(allowed: set[str]) -> pl.DataFrame:
    """Load OHLCV parquet files from data/raw/ohlcv for the allowed tickers."""
    frames: list[pl.DataFrame] = []
    raw_dir = ROOT / "data" / "raw" / "ohlcv"
    if not raw_dir.exists():
        raise FileNotFoundError(f"OHLCV directory not found: {raw_dir}")

    for ticker_dir in sorted(raw_dir.iterdir()):
        if not ticker_dir.is_dir():
            continue
        ticker = ticker_dir.name.upper()
        if ticker not in allowed:
            continue

        ticker_frames: list[pl.DataFrame] = []
        for pq in sorted(ticker_dir.glob("*.parquet")):
            try:
                df = pl.read_parquet(pq)
            except Exception:
                continue
            if "symbol" in df.columns:
                df = df.drop("symbol")
            if "ticker" not in df.columns:
                df = df.with_columns(pl.lit(ticker).alias("ticker"))
            if "transaction_time" in df.columns:
                df = df.drop("transaction_time")
            if "valid_time" in df.columns:
                vt = df["valid_time"]
                if vt.dtype == pl.Datetime:
                    df = df.with_columns(vt.dt.replace_time_zone("UTC").cast(pl.Date))
                elif vt.dtype != pl.Date:
                    df = df.with_columns(vt.cast(pl.Date))
            keep = [c for c in ["ticker", "valid_time", "open", "high", "low", "close", "adj_close", "volume"] if c in df.columns]
            if keep:
                ticker_frames.append(df.select(keep))

        if ticker_frames:
            frames.append(
                pl.concat(ticker_frames, how="vertical_relaxed")
                .unique(subset=["ticker", "valid_time"], keep="last")
                .sort(["ticker", "valid_time"])
            )

    if not frames:
        return pl.DataFrame()

    return pl.concat(frames, how="vertical_relaxed").unique(["ticker", "valid_time"]).sort(["ticker", "valid_time"])


def macro_path(name: str) -> str | None:
    """Return a macro parquet path if present."""
    p = ROOT / "data" / "raw" / "macro" / f"{name}.parquet"
    return str(p) if p.exists() else None
