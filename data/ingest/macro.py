"""Macro data ingestion from FRED public CSV endpoints (no API key required)."""

from __future__ import annotations

import io
from datetime import datetime
from pathlib import Path
from typing import Optional
from zoneinfo import ZoneInfo

import polars as pl
import requests
import yaml
from loguru import logger

_ROOT = Path(__file__).parents[2]

# FRED public CSV endpoint — no API key needed
_FRED_CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"

_MACRO_SCHEMA_BASE = {
    "series_id": pl.Utf8,
    "valid_time": pl.Date,
    "transaction_time": pl.Datetime("us", "UTC"),
    "value": pl.Float64,
}


def _load_config() -> dict:
    with open(_ROOT / "config" / "universe.yaml") as f:
        return yaml.safe_load(f)


def _config_settings(cfg: dict) -> dict:
    return cfg.get("settings") or cfg.get("universe", {}).get("settings", {})


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


def _fetch_fred_series(series_id: str) -> Optional[pl.DataFrame]:
    """Download a FRED series and return a Polars DataFrame with DATE + value columns."""
    url = _FRED_CSV_URL.format(series_id=series_id)
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
    except requests.RequestException as exc:
        logger.warning(f"[{series_id}] FRED download error: {exc}")
        return None

    try:
        df = pl.read_csv(
            io.StringIO(resp.text),
            null_values=[".", ""],
            try_parse_dates=True,
        )
    except Exception as exc:
        logger.warning(f"[{series_id}] CSV parse error: {exc}")
        return None

    if df.is_empty():
        logger.warning(f"[{series_id}] empty response")
        return None

    # FRED CSV headers: DATE, {series_id}
    col_names = df.columns
    date_col = col_names[0]
    val_col = col_names[1] if len(col_names) > 1 else series_id

    df = df.rename({date_col: "valid_time", val_col: "value"})

    if df["valid_time"].dtype != pl.Date:
        df = df.with_columns(pl.col("valid_time").cast(pl.Date, strict=False))

    df = df.with_columns(pl.col("value").cast(pl.Float64, strict=False))
    df = df.filter(pl.col("value").is_not_null())

    return df


def download_macro(
    start: str = "2010-01-01",
    end: Optional[str] = None,
    data_dir: Optional[Path] = None,
) -> None:
    """Download VIX, 10Y/2Y Treasury yields, S&P 500 from FRED and save Parquet.

    Derived series:
      - yield_spread = DGS10 - DGS2
      - S&P 500 rolling returns (21d, 63d, 252d)

    Args:
        start: ISO date string (inclusive filter after download).
        end: ISO date string (inclusive filter). Defaults to today.
        data_dir: Override output root; defaults to config data_dir.
    """
    cfg = _load_config()
    settings = _config_settings(cfg)
    data_dir = data_dir or (_ROOT / settings["data_dir"])
    tx_time = _transaction_time(cfg)
    series_map: dict[str, str] = cfg["macro"]["fred_series"]
    rolling_windows: list[int] = cfg["macro"]["sp500_rolling_windows"]

    end = end or datetime.today().strftime("%Y-%m-%d")
    start_date = pl.lit(start).str.to_date()
    end_date = pl.lit(end).str.to_date()

    out_dir = data_dir / "macro"
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Downloading macro data {start}..{end}")

    # ---------- VIX ----------
    vix_df = _fetch_fred_series(series_map["vix"])
    if vix_df is not None:
        vix_df = _filter_dates(vix_df, start, end)
        vix_df = _add_meta(vix_df, series_map["vix"], tx_time)
        out = out_dir / "vix.parquet"
        vix_df.write_parquet(out)
        logger.info(f"VIX: {len(vix_df)} rows → {out}")

    # ---------- Treasuries + spread ----------
    t10_df = _fetch_fred_series(series_map["treasury_10y"])
    t2_df = _fetch_fred_series(series_map["treasury_2y"])

    if t10_df is not None and t2_df is not None:
        # Join and compute spread
        joined = (
            t10_df.rename({"value": "dgs10"})
            .join(t2_df.rename({"value": "dgs2"}), on="valid_time", how="inner")
            .with_columns((pl.col("dgs10") - pl.col("dgs2")).alias("yield_spread"))
        )
        joined = _filter_dates(joined, start, end)
        joined = joined.with_columns(
            pl.lit(tx_time).cast(pl.Datetime("us", "UTC")).alias("transaction_time")
        )
        out = out_dir / "treasuries.parquet"
        joined.write_parquet(out)
        logger.info(f"Treasuries: {len(joined)} rows → {out}")

    # ---------- S&P 500 + rolling returns ----------
    sp_df = _fetch_fred_series(series_map["sp500"])
    if sp_df is not None:
        sp_df = _filter_dates(sp_df, start, end)
        sp_df = sp_df.rename({"value": "sp500_price"})

        # Daily log returns
        sp_df = sp_df.with_columns(
            (pl.col("sp500_price") / pl.col("sp500_price").shift(1) - 1.0).alias("daily_return")
        )

        # Rolling cumulative returns via log/exp:
        # product(1 + r_i) = exp(sum(log(1 + r_i)))
        for w in rolling_windows:
            col = f"return_{w}d"
            sp_df = sp_df.with_columns(
                (
                    pl.col("daily_return")
                    .log1p()
                    .rolling_sum(window_size=w, min_samples=w)
                    .exp()
                    - 1.0
                ).alias(col)
            )

        sp_df = sp_df.with_columns(
            pl.lit(tx_time).cast(pl.Datetime("us", "UTC")).alias("transaction_time")
        )
        out = out_dir / "sp500.parquet"
        sp_df.write_parquet(out)
        logger.info(f"S&P 500: {len(sp_df)} rows → {out}")


def _filter_dates(df: pl.DataFrame, start: str, end: str) -> pl.DataFrame:
    return df.filter(
        (pl.col("valid_time") >= pl.lit(start).str.to_date())
        & (pl.col("valid_time") <= pl.lit(end).str.to_date())
    )


def _add_meta(df: pl.DataFrame, series_id: str, tx_time: datetime) -> pl.DataFrame:
    return df.with_columns(
        [
            pl.lit(series_id).cast(pl.Utf8).alias("series_id"),
            pl.lit(tx_time).cast(pl.Datetime("us", "UTC")).alias("transaction_time"),
        ]
    )
