"""News ingestion from Yahoo Finance RSS feeds with deduplication."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Optional
from zoneinfo import ZoneInfo

import feedparser
import polars as pl
import yaml
from loguru import logger

_ROOT = Path(__file__).parents[2]

_RSS_URL = "https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US"

_NEWS_SCHEMA = {
    "ticker": pl.Utf8,
    "valid_time": pl.Date,
    "transaction_time": pl.Datetime("us", "UTC"),
    "title": pl.Utf8,
    "published": pl.Datetime("us", "UTC"),
    "source": pl.Utf8,
    "url": pl.Utf8,
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


def _parse_feed(ticker: str, date: str, tx_time: datetime) -> list[dict]:
    """Fetch and parse the Yahoo Finance RSS feed for one ticker."""
    url = _RSS_URL.format(ticker=ticker)
    try:
        feed = feedparser.parse(url)
    except Exception as exc:
        logger.warning(f"[{ticker}] RSS fetch error: {exc}")
        return []

    if feed.bozo and not feed.entries:
        logger.warning(f"[{ticker}] malformed or empty RSS feed (bozo={feed.bozo})")
        return []

    records: list[dict] = []
    for entry in feed.entries:
        try:
            # published_parsed is a time.struct_time in UTC
            if hasattr(entry, "published_parsed") and entry.published_parsed:
                published = datetime(*entry.published_parsed[:6], tzinfo=ZoneInfo("UTC"))
            else:
                published = tx_time

            records.append(
                {
                    "ticker": ticker,
                    "valid_time": date,
                    "transaction_time": tx_time,
                    "title": getattr(entry, "title", ""),
                    "published": published,
                    "source": getattr(entry, "source", {}).get("title", "")
                    if isinstance(getattr(entry, "source", None), dict)
                    else getattr(entry, "source", ""),
                    "url": getattr(entry, "link", ""),
                }
            )
        except Exception as exc:
            logger.warning(f"[{ticker}] error parsing entry: {exc}")
            continue

    return records


def download_news(
    tickers: Optional[list[str]] = None,
    date: Optional[str] = None,
    data_dir: Optional[Path] = None,
) -> pl.DataFrame:
    """Download news headlines for all tickers and save as daily Parquet.

    Args:
        tickers: Override list; defaults to config/universe.yaml.
        date: ISO date string (e.g. "2024-01-15"). Defaults to today.
        data_dir: Override output root; defaults to config data_dir.

    Returns:
        Polars DataFrame with deduplicated news records.
    """
    cfg = _load_config()
    settings = _config_settings(cfg)
    tickers = tickers or _config_tickers(cfg)
    date = date or datetime.today().strftime("%Y-%m-%d")
    data_dir = data_dir or (_ROOT / settings["data_dir"])
    tx_time = _transaction_time(cfg)

    logger.info(f"Downloading news for {date}: {len(tickers)} tickers")

    all_records: list[dict] = []
    for ticker in tickers:
        records = _parse_feed(ticker, date, tx_time)
        logger.info(f"[{ticker}] {len(records)} headlines")
        all_records.extend(records)

    if not all_records:
        logger.warning("No news records collected")
        return pl.DataFrame(schema=_NEWS_SCHEMA)

    df = pl.DataFrame(all_records)

    # Cast to canonical schema
    df = df.with_columns(
        [
            pl.col("ticker").cast(pl.Utf8),
            pl.col("valid_time").cast(pl.Date),
            pl.col("transaction_time").cast(pl.Datetime("us", "UTC")),
            pl.col("title").cast(pl.Utf8),
            pl.col("published").cast(pl.Datetime("us", "UTC")),
            pl.col("source").cast(pl.Utf8),
            pl.col("url").cast(pl.Utf8),
        ]
    )

    # Deduplicate by URL
    before = len(df)
    df = df.unique(subset=["url"], keep="first")
    dupes = before - len(df)
    if dupes:
        logger.info(f"Removed {dupes} duplicate URLs")

    # Save
    out_dir = data_dir / "news"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{date}.parquet"
    df.write_parquet(out_path)
    logger.info(f"News saved: {len(df)} records → {out_path}")

    return df
