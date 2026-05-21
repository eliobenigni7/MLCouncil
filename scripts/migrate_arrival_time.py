"""Backfill ``arrival_time`` on existing FeatureStore symbols.

Retro-estimates feed availability from raw ingest metadata:
- News: RSS ``published`` timestamp in ``data/raw/news/*.parquet``
- Macro: FRED observation ``valid_time`` + next-calendar-day 13:30 UTC release proxy
- Default: existing ``transaction_time`` (preserves bi-temporal PIT behavior)
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path

import polars as pl

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from data.store.arctic_store import FeatureStore  # noqa: E402

_FRED_RELEASE_UTC = time(13, 30)


def _fred_arrival_from_valid_time(valid_time: date) -> datetime:
    """Proxy FRED morning release as next calendar day 13:30 UTC."""
    release_day = valid_time + timedelta(days=1)
    return datetime.combine(release_day, _FRED_RELEASE_UTC, tzinfo=timezone.utc)


def build_news_arrival_index(raw_dir: Path) -> pl.DataFrame:
    """``ticker``, ``valid_time`` -> max ``published`` as ``arrival_time``."""
    news_dir = raw_dir / "news"
    if not news_dir.exists():
        return pl.DataFrame(
            schema={
                "ticker": pl.Utf8,
                "valid_time": pl.Date,
                "arrival_time": pl.Datetime("us", "UTC"),
            }
        )

    frames: list[pl.DataFrame] = []
    for path in sorted(news_dir.glob("*.parquet")):
        df = pl.read_parquet(path)
        if df.is_empty() or "published" not in df.columns:
            continue
        cols = ["ticker", "valid_time", "published"]
        subset = df.select(cols).with_columns(
            pl.col("published").cast(pl.Datetime("us", "UTC")).alias("arrival_time")
        )
        frames.append(subset.select("ticker", "valid_time", "arrival_time"))

    if not frames:
        return pl.DataFrame(
            schema={
                "ticker": pl.Utf8,
                "valid_time": pl.Date,
                "arrival_time": pl.Datetime("us", "UTC"),
            }
        )

    return (
        pl.concat(frames, how="diagonal_relaxed")
        .group_by("ticker", "valid_time")
        .agg(pl.col("arrival_time").max().alias("arrival_time"))
    )


def build_fred_arrival_index(raw_dir: Path) -> pl.DataFrame:
    """Union macro series ``valid_time`` with FRED release proxy."""
    macro_dir = raw_dir / "macro"
    if not macro_dir.exists():
        return pl.DataFrame(
            schema={
                "valid_time": pl.Date,
                "arrival_time": pl.Datetime("us", "UTC"),
            }
        )

    valid_times: list[date] = []
    for path in macro_dir.glob("*.parquet"):
        df = pl.read_parquet(path)
        if df.is_empty() or "valid_time" not in df.columns:
            continue
        valid_times.extend(df["valid_time"].to_list())

    if not valid_times:
        return pl.DataFrame(
            schema={
                "valid_time": pl.Date,
                "arrival_time": pl.Datetime("us", "UTC"),
            }
        )

    unique_dates = sorted(set(valid_times))
    return pl.DataFrame(
        {
            "valid_time": unique_dates,
            "arrival_time": [_fred_arrival_from_valid_time(d) for d in unique_dates],
        }
    )


def estimate_arrival_times(
    df: pl.DataFrame,
    ticker: str,
    news_index: pl.DataFrame,
    fred_index: pl.DataFrame,
) -> pl.DataFrame:
    """Return ``df`` with an ``arrival_time`` column."""
    if "arrival_time" in df.columns:
        return df

    out = df.with_columns(pl.col("transaction_time").alias("arrival_time"))

    if not news_index.is_empty() and "valid_time" in out.columns:
        ticker_news = news_index.filter(pl.col("ticker") == ticker)
        if not ticker_news.is_empty():
            out = out.join(
                ticker_news.select("valid_time", "arrival_time"),
                on="valid_time",
                how="left",
                suffix="_news",
            )
            out = out.with_columns(
                pl.coalesce(pl.col("arrival_time_news"), pl.col("arrival_time")).alias(
                    "arrival_time"
                )
            ).drop("arrival_time_news")

    if not fred_index.is_empty() and "valid_time" in out.columns:
        out = out.join(
            fred_index.select("valid_time", pl.col("arrival_time").alias("arrival_time_fred")),
            on="valid_time",
            how="left",
        )
        out = out.with_columns(
            pl.coalesce(pl.col("arrival_time_fred"), pl.col("arrival_time")).alias(
                "arrival_time"
            )
        ).drop("arrival_time_fred")

    return out.select(
        [c for c in df.columns if c != "arrival_time"] + ["arrival_time"]
    )


def migrate_symbol(
    store: FeatureStore,
    ticker: str,
    news_index: pl.DataFrame,
    fred_index: pl.DataFrame,
    *,
    dry_run: bool,
) -> dict:
    df = store.read(ticker)
    if df.is_empty():
        return {"ticker": ticker, "rows": 0, "updated": 0, "skipped": True}

    if "arrival_time" in df.columns:
        return {"ticker": ticker, "rows": df.height, "updated": 0, "skipped": True}

    migrated = estimate_arrival_times(df, ticker, news_index, fred_index)
    changed = 0
    if "transaction_time" in migrated.columns and "arrival_time" in migrated.columns:
        changed = int(
            (migrated["arrival_time"] != migrated["transaction_time"]).sum()
        )

    if not dry_run:
        store.write(ticker, migrated)

    return {
        "ticker": ticker,
        "rows": migrated.height,
        "updated": changed,
        "skipped": False,
    }


def run_migration(
    *,
    uri: str,
    library: str,
    raw_dir: Path,
    dry_run: bool,
) -> list[dict]:
    store = FeatureStore(uri=uri, library=library)
    news_index = build_news_arrival_index(raw_dir)
    fred_index = build_fred_arrival_index(raw_dir)

    results: list[dict] = []
    for ticker in sorted(store.list_symbols()):
        results.append(
            migrate_symbol(
                store,
                ticker,
                news_index,
                fred_index,
                dry_run=dry_run,
            )
        )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report planned backfill without writing to the store",
    )
    parser.add_argument(
        "--uri",
        default=os.getenv("ARCTICDB_URI", "lmdb://data/arctic/"),
        help="FeatureStore URI (default: ARCTICDB_URI or lmdb://data/arctic/)",
    )
    parser.add_argument(
        "--library",
        default="mlcouncil",
        help="ArcticDB library name (default: mlcouncil)",
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=_ROOT / "data" / "raw",
        help="Root directory for raw ingest parquet (default: data/raw)",
    )
    args = parser.parse_args()

    mode = "DRY-RUN" if args.dry_run else "APPLY"
    print(f"[migrate_arrival_time] mode={mode} uri={args.uri} raw_dir={args.raw_dir}")

    results = run_migration(
        uri=args.uri,
        library=args.library,
        raw_dir=args.raw_dir,
        dry_run=args.dry_run,
    )

    if not results:
        print("No symbols found in feature store.")
        return

    total_rows = sum(r["rows"] for r in results)
    total_changed = sum(r["updated"] for r in results)
    skipped = sum(1 for r in results if r["skipped"])

    print(f"Symbols: {len(results)} (skipped already-migrated: {skipped})")
    print(f"Rows scanned: {total_rows}")
    print(f"Rows with arrival_time != transaction_time: {total_changed}")

    for row in results:
        if row["rows"] == 0:
            continue
        status = "skip" if row["skipped"] else "plan" if args.dry_run else "write"
        print(
            f"  {row['ticker']}: rows={row['rows']} "
            f"arrival_override={row['updated']} [{status}]"
        )


if __name__ == "__main__":
    main()
