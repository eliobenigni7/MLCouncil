#!/usr/bin/env python3
"""
Backfill crypto historical data from Alpaca.
Downloads daily OHLCV bars and saves to data/raw/ohlcv/{TICKER}/YYYY-MM-DD.parquet
"""
import os
import sys
from pathlib import Path
from datetime import datetime, timedelta, timezone
import alpaca_trade_api
import pandas as pd
import pyarrow.parquet as pq
import pyarrow as pa

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

ALPACA_CRYPTO_URL = os.getenv("ALPACA_CRYPTO_URL", "https://paper-api.alpaca.markets")
API_KEY = os.getenv("ALPACA_API_KEY")
SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")

# (internal_ticker, alpaca_ticker)
TICKERS = [("BTCUSD", "BTC/USD"), ("ETHUSD", "ETH/USD")]
DAYS_BACK = 90  # Alpaca free tier limit

def get_raw_dir(internal_ticker: str) -> Path:
    root = Path(__file__).parent.parent / "data" / "raw" / "ohlcv" / internal_ticker
    root.mkdir(parents=True, exist_ok=True)
    return root

def backfill_ticker(internal_ticker: str, alpaca_ticker: str, api: alpaca_trade_api.REST):
    end = datetime.now(timezone.utc).replace(microsecond=0)
    start = end - timedelta(days=DAYS_BACK)

    print(f"[{internal_ticker}] Downloading {DAYS_BACK} days of daily bars...")

    try:
        bars = api.get_crypto_bars(alpaca_ticker, "1Day", start.isoformat(), end.isoformat()).df
    except Exception as e:
        print(f"[{internal_ticker}] ERROR fetching bars: {e}")
        return

    if bars.empty:
        print(f"[{internal_ticker}] No data returned")
        return

    # Crypto bars come multi-index (symbol, timestamp) - reset
    if isinstance(bars.index, pd.MultiIndex):
        bars = bars.reset_index()
        if "symbol" in bars.columns:
            bars = bars.drop(columns=["symbol"])

    # Parse timestamp
    if "timestamp" in bars.columns:
        bars["timestamp"] = pd.to_datetime(bars["timestamp"])
        bars = bars.set_index("timestamp").sort_index()

    # Also handle DatetimeIndex
    if isinstance(bars.index, pd.DatetimeIndex):
        bars.index = bars.index.tz_localize(None) if bars.index.tz else bars.index

    saved = 0
    for date, row in bars.iterrows():
        date_str = date.strftime("%Y-%m-%d")
        out_path = get_raw_dir(internal_ticker) / f"{date_str}.parquet"

        if out_path.exists():
            print(f"  {date_str} already exists, skipping")
            continue

        df = pd.DataFrame([row]).reset_index(names=["timestamp"])
        table = pa.Table.from_pandas(df)
        pq.write_table(table, str(out_path))
        saved += 1

    print(f"[{internal_ticker}] Saved {saved} new files, {len(bars)} total bars available")

def main():
    if not API_KEY or not SECRET_KEY:
        print("ERROR: ALPACA_API_KEY / ALPACA_SECRET_KEY not set in .env")
        sys.exit(1)

    api = alpaca_trade_api.REST(API_KEY, SECRET_KEY, ALPACA_CRYPTO_URL, api_version="v2")

    for internal_ticker, alpaca_ticker in TICKERS:
        backfill_ticker(internal_ticker, alpaca_ticker, api)

    print("\nBackfill complete! Now regenerate features:")
    print("  cd /path/to/MLCouncil && python3 -m data.pipeline --date 2026-04-10")

if __name__ == "__main__":
    main()
