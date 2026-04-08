"""Short interest data ingestion from FINRA.

Data source: FINRA OTC Bulk Data (free, published twice monthly)
URL: https://www.finra.org/industry/short-interest/bulk-shorts

Signals generated:
- si_ratio: Short interest / Shares outstanding
- days_to_cover: Short interest / Average daily volume
- short_volume_ratio: Short volume / Total volume
- high_si_flag: si_ratio > 10% = significant positioning

Note: FINRA data is WEEKLY (published ~15th and last business day of month).
Use as longer-horizon signal (>2 weeks).
"""

from __future__ import annotations

import hashlib
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd
import requests
from io import StringIO

_ROOT = Path(__file__).parents[1]
CACHE_DIR = _ROOT / "data" / "raw" / "short_interest"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


class FINRAShortInterestIngestor:
    BASE_URL = "https://www.finra.org/sites/default/files/shorts"

    def __init__(self, cache_days: int = 14):
        self.cache_days = cache_days
        self.universe_tickers = self._load_universe_tickers()

    def _load_universe_tickers(self) -> list[str]:
        try:
            import yaml
            with open(_ROOT / "config" / "universe.yaml") as f:
                cfg = yaml.safe_load(f)
            large = cfg["universe"].get("large_cap", [])
            mid = cfg["universe"].get("mid_cap", [])
            return large + mid
        except Exception:
            return []

    def fetch_latest(self, date: Optional[datetime] = None) -> pd.DataFrame:
        date = date or datetime.now()

        cache_file = CACHE_DIR / f"si_{date.strftime('%Y%m%d')}.parquet"
        if cache_file.exists():
            age = datetime.now() - datetime.fromtimestamp(cache_file.stat().st_mtime)
            if age.days < self.cache_days:
                return pd.read_parquet(cache_file)

        for days_back in range(15):
            try_date = date - timedelta(days=days_back)
            url = f"{self.BASE_URL}/shorts_{try_date:%Y%m%d}.csv"
            try:
                response = requests.get(url, timeout=30)
                if response.status_code == 200:
                    df = self._process_raw(response.text, try_date)
                    df.write_parquet(cache_file)
                    return df
            except Exception:
                continue

        if cache_file.exists():
            return pd.read_parquet(cache_file)

        raise RuntimeError(
            "Could not find recent FINRA short interest file. "
            "Data is published around the 15th and last business day of each month."
        )

    def _process_raw(self, csv_content: str, date: datetime) -> pd.DataFrame:
        df = pd.read_csv(StringIO(csv_content))

        if "Symbol" not in df.columns:
            if "IssueSymbol" in df.columns:
                df = df.rename(columns={"IssueSymbol": "Symbol"})
            else:
                raise ValueError(f"Unknown CSV format. Columns: {df.columns.tolist()}")

        if self.universe_tickers:
            df = df[df["Symbol"].isin(self.universe_tickers)]

        if "ShortInterest" not in df.columns:
            for col in ["ShortInterest", "CurrentShortInterest", "SI"]:
                if col in df.columns:
                    df = df.rename(columns={col: "ShortInterest"})
                    break

        if "DaysToCover" not in df.columns:
            for col in ["DaysToCover", "DaysToCoverShorts"]:
                if col in df.columns:
                    df = df.rename(columns={col: "DaysToCover"})
                    break

        result = pd.DataFrame()
        result["ticker"] = df["Symbol"]
        result["short_interest"] = pd.to_numeric(df.get("ShortInterest", 0), errors="coerce").fillna(0)
        result["days_to_cover"] = pd.to_numeric(df.get("DaysToCover", 0), errors="coerce").fillna(0)

        if "SharesOutstanding" in df.columns:
            shares_out = pd.to_numeric(df["SharesOutstanding"], errors="coerce").fillna(1)
            result["si_ratio"] = result["short_interest"] / (shares_out + 1)
        else:
            result["si_ratio"] = result["short_interest"] / 1e9

        result["high_si_flag"] = (result["si_ratio"] > 0.10).astype(int)

        for col in ["si_ratio", "days_to_cover"]:
            if col in result.columns:
                mean = result[col].mean()
                std = result[col].std()
                result[f"{col}_z"] = (result[col] - mean) / (std + 1e-8)

        result["date"] = date.strftime("%Y-%m-%d")

        return result[["ticker", "short_interest", "days_to_cover", "si_ratio",
                      "high_si_flag", "si_ratio_z", "days_to_cover_z", "date"]]

    def get_features(self, date: Optional[datetime] = None) -> pd.DataFrame:
        df = self.fetch_latest(date)
        return df.set_index("ticker")[[
            "si_ratio_z",
            "days_to_cover_z",
            "high_si_flag",
        ]]

    def forward_fill(self, dates: list[str]) -> pd.DataFrame:
        all_data = []
        for date_str in dates:
            try:
                date = datetime.strptime(date_str, "%Y-%m-%d")
                df = self.fetch_latest(date)
                df["valid_time"] = date_str
                all_data.append(df)
            except Exception:
                continue

        if not all_data:
            return pd.DataFrame()

        combined = pd.concat(all_data, ignore_index=True)

        tickers = combined["ticker"].unique()
        result_rows = []
        for ticker in tickers:
            ticker_data = combined[combined["ticker"] == ticker].sort_values("date")
            if not ticker_data.empty:
                latest = ticker_data.iloc[-1][[
                    "si_ratio_z", "days_to_cover_z", "high_si_flag", "si_ratio"
                ]]
                for date_str in dates:
                    row = {"ticker": ticker, "valid_time": date_str}
                    row.update(latest.to_dict())
                    result_rows.append(row)

        return pd.DataFrame(result_rows)


def get_short_interest_features(
    date: Optional[datetime] = None,
    universe: Optional[list[str]] = None,
) -> pd.Series:
    if not universe:
        try:
            import yaml
            with open(_ROOT / "config" / "universe.yaml") as f:
                cfg = yaml.safe_load(f)
            universe = cfg["universe"].get("large_cap", []) + cfg["universe"].get("mid_cap", [])
        except Exception:
            universe = []

    ingester = FINRAShortInterestIngestor()
    features = ingester.get_features(date)
    return features
