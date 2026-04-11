"""ArcticDB feature store with LMDB backend.

Point-in-time correctness
--------------------------
Each ``write()`` call appends a new ArcticDB version for the symbol.
The ``transaction_time`` column records *when* the data was written.
``read()`` filters rows to ``transaction_time <= as_of_transaction_time``
when supplied, so you can reconstruct what the store looked like at any
past point in time.

LMDB is used for local PoC work; no S3 configuration required.

ARM64 Compatibility
-------------------
On platforms where ArcticDB wheels are unavailable (ARM64),
automatically falls back to Parquet-based storage with same interface.
"""

from __future__ import annotations

from datetime import date, datetime, timezone
from typing import Optional

import polars as pl

# Try ArcticDB first, fall back to Parquet backend for ARM64 compatibility
_arcticdb_available = False
try:
    import arcticdb as adb
    _arcticdb_available = True
except ImportError:
    pass


class FeatureStore:
    """Versioned feature store backed by ArcticDB (LMDB) or Parquet fallback.

    Parameters
    ----------
    uri:
        ArcticDB connection string or Parquet directory.
        Default: ``"lmdb://data/arctic/"`` (relative to working directory).
    library:
        Name of the ArcticDB library or Parquet subdirectory.
    """

    _TICKER_PREFIX = "features/"

    def __init__(
        self,
        uri: str = "lmdb://data/arctic/",
        library: str = "mlcouncil",
    ) -> None:
        if _arcticdb_available:
            self._backend = "arcticdb"
            self._ac = adb.Arctic(uri)
            self._lib = self._ac.get_library(library, create_if_missing=True)
        else:
            # Parquet fallback for ARM64
            self._backend = "parquet"
            from pathlib import Path
            if uri.startswith("lmdb://"):
                uri = uri[7:]  # Strip lmdb:// prefix
            self._base_path = Path(uri) / library
            self._base_path.mkdir(parents=True, exist_ok=True)

    def _symbol_path(self, ticker: str):
        """Get parquet file path for ticker (parquet backend only)."""
        from pathlib import Path
        symbol = self._TICKER_PREFIX + ticker
        safe_symbol = symbol.replace("/", "_")
        return self._base_path / f"{safe_symbol}.parquet"

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def write(
        self,
        ticker: str,
        df: pl.DataFrame,
        metadata: dict | None = None,
    ) -> None:
        """Write or append features for a ticker.

        Parameters
        ----------
        ticker:
            Ticker symbol (e.g. ``"AAPL"``).
        df:
            Polars DataFrame with at least ``valid_time`` column.
        metadata:
            Optional dict stored alongside the version (ArcticDB only).
        """
        if "transaction_time" not in df.columns:
            tx = datetime.now(timezone.utc)
            df = df.with_columns(
                pl.lit(tx).cast(pl.Datetime("us", "UTC")).alias("transaction_time")
            )

        if self._backend == "arcticdb":
            symbol = self._TICKER_PREFIX + ticker
            pd_df = df.to_pandas()
            if "valid_time" in pd_df.columns:
                import pandas as pd
                pd_df["valid_time"] = pd.to_datetime(pd_df["valid_time"])
                pd_df = pd_df.set_index("valid_time")
            self._lib.write(symbol, pd_df, metadata=metadata or {})
        else:
            # Parquet backend
            path = self._symbol_path(ticker)
            if path.exists():
                existing = pl.read_parquet(path)
                df = pl.concat([existing, df], how="diagonal_relaxed")
            df.write_parquet(path)

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def read(
        self,
        ticker: str,
        start: date | None = None,
        end: date | None = None,
        as_of_transaction_time: datetime | None = None,
    ) -> pl.DataFrame:
        """Read features for a single ticker.

        Parameters
        ----------
        ticker:
            Ticker symbol.
        start, end:
            Inclusive valid_time range filter.
        as_of_transaction_time:
            Point-in-time filter. Only rows with
            ``transaction_time <= as_of_transaction_time`` are returned.

        Returns
        -------
        pl.DataFrame
            Features with valid_time as a column (not index).
        """
        if self._backend == "arcticdb":
            import pandas as pd
            symbol = self._TICKER_PREFIX + ticker
            if not self._lib.has_symbol(symbol):
                return pl.DataFrame()

            date_range = None
            if start is not None or end is not None:
                ts_start = pd.Timestamp(start) if start is not None else None
                ts_end = pd.Timestamp(end) if end is not None else None
                date_range = (ts_start, ts_end)

            vitem = self._lib.read(symbol, date_range=date_range)
            pd_df = vitem.data.reset_index()
            df = pl.from_pandas(pd_df)

            if as_of_transaction_time is not None and "transaction_time" in df.columns:
                cutoff = pl.lit(as_of_transaction_time).cast(pl.Datetime("us", "UTC"))
                df = df.filter(pl.col("transaction_time") <= cutoff)

            return df
        else:
            # Parquet backend
            path = self._symbol_path(ticker)
            if not path.exists():
                return pl.DataFrame()

            df = pl.read_parquet(path)

            if start is not None or end is not None:
                if "valid_time" in df.columns:
                    if start is not None:
                        df = df.filter(pl.col("valid_time") >= pl.date(start.year, start.month, start.day))
                    if end is not None:
                        df = df.filter(pl.col("valid_time") <= pl.date(end.year, end.month, end.day))

            if as_of_transaction_time is not None and "transaction_time" in df.columns:
                cutoff = pl.lit(as_of_transaction_time).cast(pl.Datetime("us", "UTC"))
                df = df.filter(pl.col("transaction_time") <= cutoff)

            return df

    # ------------------------------------------------------------------
    # Read universe
    # ------------------------------------------------------------------

    def read_universe(
        self,
        tickers: list[str],
        as_of_date: date,
        as_of_transaction_time: datetime | None = None,
    ) -> pl.DataFrame:
        """Read features for all tickers on a specific valid_time date.

        Parameters
        ----------
        tickers:
            List of ticker symbols.
        as_of_date:
            The valid_time date to retrieve.
        as_of_transaction_time:
            Point-in-time filter applied to each ticker.

        Returns
        -------
        pl.DataFrame
            Combined DataFrame for the full universe on ``as_of_date``.
        """
        frames: list[pl.DataFrame] = []
        for ticker in tickers:
            df = self.read(
                ticker,
                start=as_of_date,
                end=as_of_date,
                as_of_transaction_time=as_of_transaction_time,
            )
            if not df.is_empty():
                frames.append(df)

        if not frames:
            return pl.DataFrame()

        return pl.concat(frames, how="diagonal_relaxed")

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def list_symbols(self) -> list[str]:
        """Return all stored ticker symbols (without prefix)."""
        if self._backend == "arcticdb":
            prefix = self._TICKER_PREFIX
            return [
                s[len(prefix):]
                for s in self._lib.list_symbols()
                if s.startswith(prefix)
            ]
        else:
            prefix = self._TICKER_PREFIX.replace("/", "_")
            symbols = []
            for f in self._base_path.glob("*.parquet"):
                name = f.stem
                if name.startswith(prefix):
                    symbols.append(name[len(prefix):])
            return symbols

    def list_versions(self, ticker: str) -> list[dict]:
        """Return version history for a ticker."""
        if self._backend == "arcticdb":
            symbol = self._TICKER_PREFIX + ticker
            if not self._lib.has_symbol(symbol):
                return []
            return list(self._lib.list_versions(symbol))
        else:
            path = self._symbol_path(ticker)
            if not path.exists():
                return []
            mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
            return [{"version": 1, "timestamp": mtime.isoformat()}]


# Export for convenience
__all__ = ["FeatureStore"]