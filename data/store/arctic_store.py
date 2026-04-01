"""ArcticDB feature store with LMDB backend.

Point-in-time correctness
--------------------------
Each ``write()`` call appends a new ArcticDB version for the symbol.
The ``transaction_time`` column records *when* the data was written.
``read()`` filters rows to ``transaction_time <= as_of_transaction_time``
when supplied, so you can reconstruct what the store looked like at any
past point in time.

LMDB is used for local PoC work; no S3 configuration required.
"""

from __future__ import annotations

from datetime import date, datetime, timezone
from typing import Optional

import polars as pl

try:
    import arcticdb as adb
except ImportError as exc:
    raise ImportError("arcticdb is required: pip install arcticdb") from exc


class FeatureStore:
    """Versioned feature store backed by ArcticDB (LMDB).

    Parameters
    ----------
    uri:
        ArcticDB connection string.  Default: ``"lmdb://data/arctic/"``
        (relative to the working directory).
    library:
        Name of the ArcticDB library to use.
    """

    _TICKER_PREFIX = "features/"

    def __init__(
        self,
        uri: str = "lmdb://data/arctic/",
        library: str = "mlcouncil",
    ) -> None:
        self._ac = adb.Arctic(uri)
        self._lib = self._ac.get_library(library, create_if_missing=True)

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

        Each call creates a new ArcticDB version (automatic versioning).
        A ``transaction_time`` column is added if not already present,
        stamped with UTC now.

        Parameters
        ----------
        ticker:
            Ticker symbol (e.g. ``"AAPL"``).
        df:
            Polars DataFrame with at least ``valid_time`` column.
        metadata:
            Optional dict stored alongside the version in ArcticDB.
        """
        if "transaction_time" not in df.columns:
            tx = datetime.now(timezone.utc)
            df = df.with_columns(
                pl.lit(tx).cast(pl.Datetime("us", "UTC")).alias("transaction_time")
            )

        symbol = self._TICKER_PREFIX + ticker
        pd_df = df.to_pandas()

        # ArcticDB requires a DatetimeIndex for timeseries data
        if "valid_time" in pd_df.columns:
            import pandas as pd
            pd_df["valid_time"] = pd.to_datetime(pd_df["valid_time"])
            pd_df = pd_df.set_index("valid_time")

        self._lib.write(symbol, pd_df, metadata=metadata or {})

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
            Point-in-time filter.  Only rows with
            ``transaction_time <= as_of_transaction_time`` are returned.
            Pass ``None`` to return the latest snapshot.

        Returns
        -------
        pl.DataFrame
            Features with valid_time as a column (not index).
        """
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

        # Point-in-time filter
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
        prefix = self._TICKER_PREFIX
        return [
            s[len(prefix):]
            for s in self._lib.list_symbols()
            if s.startswith(prefix)
        ]

    def list_versions(self, ticker: str) -> list[dict]:
        """Return version history for a ticker."""
        symbol = self._TICKER_PREFIX + ticker
        if not self._lib.has_symbol(symbol):
            return []
        return list(self._lib.list_versions(symbol))
