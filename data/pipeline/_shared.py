"""Helper condivisi della pipeline Dagster MLCouncil.

Costanti di percorso, caricamento universo/config e helper comuni usati dagli
asset dei vari layer (data/pipeline/assets_*.py).

Namespace condiviso
-------------------
I test (tests/test_pipeline.py, tests/test_canary.py) caricano data/pipeline.py
come modulo standalone (importlib spec_from_file_location) e patchano le
costanti/helper definiti qui (es. ``_DATA_DIR``, ``_safe_pickle_load``) tramite
monkeypatch/patch.object sul modulo entry point. Per questo:

- gli asset accedono SEMPRE a questi nomi come ``_shared.NOME`` (attributo del
  modulo, mai import di valore che congelerebbe il binding);
- i moduli entry point (data/pipeline/__init__.py e data/pipeline.py)
  allineano il proprio ``__dict__`` a questo modulo condiviso, così le patch
  dei test scrivono nello stato effettivamente letto dagli asset.

Nota sui path: questo modulo vive un livello più profondo del vecchio
data/pipeline.py, quindi ``_ROOT`` usa ``parents[2]`` invece di ``parents[1]``.
"""

import sys
import types
from datetime import date as date_type
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import yaml
import dagster as dg
from dagster import AssetExecutionContext

from data.contracts import validate_asset_contract, version_payload
from data.lineage import lineage_artifact_payload


class _SharedNamespaceModule(types.ModuleType):
    """ModuleType con namespace condiviso con questo modulo.

    I test (tests/test_pipeline.py, tests/test_canary.py) caricano
    data/pipeline.py come modulo standalone e patchano costanti/helper via
    monkeypatch.setattr / patch.object sul modulo entry point. Gli entry point
    (data/pipeline.py e data/pipeline/__init__.py) adottano questa classe:
    ogni scrittura di attributo viene inoltrata anche a ``_SHARED_NAMESPACE``,
    così le patch raggiungono lo stato effettivamente letto dagli asset
    (che accedono ai nomi come ``_shared.NOME``).
    """

    def __setattr__(self, name: str, value) -> None:
        types.ModuleType.__setattr__(self, name, value)
        if not name.startswith("__"):
            _SHARED_NAMESPACE[name] = value


# Namespace condiviso: punto di inoltro per le scritture di attributo dei moduli
# entry point (vedi _SharedNamespaceModule).
_SHARED_NAMESPACE = globals()

# ---------------------------------------------------------------------------
# Path bootstrap — un livello più profondo rispetto al vecchio data/pipeline.py
# ---------------------------------------------------------------------------

_ROOT = Path(__file__).parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

_DATA_DIR       = _ROOT / "data" / "raw"
_ORDERS_DIR     = _ROOT / "data" / "orders"
_RESULTS_DIR    = _ROOT / "data" / "results"
_CHECKPOINTS    = _ROOT / "models" / "checkpoints"
_EXCLUDE_COLS   = {"ticker", "valid_time", "transaction_time"}
_MIN_ALPHA_FEATURES = 50
_DEFAULT_PORTFOLIO_VALUE = 100_000.0


class LivePortfolioSnapshotError(RuntimeError):
    """Errore pipeline per snapshot live Alpaca non disponibile o non valido."""


# ---------------------------------------------------------------------------
# Shared config
# ---------------------------------------------------------------------------

_DAILY_PARTITIONS = dg.DailyPartitionsDefinition(start_date="2018-01-01")
_RETRY            = dg.RetryPolicy(max_retries=2, delay=30)


def _safe_pickle_load(path: Path):
    """Carica un checkpoint pickle solo con sidecar SHA-256 obbligatorio."""
    from council.pickle_security import trusted_pickle_load

    return trusted_pickle_load(path, require_hash=True)


def _load_universe(include_crypto: bool = True) -> list[str]:
    """Carica la lista dei ticker da config/universe.yaml.

    Supporta sia il formato legacy con `universe.tickers` sia il formato
    bucketed corrente (`large_cap`, `mid_cap`, ...), ignorando la sezione
    `settings`. Include anche `crypto_universe` se presente e include_crypto=True.
    """
    with open(_ROOT / "config" / "universe.yaml") as f:
        cfg = yaml.safe_load(f)

    tickers: list[str] = []
    seen: set[str] = set()

    # Equity universe
    universe_cfg = cfg.get("universe", {})
    if isinstance(universe_cfg.get("tickers"), list):
        equity_tickers = universe_cfg["tickers"]
    else:
        equity_tickers = []
        for bucket_name, bucket_values in universe_cfg.items():
            if bucket_name == "settings" or not isinstance(bucket_values, list):
                continue
            equity_tickers.extend(bucket_values)

    for ticker in equity_tickers:
        if ticker not in seen:
            seen.add(ticker)
            tickers.append(ticker)

    # Crypto universe
    if include_crypto:
        crypto_cfg = cfg.get("crypto_universe", {})
        if isinstance(crypto_cfg, dict):
            for bucket_values in crypto_cfg.values():
                if not isinstance(bucket_values, list):
                    continue
                for ticker in bucket_values:
                    if ticker not in seen:
                        seen.add(ticker)
                        tickers.append(ticker)
        elif isinstance(crypto_cfg, list):
            # Flat list format
            for ticker in crypto_cfg:
                if ticker not in seen:
                    seen.add(ticker)
                    tickers.append(ticker)

    return tickers


def load_universe_as_of(
    as_of_date: str | date_type | None = None,
    include_crypto: bool = True,
) -> list[str]:
    """Return only tickers that were universe members on *as_of_date*.

    Uses ``config/universe_history.yaml`` which records ``added`` /
    ``removed`` dates per ticker.  Falls back to :func:`_load_universe`
    (full current universe) when the history file is missing or
    *as_of_date* is ``None``.

    Parameters
    ----------
    as_of_date:
        ISO-8601 date string or ``datetime.date``.  ``None`` → current
        universe (no survivorship filtering).
    include_crypto:
        Whether to include crypto tickers (BTCUSD, ETHUSD …).
    """
    if as_of_date is None:
        return _load_universe(include_crypto=include_crypto)

    if isinstance(as_of_date, str):
        as_of_date = date_type.fromisoformat(as_of_date)

    history_path = _ROOT / "config" / "universe_history.yaml"
    if not history_path.exists():
        return _load_universe(include_crypto=include_crypto)

    with open(history_path) as f:
        history = yaml.safe_load(f) or {}

    membership = history.get("membership", {})
    if not membership:
        return _load_universe(include_crypto=include_crypto)

    # Also load the current universe to know which tickers are equity vs crypto
    with open(_ROOT / "config" / "universe.yaml") as f:
        cfg = yaml.safe_load(f)
    crypto_tickers: set[str] = set()
    crypto_cfg = cfg.get("crypto_universe", {})
    if isinstance(crypto_cfg, dict):
        for bucket_values in crypto_cfg.values():
            if isinstance(bucket_values, list):
                crypto_tickers.update(bucket_values)
    elif isinstance(crypto_cfg, list):
        crypto_tickers.update(crypto_cfg)

    tickers: list[str] = []
    for ticker, periods in membership.items():
        if not include_crypto and ticker in crypto_tickers:
            continue
        added = date_type.fromisoformat(str(periods.get("added", "2018-01-01")))
        removed_raw = periods.get("removed")
        removed = date_type.fromisoformat(str(removed_raw)) if removed_raw else None

        if as_of_date >= added and (removed is None or as_of_date < removed):
            tickers.append(ticker)

    return tickers


def _normalize_df(df: pl.DataFrame) -> pl.DataFrame:
    """Normalize datetime columns to UTC timezone for Polars 1.x strict concat.
    Also cast Datetime to Date for compatibility with existing parquet files."""
    if df.is_empty():
        return df
    # Cast Datetime -> Date (UTC midnight) for compatibility
    for c in df.columns:
        if df[c].dtype == pl.Datetime:
            df = df.with_columns(
                pl.col(c).dt.replace_time_zone("UTC").dt.convert_time_zone("UTC").cast(pl.Date)
            )
    return df


def _load_all_ohlcv(extra: pl.DataFrame | None = None) -> pl.DataFrame:
    """Legge tutti i parquet OHLCV storici, con eventuale append di `extra`."""
    ohlcv_dir = _DATA_DIR / "ohlcv"
    frames: list[pl.DataFrame] = []
    if ohlcv_dir.exists():
        for ticker_dir in sorted(ohlcv_dir.iterdir()):
            if not ticker_dir.is_dir():
                continue
            for pq in sorted(ticker_dir.glob("*.parquet")):
                try:
                    frames.append(_normalize_df(pl.read_parquet(pq)))
                except Exception:
                    pass
    if extra is not None and not extra.is_empty():
        frames.append(_normalize_df(extra))
    if not frames:
        return pl.DataFrame()
    return (
        pl.concat(frames)
        .unique(["ticker", "valid_time"])
        .sort(["ticker", "valid_time"])
    )


def _load_partitioned_parquet(subdir: str, partition_date: str) -> pl.DataFrame:
    """Load a partitioned parquet written under data/raw/<subdir>/<date>.parquet."""
    path = _DATA_DIR / "raw" / subdir / f"{partition_date}.parquet"
    if not path.exists():
        return pl.DataFrame()
    try:
        return _normalize_df(pl.read_parquet(path))
    except Exception:
        return pl.DataFrame()


def _load_macro_context_from_disk() -> pl.DataFrame:
    """Load the macro context parquet files saved by download_macro."""
    macro_dir = _DATA_DIR / "macro"

    def _path(name: str) -> str | None:
        p = macro_dir / f"{name}.parquet"
        return str(p) if p.exists() else None

    from data.features.alpha158 import build_macro_context

    return build_macro_context(
        vix_path=_path("vix"),
        treasuries_path=_path("treasuries"),
        sp500_path=_path("sp500"),
    )


def _record_asset_metadata(
    context: AssetExecutionContext,
    asset_name: str,
    payload,
    partition_date: str,
    lineage: dict[str, str] | None = None,
) -> dict[str, object]:
    contract_summary = validate_asset_contract(asset_name, payload, partition_date)
    metadata: dict[str, object] = {
        "asset_name": asset_name,
        "partition_date": partition_date,
        "row_count": contract_summary["row_count"],
        "column_count": contract_summary["column_count"],
        "payload_version": version_payload(asset_name, payload, partition_date),
    }
    if lineage:
        metadata.update(lineage_artifact_payload(lineage))
    context.add_output_metadata(metadata)
    return metadata


def _contract_check_result(asset_name: str, payload, partition_date: str | None = None) -> dg.AssetCheckResult:
    try:
        summary = validate_asset_contract(asset_name, payload, partition_date)
    except Exception as exc:
        return dg.AssetCheckResult(
            passed=False,
            metadata={
                "asset_name": asset_name,
                "error": str(exc),
                "partition_date": partition_date or "n/a",
            },
        )

    return dg.AssetCheckResult(
        passed=True,
        metadata={
            "asset_name": asset_name,
            "row_count": summary["row_count"],
            "column_count": summary["column_count"],
            "partition_date": partition_date or "n/a",
        },
    )


def _load_live_portfolio_snapshot(
    target_tickers: list[str] | None = None,
) -> tuple[pd.Series, float]:
    zero_weights = pd.Series(dtype=float, name="current_weight")
    if target_tickers is not None:
        zero_weights = pd.Series(
            0.0, index=target_tickers, dtype=float, name="current_weight"
        )
    try:
        from execution.alpaca_adapter import AlpacaConfig, AlpacaLiveNode

        node = AlpacaLiveNode(AlpacaConfig.from_env())
        account = node.get_account_info()
        portfolio_value = float(account.get("portfolio_value", 0.0) or 0.0)
        if not np.isfinite(portfolio_value) or portfolio_value <= 0:
            raise LivePortfolioSnapshotError(
                f"live portfolio snapshot: invalid portfolio value {portfolio_value!r}"
            )

        positions_df = node.get_all_positions(strict=True)
        if positions_df.empty:
            return zero_weights, portfolio_value

        required_cols = {"symbol", "current_value"}
        missing_cols = sorted(required_cols - set(positions_df.columns))
        if missing_cols:
            raise LivePortfolioSnapshotError(
                "live portfolio snapshot: malformed positions payload "
                f"(missing columns: {', '.join(missing_cols)})"
            )

        if positions_df["symbol"].isna().any():
            raise LivePortfolioSnapshotError(
                "live portfolio snapshot: malformed positions payload (null symbols)"
            )

        current_values = pd.to_numeric(
            positions_df["current_value"], errors="coerce"
        ).astype(float)
        if current_values.isna().any() or not np.isfinite(current_values).all():
            raise LivePortfolioSnapshotError(
                "live portfolio snapshot: malformed positions payload "
                "(invalid current_value)"
            )

        normalized_positions = positions_df.assign(current_value=current_values)
        if normalized_positions["symbol"].duplicated().any():
            # Alpaca puo' restituire lo stesso simbolo da sorgenti multiple
            # (es. TradingClient + endpoint crypto). Manteniamo una sola riga
            # per ticker per evitare di contare due volte la stessa esposizione.
            normalized_positions = normalized_positions.drop_duplicates(
                subset=["symbol"], keep="last"
            )

        current_weights = (
            normalized_positions
            .set_index("symbol")["current_value"]
            .astype(float)
            .div(portfolio_value)
            .rename("current_weight")
        )
        if target_tickers is None:
            return current_weights.sort_index(), portfolio_value
        return current_weights.reindex(target_tickers).fillna(0.0), portfolio_value
    except LivePortfolioSnapshotError:
        raise
    except Exception as exc:
        raise LivePortfolioSnapshotError(
            f"live portfolio snapshot unavailable: {exc}"
        ) from exc


def _load_returns_wide(tickers: list[str], tail_days: int = 90) -> "pd.DataFrame | None":
    """Carica i rendimenti giornalieri multi-ticker (pivot wide) per i check settimanali.

    Ritorna None se non ci sono parquet OHLCV o se il pivot è vuoto.
    """
    ohlcv_dir = _DATA_DIR / "ohlcv"
    frames: list[pl.DataFrame] = []
    for ticker in tickers:
        ticker_dir = ohlcv_dir / ticker
        if ticker_dir.exists():
            for pq in sorted(ticker_dir.glob("*.parquet")):
                try:
                    frames.append(_normalize_df(pl.read_parquet(pq)))
                except Exception:
                    pass
    if not frames:
        return None
    ohlcv = (
        pl.concat(frames)
        .sort(["ticker", "valid_time", "transaction_time"])
        .unique(["ticker", "valid_time"], keep="last")
        .sort(["ticker", "valid_time"])
    )
    returns_wide = (
        ohlcv.select(["ticker", "valid_time", "adj_close"])
        .with_columns(
            (pl.col("adj_close") / pl.col("adj_close").shift(1) - 1)
            .over("ticker")
            .alias("ret_1d")
        )
        .filter(pl.col("ret_1d").is_not_null())
        .pivot(values="ret_1d", index="valid_time", on="ticker")
        .to_pandas()
        .set_index("valid_time")
        .tail(tail_days)
    )
    if returns_wide.empty:
        return None
    return returns_wide


def _compute_covariance(tickers: list[str]) -> pd.DataFrame:
    """Carica OHLCV da disco e calcola la matrice di covarianza (ultime 90 sessioni)."""
    ohlcv_dir = _DATA_DIR / "ohlcv"
    frames: list[pl.DataFrame] = []

    for ticker in tickers:
        ticker_dir = ohlcv_dir / ticker
        if ticker_dir.exists():
            for pq in sorted(ticker_dir.glob("*.parquet")):
                try:
                    frames.append(_normalize_df(pl.read_parquet(pq)))
                except Exception:
                    pass

    if not frames:
        n = len(tickers)
        return pd.DataFrame(np.eye(n) * 0.0001, index=tickers, columns=tickers)

    ohlcv = (
        pl.concat(frames)
        .sort(["ticker", "valid_time", "transaction_time"])
        .unique(["ticker", "valid_time"], keep="last")
        .sort(["ticker", "valid_time"])
    )
    # drop_nulls before pivot would discard every row where *any* ticker has a
    # missing return (e.g. halts, sparse mid-caps).  Instead, compute returns
    # per ticker (nulls only at each ticker's first row) then pivot and use
    # pairwise covariance so tickers with partial overlap still contribute.
    returns_wide = (
        ohlcv
        .select(["ticker", "valid_time", "adj_close"])
        .with_columns(
            (pl.col("adj_close") / pl.col("adj_close").shift(1) - 1)
            .over("ticker")
            .alias("ret_1d")
        )
        .filter(pl.col("ret_1d").is_not_null())
        .pivot(values="ret_1d", index="valid_time", on="ticker")
        .to_pandas()
        .set_index("valid_time")
        .tail(90)
    )
    from council.risk.covariance_dynamic import compute_covariance_from_returns

    return compute_covariance_from_returns(returns_wide)


def _load_market_returns() -> pd.Series | None:
    """Carica i ritorni di mercato (SPY o S&P 500) per beta neutrality."""
    spy_path = _DATA_DIR / "ohlcv" / "SPY"
    if not spy_path.exists():
        # Try alternative location
        spy_path = _DATA_DIR / "raw" / "ohlcv" / "SPY"
    if not spy_path.exists():
        return None
    try:
        all_files = sorted(spy_path.glob("*.parquet"))
        if not all_files:
            return None
        df = pd.concat([pd.read_parquet(f) for f in all_files], ignore_index=True)
        if "adj_close" in df.columns and "ticker" in df.columns:
            spy = df[df["ticker"] == "SPY"].sort_values("valid_time")
            spy_returns = spy["adj_close"].pct_change().dropna()
            spy_returns.index = spy["valid_time"].iloc[1:].values
            return spy_returns
    except Exception:
        pass
    return None
