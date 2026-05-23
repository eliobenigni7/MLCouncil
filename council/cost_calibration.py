"""Self-calibrating transaction cost model — engine (ADR-0003 Stage B).

Reads :class:`execution.fill_log.FillRecord` history and derives per-ticker
(and per-tier) ``kappa_slippage_bps`` from realised implementation shortfall.

The artifact is written as JSON with a SHA-256 sidecar manifest so that
``TransactionCostModel.from_env()`` can fail closed on tampered files,
mirroring the policy in :mod:`council.pickle_security`.

Calibration formula (v1)::

    IS_bps  = 10_000 * (fill_price - decision_price) / decision_price * sign(side)
    kappa_t = rolling_median(IS_bps over last N fills for ticker t)
    kappa_T = rolling_median(IS_bps over all fills in tier T)

Tier mapping uses the same buckets as ``ILLIQUIDITY_MAP`` in
:mod:`council.transaction_costs` (``mega``, ``large``, ``mid``, ``crypto``,
fallback ``default``).
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import polars as pl

_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CALIBRATION_PATH = _ROOT / "data" / "operations" / "cost_calibration.json"
DEFAULT_FILLS_DIR = _ROOT / "data" / "operations" / "fills"
DEFAULT_MIN_FILLS = 30

# Coarse ticker → tier mapping. Aligned with ILLIQUIDITY_MAP in
# council/transaction_costs.py. Tickers not listed fall back to "default".
TIER_BY_TICKER: dict[str, str] = {
    # mega (universe large_cap)
    "AAPL": "mega", "MSFT": "mega", "GOOGL": "mega", "AMZN": "mega",
    "META": "mega", "NVDA": "mega", "TSLA": "mega", "JPM": "mega",
    "V": "mega", "MA": "mega", "LLY": "mega", "UNH": "mega",
    "JNJ": "mega", "WMT": "mega", "PG": "mega", "KO": "mega", "PEP": "mega",
    "XOM": "mega", "CVX": "mega", "HD": "mega", "DIS": "mega", "TMUS": "mega",
    # large
    "NKE": "large", "CAT": "large", "BA": "large", "HON": "large",
    "UNP": "large", "NEE": "large", "DUK": "large", "GS": "large",
    "BAC": "large", "LIN": "large", "APD": "large",
    "AMT": "large", "PLD": "large",
    "UBER": "large", "PLTR": "large", "CRWD": "large", "DDOG": "large",
    "SHOP": "large", "MRK": "large", "ABT": "large", "PFE": "large", "COP": "large",
    # mid (universe mid_cap + legacy names)
    "AIG": "mid", "CB": "mid", "MET": "mid", "TFC": "mid", "USB": "mid",
    "CL": "mid", "MDLZ": "mid", "AMGN": "mid", "BMY": "mid", "GILD": "mid",
    "GE": "mid", "LMT": "mid", "RTX": "mid", "ADP": "mid", "INTU": "mid",
    "ETSY": "mid", "FVRR": "mid", "ROKU": "mid", "DOCU": "mid",
    "ABNB": "mid", "NET": "mid", "SQ": "mid", "SNOW": "mid",
    # crypto
    "BTCUSD": "crypto", "ETHUSD": "crypto",
}


def ticker_tier(ticker: str) -> str:
    return TIER_BY_TICKER.get(ticker, "default")


@dataclass(frozen=True)
class CalibrationArtifact:
    """In-memory view of ``cost_calibration.json``."""

    generated_at: datetime
    calibration_window_end: datetime
    fill_sample_count: int
    min_fills: int
    kappa_by_ticker: dict[str, float]
    fill_count_by_ticker: dict[str, int]
    kappa_by_tier: dict[str, float]
    fill_count_by_tier: dict[str, int]
    pipeline_run_id: str = ""
    config_hash: str = ""
    version: str = ""

    def to_dict(self) -> dict:
        return {
            "generated_at": self.generated_at.isoformat(),
            "calibration_window_end": self.calibration_window_end.isoformat(),
            "fill_sample_count": self.fill_sample_count,
            "min_fills": self.min_fills,
            "kappa_by_ticker": self.kappa_by_ticker,
            "fill_count_by_ticker": self.fill_count_by_ticker,
            "kappa_by_tier": self.kappa_by_tier,
            "fill_count_by_tier": self.fill_count_by_tier,
            "pipeline_run_id": self.pipeline_run_id,
            "config_hash": self.config_hash,
            "version": self.version,
        }


# ---------------------------------------------------------------------------
# Calibration math
# ---------------------------------------------------------------------------


def compute_is_bps(df: pl.DataFrame) -> pl.DataFrame:
    """Add an ``is_bps`` column to a fills frame (idempotent)."""
    if df.height == 0:
        return df.with_columns(pl.lit(None, dtype=pl.Float64).alias("is_bps"))

    sign_expr = (
        pl.when(pl.col("side") == "buy").then(1.0).otherwise(-1.0)
    )
    is_bps = (
        (pl.col("fill_price") - pl.col("decision_price"))
        / pl.col("decision_price")
        * 10_000.0
        * sign_expr
    ).alias("is_bps")
    return df.with_columns(is_bps)


def _median_by(df: pl.DataFrame, group: str) -> tuple[dict, dict]:
    """Return ``(median, count)`` dicts grouped by *group* column."""
    if df.height == 0:
        return {}, {}
    grouped = df.group_by(group).agg(
        pl.col("is_bps").median().alias("kappa"),
        pl.len().alias("n"),
    )
    kappa = {row[group]: float(row["kappa"]) for row in grouped.iter_rows(named=True)}
    counts = {row[group]: int(row["n"]) for row in grouped.iter_rows(named=True)}
    return kappa, counts


# ---------------------------------------------------------------------------
# Calibrator
# ---------------------------------------------------------------------------


@dataclass
class CostCalibrator:
    """Build a :class:`CalibrationArtifact` from a fills frame."""

    min_fills: int = DEFAULT_MIN_FILLS

    def calibrate(
        self,
        fills: pl.DataFrame,
        *,
        pipeline_run_id: str = "",
        config_hash: str = "",
    ) -> CalibrationArtifact:
        df = compute_is_bps(fills)

        # per-ticker
        kappa_ticker_all, count_ticker = _median_by(df, "ticker")
        kappa_ticker = {
            t: k for t, k in kappa_ticker_all.items() if count_ticker.get(t, 0) >= self.min_fills
        }

        # per-tier — always include tiers with enough data, regardless of
        # whether individual tickers cleared the threshold.
        tier_df = df.with_columns(
            pl.col("ticker").map_elements(ticker_tier, return_dtype=pl.Utf8).alias("tier")
        )
        kappa_tier_all, count_tier = _median_by(tier_df, "tier")
        kappa_tier = {
            t: k for t, k in kappa_tier_all.items() if count_tier.get(t, 0) >= self.min_fills
        }

        window_end = (
            df["fill_ts"].max()
            if df.height > 0 and "fill_ts" in df.columns
            else datetime.now(timezone.utc)
        )

        return CalibrationArtifact(
            generated_at=datetime.now(timezone.utc),
            calibration_window_end=window_end,
            fill_sample_count=df.height,
            min_fills=self.min_fills,
            kappa_by_ticker=kappa_ticker,
            fill_count_by_ticker={
                t: int(c) for t, c in count_ticker.items() if t in kappa_ticker
            },
            kappa_by_tier=kappa_tier,
            fill_count_by_tier={
                t: int(c) for t, c in count_tier.items() if t in kappa_tier
            },
            pipeline_run_id=pipeline_run_id,
            config_hash=config_hash,
        )


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def _calibration_version(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def write_calibration(
    artifact: CalibrationArtifact,
    path: Path = DEFAULT_CALIBRATION_PATH,
) -> str:
    """Write ``cost_calibration.json`` + ``.manifest`` sidecar.

    Returns the SHA-256 version string also stored in
    ``artifact.version`` and inside the manifest.
    """
    payload = json.dumps(artifact.to_dict(), indent=2, sort_keys=True).encode("utf-8")
    version = _calibration_version(payload)

    path.parent.mkdir(parents=True, exist_ok=True)

    # write atomically
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(payload)
    os.replace(tmp, path)

    manifest = {
        "artifact_path": str(path),
        "sha256": version,
        "fill_sample_count": artifact.fill_sample_count,
        "calibration_window_end": artifact.calibration_window_end.isoformat(),
        "pipeline_run_id": artifact.pipeline_run_id,
        "config_hash": artifact.config_hash,
        "generated_at": artifact.generated_at.isoformat(),
    }
    manifest_path = path.with_suffix(path.suffix + ".manifest")
    manifest_payload = json.dumps(manifest, indent=2, sort_keys=True).encode("utf-8")
    tmp_m = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
    tmp_m.write_bytes(manifest_payload)
    os.replace(tmp_m, manifest_path)

    return version


class CalibrationHashError(ValueError):
    """Raised when the manifest hash doesn't match the artifact."""


def load_calibration(
    path: Path = DEFAULT_CALIBRATION_PATH,
    *,
    require_manifest: bool = True,
) -> CalibrationArtifact:
    """Load a calibration artifact, fail closed on hash mismatch.

    When ``require_manifest=True`` (default) the ``.manifest`` sidecar must
    exist and match the file content's SHA-256. Set ``require_manifest=False``
    only in tests or rollback scenarios.
    """
    payload = path.read_bytes()
    actual_sha = _calibration_version(payload)

    if require_manifest:
        manifest_path = path.with_suffix(path.suffix + ".manifest")
        if not manifest_path.exists():
            raise CalibrationHashError(
                f"Missing manifest sidecar for {path}; use require_manifest=False to bypass."
            )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        expected_sha = manifest.get("sha256", "")
        if expected_sha != actual_sha:
            raise CalibrationHashError(
                f"Cost calibration hash mismatch: expected {expected_sha}, got {actual_sha}"
            )

    data = json.loads(payload)

    def _parse_dt(s: str) -> datetime:
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt

    return CalibrationArtifact(
        generated_at=_parse_dt(data["generated_at"]),
        calibration_window_end=_parse_dt(data["calibration_window_end"]),
        fill_sample_count=int(data["fill_sample_count"]),
        min_fills=int(data["min_fills"]),
        kappa_by_ticker=dict(data.get("kappa_by_ticker", {})),
        fill_count_by_ticker={k: int(v) for k, v in data.get("fill_count_by_ticker", {}).items()},
        kappa_by_tier=dict(data.get("kappa_by_tier", {})),
        fill_count_by_tier={k: int(v) for k, v in data.get("fill_count_by_tier", {}).items()},
        pipeline_run_id=str(data.get("pipeline_run_id", "")),
        config_hash=str(data.get("config_hash", "")),
        version=actual_sha,
    )


# ---------------------------------------------------------------------------
# End-to-end helper
# ---------------------------------------------------------------------------


def run_calibration_job(
    *,
    fills_dir: Path = DEFAULT_FILLS_DIR,
    out_path: Path = DEFAULT_CALIBRATION_PATH,
    min_fills: int = DEFAULT_MIN_FILLS,
    pipeline_run_id: str = "",
    config_hash: str = "",
) -> Optional[CalibrationArtifact]:
    """Read fills, build the artifact, write it. Returns None on empty input."""
    from execution.fill_log import read_fills

    fills = read_fills(base=fills_dir)
    if fills.height == 0:
        return None

    artifact = CostCalibrator(min_fills=min_fills).calibrate(
        fills,
        pipeline_run_id=pipeline_run_id,
        config_hash=config_hash,
    )
    version = write_calibration(artifact, path=out_path)
    return CalibrationArtifact(
        generated_at=artifact.generated_at,
        calibration_window_end=artifact.calibration_window_end,
        fill_sample_count=artifact.fill_sample_count,
        min_fills=artifact.min_fills,
        kappa_by_ticker=artifact.kappa_by_ticker,
        fill_count_by_ticker=artifact.fill_count_by_ticker,
        kappa_by_tier=artifact.kappa_by_tier,
        fill_count_by_tier=artifact.fill_count_by_tier,
        pipeline_run_id=artifact.pipeline_run_id,
        config_hash=artifact.config_hash,
        version=version,
    )
