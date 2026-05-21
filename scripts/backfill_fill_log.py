"""Backfill historical paper trades into the structured FillRecord log.

Reads existing ``data/paper_trades/{YYYY-MM-DD}.json`` files and creates
:class:`execution.fill_log.FillRecord` entries in
``data/operations/fills/{YYYY-MM}.parquet``.

Caveat
------
The legacy paper-trades JSON only captures **submission** metadata, not the
realised fill price. This script can run in two modes:

* ``--from-status`` (default): includes only records that already carry a
  ``filled_avg_price`` (added by order-status enrichment); skips
  submission-only records with a count.
* ``--include-submissions``: also writes FillRecord rows with
  ``fill_price == decision_price``, producing zero implementation-shortfall
  records. Useful for smoke-testing the calibration pipeline against
  whatever telemetry exists.

For production-grade backfill, run Alpaca order-status enrichment first to
populate ``filled_avg_price`` and ``filled_at`` before invoking this script.

Usage
-----
::

    python scripts/backfill_fill_log.py --dry-run
    python scripts/backfill_fill_log.py --include-submissions
"""

from __future__ import annotations

import argparse
import json
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from execution.fill_log import FillRecord, append_fills

PAPER_TRADES_DIR = _REPO / "data" / "paper_trades"


def _parse_dt(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except (ValueError, TypeError):
        return None


def _record_to_fill(record: dict, include_submissions: bool) -> FillRecord | None:
    """Map one paper_trades JSON entry to a FillRecord, or None to skip."""
    side = (record.get("side") or "").lower()
    if side not in {"buy", "sell"}:
        return None

    try:
        qty = abs(float(record.get("qty") or 0))
    except (TypeError, ValueError):
        return None
    if qty <= 0:
        return None

    fill_price = record.get("filled_avg_price") or record.get("fill_price")
    if fill_price is None:
        if not include_submissions:
            return None
        # submission-only fallback: use limit_price if available
        fill_price = record.get("limit_price") or record.get("decision_price")
    if fill_price is None:
        return None
    try:
        fill_price = float(fill_price)
    except (TypeError, ValueError):
        return None
    if fill_price <= 0:
        return None

    decision_price = record.get("decision_price") or record.get("limit_price") or fill_price
    try:
        decision_price = float(decision_price)
    except (TypeError, ValueError):
        return None

    submitted = _parse_dt(record.get("submitted_at")) or datetime.now(timezone.utc)
    filled = _parse_dt(record.get("filled_at")) or submitted

    order_id = str(record.get("order_id") or "")
    fill_id = f"{order_id}_F001" if order_id else f"backfill_{uuid.uuid4().hex[:12]}"

    return FillRecord(
        fill_id=fill_id,
        order_id=order_id,
        ticker=record.get("symbol", ""),
        side=side,
        qty=qty,
        fill_price=fill_price,
        decision_price=decision_price,
        decision_ts=submitted,
        fill_ts=filled,
        broker="alpaca",
        venue=("ALPACA_CRYPTO" if record.get("asset_class") == "crypto" else "ALPACA"),
        pipeline_run_id=record.get("pipeline_run_id", ""),
        config_hash=record.get("config_hash", ""),
        commission_bps=float(record.get("commission_bps", 0.0)),
        slippage_bps_assumed=float(record.get("slippage_bps_assumed", 0.0)),
        cost_calibration_version=record.get("cost_calibration_version", "backfill"),
    )


def backfill(
    paper_trades_dir: Path = PAPER_TRADES_DIR,
    include_submissions: bool = False,
    dry_run: bool = False,
) -> dict:
    """Run the backfill. Returns a summary dict."""
    if not paper_trades_dir.exists():
        return {"files": 0, "records_in": 0, "records_out": 0, "skipped": 0}

    # group by month to satisfy the partition contract
    by_month: dict[str, list[FillRecord]] = {}
    summary = {"files": 0, "records_in": 0, "records_out": 0, "skipped": 0}

    for json_file in sorted(paper_trades_dir.glob("*.json")):
        summary["files"] += 1
        try:
            entries = json.loads(json_file.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        if not isinstance(entries, list):
            continue
        for entry in entries:
            summary["records_in"] += 1
            fr = _record_to_fill(entry, include_submissions=include_submissions)
            if fr is None:
                summary["skipped"] += 1
                continue
            by_month.setdefault(fr.fill_ts.strftime("%Y-%m"), []).append(fr)
            summary["records_out"] += 1

    if dry_run:
        return summary

    for month, records in by_month.items():
        # de-dup within a month before writing (idempotent across runs)
        seen: set[str] = set()
        unique = []
        for r in records:
            if r.fill_id in seen:
                continue
            seen.add(r.fill_id)
            unique.append(r)
        append_fills(unique)

    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="parse only, do not write parquet")
    parser.add_argument(
        "--include-submissions",
        action="store_true",
        help="also emit zero-IS records for submission-only entries",
    )
    parser.add_argument(
        "--paper-trades-dir",
        default=str(PAPER_TRADES_DIR),
        help="path to data/paper_trades/ (default: %(default)s)",
    )
    args = parser.parse_args()

    summary = backfill(
        paper_trades_dir=Path(args.paper_trades_dir),
        include_submissions=args.include_submissions,
        dry_run=args.dry_run,
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
