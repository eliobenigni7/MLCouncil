#!/usr/bin/env python
"""Weekly walk-forward champion/challenger promotion orchestrator (T1.1).

Retrains a shadow challenger, evaluates purged walk-forward metrics, and runs
``validate_model_promotion``. Challengers never enter the daily Dagster pipeline
until manually promoted after consecutive CI passes.

Usage:
    python scripts/run_walkforward_promotion.py --model lightgbm --dry-run
    python scripts/run_walkforward_promotion.py --model sentiment
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from council.walkforward_promotion_gate import (  # noqa: E402
    SUPPORTED_MODELS,
    run_model_promotion_gate,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Walk-forward model promotion gate")
    parser.add_argument(
        "--model",
        required=True,
        choices=sorted(SUPPORTED_MODELS),
        help="Alpha model to evaluate (champion vs shadow challenger; tft compares vs lightgbm)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Skip retrain and streak persistence; still evaluate cached signals",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=ROOT,
        help="Repository root (default: project root)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print full gate report as JSON on stdout",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    report = run_model_promotion_gate(
        args.model,
        root=args.root,
        dry_run=args.dry_run,
    )
    if args.json:
        print(json.dumps(report, indent=2, default=str))
    else:
        print(f"Model: {report['model']}")
        print(f"Status: {report['status']}")
        print(f"Dry run: {report['dry_run']}")
        print(f"Promotion passed: {report['promotion_passed']}")
        if report.get("reasons"):
            print("Reasons:")
            for reason in report["reasons"]:
                print(f"  - {reason}")
        print(f"Report: {report['report_path']}")

    if report["status"] in {"gate_passed_shadow", "skipped_no_signal_cache", "skipped"}:
        return 0
    if report.get("promotion_passed"):
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
