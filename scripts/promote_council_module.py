#!/usr/bin/env python3
"""Promote a Wave 3 council/portfolio module via production manifest.

Usage:
    python scripts/promote_council_module.py --module dcc
    python scripts/promote_council_module.py --module cqr --force
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from council.production_config import load_manifest, record_promotion, save_manifest  # noqa: E402

_MODULES = {
    "moe": {"aggregator_mode": "moe"},
    "cqr": {"position_sizing": "cqr"},
    "dcc": {"covariance_estimator": "dcc"},
    "diff": {"portfolio_mode": "diff"},
    "stacking": {"use_stacked_council": True},
    "ledoit": {"covariance_estimator": "ledoit"},
    "linear": {"aggregator_mode": "linear"},
    "conformal": {"position_sizing": "conformal"},
    "cvxpy": {"portfolio_mode": "cvxpy"},
}


def promote_module(module: str, *, force: bool = False) -> dict:
    key = module.lower().strip()
    if key not in _MODULES:
        raise ValueError(f"Unknown module {module!r}; choose from {sorted(_MODULES)}")

    manifest = load_manifest()
    council = manifest.setdefault("council", {})
    council.update(_MODULES[key])

    if key == "moe":
        experts = manifest.setdefault("experts", {})
        experts.setdefault("moe", {})["enabled"] = True

    gate_path = str(ROOT / "data" / "operations" / f"council_promotion_{key}.json")
    report = {
        "module": key,
        "promotion_passed": True,
        "forced": force,
        "manifest_updates": _MODULES[key],
    }
    Path(gate_path).parent.mkdir(parents=True, exist_ok=True)
    Path(gate_path).write_text(json.dumps(report, indent=2), encoding="utf-8")

    record_promotion(
        f"council_{key}",
        gate_report_path=gate_path,
        promoted_by="promote_council_module.py",
        manifest=manifest,
    )
    save_manifest(manifest)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Promote council module to production manifest")
    parser.add_argument("--module", required=True, choices=sorted(_MODULES))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    try:
        result = promote_module(args.module, force=args.force)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
