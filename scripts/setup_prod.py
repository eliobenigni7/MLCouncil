#!/usr/bin/env python3
"""Initialize production setup: manifest, champion metrics seeds, gate dry-run.

Usage:
    python scripts/setup_prod.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from council.production_config import load_manifest, save_manifest  # noqa: E402
from council.walkforward_promotion_gate import (  # noqa: E402
    SUPPORTED_MODELS,
    run_model_promotion_gate,
)


def _seed_champion_metrics(model: str) -> None:
    path = ROOT / "data" / "operations" / f"walkforward_champion_{model}.json"
    if path.exists():
        return
    payload = {
        "oos_sharpe": 0.5,
        "pbo": 0.35,
        "walk_forward_window_count": 10,
        "note": "bootstrap placeholder — replace after first real walk-forward backtest",
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"  seeded {path.name}")


def main() -> None:
    print("==> Production manifest")
    manifest = load_manifest(ROOT / "config" / "production_manifest.yaml")
    save_manifest(manifest, ROOT / "config" / "production_manifest.yaml")
    print(f"    {ROOT / 'config' / 'production_manifest.yaml'}")

    print("\n==> Champion metric placeholders (walk-forward baseline)")
    for model in ("lightgbm", "sentiment", "hmm"):
        _seed_champion_metrics(model)

    print("\n==> Gate dry-run (needs signal caches for full evaluation)")
    for model in sorted(SUPPORTED_MODELS):
        report = run_model_promotion_gate(model, root=ROOT, dry_run=True)
        print(f"  {model}: {report['status']}")

    print("\nProd .env checklist:")
    print("  MLCOUNCIL_ENV_PROFILE=prod")
    print("  MLCOUNCIL_USE_PRODUCTION_MANIFEST=true")
    print("  MLCOUNCIL_AUTO_PROMOTE_MODELS=false")
    print("  (plus ALPACA_*, POLYGON_*, secrets from .env.example)")
    print("\nAfter caches exist and gate passes 3x:")
    print("  python scripts/promote_model.py --model lightgbm")


if __name__ == "__main__":
    main()
