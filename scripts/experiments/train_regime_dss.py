#!/usr/bin/env python
"""Train deep state-space regime challenger (T2.3) — shadow only.

Does not modify the Dagster daily path (HMM remains default). Writes
``models/checkpoints/regime_dss_latest.pkl`` + hash sidecar for shadow inference.

Usage:
    python scripts/train_regime_dss.py
    python scripts/train_regime_dss.py --epochs 40 --compare-hmm
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import polars as pl

from models.regime import RegimeModel
from models.regime_dss import DeepRegimeModel


def _load_macro(root: Path) -> pl.DataFrame:
    macro_dir = root / "data" / "macro"

    def _path(name: str) -> str | None:
        p = macro_dir / f"{name}.parquet"
        return str(p) if p.exists() else None

    try:
        from data.features.alpha158 import build_macro_context

        macro = build_macro_context(
            vix_path=_path("vix"),
            treasuries_path=_path("treasuries"),
            sp500_path=_path("sp500"),
        )
        if not macro.is_empty():
            return macro
    except Exception as exc:
        print(f"[warn] build_macro_context failed: {exc}")

    print("[info] Using synthetic macro fallback for DSS training")
    from datetime import date, timedelta

    import numpy as np

    rng = np.random.default_rng(42)
    n = 400
    start = date(2022, 1, 3)
    dates = [start + timedelta(days=i) for i in range(n)]
    return pl.DataFrame(
        {
            "valid_time": dates,
            "sp500_ret_20d": rng.normal(0.0, 0.02, n).tolist(),
            "vix": (15.0 + rng.normal(0, 2, n)).tolist(),
            "yield_spread": (1.0 + rng.normal(0, 0.1, n)).tolist(),
        }
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train DSS regime challenger (shadow)")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--epochs", type=int, default=None, help="Override config n_epochs")
    parser.add_argument(
        "--compare-hmm",
        action="store_true",
        help="Print ELBO vs HMM log-likelihood proxy after training",
    )
    parser.add_argument("--json", action="store_true", help="Emit metadata JSON on stdout")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    macro = _load_macro(args.root)
    if macro.is_empty():
        print("No macro data available.", file=sys.stderr)
        return 1

    model = DeepRegimeModel()
    metrics = model.fit(macro, n_epochs=args.epochs)
    out = args.root / "models" / "checkpoints" / "regime_dss_latest.pkl"
    out.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(out))

    report: dict = {
        "checkpoint": str(out),
        "metrics": {
            "elbo_final": metrics.elbo_final,
            "elbo_initial": metrics.elbo_initial,
            "elbo_improvement_pct": metrics.elbo_improvement_pct,
            "backend": metrics.backend,
            "embed_dim": metrics.embed_dim,
        },
        "metadata": model.get_metadata(),
        "latest_regime": model.predict_regime(macro),
        "latest_embedding": model.predict_embedding(macro).tolist(),
    }

    if args.compare_hmm:
        report["hmm_comparison"] = model.compare_hmm_elbo(macro, hmm=RegimeModel())

    print(
        f"[regime_dss] trained backend={metrics.backend} "
        f"ELBO {metrics.elbo_initial:.4f} -> {metrics.elbo_final:.4f} "
        f"({metrics.elbo_improvement_pct:+.1f}%)"
    )
    print(f"[regime_dss] shadow checkpoint -> {out}")
    print(f"[regime_dss] latest label={report['latest_regime']}")

    if args.json:
        print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
