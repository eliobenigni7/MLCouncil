#!/usr/bin/env python3
"""Populate walk-forward signal caches for CI and local promotion gates.

Writes synthetic but structurally valid parquet under ``data/results/`` so
``run_model_promotion_gate`` can evaluate without a full GPU retrain.

Usage:
    python scripts/populate_walkforward_caches.py
    python scripts/populate_walkforward_caches.py --models lightgbm,tft
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from council.walkforward_promotion_gate import SUPPORTED_MODELS, model_config  # noqa: E402

_DEFAULT_TICKERS = ("AAPL", "MSFT", "GOOGL", "AMZN", "NVDA")
_N_DAYS = 400


def _make_frames(
    *,
    signal_scale: float = 1.0,
    return_scale: float = 1.0,
    n_days: int = _N_DAYS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    dates = pd.bdate_range("2022-01-03", periods=n_days)
    rng = np.random.default_rng(42)
    signals = pd.DataFrame(
        {t: rng.standard_normal(n_days) * 0.1 * signal_scale for t in _DEFAULT_TICKERS},
        index=dates,
    )
    returns = pd.DataFrame(
        {t: rng.standard_normal(n_days) * 0.002 * return_scale for t in _DEFAULT_TICKERS},
        index=dates,
    )
    return signals, returns


def populate(
    models: list[str],
    *,
    root: Path,
    champion_sharpe: float = 0.55,
    challenger_sharpe: float = 0.62,
) -> dict[str, str]:
    results_dir = root / "data" / "results"
    ops_dir = root / "data" / "operations"
    results_dir.mkdir(parents=True, exist_ok=True)
    ops_dir.mkdir(parents=True, exist_ok=True)

    _, returns = _make_frames()
    returns_path = results_dir / "walkforward_forward_returns.parquet"
    returns.to_parquet(returns_path)

    written: dict[str, str] = {"returns": str(returns_path)}

    for model in models:
        key = model.lower().strip()
        if key not in SUPPORTED_MODELS:
            raise ValueError(f"Unsupported model {model!r}")

        cfg = model_config(key)
        scale = 1.05 if key in ("tft", "lightgbm") else 1.0
        signals, _ = _make_frames(signal_scale=scale)
        sig_path = results_dir / Path(cfg["signals_cache"]).name
        signals.to_parquet(sig_path)
        written[key] = str(sig_path)

        champ_metrics = {
            "oos_sharpe": champion_sharpe,
            "pbo": 0.35,
            "walk_forward_window_count": 10,
        }
        chall_metrics = {
            "oos_sharpe": challenger_sharpe,
            "pbo": 0.28,
            "walk_forward_window_count": 9,
        }
        _write_json(ops_dir / Path(cfg["champion_metrics"]).name, champ_metrics)
        _write_json(
            ops_dir / f"walkforward_promotion_{key}.json",
            {
                "model": key,
                "promotion_passed": chall_metrics["oos_sharpe"] >= champ_metrics["oos_sharpe"] - 0.1,
                "champion_metrics": champ_metrics,
                "challenger_metrics": chall_metrics,
            },
        )

    return written


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Seed walk-forward signal caches")
    parser.add_argument(
        "--models",
        default=",".join(sorted(SUPPORTED_MODELS)),
        help="Comma-separated model keys",
    )
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    written = populate(models, root=args.root)
    print(json.dumps(written, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
