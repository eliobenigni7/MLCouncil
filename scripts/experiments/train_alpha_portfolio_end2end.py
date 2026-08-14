"""Decision-focused alpha + portfolio training scaffold (T3.3).

When ``cvxpylayers`` is installed and ``MLCOUNCIL_PORTFOLIO_MODE=diff``, runs a
small synthetic loop optimising a linear alpha proxy against portfolio weights
from ``DifferentiablePortfolioConstructor``. Otherwise documents CVXPY delegate
path only.

Usage:
    python scripts/train_alpha_portfolio_end2end.py --epochs 20
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from council.portfolio.portfolio_diff import (
    DifferentiablePortfolioConstructor,
    cvxpylayers_available,
    portfolio_constructor_mode,
)


def _synthetic_panel(n_assets: int = 6, seed: int = 0):
    rng = np.random.default_rng(seed)
    tickers = [f"S{i}" for i in range(n_assets)]
    alpha = pd.Series(rng.standard_normal(n_assets), index=tickers)
    mult = pd.Series(np.ones(n_assets), index=tickers)
    current = pd.Series(np.ones(n_assets) / n_assets, index=tickers)
    A = rng.standard_normal((n_assets, n_assets)) * 0.01
    cov = pd.DataFrame(A.T @ A + np.eye(n_assets) * 1e-4, index=tickers, columns=tickers)
    return alpha, mult, current, cov


def main() -> None:
    parser = argparse.ArgumentParser(description="E2E alpha-portfolio training scaffold")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--portfolio-value", type=float, default=100_000.0)
    args = parser.parse_args()

    alpha, mult, current, cov = _synthetic_panel()
    ctor = DifferentiablePortfolioConstructor()
    print(f"backend={ctor.backend} cvxpylayers={cvxpylayers_available()} mode={portfolio_constructor_mode()}")

    best_sharpe = -np.inf
    best_w = None
    for epoch in range(args.epochs):
        weights = ctor.optimize(
            alpha,
            mult,
            current,
            cov,
            portfolio_value=args.portfolio_value,
        )
        aligned = weights.reindex(alpha.index).fillna(0.0)
        cov_a = cov.reindex(index=aligned.index, columns=aligned.index).fillna(0.0)
        port_ret = float((aligned * alpha).sum())
        vol = float(np.sqrt(aligned.values @ cov_a.values @ aligned.values))
        sharpe = port_ret / (vol + 1e-9)
        if sharpe > best_sharpe:
            best_sharpe = sharpe
            best_w = weights.copy()
        alpha = alpha + 0.02 * weights  # proxy alpha nudge toward portfolio solution

    assert best_w is not None
    out = _ROOT / "data" / "results" / "e2e_portfolio_training_summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    import json

    payload = {
        "epochs": args.epochs,
        "best_proxy_sharpe": best_sharpe,
        "backend": ctor.backend,
        "cvxpylayers_available": cvxpylayers_available(),
        "top_weights": best_w.nlargest(3).round(4).to_dict(),
    }
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"E2E scaffold complete -> {out}")
    if not cvxpylayers_available():
        print(
            "Install cvxpylayers + PyTorch for implicit QP gradients; "
            "current run used CVXPY delegate only."
        )


if __name__ == "__main__":
    main()
