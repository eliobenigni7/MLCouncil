"""Mini-spike: robust mean-variance vs plain MV on synthetic alpha/cov."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

RESULTS = ROOT / "artifacts" / "spikes" / "spike_robust_opt.json"


def _solve(alpha: np.ndarray, cov: np.ndarray, kappa: float) -> np.ndarray | None:
    try:
        import cvxpy as cp
    except ModuleNotFoundError:
        return None

    n = len(alpha)
    w = cp.Variable(n)
    risk = cp.quad_form(w, cov)
    objective = cp.Maximize(alpha @ w - kappa * cp.sqrt(risk))
    constraints = [cp.sum(w) == 1.0, w >= 0.0, w <= 0.25]
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.SCS, verbose=False)
    if w.value is None:
        return None
    return np.clip(w.value, 0.0, None)


def main() -> None:
    rng = np.random.default_rng(7)
    n = 6
    alpha = rng.standard_normal(n)
    a = rng.standard_normal((60, n))
    cov = np.cov(a, rowvar=False) + np.eye(n) * 1e-4

    mv_w = _solve(alpha, cov, kappa=0.0)
    scans = []
    for kappa in [0.0, 0.5, 1.0, 2.0, 5.0]:
        w = _solve(alpha, cov, kappa=kappa)
        if w is None:
            continue
        port_vol = float(np.sqrt(w @ cov @ w))
        scans.append({"kappa": kappa, "weights": w.tolist(), "vol": port_vol})

    payload = {
        "mv_weights": mv_w.tolist() if mv_w is not None else [],
        "robust_scan": scans,
        "covariance_condition_number": float(np.linalg.cond(cov)),
        "recommendation": "go" if mv_w is not None and len(scans) >= 3 else "no-go",
    }
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    RESULTS.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
