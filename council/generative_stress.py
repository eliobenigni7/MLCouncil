"""Generative stress scenario sampler (T4.3 VAE/Diffusion scaffold).

Falls back to multivariate Gaussian with regime conditioning when torch is absent.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

_TRUTHY = frozenset({"1", "true", "yes", "on"})


def generative_stress_enabled() -> bool:
    return os.getenv("MLCOUNCIL_GENERATIVE_STRESS", "").strip().lower() in _TRUTHY


@dataclass
class GenerativeStressResult:
    scenarios: np.ndarray
    method: str
    n_scenarios: int
    var_95: float

    def summary(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "n_scenarios": self.n_scenarios,
            "var_95": self.var_95,
            "scenario_mean": float(np.mean(self.scenarios)),
            "scenario_std": float(np.std(self.scenarios)),
        }


class GenerativeStressEngine:
    """Sample multivariate return scenarios for Monte Carlo VaR augmentation."""

    def __init__(self, *, n_scenarios: int = 10_000, random_state: int = 42) -> None:
        self.n_scenarios = n_scenarios
        self.random_state = random_state

    def sample_scenarios(
        self,
        returns_wide: pd.DataFrame,
        *,
        regime_scale: float = 1.0,
    ) -> GenerativeStressResult:
        if returns_wide.empty:
            raise ValueError("returns_wide is empty")

        tail = returns_wide.dropna(how="all").tail(252)
        mu = tail.mean().values
        cov = tail.cov().values
        cov = cov + np.eye(len(mu)) * 1e-6
        rng = np.random.default_rng(self.random_state)

        method = "gaussian_fallback"
        try:
            import torch  # noqa: F401

            method = "generative_stub"
        except ImportError:
            pass

        draws = rng.multivariate_normal(
            mu * regime_scale,
            cov * (regime_scale**2),
            size=self.n_scenarios,
        )
        portfolio_returns = draws.mean(axis=1)
        var_95 = float(np.quantile(portfolio_returns, 0.05))
        logger.info(
            f"Generative stress ({method}): n={self.n_scenarios} var_95={var_95:.4f}"
        )
        return GenerativeStressResult(
            scenarios=portfolio_returns,
            method=method,
            n_scenarios=self.n_scenarios,
            var_95=var_95,
        )
