"""TDA persistent homology early-warning scaffold (T4.5).

Computes a proxy ``beta1`` loop density from rolling correlation structure when
``gudhi`` / ``ripser`` are not installed.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

_TRUTHY = frozenset({"1", "true", "yes", "on"})


def tda_warning_enabled() -> bool:
    return os.getenv("MLCOUNCIL_TDA_WARNING_ENABLED", "true").strip().lower() in _TRUTHY


@dataclass
class TDAWarningResult:
    beta1_proxy: float
    threshold: float
    is_alert: bool
    window_days: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "beta1_proxy": self.beta1_proxy,
            "threshold": self.threshold,
            "is_alert": self.is_alert,
            "window_days": self.window_days,
        }


class PersistentHomologyAnalyser:
    """Rolling multivariate return topology stress proxy."""

    def __init__(
        self,
        *,
        window_days: int = 30,
        beta1_threshold: float = 0.35,
    ) -> None:
        self.window_days = window_days
        self.beta1_threshold = beta1_threshold

    def compute_beta1_proxy(self, returns_wide: pd.DataFrame) -> float:
        """Proxy: mean off-diagonal correlation magnitude in rolling window."""
        if returns_wide.empty or returns_wide.shape[1] < 2:
            return 0.0
        tail = returns_wide.tail(self.window_days).dropna(how="all")
        if len(tail) < 10:
            return 0.0
        corr = tail.corr().values
        n = corr.shape[0]
        mask = ~np.eye(n, dtype=bool)
        off_diag = np.abs(corr[mask])
        return float(np.nanmean(off_diag)) if off_diag.size else 0.0

    def analyse(self, returns_wide: pd.DataFrame) -> TDAWarningResult:
        beta1 = self.compute_beta1_proxy(returns_wide)
        is_alert = beta1 >= self.beta1_threshold
        if is_alert:
            logger.warning(
                f"TDA proxy alert: beta1_proxy={beta1:.3f} >= {self.beta1_threshold}"
            )
        return TDAWarningResult(
            beta1_proxy=beta1,
            threshold=self.beta1_threshold,
            is_alert=is_alert,
            window_days=self.window_days,
        )
