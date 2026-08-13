"""PCMCI-style causal graph drift detector (T4.4 shadow scaffold).

Uses lightweight correlation-based graph proxy when ``tigramite`` is unavailable.
Production monitor integrates via :func:`PCMCIDriftDetector.check`.
Canary status: shadow — target: P-2 — expiry: 2027-12-01 (promote via canary o retire)
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

_TRUTHY = frozenset({"1", "true", "yes", "on"})


def causal_drift_enabled() -> bool:
    return os.getenv("MLCOUNCIL_CAUSAL_DRIFT_ENABLED", "").strip().lower() in _TRUTHY


@dataclass
class CausalGraphSnapshot:
    """Adjacency summary for feature → return links."""

    links: set[tuple[str, str]] = field(default_factory=set)
    threshold: float = 0.15

    def to_dict(self) -> dict[str, Any]:
        return {
            "link_count": len(self.links),
            "links": sorted([f"{a}->{b}" for a, b in self.links]),
            "threshold": self.threshold,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "CausalGraphSnapshot":
        """Rebuild a snapshot from ``to_dict()`` output (graceful on garbage)."""
        if not isinstance(data, dict):
            return cls()
        links: set[tuple[str, str]] = set()
        for link in data.get("links", []) or []:
            if isinstance(link, str) and "->" in link:
                a, b = link.split("->", 1)
                links.add((a.strip(), b.strip()))
        threshold = 0.15
        try:
            threshold = float(data.get("threshold", threshold))
        except (TypeError, ValueError):
            threshold = 0.15
        return cls(links=links, threshold=threshold)


class PCMCIDriftDetector:
    """Detect structural changes in feature-return dependency graph."""

    def __init__(
        self,
        *,
        corr_threshold: float = 0.15,
        min_samples: int = 60,
        link_change_fraction: float = 0.25,
    ) -> None:
        self.corr_threshold = corr_threshold
        self.min_samples = min_samples
        self.link_change_fraction = link_change_fraction
        self._baseline: CausalGraphSnapshot | None = None
        self.last_diagnostics: dict[str, Any] | None = None

    def fit_baseline(self, features: pd.DataFrame, returns: pd.Series) -> CausalGraphSnapshot:
        self._baseline = self._build_graph(features, returns)
        return self._baseline

    @property
    def baseline(self) -> CausalGraphSnapshot | None:
        """The currently installed baseline snapshot (None before first check)."""
        return self._baseline

    def set_baseline(self, snapshot: CausalGraphSnapshot | None) -> None:
        """Install a persisted baseline snapshot (e.g. from a previous run)."""
        self._baseline = snapshot

    def _build_graph(self, features: pd.DataFrame, returns: pd.Series) -> CausalGraphSnapshot:
        aligned = features.copy()
        aligned["__ret__"] = returns.reindex(features.index).values
        aligned = aligned.dropna()
        if len(aligned) < self.min_samples:
            return CausalGraphSnapshot(threshold=self.corr_threshold)

        links: set[tuple[str, str]] = set()
        ret = aligned["__ret__"]
        for col in features.columns:
            if col == "__ret__":
                continue
            corr = float(aligned[col].corr(ret))
            if abs(corr) >= self.corr_threshold:
                links.add((col, "forward_return"))
        return CausalGraphSnapshot(links=links, threshold=self.corr_threshold)

    def check(
        self,
        features: pd.DataFrame,
        returns: pd.Series,
    ) -> tuple[bool, dict[str, Any]]:
        """Return (is_alert, diagnostics)."""
        current = self._build_graph(features, returns)
        if self._baseline is None:
            self._baseline = current
            self.last_diagnostics = {"status": "baseline_initialized", **current.to_dict()}
            return False, self.last_diagnostics

        base_links = self._baseline.links
        cur_links = current.links
        if not base_links:
            self.last_diagnostics = {"status": "empty_baseline", **current.to_dict()}
            return False, self.last_diagnostics

        added = cur_links - base_links
        removed = base_links - cur_links
        change_frac = (len(added) + len(removed)) / max(len(base_links), 1)
        is_alert = change_frac >= self.link_change_fraction
        diag = {
            "status": "alert" if is_alert else "ok",
            "change_fraction": change_frac,
            "links_added": len(added),
            "links_removed": len(removed),
            "baseline_link_count": len(base_links),
            "current_link_count": len(cur_links),
        }
        self.last_diagnostics = diag
        if is_alert:
            logger.warning(f"PCMCI proxy drift: {diag}")
        return is_alert, diag


# ---------------------------------------------------------------------------
# Baseline persistence (weekly asset keeps the baseline across runs)
# ---------------------------------------------------------------------------

def save_causal_baseline(path: str | Path, snapshot: CausalGraphSnapshot | None) -> None:
    """Persist the baseline snapshot as JSON for the next weekly run."""
    if snapshot is None:
        return
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(snapshot.to_dict(), indent=2), encoding="utf-8")


def load_causal_baseline(path: str | Path) -> CausalGraphSnapshot | None:
    """Load a persisted baseline; returns None when missing or malformed."""
    p = Path(path)
    if not p.exists():
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    return CausalGraphSnapshot.from_dict(data)
