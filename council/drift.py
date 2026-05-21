"""Streaming drift detectors for daily return monitoring (River ADWIN / DDM).

ADWIN runs on a rolling window of equal-weight portfolio daily returns
(default 60 business days). DDM (Page-Hinckley) provides a complementary
concept-drift signal for scheduling heavy walk-forward retrains.
"""

from __future__ import annotations

from collections import deque
from typing import Deque

import pandas as pd


class ADWINDetector:
    """ADWIN drift detector with bounded rolling history."""

    def __init__(
        self,
        *,
        delta: float = 0.002,
        window_days: int = 60,
    ) -> None:
        from river import drift

        self._detector = drift.ADWIN(delta=delta)
        self._window_days = window_days
        self._buffer: Deque[float] = deque(maxlen=window_days)
        self.drift_detected: bool = False

    def update(self, value: float) -> bool:
        """Ingest one observation; return True if ADWIN reports drift."""
        self._buffer.append(float(value))
        self._detector.update(float(value))
        self.drift_detected = bool(getattr(self._detector, "drift_detected", False))
        return self.drift_detected

    def update_series(self, returns: pd.Series) -> bool:
        """Feed a return series; return True if drift detected on final point."""
        if returns.empty:
            return False
        detected = False
        for val in returns.astype(float).values:
            detected = self.update(val)
        return detected

    @property
    def window_size(self) -> int:
        return len(self._buffer)


class DDMDetector:
    """Drift Detection Method (DDM) via ``river.drift.binary.DDM``.

    Expects a stream of binary error indicators (0 = correct, 1 = error).
    For continuous returns, use ``returns_to_error_indicators()`` first.
    """

    def __init__(
        self,
        *,
        warm_start: int = 30,
        warning_threshold: float = 2.0,
        drift_threshold: float = 3.0,
    ) -> None:
        from river.drift.binary import DDM

        self._detector = DDM(
            warm_start=warm_start,
            warning_threshold=warning_threshold,
            drift_threshold=drift_threshold,
        )
        self.in_warning: bool = False
        self.drift_detected: bool = False

    @staticmethod
    def returns_to_error_indicators(returns: pd.Series) -> pd.Series:
        """Map daily returns to binary miss indicators (negative return = 1)."""
        return (returns.astype(float) < 0).astype(int)

    def update(self, value: float | int) -> bool:
        """Ingest one binary error indicator; return True on DDM drift."""
        self._detector.update(int(value))
        self.in_warning = bool(getattr(self._detector, "in_warning_zone", False))
        self.drift_detected = bool(getattr(self._detector, "drift_detected", False))
        return self.drift_detected

    def update_series(self, returns: pd.Series) -> bool:
        if returns.empty:
            return False
        detected = False
        for val in returns.astype(float).values:
            detected = self.update(val)
        return detected
