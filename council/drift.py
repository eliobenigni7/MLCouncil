"""Streaming drift detectors for daily return monitoring (pure Python).

ADWIN runs on a rolling window of equal-weight portfolio daily returns
(default 60 business days). DDM (Page-Hinckley) provides a complementary
concept-drift signal for scheduling heavy walk-forward retrains.

Pure Python implementations — no river dependency, ARM64-compatible.
"""

from __future__ import annotations

import math
from collections import deque
from typing import Deque

import pandas as pd


class _ADWINBucket:
    """Single bucket in the ADWIN histogram."""

    __slots__ = ("total", "variance", "size")

    def __init__(self, total: float, variance: float, size: int) -> None:
        self.total = total
        self.variance = variance
        self.size = size


class ADWINDetector:
    """ADWIN drift detector with bounded rolling history.

    Pure Python reimplementation of the Adaptive Windowing algorithm
    (Bifet & Gavaldà, 2007). No river dependency.
    """

    def __init__(self, *, delta: float = 0.002, window_days: int = 60) -> None:
        if not (0 < delta < 1):
            raise ValueError("delta must be in (0, 1)")
        self._delta = delta
        self._window_days = window_days
        self._buckets: list[_ADWINBucket] = []
        self._total: float = 0.0
        self._variance: float = 0.0
        self._width: int = 0
        self._bucket_size: int = 1  # grows adaptively
        self._buffer: Deque[float] = deque(maxlen=window_days)
        self.drift_detected: bool = False

    def update(self, value: float) -> bool:
        """Ingest one observation; return True if ADWIN reports drift."""
        v = float(value)
        self._buffer.append(v)
        self._total += v
        self._variance += v * v
        self._width += 1

        # Adaptive bucket sizing
        if self._width % self._bucket_size == 0:
            mean = self._total / self._width
            var = max(0.0, (self._variance / self._width) - (mean * mean))
            self._buckets.append(_ADWINBucket(mean * self._bucket_size, var * self._bucket_size, self._bucket_size))
            if self._width >= 2:
                self._bucket_size = min(max(1, self._width // 20), 100)

        # Check for drift by scanning cut points
        self.drift_detected = self._detect_drift()
        if self.drift_detected:
            self._drop_oldest_bucket()
        return self.drift_detected

    def _detect_drift(self) -> bool:
        """Scan bucket cut points for a significant mean shift."""
        n_buckets = len(self._buckets)
        if n_buckets < 4:
            return False

        total_right = self._total
        width_right = self._width
        var_right = self._variance

        total_left = 0.0
        width_left = 0
        var_left = 0.0

        delta_prime = self._delta / math.log2(max(2, n_buckets))

        for i in range(n_buckets - 1):
            b = self._buckets[i]
            total_left += b.total
            width_left += b.size
            var_left += b.variance
            total_right -= b.total
            width_right -= b.size
            var_right -= b.variance

            if width_left < 2 or width_right < 2:
                continue

            mean_left = total_left / width_left
            mean_right = total_right / width_right

            # Pooled variance estimate
            pooled_var_left = max(0.0, (var_left / width_left) - (mean_left * mean_left))
            pooled_var_right = max(0.0, (var_right / width_right) - (mean_right * mean_right))

            # Harmonic mean of widths
            m = 1.0 / ((1.0 / width_left) + (1.0 / width_right))

            epsilon_cut = math.sqrt(2.0 * m * pooled_var_left * delta_prime) + (2.0 / (3.0 * m)) * delta_prime
            if abs(mean_left - mean_right) > epsilon_cut:
                return True

        return False

    def _drop_oldest_bucket(self) -> None:
        """Evict the oldest bucket after drift detection."""
        if not self._buckets:
            return
        oldest = self._buckets.pop(0)
        self._total -= oldest.total
        self._width -= oldest.size
        self._variance -= oldest.variance

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
    """Drift Detection Method (DDM) — Page-Hinckley on binary error stream.

    Pure Python reimplementation. No river dependency.
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
        self._warm_start = warm_start
        self._warning_threshold = warning_threshold
        self._drift_threshold = drift_threshold
        self._n: int = 0
        self._p_min: float = float("inf")
        self._s_min: float = float("inf")
        self._sum_: float = 0.0
        self.in_warning: bool = False
        self.drift_detected: bool = False

    @staticmethod
    def returns_to_error_indicators(returns: pd.Series) -> pd.Series:
        """Map daily returns to binary miss indicators (negative return = 1)."""
        return (returns.astype(float) < 0).astype(int)

    def update(self, value: float | int) -> bool:
        """Ingest one binary error indicator; return True on DDM drift."""
        self._n += 1
        error = int(value)
        self._sum_ += error
        p_i = self._sum_ / self._n

        if self._n < self._warm_start:
            return False

        if p_i + math.sqrt(p_i * (1 - p_i) / self._n) < self._p_min:
            self._p_min = p_i + math.sqrt(p_i * (1 - p_i) / self._n)
            self._s_min = math.sqrt(p_i * (1 - p_i) / self._n)

        if self._s_min == float("inf"):
            return False

        warning_level = self._p_min + self._warning_threshold * self._s_min
        drift_level = self._p_min + self._drift_threshold * self._s_min

        if p_i + math.sqrt(p_i * (1 - p_i) / self._n) > drift_level:
            self.drift_detected = True
            self.in_warning = False
            return True

        self.in_warning = p_i + math.sqrt(p_i * (1 - p_i) / self._n) > warning_level
        return False

    def update_series(self, returns: pd.Series) -> bool:
        if returns.empty:
            return False
        detected = False
        for val in returns.astype(float).values:
            detected = self.update(val)
        return detected
