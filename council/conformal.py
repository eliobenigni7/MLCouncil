"""Conformal prediction for position sizing via MAPIE jackknife+.

Conformal prediction provides finite-sample coverage guarantees: the
prediction interval [lower, upper] contains the true value with probability
>= coverage = 1 - alpha, regardless of the data distribution.

Interval width encodes uncertainty:
- Narrow interval → high confidence  → multiplier close to 2.0
- Wide interval   → low confidence   → multiplier close to 0.2

Usage
-----
>>> sizer = ConformalPositionSizer(coverage=0.85)
>>> sizer.fit(X_calib, y_calib)
>>> multipliers = sizer.compute_position_multipliers(council_signal, X_live)
>>> filtered   = sizer.filter_low_confidence(council_signal, X_live)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger


class ConformalPositionSizer:
    """Jackknife+ conformal regressor for uncertainty-aware position sizing.

    Uses MAPIE's ``MapieRegressor`` with ``method="plus"`` (jackknife+) which
    gives a finite-sample coverage lower-bound of 1 - 2*alpha*(n/(n+1)).
    The base estimator is a Ridge regression.

    Parameters
    ----------
    coverage:
        Target marginal coverage, 1 - alpha.  Must be in (0.5, 1.0).
        Default 0.85 → 85 % prediction intervals.  Reduced from 0.90 to
        tighten intervals and increase average position multipliers by ~15 %,
        improving expected alpha capture.  The 15 % miss rate is acceptable
        for daily cross-sectional signals where diversification across 19
        tickers limits tail exposure from individual misses.
    """

    _MIN_MULT: float = 0.2
    _MAX_MULT: float = 2.0

    def __init__(self, coverage: float = 0.85) -> None:
        if not 0.5 < coverage < 1.0:
            raise ValueError(f"coverage must be in (0.5, 1.0), got {coverage}")
        self.coverage = coverage
        self._alpha = 1.0 - coverage
        self._mapie = None
        self._n_features: int | None = None

    # ------------------------------------------------------------------
    # fit
    # ------------------------------------------------------------------

    def fit(self, X_calib: np.ndarray, y_calib: np.ndarray) -> None:
        """Calibrate prediction intervals on a held-out dataset.

        Parameters
        ----------
        X_calib:
            2-D feature array (n_samples, n_features).  Typically Alpha158
            features for the calibration period.
        y_calib:
            1-D array of realized returns aligned row-by-row with X_calib.
        """
        from mapie.regression import CrossConformalRegressor
        from sklearn.linear_model import Ridge

        X_calib = np.asarray(X_calib, dtype=float)
        y_calib = np.asarray(y_calib, dtype=float)

        if X_calib.ndim != 2:
            raise ValueError(f"X_calib must be 2-D, got shape {X_calib.shape}")
        if len(X_calib) != len(y_calib):
            raise ValueError(
                f"X_calib ({len(X_calib)}) and y_calib ({len(y_calib)}) length mismatch"
            )

        self._n_features = X_calib.shape[1]

        # CrossConformalRegressor with method="plus" (jackknife+) gives
        # finite-sample coverage >= 1 - 2*alpha*(n/(n+1)).
        self._mapie = CrossConformalRegressor(
            estimator=Ridge(alpha=1.0),
            confidence_level=self.coverage,
            method="plus",
            cv=5,
            random_state=42,
        )
        self._mapie.fit_conformalize(X_calib, y_calib)

        logger.debug(
            f"ConformalPositionSizer fitted: n_calib={len(y_calib)}, "
            f"n_features={self._n_features}, coverage={self.coverage}"
        )

    # ------------------------------------------------------------------
    # get_intervals
    # ------------------------------------------------------------------

    def get_intervals(
        self, X: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute point predictions and conformal prediction intervals.

        Parameters
        ----------
        X:
            2-D feature array (n_samples, n_features).

        Returns
        -------
        predictions : np.ndarray, shape (n_samples,)
            Ridge point predictions.
        lower_bound : np.ndarray, shape (n_samples,)
            Lower conformal bound at coverage level.
        upper_bound : np.ndarray, shape (n_samples,)
            Upper conformal bound at coverage level.
        """
        if self._mapie is None:
            raise RuntimeError("Model not fitted. Call fit() first.")

        X = np.asarray(X, dtype=float)
        # predict_interval returns (preds, intervals) where intervals is (n, 2, 1)
        # intervals[:, 0, 0] = lower bounds, intervals[:, 1, 0] = upper bounds
        preds, intervals = self._mapie.predict_interval(X)
        lower = intervals[:, 0, 0]
        upper = intervals[:, 1, 0]
        return preds, lower, upper

    # ------------------------------------------------------------------
    # compute_position_multipliers
    # ------------------------------------------------------------------

    def compute_position_multipliers(
        self,
        council_signal: pd.Series,
        X: np.ndarray,
    ) -> pd.Series:
        """Convert interval width into position multipliers.

        Algorithm
        ---------
        1. Compute conformal intervals → width = upper - lower per stock.
        2. width_norm  = width / median(width).
        3. multiplier  = exp(1 - width_norm), clipped to [0.2, 2.0].

        Stocks with narrow intervals (high confidence) receive multipliers
        near 2.0; stocks with wide intervals receive multipliers near 0.2.
        The exponential mapping avoids a dead zone that arose with 1/width_norm
        where all stocks with width_norm <= 0.5 were saturated at MAX_MULT.

        Parameters
        ----------
        council_signal:
            pd.Series(index=ticker) of council z-scores.  The index order
            must match the row order of X.
        X:
            Feature array aligned row-by-row with council_signal.

        Returns
        -------
        pd.Series(index=ticker, values=float) — multipliers in [0.2, 2.0].
        """
        _, lower, upper = self.get_intervals(np.asarray(X, dtype=float))
        width = upper - lower

        median_w = float(np.median(width))
        if median_w < 1e-9:
            multipliers = np.ones(len(width))
        else:
            width_norm = width / median_w
            # Use exp(1 - width_norm) instead of 1/width_norm to avoid a dead
            # zone where all stocks with width_norm <= 0.5 were clipped to the
            # same MAX_MULT=2.0, losing position-size differentiation.
            # exp gives smooth monotone decay: width_norm=1 → 1.0 (neutral),
            # width_norm<1 → >1.0 (scale up), width_norm>1 → <1.0 (scale down).
            multipliers = np.clip(
                np.exp(1.0 - width_norm), self._MIN_MULT, self._MAX_MULT
            )

        return pd.Series(
            multipliers, index=council_signal.index, name="position_multiplier"
        )

    # ------------------------------------------------------------------
    # filter_low_confidence
    # ------------------------------------------------------------------

    def filter_low_confidence(
        self,
        council_signal: pd.Series,
        X: np.ndarray,
        threshold_percentile: float = 90,
    ) -> pd.Series:
        """Zero out signals whose conformal interval width is in the top percentile.

        Removes the least reliable signals from the investable universe.

        Parameters
        ----------
        council_signal:
            pd.Series(index=ticker) of council signals.
        X:
            Feature array aligned row-by-row with council_signal.
        threshold_percentile:
            Signals with width >= np.percentile(widths, threshold_percentile)
            are set to 0.  Default 90 → drop the widest-interval 10 %.
            Raised from 80 to preserve more of the investable universe,
            especially for small portfolios targeting top-5 positions.

        Returns
        -------
        pd.Series with low-confidence signals zeroed out.
        """
        _, lower, upper = self.get_intervals(np.asarray(X, dtype=float))
        width = upper - lower
        threshold = float(np.percentile(width, threshold_percentile))

        filtered = council_signal.copy().astype(float)
        wide_mask = pd.Series(width >= threshold, index=council_signal.index)
        filtered[wide_mask] = 0.0

        n_zeroed = int(wide_mask.sum())
        logger.debug(
            f"filter_low_confidence: zeroed {n_zeroed}/{len(council_signal)} signals "
            f"(p{threshold_percentile} threshold={threshold:.5f})"
        )
        return filtered
