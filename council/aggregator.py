"""Council aggregator: weighted ensemble of model signals with adaptive weighting.

Architecture
------------
* Base weights per market regime are loaded from ``config/regime_weights.yaml``.
* After ``min_history_days`` of observed IC history, weights are scaled by each
  model's rolling 60-day Information Ratio (mean IC / std IC * sqrt(252)).
* Models with consistently negative Sharpe are down-weighted toward their floor
  (``weight_clip.min``); no model can exceed ``weight_clip.max``.
* Every call to ``aggregate()`` is logged (weights + contributions) so that
  ``get_attribution()`` can reconstruct per-model P&L attribution.

Typical daily flow
------------------
    T:    aggregator.aggregate(signals, regime, date=today) → council signal
    T+1:  aggregator.update_performance(signals_history, returns_history, date=today)
"""

from __future__ import annotations

from datetime import date
from typing import Any

import numpy as np
import pandas as pd
import yaml
from loguru import logger
from scipy.stats import spearmanr


# ---------------------------------------------------------------------------
# Default config (fallback when YAML is missing, e.g. in tests)
# ---------------------------------------------------------------------------

_DEFAULT_CONFIG: dict[str, Any] = {
    "regime_weights": {
        "bull":       {"lgbm": 0.50, "sentiment": 0.30, "hmm": 0.20},
        "bear":       {"lgbm": 0.40, "sentiment": 0.20, "hmm": 0.40},
        "transition": {"lgbm": 0.45, "sentiment": 0.25, "hmm": 0.30},
    },
    "weight_clip": {"min": 0.05, "max": 0.70},
    "performance": {
        "min_history_days":    30,
        "ic_rolling_window":   30,
        "sharpe_rolling_window": 60,
    },
}


# ---------------------------------------------------------------------------
# CouncilAggregator
# ---------------------------------------------------------------------------

class CouncilAggregator:
    """Aggregate signals from N specialised models into a single council signal.

    The weight of each model depends on the current market regime and its
    rolling Sharpe over the last 60 days of IC observations.
    """

    def __init__(self, config_path: str = "config/regime_weights.yaml") -> None:
        try:
            with open(config_path) as f:
                cfg: dict[str, Any] = yaml.safe_load(f)
        except FileNotFoundError:
            logger.warning(
                f"Config not found at {config_path!r}; using built-in defaults."
            )
            cfg = _DEFAULT_CONFIG

        self._base_weights: dict[str, dict[str, float]] = cfg["regime_weights"]
        self._min_weight: float = cfg["weight_clip"]["min"]
        self._max_weight: float = cfg["weight_clip"]["max"]
        self._min_history: int = cfg["performance"]["min_history_days"]
        self._ic_window: int = cfg["performance"]["ic_rolling_window"]
        self._sharpe_window: int = cfg["performance"]["sharpe_rolling_window"]

        # model_name → {date: ic_value}  (populated by update_performance)
        self._ic_by_date: dict[str, dict[date, float]] = {}

        # date → {"weights": {...}, "regime": str, "contributions": {...}}
        self._weights_log: dict[date, dict[str, Any]] = {}

    # ------------------------------------------------------------------
    # aggregate
    # ------------------------------------------------------------------

    def aggregate(
        self,
        signals: dict[str, pd.Series],
        regime: str,
        date: date,
    ) -> pd.Series:
        """Combine model signals into a single z-scored council signal.

        Parameters
        ----------
        signals:
            Mapping model_name → pd.Series(index=ticker, values=float z-score).
        regime:
            Current market regime label: "bull", "bear", or "transition".
        date:
            Valuation date — used for logging and attribution lookup.

        Returns
        -------
        pd.Series(index=ticker, values=council_signal)
            Cross-sectional z-score across all active tickers.
        """
        base_weights = self._base_weights.get(
            regime, self._base_weights.get("transition", {})
        )

        # Only consider models that have both a configured weight and a live signal
        active_models = [m for m in base_weights if m in signals]
        if not active_models:
            logger.warning(
                f"aggregate() [{date}]: no overlap between config models "
                f"{list(base_weights)} and provided signals {list(signals)}. "
                "Using equal-weight fallback over all provided signals."
            )
            active_models = list(signals.keys())
            uw = 1.0 / max(len(active_models), 1)
            base_weights = {m: uw for m in active_models}

        # Adaptive weighting once sufficient IC history is available
        if self._has_sufficient_history(active_models):
            sharpe = self._compute_rolling_sharpe(active_models)
            weights = self._adjust_weights_for_performance(
                {m: base_weights[m] for m in active_models}, sharpe
            )
        else:
            raw = {m: base_weights[m] for m in active_models}
            total = sum(raw.values()) or 1.0
            weights = {m: v / total for m, v in raw.items()}

        # Union of all tickers across active model signals
        all_tickers: set[str] = set()
        for m in active_models:
            all_tickers.update(signals[m].index.tolist())
        tickers = sorted(all_tickers)

        # Weighted average
        combined = pd.Series(0.0, index=tickers)
        contributions: dict[str, float] = {}
        for model in active_models:
            sig = signals[model].reindex(tickers).fillna(0.0)
            combined += weights[model] * sig
            contributions[model] = float((weights[model] * sig).mean())

        # Cross-sectional z-score
        std = float(combined.std())
        if std > 1e-9:
            combined = (combined - combined.mean()) / std

        # Log for attribution
        self._weights_log[date] = {
            "weights": weights.copy(),
            "regime": regime,
            "contributions": contributions,
        }
        logger.debug(
            f"[{date}] regime={regime} weights={weights} "
            f"n_tickers={len(tickers)} active_models={active_models}"
        )

        combined.index.name = "ticker"
        return combined

    # ------------------------------------------------------------------
    # update_performance
    # ------------------------------------------------------------------

    def update_performance(
        self,
        signals_history: dict[str, pd.DataFrame],
        returns_history: pd.DataFrame,
        date: date,
    ) -> None:
        """Update IC history after T+1 returns become available.

        Called daily after realized returns are known.  Computes the
        cross-sectional IC (Spearman rank correlation) between each
        model's signal and the realized returns for each date in
        ``signals_history`` / ``returns_history``.

        Parameters
        ----------
        signals_history:
            model_name → pd.DataFrame(index=date, columns=ticker).
        returns_history:
            pd.DataFrame(index=date, columns=ticker) of realized returns.
        date:
            Valuation date of the update call (for logging).
        """
        for model_name, signals_df in signals_history.items():
            if model_name not in self._ic_by_date:
                self._ic_by_date[model_name] = {}

            common_dates = signals_df.index.intersection(returns_history.index)
            for d in common_dates:
                if d in self._ic_by_date[model_name]:
                    continue  # already computed — skip

                sig = signals_df.loc[d].dropna()
                ret = returns_history.loc[d].dropna()
                common_tickers = sig.index.intersection(ret.index)
                if len(common_tickers) < 3:
                    continue

                ic_val, _ = spearmanr(
                    sig[common_tickers].values, ret[common_tickers].values
                )
                if not np.isnan(ic_val):
                    self._ic_by_date[model_name][d] = float(ic_val)

        # Back-fill contributions in the weights log with IC × weight
        for log_date, log_entry in self._weights_log.items():
            for model_name, w in log_entry["weights"].items():
                ic = self._ic_by_date.get(model_name, {}).get(log_date, np.nan)
                log_entry["contributions"][model_name] = w * ic if not np.isnan(ic) else np.nan

        logger.debug(
            f"[{date}] update_performance: IC history updated for "
            f"{list(signals_history.keys())}"
        )

    # ------------------------------------------------------------------
    # get_attribution
    # ------------------------------------------------------------------

    def get_attribution(self, date: date) -> pd.DataFrame:
        """Per-model attribution for a given valuation date.

        Returns
        -------
        pd.DataFrame with columns:
            model_name, weight, ic_rolling_30d, sharpe_rolling_60d, pnl_contribution
        """
        log = self._weights_log.get(date, {})
        weights_used = log.get("weights", {})
        contributions = log.get("contributions", {})

        all_models = sorted(
            set(weights_used.keys()) | set(self._ic_by_date.keys())
        )

        rows = []
        for model_name in all_models:
            ic_entries = sorted(self._ic_by_date.get(model_name, {}).items())
            recent_30 = [v for _, v in ic_entries[-self._ic_window:]]
            recent_60 = [v for _, v in ic_entries[-self._sharpe_window:]]

            ic_30d = (
                float(np.mean(recent_30))
                if len(recent_30) >= self._ic_window
                else np.nan
            )
            sharpe_60d = (
                float(
                    np.mean(recent_60) / (np.std(recent_60) + 1e-9) * np.sqrt(252)
                )
                if len(recent_60) >= 2
                else np.nan
            )

            rows.append(
                {
                    "model_name": model_name,
                    "weight": weights_used.get(model_name, np.nan),
                    "ic_rolling_30d": ic_30d,
                    "sharpe_rolling_60d": sharpe_60d,
                    "pnl_contribution": contributions.get(model_name, np.nan),
                }
            )

        return pd.DataFrame(
            rows,
            columns=[
                "model_name",
                "weight",
                "ic_rolling_30d",
                "sharpe_rolling_60d",
                "pnl_contribution",
            ],
        )

    # ------------------------------------------------------------------
    # _adjust_weights_for_performance
    # ------------------------------------------------------------------

    def _adjust_weights_for_performance(
        self,
        base_weights: dict[str, float],
        sharpe_rolling: dict[str, float],
    ) -> dict[str, float]:
        """Scale base weights by max(0, Sharpe), clip, then renormalize.

        Algorithm
        ---------
        1. adjusted[m] = base_weight[m] * max(0, sharpe_rolling[m])
        2. Normalize adjusted weights to sum 1.
        3. Clip each weight to [min_weight, max_weight].
        4. Renormalize to sum 1.

        Falls back to renormalized base weights when every model has
        non-positive rolling Sharpe (avoids an all-zero weight vector).
        """
        adjusted = {
            m: w * max(0.0, sharpe_rolling.get(m, 0.0))
            for m, w in base_weights.items()
        }
        total = sum(adjusted.values())

        if total < 1e-9:
            # All models have non-positive Sharpe — fall back to base weights
            total_base = sum(base_weights.values()) or 1.0
            return {m: v / total_base for m, v in base_weights.items()}

        # Normalize, then clip to [min_weight, max_weight]
        normalized = {m: v / total for m, v in adjusted.items()}
        clipped = {
            m: max(self._min_weight, min(self._max_weight, v))
            for m, v in normalized.items()
        }

        # Final renormalization after clipping
        total_clipped = sum(clipped.values()) or 1.0
        return {m: v / total_clipped for m, v in clipped.items()}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _has_sufficient_history(self, models: list[str]) -> bool:
        """True if at least one model has >= min_history_days of IC entries."""
        return any(
            len(self._ic_by_date.get(m, {})) >= self._min_history
            for m in models
        )

    def _compute_rolling_sharpe(self, models: list[str]) -> dict[str, float]:
        """Rolling Sharpe (60-day IC mean / std * sqrt(252)) per model."""
        result: dict[str, float] = {}
        for m in models:
            entries = sorted(self._ic_by_date.get(m, {}).items())
            recent = [v for _, v in entries[-self._sharpe_window:]]
            if len(recent) >= 2:
                result[m] = (
                    float(np.mean(recent))
                    / (float(np.std(recent)) + 1e-9)
                    * np.sqrt(252)
                )
            else:
                result[m] = 0.0
        return result
