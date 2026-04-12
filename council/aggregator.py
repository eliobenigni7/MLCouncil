"""Council aggregator: weighted ensemble of model signals with adaptive weighting.

Architecture
------------
* Base weights per market regime are loaded from ``config/regime_weights.yaml``.
* After ``min_history_days`` of observed IC history, weights are scaled by each
  model's rolling 100-day Information Ratio (mean IC / std IC * sqrt(252)).
* Models with consistently negative Sharpe are down-weighted toward their floor
  (``weight_clip.min``); no model can exceed ``weight_clip.max``.
* Orthogonality constraints: correlated models are down-weighted to maintain
  maximum pairwise correlation below threshold.
* Every call to ``aggregate()`` is logged (weights + contributions) so that
  ``get_attribution()`` can reconstruct per-model P&L attribution.

Typical daily flow
------------------
    T:    aggregator.aggregate(signals, regime, date=today) → council signal
    T+1:  aggregator.update_performance(signals_history, returns_history, date=today)
"""

from __future__ import annotations

from datetime import date
from typing import Any, Optional

import numpy as np
import pandas as pd
import yaml
from loguru import logger
from scipy.stats import spearmanr


DEFAULT_CONFIG: dict[str, Any] = {
    "regime_weights": {
        "bull":       {"lgbm": 0.50, "sentiment": 0.30, "hmm": 0.20},
        "bear":       {"lgbm": 0.40, "sentiment": 0.20, "hmm": 0.40},
        "transition": {"lgbm": 0.45, "sentiment": 0.25, "hmm": 0.30},
    },
    "weight_clip": {"min": 0.05, "max": 0.70},
    "performance": {
        "min_history_days":    30,
        "ic_rolling_window":   30,
        # 60 days is too noisy for equity IC-Sharpe; 100 days is the minimum
        # recommended window for stable Sharpe estimation in the literature.
        "sharpe_rolling_window": 100,
    },
    "orthogonality": {
        "max_correlation": 0.70,
        "correlation_window": 60,
        "auto_downweight": True,
        "downweight_factor": 0.5,
    },
}


class OrthogonalityMonitor:
    """Monitor and enforce orthogonality between model signals.

    Tracks rolling correlations between model signals and automatically
    downweights models that become too correlated with others.
    """

    def __init__(
        self,
        max_correlation: float = 0.70,
        correlation_window: int = 60,
        auto_downweight: bool = True,
        downweight_factor: float = 0.5,
    ):
        self.max_correlation = max_correlation
        self.correlation_window = correlation_window
        self.auto_downweight = auto_downweight
        self.downweight_factor = downweight_factor
        self._signal_history: dict[str, pd.DataFrame] = {}
        self._correlation_alerts: list[dict] = []

    def update_signals(
        self,
        signals: dict[str, pd.Series],
        date: date,
    ) -> None:
        for model_name, sig in signals.items():
            if model_name not in self._signal_history:
                self._signal_history[model_name] = pd.DataFrame(columns=sig.index)
            df = self._signal_history[model_name].copy()
            df = df.reindex(columns=df.columns.union(sig.index, sort=False))
            df.loc[date, sig.index] = sig.values
            # Trim to correlation_window rows to bound memory growth.
            if len(df) > self.correlation_window:
                df = df.iloc[-self.correlation_window:]
            self._signal_history[model_name] = df

    def compute_correlation_matrix(
        self,
        models: list[str],
        end_date: Optional[date] = None,
    ) -> pd.DataFrame:
        if len(models) < 2:
            return pd.DataFrame()

        series_list = []
        for model in models:
            if model in self._signal_history:
                df = self._signal_history[model]
                if end_date and end_date in df.index:
                    df = df.loc[:end_date]
                if len(df) >= 10:
                    series_list.append(df.mean(axis=1).tail(self.correlation_window))

        if len(series_list) < 2:
            return pd.DataFrame()

        combined = pd.concat(series_list, axis=1)
        combined.columns = models[: len(series_list)]
        return combined.corr()

    def get_correlated_pairs(
        self,
        models: list[str],
        end_date: Optional[date] = None,
    ) -> list[tuple[str, str, float]]:
        corr_matrix = self.compute_correlation_matrix(models, end_date)
        if corr_matrix.empty:
            return []

        correlated = []
        for i, model1 in enumerate(corr_matrix.columns):
            for j, model2 in enumerate(corr_matrix.columns):
                if i < j:
                    corr = corr_matrix.loc[model1, model2]
                    if abs(corr) > self.max_correlation:
                        correlated.append((model1, model2, corr))

        return correlated

    def compute_orthogonality_penalty(
        self,
        models: list[str],
        base_weights: dict[str, float],
        end_date: Optional[date] = None,
    ) -> dict[str, float]:
        penalties = {m: 1.0 for m in models}
        correlated = self.get_correlated_pairs(models, end_date)

        if not correlated:
            return penalties

        for model1, model2, corr in correlated:
            if model1 in base_weights and model2 in base_weights:
                if base_weights[model1] >= base_weights[model2]:
                    penalties[model2] *= self.downweight_factor
                else:
                    penalties[model1] *= self.downweight_factor

                self._correlation_alerts.append({
                    "date": end_date,
                    "model1": model1,
                    "model2": model2,
                    "correlation": corr,
                    "action": "downweighted",
                })

        return penalties

    def get_orthogonality_report(
        self,
        models: list[str],
        end_date: Optional[date] = None,
    ) -> dict:
        corr_matrix = self.compute_correlation_matrix(models, end_date)
        correlated = self.get_correlated_pairs(models, end_date)

        if corr_matrix.empty:
            return {"status": "insufficient_data"}

        upper_tri = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        correlations = upper_tri.stack().values

        return {
            "status": "ok" if not correlated else "correlated",
            "max_correlation": float(correlations.max()) if len(correlations) > 0 else 0.0,
            "mean_correlation": float(correlations.mean()) if len(correlations) > 0 else 0.0,
            "correlated_pairs": [
                {"model1": m1, "model2": m2, "correlation": corr}
                for m1, m2, corr in correlated
            ],
            "n_alerts": len(self._correlation_alerts),
        }


class CouncilAggregator:
    """Aggregate signals from N specialised models into a single council signal.

    The weight of each model depends on:
    1. Current market regime (base weights from config)
    2. Rolling Sharpe over the last 60 days of IC observations
    3. Orthogonality constraints (correlation-based downweighting)

    Supports up to 5 alpha models:
    - lgbm: LightGBM technical model
    - sentiment: FinBERT sentiment model
    - short_interest: FINRA short interest model
    - earnings_nlp: EDGAR earnings NLP model
    - hmm: Hidden Markov Model regime
    """

    def __init__(
        self,
        config_path: str = "config/regime_weights.yaml",
        use_orthogonality: bool = True,
    ) -> None:
        try:
            with open(config_path) as f:
                cfg: dict[str, Any] = yaml.safe_load(f)
        except FileNotFoundError:
            logger.warning(
                f"Config not found at {config_path!r}; using built-in defaults."
            )
            cfg = DEFAULT_CONFIG

        self._base_weights: dict[str, dict[str, float]] = cfg["regime_weights"]
        self._min_weight: float = cfg["weight_clip"]["min"]
        self._max_weight: float = cfg["weight_clip"]["max"]
        self._min_history: int = cfg["performance"]["min_history_days"]
        self._ic_window: int = cfg["performance"]["ic_rolling_window"]
        self._sharpe_window: int = cfg["performance"]["sharpe_rolling_window"]

        ortho_cfg = cfg.get("orthogonality", DEFAULT_CONFIG["orthogonality"])
        self._ortho_monitor = OrthogonalityMonitor(
            max_correlation=ortho_cfg.get("max_correlation", 0.70),
            correlation_window=ortho_cfg.get("correlation_window", 60),
            auto_downweight=ortho_cfg.get("auto_downweight", True),
            downweight_factor=ortho_cfg.get("downweight_factor", 0.5),
        ) if use_orthogonality else None

        self._ic_by_date: dict[str, dict[date, float]] = {}
        self._weights_log: dict[date, dict[str, Any]] = {}

    def aggregate(
        self,
        signals: dict[str, pd.Series],
        regime: str,
        date: date,
    ) -> pd.Series:
        base_weights = self._base_weights.get(
            regime, self._base_weights.get("transition", {})
        )

        active_models = [m for m in base_weights if m in signals]
        if not active_models:
            logger.warning(
                f"aggregate() [{date}]: no overlap between config models "
                f"{list(base_weights)} and provided signals {list(signals)}. "
                "Using equal-weight fallback."
            )
            active_models = list(signals.keys())
            uw = 1.0 / max(len(active_models), 1)
            base_weights = {m: uw for m in active_models}

        if self._ortho_monitor:
            self._ortho_monitor.update_signals(signals, date)

        if self._has_sufficient_history(active_models):
            sharpe = self._compute_rolling_sharpe(active_models)
            weights = self._adjust_weights_for_performance(
                {m: base_weights[m] for m in active_models}, sharpe
            )
        else:
            raw = {m: base_weights[m] for m in active_models}
            total = sum(raw.values()) or 1.0
            weights = {m: v / total for m, v in raw.items()}

        if self._ortho_monitor and self._ortho_monitor.auto_downweight:
            penalties = self._ortho_monitor.compute_orthogonality_penalty(
                active_models, weights, date
            )
            for m in weights:
                weights[m] *= penalties.get(m, 1.0)
            # Do NOT re-normalise here: doing so negates the penalty entirely
            # (penalised models' share is redistributed back to the others).
            # The combined signal is z-scored downstream, so the absolute sum
            # does not affect scale. We only cap at max_weight to prevent a
            # single un-penalised model from dominating.
            for m in weights:
                weights[m] = min(weights[m], self._max_weight)

        all_tickers: set[str] = set()
        for m in active_models:
            all_tickers.update(signals[m].index.tolist())
        tickers = sorted(all_tickers)

        combined = pd.Series(0.0, index=tickers)
        contributions: dict[str, float] = {}
        for model in active_models:
            sig = signals[model].reindex(tickers).fillna(0.0)
            combined += weights[model] * sig
            contributions[model] = float((weights[model] * sig).mean())

        std = float(combined.std())
        if std > 1e-9:
            combined = (combined - combined.mean()) / std

        self._weights_log[date] = {
            "weights": weights.copy(),
            "regime": regime,
            "contributions": contributions,
        }

        if self._ortho_monitor:
            ortho_report = self._ortho_monitor.get_orthogonality_report(
                active_models, date
            )
            self._weights_log[date]["orthogonality"] = ortho_report

        logger.info(
            f"[{date}] regime={regime} weights={weights} "
            f"n_tickers={len(tickers)} active_models={active_models}"
        )

        combined.index.name = "ticker"
        return combined

    def update_performance(
        self,
        signals_history: dict[str, pd.DataFrame],
        returns_history: pd.DataFrame,
        date: date,
    ) -> None:
        for model_name, signals_df in signals_history.items():
            if model_name not in self._ic_by_date:
                self._ic_by_date[model_name] = {}

            # Signals generated at T are meant to predict returns from T to T+1.
            # IC must therefore be computed between signals[T-1] and returns[T],
            # not signals[T] vs returns[T] (same-day, not predictive).
            signal_dates = signals_df.index
            common_return_dates = returns_history.index.intersection(signal_dates)
            for d in common_return_dates:
                if d in self._ic_by_date[model_name]:
                    continue

                # Find the signal date immediately preceding d.
                pos = signal_dates.searchsorted(d)
                if pos == 0:
                    continue  # No prior signal date available for this return date.
                prev_date = signal_dates[pos - 1]

                sig = signals_df.loc[prev_date].dropna()
                ret = returns_history.loc[d].dropna()
                common_tickers = sig.index.intersection(ret.index)
                if len(common_tickers) < 3:
                    continue

                ic_val, _ = spearmanr(
                    sig[common_tickers].values, ret[common_tickers].values
                )
                if not np.isnan(ic_val):
                    self._ic_by_date[model_name][d] = float(ic_val)

            # Evict oldest IC entries beyond the Sharpe window to bound memory.
            if len(self._ic_by_date[model_name]) > self._sharpe_window:
                oldest = sorted(self._ic_by_date[model_name])[:-self._sharpe_window]
                for old_d in oldest:
                    del self._ic_by_date[model_name][old_d]

        # Evict oldest weights-log entries beyond the Sharpe window.
        if len(self._weights_log) > self._sharpe_window:
            oldest = sorted(self._weights_log)[:-self._sharpe_window]
            for old_d in oldest:
                del self._weights_log[old_d]

        for log_date, log_entry in self._weights_log.items():
            for model_name, w in log_entry["weights"].items():
                ic = self._ic_by_date.get(model_name, {}).get(log_date, np.nan)
                log_entry["contributions"][model_name] = w * ic if not np.isnan(ic) else np.nan

        logger.debug(
            f"[{date}] update_performance: IC history updated for "
            f"{list(signals_history.keys())}"
        )

    def get_attribution(self, date: date) -> pd.DataFrame:
        log = self._weights_log.get(date, {})
        weights_used = log.get("weights", {})
        contributions = log.get("contributions", {})
        ortho_report = log.get("orthogonality", {})

        all_models = sorted(
            set(weights_used.keys()) | set(self._ic_by_date.keys())
        )

        rows = []
        for model_name in all_models:
            ic_entries = sorted(self._ic_by_date.get(model_name, {}).items())
            recent_30 = [v for _, v in ic_entries[-self._ic_window:]]
            recent_60 = [v for _, v in ic_entries[-self._sharpe_window:]]

            ic_30d = float(np.mean(recent_30)) if len(recent_30) >= self._ic_window else np.nan
            sharpe_60d = (
                float(np.mean(recent_60) / (np.std(recent_60) + 1e-9) * np.sqrt(252))
                if len(recent_60) >= 2
                else np.nan
            )

            rows.append({
                "model_name": model_name,
                "weight": weights_used.get(model_name, np.nan),
                "ic_rolling_30d": ic_30d,
                "sharpe_rolling_60d": sharpe_60d,
                "pnl_contribution": contributions.get(model_name, np.nan),
            })

        return pd.DataFrame(
            rows,
            columns=["model_name", "weight", "ic_rolling_30d", "sharpe_rolling_60d", "pnl_contribution"],
        )

    def get_orthogonality_status(self, date: Optional[date] = None) -> dict:
        if not self._ortho_monitor:
            return {"status": "disabled"}
        models = list(self._ic_by_date.keys())
        return self._ortho_monitor.get_orthogonality_report(models, date)

    def _adjust_weights_for_performance(
        self,
        base_weights: dict[str, float],
        sharpe_rolling: dict[str, float],
    ) -> dict[str, float]:
        # Use a soft floor (0.1) instead of hard-zero for negative-Sharpe models.
        # This preserves a minimum contribution from temporarily underperforming
        # models, allowing them to recover without restarting from 0.
        adjusted = {
            m: w * max(0.1, sharpe_rolling.get(m, 0.0))
            for m, w in base_weights.items()
        }
        total = sum(adjusted.values())

        if total < 1e-9:
            total_base = sum(base_weights.values()) or 1.0
            return {m: v / total_base for m, v in base_weights.items()}

        normalized = {m: v / total for m, v in adjusted.items()}

        # Project onto the [min_weight, max_weight] constrained simplex.
        # Simple clip-renorm diverges for extreme Sharpe ratios because renorm
        # can push a just-clipped weight back above max_weight.
        # Correct approach: at each step, separate "fixed" models (at bounds)
        # from "free" models and redistribute the remaining budget proportionally
        # among the free models.  Converges in O(n) iterations for n models.
        w = dict(normalized)
        for _ in range(len(w) + 2):
            total = sum(w.values()) or 1.0
            w = {m: v / total for m, v in w.items()}

            if all(
                self._min_weight - 1e-8 <= v <= self._max_weight + 1e-8
                for v in w.values()
            ):
                break

            fixed: dict[str, float] = {}
            free: dict[str, float] = {}
            for m, v in w.items():
                if v <= self._min_weight:
                    fixed[m] = self._min_weight
                elif v >= self._max_weight:
                    fixed[m] = self._max_weight
                else:
                    free[m] = v

            remaining = 1.0 - sum(fixed.values())
            if free and remaining > 1e-9:
                free_total = sum(free.values())
                if free_total > 1e-9:
                    w = {**fixed, **{m: remaining * v / free_total for m, v in free.items()}}
                else:
                    w = {**fixed, **{m: remaining / len(free) for m in free}}
            else:
                # All models are at bounds; absorb remainder into min-bound ones.
                w = fixed
                if remaining > 1e-9:
                    min_models = [m for m, v in fixed.items() if v <= self._min_weight + 1e-9]
                    if min_models:
                        add = min(remaining / len(min_models), self._max_weight - self._min_weight)
                        for m in min_models:
                            w[m] = w[m] + add

        total_final = sum(w.values()) or 1.0
        return {m: v / total_final for m, v in w.items()}

    def _has_sufficient_history(self, models: list[str]) -> bool:
        return all(
            len(self._ic_by_date.get(m, {})) >= self._min_history
            for m in models
        )

    def _compute_rolling_sharpe(self, models: list[str]) -> dict[str, float]:
        """Compute IC-Sharpe using exponentially weighted IC observations.

        EWM with halflife=20 days gives ~4× more weight to the most recent
        month vs. observations from 60 days ago, enabling faster adaptation
        to regime changes compared to a simple rolling mean.
        """
        result: dict[str, float] = {}
        for m in models:
            entries = sorted(self._ic_by_date.get(m, {}).items())
            recent = [v for _, v in entries[-self._sharpe_window:]]
            if len(recent) >= 2:
                ic_series = pd.Series(recent, dtype=float)
                halflife = min(20, len(recent) // 2)
                ewm = ic_series.ewm(halflife=halflife, adjust=True)
                ewm_mean = float(ewm.mean().iloc[-1])
                ewm_std = float(ewm.std().iloc[-1])
                result[m] = ewm_mean / (ewm_std + 1e-9) * np.sqrt(252)
            else:
                result[m] = 0.0
        return result
