"""HMM Gaussian 3-state regime detector.

This model is NOT a BaseModel — it does not generate alpha signals.
It generates a regime label ("bull" / "bear" / "transition") that is
used by the council ensemble to condition portfolio weights.

State labelling convention
--------------------------
After fitting, HMM states are sorted by the mean of the first feature
column (expected to be an equity return series).  The state with the
highest mean return is labelled "bull", the lowest "bear", and the
middle state "transition".  This ordering is deterministic regardless
of the arbitrary internal numbering hmmlearn assigns.

Macro DataFrame schema (minimum required columns — flexible)
------------------------------------------------------------
Preference order (first matching set is used):
  1. sp500_ret_21d, vix_level, yield_spread
  2. sp500_ret_20d, vix,       yield_spread
  3. sp500_ret_5d,  vix,       yield_spread
  4. Any 3 numeric columns (last resort)
"""

from __future__ import annotations

import pickle
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
import polars as pl
import yaml
from hmmlearn import hmm
from sklearn.preprocessing import StandardScaler


# Ordered preference lists for feature column selection
_FEATURE_PREFERENCES: list[list[str]] = [
    ["sp500_ret_21d", "vix_level", "yield_spread"],
    ["sp500_ret_20d", "vix_level", "yield_spread"],
    ["sp500_ret_20d", "vix", "yield_spread"],
    ["sp500_ret_5d", "vix", "yield_spread"],
]


class RegimeModel:
    """Gaussian HMM regime detector.

    Attributes
    ----------
    name : str
        Model identifier.
    n_states : int
        Number of hidden Markov states (default 3).
    """

    name = "hmm_regime"

    def __init__(
        self,
        n_states: int = 3,
        config_path: str = "config/models.yaml",
    ) -> None:
        self.n_states = n_states

        try:
            with open(config_path) as f:
                cfg = yaml.safe_load(f)
            hmm_cfg = cfg.get("hmm", {})
            self._n_iter: int = hmm_cfg.get("n_iter", 100)
            self._covariance_type: str = hmm_cfg.get("covariance_type", "full")
            self._random_state: int = hmm_cfg.get("random_state", 42)
            self.n_states = hmm_cfg.get("n_states", n_states)
        except FileNotFoundError:
            self._n_iter = 100
            self._covariance_type = "full"
            self._random_state = 42

        self._model: hmm.GaussianHMM | None = None
        self._scaler: StandardScaler | None = None
        self._feature_cols: list[str] | None = None
        # Maps HMM state int → label str
        self._state_map: dict[int, str] = {}
        self._last_trained: str | None = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _select_feature_cols(df: pd.DataFrame) -> list[str]:
        """Pick the first matching feature column set from preferences."""
        for cols in _FEATURE_PREFERENCES:
            if all(c in df.columns for c in cols):
                return cols

        # Last resort: any 3 numeric columns (exclude valid_time)
        numeric = [
            c
            for c in df.columns
            if c != "valid_time" and pd.api.types.is_numeric_dtype(df[c])
        ]
        if len(numeric) < 1:
            raise ValueError(
                "macro_df has no usable numeric columns for HMM training."
            )
        return numeric[:3]

    def _prepare_X(self, macro_df: pl.DataFrame) -> tuple[np.ndarray, pd.DataFrame]:
        """Convert macro Polars DataFrame → scaled numpy array.

        Also returns the pandas DataFrame (sorted by valid_time) for alignment.
        """
        if self._scaler is None or self._feature_cols is None:
            raise RuntimeError("Model not fitted. Call fit() first.")

        df = macro_df.sort("valid_time").to_pandas()
        X = df[self._feature_cols].ffill().bfill().fillna(0.0).values
        return self._scaler.transform(X), df

    # ------------------------------------------------------------------
    # fit
    # ------------------------------------------------------------------

    def fit(self, macro_df: pl.DataFrame) -> None:
        """Fit the Gaussian HMM on macro time-series data.

        Parameters
        ----------
        macro_df:
            Polars DataFrame with at least `valid_time` and macro feature
            columns.  See module docstring for accepted column sets.
        """
        df = macro_df.sort("valid_time").to_pandas()
        feat_cols = self._select_feature_cols(df)
        self._feature_cols = feat_cols

        X = df[feat_cols].ffill().bfill().fillna(0.0).values

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        model = hmm.GaussianHMM(
            n_components=self.n_states,
            covariance_type=self._covariance_type,
            n_iter=self._n_iter,
            random_state=self._random_state,
        )
        model.fit(X_scaled)

        # Label states by mean of the first feature (equity return proxy)
        states = model.predict(X_scaled)
        df["_hmm_state"] = states

        return_col = feat_cols[0]
        state_means = df.groupby("_hmm_state")[return_col].mean()

        # Sort descending: highest return → bull, lowest → bear, middle → transition
        sorted_states = state_means.sort_values(ascending=False).index.tolist()

        if self.n_states == 3:
            self._state_map = {
                sorted_states[0]: "bull",
                sorted_states[1]: "transition",
                sorted_states[2]: "bear",
            }
        else:
            labels = ["bull"] + ["transition"] * (self.n_states - 2) + ["bear"]
            self._state_map = {s: l for s, l in zip(sorted_states, labels)}

        self._model = model
        self._scaler = scaler
        self._last_trained = datetime.now().isoformat()

    # ------------------------------------------------------------------
    # predict
    # ------------------------------------------------------------------

    def predict_regime(self, macro_df: pl.DataFrame) -> str:
        """Return the regime label for the most recent observation.

        Returns
        -------
        str
            One of: "bull", "bear", "transition".
        """
        X_scaled, _ = self._prepare_X(macro_df)
        state = int(self._model.predict(X_scaled)[-1])
        return self._state_map.get(state, "transition")

    def predict_probabilities(self, macro_df: pl.DataFrame) -> dict[str, float]:
        """Return regime probabilities for the most recent observation.

        Returns
        -------
        dict
            {"bull": float, "bear": float, "transition": float}
            Values sum to 1.0.
        """
        X_scaled, _ = self._prepare_X(macro_df)
        # predict_proba returns (n_samples, n_states) posterior probabilities
        probs = self._model.predict_proba(X_scaled)
        last_probs = probs[-1]

        return {
            self._state_map.get(i, f"state_{i}"): float(last_probs[i])
            for i in range(self.n_states)
        }

    def get_regime_history(self, macro_df: pl.DataFrame) -> pd.DataFrame:
        """Return full regime history with per-state probabilities.

        Used by the dashboard and backtesting engine.

        Returns
        -------
        pd.DataFrame
            Columns: valid_time, regime, prob_bull, prob_bear, prob_transition.
        """
        X_scaled, df = self._prepare_X(macro_df)
        states = self._model.predict(X_scaled)
        probs = self._model.predict_proba(X_scaled)

        result = df[["valid_time"]].copy().reset_index(drop=True)
        result["regime"] = [
            self._state_map.get(int(s), f"state_{s}") for s in states
        ]
        for i in range(self.n_states):
            label = self._state_map.get(i, f"state_{i}")
            result[f"prob_{label}"] = probs[:, i]

        return result

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Serialize to disk using pickle."""
        with open(path, "wb") as f:
            pickle.dump(self, f)

    def load(self, path: str) -> None:
        """Load from pickle file into this instance."""
        with open(path, "rb") as f:
            saved = pickle.load(f)
        self.__dict__.update(saved.__dict__)

    def get_metadata(self) -> dict:
        return {
            "name": self.name,
            "n_states": self.n_states,
            "last_trained": self._last_trained,
            "state_map": self._state_map,
            "feature_cols": self._feature_cols,
        }
