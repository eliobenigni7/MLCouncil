"""HMM Gaussian 3-state regime detector.

This model is NOT a BaseModel - it does not generate alpha signals.
It generates a regime label ("bull" / "bear" / "transition") that is
used by the council ensemble to condition portfolio weights.

State labelling convention
--------------------------
After fitting, latent states are sorted by the mean of the first feature
column (expected to be an equity return series). The state with the
highest mean return is labelled "bull", the lowest "bear", and the
middle state "transition". This ordering is deterministic regardless
of the arbitrary internal numbering assigned by the backend.

Macro DataFrame schema (minimum required columns - flexible)
------------------------------------------------------------
Preference order (first matching set is used):
  1. sp500_ret_21d, vix_level, yield_spread
  2. sp500_ret_20d, vix,       yield_spread
  3. sp500_ret_5d,  vix,       yield_spread
  4. Any 3 numeric columns (last resort)
"""

from __future__ import annotations

import hashlib
import pickle
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import polars as pl
import yaml
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

try:
    from hmmlearn import hmm
except ModuleNotFoundError:  # pragma: no cover - exercised via fallback tests
    hmm = None


# Ordered preference lists for feature column selection
_FEATURE_PREFERENCES: list[list[str]] = [
    ["sp500_ret_21d", "vix_level", "yield_spread"],
    ["sp500_ret_20d", "vix_level", "yield_spread"],
    ["sp500_ret_20d", "vix", "yield_spread"],
    ["sp500_ret_5d", "vix", "yield_spread"],
]


class RegimeModel:
    """Gaussian regime detector with a graceful fallback backend."""

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

        self._model: Any | None = None
        self._scaler: StandardScaler | None = None
        self._feature_cols: list[str] | None = None
        self._state_map: dict[int, str] = {}
        self._last_trained: str | None = None
        self._training_returns: np.ndarray | None = None
        self._backend = (
            "hmmlearn" if hmm is not None else "gaussian_mixture_fallback"
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _select_feature_cols(df: pd.DataFrame) -> list[str]:
        """Pick the first matching feature column set from preferences."""
        for cols in _FEATURE_PREFERENCES:
            if all(c in df.columns for c in cols):
                return cols

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
        """Convert macro Polars DataFrame to scaled numpy array."""
        if self._scaler is None or self._feature_cols is None:
            raise RuntimeError("Model not fitted. Call fit() first.")

        df = macro_df.sort("valid_time").to_pandas()
        X = df[self._feature_cols].ffill().bfill().fillna(0.0).values
        return self._scaler.transform(X), df

    def _build_model(self) -> Any:
        """Build the preferred regime model backend."""
        if hmm is not None:
            return hmm.GaussianHMM(
                n_components=self.n_states,
                covariance_type=self._covariance_type,
                n_iter=self._n_iter,
                random_state=self._random_state,
            )

        return GaussianMixture(
            n_components=self.n_states,
            covariance_type=self._covariance_type,
            max_iter=self._n_iter,
            random_state=self._random_state,
        )

    def _label_states(
        self,
        model: Any,
        states: np.ndarray,
        training_returns: np.ndarray,
    ) -> dict[int, str]:
        """Create a deterministic mapping from latent state id to regime label.

        Parameters
        ----------
        model:
            Fitted HMM or GaussianMixture instance.
        states:
            Predicted state sequence from the current fit (same length as
            ``training_returns``).
        training_returns:
            First-feature column values used in the current fit.  Passed
            explicitly to avoid relying on ``self._training_returns``, which
            could be stale if a previous ``fit()`` call raised an exception
            after updating the attribute but before completing state labelling.
        """
        means = getattr(model, "means_", None)
        if means is not None and len(means) >= self.n_states:
            sorted_states = sorted(
                range(self.n_states),
                key=lambda idx: float(np.ravel(means[idx])[0]),
                reverse=True,
            )
        else:
            summary = (
                pd.DataFrame({"state": states, "value": training_returns})
                .groupby("state")["value"]
                .mean()
            )
            sorted_states = summary.sort_values(ascending=False).index.tolist()

        if self.n_states == 1:
            return {sorted_states[0]: "transition"}

        if self.n_states == 2:
            return {
                sorted_states[0]: "bull",
                sorted_states[-1]: "bear",
            }

        labels = ["bull"] + ["transition"] * (self.n_states - 2) + ["bear"]
        return {state: label for state, label in zip(sorted_states, labels)}

    # ------------------------------------------------------------------
    # fit
    # ------------------------------------------------------------------

    def fit(self, macro_df: pl.DataFrame) -> None:
        """Fit the regime model on macro time-series data."""
        df = macro_df.sort("valid_time").to_pandas()
        feat_cols = self._select_feature_cols(df)
        self._feature_cols = feat_cols

        X = df[feat_cols].ffill().bfill().fillna(0.0).values

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        model = self._build_model()
        model.fit(X_scaled)

        states = np.asarray(model.predict(X_scaled), dtype=int)
        training_returns = df[feat_cols[0]].to_numpy(copy=True)
        self._training_returns = training_returns
        # Pass training_returns explicitly so _label_states never reads a stale
        # self._training_returns left over from a failed previous fit() call.
        self._state_map = self._label_states(model, states, training_returns)

        self._model = model
        self._scaler = scaler
        self._last_trained = datetime.now().isoformat()

    # ------------------------------------------------------------------
    # predict
    # ------------------------------------------------------------------

    def predict_regime(self, macro_df: pl.DataFrame) -> str:
        """Return the regime label for the most recent observation."""
        if self._model is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        X_scaled, _ = self._prepare_X(macro_df)
        state = int(self._model.predict(X_scaled)[-1])
        return self._state_map.get(state, "transition")

    def predict_probabilities(self, macro_df: pl.DataFrame) -> dict[str, float]:
        """Return regime probabilities for the most recent observation."""
        if self._model is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        X_scaled, _ = self._prepare_X(macro_df)
        probs = self._model.predict_proba(X_scaled)
        last_probs = probs[-1]

        return {
            self._state_map.get(i, f"state_{i}"): float(last_probs[i])
            for i in range(self.n_states)
        }

    def get_regime_history(self, macro_df: pl.DataFrame) -> pd.DataFrame:
        """Return full regime history with per-state probabilities."""
        if self._model is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
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
        """Serialize to disk using pickle with SHA-256 hash sidecar."""
        p = Path(path)
        with open(p, "wb") as f:
            pickle.dump(self, f)
        # Write hash sidecar for tamper detection on load
        digest = hashlib.sha256(p.read_bytes()).hexdigest()
        p.with_suffix(p.suffix + ".hash").write_text(digest)

    def load(self, path: str) -> None:
        """Load from pickle file with SHA-256 hash verification.

        Raises ``ValueError`` when a ``.hash`` sidecar exists and the
        digest does not match, preventing execution of tampered files.
        """
        p = Path(path)
        hash_path = p.with_suffix(p.suffix + ".hash")
        if hash_path.exists():
            expected = hash_path.read_text().strip()
            actual = hashlib.sha256(p.read_bytes()).hexdigest()
            if actual != expected:
                raise ValueError(
                    f"Checkpoint hash mismatch for {p}: "
                    f"expected {expected}, got {actual}. "
                    "File may be corrupted or tampered with."
                )
        with open(p, "rb") as f:
            saved = pickle.load(f)
        self.__dict__.update(saved.__dict__)

    def get_metadata(self) -> dict:
        return {
            "name": self.name,
            "n_states": self.n_states,
            "last_trained": self._last_trained,
            "state_map": self._state_map,
            "feature_cols": self._feature_cols,
            "backend": self._backend,
        }
