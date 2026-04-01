"""Abstract base class for all MLCouncil models.

Every council model must implement fit() and predict().
The predict() contract:
  - Input:  pl.DataFrame with ticker, valid_time, and feature columns
  - Output: pd.Series(index=ticker, values=float)
             Values are cross-sectional z-scores.
             Positive = long bias, negative = short bias.
"""

from __future__ import annotations

import pickle
from abc import ABC, abstractmethod
from datetime import datetime

import pandas as pd
import polars as pl


class BaseModel(ABC):
    name: str  # unique model identifier used in MLflow and logs

    @abstractmethod
    def fit(self, features: pl.DataFrame, targets: pd.Series) -> None:
        """Train the model.

        Parameters
        ----------
        features:
            pl.DataFrame with columns [ticker, valid_time, *feature_cols].
            Feature values at row T use only data available at close of T-1
            (no look-ahead bias).
        targets:
            pd.Series indexed by (ticker, valid_time) MultiIndex or a flat
            index aligned with `features` rows. Values are cross-sectional
            rank percentiles [0, 1] from compute_targets().
        """

    @abstractmethod
    def predict(self, features: pl.DataFrame) -> pd.Series:
        """Generate alpha signals.

        Parameters
        ----------
        features:
            pl.DataFrame with the same schema as used during fit().

        Returns
        -------
        pd.Series
            index = ticker (str), values = float z-scores.
            Positive = long bias, negative = short bias.
            When multiple dates are present in `features`, signals are
            z-scored cross-sectionally within each date then concatenated.
        """

    def save(self, path: str) -> None:
        """Serialize the model to disk using pickle."""
        with open(path, "wb") as f:
            pickle.dump(self, f)

    def load(self, path: str) -> None:
        """Load model state from a pickle file into this instance."""
        with open(path, "rb") as f:
            saved = pickle.load(f)
        self.__dict__.update(saved.__dict__)

    def get_metadata(self) -> dict:
        """Return a metadata dictionary for logging and provenance tracking.

        Keys
        ----
        name          : str   model identifier
        last_trained  : str   ISO-8601 timestamp of last fit() call
        n_features    : int   number of input features used
        params        : dict  hyperparameters passed at construction
        """
        return {
            "name": self.name,
            "last_trained": getattr(self, "_last_trained", None),
            "n_features": getattr(self, "_n_features", None),
            "params": getattr(self, "_params", {}),
        }
