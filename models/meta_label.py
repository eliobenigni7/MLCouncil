"""Meta-labeling gate for primary LightGBM signals.

A secondary binary classifier estimates P(primary direction is correct | features).
Signals below ``MLCOUNCIL_META_LABEL_THRESHOLD`` are zeroed when enabled.

Enable filtering: ``MLCOUNCIL_META_LABEL=true``
Shadow (log only): ``MLCOUNCIL_META_LABEL_SHADOW=true``
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import polars as pl
import yaml

from council.pickle_security import verify_pickle_hash_sidecar

_TRUTHY = frozenset({"1", "true", "yes", "on"})
_EXCLUDE_COLS = frozenset({"ticker", "valid_time", "transaction_time"})
_DEFAULT_CHECKPOINT = Path("models/checkpoints/meta_label_latest.pkl")


def meta_label_enabled() -> bool:
    return os.getenv("MLCOUNCIL_META_LABEL", "").strip().lower() in _TRUTHY


def meta_label_shadow_mode() -> bool:
    return os.getenv("MLCOUNCIL_META_LABEL_SHADOW", "").strip().lower() in _TRUTHY


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    return float(raw)


@dataclass
class MetaLabelFilterStats:
    """Outcome of applying the meta-label gate."""

    n_total: int
    n_filtered: int
    filtered_fraction: float
    threshold: float
    shadow: bool
    mean_prob_kept: float | None = None
    mean_prob_filtered: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_total": self.n_total,
            "n_filtered": self.n_filtered,
            "filtered_fraction": self.filtered_fraction,
            "threshold": self.threshold,
            "shadow": self.shadow,
            "mean_prob_kept": self.mean_prob_kept,
            "mean_prob_filtered": self.mean_prob_filtered,
            **self.metadata,
        }


def build_meta_labels(
    primary_signal: np.ndarray | pd.Series,
    forward_return: np.ndarray | pd.Series,
    *,
    min_abs_signal: float = 0.05,
) -> np.ndarray:
    """Binary label: 1 if primary direction matches realized forward return."""
    sig = np.asarray(primary_signal, dtype=float)
    ret = np.asarray(forward_return, dtype=float)
    valid = (
        np.isfinite(sig)
        & np.isfinite(ret)
        & (np.abs(sig) >= min_abs_signal)
        & (np.abs(ret) > 1e-12)
    )
    labels = np.full(len(sig), np.nan, dtype=float)
    labels[valid] = (np.sign(sig[valid]) == np.sign(ret[valid])).astype(float)
    return labels


class MetaLabelClassifier:
    """Secondary classifier: P(primary LGBM direction correct)."""

    name = "meta_label"

    def __init__(self, config_path: str = "config/models.yaml") -> None:
        try:
            with open(config_path) as f:
                cfg = yaml.safe_load(f) or {}
        except FileNotFoundError:
            cfg = {}
        ml_cfg = cfg.get("meta_label", {})
        self._params: dict = ml_cfg.get(
            "classifier",
            {
                "n_estimators": 100,
                "learning_rate": 0.05,
                "num_leaves": 15,
                "min_child_samples": 50,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "random_state": 42,
                "verbose": -1,
            },
        )
        self._threshold: float = float(
            ml_cfg.get("threshold", _env_float("MLCOUNCIL_META_LABEL_THRESHOLD", 0.55))
        )
        self._min_abs_signal: float = float(ml_cfg.get("min_abs_signal", 0.05))
        self._model: lgb.LGBMClassifier | None = None
        self._feature_cols: list[str] | None = None

    @property
    def threshold(self) -> float:
        return self._threshold

    def _feature_cols_from(self, features: pl.DataFrame) -> list[str]:
        return [c for c in features.columns if c not in _EXCLUDE_COLS]

    def _aligned_frame(
        self,
        features: pl.DataFrame,
        primary_signal: pd.Series,
        forward_return: pd.Series | None = None,
    ) -> pd.DataFrame:
        feat_cols = self._feature_cols_from(features)
        df = features.select(["ticker", "valid_time"] + feat_cols).to_pandas()

        if pd.api.types.is_datetime64_any_dtype(df["valid_time"]):
            df["valid_time"] = df["valid_time"].dt.date
        else:
            df["valid_time"] = pd.to_datetime(df["valid_time"], errors="coerce").dt.date

        sig = primary_signal.rename("__signal__")
        if isinstance(sig.index, pd.MultiIndex):
            sig = sig.reset_index()
            sig.columns = list(sig.columns[:-1]) + ["__signal__"]
            if pd.api.types.is_datetime64_any_dtype(sig["valid_time"]):
                sig["valid_time"] = sig["valid_time"].dt.date
            else:
                sig["valid_time"] = pd.to_datetime(sig["valid_time"], errors="coerce").dt.date
            on = ["ticker", "valid_time"]
            df = df.merge(sig[on + ["__signal__"]], on=on, how="inner")
        else:
            df = df.copy()
            df["__signal__"] = sig.reindex(df["ticker"]).values

        if forward_return is not None:
            ret = forward_return.rename("__ret__")
            if isinstance(ret.index, pd.MultiIndex):
                ret = ret.reset_index()
                ret.columns = list(ret.columns[:-1]) + ["__ret__"]
                if pd.api.types.is_datetime64_any_dtype(ret["valid_time"]):
                    ret["valid_time"] = ret["valid_time"].dt.date
                else:
                    ret["valid_time"] = pd.to_datetime(ret["valid_time"], errors="coerce").dt.date
                df = df.merge(ret[["ticker", "valid_time", "__ret__"]], on=["ticker", "valid_time"], how="inner")
            else:
                df["__ret__"] = ret.reindex(df["ticker"]).values

        return df

    def fit(
        self,
        features: pl.DataFrame,
        primary_signal: pd.Series,
        forward_return: pd.Series,
    ) -> None:
        """Train on rows where primary signal direction can be scored vs realized return."""
        df = self._aligned_frame(features, primary_signal, forward_return)
        labels = build_meta_labels(
            df["__signal__"].values,
            df["__ret__"].values,
            min_abs_signal=self._min_abs_signal,
        )
        mask = np.isfinite(labels)
        if mask.sum() < 50:
            raise ValueError(
                f"Need at least 50 labeled meta-label rows, got {int(mask.sum())}"
            )

        feat_cols = self._feature_cols_from(features)
        df.loc[:, "meta_primary_signal"] = df["__signal__"]
        if "meta_primary_signal" not in feat_cols:
            feat_cols = feat_cols + ["meta_primary_signal"]
        self._feature_cols = feat_cols
        X = df.loc[mask, feat_cols].fillna(0.0)
        y = labels[mask].astype(int)

        self._model = lgb.LGBMClassifier(**self._params)
        self._model.fit(X, y)

    def predict_proba(
        self,
        features: pl.DataFrame,
        primary_signal: pd.Series | None = None,
    ) -> pd.Series:
        """P(correct) per row; index matches ``primary_signal`` or ticker column."""
        if self._model is None:
            raise RuntimeError("MetaLabelClassifier not fitted")

        feat_cols = list(self._feature_cols or self._feature_cols_from(features))
        if "meta_primary_signal" in feat_cols and primary_signal is not None:
            feat_cols = [c for c in feat_cols if c != "meta_primary_signal"]

        df = features.select(["ticker", "valid_time"] + feat_cols).to_pandas()
        for col in feat_cols:
            if col not in df.columns:
                df[col] = 0.0

        if primary_signal is not None:
            if isinstance(primary_signal.index, pd.MultiIndex):
                sig_df = primary_signal.rename("meta_primary_signal").reset_index()
                df = df.merge(
                    sig_df,
                    on=["ticker", "valid_time"],
                    how="left",
                )
            else:
                df["meta_primary_signal"] = primary_signal.reindex(df["ticker"]).values
            if "meta_primary_signal" not in feat_cols:
                feat_cols = feat_cols + ["meta_primary_signal"]

        X = df[feat_cols].fillna(0.0)
        proba = self._model.predict_proba(X)[:, 1]
        index = (
            primary_signal.index
            if primary_signal is not None and len(primary_signal.index) == len(proba)
            else df["ticker"].values
        )
        return pd.Series(proba, index=index, name="meta_prob")

    def filter_signals(
        self,
        signals: pd.Series,
        features: pl.DataFrame,
        *,
        threshold: float | None = None,
        shadow: bool = False,
    ) -> tuple[pd.Series, MetaLabelFilterStats]:
        """Zero signals with meta-label probability below threshold."""
        thr = self._threshold if threshold is None else threshold
        probs = self.predict_proba(features, primary_signal=signals)

        aligned = signals.reindex(probs.index).fillna(0.0)
        n_total = len(aligned)
        low = probs < thr
        n_filtered = int(low.sum())

        kept_mask = ~low
        stats = MetaLabelFilterStats(
            n_total=n_total,
            n_filtered=n_filtered,
            filtered_fraction=n_filtered / n_total if n_total else 0.0,
            threshold=thr,
            shadow=shadow,
            mean_prob_kept=float(probs[kept_mask].mean()) if kept_mask.any() else None,
            mean_prob_filtered=float(probs[low].mean()) if n_filtered else None,
        )

        if shadow:
            return signals, stats

        out = aligned.copy()
        out[low.reindex(out.index, fill_value=False)] = 0.0
        return out, stats

    def save(self, path: str | Path) -> None:
        import hashlib

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        path.with_suffix(path.suffix + ".hash").write_text(digest)

    @classmethod
    def load(cls, path: str | Path) -> MetaLabelClassifier:
        path = Path(path)
        verify_pickle_hash_sidecar(path)
        loaded = joblib.load(path)
        if not isinstance(loaded, cls):
            raise TypeError(f"Expected {cls.__name__}, got {type(loaded)}")
        return loaded


def apply_meta_label_gate(
    signals: pd.Series,
    features: pl.DataFrame,
    checkpoint: str | Path | None = None,
) -> tuple[pd.Series, MetaLabelFilterStats | None]:
    """Apply meta-label filter when enabled; no-op when disabled or no checkpoint."""
    if not meta_label_enabled():
        return signals, None

    ckpt = Path(checkpoint) if checkpoint else _DEFAULT_CHECKPOINT
    if not ckpt.exists():
        return signals, MetaLabelFilterStats(
            n_total=len(signals),
            n_filtered=0,
            filtered_fraction=0.0,
            threshold=_env_float("MLCOUNCIL_META_LABEL_THRESHOLD", 0.55),
            shadow=meta_label_shadow_mode(),
            metadata={"skipped": "no_checkpoint"},
        )

    clf = MetaLabelClassifier.load(ckpt)
    return clf.filter_signals(
        signals,
        features,
        shadow=meta_label_shadow_mode(),
    )
