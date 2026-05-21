"""Deep state-space regime challenger (T2.3).

Continuous latent regime :math:`z_t \\in \\mathbb{R}^d` with amortized
variational inference. Production Dagster path keeps ``RegimeModel`` (HMM) as
default; this module runs in **shadow** until walk-forward promotion.

Backend priority: ``mamba-ssm`` (if installed) → NumPy recurrent VAE stub.
Torch is optional for the stub when available; tests run CPU-only on NumPy.
"""

from __future__ import annotations

import hashlib
import os
import pickle
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import polars as pl
import yaml

from models.regime import RegimeModel, _FEATURE_PREFERENCES

_TRUTHY = frozenset({"1", "true", "yes", "on"})
_REGIME_LABELS = ("bull", "bear", "transition")


def shadow_regime_enabled() -> bool:
    """True when DSS shadow inference should run alongside HMM."""
    return os.getenv("MLCOUNCIL_REGIME_DSS_SHADOW", "").strip().lower() in _TRUTHY


def _try_import_mamba() -> Any | None:
    try:
        import mamba_ssm  # noqa: F401

        return mamba_ssm
    except ImportError:
        return None


def _try_import_torch() -> Any | None:
    try:
        import torch  # noqa: F401

        return torch
    except ImportError:
        return None


@dataclass
class TrainMetrics:
    """Training summary for shadow logging and gating."""

    elbo_final: float
    elbo_initial: float
    n_steps: int
    backend: str
    embed_dim: int

    @property
    def elbo_improvement_pct(self) -> float:
        if abs(self.elbo_initial) < 1e-12:
            return 0.0
        return 100.0 * (self.elbo_final - self.elbo_initial) / abs(self.elbo_initial)


@dataclass
class ShadowRegimeRecord:
    """Shadow-only regime output for a single as-of date."""

    regime_label: str
    embedding: np.ndarray
    probabilities: dict[str, float]
    elbo: float | None = None
    backend: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


class _NumpyRecurrentVAE:
    """Lightweight recurrent VAE for CPU training without mamba-ssm."""

    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 8,
        hidden_dim: int = 16,
        random_state: int = 42,
    ) -> None:
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        rng = np.random.default_rng(random_state)
        scale = 0.05
        self.W_xh = rng.standard_normal((hidden_dim, input_dim)) * scale
        self.b_h = np.zeros(hidden_dim)
        self.W_zz = rng.standard_normal((latent_dim, latent_dim)) * scale
        self.W_hz = rng.standard_normal((latent_dim, hidden_dim)) * scale
        self.b_z = np.zeros(latent_dim)
        self.W_dec = rng.standard_normal((input_dim, latent_dim)) * scale
        self.b_dec = np.zeros(input_dim)
        self._last_elbo: float | None = None

    def _forward(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return latent path (T, d) and reconstructions (T, F)."""
        T = len(X)
        h = np.zeros(self.hidden_dim)
        z_path = np.zeros((T, self.latent_dim))
        recon = np.zeros_like(X)
        for t in range(T):
            z_prev = z_path[max(t - 1, 0)]
            h = np.tanh(self.W_xh @ X[t] + self.b_h)
            z = np.tanh(self.W_hz @ h + self.W_zz @ z_prev + self.b_z)
            z_path[t] = z
            recon[t] = self.W_dec @ z + self.b_dec
        return z_path, recon

    def elbo(self, X: np.ndarray, kl_weight: float = 0.01) -> float:
        z_path, recon = self._forward(X)
        recon_err = float(np.mean((X - recon) ** 2))
        smooth = float(np.mean(np.diff(z_path, axis=0) ** 2)) if len(z_path) > 1 else 0.0
        l2 = float(np.mean(z_path**2))
        return -(recon_err + kl_weight * (l2 + smooth))

    def _flatten(self) -> np.ndarray:
        parts = [
            self.W_xh.ravel(),
            self.b_h.ravel(),
            self.W_zz.ravel(),
            self.W_hz.ravel(),
            self.b_z.ravel(),
            self.W_dec.ravel(),
            self.b_dec.ravel(),
        ]
        return np.concatenate(parts)

    def _unflatten(self, params: np.ndarray) -> None:
        idx = 0
        for attr, shape in (
            ("W_xh", self.W_xh.shape),
            ("b_h", self.b_h.shape),
            ("W_zz", self.W_zz.shape),
            ("W_hz", self.W_hz.shape),
            ("b_z", self.b_z.shape),
            ("W_dec", self.W_dec.shape),
            ("b_dec", self.b_dec.shape),
        ):
            size = int(np.prod(shape))
            setattr(self, attr, params[idx : idx + size].reshape(shape))
            idx += size

    def fit(
        self,
        X: np.ndarray,
        *,
        n_epochs: int = 80,
        lr: float = 0.02,
        kl_weight: float = 0.01,
    ) -> TrainMetrics:
        elbo_initial = self.elbo(X, kl_weight=kl_weight)

        def objective(params: np.ndarray) -> float:
            self._unflatten(params)
            return -self.elbo(X, kl_weight=kl_weight)

        try:
            from scipy.optimize import minimize

            result = minimize(
                objective,
                self._flatten(),
                method="L-BFGS-B",
                options={"maxiter": max(n_epochs, 10)},
            )
            self._unflatten(result.x)
        except Exception:
            # Fallback: single-step finite difference on decoder only
            eps = 1e-3
            for _ in range(min(n_epochs, 20)):
                flat = self.W_dec.ravel()
                grad = np.zeros_like(flat)
                for i in range(flat.size):
                    old = flat[i]
                    flat[i] = old + eps
                    self.W_dec = flat.reshape(self.W_dec.shape)
                    plus = self.elbo(X, kl_weight=kl_weight)
                    flat[i] = old
                    self.W_dec = flat.reshape(self.W_dec.shape)
                    grad[i] = (plus - elbo_initial) / eps
                flat += lr * grad
                self.W_dec = flat.reshape(self.W_dec.shape)

        elbo_final = self.elbo(X, kl_weight=kl_weight)
        self._last_elbo = elbo_final
        return TrainMetrics(
            elbo_final=elbo_final,
            elbo_initial=elbo_initial,
            n_steps=len(X),
            backend="numpy_vae",
            embed_dim=self.latent_dim,
        )

    def encode(self, X: np.ndarray) -> np.ndarray:
        z_path, _ = self._forward(X)
        return z_path


class _MambaStubBackend(_NumpyRecurrentVAE):
    """Placeholder when mamba-ssm is importable but full CUDA stack is absent."""

    def fit(self, X: np.ndarray, **kwargs: Any) -> TrainMetrics:
        metrics = super().fit(X, **kwargs)
        metrics.backend = "mamba_stub_numpy"
        return metrics


class DeepRegimeModel:
    """Deep state-space regime model with continuous embeddings."""

    name = "regime_dss"

    def __init__(
        self,
        embed_dim: int = 8,
        config_path: str = "config/models.yaml",
    ) -> None:
        self.embed_dim = embed_dim
        self._feature_cols: list[str] | None = None
        self._scaler_mean: np.ndarray | None = None
        self._scaler_scale: np.ndarray | None = None
        self._backend: Any | None = None
        self._backend_name: str = "unfitted"
        self._prototypes: dict[str, np.ndarray] = {}
        self._last_trained: str | None = None
        self._train_metrics: TrainMetrics | None = None

        try:
            with open(config_path) as f:
                cfg = yaml.safe_load(f) or {}
            dss_cfg = cfg.get("regime_dss", {})
            self.embed_dim = int(dss_cfg.get("embed_dim", embed_dim))
            self._n_epochs = int(dss_cfg.get("n_epochs", 80))
            self._lr = float(dss_cfg.get("learning_rate", 0.02))
            self._kl_weight = float(dss_cfg.get("kl_weight", 0.01))
            self._random_state = int(dss_cfg.get("random_state", 42))
        except FileNotFoundError:
            self._n_epochs = 80
            self._lr = 0.02
            self._kl_weight = 0.01
            self._random_state = 42

    @staticmethod
    def _select_feature_cols(df: pd.DataFrame) -> list[str]:
        for cols in _FEATURE_PREFERENCES:
            if all(c in df.columns for c in cols):
                return cols
        numeric = [
            c
            for c in df.columns
            if c != "valid_time" and pd.api.types.is_numeric_dtype(df[c])
        ]
        if len(numeric) < 1:
            raise ValueError("macro_df has no usable numeric columns for DSS training.")
        return numeric[:3]

    def _macro_to_matrix(self, macro_df: pl.DataFrame) -> tuple[np.ndarray, pd.DataFrame]:
        if self._feature_cols is None or self._scaler_mean is None or self._scaler_scale is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        df = macro_df.sort("valid_time").to_pandas()
        X = df[self._feature_cols].ffill().bfill().fillna(0.0).values.astype(float)
        X_scaled = (X - self._scaler_mean) / self._scaler_scale
        return X_scaled, df

    def _build_backend(self, input_dim: int) -> tuple[Any, str]:
        if _try_import_mamba() is not None:
            return (
                _MambaStubBackend(
                    input_dim,
                    latent_dim=self.embed_dim,
                    random_state=self._random_state,
                ),
                "mamba_stub_numpy",
            )
        return (
            _NumpyRecurrentVAE(
                input_dim,
                latent_dim=self.embed_dim,
                random_state=self._random_state,
            ),
            "numpy_vae",
        )

    def _fit_prototypes(self, z_path: np.ndarray, returns: np.ndarray) -> None:
        """Map latent states to bull/bear/transition via return ordering on z[:,0]."""
        z_last = z_path[-1]
        order = np.argsort(z_path[:, 0])
        n = len(order)
        if n < 3:
            self._prototypes = {
                "bull": z_last.copy(),
                "bear": z_last.copy(),
                "transition": z_last.copy(),
            }
            return
        thirds = np.array_split(order, 3)
        ret_means = [
            (float(returns[idx].mean()), idx) for idx in thirds if len(idx) > 0
        ]
        ret_means.sort(key=lambda x: x[0], reverse=True)
        labels = list(_REGIME_LABELS)
        for label, (_, idx) in zip(labels, ret_means):
            self._prototypes[label] = z_path[idx].mean(axis=0)
        for label in _REGIME_LABELS:
            self._prototypes.setdefault(label, z_last.copy())

    def fit(self, macro_df: pl.DataFrame, *, n_epochs: int | None = None) -> TrainMetrics:
        """Fit variational state-space model on macro features."""
        df = macro_df.sort("valid_time").to_pandas()
        feat_cols = self._select_feature_cols(df)
        self._feature_cols = feat_cols
        X = df[feat_cols].ffill().bfill().fillna(0.0).values.astype(float)
        self._scaler_mean = X.mean(axis=0)
        self._scaler_scale = np.where(X.std(axis=0) < 1e-9, 1.0, X.std(axis=0))
        X_scaled = (X - self._scaler_mean) / self._scaler_scale

        backend, backend_name = self._build_backend(X_scaled.shape[1])
        metrics = backend.fit(
            X_scaled,
            n_epochs=n_epochs or self._n_epochs,
            lr=self._lr,
            kl_weight=self._kl_weight,
        )
        metrics.backend = backend_name
        z_path = backend.encode(X_scaled)
        self._fit_prototypes(z_path, df[feat_cols[0]].to_numpy())
        self._backend = backend
        self._backend_name = backend_name
        self._train_metrics = metrics
        self._last_trained = datetime.now().isoformat()
        return metrics

    def predict_embedding(self, macro_df: pl.DataFrame) -> np.ndarray:
        """Return continuous regime embedding for the latest observation."""
        if self._backend is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        X_scaled, _ = self._macro_to_matrix(macro_df)
        z_path = self._backend.encode(X_scaled)
        return np.asarray(z_path[-1], dtype=float)

    def predict_probabilities(self, macro_df: pl.DataFrame) -> dict[str, float]:
        """Softmax weights over regime prototypes in embedding space."""
        emb = self.predict_embedding(macro_df)
        if not self._prototypes:
            return {label: 1.0 / len(_REGIME_LABELS) for label in _REGIME_LABELS}
        labels = list(self._prototypes.keys())
        dists = np.array(
            [np.sum((emb - self._prototypes[label]) ** 2) for label in labels],
            dtype=float,
        )
        logits = -dists
        logits -= logits.max()
        exp = np.exp(logits)
        probs = exp / exp.sum()
        return {label: float(p) for label, p in zip(labels, probs)}

    def predict_regime(self, macro_df: pl.DataFrame) -> str:
        """Discrete regime label via nearest prototype (shadow / logging)."""
        probs = self.predict_probabilities(macro_df)
        return max(probs, key=probs.get)

    def compute_elbo(self, macro_df: pl.DataFrame) -> float:
        if self._backend is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        X_scaled, _ = self._macro_to_matrix(macro_df)
        return float(self._backend.elbo(X_scaled, kl_weight=self._kl_weight))

    def compare_hmm_elbo(
        self,
        macro_df: pl.DataFrame,
        *,
        hmm: RegimeModel | None = None,
    ) -> dict[str, float]:
        """Shadow gating helper: DSS ELBO vs HMM log-likelihood proxy."""
        dss_elbo = self.compute_elbo(macro_df)
        hmm_model = hmm or RegimeModel()
        if hmm_model._model is None:
            hmm_model.fit(macro_df)
        X_scaled, _ = hmm_model._prepare_X(macro_df)
        if hasattr(hmm_model._model, "score"):
            hmm_score = float(hmm_model._model.score(X_scaled))
        else:
            hmm_score = float(-np.mean((X_scaled - X_scaled.mean(axis=0)) ** 2))
        return {
            "dss_elbo": dss_elbo,
            "hmm_loglik_proxy": hmm_score,
            "elbo_delta_pct": (
                100.0 * (dss_elbo - hmm_score) / abs(hmm_score)
                if abs(hmm_score) > 1e-12
                else 0.0
            ),
        }

    def get_regime_history(self, macro_df: pl.DataFrame) -> pd.DataFrame:
        if self._backend is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        X_scaled, df = self._macro_to_matrix(macro_df)
        z_path = self._backend.encode(X_scaled)
        result = df[["valid_time"]].copy().reset_index(drop=True)
        for j in range(z_path.shape[1]):
            result[f"z_{j}"] = z_path[:, j]
        labels = []
        prob_rows: list[dict[str, float]] = []
        for i in range(len(z_path)):
            emb = z_path[i]
            if not self._prototypes:
                probs = {label: 1.0 / 3 for label in _REGIME_LABELS}
            else:
                dists = {
                    label: np.sum((emb - proto) ** 2)
                    for label, proto in self._prototypes.items()
                }
                logits = -np.array(list(dists.values()))
                logits -= logits.max()
                exp = np.exp(logits)
                exp /= exp.sum()
                probs = dict(zip(dists.keys(), exp.tolist()))
            prob_rows.append(probs)
            labels.append(max(probs, key=probs.get))
        result["regime"] = labels
        for label in _REGIME_LABELS:
            result[f"prob_{label}"] = [row.get(label, 0.0) for row in prob_rows]
        return result

    def shadow_record(self, macro_df: pl.DataFrame) -> ShadowRegimeRecord:
        """Bundle shadow outputs for logging without touching council weights."""
        return ShadowRegimeRecord(
            regime_label=self.predict_regime(macro_df),
            embedding=self.predict_embedding(macro_df),
            probabilities=self.predict_probabilities(macro_df),
            elbo=self.compute_elbo(macro_df) if self._backend is not None else None,
            backend=self._backend_name,
            metadata=self.get_metadata(),
        )

    def save(self, path: str) -> None:
        p = Path(path)
        with open(p, "wb") as f:
            pickle.dump(self, f)
        digest = hashlib.sha256(p.read_bytes()).hexdigest()
        p.with_suffix(p.suffix + ".hash").write_text(digest)

    def load(self, path: str, *, require_hash: bool = True) -> None:
        from council.pickle_security import trusted_pickle_load

        saved = trusted_pickle_load(path, require_hash=require_hash)
        self.__dict__.update(saved.__dict__)

    def get_metadata(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "embed_dim": self.embed_dim,
            "last_trained": self._last_trained,
            "feature_cols": self._feature_cols,
            "backend": self._backend_name,
            "prototypes": {k: v.tolist() for k, v in self._prototypes.items()},
            "train_metrics": (
                None
                if self._train_metrics is None
                else {
                    "elbo_final": self._train_metrics.elbo_final,
                    "elbo_initial": self._train_metrics.elbo_initial,
                    "elbo_improvement_pct": self._train_metrics.elbo_improvement_pct,
                    "backend": self._train_metrics.backend,
                }
            ),
        }

    def regime_centroids(self) -> dict[str, np.ndarray]:
        """Prototype vectors for council embedding mode."""
        return {k: np.asarray(v, dtype=float) for k, v in self._prototypes.items()}
