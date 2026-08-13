"""Temporal Fusion Transformer (TFT) alpha challenger — shadow mode only.

Lightweight PyTorch implementation inspired by Lim et al. (2021): variable
selection gates, GRU encoder, scaled dot-product attention, and pinball loss
for quantile outputs. Does **not** wire into the daily Dagster council path;
signals are written to ``data/results/tft_shadow_signals.parquet`` for
walk-forward promotion (T1.1) and offline comparison vs LightGBM champion.

Inference target (documented): <300 ms CPU for a typical daily batch
(~20 tickers × 30 features × encoder_length 20) on a single core after fit.
Canary status: shadow — target: P-1.1 — expiry: 2027-02-01 (promote via canary o retire)
"""

from __future__ import annotations

import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import polars as pl
import yaml

from .base import BaseModel

logger = logging.getLogger(__name__)

_EXCLUDE_COLS = frozenset({"ticker", "valid_time", "transaction_time", "arrival_time"})
_SHADOW_SIGNALS_PATH = "data/results/tft_shadow_signals.parquet"

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    _TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised when torch optional
    _TORCH_AVAILABLE = False
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]


def _require_torch() -> None:
    if not _TORCH_AVAILABLE:
        raise ImportError(
            "PyTorch is required for TFT. Install with: pip install torch>=2.0"
        )


# ---------------------------------------------------------------------------
# TFT-inspired network (PyTorch, CPU-friendly)
# ---------------------------------------------------------------------------


if _TORCH_AVAILABLE:

    class _VariableSelection(nn.Module):
        """Gated feature selection (softmax weights per timestep)."""

        def __init__(self, n_features: int, hidden: int) -> None:
            super().__init__()
            self.grn = nn.Sequential(
                nn.Linear(n_features, hidden),
                nn.ELU(),
                nn.Linear(hidden, n_features),
            )

        def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            # x: (batch, seq, features)
            gates = torch.softmax(self.grn(x), dim=-1)
            selected = x * gates
            return selected, gates

    class _TFTCore(nn.Module):
        def __init__(
            self,
            n_features: int,
            hidden_size: int,
            num_layers: int,
            dropout: float,
            n_quantiles: int,
        ) -> None:
            super().__init__()
            self.vsn = _VariableSelection(n_features, hidden_size)
            self.encoder = nn.GRU(
                n_features,
                hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0,
            )
            self.attn_q = nn.Linear(hidden_size, hidden_size)
            self.attn_k = nn.Linear(hidden_size, hidden_size)
            self.attn_v = nn.Linear(hidden_size, hidden_size)
            self.head = nn.Linear(hidden_size, n_quantiles)

        def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            x_sel, gates = self.vsn(x)
            enc_out, _ = self.encoder(x_sel)
            # Self-attention over time (single head)
            q = self.attn_q(enc_out)
            k = self.attn_k(enc_out)
            v = self.attn_v(enc_out)
            scores = torch.matmul(q, k.transpose(-2, -1)) / (q.size(-1) ** 0.5)
            weights = torch.softmax(scores, dim=-1)
            context = torch.matmul(weights, v)
            # Use last timestep representation
            h = context[:, -1, :]
            quantiles = self.head(h)
            return quantiles, gates.mean(dim=(0, 1))

    def _pinball_loss(
        pred: torch.Tensor, target: torch.Tensor, quantiles: torch.Tensor
    ) -> torch.Tensor:
        """Multi-quantile pinball loss; pred shape (batch, n_quantiles)."""
        errors = target.unsqueeze(-1) - pred
        loss = torch.maximum(quantiles * errors, (quantiles - 1.0) * errors)
        return loss.mean()


class TemporalFusionAlpha(BaseModel):
    """TFT-style panel sequence model for cross-sectional alpha (shadow challenger)."""

    name = "tft"

    def __init__(self, config_path: str = "config/models.yaml") -> None:
        _require_torch()
        with open(config_path) as f:
            cfg = yaml.safe_load(f) or {}
        self._params: dict[str, Any] = dict(cfg.get("tft", {}))
        self._feature_cols: list[str] | None = None
        self._n_features: int | None = None
        self._last_trained: str | None = None
        self._net: Any = None
        self._device = torch.device("cpu")
        self._quantile_levels: list[float] = list(
            self._params.get("quantiles", [0.05, 0.5, 0.95])
        )
        self._median_idx = int(
            np.argmin(np.abs(np.array(self._quantile_levels) - 0.5))
        )
        self._selection_weights: pd.Series | None = None
        self._train_loss_history: list[float] = []

    # ------------------------------------------------------------------
    # Data preparation
    # ------------------------------------------------------------------

    def _feature_cols_from(self, features: pl.DataFrame) -> list[str]:
        return [c for c in features.columns if c not in _EXCLUDE_COLS]

    def _panel_to_numpy(
        self, features: pl.DataFrame, targets: pd.Series | None = None
    ) -> tuple[np.ndarray, np.ndarray | None, list[tuple[str, Any]]]:
        """Build sequence tensors and row keys (ticker, valid_time) for each sample."""
        feat_cols = self._feature_cols_from(features)
        df = features.select(["ticker", "valid_time"] + feat_cols).to_pandas()
        if pd.api.types.is_datetime64_any_dtype(df["valid_time"]):
            df["valid_time"] = df["valid_time"].dt.date
        else:
            df["valid_time"] = pd.to_datetime(df["valid_time"]).dt.date

        if targets is not None:
            if isinstance(targets.index, pd.MultiIndex):
                t = targets.reset_index()
                t.columns = ["ticker", "valid_time", "__target__"]
                if pd.api.types.is_datetime64_any_dtype(t["valid_time"]):
                    t["valid_time"] = t["valid_time"].dt.date
                else:
                    t["valid_time"] = pd.to_datetime(t["valid_time"]).dt.date
                df = df.merge(t, on=["ticker", "valid_time"], how="inner")
            else:
                df["__target__"] = targets.values

        encoder_len = int(self._params.get("encoder_length", 20))
        min_history = int(self._params.get("min_history", 5))

        seqs: list[np.ndarray] = []
        ys: list[float] = []
        keys: list[tuple[str, Any]] = []

        for ticker, grp in df.groupby("ticker", sort=False):
            grp = grp.sort_values("valid_time")
            mat = grp[feat_cols].astype(np.float32).fillna(0.0).values
            if len(mat) < min_history:
                continue
            tgt_col = "__target__" if "__target__" in grp.columns else None
            for i in range(encoder_len - 1, len(mat)):
                window = mat[i - encoder_len + 1 : i + 1]
                if window.shape[0] < encoder_len:
                    continue
                seqs.append(window)
                keys.append((ticker, grp["valid_time"].iloc[i]))
                if tgt_col is not None:
                    val = grp[tgt_col].iloc[i]
                    if pd.notna(val):
                        ys.append(float(val))
                    else:
                        ys.append(np.nan)
        X = np.stack(seqs, axis=0) if seqs else np.zeros((0, encoder_len, len(feat_cols)))
        y = np.array(ys, dtype=np.float32) if ys else None
        return X, y, keys

    @staticmethod
    def _zscore_series(s: pd.Series) -> pd.Series:
        std = s.std()
        if std == 0 or np.isnan(std):
            return s - s.mean()
        return (s - s.mean()) / std

    # ------------------------------------------------------------------
    # fit / predict
    # ------------------------------------------------------------------

    def fit(self, features: pl.DataFrame, targets: pd.Series) -> None:
        _require_torch()
        self._feature_cols = self._feature_cols_from(features)
        self._n_features = len(self._feature_cols)

        X, y, _ = self._panel_to_numpy(features, targets)
        if len(X) == 0 or y is None:
            raise ValueError("Insufficient sequence samples for TFT fit")

        mask = ~np.isnan(y)
        X, y = X[mask], y[mask]
        if len(X) < 10:
            raise ValueError(f"Need at least 10 training sequences, got {len(X)}")

        hidden = int(self._params.get("hidden_size", 64))
        layers = int(self._params.get("num_layers", 1))
        dropout = float(self._params.get("dropout", 0.1))
        lr = float(self._params.get("learning_rate", 1e-3))
        epochs = int(self._params.get("max_epochs", 30))
        batch_size = int(self._params.get("batch_size", 256))

        n_q = len(self._quantile_levels)
        self._net = _TFTCore(
            n_features=self._n_features,
            hidden_size=hidden,
            num_layers=layers,
            dropout=dropout,
            n_quantiles=n_q,
        ).to(self._device)

        q_tensor = torch.tensor(self._quantile_levels, dtype=torch.float32, device=self._device)
        optimizer = torch.optim.Adam(self._net.parameters(), lr=lr)
        Xt = torch.from_numpy(X).to(self._device)
        yt = torch.from_numpy(y).to(self._device)

        self._net.train()
        self._train_loss_history = []
        for _ in range(epochs):
            perm = torch.randperm(len(Xt))
            epoch_loss = 0.0
            n_batches = 0
            for start in range(0, len(Xt), batch_size):
                idx = perm[start : start + batch_size]
                xb = Xt[idx]
                yb = yt[idx]
                optimizer.zero_grad()
                pred, gates = self._net(xb)
                loss = _pinball_loss(pred, yb, q_tensor)
                loss.backward()
                optimizer.step()
                epoch_loss += float(loss.item())
                n_batches += 1
            self._train_loss_history.append(epoch_loss / max(n_batches, 1))

        self._net.eval()
        with torch.no_grad():
            _, gates = self._net(Xt[: min(512, len(Xt))])
            weights = gates.detach().cpu().numpy()
            self._selection_weights = pd.Series(
                weights,
                index=self._feature_cols,
                name="vsn_weight",
            ).sort_values(ascending=False)

        self._last_trained = datetime.now().isoformat()
        logger.info(
            "TFT fit complete: %d sequences, final loss=%.4f",
            len(X),
            self._train_loss_history[-1] if self._train_loss_history else 0.0,
        )

    def predict(self, features: pl.DataFrame) -> pd.Series:
        if self._net is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        _require_torch()

        feat_cols = list(self._feature_cols or self._feature_cols_from(features))
        present = [c for c in feat_cols if c in features.columns]
        missing = [c for c in feat_cols if c not in features.columns]
        subset = ["ticker", "valid_time"] + present
        df_pl = features.select(subset)
        if missing:
            for c in missing:
                df_pl = df_pl.with_columns(pl.lit(0.0).alias(c))
            df_pl = df_pl.select(["ticker", "valid_time"] + feat_cols)

        X, _, keys = self._panel_to_numpy(df_pl, targets=None)
        if len(X) == 0:
            return pd.Series(dtype=float)

        self._net.eval()
        scores: list[float] = []
        tickers: list[str] = []
        dates: list[Any] = []
        batch_size = int(self._params.get("batch_size", 512))

        with torch.no_grad():
            for start in range(0, len(X), batch_size):
                xb = torch.from_numpy(X[start : start + batch_size]).to(self._device)
                pred, _ = self._net(xb)
                median = pred[:, self._median_idx].cpu().numpy()
                scores.extend(median.tolist())
                for ticker, dt in keys[start : start + batch_size]:
                    tickers.append(ticker)
                    dates.append(dt)

        out = pd.DataFrame({"ticker": tickers, "valid_time": dates, "score": scores})
        out["score_z"] = out.groupby("valid_time")["score"].transform(self._zscore_series)
        return pd.Series(out["score_z"].values, index=pd.Index(out["ticker"].values, name="ticker"))

    def get_selection_weights(self) -> pd.Series:
        """Top variable-selection gate weights (for interpretability gating)."""
        if self._selection_weights is None:
            raise RuntimeError("No selection weights. Call fit() first.")
        return self._selection_weights.copy()

    def measure_inference_latency_ms(
        self, features: pl.DataFrame, *, n_warmup: int = 2
    ) -> float:
        """Wall-clock median inference latency in milliseconds (CPU)."""
        for _ in range(n_warmup):
            self.predict(features)
        t0 = time.perf_counter()
        self.predict(features)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return float(elapsed_ms)


def write_shadow_signals(
    signals: pd.DataFrame,
    path: str | Path | None = None,
) -> Path:
    """Persist shadow TFT signals for walk-forward CI (wide: dates × tickers).

    Parameters
    ----------
    signals:
        DataFrame indexed by ``valid_time`` (or with ``valid_time`` column) and
        columns = tickers, values = cross-sectional z-scores. Alternatively pass
        long format with columns ``ticker``, ``valid_time``, ``signal``.
    path:
        Destination parquet; default ``data/results/tft_shadow_signals.parquet``.
    """
    dest = Path(path or _SHADOW_SIGNALS_PATH)
    dest.parent.mkdir(parents=True, exist_ok=True)

    if (
        isinstance(signals.index, pd.DatetimeIndex)
        and len(signals.columns) > 0
    ):
        wide = signals.copy()
    elif {"ticker", "valid_time"}.issubset(signals.columns):
        wide = signals.pivot(index="valid_time", columns="ticker", values="signal")
        wide.index = pd.to_datetime(wide.index)
    else:
        raise ValueError(
            "signals must be wide (DatetimeIndex × tickers) or long "
            "(ticker, valid_time, signal)"
        )

    wide = wide.sort_index()
    wide.to_parquet(dest)
    logger.info("Wrote TFT shadow signals: %s (%d rows)", dest, len(wide))
    return dest


def build_shadow_signal_matrix(
    features: pl.DataFrame,
    model: TemporalFusionAlpha,
) -> pd.DataFrame:
    """Batch-predict sequence endpoints and pivot to walk-forward wide matrix."""
    if model._net is None:
        raise RuntimeError("Model not fitted. Call fit() first.")

    feat_cols = list(model._feature_cols or model._feature_cols_from(features))
    subset = ["ticker", "valid_time"] + [c for c in feat_cols if c in features.columns]
    pl_feat = features.select(subset)
    for c in feat_cols:
        if c not in pl_feat.columns:
            pl_feat = pl_feat.with_columns(pl.lit(0.0).alias(c))

    X, _, keys = model._panel_to_numpy(pl_feat, targets=None)
    if len(X) == 0:
        return pd.DataFrame()

    model._net.eval()
    batch_size = int(model._params.get("batch_size", 512))
    scores: list[float] = []
    with torch.no_grad():
        for start in range(0, len(X), batch_size):
            xb = torch.from_numpy(X[start : start + batch_size]).to(model._device)
            pred, _ = model._net(xb)
            median = pred[:, model._median_idx].cpu().numpy()
            scores.extend(median.tolist())

    long = pd.DataFrame(
        {
            "ticker": [k[0] for k in keys],
            "valid_time": [k[1] for k in keys],
            "score": scores,
        }
    )
    long["score_z"] = long.groupby("valid_time")["score"].transform(
        TemporalFusionAlpha._zscore_series
    )
    wide = long.pivot(index="valid_time", columns="ticker", values="score_z")
    wide.index = pd.to_datetime(wide.index)
    wide.index.name = "valid_time"
    return wide.astype(float)
