"""Mixture-of-Experts gating for council signal aggregation (T3.1 shadow).

Non-linear expert weights: y_hat = sum_k g_k(x) * f_k(x), sum_k g_k = 1.
Production default remains linear aggregation in ``CouncilAggregator``;
enable via ``MLCOUNCIL_AGGREGATOR_MODE=moe``.
"""

from __future__ import annotations

import os
import pickle
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from loguru import logger

DEFAULT_MOE_CHECKPOINT = Path(__file__).resolve().parents[1] / "models" / "checkpoints" / "moe_gate.pkl"
SHADOW_MOE_DIR = Path(__file__).resolve().parents[1] / "data" / "results" / "shadow_moe"

_REGIME_INDEX = {"bull": 0, "bear": 1, "transition": 2}


def aggregator_mode() -> str:
    """``linear`` (default) or ``moe`` shadow gating."""
    raw = os.getenv("MLCOUNCIL_AGGREGATOR_MODE", "linear").strip().lower()
    return raw if raw in ("linear", "moe") else "linear"


def build_regime_context(
    regime: str,
    expert_ic: dict[str, float] | None = None,
) -> np.ndarray:
    """Context vector: one-hot regime (3) + mean IC (1)."""
    one_hot = np.zeros(3, dtype=float)
    idx = _REGIME_INDEX.get(regime.strip().lower(), 2)
    one_hot[idx] = 1.0
    ic_vals = list((expert_ic or {}).values())
    mean_ic = float(np.mean(ic_vals)) if ic_vals else 0.0
    return np.concatenate([one_hot, [mean_ic]])


class MoEGatingNetwork:
    """Softmax gating over expert model signals (numpy scaffold).

    Parameters
    ----------
    n_experts:
        Number of council models (experts).
    context_dim:
        Input dimension for gating logits (default 4 = regime OH + mean IC).
    temperature:
        Softmax temperature; higher → smoother expert mix.
    seed:
        RNG seed for initial gate weights (shadow / untrained scaffold).
    """

    def __init__(
        self,
        n_experts: int,
        *,
        context_dim: int = 4,
        temperature: float = 1.0,
        seed: int = 42,
    ) -> None:
        if n_experts < 1:
            raise ValueError(f"n_experts must be >= 1, got {n_experts}")
        self.n_experts = n_experts
        self.context_dim = context_dim
        self.temperature = max(float(temperature), 1e-6)
        rng = np.random.default_rng(seed)
        # Small random init until ``scripts/train_moe_gating.py`` writes a checkpoint.
        self._gate_weights = rng.normal(0.0, 0.05, size=(context_dim, n_experts))

    @property
    def gate_weight_matrix(self) -> np.ndarray:
        return np.asarray(self._gate_weights, dtype=float)

    def set_gate_weights(self, weights: np.ndarray) -> None:
        w = np.asarray(weights, dtype=float)
        if w.shape != (self.context_dim, self.n_experts):
            raise ValueError(
                f"expected gate shape ({self.context_dim}, {self.n_experts}), got {w.shape}"
            )
        self._gate_weights = w

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as fh:
            pickle.dump(
                {
                    "n_experts": self.n_experts,
                    "context_dim": self.context_dim,
                    "temperature": self.temperature,
                    "gate_weights": self._gate_weights,
                },
                fh,
            )
        logger.info(f"MoEGatingNetwork saved → {path}")

    @classmethod
    def load(cls, path: str | Path) -> "MoEGatingNetwork":
        path = Path(path)
        with open(path, "rb") as fh:
            payload = pickle.load(fh)
        net = cls(
            int(payload["n_experts"]),
            context_dim=int(payload["context_dim"]),
            temperature=float(payload.get("temperature", 1.0)),
        )
        net.set_gate_weights(np.asarray(payload["gate_weights"], dtype=float))
        return net

    @classmethod
    def load_or_create(
        cls,
        n_experts: int,
        path: str | Path | None = None,
        *,
        context_dim: int = 4,
    ) -> "MoEGatingNetwork":
        ckpt = Path(path) if path is not None else DEFAULT_MOE_CHECKPOINT
        if ckpt.exists():
            net = cls.load(ckpt)
            if net.n_experts != n_experts:
                logger.warning(
                    f"MoE checkpoint n_experts={net.n_experts} != {n_experts}; using fresh init"
                )
                return cls(n_experts, context_dim=context_dim)
            return net
        return cls(n_experts, context_dim=context_dim)

    def gate_weights(self, context: Sequence[float] | np.ndarray) -> np.ndarray:
        """Return softmax expert weights summing to 1."""
        ctx = np.asarray(context, dtype=float).ravel()
        if ctx.shape[0] != self.context_dim:
            raise ValueError(
                f"context length {ctx.shape[0]} != context_dim {self.context_dim}"
            )
        logits = ctx @ self._gate_weights
        logits = logits / self.temperature
        logits -= float(np.max(logits))
        exp_logits = np.exp(logits)
        denom = float(exp_logits.sum())
        if denom < 1e-12:
            return np.full(self.n_experts, 1.0 / self.n_experts)
        return exp_logits / denom

    def combine_signals(
        self,
        signals: dict[str, pd.Series],
        expert_order: list[str],
        gate: np.ndarray,
        *,
        performance_weights: dict[str, float] | None = None,
    ) -> tuple[pd.Series, dict[str, float]]:
        """Blend expert z-scores with MoE gate × optional performance weights."""
        if len(gate) != len(expert_order):
            raise ValueError("gate length must match expert_order")

        perf = performance_weights or {}
        effective: dict[str, float] = {}
        for i, model in enumerate(expert_order):
            base = float(perf.get(model, 1.0))
            effective[model] = float(gate[i]) * base

        total = sum(effective.values()) or 1.0
        effective = {m: w / total for m, w in effective.items()}

        tickers = sorted(
            {t for m in expert_order if m in signals for t in signals[m].index}
        )
        combined = pd.Series(0.0, index=tickers)
        for model in expert_order:
            if model not in signals:
                continue
            sig = signals[model].reindex(tickers).fillna(0.0)
            combined += effective[model] * sig

        std = float(combined.std())
        if std > 1e-9:
            combined = (combined - combined.mean()) / std

        combined.index.name = "ticker"
        logger.debug(f"MoE combine: gate={gate.round(4).tolist()} effective={effective}")
        return combined, effective


def moe_enabled() -> bool:
    return aggregator_mode() == "moe"


def log_moe_shadow(
    partition_date: str,
    *,
    linear_signal: pd.Series,
    moe_signal: pd.Series,
    gate_weights: list[float] | None,
    expert_order: list[str] | None,
    effective_weights: dict[str, float] | None = None,
    out_dir: Path | None = None,
) -> Path:
    """Write MoE vs linear council comparison parquet (no production effect)."""
    out_dir = out_dir or SHADOW_MOE_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    idx = linear_signal.index.union(moe_signal.index)
    payload = pd.DataFrame(
        {
            "ticker": idx.astype(str),
            "linear_signal": linear_signal.reindex(idx).fillna(0.0).values,
            "moe_signal": moe_signal.reindex(idx).fillna(0.0).values,
            "gate_weights": [gate_weights] * len(idx) if gate_weights else [None] * len(idx),
            "expert_order": [expert_order] * len(idx) if expert_order else [None] * len(idx),
            "effective_weights": [effective_weights] * len(idx) if effective_weights else [None] * len(idx),
        }
    )
    path = out_dir / f"{partition_date}.parquet"
    payload.to_parquet(path, index=False)
    logger.info(f"MoE shadow logged → {path} ({len(payload)} rows)")
    return path
