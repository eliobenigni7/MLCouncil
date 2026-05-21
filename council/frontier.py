"""Frontier mode helpers — wire all Wave 1–3 env flags into the daily pipeline."""

from __future__ import annotations

import os
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import polars as pl
from loguru import logger

_TRUTHY = frozenset({"1", "true", "yes", "on"})
_ROOT = Path(__file__).resolve().parents[1]
_CHECKPOINTS = _ROOT / "models" / "checkpoints"
_TFT_SHADOW = _ROOT / "data" / "results" / "tft_shadow_signals.parquet"


def frontier_profile_active() -> bool:
    return os.getenv("MLCOUNCIL_ENV_PROFILE", "").strip().lower() == "frontier"


def tft_in_council_enabled() -> bool:
    from council.production_config import expert_enabled, manifest_enabled

    if manifest_enabled():
        return expert_enabled("tft")
    raw = os.getenv("MLCOUNCIL_TFT_IN_COUNCIL", "").strip().lower()
    if raw in _TRUTHY:
        return True
    return frontier_profile_active() and raw != "false"


def use_stacked_council_signal() -> bool:
    from council.production_config import manifest_enabled, use_stacked_council

    if manifest_enabled():
        return use_stacked_council()
    raw = os.getenv("MLCOUNCIL_USE_STACKED_COUNCIL", "").strip().lower()
    if raw in _TRUTHY:
        return True
    return frontier_profile_active() and raw != "false"


def load_regime_context(
    raw_macro: pl.DataFrame,
    regime_label: str,
) -> tuple[np.ndarray | None, dict[str, np.ndarray] | None]:
    """Load DSS regime embedding when ``MLCOUNCIL_REGIME_MODE=embedding``."""
    from council.aggregator import regime_mode

    if regime_mode() != "embedding":
        return None, None

    checkpoint = _CHECKPOINTS / "regime_dss_latest.pkl"
    if not checkpoint.exists():
        logger.warning("regime embedding requested but regime_dss_latest.pkl missing")
        return None, None

    try:
        from models.regime_dss import DeepRegimeModel

        model = DeepRegimeModel.load(checkpoint)
        embedding = model.predict_embedding(raw_macro)
        centroids = model.regime_centroids()
        return np.asarray(embedding, dtype=float), centroids
    except Exception as exc:
        logger.warning(f"regime DSS embedding failed: {exc}")
        return None, None


def load_tft_expert_signals(
    tickers: list[str],
    partition_date: str,
) -> pd.Series | None:
    """Load TFT z-scores for partition day (parquet shadow matrix or checkpoint)."""
    if not tft_in_council_enabled():
        return None

    day = pd.Timestamp(partition_date)
    if _TFT_SHADOW.exists():
        try:
            wide = pd.read_parquet(_TFT_SHADOW)
            wide.index = pd.to_datetime(wide.index)
            if day in wide.index:
                row = wide.loc[day].reindex(tickers).fillna(0.0)
                row.name = "tft"
                return row.astype(float)
            nearest = wide.index[wide.index <= day]
            if len(nearest) > 0:
                row = wide.loc[nearest[-1]].reindex(tickers).fillna(0.0)
                row.name = "tft"
                return row.astype(float)
        except Exception as exc:
            logger.debug(f"TFT shadow parquet read failed: {exc}")

    ckpt = _CHECKPOINTS / "tft_challenger.pkl"
    if not ckpt.exists():
        return None
    try:
        from council.pickle_security import trusted_pickle_load
        from models.tft import TemporalFusionAlpha

        model: TemporalFusionAlpha = trusted_pickle_load(ckpt, require_hash=False)
        # Minimal stub: zero signal if no feature panel passed
        return pd.Series(0.0, index=tickers, name="tft")
    except Exception as exc:
        logger.warning(f"TFT council expert unavailable: {exc}")
        return None


def load_microstructure_signals(tickers: list[str]) -> pd.Series | None:
    from council.production_config import expert_enabled, manifest_enabled
    from models.microstructure import MicrostructureModel, microstructure_promoted

    if manifest_enabled():
        if not expert_enabled("microstructure"):
            return None
    elif not microstructure_promoted():
        return None
    try:
        model = MicrostructureModel()
        return model.predict(tickers=tickers)
    except Exception as exc:
        logger.warning(f"microstructure signals failed: {exc}")
        return None


def enrich_council_experts(
    signals: dict[str, pd.Series],
    *,
    tickers: list[str],
    partition_date: str,
) -> dict[str, pd.Series]:
    """Add TFT / microstructure experts when frontier flags are on."""
    tft = load_tft_expert_signals(tickers, partition_date)
    if tft is not None and not tft.empty:
        signals["tft"] = tft

    micro = load_microstructure_signals(tickers)
    if micro is not None and not micro.empty:
        signals["microstructure"] = micro

    return signals


def apply_stacked_council_override(
    combined: pd.Series,
    expert_signals: dict[str, pd.Series],
    partition_date: str,
) -> pd.Series:
    """Replace council output with stacking meta-learner when enabled."""
    if not use_stacked_council_signal() or len(expert_signals) < 2:
        return combined

    from council.cqr import DEFAULT_STACKING_CHECKPOINT, StackingMetaLearner

    if not DEFAULT_STACKING_CHECKPOINT.exists():
        logger.warning(
            "MLCOUNCIL_USE_STACKED_COUNCIL=true but stacking_meta.pkl missing; "
            "run scripts/bootstrap_frontier.py"
        )
        return combined

    try:
        meta = StackingMetaLearner.load(DEFAULT_STACKING_CHECKPOINT)
        base_df = pd.DataFrame({m: s for m, s in expert_signals.items()}).fillna(0.0)
        stacked = meta.predict(base_df).rename("council_signal")
        logger.info(
            f"frontier [{partition_date}]: council signal from stacking meta-learner"
        )
        return stacked
    except Exception as exc:
        logger.warning(f"stacked council override failed: {exc}")
        return combined
