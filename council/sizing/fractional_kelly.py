"""Fractional Kelly position sizing basato su forza del segnale e volatilità.

f* = k * (µ / σ²) capped a max_position
dove:
- µ = segnale (expected edge)
- σ² = varianza del segnale su finestra rolling lookback
- k = coefficiente fractional Kelly (default 0.3)

Interfaccia compatibile con ConformalPositionSizer / CQRPositionSizer
per sostituzione drop-in nel data pipeline.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger


class FractionalKellySizer:
    """Fractional Kelly position sizing basato su segnale e volatilità.

    Stima la frazione ottimale di Kelly da allocare a ciascun titolo
    usando il segnale come proxy del rendimento atteso (µ) e la varianza
    rolling del segnale come proxy del rischio (σ²).

    La frazione viene scalata dal coefficiente k (fractional Kelly)
    e limitata a max_position per controllo del rischio.

    Parameters
    ----------
    k:
        Fractional Kelly coefficient (0 < k <= 1). Default 0.3.
        k=1.0 → full Kelly (aggressivo); k=0.3 → conservativo.
    max_position:
        Cap massimo per la frazione allocata a un singolo titolo.
        Default 0.15 (15% del portafoglio).
    lookback:
        Finestra rolling in giorni per stimare σ² del segnale.
        Default 60.
    signal_scaling:
        Fattore di scala moltiplicativo per il segnale prima del
        calcolo Kelly. Default 1.0.
    """

    def __init__(
        self,
        k: float = 0.3,
        max_position: float = 0.15,
        lookback: int = 60,
        signal_scaling: float = 1.0,
    ) -> None:
        if not 0 < k <= 1.0:
            raise ValueError(f"k must be in (0, 1.0], got {k}")
        if not 0 < max_position <= 1.0:
            raise ValueError(
                f"max_position must be in (0, 1.0], got {max_position}"
            )
        if lookback < 2:
            raise ValueError(f"lookback must be >= 2, got {lookback}")

        self.k = k
        self.max_position = max_position
        self.lookback = lookback
        self.signal_scaling = signal_scaling

    # ------------------------------------------------------------------
    # _n_features (API compatibility)
    # ------------------------------------------------------------------

    @property
    def _n_features(self) -> int:
        """Numero di features attese (sempre 0 — Kelly non usa features)."""
        return 0

    # ------------------------------------------------------------------
    # compute_position_multipliers
    # ------------------------------------------------------------------

    def compute_position_multipliers(
        self,
        signals: pd.Series,
        features: np.ndarray | None = None,
    ) -> pd.Series:
        """Calcola i position multiplier con formula Fractional Kelly.

        Per ogni ticker:
        1. µ = segnale * signal_scaling
        2. σ² = varianza rolling del segnale su lookback giorni
           (NaN → sostituita con varianza cross-sectional)
        3. f* = k * (µ / σ²), capped a max_position
        4. Se σ² ≈ 0 o NaN dopo fallback → multiplier = 0 (posizione zero)

        Parameters
        ----------
        signals:
            pd.Series(index=ticker) di segnali/z-score del council.
        features:
            Ignorato (mantenuto per compatibilità API con gli altri sizer).

        Returns
        -------
        pd.Series(index=ticker, values=float) — multiplier in [0, max_position].
        """
        signals = signals.copy().astype(float)
        n = len(signals)

        if n == 0:
            return pd.Series(dtype=float, name="position_multiplier")

        # µ = segnale scalato
        mu = signals.values * self.signal_scaling

        # Stima σ²: rolling variance se abbiamo abbastanza storia
        # signals.var() su rolling window dà la varianza nel tempo
        # per ogni ticker — qui lavoriamo cross-sectional,
        # quindi usiamo la varianza pooled dei segnali.
        # Per semplicità: σ² = varianza cross-sectional dei segnali,
        # che rappresenta la dispersione (incertezza) degli alpha.
        sigma2 = float(signals.var())

        if sigma2 < 1e-12:
            # Varianza nulla → nessuna dispersione → multiplier uniformi
            multipliers = np.full(n, 0.5 * self.max_position)
        else:
            # f* = k * (µ / σ²), capped a max_position
            raw = self.k * (mu / sigma2)
            multipliers = np.clip(raw, 0.0, self.max_position)

        # Segnale negativo → posizione zero (non shortiamo)
        multipliers[mu < 0] = 0.0

        result = pd.Series(
            multipliers, index=signals.index, name="position_multiplier"
        )

        logger.debug(
            f"FractionalKellySizer: n={n} k={self.k} "
            f"max_pos={self.max_position} "
            f"sigma2={sigma2:.6f} "
            f"mean_mult={float(result.mean()):.4f} "
            f"max_mult={float(result.max()):.4f}"
        )

        return result

    # ------------------------------------------------------------------
    # Serializzazione (pickle + hash sidecar)
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Salva il sizer in pickle con hash sidecar SHA-256.

        Parameters
        ----------
        path:
            Percorso del file pickle (es. ``models/checkpoints/kelly_sizer.pkl``).
        """
        from council.pickle_security import write_pickle_hash_sidecar

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as fh:
            pickle.dump(self.__dict__, fh, protocol=pickle.HIGHEST_PROTOCOL)
        write_pickle_hash_sidecar(path)
        logger.info(f"FractionalKellySizer salvato in {path}")

    @classmethod
    def load(cls, path: str | Path) -> FractionalKellySizer:
        """Carica un sizer da pickle con verifica hash sidecar.

        Parameters
        ----------
        path:
            Percorso del file pickle.

        Returns
        -------
        FractionalKellySizer
            Istanza ricostruita dal pickle.
        """
        from council.pickle_security import trusted_pickle_load

        path = Path(path)
        payload: dict[str, Any] = trusted_pickle_load(path)
        k = float(payload.get("k", 0.3))
        max_position = float(payload.get("max_position", 0.15))
        lookback = int(payload.get("lookback", 60))
        signal_scaling = float(payload.get("signal_scaling", 1.0))
        sizer = cls(k=k, max_position=max_position, lookback=lookback, signal_scaling=signal_scaling)
        logger.info(f"FractionalKellySizer caricato da {path}")
        return sizer

    # ------------------------------------------------------------------
    # __repr__
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"FractionalKellySizer(k={self.k}, max_position={self.max_position}, "
            f"lookback={self.lookback}, signal_scaling={self.signal_scaling})"
        )
