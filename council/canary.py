"""Canary feature controller (F-0.4) — flag governance con revert automatico.

Implements the P2 principle (shadow → canary → production) and the P4 kill
switch: a feature is activated in ``config/canary.yaml`` only after the G1 gate
(owner decision); the controller applies the env vars as a **run policy** and,
when the daily council metric stays below ``floor`` for ``min_days`` consecutive
runs, it disables the feature persistently (``data/results/canary_state.json``)
and raises a health alert through ``council/monitoring/alerting.py`` (F-0.2 infrastructure,
log + dashboard state + email for CRITICAL).

Conceptual difference vs. the old switch-and-restore pattern removed in F-0.2:
``apply()`` must be called at the **start of a job run** (before any asset that
reads the flags executes) and env vars are never mutated mid-run. A revert only
disables the feature in the persisted state, so it takes effect from the next
run — no mid-run mutation, no restore dance.

Design notes
------------
* ``os.environ.setdefault`` semantics: an explicit operator env (or the
  production manifest applied at import time) always wins over the canary value.
* Revert is sticky (kill switch): after a revert the feature stays disabled even
  if ``config/canary.yaml`` is flipped back to ``enabled: true``; re-enabling
  requires resetting the feature entry in ``canary_state.json``.
* ``check_revert`` evaluates the trailing ``min_days`` history entries (each
  entry = one daily run; calendar gaps such as weekends do not count as an
  interruption). No data for a feature → no revert (wait for data).
* Loader and state persistence are best-effort: missing/empty/corrupt files
  degrade to an empty configuration / fresh state, never raising.
* With a multiprocess executor, env mutation inside one asset process does not
  propagate to other step processes: apply the flags at module import (like
  ``council.production_config.apply_manifest_to_environ``) before moving to a
  multiprocess deployment.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from loguru import logger

from council.monitoring.alerting import dispatch_health_alerts

_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_CONFIG_PATH = _ROOT / "config" / "canary.yaml"
_DEFAULT_STATE_PATH = _ROOT / "data" / "results" / "canary_state.json"


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class CanaryFeature:
    """Una feature canary dichiarata in ``config/canary.yaml``."""

    name: str
    env: str
    value: str
    enabled: bool = False
    floor: float = 0.0
    min_days: int = 5


def load_canary_config(path: str | Path | None = None) -> list[CanaryFeature]:
    """Load ``config/canary.yaml`` into a list of :class:`CanaryFeature`.

    Missing path, empty file, or corrupt YAML → empty list, **never raises**
    (the canary layer must stay zero-risk for the pipeline).
    """
    config_path = Path(path) if path is not None else _DEFAULT_CONFIG_PATH
    try:
        if not config_path.exists():
            return []
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return []
    if not isinstance(raw, dict):
        return []
    entries = raw.get("features")
    if not isinstance(entries, list):
        return []
    features: list[CanaryFeature] = []
    for entry in entries:
        feature = _parse_feature(entry)
        if feature is not None:
            features.append(feature)
    return features


def _parse_feature(entry: Any) -> CanaryFeature | None:
    """Parse a single YAML entry; malformed entries are skipped, never raised."""
    if not isinstance(entry, dict):
        return None
    name = entry.get("name")
    env = entry.get("env")
    if not isinstance(name, str) or not name or not isinstance(env, str) or not env:
        return None
    metrics = entry.get("metrics")
    metrics = metrics if isinstance(metrics, dict) else {}
    floor_raw = metrics.get("floor", entry.get("floor"))
    min_days_raw = metrics.get("min_days", entry.get("min_days"))
    floor = _to_float(floor_raw) if floor_raw is not None else 0.0
    min_days = _to_int(min_days_raw) if min_days_raw is not None else 5
    value = entry.get("value")
    return CanaryFeature(
        name=name,
        env=env,
        value=str(value) if value is not None else "true",
        enabled=bool(entry.get("enabled", False)),
        floor=floor if floor is not None else 0.0,
        min_days=max(1, min_days) if min_days is not None else 5,
    )


# ---------------------------------------------------------------------------
# Persisted state
# ---------------------------------------------------------------------------

class CanaryState:
    """Persistenza JSON dello stato canary (``data/results/canary_state.json``).

    Struttura::

        {
          "features": {"<name>": {"enabled": bool, "reverted_at": str|None,
                                  "revert_reason": str|None,
                                  "revert_value": float|None,
                                  "revert_floor": float|None}},
          "history":  {"<name>": [{"date": "YYYY-MM-DD", "value": float}, ...]}
        }
    """

    def __init__(self, config: list[CanaryFeature] | None = None) -> None:
        self.features: dict[str, dict[str, Any]] = {}
        self.history: dict[str, list[dict[str, Any]]] = {}
        for feature in config or []:
            self.features[feature.name] = {
                "enabled": feature.enabled,
                "reverted_at": None,
                "revert_reason": None,
                "revert_value": None,
                "revert_floor": None,
            }

    @classmethod
    def load(
        cls,
        path: str | Path,
        config: list[CanaryFeature] | None = None,
    ) -> "CanaryState":
        """Load state from disk; missing/corrupt file → fresh state (never raises)."""
        state = cls(config)
        try:
            path = Path(path)
            if not path.exists():
                return state
            data = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                return state
            stored_features = data.get("features")
            if isinstance(stored_features, dict):
                for name, entry in stored_features.items():
                    if isinstance(entry, dict) and name in state.features:
                        state.features[name].update(
                            {
                                key: entry[key]
                                for key in (
                                    "enabled",
                                    "reverted_at",
                                    "revert_reason",
                                    "revert_value",
                                    "revert_floor",
                                )
                                if key in entry
                            }
                        )
            stored_history = data.get("history")
            if isinstance(stored_history, dict):
                for name, entries in stored_history.items():
                    if isinstance(entries, list):
                        state.history[name] = [
                            e for e in entries if isinstance(e, dict)
                        ]
        except (OSError, json.JSONDecodeError):
            pass  # stato corrotto → riparte dalla config, mai eccezioni
        return state

    def is_enabled(self, name: str) -> bool:
        entry = self.features.get(name)
        return bool(entry["enabled"]) if entry else False

    def disable(
        self,
        name: str,
        *,
        reason: str,
        last_value: float,
        floor: float,
        date: Any,
    ) -> None:
        """Disabilita la feature (kill switch sticky) e registra il revert."""
        entry = self.features.setdefault(
            name,
            {
                "enabled": True,
                "reverted_at": None,
                "revert_reason": None,
                "revert_value": None,
                "revert_floor": None,
            },
        )
        entry.update(
            {
                "enabled": False,
                "reverted_at": str(date),
                "revert_reason": reason,
                "revert_value": float(last_value),
                "revert_floor": float(floor),
            }
        )

    def save(self, path: str | Path) -> None:
        """Persistenza best-effort: un errore di scrittura non deve far fallire la run."""
        try:
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            payload = {"features": self.features, "history": self.history}
            path.write_text(
                json.dumps(payload, indent=2, default=str), encoding="utf-8"
            )
        except OSError as exc:
            logger.warning(f"canary_state: persistenza fallita ({exc})")


# ---------------------------------------------------------------------------
# Revert events
# ---------------------------------------------------------------------------

@dataclass
class RevertEvent:
    """Evento di revert automatico di una feature canary (F-0.4)."""

    name: str
    reason: str
    last_value: float
    floor: float
    date: str

    def to_health_signal(self) -> dict[str, Any]:
        """Rendering compatibile con ``dispatch_health_alerts`` (livello "alert")."""
        return {
            "level": "alert",
            "value": self.last_value,
            "threshold": self.floor,
            "note": self.reason,
        }


# ---------------------------------------------------------------------------
# Controller
# ---------------------------------------------------------------------------

class CanaryController:
    """Controllore canary: apply a inizio run, record metriche, revert automatico.

    Parameters
    ----------
    config:
        Feature canary caricate da ``config/canary.yaml``.
    state_path:
        Percorso del JSON di stato (``data/results/canary_state.json`` di default).
    """

    def __init__(
        self,
        config: list[CanaryFeature],
        state_path: str | Path | None = None,
    ) -> None:
        self.config = config
        self.state_path = (
            Path(state_path) if state_path is not None else _DEFAULT_STATE_PATH
        )
        self.state = CanaryState.load(self.state_path, config=config)

    # -- attivazione (policy di run) --------------------------------------

    def _active_features(self) -> list[CanaryFeature]:
        """Feature abilitate in config E non revertite nello stato persistito."""
        return [f for f in self.config if f.enabled and self.state.is_enabled(f.name)]

    def apply(self) -> list[str]:
        """Applica l'env delle feature canary attive; ritorna i nomi applicati.

        Run policy: da chiamare a inizio job, PRIMA che gli asset che leggono i
        flag eseguano. Usa ``os.environ.setdefault`` — l'env esplicito
        dell'operatore (o il manifest di produzione) vince. Il revert
        disabilita la feature nello stato persistito: l'effetto si manifesta
        dalla run successiva (niente mutazione mid-run).
        """
        applied: list[str] = []
        for feature in self._active_features():
            os.environ.setdefault(feature.env, feature.value)
            applied.append(feature.name)
        return applied

    # -- metriche & revert -------------------------------------------------

    def record(self, date: Any, metrics: dict[str, float]) -> None:
        """Appende le metriche giornaliere alla history persistita.

        La chiave metrica di una feature è il suo ``name``; se assente nei
        record usa la prima chiave numerica disponibile (fallback). Voci senza
        valore numerico non vengono registrate.
        """
        appended = False
        for feature in self.config:
            value = _resolve_metric_value(feature.name, metrics)
            if value is None:
                continue
            self.state.history.setdefault(feature.name, []).append(
                {"date": str(date), "value": value}
            )
            appended = True
        if appended:
            self.state.save(self.state_path)

    def check_revert(self) -> list[RevertEvent]:
        """Revert automatico per le feature attive sotto floor per ``min_days`` run.

        Valuta le ultime ``min_days`` voci della history della feature (ogni
        voce = un run giornaliero; i gap di calendario come i weekend non
        interrompono la sequenza). Se tutte sono sotto ``floor`` → revert:
        feature disabilitata nello stato persistito (kill switch P4, effettivo
        dalla run successiva) e evento ritornato. History insufficiente o
        metrica mancante → nessun revert (aspetta dati).
        """
        events: list[RevertEvent] = []
        for feature in self._active_features():
            entries = [
                e
                for e in self.state.history.get(feature.name, [])
                if isinstance(e, dict) and _to_float(e.get("value")) is not None
            ]
            entries = sorted(
                entries, key=lambda e: str(e.get("date", ""))
            )[-feature.min_days:]
            if len(entries) < feature.min_days:
                continue  # history insufficiente → aspetta dati
            values = [float(e["value"]) for e in entries]
            if not all(v < feature.floor for v in values):
                continue  # metrica sopra floor → nessun revert
            last = entries[-1]
            last_value = values[-1]
            reason = (
                f"metrica sotto floor per {feature.min_days} run consecutivi "
                f"(last value {last_value:.4f} < {feature.floor})"
            )
            date = str(last.get("date", ""))
            self.state.disable(
                feature.name,
                reason=reason,
                last_value=last_value,
                floor=feature.floor,
                date=date,
            )
            events.append(
                RevertEvent(
                    name=feature.name,
                    reason=reason,
                    last_value=last_value,
                    floor=feature.floor,
                    date=date,
                )
            )
        if events:
            self.state.save(self.state_path)
        return events


# ---------------------------------------------------------------------------
# Convenience entry points
# ---------------------------------------------------------------------------

def apply_canary_features(state_path: str | Path | None = None) -> list[str]:
    """Helper pipeline: applica le feature canary attive a inizio job.

    Da chiamare nel primo asset del job (radice del grafo), prima che gli asset
    che leggono i flag eseguano. No-op sicuro con config assente o vuota.
    """
    return CanaryController(load_canary_config(), state_path=state_path).apply()


def run_canary_health(
    date: Any,
    metrics: dict[str, float],
    dispatcher: Any | None = None,
    config: list[CanaryFeature] | None = None,
    state_path: str | Path | None = None,
) -> list[RevertEvent]:
    """Convenienza: record + check_revert + dispatch health alert per ogni revert.

    Ogni revert viene instradato in ``dispatch_health_alerts`` con chiave
    ``canary_<name>`` (livello "alert" → CRITICAL: log + dashboard + email),
    riusando l'infrastruttura alerting di F-0.2. ``dispatcher`` è iniettabile
    per i test. No-op completo (nessuno stato scritto, nessun alert) se il
    config è assente o senza feature abilitate.
    """
    if config is None:
        config = load_canary_config()
    if not any(f.enabled for f in config):
        return []
    controller = CanaryController(config, state_path=state_path)
    controller.record(date, metrics)
    events = controller.check_revert()
    for event in events:
        dispatch_health_alerts(
            {f"canary_{event.name}": event.to_health_signal()},
            dispatcher=dispatcher,
            check_date=date,
        )
    return events


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _resolve_metric_value(name: str, metrics: Any) -> float | None:
    """Metrica per la feature: chiave = nome feature, fallback prima chiave numerica."""
    if not isinstance(metrics, dict):
        return None
    if name in metrics:
        return _to_float(metrics[name])
    for value in metrics.values():
        resolved = _to_float(value)
        if resolved is not None:
            return resolved
    return None


def _to_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _to_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
