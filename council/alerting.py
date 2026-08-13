"""Unified health-signal aggregation for the immune system (F-0.2).

``collect_health_signals`` merges the outputs of the four drift / warning
families into a single structured dict:

- ``tda_warning``     — TDA beta1 proxy alert flag (council/tda_warning.py)
- ``causal_drift``    — causal graph change_fraction vs 0.25 (council/causal_drift.py)
- ``adwin_drift``     — ADWIN streaming drift flag (council/drift.py)
- ``ddm_drift``       — DDM streaming drift flag (council/drift.py)
- ``evidently_drift`` — dataset drift fraction (KS p < 0.05) vs 0.5
                        (council/evidently_reports.py)

Each signal is rendered as::

    {"level": "ok" | "warn" | "alert", "value": ..., "threshold": ..., "note": ...}

Missing / malformed inputs degrade to level ``"ok"`` with a ``note`` instead of
raising, so the API endpoint stays up before the weekly assets have run at all.

``dispatch_health_alerts`` closes the composition gap with the existing alert
layer (council/alerts.py): every ``warn``/``alert`` signal is converted into an
``AlertResult`` (WARNING / CRITICAL) and routed through ``AlertDispatcher``
(log file, dashboard state, email for CRITICAL). Dispatch is meant to run on a
weekly cadence from the Dagster asset — the GET health endpoint stays read-only.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from council.alerts import AlertDispatcher, AlertResult, Severity

HEALTH_LEVELS = ("ok", "warn", "alert")

# Soglie di default allineate ai detector: causal drift 0.25, evidently 0.5.
DEFAULT_CAUSAL_THRESHOLD = 0.25
DEFAULT_EVIDENTLY_THRESHOLD = 0.5
DEFAULT_TDA_THRESHOLD = 0.35

_DEFAULT_RESULTS_DIR = Path(__file__).resolve().parents[1] / "data" / "results"


def collect_health_signals(
    *,
    tda_alert: dict[str, Any] | None = None,
    causal_drift: dict[str, Any] | None = None,
    adwin_drift: dict[str, Any] | None = None,
    ddm_drift: dict[str, Any] | None = None,
    evidently_drift: dict[str, Any] | None = None,
    causal_threshold: float = DEFAULT_CAUSAL_THRESHOLD,
    evidently_threshold: float = DEFAULT_EVIDENTLY_THRESHOLD,
) -> dict[str, dict[str, Any]]:
    """Aggregate all immune-system signals into a single health dict.

    Parameters
    ----------
    tda_alert:
        Payload from ``data/results/tda_warning_latest.json``
        (keys: ``is_alert``, ``beta1_proxy``, ``threshold``).
    causal_drift:
        Payload from ``data/results/causal_drift_latest.json``
        (keys: ``change_fraction``, ``status``, ...).
    adwin_drift, ddm_drift:
        Payloads from the streaming detectors (key ``drift_detected``),
        or a raw bool flag.
    evidently_drift:
        Summary dict from :func:`council.evidently_reports.generate_drift_report`
        (key ``drift_fraction``).
    causal_threshold:
        Change-fraction threshold for the causal graph check. Default 0.25.
    evidently_threshold:
        Drift-fraction threshold for the dataset drift check. Default 0.5.

    Returns
    -------
    dict[str, dict[str, Any]]
        One entry per signal: ``{signal_name: {"level", "value",
        "threshold", "note"}}``. Inputs that are ``None`` / empty / malformed
        are rendered as level ``"ok"`` with a note (never raised).
    """
    return {
        "tda_warning": _flag_signal(
            _as_dict(tda_alert),
            flag_key="is_alert",
            name="tda_warning",
            threshold=_safe_float(_as_dict(tda_alert), "threshold", DEFAULT_TDA_THRESHOLD),
            detail_key="beta1_proxy",
        ),
        "causal_drift": _threshold_signal(
            _as_dict(causal_drift),
            value_key="change_fraction",
            name="causal_drift",
            threshold=causal_threshold,
        ),
        "adwin_drift": _flag_signal(
            _as_dict(adwin_drift),
            flag_key="drift_detected",
            name="adwin_drift",
            threshold=1.0,
        ),
        "ddm_drift": _flag_signal(
            _as_dict(ddm_drift),
            flag_key="drift_detected",
            name="ddm_drift",
            threshold=1.0,
        ),
        "evidently_drift": _threshold_signal(
            _as_dict(evidently_drift),
            value_key="drift_fraction",
            name="evidently_drift",
            threshold=evidently_threshold,
        ),
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _as_dict(payload: Any) -> dict[str, Any] | None:
    """Return a dict when the payload is a non-empty mapping, else None.

    Accepts raw bool flags (adwin/ddm) by wrapping them, so callers can pass
    either a detector payload dict or a bare ``True``/``False``.
    """
    if payload is None:
        return None
    if isinstance(payload, bool):
        return {"drift_detected": payload}
    if isinstance(payload, dict) and payload:
        return payload
    return None


def _safe_float(payload: dict[str, Any] | None, key: str, default: float) -> float:
    if payload is None:
        return default
    try:
        return float(payload.get(key, default))
    except (TypeError, ValueError):
        return default


def _flag_signal(
    payload: dict[str, Any] | None,
    *,
    flag_key: str,
    name: str,
    threshold: float,
    detail_key: str | None = None,
) -> dict[str, Any]:
    """Render a boolean flag signal: True → ``alert``, False → ``ok``."""
    if payload is None:
        return {
            "level": "ok",
            "value": None,
            "threshold": threshold,
            "note": f"{name}: input non disponibile",
        }
    flag = bool(payload.get(flag_key, False))
    note = None
    if detail_key is not None and payload.get(detail_key) is not None:
        try:
            note = f"{detail_key}={float(payload[detail_key]):.4f}"
        except (TypeError, ValueError):
            note = None
    return {
        "level": "alert" if flag else "ok",
        "value": flag,
        "threshold": threshold,
        "note": note,
    }


def _threshold_signal(
    payload: dict[str, Any] | None,
    *,
    value_key: str,
    name: str,
    threshold: float,
) -> dict[str, Any]:
    """Render a threshold signal: value >= threshold → ``alert``."""
    if payload is None:
        return {
            "level": "ok",
            "value": None,
            "threshold": threshold,
            "note": f"{name}: input non disponibile",
        }
    value = _safe_float(payload, value_key, 0.0)
    return {
        "level": "alert" if value >= threshold else "ok",
        "value": value,
        "threshold": threshold,
        "note": None,
    }


# ---------------------------------------------------------------------------
# Disk loading + dispatch
# ---------------------------------------------------------------------------

def collect_health_signals_from_disk(
    results_dir: str | Path | None = None,
) -> dict[str, dict[str, Any]]:
    """Read the weekly check payloads from disk and aggregate the health dict.

    Parameters
    ----------
    results_dir:
        Directory holding the ``*_latest.json`` payloads (``tda_warning_latest``,
        ``causal_drift_latest``, ``adwin_latest``, ``ddm_latest``,
        ``evidently_drift_latest``). Defaults to ``<repo>/data/results``.
        Missing or malformed files are treated as absent → level ``"ok"`` with
        a note (never raised).

    Returns
    -------
    The output of :func:`collect_health_signals` for the payloads found.
    """
    import json

    directory = Path(results_dir) if results_dir is not None else _DEFAULT_RESULTS_DIR

    def _load(name: str) -> dict[str, Any] | None:
        p = directory / f"{name}.json"
        try:
            if p.exists():
                data = json.loads(p.read_text(encoding="utf-8"))
                return data if isinstance(data, dict) else None
        except (json.JSONDecodeError, OSError):
            return None
        return None

    return collect_health_signals(
        tda_alert=_load("tda_warning_latest"),
        causal_drift=_load("causal_drift_latest"),
        adwin_drift=_load("adwin_latest"),
        ddm_drift=_load("ddm_latest"),
        evidently_drift=_load("evidently_drift_latest"),
    )


def dispatch_health_alerts(
    health: dict[str, dict[str, Any]] | None,
    dispatcher: Any | None = None,
    check_date: str | None = None,
) -> list[AlertResult]:
    """Dispatch every ``warn``/``alert`` signal through ``AlertDispatcher``.

    Level mapping → Severity:

    - ``"alert"`` → CRITICAL (log file + dashboard state + email)
    - ``"warn"``  → WARNING (log file + dashboard state)
    - ``"ok"``    → skipped

    Parameters
    ----------
    health:
        Output of :func:`collect_health_signals` (or
        :func:`collect_health_signals_from_disk`).
    dispatcher:
        Injectable ``AlertDispatcher`` (or duck-typed object with a
        ``dispatch`` method) for tests; defaults to a real
        :class:`~council.alerts.AlertDispatcher`.
    check_date:
        Optional ISO date used in the alert timestamp; when None the
        timestamp is auto-set to UTC now.

    Returns
    -------
    list[AlertResult]
        The created AlertResult entries (empty when nothing to dispatch).
        Non-dict entries and level ``"ok"`` are skipped gracefully.
    """
    results: list[AlertResult] = []
    if not isinstance(health, dict):
        return results

    for name, signal in health.items():
        if not isinstance(signal, dict):
            continue
        level = str(signal.get("level", "ok")).lower()
        if level not in ("warn", "alert"):
            continue
        severity = Severity.CRITICAL if level == "alert" else Severity.WARNING
        threshold = _safe_float(signal, "threshold", 0.0)
        value = signal.get("value")
        note = signal.get("note")
        message = (
            f"council/{name}: health level={level} value={value} "
            f"threshold={threshold}"
            + (f" ({note})" if note else "")
        )
        results.append(
            AlertResult(
                is_alert=True,
                severity=severity,
                model_name="council",
                check_type=name,
                message=message,
                recommendation=(
                    "Review the weekly monitoring outputs "
                    "(data/results/*_latest.json) and the health endpoint "
                    "/api/monitoring/health."
                ),
                metric_value=_safe_float(signal, "value", 0.0),
                threshold=threshold,
                timestamp=_format_timestamp(check_date),
            )
        )

    if results:
        (dispatcher or AlertDispatcher()).dispatch(results)
    return results


def _format_timestamp(check_date: Any) -> str:
    """Render check_date as an ISO UTC midnight timestamp ('' → auto now)."""
    if check_date is None:
        return ""
    if hasattr(check_date, "isoformat"):
        check_date = check_date.isoformat()
    return f"{check_date}T00:00:00Z"
