"""Alert data structures and dispatcher for the MLCouncil monitoring system.

Severity levels
---------------
INFO     — informational, logged to file only
WARNING  — written to dashboard state (current_alerts.json) and logged
CRITICAL — email notification + dashboard state + logged to file

File outputs
------------
* data/alerts/{date}.json          — append-only daily alert log
* data/monitoring/current_alerts.json  — latest alerts for the Streamlit sidebar
"""

from __future__ import annotations

import json
import logging
import os
import smtplib
import socket
from dataclasses import asdict, dataclass
from datetime import date, datetime
from email.message import EmailMessage
from enum import Enum
from pathlib import Path
from typing import Any

from loguru import logger
from runtime_env import get_secret

_ROOT = Path(__file__).parents[1]
_ALERTS_DIR = _ROOT / "data" / "alerts"
_MONITORING_DIR = _ROOT / "data" / "monitoring"
_DEADLETTER_DIR = _ALERTS_DIR / "deadletter"


# ---------------------------------------------------------------------------
# Enums & dataclasses
# ---------------------------------------------------------------------------

class Severity(Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class AlertResult:
    """Single monitoring check result.

    Attributes
    ----------
    is_alert:
        True when the check triggered an alert (threshold breached).
    severity:
        Severity level: INFO / WARNING / CRITICAL.
    model_name:
        Identifier of the model being checked, or "council" for system-wide checks.
    check_type:
        One of: "alpha_decay", "feature_drift", "shap_stability", "regime_change".
    message:
        Human-readable summary of what was detected.
    recommendation:
        Actionable next step for the trading team.
    metric_value:
        The measured metric value that triggered (or didn't trigger) the alert.
    threshold:
        The configured threshold being compared against.
    timestamp:
        UTC timestamp of when the check was run (auto-set on creation when None).
    """

    is_alert: bool
    severity: Severity
    model_name: str
    check_type: str
    message: str
    recommendation: str
    metric_value: float
    threshold: float
    timestamp: str = ""

    def __post_init__(self) -> None:
        if not self.timestamp:
            self.timestamp = datetime.utcnow().isoformat() + "Z"

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["severity"] = self.severity.value
        return d


# ---------------------------------------------------------------------------
# AlertDispatcher
# ---------------------------------------------------------------------------

class AlertDispatcher:
    """Route alerts to file, dashboard state, and (optionally) email.

    Severity routing
    ----------------
    INFO     → _log_to_file
    WARNING  → _log_to_file + _write_to_dashboard_state
    CRITICAL → _log_to_file + _write_to_dashboard_state + _send_email
    """

    def __init__(self) -> None:
        _ALERTS_DIR.mkdir(parents=True, exist_ok=True)
        _MONITORING_DIR.mkdir(parents=True, exist_ok=True)
        _DEADLETTER_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def dispatch(self, alerts: list[AlertResult]) -> None:
        """Dispatch all triggered alerts through the appropriate channels."""
        for alert in alerts:
            if not alert.is_alert:
                continue

            self._log_to_file(alert)

            if alert.severity in (Severity.WARNING, Severity.CRITICAL):
                self._write_to_dashboard_state(alert)

            if alert.severity == Severity.CRITICAL:
                self._send_email(alert)

    # ------------------------------------------------------------------
    # Channels
    # ------------------------------------------------------------------

    def _log_to_file(self, alert: AlertResult) -> None:
        """Append alert JSON to data/alerts/{today}.json."""
        today = date.today().isoformat()
        log_path = _ALERTS_DIR / f"{today}.json"

        existing: list[dict] = []
        if log_path.exists():
            try:
                with open(log_path) as f:
                    existing = json.load(f)
                if not isinstance(existing, list):
                    existing = []
            except (json.JSONDecodeError, OSError):
                existing = []

        existing.append(alert.to_dict())
        try:
            with open(log_path, "w") as f:
                json.dump(existing, f, indent=2, default=str)
            logger.debug(
                f"Alert logged → {log_path.name}: "
                f"[{alert.severity.value.upper()}] {alert.check_type} / {alert.model_name}"
            )
        except OSError as exc:
            logger.warning(f"Could not write alert log: {exc}")

    def _write_to_dashboard_state(self, alert: AlertResult) -> None:
        """Merge alert into data/monitoring/current_alerts.json (deduplicated by key).

        The dashboard sidebar reads this file to display live alerts.
        Key = (model_name, check_type) — newer alert overwrites older.
        """
        state_path = _MONITORING_DIR / "current_alerts.json"

        current: list[dict] = []
        if state_path.exists():
            try:
                with open(state_path) as f:
                    current = json.load(f)
                if not isinstance(current, list):
                    current = []
            except (json.JSONDecodeError, OSError):
                current = []

        # Deduplicate: remove stale entry with same (model_name, check_type)
        key = (alert.model_name, alert.check_type)
        current = [
            a for a in current
            if (a.get("model_name"), a.get("check_type")) != key
        ]
        current.append(alert.to_dict())

        try:
            with open(state_path, "w") as f:
                json.dump(current, f, indent=2, default=str)
            logger.info(
                f"Dashboard state updated: [{alert.severity.value.upper()}] "
                f"{alert.check_type} — {alert.message}"
            )
        except OSError as exc:
            logger.warning(f"Could not update dashboard state: {exc}")

    def _send_email(self, alert: AlertResult) -> None:
        """Send email via smtplib (Gmail app password).

        Requires environment variables:
            ALERT_EMAIL     — sender and recipient address
            SMTP_PASSWORD   — Gmail app password (not account password)

        Silently skips when credentials are not configured.
        """
        email_addr = os.environ.get("ALERT_EMAIL", "")
        smtp_password = get_secret("SMTP_PASSWORD")

        if not email_addr or not smtp_password:
            logger.debug(
                "CRITICAL alert not emailed: ALERT_EMAIL / SMTP_PASSWORD not set."
            )
            return

        hostname = _safe_hostname()
        msg = EmailMessage()
        msg["Subject"] = (
            f"[MLCouncil CRITICAL] {alert.check_type.upper()} — {alert.model_name}"
        )
        msg["From"] = email_addr
        msg["To"] = email_addr
        msg.set_content(
            f"CRITICAL ALERT — MLCouncil monitoring\n"
            f"{'=' * 60}\n"
            f"Host:        {hostname}\n"
            f"Timestamp:   {alert.timestamp}\n"
            f"Model:       {alert.model_name}\n"
            f"Check:       {alert.check_type}\n"
            f"Metric:      {alert.metric_value:.6f}  (threshold: {alert.threshold:.6f})\n\n"
            f"Message:\n{alert.message}\n\n"
            f"Recommendation:\n{alert.recommendation}\n"
        )

        try:
            with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
                smtp.login(email_addr, smtp_password)
                smtp.send_message(msg)
            logger.info(f"CRITICAL alert email sent to {email_addr}")
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"Failed to send alert email: {exc}")
            self._save_to_deadletter(alert)

    # ------------------------------------------------------------------
    # Deadletter queue
    # ------------------------------------------------------------------

    @staticmethod
    def _save_to_deadletter(alert: AlertResult) -> None:
        """Persist a failed-to-email alert so it can be retried later."""
        _DEADLETTER_DIR.mkdir(parents=True, exist_ok=True)
        ts = datetime.utcnow().strftime("%Y%m%dT%H%M%S%f")
        dl_path = _DEADLETTER_DIR / f"{ts}_{alert.check_type}.json"
        try:
            dl_path.write_text(json.dumps(alert.to_dict(), indent=2, default=str))
            logger.warning(f"CRITICAL alert saved to deadletter: {dl_path.name}")
        except OSError as exc:
            logger.error(f"Could not write deadletter file: {exc}")

    def retry_deadletter(self) -> int:
        """Attempt to re-send all deadlettered CRITICAL alerts.

        Returns the number of alerts successfully sent and removed from
        the deadletter queue.
        """
        if not _DEADLETTER_DIR.exists():
            return 0

        sent = 0
        for dl_file in sorted(_DEADLETTER_DIR.glob("*.json")):
            try:
                data = json.loads(dl_file.read_text())
                alert = AlertResult(
                    is_alert=True,
                    severity=Severity.CRITICAL,
                    model_name=data.get("model_name", "unknown"),
                    check_type=data.get("check_type", "unknown"),
                    message=data.get("message", ""),
                    recommendation=data.get("recommendation", ""),
                    metric_value=float(data.get("metric_value", 0.0)),
                    threshold=float(data.get("threshold", 0.0)),
                    timestamp=data.get("timestamp", ""),
                )
                # Try sending — if it fails again, _send_email will
                # re-save to deadletter (but we remove this file first
                # to avoid duplicates).
                dl_file.unlink()
                self._send_email(alert)
                sent += 1
            except Exception as exc:  # noqa: BLE001
                logger.debug(f"Deadletter retry failed for {dl_file.name}: {exc}")

        if sent:
            logger.info(f"Deadletter retry: {sent} alert(s) re-sent successfully.")
        return sent


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _safe_hostname() -> str:
    try:
        return socket.gethostname()
    except Exception:  # noqa: BLE001
        return "unknown"


def load_current_alerts() -> list[dict]:
    """Read data/monitoring/current_alerts.json for the dashboard sidebar.

    Returns an empty list when the file does not exist or is malformed.
    """
    state_path = _MONITORING_DIR / "current_alerts.json"
    if not state_path.exists():
        return []
    try:
        with open(state_path) as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except (json.JSONDecodeError, OSError):
        return []
