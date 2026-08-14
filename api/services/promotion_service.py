from __future__ import annotations

import json
import os
from pathlib import Path

import yaml

from api.errors import ApiError

MANIFEST_PATH = Path(os.getenv("MLCOUNCIL_PRODUCTION_MANIFEST", "config/production_manifest.yaml"))
OPERATIONS_DIR = Path("data/operations")
SHADOW_ARTIFACTS = [
    Path("data/results/tft_shadow_signals.parquet"),
    Path("data/results/shadow_sentiment_llm"),
    Path("data/results/tda_warning_latest.json"),
]


def get_manifest() -> dict:
    if not MANIFEST_PATH.exists():
        raise ApiError(404, "artifact_not_found", "Production manifest not found", str(MANIFEST_PATH))
    return yaml.safe_load(MANIFEST_PATH.read_text(encoding="utf-8"))


def get_reports() -> dict:
    reports = {}
    for path in sorted(OPERATIONS_DIR.glob("walkforward_promotion_*.json")):
        try:
            reports[path.stem.replace("walkforward_promotion_", "")] = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
    streaks = {}
    for path in sorted(OPERATIONS_DIR.glob("walkforward_streak_*.json")):
        try:
            streaks[path.stem.replace("walkforward_streak_", "")] = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
    return {"reports": reports, "streaks": streaks}


def get_shadow_artifacts() -> dict:
    out = []
    for path in SHADOW_ARTIFACTS:
        try:
            mtime = path.stat().st_mtime if path.exists() else None
        except OSError:
            mtime = None
        out.append({"path": str(path), "exists": path.exists(), "mtime": mtime})
    return {"artifacts": out}
