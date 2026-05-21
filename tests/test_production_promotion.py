"""Tests for production manifest and promotion gate."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml


def test_manifest_defaults_linear_conformal(tmp_path, monkeypatch):
    manifest_path = tmp_path / "production_manifest.yaml"
    baseline = {
        "schema_version": 1,
        "models": {
            "technical": {"family": "lightgbm", "checkpoint": "models/checkpoints/lgbm_latest.pkl"},
            "sentiment": {"family": "finbert", "checkpoint": "models/checkpoints/sentiment_latest.pkl"},
            "regime": {"family": "hmm", "checkpoint": "models/checkpoints/hmm_latest.pkl"},
        },
        "council": {
            "aggregator_mode": "linear",
            "position_sizing": "conformal",
            "covariance_estimator": "ledoit",
            "portfolio_mode": "cvxpy",
            "use_stacked_council": False,
            "regime_mode": "label",
        },
        "experts": {"tft": {"enabled": False}, "microstructure": {"enabled": False}},
        "features": {"online_learning": False, "otel_enabled": False},
        "promotion_history": [],
    }
    manifest_path.write_text(yaml.safe_dump(baseline), encoding="utf-8")

    monkeypatch.setenv("MLCOUNCIL_ENV_PROFILE", "prod")
    monkeypatch.setenv("MLCOUNCIL_USE_PRODUCTION_MANIFEST", "true")

    from council import production_config

    production_config._MANIFEST_PATH = manifest_path  # type: ignore[attr-defined]

    assert production_config.get_aggregator_mode() == "linear"
    assert production_config.get_position_sizing_mode() == "conformal"
    assert production_config.get_covariance_estimator() == "ledoit"
    assert production_config.expert_enabled("tft") is False


def test_promote_model_requires_gate_report(tmp_path):
    from council.walkforward_promotion_gate import promote_model_to_production

    with pytest.raises(RuntimeError, match="No gate report"):
        promote_model_to_production("lightgbm", root=tmp_path)


def test_promote_model_with_force(tmp_path, monkeypatch):
    ops = tmp_path / "data" / "operations"
    ops.mkdir(parents=True)
    ckpt_dir = tmp_path / "models" / "checkpoints"
    ckpt_dir.mkdir(parents=True)

    challenger = ckpt_dir / "lgbm_challenger.pkl"
    challenger.write_bytes(b"challenger")
    champion = ckpt_dir / "lgbm_latest.pkl"

    gate = {
        "promotion_passed": True,
        "challenger_metrics": {"oos_sharpe": 0.6, "pbo": 0.4, "walk_forward_window_count": 10},
    }
    (ops / "walkforward_promotion_lightgbm.json").write_text(
        json.dumps(gate), encoding="utf-8"
    )
    (ops / "walkforward_streak_lightgbm.json").write_text(
        json.dumps({"consecutive_passes": 3, "auto_promote_eligible": True}),
        encoding="utf-8",
    )

    manifest_src = Path(__file__).resolve().parents[1] / "config" / "production_manifest.yaml"
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    (config_dir / "production_manifest.yaml").write_text(
        manifest_src.read_text(encoding="utf-8"), encoding="utf-8"
    )

    from council.walkforward_promotion_gate import promote_model_to_production

    result = promote_model_to_production("lightgbm", root=tmp_path, force=True)
    assert result["status"] == "ok"
    assert champion.exists()

    manifest = yaml.safe_load((config_dir / "production_manifest.yaml").read_text())
    assert len(manifest.get("promotion_history", [])) >= 1
