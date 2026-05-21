"""Tests for scripts/promote_council_module.py."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def _load_promote():
    spec = importlib.util.spec_from_file_location(
        "promote_council",
        ROOT / "scripts" / "promote_council_module.py",
    )
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def test_promote_dcc_updates_manifest(tmp_path, monkeypatch):
    import council.production_config as pc

    manifest_src = ROOT / "config" / "production_manifest.yaml"
    manifest_dst = tmp_path / "config" / "production_manifest.yaml"
    manifest_dst.parent.mkdir(parents=True, exist_ok=True)
    manifest_dst.write_text(manifest_src.read_text(encoding="utf-8"), encoding="utf-8")

    monkeypatch.setattr(pc, "_MANIFEST_PATH", manifest_dst)
    mod = _load_promote()
    result = mod.promote_module("dcc", force=True)
    assert result["promotion_passed"]
    data = yaml.safe_load(manifest_dst.read_text(encoding="utf-8"))
    assert data["council"]["covariance_estimator"] == "dcc"
