"""Tests for scripts/populate_walkforward_caches.py."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def _load_populate():
    spec = importlib.util.spec_from_file_location(
        "populate_wf",
        ROOT / "scripts" / "populate_walkforward_caches.py",
    )
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def test_populate_writes_caches(tmp_path):
    mod = _load_populate()
    written = mod.populate(["lightgbm"], root=tmp_path)
    sig_path = Path(written["lightgbm"])
    ret_path = Path(written["returns"])
    assert sig_path.exists()
    assert ret_path.exists()
    signals = pd.read_parquet(sig_path)
    assert len(signals) >= 200
