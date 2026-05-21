"""Challenger promotion status — walk-forward gates and manifest (Wave 1–3)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

st.set_page_config(page_title="Challenger Promotion", page_icon="🏁", layout="wide")
st.title("Challenger promotion & shadow gates")

ops = _ROOT / "data" / "operations"
results = _ROOT / "data" / "results"


def _load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


manifest_path = _ROOT / "config" / "production_manifest.yaml"
if manifest_path.exists():
    import yaml

    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8")) or {}
    st.subheader("Production manifest")
    st.json(
        {
            "updated_at": manifest.get("updated_at"),
            "council": manifest.get("council"),
            "experts": manifest.get("experts"),
            "promotion_history": manifest.get("promotion_history", [])[-5:],
        }
    )

st.subheader("Walk-forward promotion reports")
models = ["lightgbm", "sentiment", "hmm", "tft"]
rows = []
for model in models:
    report = _load_json(ops / f"walkforward_promotion_{model}.json")
    streak = _load_json(ops / f"walkforward_streak_{model}.json")
    rows.append(
        {
            "model": model,
            "status": (report or {}).get("status", "missing"),
            "passed": (report or {}).get("promotion_passed"),
            "eligible": (streak or {}).get("auto_promote_eligible"),
            "consecutive_passes": (streak or {}).get("consecutive_passes", 0),
        }
    )
st.dataframe(pd.DataFrame(rows), use_container_width=True)

st.subheader("Shadow artifacts")
shadow_paths = [
    results / "tft_shadow_signals.parquet",
    results / "shadow_sentiment_llm",
    results / "tda_warning_latest.json",
]
for p in shadow_paths:
    st.write(f"`{p}`", "✅" if p.exists() else "— missing")

st.caption(
    "Populate caches: `python scripts/populate_walkforward_caches.py` · "
    "Staging TFT: `python scripts/establish_wave2_staging_promotion.py`"
)
