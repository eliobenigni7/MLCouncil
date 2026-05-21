"""Fill quality and cost calibration panel."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import streamlit as st

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from council.cost_calibration import DEFAULT_CALIBRATION_PATH, load_calibration
from council.transaction_costs import get_active_calibration_version, get_calibration_path
from dashboard.data_loader import load_fill_quality_summary

st.set_page_config(page_title="Fill Quality — ML Council", layout="wide")
st.title("Fill Quality & Cost Calibration")

version = get_active_calibration_version()
st.caption(f"Active calibration version: `{version or 'static lookup only'}`")

summary = load_fill_quality_summary()
if summary.empty:
    st.info("No fill records in data/operations/fills/ yet.")
else:
    st.dataframe(summary.round(3), use_container_width=True)

calib_path = get_calibration_path() or DEFAULT_CALIBRATION_PATH
if calib_path.exists():
    try:
        artifact = load_calibration(calib_path)
        st.subheader("Calibration artifact")
        st.json(
            {
                "version": artifact.version,
                "fill_sample_count": artifact.fill_sample_count,
                "kappa_by_tier": artifact.kappa_by_tier,
                "fill_count_by_tier": artifact.fill_count_by_tier,
            }
        )
    except Exception as exc:  # noqa: BLE001
        st.warning(f"Could not load calibration: {exc}")
else:
    st.info("No cost_calibration.json — nightly job has not produced an artifact yet.")
