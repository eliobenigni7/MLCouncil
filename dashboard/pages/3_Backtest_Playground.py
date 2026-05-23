"""Backtest Playground — launch ad-hoc backtests with custom council & portfolio params.

Runs the council aggregator + portfolio optimiser on a user-chosen window and
universe, with **no Alpaca dependency**. Backtests execute in a background
thread; the page polls until the future is done and then renders results.
Each run is persisted to ``data/results_playground/<ts>/`` for comparison.
"""

from __future__ import annotations

import sys
import time
import traceback
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import streamlit as st

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from backtest.playground import (
    PlaygroundParams,
    PlaygroundResult,
    list_snapshots,
    load_available_universe,
    load_snapshot_equity,
    load_snapshot_params,
    run_playground_backtest,
)
from dashboard.charts import (
    drawdown_chart,
    monthly_returns_heatmap,
    playground_overlay_chart,
    rolling_sharpe_chart,
)

st.set_page_config(
    page_title="Backtest Playground — ML Council",
    page_icon="🧪",
    layout="wide",
)
st.title("🧪 Backtest Playground")
st.caption(
    "Esegui backtest ad-hoc senza Alpaca: gioca con i pesi del council, "
    "i vincoli di portfolio e i costi, e confronta gli snapshot. "
    "I segnali per-modello sono **proxy** veloci derivati dai prezzi; "
    "per i modelli production (LGBM + FinBERT + HMM addestrati) usa "
    "`python scripts/run_strategy_backtest.py`."
)

# ---------------------------------------------------------------------------
# Background executor — module-level singleton in session state
# ---------------------------------------------------------------------------

if "playground_executor" not in st.session_state:
    st.session_state["playground_executor"] = ThreadPoolExecutor(max_workers=1)
if "playground_future" not in st.session_state:
    st.session_state["playground_future"] = None
if "playground_progress" not in st.session_state:
    # Mutable dict shared between worker thread and main thread.
    # Updating a dict key is atomic in CPython and avoids Streamlit's
    # warning about accessing session_state from non-script threads.
    st.session_state["playground_progress"] = {"p": 0.0, "msg": "Idle"}
if "playground_last_result" not in st.session_state:
    st.session_state["playground_last_result"] = None
if "playground_last_error" not in st.session_state:
    st.session_state["playground_last_error"] = None


def _make_progress_cb():
    bucket = st.session_state["playground_progress"]

    def _cb(p: float, msg: str) -> None:
        bucket["p"] = float(p)
        bucket["msg"] = str(msg)

    return _cb


# ---------------------------------------------------------------------------
# Sidebar — parameter form
# ---------------------------------------------------------------------------

def _normalize_weights(weights: dict[str, float]) -> dict[str, float]:
    total = sum(max(0.0, float(v)) for v in weights.values())
    if total <= 1e-9:
        return {k: 1.0 / max(len(weights), 1) for k in weights}
    return {k: max(0.0, float(v)) / total for k, v in weights.items()}


with st.sidebar:
    st.header("Parametri")

    universe_options = load_available_universe()
    default_universe = [t for t in ["AAPL", "MSFT", "GOOGL", "AMZN", "META"] if t in universe_options]
    if not default_universe:
        default_universe = universe_options[:5]

    with st.expander("Window & costi", expanded=True):
        end_default = date.today() - timedelta(days=1)
        start_default = end_default - timedelta(days=365)
        col_s, col_e = st.columns(2)
        with col_s:
            start_date = st.date_input("Start", value=start_default, key="pg_start")
        with col_e:
            end_date = st.date_input("End", value=end_default, key="pg_end")
        initial_capital = st.number_input(
            "Capitale iniziale (USD)",
            min_value=1_000.0,
            max_value=10_000_000.0,
            value=100_000.0,
            step=10_000.0,
            key="pg_capital",
        )
        slippage_bps = st.number_input(
            "Slippage (bps)", min_value=0.0, max_value=200.0, value=3.0, step=0.5, key="pg_slip"
        )
        commission_bps = st.number_input(
            "Commission (bps)", min_value=0.0, max_value=50.0, value=0.5, step=0.1, key="pg_comm"
        )

    with st.expander("Universo", expanded=True):
        universe = st.multiselect(
            "Tickers",
            options=universe_options,
            default=default_universe,
            key="pg_universe",
        )
        st.caption(f"{len(universe)} ticker selezionati su {len(universe_options)} disponibili.")

    with st.expander("Council — regime weights", expanded=False):
        st.caption("I pesi vengono normalizzati per regime in modo da sommare a 1.")
        regime_weights: dict[str, dict[str, float]] = {}
        for regime, defaults in (
            ("bull",       {"lgbm": 0.55, "sentiment": 0.25, "hmm": 0.20}),
            ("bear",       {"lgbm": 0.35, "sentiment": 0.15, "hmm": 0.50}),
            ("transition", {"lgbm": 0.45, "sentiment": 0.20, "hmm": 0.35}),
        ):
            st.markdown(f"**{regime.title()}**")
            cols = st.columns(3)
            row: dict[str, float] = {}
            for i, (model, dval) in enumerate(defaults.items()):
                with cols[i]:
                    row[model] = st.slider(
                        f"{model}",
                        min_value=0.0,
                        max_value=1.0,
                        value=float(dval),
                        step=0.05,
                        key=f"pg_w_{regime}_{model}",
                    )
            regime_weights[regime] = _normalize_weights(row)

    with st.expander("Council — bounds & orthogonality", expanded=False):
        col_a, col_b = st.columns(2)
        with col_a:
            weight_clip_min = st.number_input(
                "weight_clip.min", min_value=0.0, max_value=0.5, value=0.05, step=0.05, key="pg_wcmin"
            )
        with col_b:
            weight_clip_max = st.number_input(
                "weight_clip.max", min_value=0.10, max_value=1.0, value=0.60, step=0.05, key="pg_wcmax"
            )
        ic_rolling_window = st.number_input(
            "IC rolling window (days)", min_value=20, max_value=252, value=60, step=5, key="pg_icwin"
        )
        sharpe_rolling_window = st.number_input(
            "Sharpe rolling window (days)", min_value=60, max_value=504, value=120, step=10, key="pg_shwin"
        )
        use_orthogonality = st.checkbox("Enable orthogonality monitor", value=True, key="pg_ortho")
        max_correlation = st.slider(
            "max pairwise correlation", min_value=0.30, max_value=0.95,
            value=0.65, step=0.05, key="pg_maxcorr", disabled=not use_orthogonality,
        )

    with st.expander("Portfolio constraints", expanded=False):
        max_position = st.slider(
            "max_position", min_value=0.02, max_value=0.50, value=0.08, step=0.01, key="pg_maxpos"
        )
        max_turnover = st.slider(
            "max_turnover", min_value=0.05, max_value=1.00, value=0.20, step=0.05, key="pg_turn"
        )
        max_vol_ann = st.slider(
            "max_vol_ann", min_value=0.05, max_value=0.80, value=0.30, step=0.05, key="pg_vol"
        )
        sector_cap = st.slider(
            "sector_cap", min_value=0.10, max_value=1.00, value=0.45, step=0.05, key="pg_sect"
        )
        min_signal_strength = st.slider(
            "min_signal_strength", min_value=0.0, max_value=1.0, value=0.20, step=0.05, key="pg_mss"
        )

    note = st.text_input("Note (opzionale)", value="", key="pg_note", help="Salvata nello snapshot.")

    run_clicked = st.button(
        "▶ Run Backtest",
        type="primary",
        use_container_width=True,
        disabled=(st.session_state["playground_future"] is not None
                  and not st.session_state["playground_future"].done()),
    )


# ---------------------------------------------------------------------------
# Submit a run
# ---------------------------------------------------------------------------

def _build_params() -> PlaygroundParams:
    return PlaygroundParams(
        start_date=str(start_date),
        end_date=str(end_date),
        universe=list(universe),
        initial_capital=float(initial_capital),
        slippage_bps=float(slippage_bps),
        commission_bps=float(commission_bps),
        regime_weights=regime_weights,
        weight_clip_min=float(weight_clip_min),
        weight_clip_max=float(weight_clip_max),
        ic_rolling_window=int(ic_rolling_window),
        sharpe_rolling_window=int(sharpe_rolling_window),
        use_orthogonality=bool(use_orthogonality),
        max_correlation=float(max_correlation),
        max_position=float(max_position),
        max_turnover=float(max_turnover),
        max_vol_ann=float(max_vol_ann),
        sector_cap=float(sector_cap),
        min_signal_strength=float(min_signal_strength),
        note=note,
    )


if run_clicked:
    if not universe:
        st.error("Seleziona almeno un ticker.")
    elif start_date >= end_date:
        st.error("La start date deve essere precedente alla end date.")
    else:
        st.session_state["playground_last_error"] = None
        st.session_state["playground_progress"] = {"p": 0.0, "msg": "Submitting…"}
        params = _build_params()
        executor: ThreadPoolExecutor = st.session_state["playground_executor"]
        future: Future = executor.submit(run_playground_backtest, params, _make_progress_cb())
        st.session_state["playground_future"] = future
        st.rerun()


# ---------------------------------------------------------------------------
# Main layout — progress / results on the left, snapshots on the right
# ---------------------------------------------------------------------------

col_main, col_snap = st.columns([3, 2], gap="large")


def _render_progress(container) -> None:
    bucket = st.session_state["playground_progress"]
    p = float(bucket.get("p", 0.0))
    msg = str(bucket.get("msg", "…"))
    container.progress(min(max(p, 0.0), 1.0), text=msg)


def _render_stats(stats: dict) -> None:
    cols = st.columns(4)
    cols[0].metric("Sharpe (net)", f"{stats.get('sharpe', 0.0):.2f}")
    cols[1].metric("Max DD", f"{stats.get('max_drawdown', 0.0) * 100:.1f}%")
    cols[2].metric("CAGR", f"{stats.get('cagr', 0.0) * 100:.1f}%")
    cols[3].metric("Turnover (avg/d)", f"{stats.get('turnover', 0.0) * 100:.1f}%")

    cols2 = st.columns(4)
    cols2[0].metric("Calmar", f"{stats.get('calmar', 0.0):.2f}")
    cols2[1].metric("Final Equity", f"${stats.get('final_equity', 0.0):,.0f}")
    cols2[2].metric("Est. Costs", f"${stats.get('estimated_costs_usd', 0.0):,.0f}")
    cols2[3].metric("Rebalances", f"{stats.get('n_trades', 0)}")


def _render_result(result: PlaygroundResult, overlay_snapshots: dict[str, pd.Series]) -> None:
    st.subheader("Risultati")
    _render_stats(result.stats)

    curves: dict[str, pd.Series] = {}
    label_self = f"current ({result.params.start_date} → {result.params.end_date})"
    if not result.equity_curve.empty:
        curves[label_self] = result.equity_curve
    curves.update(overlay_snapshots)

    fig = playground_overlay_chart(curves, benchmark=result.benchmark_curve)
    st.plotly_chart(fig, use_container_width=True)

    tab_dd, tab_sh, tab_mh, tab_w, tab_c, tab_p = st.tabs(
        ["Drawdown", "Rolling Sharpe", "Monthly Heatmap", "Weights", "Contributions", "Params"]
    )

    with tab_dd:
        if not result.equity_curve.empty:
            st.plotly_chart(drawdown_chart(result.equity_curve), use_container_width=True)
        else:
            st.info("Nessuna equity curve disponibile.")

    with tab_sh:
        rets = result.equity_curve.pct_change().dropna()
        if not rets.empty:
            window = min(63, max(20, len(rets) // 4))
            st.plotly_chart(rolling_sharpe_chart(rets, window=window), use_container_width=True)
        else:
            st.info("Returns insufficienti.")

    with tab_mh:
        rets = result.equity_curve.pct_change().dropna()
        if not rets.empty:
            st.plotly_chart(monthly_returns_heatmap(rets), use_container_width=True)
        else:
            st.info("Returns insufficienti.")

    with tab_w:
        if result.weights.empty:
            st.info("Weights vuoti.")
        else:
            avg_w = result.weights.mean().sort_values(ascending=False)
            avg_w = avg_w[avg_w > 1e-4]
            st.bar_chart(avg_w)
            with st.expander("Tabella weights giornalieri"):
                st.dataframe(result.weights.round(4), use_container_width=True, height=300)

    with tab_c:
        if result.council_contributions.empty:
            st.info("Contributi council non disponibili (run troppo breve?).")
        else:
            contrib_cols = [c for c in result.council_contributions.columns if c.startswith("contrib_")]
            if contrib_cols:
                st.line_chart(result.council_contributions[contrib_cols])
            with st.expander("Tabella contributi"):
                st.dataframe(result.council_contributions.round(4),
                             use_container_width=True, height=260)

    with tab_p:
        st.json(result.params.to_dict())
        if result.snapshot_path:
            st.caption(f"Snapshot salvato in: `{result.snapshot_path}`")


with col_main:
    future: Future | None = st.session_state["playground_future"]

    if future is not None and not future.done():
        progress_holder = st.empty()
        status_holder = st.empty()
        _render_progress(progress_holder)
        status_holder.info("Backtest in esecuzione in background…")
        time.sleep(0.5)
        st.rerun()

    if future is not None and future.done():
        try:
            result: PlaygroundResult = future.result()
            st.session_state["playground_last_result"] = result
        except Exception as exc:  # noqa: BLE001
            tb = traceback.format_exc()
            st.session_state["playground_last_error"] = (str(exc), tb)
            st.session_state["playground_last_result"] = None
        st.session_state["playground_future"] = None
        st.rerun()

    err = st.session_state["playground_last_error"]
    if err is not None:
        msg, tb = err
        st.error(f"Backtest fallito: {msg}")
        with st.expander("Traceback"):
            st.code(tb, language="text")

    result = st.session_state["playground_last_result"]
    if result is not None:
        selected_paths: list[str] = st.session_state.get("playground_overlay_paths", [])
        overlay: dict[str, pd.Series] = {}
        for p in selected_paths:
            try:
                eq = load_snapshot_equity(Path(p))
                pm = load_snapshot_params(Path(p))
                label = f"{Path(p).name} ({pm.get('start_date', '')}→{pm.get('end_date', '')})"
                if not eq.empty:
                    overlay[label] = eq
            except Exception:
                continue
        _render_result(result, overlay)
    elif future is None:
        st.info(
            "Imposta i parametri nella sidebar e clicca **Run Backtest** per lanciare "
            "un esperimento. Tutto è eseguito localmente — nessuna chiave Alpaca richiesta."
        )

with col_snap:
    st.subheader("Snapshots")
    snaps = list_snapshots()
    if snaps.empty:
        st.info("Nessuno snapshot ancora. Lancia un backtest per crearne uno.")
    else:
        display_cols = ["timestamp", "start_date", "end_date", "n_tickers",
                        "sharpe", "max_drawdown", "cagr", "note"]
        st.dataframe(
            snaps[display_cols].round(3),
            use_container_width=True,
            height=260,
        )
        labels = [f"{r.timestamp}  ({r.start_date}→{r.end_date}, Sh={r.sharpe})"
                  for r in snaps.itertuples()]
        path_by_label = dict(zip(labels, snaps["path"].tolist()))
        selected = st.multiselect(
            "Overlay sul chart corrente",
            options=labels,
            default=[],
            key="pg_overlay_pick",
            help="Seleziona uno o più snapshot per sovrapporli all'equity curve.",
        )
        st.session_state["playground_overlay_paths"] = [path_by_label[s] for s in selected]

        st.caption("Gli snapshot sono persistenti in `data/results_playground/`.")
