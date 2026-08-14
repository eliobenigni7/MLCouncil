"""Layer 5 — Monitoring: canary_health, tda_warning_signal, causal_drift_check,
model_promotion_gate (check settimanali/shadow + walk-forward promotion).

Parte del package data/pipeline (ex data/pipeline.py). Gli asset accedono alle
costanti/helper condivisi via ``_shared.NOME`` (vedi data/pipeline/_shared.py).
"""

import os
import numpy as np
import pandas as pd
import dagster as dg
from dagster import AssetExecutionContext
from datetime import date as date_type

from . import _shared
from ._shared import (
    _DAILY_PARTITIONS,
    _RETRY,
    _load_universe,
)


# ---------------------------------------------------------------------------
# Canary health (F-0.4) — metriche same-day del council + revert automatico
# ---------------------------------------------------------------------------

# Proxy di default tra feature canary e metriche same-day del council. Il
# mapping è un'euristica iniziale (nessuna metrica per-feature dedicata senza
# refactor invasivo): l'owner (gate G1) può raffinarlo ridenominando le feature
# in config/canary.yaml o estendendo questa mappa.
_CANARY_METRIC_PROXIES: dict[str, str] = {
    "online_learning": "council_signal_mean_abs",
    "moe_gating": "council_signal_mean_abs",
    "position_sizing_cqr": "portfolio_turnover",
    "dynamic_slippage": "portfolio_turnover",
}


def _build_canary_metrics(
    council_signal: pd.Series,
    portfolio_weights: pd.Series,
    feature_names: list[str] | None = None,
) -> dict[str, float]:
    """Metriche canary same-day (chiave = nome feature, proxy di default).

    Disponibili senza refactor invasivo:
    - ``council_signal_mean_abs`` — mean|z| del segnale combinato (qualità del segnale);
    - ``portfolio_turnover``      — 0.5 * Σ|target - current| vs snapshot live;
    - ``realized_vol_20d``        — std dei rendimenti medi giornalieri (20 sessioni).

    Le metriche non disponibili (es. snapshot live assente, OHLCV mancante)
    vengono semplicemente omesse: nessuna eccezione propagata.
    """
    base: dict[str, float] = {}
    if not council_signal.empty:
        base["council_signal_mean_abs"] = float(np.abs(council_signal).mean())
    if not portfolio_weights.empty:
        try:
            current_w, _ = _shared._load_live_portfolio_snapshot(
                portfolio_weights.index.tolist()
            )
            delta = (
                portfolio_weights
                - current_w.reindex(portfolio_weights.index).fillna(0.0)
            )
            base["portfolio_turnover"] = float(0.5 * delta.abs().sum())
        except Exception:  # noqa: BLE001
            pass  # snapshot live non disponibile → turnover non registrato
    try:
        returns_wide = _shared._load_returns_wide(
            sorted(council_signal.index.tolist()), tail_days=20
        )
        if returns_wide is not None and len(returns_wide) >= 2:
            base["realized_vol_20d"] = float(returns_wide.mean(axis=1).std())
    except Exception:  # noqa: BLE001
        pass  # OHLCV non disponibile → vol non registrata

    metrics = dict(base)
    for name in feature_names or []:
        proxy = _CANARY_METRIC_PROXIES.get(name, "council_signal_mean_abs")
        if proxy in base:
            metrics[name] = base[proxy]
    return metrics


@dg.asset(
    partitions_def=_DAILY_PARTITIONS,
    retry_policy=_RETRY,
    description=(
        "Canary health check giornaliero (F-0.4): registra le metriche "
        "same-day del council e applica il revert automatico delle feature "
        "canary attive (config/canary.yaml)."
    ),
)
def canary_health(
    context: AssetExecutionContext,
    council_signal: pd.Series,
    portfolio_weights: pd.Series,
) -> dict:
    """Registra le metriche del council e fa il check revert delle feature canary.

    No-op completo se config/canary.yaml è assente o senza feature abilitate
    (nessuno stato scritto, nessun alert, nessun side effect). Ordine logico:
    dopo council_signal/portfolio_weights, prima della materializzazione dei
    risultati operativi.
    """
    from council.canary import load_canary_config, run_canary_health

    partition_date = context.partition_key

    config = load_canary_config()
    active = [f.name for f in config if f.enabled]
    if not active:
        context.log.info(
            f"canary_health [{partition_date}]: nessuna feature canary "
            "abilitata — no-op"
        )
        return {"status": "noop"}

    metrics = _build_canary_metrics(
        council_signal, portfolio_weights, feature_names=active
    )
    if not metrics:
        context.log.warning(
            f"canary_health [{partition_date}]: nessuna metrica disponibile"
        )
        return {"status": "no_metrics"}

    events = run_canary_health(partition_date, metrics, config=config)
    for event in events:
        context.log.warning(
            f"canary_health [{partition_date}]: REVERT {event.name} — {event.reason}"
        )
    context.add_output_metadata(
        {
            "partition_date": partition_date,
            "metrics": {k: round(float(v), 6) for k, v in metrics.items()},
            "reverted": [event.name for event in events],
        }
    )
    context.log.info(
        f"canary_health [{partition_date}]: {len(active)} feature attive, "
        f"{len(events)} revert"
    )
    return {
        "status": "ok",
        "metrics": metrics,
        "reverts": [event.name for event in events],
    }


@dg.asset(
    retry_policy=_RETRY,
    description="Weekly TDA topology stress signal (T4.5 shadow).",
)
def tda_warning_signal(context: AssetExecutionContext) -> dict:
    """Compute rolling beta1 proxy on multivariate returns; log alert metadata."""
    from council.risk.tda_warning import PersistentHomologyAnalyser, tda_warning_enabled

    if not tda_warning_enabled():
        return {"status": "disabled"}

    tickers = _load_universe()[:12]
    returns_wide = _shared._load_returns_wide(tickers)
    if returns_wide is None:
        return {"status": "skipped_no_returns"}
    analyser = PersistentHomologyAnalyser()
    result = analyser.analyse(returns_wide)
    out_path = _shared._RESULTS_DIR / "tda_warning_latest.json"
    _shared._RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    import json

    payload = result.to_dict()
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    context.log.info(f"tda_warning_signal: {payload}")
    if result.is_alert:
        context.log.warning(
            f"tda_warning_signal: beta1_proxy={result.beta1_proxy:.3f} "
            f">= {result.threshold}"
        )
    return payload


@dg.asset(
    retry_policy=_RETRY,
    description="Weekly causal graph drift check (T4.4 shadow).",
)
def causal_drift_check(context: AssetExecutionContext) -> dict:
    """Rileva cambi strutturali feature→return vs la baseline persistita.

    La baseline del grafo causale viene salvata in ``causal_baseline.json``
    tra una run settimanale e l'altra; l'esito corrente va in
    ``data/results/causal_drift_latest.json``.
    """
    from council.risk.causal_drift import (
        PCMCIDriftDetector,
        causal_drift_enabled,
        load_causal_baseline,
        save_causal_baseline,
    )
    from council.monitoring.monitor import CouncilMonitor

    if not causal_drift_enabled():
        return {"status": "disabled"}

    tickers = _load_universe()[:12]
    returns_wide = _shared._load_returns_wide(tickers)
    if returns_wide is None:
        return {"status": "skipped_no_returns"}

    # Feature = rendimenti per ticker; target = rendimento medio del portafoglio.
    features = returns_wide
    forward_return = returns_wide.mean(axis=1)

    detector = PCMCIDriftDetector()
    baseline = load_causal_baseline(_shared._RESULTS_DIR / "causal_baseline.json")
    if baseline is not None:
        detector.set_baseline(baseline)

    result = CouncilMonitor().check_causal_graph_drift(
        features, forward_return, detector=detector
    )
    save_causal_baseline(_shared._RESULTS_DIR / "causal_baseline.json", detector.baseline)

    import json

    payload = result.to_dict()
    diag = detector.last_diagnostics or {}
    # "threshold" nel diag è la soglia di correlazione del grafo (0.15);
    # teniamo quella dell'alert (link_change_fraction) dal risultato.
    payload.update({k: v for k, v in diag.items() if k != "threshold"})

    out_path = _shared._RESULTS_DIR / "causal_drift_latest.json"
    _shared._RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    context.log.info(f"causal_drift_check: {payload}")
    if result.is_alert:
        context.log.warning(
            f"causal_drift_check: change_fraction="
            f"{diag.get('change_fraction', 0):.3f} >= {detector.link_change_fraction}"
        )

    # Dispatch unificato del sistema immunitario: instrada gli health alert
    # nel layer AlertDispatcher (log + dashboard state + email per CRITICAL).
    # Solo qui, con cadenza settimanale — l'endpoint GET /api/monitoring/health
    # resta read-only (nessun dispatch per-request).
    try:
        from council.monitoring.alerting import (
            collect_health_signals_from_disk,
            dispatch_health_alerts,
        )

        health = collect_health_signals_from_disk(_shared._RESULTS_DIR)
        dispatched = dispatch_health_alerts(
            health, check_date=date_type.today().isoformat()
        )
        if dispatched:
            context.log.info(
                f"causal_drift_check: {len(dispatched)} health alert(s) dispatched "
                f"({[d.check_type for d in dispatched]})"
            )
    except Exception as exc:  # noqa: BLE001
        # Il dispatch non deve mai far fallire il check settimanale.
        context.log.warning(f"causal_drift_check: health dispatch failed ({exc})")
    return payload


@dg.asset(
    retry_policy=_RETRY,
    description=(
        "Weekly alpha model promotion gate (T1.1). Evaluates shadow challengers vs "
        "champion walk-forward metrics. Production promotion requires "
        "scripts/promote_model.py after 3 consecutive passes."
    ),
)
def model_promotion_gate(context: AssetExecutionContext) -> dict:
    """Run walk-forward gate for production alpha models (shadow only)."""
    from council.walkforward_promotion_gate import SUPPORTED_MODELS, run_model_promotion_gate

    auto_promote = os.getenv("MLCOUNCIL_AUTO_PROMOTE_MODELS", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    reports: dict[str, dict] = {}
    for model in sorted(SUPPORTED_MODELS):
        report = run_model_promotion_gate(model, dry_run=False)
        reports[model] = report
        context.log.info(
            f"model_promotion_gate [{model}]: status={report.get('status')} "
            f"passed={report.get('promotion_passed')} "
            f"eligible={report.get('auto_promote_eligible')}"
        )
        if auto_promote and report.get("auto_promote_eligible"):
            try:
                from council.walkforward_promotion_gate import promote_model_to_production

                promote_model_to_production(model, force=False)
                context.log.info(f"model_promotion_gate: auto-promoted {model}")
            except Exception as exc:
                context.log.warning(f"model_promotion_gate: auto-promote {model} failed: {exc}")

    return {"models": reports, "auto_promote": auto_promote}


walkforward_promotion_job = dg.define_asset_job(
    name="walkforward_promotion_job",
    selection=dg.AssetSelection.assets(model_promotion_gate),
    description="Weekly walk-forward champion/challenger gate (alpha models).",
)


@dg.schedule(
    cron_schedule="0 2 * * 1",
    execution_timezone="UTC",
    job=walkforward_promotion_job,
)
def walkforward_promotion_schedule(context: "dg.ScheduleEvaluationContext"):
    """Monday 02:00 UTC — aligns with .github/workflows/walk-forward-ci.yml."""
    return dg.RunRequest(tags={"mlcouncil/job": "walkforward_promotion"})
