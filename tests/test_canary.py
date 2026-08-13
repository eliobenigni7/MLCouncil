"""Tests per il canary controller (F-0.4): flag governance con revert automatico.

Copertura
---------
1. Config parsing: valida / vuota / corrotta (mai eccezioni)
2. apply(): setdefault rispetta l'env pre-esistente dell'operatore
3. record + check_revert: metrica sotto floor per min_days → revert + disabilitazione
4. Metrica sopra floor / history insufficiente → nessun revert
5. Revert → dispatch alert (dispatcher mock) via council/alerting.py
6. No-op con config vuota (zero side effect)
7. Asset pipeline canary_health: no-op senza feature abilitate
"""

from __future__ import annotations

import inspect
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

# Aggiungi la root del progetto al path
_ROOT = Path(__file__).parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from council.alerts import Severity
from council.canary import (
    CanaryController,
    CanaryFeature,
    CanaryState,
    RevertEvent,
    apply_canary_features,
    load_canary_config,
    run_canary_health,
)


# ---------------------------------------------------------------------------
# 1. Config parsing
# ---------------------------------------------------------------------------

class TestLoadConfig:
    def test_loads_valid_config(self, tmp_path):
        p = tmp_path / "canary.yaml"
        p.write_text(
            """
features:
  - name: online_learning
    env: MLCOUNCIL_ONLINE_LEARNING
    value: "true"
    enabled: true
    metrics:
      floor: 0.02
      min_days: 7
  - name: moe_gating
    env: MLCOUNCIL_AGGREGATOR_MODE
    value: "moe"
    enabled: false
""",
            encoding="utf-8",
        )

        feats = load_canary_config(p)

        assert len(feats) == 2
        assert feats[0].name == "online_learning"
        assert feats[0].env == "MLCOUNCIL_ONLINE_LEARNING"
        assert feats[0].value == "true"
        assert feats[0].enabled is True
        assert feats[0].floor == pytest.approx(0.02)
        assert feats[0].min_days == 7
        assert feats[1].name == "moe_gating"
        assert feats[1].enabled is False
        assert feats[1].floor == pytest.approx(0.0)  # default
        assert feats[1].min_days == 5  # default

    def test_missing_path_returns_empty(self, tmp_path):
        assert load_canary_config(tmp_path / "nope.yaml") == []

    def test_empty_file_returns_empty(self, tmp_path):
        p = tmp_path / "canary.yaml"
        p.write_text("", encoding="utf-8")
        assert load_canary_config(p) == []

    def test_corrupt_yaml_returns_empty(self, tmp_path):
        p = tmp_path / "canary.yaml"
        p.write_text("features: [unclosed", encoding="utf-8")
        assert load_canary_config(p) == []

    def test_malformed_entries_skipped(self, tmp_path):
        p = tmp_path / "canary.yaml"
        p.write_text(
            """
features:
  - name: ok_feature
    env: MLCOUNCIL_OK
    value: "true"
  - enabled: true
  - name: 123
    env: MLCOUNCIL_BAD
  - "string"
""",
            encoding="utf-8",
        )

        feats = load_canary_config(p)

        assert [f.name for f in feats] == ["ok_feature"]


# ---------------------------------------------------------------------------
# 2. apply(): attivazione come policy di run
# ---------------------------------------------------------------------------

class TestApply:
    @staticmethod
    def _feature(**overrides) -> CanaryFeature:
        base = dict(
            name="moe_gating",
            env="MLCOUNCIL_AGGREGATOR_MODE",
            value="moe",
            enabled=True,
            floor=0.0,
            min_days=2,
        )
        base.update(overrides)
        return CanaryFeature(**base)

    def test_apply_sets_env_for_enabled_features(self, tmp_path, monkeypatch):
        monkeypatch.delenv("MLCOUNCIL_AGGREGATOR_MODE", raising=False)

        controller = CanaryController(
            [self._feature()], state_path=tmp_path / "canary_state.json"
        )

        applied = controller.apply()

        assert applied == ["moe_gating"]
        assert __import__("os").environ["MLCOUNCIL_AGGREGATOR_MODE"] == "moe"

    def test_apply_respects_operator_env(self, tmp_path, monkeypatch):
        monkeypatch.setenv("MLCOUNCIL_AGGREGATOR_MODE", "linear")

        controller = CanaryController(
            [self._feature()], state_path=tmp_path / "canary_state.json"
        )

        applied = controller.apply()

        assert applied == ["moe_gating"]
        # L'env esplicito dell'operatore vince (setdefault)
        assert __import__("os").environ["MLCOUNCIL_AGGREGATOR_MODE"] == "linear"

    def test_apply_skips_disabled_features(self, tmp_path, monkeypatch):
        monkeypatch.delenv("MLCOUNCIL_AGGREGATOR_MODE", raising=False)

        controller = CanaryController(
            [self._feature(enabled=False)], state_path=tmp_path / "canary_state.json"
        )

        assert controller.apply() == []
        assert __import__("os").getenv("MLCOUNCIL_AGGREGATOR_MODE") is None

    def test_apply_skips_reverted_feature(self, tmp_path, monkeypatch):
        monkeypatch.delenv("MLCOUNCIL_AGGREGATOR_MODE", raising=False)
        state_path = tmp_path / "canary_state.json"

        controller = CanaryController([self._feature(min_days=1)], state_path=state_path)
        controller.record("2026-08-12", {"moe_gating": -0.1})
        assert controller.check_revert()  # 1 run sotto floor → revert

        assert controller.apply() == []  # revert sticky → non riapplicata
        assert __import__("os").getenv("MLCOUNCIL_AGGREGATOR_MODE") is None

        # Lo stato persistito conferma la disabilitazione alla run successiva
        reloaded = CanaryController([self._feature(min_days=1)], state_path=state_path)
        assert reloaded.apply() == []


# ---------------------------------------------------------------------------
# 3. record + check_revert
# ---------------------------------------------------------------------------

class TestRecordAndRevert:
    @staticmethod
    def _feature(**overrides) -> CanaryFeature:
        base = dict(
            name="moe_gating",
            env="MLCOUNCIL_AGGREGATOR_MODE",
            value="moe",
            enabled=True,
            floor=0.0,
            min_days=3,
        )
        base.update(overrides)
        return CanaryFeature(**base)

    def test_revert_after_min_days_below_floor(self, tmp_path):
        state_path = tmp_path / "canary_state.json"
        controller = CanaryController([self._feature()], state_path=state_path)

        for i in range(3):
            controller.record(f"2026-08-{10 + i:02d}", {"moe_gating": -0.05 - 0.01 * i})

        events = controller.check_revert()

        assert len(events) == 1
        event = events[0]
        assert isinstance(event, RevertEvent)
        assert event.name == "moe_gating"
        assert event.last_value == pytest.approx(-0.07)
        assert event.floor == pytest.approx(0.0)
        assert event.date == "2026-08-12"
        assert "sotto floor" in event.reason
        # Feature disabilitata in memoria e nello stato persistito
        assert not controller.state.is_enabled("moe_gating")
        saved = CanaryState.load(state_path, config=[self._feature()])
        assert not saved.is_enabled("moe_gating")
        assert saved.features["moe_gating"]["reverted_at"] == "2026-08-12"
        assert saved.features["moe_gating"]["revert_value"] == pytest.approx(-0.07)

    def test_revert_fires_only_once(self, tmp_path):
        state_path = tmp_path / "canary_state.json"
        controller = CanaryController([self._feature()], state_path=state_path)
        for i in range(3):
            controller.record(f"2026-08-{10 + i:02d}", {"moe_gating": -0.1})

        assert len(controller.check_revert()) == 1
        # Dopo il revert la feature non è più attiva → nessun secondo evento
        assert controller.check_revert() == []

    def test_no_revert_above_floor(self, tmp_path):
        controller = CanaryController(
            [self._feature()], state_path=tmp_path / "canary_state.json"
        )

        for i in range(3):
            controller.record(f"2026-08-{10 + i:02d}", {"moe_gating": 0.05 + i})

        assert controller.check_revert() == []
        assert controller.state.is_enabled("moe_gating")

    def test_no_revert_insufficient_history(self, tmp_path):
        controller = CanaryController(
            [self._feature(min_days=5)], state_path=tmp_path / "canary_state.json"
        )

        for i in range(3):
            controller.record(f"2026-08-{10 + i:02d}", {"moe_gating": -0.1})

        assert controller.check_revert() == []  # aspetta dati
        assert controller.state.is_enabled("moe_gating")

    def test_no_revert_without_metric_data(self, tmp_path):
        controller = CanaryController(
            [self._feature()], state_path=tmp_path / "canary_state.json"
        )

        controller.record("2026-08-10", {})  # nessuna chiave numerica

        assert controller.check_revert() == []
        assert controller.state.is_enabled("moe_gating")

    def test_fallback_to_first_numeric_metric_key(self, tmp_path):
        controller = CanaryController(
            [self._feature()], state_path=tmp_path / "canary_state.json"
        )

        for i in range(3):
            controller.record(
                f"2026-08-{10 + i:02d}",
                {"council_signal_mean_abs": -0.1 - 0.01 * i, "realized_vol_20d": 0.02},
            )

        events = controller.check_revert()

        # La feature non ha chiave propria nei record → fallback prima numerica
        assert len(events) == 1
        assert events[0].last_value == pytest.approx(-0.12)

    def test_revert_event_to_health_signal(self):
        event = RevertEvent(
            name="moe_gating",
            reason="metrica sotto floor per 3 run consecutivi",
            last_value=-0.07,
            floor=0.0,
            date="2026-08-12",
        )

        signal = event.to_health_signal()

        assert signal["level"] == "alert"
        assert signal["value"] == pytest.approx(-0.07)
        assert signal["threshold"] == pytest.approx(0.0)
        assert "sotto floor" in signal["note"]


# ---------------------------------------------------------------------------
# 4. run_canary_health: record + revert + dispatch alert
# ---------------------------------------------------------------------------

class TestRunCanaryHealth:
    @staticmethod
    def _feature(**overrides) -> CanaryFeature:
        base = dict(
            name="online_learning",
            env="MLCOUNCIL_ONLINE_LEARNING",
            value="true",
            enabled=True,
            floor=0.0,
            min_days=2,
        )
        base.update(overrides)
        return CanaryFeature(**base)

    def test_dispatches_alert_on_revert(self, tmp_path):
        dispatcher = MagicMock()
        config = [self._feature()]

        events = run_canary_health(
            "2026-08-12",
            {"online_learning": -0.1},
            dispatcher=dispatcher,
            config=config,
            state_path=tmp_path / "canary_state.json",
        )
        assert events == []
        dispatcher.dispatch.assert_not_called()

        events = run_canary_health(
            "2026-08-13",
            {"online_learning": -0.1},
            dispatcher=dispatcher,
            config=config,
            state_path=tmp_path / "canary_state.json",
        )

        assert len(events) == 1
        dispatcher.dispatch.assert_called_once()
        results = dispatcher.dispatch.call_args.args[0]
        assert len(results) == 1
        alert = results[0]
        assert alert.check_type == "canary_online_learning"
        assert alert.severity == Severity.CRITICAL  # livello "alert"
        assert alert.model_name == "council"
        assert alert.metric_value == pytest.approx(-0.1)
        assert alert.threshold == pytest.approx(0.0)
        assert alert.timestamp.startswith("2026-08-13")

    def test_no_dispatch_without_revert(self, tmp_path):
        dispatcher = MagicMock()
        config = [self._feature()]

        run_canary_health(
            "2026-08-12",
            {"online_learning": 0.2},
            dispatcher=dispatcher,
            config=config,
            state_path=tmp_path / "canary_state.json",
        )

        dispatcher.dispatch.assert_not_called()

    def test_noop_without_enabled_features(self, tmp_path):
        dispatcher = MagicMock()
        config = [self._feature(enabled=False)]
        state_path = tmp_path / "canary_state.json"

        events = run_canary_health(
            "2026-08-12",
            {"online_learning": -0.1},
            dispatcher=dispatcher,
            config=config,
            state_path=state_path,
        )

        assert events == []
        dispatcher.dispatch.assert_not_called()
        assert not state_path.exists()  # zero side effect

    def test_noop_with_missing_config(self, tmp_path, monkeypatch):
        dispatcher = MagicMock()
        monkeypatch.setattr(
            "council.canary.load_canary_config",
            lambda *a, **k: [],
        )

        events = run_canary_health(
            "2026-08-12",
            {"online_learning": -0.1},
            dispatcher=dispatcher,
        )

        assert events == []
        dispatcher.dispatch.assert_not_called()


# ---------------------------------------------------------------------------
# 5. Helper pipeline: apply_canary_features
# ---------------------------------------------------------------------------

class TestApplyHelper:
    def test_apply_canary_features_noop_without_config(self, monkeypatch, tmp_path):
        monkeypatch.setattr("council.canary.load_canary_config", lambda *a, **k: [])

        assert apply_canary_features(state_path=tmp_path / "state.json") == []

    def test_apply_canary_features_applies_enabled(self, tmp_path, monkeypatch):
        monkeypatch.delenv("MLCOUNCIL_AGGREGATOR_MODE", raising=False)
        feat = CanaryFeature(
            name="moe_gating",
            env="MLCOUNCIL_AGGREGATOR_MODE",
            value="moe",
            enabled=True,
        )
        monkeypatch.setattr("council.canary.load_canary_config", lambda *a, **k: [feat])

        assert apply_canary_features(state_path=tmp_path / "state.json") == ["moe_gating"]
        assert __import__("os").environ["MLCOUNCIL_AGGREGATOR_MODE"] == "moe"


# ---------------------------------------------------------------------------
# 6. Asset pipeline canary_health
# ---------------------------------------------------------------------------

import importlib.util  # noqa: E402


def _load_pipeline():
    """Carica data/pipeline.py come modulo standalone (pattern test_pipeline.py)."""
    mod_name = "pipeline_module_canary_test"
    if mod_name in sys.modules:
        return sys.modules[mod_name]
    spec = importlib.util.spec_from_file_location(mod_name, _ROOT / "data" / "pipeline.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod


_pipeline = _load_pipeline()


def _make_context(partition_date: str = "2024-01-15") -> MagicMock:
    ctx = MagicMock()
    ctx.partition_key = partition_date
    ctx.log = MagicMock()
    return ctx


def _call_asset(asset_def, *args):
    """Chiama la funzione decorata di un asset (pattern test_pipeline.py)."""
    fn = asset_def.op.compute_fn.decorated_fn
    signature = inspect.signature(fn)
    positional_params = [
        p
        for p in signature.parameters.values()
        if p.kind
        in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        )
    ]
    call_args = list(args[: len(positional_params)])
    for param in positional_params[len(call_args):]:
        call_args.append(pd.Series(dtype=float) if param.name != "context" else None)
    return fn(*call_args)


class TestCanaryHealthAsset:
    def test_noop_without_enabled_features(self):
        """Senza feature abilitate l'asset è un no-op (early-return, zero side effect).

        Ermetico: il loader della config viene patchato a config vuota, quindi il
        test non dipende dallo stato reale di config/canary.yaml (che può avere
        feature abilitate dopo l'approvazione del gate G1).
        """
        from council import canary as canary_mod

        ctx = _make_context("2024-01-15")

        with patch.object(canary_mod, "load_canary_config", return_value=[]), patch.object(
            canary_mod, "run_canary_health", return_value=[]
        ) as mock_run:
            result = _call_asset(
                _pipeline.canary_health,
                ctx,
                pd.Series(dtype=float),
                pd.Series(dtype=float),
            )

        assert result == {"status": "noop"}
        mock_run.assert_not_called()  # nessun recording/dispatch/state

    def test_records_metrics_for_enabled_feature(self, tmp_path, monkeypatch):
        """Con una feature abilitata l'asset registra metriche chiave = nome feature."""
        from council import canary as canary_mod

        ctx = _make_context("2024-01-15")
        council = pd.Series([0.3, -0.2, 0.5], index=["AAPL", "MSFT", "GOOGL"])
        weights = pd.Series([0.4, 0.3, 0.3], index=["AAPL", "MSFT", "GOOGL"])
        feat = canary_mod.CanaryFeature(
            name="moe_gating",
            env="MLCOUNCIL_AGGREGATOR_MODE",
            value="moe",
            enabled=True,
            floor=0.0,
            min_days=2,
        )

        with patch.object(canary_mod, "load_canary_config", return_value=[feat]), patch.object(
            canary_mod, "run_canary_health", return_value=[]
        ) as mock_run, patch.object(
            _pipeline,
            "_load_live_portfolio_snapshot",
            return_value=(pd.Series(0.0, index=["AAPL", "MSFT", "GOOGL"]), 100000.0),
        ), patch.object(
            _pipeline, "_load_returns_wide", return_value=None
        ):
            result = _call_asset(_pipeline.canary_health, ctx, council, weights)

        assert result["status"] == "ok"
        mock_run.assert_called_once()
        call = mock_run.call_args
        metrics = call.args[1] if len(call.args) > 1 else call.kwargs.get("metrics", {})
        assert "moe_gating" in metrics
        assert metrics["moe_gating"] == pytest.approx(
            float(np.abs(council).mean()), rel=1e-6
        )
        assert "council_signal_mean_abs" in metrics
        assert call.kwargs.get("config") == [feat]

    def test_no_metrics_with_empty_inputs(self, tmp_path, monkeypatch):
        """Con input vuoti e feature abilitata → status no_metrics, senza side effect.

        Il ramo no_metrics ritorna prima di run_canary_health: nessun recording,
        nessun alert, nessuno stato scritto.
        """
        from council import canary as canary_mod

        ctx = _make_context("2024-01-15")
        feat = canary_mod.CanaryFeature(
            name="moe_gating",
            env="MLCOUNCIL_AGGREGATOR_MODE",
            value="moe",
            enabled=True,
        )

        with patch.object(canary_mod, "load_canary_config", return_value=[feat]), patch.object(
            canary_mod, "run_canary_health", return_value=[]
        ) as mock_run:
            result = _call_asset(
                _pipeline.canary_health, ctx, pd.Series(dtype=float), pd.Series(dtype=float)
            )

        assert result["status"] == "no_metrics"
        mock_run.assert_not_called()  # nessun recording/dispatch/state
