"""Overlay pending_apply: additivo a CanaryState, ignorato da check_revert."""
from pathlib import Path

import pytest

from council.canary import CanaryFeature, CanaryState, CanaryController


@pytest.fixture
def config() -> list[CanaryFeature]:
    return [
        CanaryFeature(name="online_learning", env="MLCOUNCIL_ONLINE_LEARNING", value="true", enabled=True, floor=0.0, min_days=5),
        CanaryFeature(name="moe_gating", env="MLCOUNCIL_AGGREGATOR_MODE", value="moe", enabled=False, floor=0.0, min_days=5),
    ]


def test_pending_apply_is_persisted(config, tmp_path):
    state_path = tmp_path / "canary_state.json"
    state = CanaryState()
    state.set_pending("moe_gating", True)
    state.save(str(state_path))
    loaded = CanaryState.load(str(state_path), config=config)
    assert loaded.pending_apply["moe_gating"]["enabled"] is True


def test_active_features_honor_pending(config, tmp_path):
    state = CanaryState()
    state.set_pending("moe_gating", True)
    controller = CanaryController(config, state_path=str(tmp_path / "canary_state.json"))
    controller.state = state
    active = [f.name for f in controller._active_features()]
    assert "moe_gating" in active


def test_revert_still_wins_over_pending(config, tmp_path):
    state = CanaryState()
    state.disable("online_learning", reason="test", last_value=0.01, floor=0.05, date="2026-08-14")
    state.set_pending("online_learning", True)
    controller = CanaryController(config, state_path=str(tmp_path / "canary_state.json"))
    controller.state = state
    assert "online_learning" not in [f.name for f in controller._active_features()]
