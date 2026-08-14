from __future__ import annotations

import os
from pathlib import Path

from api.errors import ApiError
from council.canary import CanaryController, CanaryState, load_canary_config

CANARY_CONFIG_PATH = Path(os.getenv("MLCOUNCIL_CANARY_CONFIG", "config/canary.yaml"))
STATE_PATH = Path("data/results/canary_state.json")


def _controller() -> CanaryController:
    config = load_canary_config(str(CANARY_CONFIG_PATH)) if CANARY_CONFIG_PATH.exists() else []
    return CanaryController(config, state_path=str(STATE_PATH))


def get_flags() -> dict:
    if not CANARY_CONFIG_PATH.exists():
        raise ApiError(404, "artifact_not_found", "canary.yaml not found", str(CANARY_CONFIG_PATH))
    controller = _controller()
    state = controller.state
    out = []
    for f in controller.config:
        reverted = not state.is_enabled(f.name)
        pending = state.pending_apply.get(f.name)
        out.append({
            "name": f.name, "env": f.env, "value": f.value,
            "config_enabled": f.enabled, "reverted": reverted,
            "pending_enabled": pending["enabled"] if pending else None,
            "effective_enabled": (pending["enabled"] if pending else f.enabled) and not reverted,
            "floor": f.floor, "min_days": f.min_days,
        })
    return {"features": out}


def get_state() -> dict:
    state = CanaryState.load(str(STATE_PATH), config=_controller().config)
    return {
        "state_file": str(STATE_PATH),
        "exists": STATE_PATH.exists(),
        "reverted_features": {
            name: {"reverted_at": v.get("reverted_at"), "reason": v.get("revert_reason")}
            for name, v in state.features.items() if not v.get("enabled")
        },
        "pending_apply": state.pending_apply,
        "history": state.history,
    }


def apply_pending(name: str, enabled: bool) -> dict:
    controller = _controller()
    if name not in {f.name for f in controller.config}:
        raise ApiError(404, "unknown_flag", f"Unknown canary flag: {name}")
    controller.state.set_pending(name, enabled)
    controller.state.save(str(STATE_PATH))
    return preview()


def preview() -> dict:
    controller = _controller()
    changes = []
    for f in controller.config:
        pending = controller.state.pending_apply.get(f.name)
        if pending is not None:
            changes.append({
                "name": f.name, "from": f.enabled,
                "to": bool(pending["enabled"]), "at": pending.get("at"),
            })
    return {"pending_changes": changes, "flags": get_flags()["features"]}


def clear_pending(name: str) -> dict:
    controller = _controller()
    controller.state.clear_pending(name)
    controller.state.save(str(STATE_PATH))
    return preview()
