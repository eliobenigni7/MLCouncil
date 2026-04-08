from __future__ import annotations

import importlib.util
import sys
import types


def _install_slowapi_stub() -> None:
    if importlib.util.find_spec("slowapi") is not None:
        return

    slowapi_mod = types.ModuleType("slowapi")
    util_mod = types.ModuleType("slowapi.util")

    class DummyLimiter:
        def __init__(self, key_func):
            self.key_func = key_func

    slowapi_mod.Limiter = DummyLimiter
    util_mod.get_remote_address = lambda request: "127.0.0.1"
    sys.modules["slowapi"] = slowapi_mod
    sys.modules["slowapi.util"] = util_mod


_install_slowapi_stub()
