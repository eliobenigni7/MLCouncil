from __future__ import annotations

import importlib.util
import sys
import types


def _install_slowapi_stub() -> None:
    if importlib.util.find_spec("slowapi") is not None:
        return

    slowapi_mod = types.ModuleType("slowapi")
    errors_mod = types.ModuleType("slowapi.errors")
    extension_mod = types.ModuleType("slowapi.extension")
    util_mod = types.ModuleType("slowapi.util")

    class DummyLimiter:
        def __init__(self, key_func):
            self.key_func = key_func

        def limit(self, _value):
            def decorator(func):
                return func

            return decorator

    class RateLimitExceeded(Exception):
        pass

    slowapi_mod.Limiter = DummyLimiter
    errors_mod.RateLimitExceeded = RateLimitExceeded
    extension_mod._rate_limit_exceeded_handler = lambda request, exc: None
    util_mod.get_remote_address = lambda request: "127.0.0.1"
    sys.modules["slowapi"] = slowapi_mod
    sys.modules["slowapi.errors"] = errors_mod
    sys.modules["slowapi.extension"] = extension_mod
    sys.modules["slowapi.util"] = util_mod


_install_slowapi_stub()
