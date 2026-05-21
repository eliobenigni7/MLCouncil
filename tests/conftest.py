from __future__ import annotations

import importlib.util
import os
import sys
import types

import pytest
from unittest.mock import patch


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


def _install_feedparser_stub() -> None:
    if importlib.util.find_spec("feedparser") is not None:
        return

    feedparser_mod = types.ModuleType("feedparser")

    def _parse(_url: str):
        mock = types.SimpleNamespace(entries=[], bozo=False)
        return mock

    feedparser_mod.parse = _parse
    sys.modules["feedparser"] = feedparser_mod


_install_feedparser_stub()


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "requires_api_key: test exercises real API key authentication behavior",
    )
    config.addinivalue_line(
        "markers",
        "asyncio: async test (pytest-asyncio)",
    )


@pytest.fixture(scope="session", autouse=True)
def _default_runtime_profile():
    old_profile = os.environ.get("MLCOUNCIL_ENV_PROFILE")
    old_require = os.environ.get("MLCOUNCIL_REQUIRE_API_KEY")
    old_key = os.environ.get("MLCOUNCIL_API_KEY")
    os.environ["MLCOUNCIL_ENV_PROFILE"] = "local"
    os.environ["MLCOUNCIL_REQUIRE_API_KEY"] = "false"
    os.environ.pop("MLCOUNCIL_API_KEY", None)
    yield
    if old_profile is None:
        os.environ.pop("MLCOUNCIL_ENV_PROFILE", None)
    else:
        os.environ["MLCOUNCIL_ENV_PROFILE"] = old_profile
    if old_require is None:
        os.environ.pop("MLCOUNCIL_REQUIRE_API_KEY", None)
    else:
        os.environ["MLCOUNCIL_REQUIRE_API_KEY"] = old_require
    if old_key is None:
        os.environ.pop("MLCOUNCIL_API_KEY", None)
    else:
        os.environ["MLCOUNCIL_API_KEY"] = old_key


@pytest.fixture(autouse=True)
def _disable_api_key(request):
    """Ensure API key auth is disabled for all tests.

    The app loads .env via runtime_env which sets MLCOUNCIL_API_KEY,
    overriding the session-scoped fixture. We patch get_configured_api_key
    to return empty string so all endpoints are accessible without auth.
    """
    if request.node.get_closest_marker("requires_api_key"):
        yield
        return

    with patch("api.auth.get_configured_api_key", return_value=""):
        yield
