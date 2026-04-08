from __future__ import annotations

from importlib import import_module

__all__ = ["BaseModel", "TechnicalModel", "RegimeModel", "SentimentModel"]

_EXPORT_MAP = {
    "BaseModel": ("models.base", "BaseModel"),
    "TechnicalModel": ("models.technical", "TechnicalModel"),
    "RegimeModel": ("models.regime", "RegimeModel"),
    "SentimentModel": ("models.sentiment", "SentimentModel"),
}


def __getattr__(name: str):
    if name not in _EXPORT_MAP:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = _EXPORT_MAP[name]
    module = import_module(module_name)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value
