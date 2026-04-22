from __future__ import annotations

MASKED_SECRET_VALUE = "********"

_SECRET_TOKENS = ("KEY", "SECRET", "TOKEN", "PASSWORD")


def is_secret_key(key: str) -> bool:
    upper_key = (key or "").upper()
    return any(token in upper_key for token in _SECRET_TOKENS)


def mask_secret_value(value: str | None) -> str:
    if not (value or "").strip():
        return ""
    return MASKED_SECRET_VALUE


def is_masked_secret_value(value: str | None) -> bool:
    return (value or "").strip() == MASKED_SECRET_VALUE
