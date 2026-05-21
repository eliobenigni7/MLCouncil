"""Coinbase Advanced Trade adapter stub (T4.2)."""

from __future__ import annotations

from loguru import logger


class CoinbaseAdapter:
    """Crypto venue stub for smart routing tests."""

    def __init__(self) -> None:
        self._connected = False

    def is_available(self) -> bool:
        return False

    def connect(self) -> bool:
        logger.warning("CoinbaseAdapter: stub only — configure API keys for production")
        return False

    def submit_order(self, product_id: str, qty: float, **kwargs) -> dict:
        raise NotImplementedError("Coinbase execution not configured")
