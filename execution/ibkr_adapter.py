"""Interactive Brokers adapter stub (T4.2)."""

from __future__ import annotations

from loguru import logger


class IBKRAdapter:
    """Paper/sandbox IBKR gateway stub."""

    def __init__(self, *, host: str = "127.0.0.1", port: int = 7497) -> None:
        self.host = host
        self.port = port
        self._connected = False

    def is_available(self) -> bool:
        return False

    def connect(self) -> bool:
        logger.warning("IBKRAdapter: stub only — install ib_insync and configure credentials")
        return False

    def submit_order(self, symbol: str, qty: int, **kwargs) -> dict:
        raise NotImplementedError("IBKR paper trading not configured")
