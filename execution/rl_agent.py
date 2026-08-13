"""PPO RL execution agent scaffold (T4.1 shadow).

Routes through TWAP when disabled or when stable-baselines3 is unavailable.
Canary status: shadow — target: P-2 — expiry: 2027-12-01 (promote via canary o retire)
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Optional

from loguru import logger

from execution.lob_simulator import OrderBookSimulator, SimulatedFill
from execution.slicer import ChildOrder, OrderSlicer

_TRUTHY = frozenset({"1", "true", "yes", "on"})


def rl_execution_enabled() -> bool:
    return os.getenv("MLCOUNCIL_RL_EXECUTION_ENABLED", "").strip().lower() in _TRUTHY


@dataclass
class ExecutionDecision:
    strategy: str
    child_orders: list[ChildOrder]
    expected_shortfall_bps: float


class PPOExecutionAgent:
    """Shadow RL execution agent — falls back to OrderSlicer TWAP/VWAP."""

    def __init__(
        self,
        *,
        adv_lookup: Optional[dict[str, float]] = None,
        simulator: Optional[OrderBookSimulator] = None,
    ) -> None:
        self._slicer = OrderSlicer(adv_lookup=adv_lookup or {})
        self._sim = simulator or OrderBookSimulator()
        self._model: Any = None
        if rl_execution_enabled():
            self._try_load_ppo()

    def _try_load_ppo(self) -> None:
        try:
            from stable_baselines3 import PPO  # noqa: F401

            logger.info("RL execution: stable-baselines3 available (training not bundled)")
        except ImportError:
            logger.debug("RL execution: stable-baselines3 not installed — TWAP fallback")

    def execute(
        self,
        symbol: str,
        qty: int,
        *,
        mid_price: float = 100.0,
        volume_profile: Optional[dict[str, float]] = None,
    ) -> ExecutionDecision:
        if not rl_execution_enabled() or self._model is None:
            if self._slicer.should_slice(symbol, abs(qty), mid_price):
                children = self._slicer.slice_vwap(symbol, qty, volume_profile)
                return ExecutionDecision(
                    strategy="vwap",
                    child_orders=children,
                    expected_shortfall_bps=8.0,
                )
            return ExecutionDecision(
                strategy="market",
                child_orders=[
                    ChildOrder(symbol=symbol, qty=qty, start_time="09:30", end_time="09:31")
                ],
                expected_shortfall_bps=5.0,
            )

        fill = self._sim.simulate_twap_fill(symbol, qty, mid_price)
        return ExecutionDecision(
            strategy="ppo_stub",
            child_orders=[
                ChildOrder(
                    symbol=symbol,
                    qty=qty,
                    start_time="09:30",
                    end_time="09:31",
                    limit_price=fill.fill_price,
                )
            ],
            expected_shortfall_bps=fill.slippage_bps,
        )
