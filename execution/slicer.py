"""Order slicing for large trades.

Reduces market impact by splitting large orders into smaller child orders.
Supports TWAP and VWAP execution strategies.

Usage:
    from execution.slicer import OrderSlicer

    slicer = OrderSlicer(adv_lookup={...})
    if slicer.should_slice("AAPL", 5000, 150.0):
        child_orders = slicer.slice_vwap("AAPL", 5000, volume_profile)
        for order in child_orders:
            submit_order(**order)
"""

from __future__ import annotations

import pandas as pd
from dataclasses import dataclass
from datetime import datetime, time
from typing import Optional

import numpy as np


@dataclass
class ChildOrder:
    symbol: str
    qty: int
    start_time: str
    end_time: str
    order_type: str = "limit"
    limit_price: Optional[float] = None
    slice_id: int = 0


class OrderSlicer:
    """Slice large orders into smaller child orders.

    Attributes
    ----------
    adv_lookup : dict
        Dictionary mapping ticker -> 20-day average daily volume (in shares).
    slice_fraction : float
        Slice if order > slice_fraction * ADV (default 0.05 = 5%).
    market_open : time
        Market open time (default 9:30 AM ET).
    market_close : time
        Market close time (default 4:00 PM ET).
    """

    DEFAULT_VOLUME_PROFILE = {
        "09:30-10:30": 0.20,
        "10:30-11:30": 0.15,
        "11:30-12:30": 0.12,
        "12:30-13:30": 0.10,
        "13:30-14:30": 0.13,
        "14:30-15:30": 0.15,
        "15:30-16:00": 0.15,
    }

    def __init__(
        self,
        adv_lookup: Optional[dict[str, float]] = None,
        slice_fraction: float = 0.05,
        market_open: time = None,
        market_close: time = None,
    ):
        self.adv_lookup = adv_lookup or {}
        self.slice_fraction = slice_fraction
        self.market_open = market_open or time(9, 30)
        self.market_close = market_close or time(16, 0)

    def should_slice(
        self,
        symbol: str,
        qty: int,
        price: float,
        lookback_days: int = 20,
    ) -> bool:
        if symbol not in self.adv_lookup:
            estimated_adv = qty * 2
        else:
            estimated_adv = self.adv_lookup[symbol]

        notional = qty * price
        adv_notional = estimated_adv * price

        if adv_notional <= 0:
            return False

        fraction = notional / adv_notional
        return fraction > self.slice_fraction

    def slice_twap(
        self,
        symbol: str,
        qty: int,
        n_slices: int = 8,
        start_time: Optional[time] = None,
        end_time: Optional[time] = None,
    ) -> list[ChildOrder]:
        start_time = start_time or self.market_open
        end_time = end_time or self.market_close

        total_minutes = (
            (end_time.hour - start_time.hour) * 60
            + (end_time.minute - start_time.minute)
        )
        slice_minutes = total_minutes // n_slices

        slice_qty = qty // n_slices
        remaining = qty - slice_qty * n_slices

        orders = []
        current_hour = start_time.hour
        current_min = start_time.minute

        for i in range(n_slices):
            extra = 1 if i < remaining else 0
            order_qty = slice_qty + extra

            start = f"{current_hour:02d}:{current_min:02d}"
            current_min += slice_minutes
            if current_min >= 60:
                current_hour += 1
                current_min -= 60
            end = f"{current_hour:02d}:{current_min:02d}"

            orders.append(ChildOrder(
                symbol=symbol,
                qty=order_qty,
                start_time=start,
                end_time=end,
                order_type="limit",
                slice_id=i,
            ))

        return orders

    def slice_vwap(
        self,
        symbol: str,
        qty: int,
        volume_profile: Optional[dict[str, float]] = None,
        start_time: Optional[time] = None,
        end_time: Optional[time] = None,
    ) -> list[ChildOrder]:
        volume_profile = volume_profile or self.DEFAULT_VOLUME_PROFILE

        start_time = start_time or self.market_open
        end_time = end_time or self.market_close

        total_minutes = (
            (end_time.hour - start_time.hour) * 60
            + (end_time.minute - start_time.minute)
        )

        orders = []
        current_time = datetime.combine(datetime.today(), start_time)
        remaining_qty = qty
        slice_id = 0

        for bucket, fraction in volume_profile.items():
            start_str, end_str = bucket.split("-")
            bucket_start = datetime.strptime(start_str, "%H:%M").time()
            bucket_end = datetime.strptime(end_str, "%H:%M").time()

            if bucket_start < start_time:
                continue
            if bucket_end > end_time:
                continue

            bucket_minutes = (
                (bucket_end.hour - bucket_start.hour) * 60
                + (bucket_end.minute - bucket_start.minute)
            )

            if total_minutes <= 0:
                continue

            order_qty = int(round(qty * fraction))
            order_qty = min(order_qty, remaining_qty)

            if order_qty <= 0:
                continue

            remaining_qty -= order_qty

            orders.append(ChildOrder(
                symbol=symbol,
                qty=order_qty,
                start_time=start_str,
                end_time=end_str,
                order_type="limit",
                slice_id=slice_id,
            ))
            slice_id += 1

        if remaining_qty > 0 and orders:
            orders[-1].qty += remaining_qty

        return orders

    def slice_adaptive(
        self,
        symbol: str,
        qty: int,
        price: float,
        urgency: str = "normal",
    ) -> list[ChildOrder]:
        urgency_map = {
            "low": 4,
            "normal": 8,
            "high": 12,
            "aggressive": 16,
        }
        n_slices = urgency_map.get(urgency, 8)

        if urgency in ("low", "normal"):
            return self.slice_vwap(symbol, qty)
        else:
            return self.slice_twap(symbol, qty, n_slices=n_slices)


def estimate_adv(
    ticker: str,
    lookback: int = 20,
    price: float = None,
) -> float:
    try:
        import yfinance as yf
        data = yf.download(ticker, period=f"{lookback}d", auto_adjust=False, progress=False)
        if data is None or data.empty:
            return 0.0
        volumes = data["Volume"].tail(lookback)
        adv = volumes.mean()
        if price is not None:
            return adv * price
        return adv
    except Exception:
        return 0.0
