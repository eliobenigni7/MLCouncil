"""Tests for execution.router (T4.2)."""

from __future__ import annotations

from execution.router import SmartRouter, Venue


def test_router_prefers_lower_cost():
    class _VenueStub:
        def is_available(self) -> bool:
            return True

    router = SmartRouter(
        {Venue.ALPACA: _VenueStub(), Venue.IBKR: _VenueStub()}
    )
    decision = router.route("AAPL", 100, cost_estimates_bps={Venue.ALPACA: 6.0, Venue.IBKR: 4.0})
    assert decision.venue == Venue.IBKR
    assert decision.expected_cost_bps <= 10.0
