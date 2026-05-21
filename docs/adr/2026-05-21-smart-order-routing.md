# ADR: Smart Order Routing (T4.2 Shadow)

- Date: 2026-05-21
- Status: Accepted (scaffolding)

## Decision

`execution/router.py` with Alpaca default; `ibkr_adapter` / `coinbase_adapter` stubs.
Enable with `MLCOUNCIL_SMART_ROUTING_ENABLED=true`.

## Rollback

Unset routing flag; orders flow through Alpaca adapter only.
