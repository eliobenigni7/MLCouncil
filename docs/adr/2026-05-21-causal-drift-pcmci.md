# ADR: PCMCI Causal Graph Drift (T4.4 Shadow)

- Date: 2026-05-21
- Status: Accepted (scaffolding)
- Related: Wave 4 T4.4

## Decision

`council/causal_drift.py` provides `PCMCIDriftDetector` with correlation-proxy graph;
`CouncilMonitor.check_causal_graph_drift` integrates when `MLCOUNCIL_CAUSAL_DRIFT_ENABLED=true`.

Full `tigramite` PCMCI deferred; proxy sufficient for CI and monitor wiring.

## Rollback

Unset `MLCOUNCIL_CAUSAL_DRIFT_ENABLED`.
