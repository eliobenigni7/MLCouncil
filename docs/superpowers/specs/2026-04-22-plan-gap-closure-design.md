# Plan Gap Closure Design

## Scope

Close the remaining implementation gaps from `docs/plan_20260421.md` with a minimal, root-repo-focused scope:

- reproducibility metadata completion
- deterministic dependency locking baseline
- missing retraining / registry module in `data/retraining.py`
- operator-facing validation/backtest summary and config/runtime inconsistency signals
- only the process/docs updates strictly needed to reflect the above

## Design

### Reproducibility

- Add a stable config fingerprint helper in `runtime_env.py`.
- Include the config fingerprint in artifact manifests from `council/artifacts.py`.
- Reuse the same fingerprint in retrospective validation environment metadata when practical.
- Introduce a pinned lock-style requirements file for CI/runtime reproducibility without replacing the current requirements layout.

### Retraining / Registry

- Add `data/retraining.py` in the root repo.
- Implement small, testable units:
  - `ValidationResult`
  - `ModelValidator`
  - `ModelRegistry`
  - `RetrainingPipeline`
- `RetrainingPipeline.train_candidate_model()` computes validation metrics plus walk-forward diagnostics using existing helpers in `backtest.validation`.
- `ModelRegistry.save_model()` persists the checkpoint and writes a `.manifest` sidecar.

### Operator UX

- Extend `/api/health` to return:
  - latest validation/backtest summary
  - config/runtime inconsistency summary
- Keep the UI changes small:
  - one overview card for validation/backtest summary
  - stronger operator banner text when inconsistency signals exist
  - one config summary panel showing fingerprint/drift warnings

### Process / Docs

- Update docs only where needed to document:
  - config fingerprint in manifests
  - pinned dependency file usage
  - admin validation summary visibility

## Testing Strategy

- Add/extend tests first for:
  - config fingerprint presence in manifests
  - retraining module behaviors already expected by `tests/test_retraining.py`
  - health endpoint summary payload
- Run focused pytest targets, then broader relevant suite.
