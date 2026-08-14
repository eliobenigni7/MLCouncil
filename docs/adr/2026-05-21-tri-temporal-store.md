# ADR: Tri-Temporal Feature Store (arrival_time)

- Date: 2026-05-21
- Status: Accepted
- Track: T1.3 (Wave 1 — Foundations)
- Related: `docs/internal/disruptive-roadmap-2026-05-21.md`, `data/store/arctic_store.py`

## Context

The feature store already supports bi-temporal correctness via `valid_time`
(business date) and `transaction_time` (store write time). News and macro feeds
have variable latency: a headline may publish hours before the nightly ingest,
and FRED observations become available on a different clock than our Dagster run.

Walk-forward replay and challenger shadow mode need to distinguish:

1. **valid_time** — what period the observation describes
2. **transaction_time** — what the store contained at replay time (PIT)
3. **arrival_time** — when the feed made the datum knowable

Without `arrival_time`, backtests that filter only on `transaction_time` can
leak future news/macro availability into earlier decision points.

## Decision

Add a third temporal column `arrival_time` to `FeatureStore`:

- `write(ticker, df, arrival_time=None)` — optional batch timestamp; if omitted
  and the DataFrame has no `arrival_time` column, default to `transaction_time`.
- `read(..., as_of_arrival_time=None)` — filter `arrival_time <= cutoff`; rows
  without the column fall back to `transaction_time` for the filter.
- `read_universe(..., as_of_arrival_time=None)` — same filter per ticker.

Historical backfill via `scripts/migrate_arrival_time.py`:

| Source | Retro-estimate |
|--------|----------------|
| `data/raw/news/*.parquet` | `published` (max per ticker + valid_time) |
| `data/raw/macro/*.parquet` | `valid_time + 1 day @ 13:30 UTC` (FRED release proxy) |
| Otherwise | `transaction_time` |

## Consequences

- Positive: realistic replay for sentiment/macro-dependent models; no change to
  existing callers that omit `arrival_time` (defaults preserve prior behavior).
- Trade-off: migration heuristics are approximate; production ingest should set
  `arrival_time` explicitly as feeds are parsed.
- Operational: run migration once per environment after deploy; idempotent on
  symbols that already have `arrival_time`.

## Alternatives Considered

1. **Separate news/macro stores only** — rejected; council features are per-ticker
   and need unified PIT reads.
2. **Infer arrival only at read time** — rejected; persistence is required for
   reproducible walk-forward CI (T1.1).
3. **Replace transaction_time with arrival_time** — rejected; store versioning and
   existing PIT tests depend on `transaction_time`.

## Rollout Plan

1. Deploy `arctic_store.py` changes (backward compatible defaults).
2. `python scripts/migrate_arrival_time.py --dry-run` on staging.
3. `python scripts/migrate_arrival_time.py` (apply) during maintenance window.
4. Update ingest paths to pass explicit `arrival_time` on write (follow-up).

## Verification

```bash
python scripts/migrate_arrival_time.py --dry-run
python -m pytest tests/test_arctic_store.py -k tri_temporal -v
```

## Rollback

Reads without `as_of_arrival_time` behave as before. Unset `arrival_time` on write
by omitting the kwarg (defaults to `transaction_time`). No env flag required.
