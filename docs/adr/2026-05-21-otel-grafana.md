# ADR: OpenTelemetry tracing + Grafana observability (T1.4)

- Date: 2026-05-21
- Status: Accepted
- Track: T1.4 (Wave 1 — Foundations)
- Related: `docs/disruptive-roadmap-2026-05-21.md`, `observability/tracing.py`

## Context

MLCouncil's daily Dagster path spans four layers (ingest → features → signals → council).
Without distributed tracing, debugging latency regressions or partial failures across assets
is slow. Wave 2–4 challengers (TFT, FinMA, MoE, RL execution) need production-grade
observability before promotion.

Constraints:

- Tracing must be **off by default** (no overhead, no broken CI when OTel packages absent).
- Minimal dependency footprint; manual spans on representative assets per layer.
- Grafana on **:3001** to avoid clashing with Dagster UI (:3000).

## Decision

1. **`observability/tracing.py`** — `init_tracing()` configures OTLP HTTP export when
   `MLCOUNCIL_OTEL_ENABLED=true`. `trace_span()` is a no-op context manager otherwise.
2. **Dagster instrumentation** — lightweight spans on:
   - Layer 1 ingest: `raw_ohlcv`
   - Layer 2 features: `alpha158_features`
   - Layer 3 signals: `lgbm_signals`
   - Layer 4 council: `daily_orders`
3. **Sidecar stack** — `docker-compose.observability.yml`: OTel Collector → Tempo;
   Prometheus scrapes collector metrics; Grafana provisions Tempo + Prometheus + dashboard JSON.
4. **Dashboard-as-code** — `dashboards/grafana/mlcouncil.json` with TraceQL filters per layer.

## Span attributes

| Attribute | Example | Purpose |
|-----------|---------|---------|
| `mlcouncil.layer` | `ingest` | Layer SLO / filtering |
| `dagster.asset` | `raw_ohlcv` | Asset identity |
| `dagster.partition` | `2026-05-20` | Partition audit trail |
| `service.name` | `mlcouncil-dagster` | Service map |

## Consequences

- **Positive:** End-to-end trace visibility; audit-friendly partition tags; rollback via env flag.
- **Trade-off:** Not full auto-instrumentation (Dagster/FastAPI hooks deferred); metrics limited to collector `up` panels until span-metrics connector is added.
- **Operational:** Run observability compose separately from main `docker-compose.yml`.

## Alternatives considered

1. **Jaeger all-in-one** — Simpler but Grafana Cloud/on-prem standard is Tempo + Grafana.
2. **Always-on OTel** — Rejected; breaks test ergonomics and adds noise locally.
3. **Only dashboard math-trace** — Insufficient for cross-service latency (existing Streamlit trace stays complementary).

## Rollback

```bash
unset MLCOUNCIL_OTEL_ENABLED
# or
export MLCOUNCIL_OTEL_ENABLED=false
```

No code path changes required; spans become no-ops immediately.

## Rollout plan

1. `docker compose -f docker-compose.observability.yml up -d`
2. Enable OTel on pipeline host; run one partition materialization.
3. Open Grafana → MLCouncil Pipeline dashboard; confirm four layer trace panels.
4. Document env vars in README / AGENTS.md.

## Verification

```bash
python -m pytest tests/test_tracing.py -v
python -c "from observability.tracing import init_tracing; init_tracing()"
docker compose -f docker-compose.observability.yml up -d
# MLCOUNCIL_OTEL_ENABLED=true OTEL_EXPORTER_OTLP_ENDPOINT=http://127.0.0.1:4318/v1/traces
python scripts/run_pipeline.py --partition 2026-05-20
```

Expected: traces in Tempo with `resource.service.name=mlcouncil-dagster` and layer attributes.
