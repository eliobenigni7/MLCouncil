import { useEffect, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { api } from "../api/client";
import { ConfirmDialog } from "../components/ConfirmDialog";

interface UniverseConfig {
  universe: { tickers?: string[] };
  settings?: Record<string, unknown>;
  macro?: Record<string, unknown>;
}

interface RegimeWeightsConfig {
  regime_weights?: Record<string, Record<string, number>>;
  weight_clip?: { min?: number; max?: number };
  performance?: Record<string, number>;
  [key: string]: unknown;
}

export function ConfigPage() {
  const qc = useQueryClient();
  const [tickersText, setTickersText] = useState("");
  const [regimeWeightsText, setRegimeWeightsText] = useState("");
  const [confirmOpen, setConfirmOpen] = useState<"universe" | "weights" | null>(null);

  const universe = useQuery({
    queryKey: ["config-universe"],
    queryFn: () => api<UniverseConfig>("/api/config/universe"),
    retry: false,
  });

  const models = useQuery({
    queryKey: ["config-models"],
    queryFn: () => api<Record<string, unknown>>("/api/config/models"),
    retry: false,
  });

  const weights = useQuery({
    queryKey: ["config-regime-weights"],
    queryFn: () => api<RegimeWeightsConfig>("/api/config/regime-weights"),
    retry: false,
  });

  useEffect(() => {
    if (universe.data?.universe?.tickers) {
      setTickersText(universe.data.universe.tickers.join(", "));
    }
  }, [universe.data]);

  useEffect(() => {
    if (weights.data) {
      setRegimeWeightsText(JSON.stringify(weights.data, null, 2));
    }
  }, [weights.data]);

  const saveUniverse = useMutation({
    mutationFn: () =>
      api<unknown>("/api/config/universe", {
        method: "PUT",
        body: JSON.stringify({
          universe: { tickers: tickersText.split(",").map((t) => t.trim().toUpperCase()).filter(Boolean) },
          settings: universe.data?.settings ?? {},
          macro: universe.data?.macro ?? {},
        }),
      }),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["config-universe"] }),
  });

  const saveWeights = useMutation({
    mutationFn: () => {
      let payload: RegimeWeightsConfig;
      try {
        payload = JSON.parse(regimeWeightsText) as RegimeWeightsConfig;
      } catch {
        throw new Error("Regime weights is not valid JSON — fix the editor before saving.");
      }
      return api<unknown>("/api/config/regime-weights", {
        method: "PUT",
        body: JSON.stringify(payload),
      });
    },
    onSuccess: () => qc.invalidateQueries({ queryKey: ["config-regime-weights"] }),
  });

  return (
    <div className="page">
      <h1>Configuration</h1>
      <p className="caption">
        Universe, model registry and regime weight files. Changes are written to config/ and picked up by the next
        pipeline run.
      </p>

      <div className="panel">
        <h2 className="panel-title">Universe</h2>
        {universe.isLoading ? (
          <div className="muted">Loading…</div>
        ) : universe.error ? (
          <div className="muted">Universe file not available.</div>
        ) : (
          <div className="stack">
            <label>
              Tickers (comma-separated)
              <textarea
                rows={5}
                value={tickersText}
                onChange={(e) => setTickersText(e.target.value)}
                spellCheck={false}
              />
            </label>
            <div className="grid-2">
              <div>
                <div className="caption" style={{ marginBottom: 8 }}>Settings</div>
                <pre className="code">{JSON.stringify(universe.data?.settings ?? {}, null, 2)}</pre>
              </div>
              <div>
                <div className="caption" style={{ marginBottom: 8 }}>Macro</div>
                <pre className="code">{JSON.stringify(universe.data?.macro ?? {}, null, 2)}</pre>
              </div>
            </div>
            <div className="form-actions">
              <button className="btn btn-primary" onClick={() => setConfirmOpen("universe")} disabled={saveUniverse.isPending}>
                Save universe
              </button>
              {saveUniverse.error && <p className="form-error">{String(saveUniverse.error)}</p>}
            </div>
          </div>
        )}
      </div>

      <div>
        <h2>Model registry</h2>
        {models.isLoading ? (
          <div className="page-empty">Loading…</div>
        ) : models.error ? (
          <div className="page-empty">Model registry not available.</div>
        ) : (
          <div className="table-wrap">
            <table className="data-table">
              <thead>
                <tr>
                  <th>Key</th>
                  <th>Configuration</th>
                </tr>
              </thead>
              <tbody>
                {Object.entries(models.data ?? {}).map(([k, v]) => (
                  <tr key={k}>
                    <td className="mono">{k}</td>
                    <td>
                      <pre className="code" style={{ margin: 0 }}>{JSON.stringify(v, null, 2)}</pre>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>

      <div className="panel">
        <h2 className="panel-title">Regime weights</h2>
        {weights.isLoading ? (
          <div className="muted">Loading…</div>
        ) : weights.error ? (
          <div className="muted">Regime weights file not available.</div>
        ) : (
          <div className="stack">
            <label>
              Regime weights (JSON — regime → model → weight, plus weight_clip and performance)
              <textarea
                rows={16}
                value={regimeWeightsText}
                onChange={(e) => setRegimeWeightsText(e.target.value)}
                spellCheck={false}
                className="mono"
              />
            </label>
            <div className="form-actions">
              <button className="btn btn-primary" onClick={() => setConfirmOpen("weights")} disabled={saveWeights.isPending}>
                Save regime weights
              </button>
              {saveWeights.error && <p className="form-error">{String(saveWeights.error)}</p>}
            </div>
          </div>
        )}
      </div>

      <ConfirmDialog
        open={confirmOpen !== null}
        title={confirmOpen === "universe" ? "Save universe changes?" : "Save regime weights?"}
        body="Writes the edited configuration to disk. It applies to the next pipeline run, not the current one."
        confirmLabel="Confirm"
        onCancel={() => setConfirmOpen(null)}
        onConfirm={() => {
          if (confirmOpen === "universe") saveUniverse.mutate();
          if (confirmOpen === "weights") saveWeights.mutate();
          setConfirmOpen(null);
        }}
      />
    </div>
  );
}
