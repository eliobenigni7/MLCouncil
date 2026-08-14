// Plotly chart builders, ported from dashboard/charts.py (12 builders).
// Pure functions returning { data, layout } figures configured for the dark theme.

export interface Series {
  dates: string[];
  values: number[];
}

export interface RegimeRow {
  date: string;
  regime: string;
}

export interface Trace {
  type?: string;
  mode?: string;
  name?: string;
  x?: (string | number | null)[];
  y?: (number | null)[];
  z?: (number | string)[][];
  text?: (string | number | null)[];
  domain?: { x: number[]; y: number[] };
  [key: string]: unknown;
}

export interface Layout {
  title?: { text?: string };
  shapes?: Record<string, unknown>[];
  [key: string]: unknown;
}

export interface Figure {
  data: Trace[];
  layout: Layout;
}

export const DARK_LAYOUT = {
  paper_bgcolor: "#17181c",
  plot_bgcolor: "#17181c",
  font: { color: "#c9ccd4", size: 12 },
  margin: { l: 60, r: 20, t: 50, b: 40 },
};

export const MODEL_COLORS: Record<string, string> = { lgbm: "#4e9bde", sentiment: "#e6a23c", hmm: "#9b6bd6" };
export const REGIME_COLORS: Record<string, string> = {
  bull: "rgba(44, 160, 44, 0.25)",
  bear: "rgba(214, 39, 40, 0.25)",
  transition: "rgba(255, 187, 0, 0.25)",
};

function colorWithAlpha(color: string, alpha = 0.25): string {
  if (color.startsWith("rgba(") || color.startsWith("rgb(")) return color;
  if (color.startsWith("#") && color.length === 7) {
    const red = parseInt(color.slice(1, 3), 16);
    const green = parseInt(color.slice(3, 5), 16);
    const blue = parseInt(color.slice(5, 7), 16);
    return `rgba(${red}, ${green}, ${blue}, ${alpha})`;
  }
  return color;
}

// charts.py:69 — equity_curve_chart
export function equityCurveChart(equity: Series, benchmark: Series) {
  return {
    data: [
      {
        x: equity.dates,
        y: equity.values,
        type: "scatter",
        mode: "lines",
        name: "Equity",
        line: { color: "#4e9bde", width: 2 },
      },
      {
        x: benchmark.dates,
        y: benchmark.values,
        type: "scatter",
        mode: "lines",
        name: "SPY",
        line: { color: "#8a8f9c", width: 1.5, dash: "dot" },
      },
    ],
    layout: { ...DARK_LAYOUT, title: { text: "Equity & benchmark (base 100)" } },
  };
}

// charts.py:161 — rolling_sharpe_chart
export function rollingSharpeChart(returns: Series, window = 252) {
  const y: (number | null)[] = returns.values.map((_, i) => {
    if (i < window) return null;
    const slice = returns.values.slice(i - window, i);
    const mean = slice.reduce((a, b) => a + b, 0) / slice.length;
    const variance = slice.reduce((a, b) => a + (b - mean) ** 2, 0) / (slice.length - 1);
    const std = Math.sqrt(variance);
    return std === 0 ? null : (mean / std) * Math.sqrt(252);
  });
  return {
    data: [{ x: returns.dates, y, type: "scatter", mode: "lines", name: "Rolling Sharpe", line: { color: "#4e9bde" } }],
    layout: {
      ...DARK_LAYOUT,
      title: { text: "Rolling Sharpe (252d)" },
      shapes: [{ type: "line", xref: "paper", x0: 0, x1: 1, y0: 0, y1: 0, line: { color: "#8a8f9c", dash: "dot" } }],
    },
  };
}

// charts.py:211 — monthly_returns_heatmap
export function monthlyReturnsHeatmap(returns: Series) {
  const map = new Map<string, number>();
  returns.dates.forEach((d, i) => {
    map.set(d.slice(0, 7), (map.get(d.slice(0, 7)) ?? 0) + (returns.values[i] ?? 0));
  });
  const months = [...map.keys()].sort();
  return {
    data: [{ z: [months.map((m) => map.get(m) ?? 0)], x: months, y: ["Return"], type: "heatmap", colorscale: "RdYlGn" }],
    layout: { ...DARK_LAYOUT, title: { text: "Monthly returns" }, yaxis: { showticklabels: false } },
  };
}

// charts.py:265 — drawdown_chart
export function drawdownChart(equity: Series) {
  const dd = equity.values.map((v, i) => {
    const peak = Math.max(...equity.values.slice(0, i + 1));
    return (v / peak - 1) * 100;
  });
  return {
    data: [
      {
        x: equity.dates,
        y: dd,
        type: "scatter",
        mode: "lines",
        fill: "tozeroy",
        name: "Drawdown",
        line: { color: "#d15b5b" },
        fillcolor: "rgba(209, 91, 91, 0.25)",
      },
    ],
    layout: { ...DARK_LAYOUT, title: { text: "Drawdown (%)" } },
  };
}

// charts.py:317 — model_contribution_bar
export interface AttributionRow {
  date: string;
  model_name: string;
  weight: number | null;
  ic_rolling_30d: number | null;
  sharpe_rolling_60d: number | null;
  pnl_contribution: number | null;
}

export function modelContributionBar(rows: AttributionRow[], targetDate?: string) {
  let filtered = rows;
  if (targetDate) {
    filtered = rows.filter((r) => r.date === targetDate);
  } else if (rows.length > 0) {
    const latest = rows.reduce((a, b) => (a.date > b.date ? a : b)).date;
    filtered = rows.filter((r) => r.date === latest);
  }
  filtered = filtered
    .filter((r) => r.pnl_contribution !== null)
    .sort((a, b) => (a.pnl_contribution ?? 0) - (b.pnl_contribution ?? 0));

  return {
    data: [
      {
        x: filtered.map((r) => r.pnl_contribution),
        y: filtered.map((r) => r.model_name.toUpperCase()),
        orientation: "h",
        type: "bar",
        marker_color: filtered.map((r) => ((r.pnl_contribution ?? 0) >= 0 ? "#3fa86d" : "#d15b5b")),
        text: filtered.map((r) => `${(r.pnl_contribution ?? 0).toFixed(4)}`),
        textposition: "outside",
        customdata: filtered.map((r) => [r.weight, r.ic_rolling_30d]),
        hovertemplate:
          "<b>%{y}</b><br>P&L Contribution: %{x:.4f}<br>Weight: %{customdata[0]:.1%}<br>IC (30d): %{customdata[1]:.4f}<extra></extra>",
      },
    ],
    layout: {
      ...DARK_LAYOUT,
      title: { text: `Council P&L Attribution — ${targetDate ?? "Latest"}` },
      xaxis: { title: { text: "P&L Contribution (IC × Weight)" } },
      yaxis: { title: { text: "Model" } },
      showlegend: false,
      shapes: [{ type: "line", xref: "paper", x0: 0, x1: 1, y0: 0, y1: 0, line: { color: "#8a8f9c", dash: "dash" } }],
    },
  };
}

// charts.py:389 — ic_rolling_chart
export interface IcRow {
  date: string;
  lgbm: number | null;
  sentiment: number | null;
  hmm: number | null;
}

export function icRollingChart(icHistory: IcRow[]) {
  const modelCols: (keyof IcRow)[] = ["lgbm", "sentiment", "hmm"];
  return {
    data: modelCols.map((model) => ({
      x: icHistory.map((r) => r.date),
      y: icHistory.map((r) => r[model]),
      type: "scatter",
      mode: "lines",
      name: model.toUpperCase(),
      line: { color: MODEL_COLORS[model] ?? "#888888", width: 2 },
    })),
    layout: {
      ...DARK_LAYOUT,
      title: { text: "IC Rolling 30d per Model" },
      xaxis: { title: { text: "Date" } },
      yaxis: { title: { text: "Information Coefficient" } },
      hovermode: "x unified",
      legend: { orientation: "h", y: 1.08 },
      shapes: [
        { type: "line", xref: "paper", x0: 0, x1: 1, y0: 0.01, y1: 0.01, line: { color: "#d15b5b", dash: "dash" } },
        { type: "line", xref: "paper", x0: 0, x1: 1, y0: 0, y1: 0, line: { color: "#8a8f9c", dash: "dot" } },
      ],
      annotations: [
        {
          xref: "paper", yref: "y", x: 1, y: 0.01,
          text: "IC alert (0.01)", showarrow: false, font: { color: "#d15b5b", size: 10 }, xanchor: "right",
        },
      ],
    },
  };
}

// charts.py:438 — weight_evolution_chart
export interface WeightRow {
  date: string;
  lgbm: number | null;
  sentiment: number | null;
  hmm: number | null;
}

export function weightEvolutionChart(weightsHistory: WeightRow[]) {
  const modelCols: (keyof WeightRow)[] = ["lgbm", "sentiment", "hmm"];
  return {
    data: modelCols.map((model, i) => {
      const color = MODEL_COLORS[model] ?? `hsl(${i * 120}, 70%, 50%)`;
      return {
        x: weightsHistory.map((r) => r.date),
        y: weightsHistory.map((r) => r[model]),
        type: "scatter",
        mode: "lines",
        name: model.toUpperCase(),
        stackgroup: "one",
        line: { width: 0.5, color },
        fillcolor: colorWithAlpha(color),
      };
    }),
    layout: {
      ...DARK_LAYOUT,
      title: { text: "Council Weight Evolution" },
      xaxis: { title: { text: "Date" } },
      yaxis: { tickformat: ".0%", range: [0, 1] },
      hovermode: "x unified",
      legend: { orientation: "h", y: 1.08 },
    },
  };
}

// charts.py:484 — regime_timeline (2-row grid: regime bands + equity overlay)
export function regimeTimeline(regimeHistory: RegimeRow[], equity?: Series) {
  const shapes: Record<string, unknown>[] = [];
  const annotations: Record<string, unknown>[] = [];
  const annotationColors: Record<string, string> = {
    bull: "rgb(44, 160, 44)",
    bear: "rgb(214, 39, 40)",
    transition: "rgb(255, 187, 0)",
  };

  let i = 0;
  while (i < regimeHistory.length) {
    const regime = regimeHistory[i].regime;
    const start = regimeHistory[i].date;
    let j = i + 1;
    while (j < regimeHistory.length && regimeHistory[j].regime === regime) j += 1;
    const end = j < regimeHistory.length ? regimeHistory[j].date : regimeHistory[regimeHistory.length - 1].date;
    shapes.push({
      type: "rect", xref: "x", yref: "paper", x0: start, x1: end, y0: 0, y1: 1,
      fillcolor: REGIME_COLORS[regime] ?? "rgba(128,128,128,0.2)", layer: "below", line: { width: 0 },
    });
    if (j - i > 10) {
      annotations.push({
        xref: "x", yref: "paper", x: start, y: 0.96, showarrow: false, align: "left",
        text: regime.charAt(0).toUpperCase() + regime.slice(1),
        font: { size: 9, color: annotationColors[regime] ?? "#888888" },
      });
    }
    i = j;
  }

  return {
    data: [
      {
        x: regimeHistory.map((r) => r.date),
        y: regimeHistory.map(() => 0),
        type: "scatter", mode: "markers", name: "Regime", showlegend: false,
        marker: { size: 0 }, hoverinfo: "skip",
      },
      ...(equity && equity.values.length > 0
        ? [{
            x: equity.dates, y: equity.values, type: "scatter", mode: "lines", name: "Equity",
            line: { color: "#4e9bde", width: 2 }, xaxis: "x2", yaxis: "y2",
          }]
        : []),
      ...[["Bull", "#3fa86d"], ["Bear", "#d15b5b"], ["Transition", "#d9a441"]].map(([label, color]) => ({
        x: [null], y: [null], type: "scatter", mode: "markers",
        marker: { size: 10, color, symbol: "square" }, name: label, showlegend: true,
      })),
    ],
    layout: {
      ...DARK_LAYOUT,
      title: { text: "Market Regime History" },
      hovermode: "x unified",
      legend: { orientation: "h", y: 1.08 },
      grid: { rows: 2, columns: 1, pattern: "independent" },
      xaxis: { domain: [0, 1], anchor: "y", showticklabels: false },
      yaxis: { domain: [0, 0.28], visible: false },
      xaxis2: { domain: [0, 1], anchor: "y2" },
      yaxis2: { domain: [0.32, 1], title: { text: "Equity (base 100)" } },
      shapes,
      annotations,
    },
  };
}

// charts.py:565 — regime_probability_gauge
export function regimeProbabilityGauge(probs: { bull: number; bear: number; transition: number }) {
  const gauges: { name: string; value: number; color: string; domain: number[] }[] = [
    { name: "Bull", value: probs.bull, color: "#3fa86d", domain: [0, 0.32] },
    { name: "Transition", value: probs.transition, color: "#d9a441", domain: [0.34, 0.66] },
    { name: "Bear", value: probs.bear, color: "#d15b5b", domain: [0.68, 1] },
  ];
  return {
    data: gauges.map((g) => ({
      type: "indicator",
      mode: "gauge+number",
      value: g.value * 100,
      number: { suffix: "%", font: { size: 22, color: g.color } },
      gauge: {
        axis: { range: [0, 100], tickcolor: "#c9ccd4" },
        bar: { color: g.color },
        bgcolor: "#262a33",
        bordercolor: "#3b414d",
        steps: [
          { range: [0, 33], color: "#141519" },
          { range: [33, 66], color: "#1a1d23" },
          { range: [66, 100], color: "#20242c" },
        ],
        threshold: { line: { color: g.color, width: 3 }, thickness: 0.75, value: g.value * 100 },
      },
      domain: { x: g.domain, y: [0.25, 1] },
      title: { text: g.name },
    })),
    layout: { ...DARK_LAYOUT, title: { text: "Current Regime Probabilities" }, height: 280 },
  };
}

// charts.py:620 — current_weights_radar
export function currentWeightsRadar(weights: Record<string, number>) {
  const categories = Object.keys(weights).map((k) => k.toUpperCase());
  const values = Object.values(weights);
  const maxV = Math.max(...values, 0);
  return {
    data: [
      {
        type: "scatterpolar",
        r: [...values, values[0]],
        theta: [...categories, categories[0]],
        fill: "toself",
        fillcolor: "rgba(78, 155, 222, 0.3)",
        line: { color: "#4e9bde", width: 2 },
        name: "Current Weights",
        mode: "lines+markers",
        marker: { size: 8, color: "#4e9bde" },
      },
    ],
    layout: {
      ...DARK_LAYOUT,
      title: { text: "Council Weights (Current Regime)" },
      polar: {
        bgcolor: "#262a33",
        radialaxis: { visible: true, range: [0, Math.max(maxV * 1.2, 0.1)], tickformat: ".0%", color: "#c9ccd4", gridcolor: "#3b414d" },
        angularaxis: { color: "#c9ccd4", gridcolor: "#3b414d" },
      },
      showlegend: false,
      height: 380,
    },
  };
}

// charts.py:671 — optimizer_waterfall
export function optimizerWaterfall(diagnostics: Record<string, unknown>, _topN = 8) {
  const greedy = (diagnostics.greedy_weights ?? {}) as Record<string, number>;
  const cvxpy = (diagnostics.cvxpy_weights ?? {}) as Record<string, number>;
  const final = (diagnostics.final_weights ?? {}) as Record<string, number>;

  const deployed = (weights: Record<string, number>) =>
    Object.values(weights).reduce((a, b) => a + Math.abs(b), 0);

  const greedyTotal = deployed(greedy);
  const cvxpyTotal = Object.keys(cvxpy).length > 0 ? deployed(cvxpy) : greedyTotal;
  const finalTotal = deployed(final);

  return {
    data: [
      {
        type: "waterfall",
        orientation: "v",
        measure: ["absolute", "relative", "relative", "total"],
        x: ["Greedy (α×m)", "CVXPY Δ", "Projection Δ", "Final"],
        y: [greedyTotal, cvxpyTotal - greedyTotal, finalTotal - cvxpyTotal, finalTotal],
        connector: { line: { color: "#3b414d" } },
        increasing: { marker: { color: "#3fa86d" } },
        decreasing: { marker: { color: "#d15b5b" } },
        totals: { marker: { color: "#4e9bde" } },
      },
    ],
    layout: {
      ...DARK_LAYOUT,
      title: { text: `Optimizer constraint waterfall (solver: ${String(diagnostics.solver_status ?? "unknown")})` },
      yaxis: { title: { text: "Deployed weight (L1)" } },
      height: 400,
    },
  };
}

// charts.py:736 — playground_overlay_chart
const OVERLAY_PALETTE = [
  "#3fa86d",
  "#636EFA",
  "#EF553B",
  "#AB63FA",
  "#FFA15A",
  "#19D3F3",
  "#FF6692",
  "#B6E880",
];

export function playgroundOverlayChart(
  curves: Record<string, Series>,
  benchmark?: Series,
  title = "Playground equity curves (normalized to 100)",
) {
  const data: Trace[] = [];
  if (benchmark && benchmark.values.length > 0) {
    const base = benchmark.values[0];
    if (base > 0) {
      data.push({
        x: benchmark.dates,
        y: benchmark.values.map((v) => (v / base) * 100),
        type: "scatter",
        mode: "lines",
        name: "SPY",
        line: { color: "#888888", width: 1.4, dash: "dot" },
      });
    }
  }
  Object.entries(curves).forEach(([label, series], i) => {
    if (!series || series.values.length === 0) return;
    const base = series.values[0];
    if (base <= 0) return;
    data.push({
      x: series.dates,
      y: series.values.map((v) => (v / base) * 100),
      type: "scatter",
      mode: "lines",
      name: label,
      line: { color: OVERLAY_PALETTE[i % OVERLAY_PALETTE.length], width: 2.2 },
    });
  });
  return {
    data,
    layout: {
      ...DARK_LAYOUT,
      title: { text: title },
      xaxis: { title: { text: "Date" } },
      yaxis: { title: { text: "Equity (base 100)" } },
      hovermode: "x unified",
      legend: { orientation: "h", y: 1.12 },
      height: 420,
    },
  };
}
