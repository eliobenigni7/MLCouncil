// Minimal ambient types for react-plotly.js (the package ships no declarations).
// The shim keeps the Plot component usable under strict mode without pulling in
// the full @types/plotly.js union types.
declare module "react-plotly.js" {
  import type * as React from "react";

  interface PlotlyProps {
    data: unknown;
    layout?: unknown;
    config?: unknown;
    style?: React.CSSProperties;
    className?: string;
    useResizeHandler?: boolean;
    onInitialized?: (...args: unknown[]) => void;
    onError?: (...args: unknown[]) => void;
  }

  const Plot: React.ComponentType<PlotlyProps>;
  export default Plot;
}
