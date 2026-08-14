import { equityCurveChart } from "./charts";

it("builds an equity chart with two traces", () => {
  const fig = equityCurveChart(
    { dates: ["2024-01-01", "2024-01-02"], values: [100, 101] },
    { dates: ["2024-01-01", "2024-01-02"], values: [100, 99] },
  );
  expect(fig.data).toHaveLength(2);
  expect(fig.layout.title?.text).toContain("Equity");
});
