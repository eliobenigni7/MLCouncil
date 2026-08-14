import { Route, Routes } from "react-router-dom";
import { ProtectedRoute } from "./components/ProtectedRoute";
import { LoginPage } from "./pages/LoginPage";
import { OverviewPage } from "./pages/OverviewPage";
import { PerformancePage } from "./pages/PerformancePage";
import { AttributionPage } from "./pages/AttributionPage";
import { RegimePage } from "./pages/RegimePage";
import { BacktestPage } from "./pages/BacktestPage";
import { PromotionPage } from "./pages/PromotionPage";
import { CanaryPage } from "./pages/CanaryPage";
import { PipelinePage } from "./pages/PipelinePage";
import { TradingPage } from "./pages/TradingPage";
import { IntradayPage } from "./pages/IntradayPage";
import { PortfolioPage } from "./pages/PortfolioPage";
import { ConfigPage } from "./pages/ConfigPage";
import { MonitoringPage } from "./pages/MonitoringPage";
import { FillQualityPage } from "./pages/FillQualityPage";

export default function App() {
  return (
    <Routes>
      <Route path="/login" element={<LoginPage />} />
      <Route element={<ProtectedRoute />}>
        <Route index element={<OverviewPage />} />
        <Route path="/performance" element={<PerformancePage />} />
        <Route path="/attribution" element={<AttributionPage />} />
        <Route path="/regime" element={<RegimePage />} />
        <Route path="/experiments/backtest" element={<BacktestPage />} />
        <Route path="/experiments/promotion" element={<PromotionPage />} />
        <Route path="/experiments/canary" element={<CanaryPage />} />
        <Route path="/operations/pipeline" element={<PipelinePage />} />
        <Route path="/operations/trading" element={<TradingPage />} />
        <Route path="/operations/intraday" element={<IntradayPage />} />
        <Route path="/operations/portfolio" element={<PortfolioPage />} />
        <Route path="/system/config" element={<ConfigPage />} />
        <Route path="/system/monitoring" element={<MonitoringPage />} />
        <Route path="/system/fill-quality" element={<FillQualityPage />} />
      </Route>
    </Routes>
  );
}
