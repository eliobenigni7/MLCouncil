import { NavLink, Outlet, useNavigate } from "react-router-dom";
import { useAuth } from "../../auth/AuthContext";
import { AlertBanner } from "../AlertBanner";

const NAV_GROUPS: { group: string; links: { to: string; label: string }[] }[] = [
  { group: "", links: [{ to: "/", label: "Overview" }] },
  {
    group: "Analytics",
    links: [
      { to: "/performance", label: "Performance" },
      { to: "/attribution", label: "Attribution" },
      { to: "/regime", label: "Regime" },
    ],
  },
  {
    group: "Experiments",
    links: [
      { to: "/experiments/backtest", label: "Backtest" },
      { to: "/experiments/promotion", label: "Promotion" },
      { to: "/experiments/canary", label: "Canary" },
    ],
  },
  {
    group: "Operations",
    links: [
      { to: "/operations/pipeline", label: "Pipeline" },
      { to: "/operations/trading", label: "Trading" },
      { to: "/operations/intraday", label: "Intraday" },
      { to: "/operations/portfolio", label: "Portfolio" },
    ],
  },
  {
    group: "System",
    links: [
      { to: "/system/config", label: "Configuration" },
      { to: "/system/monitoring", label: "Monitoring" },
      { to: "/system/fill-quality", label: "Fill Quality" },
    ],
  },
];

const EXTERNAL_LINKS = [
  { href: "/mlflow/", label: "MLflow" },
  { href: "https://mlcouncil.duckdns.org:8443/", label: "Dagster" },
  { href: "http://localhost:3001", label: "Grafana" },
];

export function SidebarLayout() {
  const { username, logout } = useAuth();
  const navigate = useNavigate();
  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="sidebar-brand">MLCouncil</div>
        <nav>
          {NAV_GROUPS.map((g) => (
            <div key={g.group || "root"} className="nav-group">
              {g.group && <div className="nav-group-title">{g.group}</div>}
              {g.links.map((l) => (
                <NavLink key={l.to} to={l.to} end={l.to === "/"}
                  className={({ isActive }) => (isActive ? "nav-link active" : "nav-link")}>
                  {l.label}
                </NavLink>
              ))}
            </div>
          ))}
          <div className="nav-group-title">Links</div>
          {EXTERNAL_LINKS.map((l) => (
            <a key={l.href} className="nav-link" href={l.href} target="_blank" rel="noreferrer">
              {l.label} ↗
            </a>
          ))}
        </nav>
        <div className="sidebar-footer">
          <span>{username}</span>
          <button onClick={() => logout().then(() => navigate("/login"))} className="link-button">
            Sign out
          </button>
        </div>
      </aside>
      <main className="main-content">
        <AlertBanner />
        <Outlet />
      </main>
    </div>
  );
}
