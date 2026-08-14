import { Navigate } from "react-router-dom";
import { useAuth } from "../auth/AuthContext";
import { SidebarLayout } from "./layout/Sidebar";

export function ProtectedRoute() {
  const { status } = useAuth();
  if (status === "loading") return <div className="boot-screen">Loading…</div>;
  if (status !== "authenticated") return <Navigate to="/login" replace />;
  return <SidebarLayout />;
}
