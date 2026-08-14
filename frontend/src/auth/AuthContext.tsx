import { createContext, useCallback, useContext, useEffect, useState } from "react";
import { authApi } from "../api/client";

type AuthStatus = "loading" | "authenticated" | "unauthenticated";

interface AuthState {
  status: AuthStatus;
  username: string | null;
  login: (username: string, password: string) => Promise<void>;
  logout: () => Promise<void>;
}

const AuthContext = createContext<AuthState | null>(null);

export function AuthProvider({ children }: { children: React.ReactNode }) {
  const [status, setStatus] = useState<AuthStatus>("loading");
  const [username, setUsername] = useState<string | null>(null);

  useEffect(() => {
    authApi
      .me()
      .then((me) => {
        setStatus("authenticated");
        setUsername(me.username);
      })
      .catch(() => setStatus("unauthenticated"));
  }, []);

  const login = useCallback(async (user: string, pass: string) => {
    const me = await authApi.login(user, pass);
    setUsername(me.username);
    setStatus("authenticated");
  }, []);

  const logout = useCallback(async () => {
    try {
      await authApi.logout();
    } finally {
      setStatus("unauthenticated");
      setUsername(null);
    }
  }, []);

  return <AuthContext.Provider value={{ status, username, login, logout }}>{children}</AuthContext.Provider>;
}

export function useAuth(): AuthState {
  const ctx = useContext(AuthContext);
  if (!ctx) throw new Error("useAuth outside AuthProvider");
  return ctx;
}
