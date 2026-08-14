// Placeholder stub — replaced by the real auth context in Task 15.
export function AuthProvider({ children }: { children: React.ReactNode }) {
  return <>{children}</>;
}

export function useAuth(): never {
  throw new Error("AuthProvider stub replaced in Task 15");
}
