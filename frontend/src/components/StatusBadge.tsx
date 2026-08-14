export function StatusBadge({ state }: { state: string }) {
  const normalized = state.toLowerCase();
  return <span className={`status-badge status-${normalized}`}>{state}</span>;
}
