export function DataTable<T extends object>({
  rows,
  columns,
  renderCell,
  emptyMessage = "No data",
}: {
  rows: T[];
  columns: string[];
  renderCell?: (col: string, row: T) => React.ReactNode;
  emptyMessage?: string;
}) {
  if (rows.length === 0) {
    return (
      <div className="table-wrap">
        <div className="page-empty" style={{ padding: "28px 16px" }}>
          {emptyMessage}
        </div>
      </div>
    );
  }
  return (
    <div className="table-wrap">
      <table className="data-table">
        <thead>
          <tr>
            {columns.map((col) => (
              <th key={col}>{col.split("_").join(" ")}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row, i) => (
            <tr key={i}>
              {columns.map((col) => (
                <td key={col}>
                  {renderCell ? renderCell(col, row) : String((row as Record<string, unknown>)[col] ?? "")}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
