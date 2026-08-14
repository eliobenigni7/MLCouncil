// Placeholder stub — replaced by the real API client in Task 15.
export class ApiError extends Error {
  constructor(
    public status: number,
    public code: string,
    public message: string,
    public detail: string,
  ) {
    super(message);
  }
}

export async function api<T>(_path: string, _options: RequestInit = {}): Promise<T> {
  throw new ApiError(501, "not_implemented", "API client lands in Task 15", "");
}
