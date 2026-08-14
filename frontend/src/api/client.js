// Placeholder stub — replaced by the real API client in Task 15.
export class ApiError extends Error {
    constructor(status, code, message, detail) {
        super(message);
        this.status = status;
        this.code = code;
        this.message = message;
        this.detail = detail;
    }
}
export async function api(_path, _options = {}) {
    throw new ApiError(501, "not_implemented", "API client lands in Task 15", "");
}
