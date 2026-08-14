from __future__ import annotations

from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse


class ApiError(HTTPException):
    """Errore applicativo con envelope JSON: {error: {code, message, detail}}."""

    def __init__(self, status_code: int, code: str, message: str, detail: str = ""):
        super().__init__(status_code=status_code, detail=detail)
        self.code = code
        self.message = message
        self.error_detail = detail


def api_error_handler(_request: Request, exc: ApiError) -> JSONResponse:
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": {"code": exc.code, "message": exc.message, "detail": exc.error_detail}},
    )
