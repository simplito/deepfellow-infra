# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Simplito sp. z o.o.

"""Global exception handlers producing OpenAI-style error responses."""

import uuid
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException

from server.utils.failed_span import get_failed_span_context
from server.utils.logger import logger
from server.utils.security_headers import add_security_headers
from server.utils.sentry import attached_tags, capture_exception, is_reported

ERROR_ID_HEADER = "X-Error-Id"

OPENAI_ERROR_TYPE_BY_STATUS: dict[int, str] = {
    401: "authentication_error",
    403: "permission_error",
    404: "not_found_error",
    429: "rate_limit_error",
}


def _error_type_for_status(status_code: int) -> str:
    if status_code >= 500:
        return "api_error"
    return OPENAI_ERROR_TYPE_BY_STATUS.get(status_code, "invalid_request_error")


def build_error_body(message: str, type_: str, param: str | None = None, code: str | None = None) -> dict[str, Any]:
    """Build an OpenAI-style `{"error": {...}}` response body."""
    return {"error": {"message": message, "type": type_, "param": param, "code": code}}


def _message_and_param_from_detail(detail: Any) -> tuple[str, str | None]:  # noqa: ANN401
    if isinstance(detail, str):
        return detail, None
    if isinstance(detail, list) and detail and all(isinstance(item, dict) and "message" in item for item in detail):
        message = "; ".join(f"{item.get('field')}: {item['message']}" if item.get("field") else item["message"] for item in detail)
        param = detail[0].get("field")
        return message, param
    return str(detail), None


async def handle_http_exception(_request: Request, exc: Exception) -> JSONResponse:
    """Translate any `HTTPException` into an OpenAI-style error response."""
    assert isinstance(exc, HTTPException)
    message, param = _message_and_param_from_detail(exc.detail)
    body = build_error_body(message, _error_type_for_status(exc.status_code), param)
    return JSONResponse(status_code=exc.status_code, content=body, headers=exc.headers)


async def handle_validation_error(_request: Request, exc: Exception) -> JSONResponse:
    """Translate FastAPI's `RequestValidationError` into an OpenAI-style error response."""
    assert isinstance(exc, RequestValidationError)
    errors = exc.errors()
    error = errors[0]
    param = ".".join(str(loc) for loc in error["loc"] if loc not in ("body", "query", "path", "header")) or None
    body = build_error_body(error["msg"], "invalid_request_error", param)
    return JSONResponse(status_code=400, content=body)


async def handle_unhandled_exception(request: Request, exc: Exception) -> JSONResponse:
    """Translate any exception no other handler caught into an OpenAI-style 500 carrying an error ID.

    The same ID goes into the message, the `X-Error-Id` header and the log line, along with the
    trace/span IDs of the traced endpoint that failed, if any. The log line has no traceback:
    Starlette re-raises the exception after this handler returns, and uvicorn logs it then. If the
    response had already started (e.g. mid-stream), Starlette drops this response but the log
    line is still written. When Sentry is on, the exception is reported to it as unhandled, tagged
    with the same IDs and its attached tags, unless already reported; that never raises.
    """
    error_id = str(uuid.uuid4())
    correlation = f"error_id={error_id}"
    span_context = get_failed_span_context(request)
    if span_context is not None and span_context.is_valid:
        correlation += f", trace_id={span_context.trace_id:032x}, span_id={span_context.span_id:016x}"
    logger.error("Unhandled %s on %s %r (%s)", type(exc).__name__, request.method, request.url.path, correlation)
    if not is_reported(exc):
        capture_exception(exc, error_id=error_id, handled=False, span_context=span_context, tags=attached_tags(exc))

    body = build_error_body(f"Internal server error. Error ID: {error_id}", "api_error")
    response = JSONResponse(status_code=500, content=body, headers={ERROR_ID_HEADER: error_id})
    # This handler runs in ServerErrorMiddleware, outside SecurityHeadersMiddleware.
    add_security_headers(response.headers, request.scope)
    return response


def register_exception_handlers(app: FastAPI) -> None:
    """Register global exception handlers producing OpenAI-style error responses."""
    app.add_exception_handler(HTTPException, handle_http_exception)
    app.add_exception_handler(RequestValidationError, handle_validation_error)
    app.add_exception_handler(Exception, handle_unhandled_exception)
