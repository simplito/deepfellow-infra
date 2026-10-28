# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Simplito sp. z o.o.

"""Tests for server/error_handlers.py."""

import logging
import uuid
from collections.abc import AsyncGenerator
from unittest import mock

import pytest
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from fastapi.testclient import TestClient
from opentelemetry.trace import INVALID_SPAN_CONTEXT, SpanContext
from pydantic import BaseModel

from server.error_handlers import (
    _error_type_for_status,  # pyright: ignore[reportPrivateUsage]
    _message_and_param_from_detail,  # pyright: ignore[reportPrivateUsage]
    build_error_body,
    handle_unhandled_exception,
    register_exception_handlers,
)
from server.utils.failed_span import remember_failed_span
from server.utils.security_headers import CACHE_CONTROL, CONTENT_SECURITY_POLICY
from server.utils.sentry import attach_tags, mark_reported

TRACE_ID = 0x0AF7651916CD43DD8448EB211C80319C
SPAN_ID = 0x00F067AA0BA902B7


class _Body(BaseModel):
    name: str


def test_build_error_body_shape() -> None:
    assert build_error_body("bad field", "invalid_request_error", "model", "some_code") == {
        "error": {"message": "bad field", "type": "invalid_request_error", "param": "model", "code": "some_code"}
    }


def test_error_type_for_status_maps_known_statuses() -> None:
    assert _error_type_for_status(400) == "invalid_request_error"
    assert _error_type_for_status(401) == "authentication_error"
    assert _error_type_for_status(403) == "permission_error"
    assert _error_type_for_status(404) == "not_found_error"
    assert _error_type_for_status(429) == "rate_limit_error"


def test_error_type_for_status_defaults_to_invalid_request_error() -> None:
    assert _error_type_for_status(418) == "invalid_request_error"


def test_error_type_for_status_maps_server_errors_to_api_error() -> None:
    assert _error_type_for_status(500) == "api_error"
    assert _error_type_for_status(503) == "api_error"


def test_message_and_param_from_string_detail() -> None:
    assert _message_and_param_from_detail("Model not found") == ("Model not found", None)


def test_message_and_param_from_pydantic_error_list_detail() -> None:
    detail = [{"field": "model", "message": "field required"}, {"field": "input", "message": "field required"}]

    message, param = _message_and_param_from_detail(detail)

    assert param == "model"
    assert message == "model: field required; input: field required"


def test_message_and_param_from_dict_detail_falls_back_to_str() -> None:
    detail = {"warnings": ["image is outdated"]}

    message, param = _message_and_param_from_detail(detail)

    assert param is None
    assert message == str(detail)


async def _boom() -> None:
    raise RuntimeError("secret internal detail")


async def _traced(request: Request) -> None:
    remember_failed_span(request, SpanContext(trace_id=TRACE_ID, span_id=SPAN_ID, is_remote=False))
    raise RuntimeError("fail")


async def _invalid_span(request: Request) -> None:
    remember_failed_span(request, INVALID_SPAN_CONTEXT)
    raise RuntimeError("fail")


async def _tagged() -> None:
    exc = RuntimeError("connection refused")
    attach_tags(exc, {"service": "ollama", "model": "llama3", "error_type": "connection_error"})
    raise exc


async def _reported() -> None:
    exc = RuntimeError("stream broke after 5xx")
    mark_reported(exc)
    raise exc


async def _missing() -> None:
    raise HTTPException(status_code=404, detail="Model not found")


async def _validated(body: _Body) -> None:
    pass


async def _broken_chunks() -> AsyncGenerator[bytes]:
    yield b"first"
    raise RuntimeError("stream broke")


async def _stream() -> StreamingResponse:
    return StreamingResponse(_broken_chunks(), media_type="text/plain")


@pytest.fixture
def app() -> FastAPI:
    app = FastAPI()
    register_exception_handlers(app)
    app.add_api_route("/boom", _boom, methods=["GET"])
    app.add_api_route("/traced", _traced, methods=["POST"])
    app.add_api_route("/invalid-span", _invalid_span, methods=["POST"])
    app.add_api_route("/tagged", _tagged, methods=["GET"])
    app.add_api_route("/reported", _reported, methods=["GET"])
    app.add_api_route("/missing", _missing, methods=["GET"])
    app.add_api_route("/validated", _validated, methods=["POST"])
    app.add_api_route("/stream", _stream, methods=["GET"])
    return app


@pytest.fixture
def client(app: FastAPI) -> TestClient:
    return TestClient(app, raise_server_exceptions=False)


def test_handle_unhandled_exception_returns_openai_500_with_error_id(client: TestClient):
    resp = client.get("/boom")

    assert resp.status_code == 500
    error_id = resp.headers["X-Error-Id"]
    assert str(uuid.UUID(error_id)) == error_id
    assert resp.json() == {
        "error": {"message": f"Internal server error. Error ID: {error_id}", "type": "api_error", "param": None, "code": None}
    }


def test_handle_unhandled_exception_hides_exception_details(client: TestClient):
    resp = client.get("/boom")

    assert "secret internal detail" not in resp.text
    assert "RuntimeError" not in resp.text
    assert all("secret internal detail" not in value for value in resp.headers.values())


def test_handle_unhandled_exception_adds_security_headers(client: TestClient):
    resp = client.get("/boom")

    assert resp.headers["Content-Security-Policy"] == CONTENT_SECURITY_POLICY
    assert resp.headers["X-Frame-Options"] == "DENY"
    assert resp.headers["X-Content-Type-Options"] == "nosniff"
    assert resp.headers["Cache-Control"] == CACHE_CONTROL
    assert resp.headers["Pragma"] == "no-cache"


def test_handle_unhandled_exception_generates_new_error_id_per_failure(client: TestClient):
    first = client.get("/boom")
    second = client.get("/boom")

    assert first.headers["X-Error-Id"] != second.headers["X-Error-Id"]


@mock.patch("server.error_handlers.logger")
def test_handle_unhandled_exception_logs_error_id_with_request_context(mock_logger: mock.MagicMock, client: TestClient):
    resp = client.get("/boom?api_key=secret")

    error_id = resp.headers["X-Error-Id"]
    assert mock_logger.error.call_count == 1
    assert mock_logger.error.call_args == mock.call("Unhandled %s on %s %r (%s)", "RuntimeError", "GET", "/boom", f"error_id={error_id}")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("path", "logged_path"),
    [
        ("/items/a\x1b[31mfake", "'/items/a\\x1b[31mfake'"),
        ("/items/a\u2028fake", "'/items/a\\u2028fake'"),
    ],
)
async def test_handle_unhandled_exception_escapes_control_characters_in_logged_path(
    caplog: pytest.LogCaptureFixture, path: str, logged_path: str
):
    caplog.set_level(logging.ERROR, logger="uvicorn")
    request = Request({"type": "http", "method": "GET", "path": path, "query_string": b"", "headers": []})

    await handle_unhandled_exception(request, RuntimeError("fail"))

    assert len(caplog.records) == 1
    assert logged_path in caplog.records[0].getMessage()


@mock.patch("server.error_handlers.logger")
def test_handle_unhandled_exception_logs_trace_and_span_id_of_failed_span(mock_logger: mock.MagicMock, client: TestClient):
    resp = client.post("/traced")

    error_id = resp.headers["X-Error-Id"]
    assert mock_logger.error.call_args == mock.call(
        "Unhandled %s on %s %r (%s)",
        "RuntimeError",
        "POST",
        "/traced",
        f"error_id={error_id}, trace_id=0af7651916cd43dd8448eb211c80319c, span_id=00f067aa0ba902b7",
    )


@mock.patch("server.error_handlers.logger")
def test_handle_unhandled_exception_skips_invalid_span_context(mock_logger: mock.MagicMock, client: TestClient):
    resp = client.post("/invalid-span")

    assert mock_logger.error.call_args.args[-1] == f"error_id={resp.headers['X-Error-Id']}"


def test_handle_http_exception_keeps_response_without_error_id(client: TestClient):
    resp = client.get("/missing")

    assert resp.status_code == 404
    assert resp.json() == {"error": {"message": "Model not found", "type": "not_found_error", "param": None, "code": None}}
    assert "X-Error-Id" not in resp.headers


def test_handle_validation_error_keeps_response_without_error_id(client: TestClient):
    resp = client.post("/validated", json={})

    assert resp.status_code == 400
    assert resp.json()["error"]["type"] == "invalid_request_error"
    assert resp.json()["error"]["param"] == "name"
    assert "X-Error-Id" not in resp.headers


@mock.patch("server.error_handlers.logger")
def test_handle_unhandled_exception_logs_mid_stream_failure_without_second_response(mock_logger: mock.MagicMock, client: TestClient):
    resp = client.get("/stream")

    assert resp.status_code == 200
    assert "X-Error-Id" not in resp.headers
    assert mock_logger.error.call_count == 1
    assert mock_logger.error.call_args.args[1:4] == ("RuntimeError", "GET", "/stream")


@mock.patch("server.error_handlers.capture_exception")
def test_handle_unhandled_exception_reports_to_sentry_with_response_error_id(mock_capture_exception: mock.MagicMock, client: TestClient):
    resp = client.post("/traced")

    assert mock_capture_exception.call_count == 1
    exc = mock_capture_exception.call_args.args[0]
    assert isinstance(exc, RuntimeError)
    assert mock_capture_exception.call_args == mock.call(
        exc,
        error_id=resp.headers["X-Error-Id"],
        handled=False,
        span_context=SpanContext(trace_id=TRACE_ID, span_id=SPAN_ID, is_remote=False),
        tags={},
    )


@mock.patch("server.error_handlers.capture_exception")
def test_handle_unhandled_exception_reports_attached_tags(mock_capture_exception: mock.MagicMock, client: TestClient):
    resp = client.get("/tagged")

    assert resp.status_code == 500
    assert mock_capture_exception.call_count == 1
    assert mock_capture_exception.call_args.kwargs["error_id"] == resp.headers["X-Error-Id"]
    assert mock_capture_exception.call_args.kwargs["tags"] == {"service": "ollama", "model": "llama3", "error_type": "connection_error"}


@mock.patch("server.error_handlers.logger")
@mock.patch("server.error_handlers.capture_exception")
def test_handle_unhandled_exception_logs_but_skips_reported_exception(
    mock_capture_exception: mock.MagicMock, mock_logger: mock.MagicMock, client: TestClient
):
    resp = client.get("/reported")

    assert resp.status_code == 500
    assert "X-Error-Id" in resp.headers
    assert mock_capture_exception.call_count == 0
    assert mock_logger.error.call_count == 1


@mock.patch("server.utils.sentry.event_from_exception")
@mock.patch("server.utils.sentry.sentry_sdk")
@mock.patch("server.utils.sentry.sentry")
def test_handle_unhandled_exception_returns_500_when_sentry_reporting_fails(
    mock_sentry: mock.MagicMock, mock_sentry_sdk: mock.MagicMock, mock_event_from_exception: mock.MagicMock, client: TestClient
):
    mock_sentry.is_active = True
    mock_sentry_sdk.get_client.return_value.is_active.return_value = True
    mock_event_from_exception.return_value = ({}, {})
    mock_sentry_sdk.capture_event.side_effect = RuntimeError("sentry down")

    resp = client.get("/boom")

    assert mock_sentry_sdk.capture_event.call_count == 1
    assert resp.status_code == 500
    assert resp.json()["error"]["type"] == "api_error"
    assert resp.headers["X-Error-Id"] in resp.json()["error"]["message"]


@pytest.mark.parametrize(("method", "path"), [("GET", "/missing"), ("POST", "/validated")])
@mock.patch("server.error_handlers.capture_exception")
def test_handle_http_and_validation_errors_are_not_reported_to_sentry(
    mock_capture_exception: mock.MagicMock, client: TestClient, method: str, path: str
):
    client.request(method, path, json={})

    assert mock_capture_exception.call_count == 0
