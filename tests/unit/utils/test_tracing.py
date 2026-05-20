# DeepFellow Software Framework.
# Copyright © 2026 Simplito sp. z o.o.
#
# This file is part of the DeepFellow Software Framework (https://deepfellow.ai).
# This software is Licensed under the DeepFellow Free License.
#
# See the License for the specific language governing permissions and
# limitations under the License.

import inspect
from typing import Any
from unittest.mock import MagicMock, call, patch

import pytest
from fastapi import Request
from fastapi.responses import JSONResponse, StreamingResponse
from opentelemetry.trace.status import StatusCode
from pydantic import BaseModel

from server.utils.tracing import FuncArgs, InfraTracer


def _make_config(enabled: bool = True, endpoint: str = "http://localhost:4317") -> MagicMock:
    cfg = MagicMock()
    cfg.otel_tracing_enabled = enabled
    cfg.otel_exporter_otlp_endpoint = endpoint
    return cfg


def _make_span() -> MagicMock:
    span = MagicMock()
    span.set_attribute = MagicMock()
    span.set_status = MagicMock()
    span.record_exception = MagicMock()
    span.end = MagicMock()
    return span


def _make_request(
    method: str = "POST",
    path: str = "/path",
    headers: dict[str, str] | None = None,
) -> Request:
    if headers is None:
        headers = {
            "content-type": "application/json",
            "content-length": "42",
            "user-agent": "test-agent",
            "accept": "*/*",
            "host": "test.com",
        }
    scope = {
        "type": "http",
        "method": method,
        "path": path,
        "query_string": b"",
        "headers": [(k.lower().encode(), v.encode()) for k, v in headers.items()],
        "server": ("test.com", 80),
    }
    return Request(scope)


def _make_tracer_with_config(enabled: bool) -> InfraTracer:
    t = InfraTracer("svc")
    t.config = _make_config(enabled=enabled)
    return t


def _bind(func: Any, *args: Any, **kwargs: Any) -> inspect.BoundArguments:
    sig = inspect.signature(func)
    bound = sig.bind(*args, **kwargs)
    bound.apply_defaults()
    return bound


def _patch_otel_context_managers():
    return [
        patch("server.utils.tracing.TracerProvider"),
        patch("server.utils.tracing.OTLPSpanExporter"),
        patch("server.utils.tracing.BatchSpanProcessor"),
        patch("server.utils.tracing.trace.set_tracer_provider"),
        patch("server.utils.tracing.trace.get_tracer"),
    ]


def test_func_args_defaults_are_none() -> None:
    args = FuncArgs()

    assert args.request is None
    assert args.query is None
    assert args.model is None


def test_infra_tracer_service_name_stored() -> None:
    t = InfraTracer("my-service")

    assert t.service_name == "my-service"


def test_infra_tracer_config_and_tracer_are_none() -> None:
    t = InfraTracer("svc")

    assert t.config is None
    assert t.tracer is None


def test_get_config_loads_config_on_first_call() -> None:
    t = InfraTracer("svc")
    cfg = _make_config()

    with patch("server.utils.tracing.load_config", return_value=cfg) as mock_load:
        result = t._get_config()  # pyright: ignore[reportPrivateUsage]

    assert result is cfg
    assert mock_load.call_count == 1


def test_get_config_caches_config_on_subsequent_calls() -> None:
    t = InfraTracer("svc")
    cfg = _make_config()

    with patch("server.utils.tracing.load_config", return_value=cfg) as mock_load:
        t._get_config()  # pyright: ignore[reportPrivateUsage]
        t._get_config()  # pyright: ignore[reportPrivateUsage]

    assert mock_load.call_count == 1


def test_get_tracer_returns_tracer() -> None:
    t = InfraTracer("svc")
    t.config = _make_config(endpoint="http://otel:4317")
    fake_tracer = MagicMock()
    patches = _patch_otel_context_managers()

    with patches[0], patches[1], patches[2], patches[3], patch("server.utils.tracing.trace.get_tracer", return_value=fake_tracer):
        result = t._get_tracer()  # pyright: ignore[reportPrivateUsage]

    assert result is fake_tracer


def test_get_tracer_caches_tracer() -> None:
    t = InfraTracer("svc")
    t.config = _make_config()
    fake_tracer = MagicMock()
    patches = _patch_otel_context_managers()

    with patches[0], patches[1], patches[2], patches[3], patch("server.utils.tracing.trace.get_tracer", return_value=fake_tracer):
        t._get_tracer()  # pyright: ignore[reportPrivateUsage]
        t._get_tracer()  # pyright: ignore[reportPrivateUsage]

    assert t.tracer is fake_tracer


def test_set_success_attributes_sets_execution_time_and_ok_status() -> None:
    t = InfraTracer("svc")
    span = _make_span()

    t._set_success_attributes(span, 0.5)  # pyright: ignore[reportPrivateUsage]

    assert span.set_attribute.call_args == call("execution.time_ms", 500.0)
    assert span.set_status.call_count == 1
    status_arg = span.set_status.call_args[0][0]
    assert status_arg.status_code == StatusCode.OK


def test_set_success_attributes_rounds_execution_time() -> None:
    t = InfraTracer("svc")
    span = _make_span()

    t._set_success_attributes(span, 1.0 / 3)  # pyright: ignore[reportPrivateUsage]

    call = span.set_attribute.call_args_list[0]
    assert call[0][0] == "execution.time_ms"
    assert call[0][1] == round((1.0 / 3) * 1000, 2)


def test_set_error_attributes_sets_error_attributes() -> None:
    t = InfraTracer("svc")
    span = _make_span()
    err = ValueError("something broke")

    t._set_error_attributes(span, 0.2, err)  # pyright: ignore[reportPrivateUsage]

    keys_set = {call[0][0] for call in span.set_attribute.call_args_list}
    assert "execution.time_ms" in keys_set
    assert "http.status_code" in keys_set
    assert "error.type" in keys_set
    assert "error.message" in keys_set
    assert "error.stacktrace" in keys_set


def test_set_error_attributes_records_exception_and_error_status() -> None:
    t = InfraTracer("svc")
    span = _make_span()
    err = RuntimeError("boom")

    t._set_error_attributes(span, 0.1, err)  # pyright: ignore[reportPrivateUsage]

    assert span.record_exception.call_count == 1
    assert span.record_exception.call_args == call(err)
    assert span.set_status.call_count == 1
    status_arg = span.set_status.call_args[0][0]
    assert status_arg.status_code == StatusCode.ERROR


def test_set_error_attributes_error_type_is_class_name() -> None:
    t = InfraTracer("svc")
    span = _make_span()
    err = TypeError("bad type")

    t._set_error_attributes(span, 0.0, err)  # pyright: ignore[reportPrivateUsage]

    attr_map = {c[0][0]: c[0][1] for c in span.set_attribute.call_args_list}
    assert attr_map["error.type"] == "TypeError"
    assert attr_map["error.message"] == "bad type"


def test_bind_args_binds_correctly() -> None:
    t = InfraTracer("svc")

    def func(a: int, b: str = "x") -> None:
        pass

    bound = t._bind_args(func, (1,), {})  # pyright: ignore[reportPrivateUsage]

    assert bound is not None
    assert bound.arguments["a"] == 1
    assert bound.arguments["b"] == "x"


def test_bind_args_returns_none_on_binding_failure() -> None:
    t = InfraTracer("svc")

    def func(a: int) -> None:
        pass

    result = t._bind_args(func, (), {})  # pyright: ignore[reportPrivateUsage]

    assert result is None


def test_extract_arguments_extracts_request() -> None:
    t = InfraTracer("svc")
    req = _make_request()

    def func(request: Request) -> None:
        pass

    bound = _bind(func, req)

    result = t._extract_arguments(bound)  # pyright: ignore[reportPrivateUsage]

    assert result.request is req


def test_extract_arguments_extracts_model() -> None:
    t = InfraTracer("svc")

    class M(BaseModel):
        x: int

    def func(model: M) -> None:
        pass

    m = M(x=1)
    bound = _bind(func, m)

    result = t._extract_arguments(bound)  # pyright: ignore[reportPrivateUsage]

    assert result.model is m


def test_extract_arguments_extracts_query() -> None:
    t = InfraTracer("svc")

    class Q(BaseModel):
        q: str

    def func(query: Q) -> None:
        pass

    q = Q(q="hello")
    bound = _bind(func, q)

    result = t._extract_arguments(bound)  # pyright: ignore[reportPrivateUsage]

    assert result.query is q


def test_extract_arguments_ignores_non_request_type() -> None:
    t = InfraTracer("svc")

    def func(request: str) -> None:
        pass

    bound = _bind(func, "not-a-request")

    result = t._extract_arguments(bound)  # pyright: ignore[reportPrivateUsage]

    assert result.request is None


def test_extract_arguments_empty_when_no_matching_params() -> None:
    t = InfraTracer("svc")

    def func(x: int) -> None:
        pass

    bound = _bind(func, 42)

    result = t._extract_arguments(bound)  # pyright: ignore[reportPrivateUsage]

    assert result.request is None
    assert result.model is None
    assert result.query is None


def test_add_attributes_to_span_adds_request_attributes() -> None:
    t = InfraTracer("svc")
    span = _make_span()
    req = _make_request()
    args = FuncArgs()
    args.request = req

    t._add_attributes_to_span(args, span)  # pyright: ignore[reportPrivateUsage]

    keys = {c[0][0] for c in span.set_attribute.call_args_list}
    assert "request.method" in keys
    assert "request.url" in keys
    assert "request.path" in keys
    assert "request.content_type" in keys


def test_add_attributes_to_span_adds_model_attributes() -> None:
    t = InfraTracer("svc")
    span = _make_span()

    class M(BaseModel):
        value: int

    args = FuncArgs()
    args.model = M(value=7)

    t._add_attributes_to_span(args, span)  # pyright: ignore[reportPrivateUsage]

    keys = {c[0][0] for c in span.set_attribute.call_args_list}
    assert "model.value" in keys


def test_add_attributes_to_span_adds_query_attributes() -> None:
    t = InfraTracer("svc")
    span = _make_span()

    class Q(BaseModel):
        term: str

    args = FuncArgs()
    args.query = Q(term="hello")

    t._add_attributes_to_span(args, span)  # pyright: ignore[reportPrivateUsage]

    keys = {c[0][0] for c in span.set_attribute.call_args_list}
    assert "query.term" in keys


def test_add_attributes_to_span_silently_ignores_exceptions() -> None:
    t = InfraTracer("svc")
    span = _make_span()
    span.set_attribute.side_effect = Exception("unexpected")
    args = FuncArgs()
    req = _make_request()
    args.request = req

    t._add_attributes_to_span(args, span)  # pyright: ignore[reportPrivateUsage]


@pytest.mark.asyncio
async def test_trace_request_skips_tracing_when_disabled() -> None:
    t = _make_tracer_with_config(enabled=False)

    @t.trace_request()
    async def handler():
        return "result"

    result = await handler()

    assert result == "result"


@pytest.mark.asyncio
async def test_trace_request_returns_result_when_tracing_enabled() -> None:
    t = _make_tracer_with_config(enabled=True)
    span = _make_span()
    fake_tracer = MagicMock()
    fake_tracer.start_span.return_value = span
    t.tracer = fake_tracer

    with patch("server.utils.tracing.trace.set_span_in_context"):

        @t.trace_request()
        async def handler():
            return "ok"

        result = await handler()

    assert result == "ok"
    assert span.end.call_count == 1


@pytest.mark.asyncio
async def test_trace_request_re_raises_exception_and_ends_span() -> None:
    t = _make_tracer_with_config(enabled=True)
    span = _make_span()
    fake_tracer = MagicMock()
    fake_tracer.start_span.return_value = span
    t.tracer = fake_tracer

    with patch("server.utils.tracing.trace.set_span_in_context"):

        @t.trace_request()
        async def handler():
            raise ValueError("fail")

        with pytest.raises(ValueError, match="fail"):
            await handler()

    assert span.record_exception.call_count == 1
    assert span.end.call_count == 1


@pytest.mark.asyncio
async def test_trace_request_streaming_response_sets_attributes() -> None:
    t = _make_tracer_with_config(enabled=True)
    span = _make_span()
    fake_tracer = MagicMock()
    fake_tracer.start_span.return_value = span
    t.tracer = fake_tracer

    async def body_gen():
        yield b"data"

    with patch("server.utils.tracing.trace.set_span_in_context"):

        @t.trace_request()
        async def handler():
            return StreamingResponse(body_gen(), media_type="text/plain")

        await handler()

    keys = {c[0][0] for c in span.set_attribute.call_args_list}
    assert "response.status" in keys
    assert "response.content_type" in keys


@pytest.mark.asyncio
async def test_trace_request_json_response_sets_attributes() -> None:
    t = _make_tracer_with_config(enabled=True)
    span = _make_span()
    fake_tracer = MagicMock()
    fake_tracer.start_span.return_value = span
    t.tracer = fake_tracer

    with patch("server.utils.tracing.trace.set_span_in_context"):

        @t.trace_request()
        async def handler():
            return JSONResponse(content={"ok": True})

        await handler()

    attr_map = {c[0][0]: c[0][1] for c in span.set_attribute.call_args_list}
    assert attr_map.get("response.content_type") == "application/json"


@pytest.mark.asyncio
async def test_trace_request_base_model_response_sets_attributes() -> None:
    t = _make_tracer_with_config(enabled=True)
    span = _make_span()
    fake_tracer = MagicMock()
    fake_tracer.start_span.return_value = span
    t.tracer = fake_tracer

    class Resp(BaseModel):
        value: int

    with patch("server.utils.tracing.trace.set_span_in_context"):

        @t.trace_request()
        async def handler():
            return Resp(value=1)

        await handler()

    attr_map = {c[0][0]: c[0][1] for c in span.set_attribute.call_args_list}
    assert attr_map.get("response.status") == 200
    assert attr_map.get("response.content_type") == "application/json"


@pytest.mark.asyncio
async def test_trace_request_unknown_response_type_sets_unknown_attributes() -> None:
    t = _make_tracer_with_config(enabled=True)
    span = _make_span()
    fake_tracer = MagicMock()
    fake_tracer.start_span.return_value = span
    t.tracer = fake_tracer
    with patch("server.utils.tracing.trace.set_span_in_context"):

        @t.trace_request()
        async def handler():
            return 42

        await handler()

    attr_map = {c[0][0]: c[0][1] for c in span.set_attribute.call_args_list}
    assert attr_map.get("response.status") == "<unknown>"
    assert attr_map.get("response.content_type") == "<unknown>"


@pytest.mark.asyncio
async def test_trace_request_preserves_function_signature() -> None:
    t = _make_tracer_with_config(enabled=False)

    async def my_handler(x: int, y: str = "a") -> str:
        return f"{x}{y}"

    wrapped = t.trace_request()(my_handler)

    assert wrapped.__name__ == "my_handler"
