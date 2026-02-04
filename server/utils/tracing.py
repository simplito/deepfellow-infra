# DeepFellow Software Framework.
# Copyright © 2026 Simplito sp. z o.o.
#
# This file is part of the DeepFellow Software Framework (https://deepfellow.ai).
# This software is Licensed under the DeepFellow Free License.
#
# See the License for the specific language governing permissions and
# limitations under the License.

"""OpenTelemetry tracing utilities for request and streaming response tracing."""

import functools
import inspect
import logging
import time
import traceback
from collections.abc import Callable
from typing import Any

from fastapi import Request
from fastapi.responses import JSONResponse, StreamingResponse
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import Status, StatusCode, TracerProvider  # type: ignore
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.trace import Span, Tracer
from pydantic import BaseModel

from server.config import AppSettings, load_config

uvicorn_logger = logging.getLogger("uvicorn")


class FuncArgs:
    request: Request | None = None
    query: BaseModel | None = None
    model: BaseModel | None = None


class InfraTracer:
    service_name: str
    config: AppSettings | None
    tracer: Tracer | None

    def __init__(self, service_name: str) -> None:
        self.service_name = service_name
        self.config = None
        self.tracer = None

    def _get_config(self) -> AppSettings:
        if self.config is None:
            self.config = load_config()
        return self.config

    def _get_tracer(self) -> Tracer:
        if self.tracer is None:
            resource = Resource(attributes={"service.name": self.service_name})
            provider = TracerProvider(resource=resource)
            # Get OTLP endpoint from config

            otlp_endpoint = self._get_config().otel_exporter_otlp_endpoint

            # processor = BatchSpanProcessor(ConsoleSpanExporter())
            processor = BatchSpanProcessor(OTLPSpanExporter(endpoint=otlp_endpoint, insecure=True))
            provider.add_span_processor(processor)

            trace.set_tracer_provider(provider)
            self.tracer = trace.get_tracer(__name__)
        return self.tracer

    def _set_success_attributes(self, span: Span, execution_time: float) -> None:
        """Set success attributes on the span."""
        attributes = {"execution.time_ms": round(execution_time * 1000, 2)}

        for key, value in attributes.items():
            span.set_attribute(key, value)
        span.set_status(Status(StatusCode.OK))

    def _set_error_attributes(self, span: Span, execution_time: float, error: Exception) -> None:
        """Set error attributes on the span."""
        attributes = {
            "execution.time_ms": round(execution_time * 1000, 2),
            "http.status_code": 500,
            "error.type": type(error).__name__,
            "error.message": str(error),
            "error.stacktrace": traceback.format_exc(),
        }
        for key, value in attributes.items():
            span.set_attribute(key, value)
        span.record_exception(error)
        span.set_status(Status(StatusCode.ERROR, description=str(error)))

    def _bind_args(self, func: Callable[..., Any], args: Any, kwargs: Any) -> inspect.BoundArguments | None:  # noqa: ANN401
        try:
            sig = inspect.signature(func)
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()
            return bound_args  # noqa: TRY300
        except Exception:
            return None

    def _extract_arguments(self, bound_args: inspect.BoundArguments) -> FuncArgs:
        res = FuncArgs()
        if "request" in bound_args.arguments and isinstance(request := bound_args.arguments["request"], Request):
            res.request = request
        if "model" in bound_args.arguments and isinstance(model := bound_args.arguments["model"], BaseModel):
            res.model = model
        if "query" in bound_args.arguments and isinstance(query := bound_args.arguments["query"], BaseModel):
            res.query = query
        return res

    def _add_attributes_to_span(self, args: FuncArgs, span: Span) -> None:
        """Extract span attributes from function arguments."""
        try:
            if args.request:
                span.set_attribute("request.method", str(args.request.method))
                span.set_attribute("request.url", str(args.request.url))
                span.set_attribute("request.path", args.request.url.path)
                span.set_attribute("request.content_type", args.request.headers.get("content-type") or "<unknown>")
                span.set_attribute("request.content_length", args.request.headers.get("content-length") or "<unknown>")
                span.set_attribute("request.user-agent", args.request.headers.get("user-agent") or "<unknown>")
                span.set_attribute("request.accept", args.request.headers.get("accept") or "<unknown>")
                span.set_attribute("request.host", args.request.headers.get("host") or "<unknown>")

            if args.model:
                for key, value in args.model.model_dump().items():
                    span.set_attribute(f"model.{key}", value)
            if args.query:
                for key, value in args.query.model_dump().items():
                    span.set_attribute(f"query.{key}", value)

        except Exception:
            pass

    def trace_request(self) -> Callable[..., Any]:
        """Create a tracing decorator for async function."""

        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            @functools.wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
                if not self._get_config().otel_tracing_enabled:
                    return await func(*args, **kwargs)
                span_name = "test"
                span = self._get_tracer().start_span(span_name)
                trace.set_span_in_context(span)
                start_time = time.time()
                # Extract and set attributes from function arguments
                bound_args = self._bind_args(func, args, kwargs)
                func_args = self._extract_arguments(bound_args) if bound_args else FuncArgs()
                self._add_attributes_to_span(func_args, span)

                def span_end() -> None:
                    execution_time = time.time() - start_time
                    self._set_success_attributes(span, execution_time)
                    span.end()

                try:
                    res = await func(*args, **kwargs)
                except Exception as e:
                    span.record_exception(e)
                    span_end()
                    raise
                else:
                    if isinstance(res, StreamingResponse):
                        span.set_attribute("response.status", res.status_code)
                        span.set_attribute("response.content_type", res.media_type or "<unknown>")
                        span_end()
                    if isinstance(res, JSONResponse):
                        span.set_attribute("response.status", res.status_code)
                        span.set_attribute("response.content_type", "application/json")
                        span_end()
                    elif isinstance(res, BaseModel):
                        span.set_attribute("response.status", 200)
                        span.set_attribute("response.content_type", "application/json")
                        span_end()
                    else:
                        span.set_attribute("response.status", "<unknown>")
                        span.set_attribute("response.content_type", "<unknown>")
                        span_end()
                    return res

            return async_wrapper

        return decorator


tracer = InfraTracer(service_name="llm-audit")
