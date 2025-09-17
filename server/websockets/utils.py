"""Utils for websocket."""

import logging
import urllib.parse
from collections.abc import AsyncGenerator, Callable
from typing import TYPE_CHECKING, Any

from fastapi import Request
from fastapi.responses import StreamingResponse

from server.config import ParentInfra
from server.websockets.loadbalancer import get_infra_url, get_key
from server.websockets.models import InfraConnectData
from server.websockets.subinfra import ExternalInfraWsManager

if TYPE_CHECKING:
    from server.config import AppSettings
    from server.websockets.models import InfraInfo

logger = logging.getLogger("uvicorn.error")


def create_infra_uri(parent_infra_config: ParentInfra) -> str:
    """Create infra uri."""
    uri = ""

    if parent_infra_config.ws_url:
        uri = f"{parent_infra_config.ws_url}/ws"

    if parent_infra_config.api_key:
        uri += f"?key={parent_infra_config.api_key}"

    return uri


def auth_header(key: str) -> dict[str, str]:
    """Get aut header."""
    return {"Authorization": f"Bearer {key}"}


def get_proxy_url(url: str, request: Request) -> str:
    """Get proxy url."""
    return urllib.parse.urljoin(url, request.url.path)


async def add_usage_for_response(f: StreamingResponse, model: str, external_ws: ExternalInfraWsManager) -> AsyncGenerator[Any]:
    """Add usage for response."""
    try:
        async for chunk in f.body_iterator:
            yield chunk
    finally:
        external_ws.remove_usage(model)


def handle_usage(f: Callable[..., Any]) -> Callable[..., Any]:
    """Measures execution time."""

    async def wrapper(*args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
        request: Request = kwargs.get("request", args[1])
        model: str = kwargs.get("model", args[2])
        external_ws: ExternalInfraWsManager = request.app.state.external_ws_manager
        external_ws.add_usage(model)
        try:
            resp = await f(*args, **kwargs)
            if isinstance(resp, StreamingResponse):
                gen = add_usage_for_response(resp, model, external_ws)
                return StreamingResponse(gen, media_type=resp.media_type, status_code=resp.status_code, headers=resp.headers)
        finally:
            external_ws.remove_usage(model)
        return resp

    return wrapper


def get_lazy_infra(request: Request, model: str) -> InfraConnectData:
    """Get most lazy (with least requests) infra data."""
    config: AppSettings = request.app.state.config
    models_usage: dict[str, dict[str, int]] = request.app.state.models_usage
    infra_infos: list[InfraInfo] = request.app.state.infra_infos
    url: str = get_infra_url(model, models_usage)
    is_inside = bool(url == config.url)
    key: str = "" if is_inside else get_key(url, infra_infos)
    return InfraConnectData(url, key, is_inside)
