# DeepFellow Software Framework.
# Copyright © 2026 Simplito sp. z o.o.
#
# This file is part of the DeepFellow Software Framework (https://deepfellow.ai).
# This software is Licensed under the DeepFellow Free License.
#
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import cast
from unittest.mock import AsyncMock, MagicMock, call, patch

import pytest

from server.models.api import Model, ModelProps
from server.models.mesh import MeshInfo
from server.utils.exceptions import ApiError
from server.websockets.infra_websocket_server import (
    Authorized,
    InfraWebsocketServer,
    InfraWsData,
)
from server.websockets.models import InitRequest, UpdateModelsRequest, UsageChangeRequest

MESH_KEY = "secret-key"


def make_model(name: str = "test-model", model_type: str = "llm") -> Model:
    return Model(
        id="reg-1",
        name=name,
        type=model_type,
        props=ModelProps(private=False, type=model_type, endpoints=["chat"]),
        usage=0,
    )


def make_server() -> InfraWebsocketServer:
    config = MagicMock()
    config.mesh_key.get_secret_value.return_value = MESH_KEY
    parent_infra = MagicMock()
    endpoint_registry = MagicMock()
    return InfraWebsocketServer(
        config=config,
        parent_infra=parent_infra,
        endpoint_registry=endpoint_registry,
    )


def make_context(authorized: Authorized | None = None) -> MagicMock:
    ctx = MagicMock()
    ctx.data = InfraWsData(authorized=authorized)
    return ctx


def make_authorized(name: str = "sub", url: str = "http://sub", api_key: str = "key") -> Authorized:
    return Authorized(name=name, url=url, api_key=api_key, models=[make_model()])


def test_init_stores_dependencies():
    config = MagicMock()
    config.mesh_key.get_secret_value.return_value = MESH_KEY
    parent_infra = MagicMock()
    endpoint_registry = MagicMock()

    server = InfraWebsocketServer(config=config, parent_infra=parent_infra, endpoint_registry=endpoint_registry)

    assert server.config is config
    assert server.parent_infra is parent_infra
    assert server.endpoint_registry is endpoint_registry


def test_create_bag_returns_empty_data():
    server = make_server()

    bag = server.create_bag()

    assert isinstance(bag, InfraWsData)
    assert bag.authorized is None


def test_handle_disconnect_authorised_calls_update_models():
    server = make_server()
    auth = make_authorized()
    ctx = make_context(authorized=auth)

    server.handle_disconnect(ctx)

    assert cast("MagicMock", server.endpoint_registry).update_models.call_count == 1
    assert cast("MagicMock", server.endpoint_registry).update_models.call_args == call(auth.models, [], auth.url, auth.api_key)


def test_handle_disconnect_unauthorised_does_not_call_update_models():
    server = make_server()
    ctx = make_context(authorized=None)

    server.handle_disconnect(ctx)

    assert cast("MagicMock", server.endpoint_registry).update_models.call_count == 0


@pytest.mark.asyncio
async def test_process_message_sends_non_none_result():
    server = make_server()
    server.server.process = AsyncMock(return_value='{"result": "OK"}')
    ctx = make_context()

    await server.process_message("msg", ctx)

    assert ctx.send.call_count == 1
    assert ctx.send.call_args == call('{"result": "OK"}')


@pytest.mark.asyncio
async def test_process_message_does_not_send_none_result():
    server = make_server()
    server.server.process = AsyncMock(return_value=None)
    ctx = make_context()

    await server.process_message("msg", ctx)

    assert ctx.send.call_count == 0


@pytest.mark.asyncio
async def test_dispatch_init():
    server = make_server()
    server._on_init = AsyncMock(return_value="OK")  # pyright: ignore[reportPrivateUsage]
    params = {"auth": MESH_KEY, "name": "sub", "url": "http://sub", "api_key": "key", "models": []}
    ctx = InfraWsData(authorized=None)

    result = await server._handle_json_rpc_request("init", params, ctx)  # pyright: ignore[reportPrivateUsage]

    server._on_init.assert_awaited_once()  # pyright: ignore[reportPrivateUsage]
    assert result == "OK"


@pytest.mark.asyncio
async def test_dispatch_usage_change():
    server = make_server()
    server._on_usage_change = MagicMock(return_value="OK")  # pyright: ignore[reportPrivateUsage]
    params = {"id": "reg-1", "usage": 5}
    ctx = InfraWsData(authorized=make_authorized())

    result = await server._handle_json_rpc_request("usage_change", params, ctx)  # pyright: ignore[reportPrivateUsage]

    assert server._on_usage_change.call_count == 1  # pyright: ignore[reportPrivateUsage]
    assert result == "OK"


@pytest.mark.asyncio
async def test_dispatch_update_models():
    server = make_server()
    server._on_update_models = MagicMock(return_value="OK")  # pyright: ignore[reportPrivateUsage]
    params = {"models": []}
    ctx = InfraWsData(authorized=make_authorized())

    result = await server._handle_json_rpc_request("update_models", params, ctx)  # pyright: ignore[reportPrivateUsage]

    assert server._on_update_models.call_count == 1  # pyright: ignore[reportPrivateUsage]
    assert result == "OK"


@pytest.mark.asyncio
async def test_dispatch_unknown_method_raises_api_error():
    server = make_server()
    ctx = InfraWsData(authorized=None)

    with pytest.raises(ApiError) as exc_info:
        await server._handle_json_rpc_request("unknown_method", {}, ctx)  # pyright: ignore[reportPrivateUsage]

    assert exc_info.value.code == -32601


def test_try_parse_valid_returns_model():
    server = make_server()
    params = {"id": "reg-1", "usage": 3}

    result = server._try_parse(params, UsageChangeRequest)  # pyright: ignore[reportPrivateUsage]

    assert isinstance(result, UsageChangeRequest)
    assert result.id == "reg-1"
    assert result.usage == 3


def test_try_parse_invalid_raises_api_error():
    server = make_server()

    with pytest.raises(ApiError) as exc_info:
        server._try_parse({"bad": "data"}, UsageChangeRequest)  # pyright: ignore[reportPrivateUsage]

    assert exc_info.value.code == -32602


@pytest.mark.asyncio
async def test_on_init_already_authorised_raises():
    server = make_server()
    ctx = InfraWsData(authorized=make_authorized())
    params = InitRequest(auth=MESH_KEY, name="sub", url="http://sub", api_key="key", models=[])

    with pytest.raises(ApiError) as exc_info:
        await server._on_init(params, ctx)  # pyright: ignore[reportPrivateUsage]

    assert exc_info.value.code == 1


@pytest.mark.asyncio
async def test_on_init_wrong_key_raises():
    server = make_server()
    ctx = InfraWsData(authorized=None)
    params = InitRequest(auth="wrong-key", name="sub", url="http://sub", api_key="key", models=[])

    with pytest.raises(ApiError) as exc_info:
        await server._on_init(params, ctx)  # pyright: ignore[reportPrivateUsage]

    assert exc_info.value.code == 2


@pytest.mark.asyncio
async def test_on_init_success_without_check_key():
    server = make_server()
    ctx = InfraWsData(authorized=None)
    models = [make_model()]
    params = InitRequest(auth=MESH_KEY, name="sub", url="http://sub", api_key="key", models=models)

    result = await server._on_init(params, ctx)  # pyright: ignore[reportPrivateUsage]

    assert result == "OK"
    assert ctx.authorized is not None
    assert ctx.authorized.name == "sub"
    assert cast("MagicMock", server.endpoint_registry).update_models.call_count == 1
    assert cast("MagicMock", server.endpoint_registry).update_models.call_args == call([], models, "http://sub", "key")


@pytest.mark.asyncio
async def test_on_init_with_check_key_http_200():
    server = make_server()
    ctx = InfraWsData(authorized=None)
    models = [make_model()]
    params = InitRequest(auth=MESH_KEY, name="sub", url="http://sub", api_key="key", models=models, check_key="ck")

    http_response = MagicMock()
    http_response.response.status = 200

    with patch("server.websockets.infra_websocket_server.make_http_request", new=AsyncMock(return_value=http_response)):
        result = await server._on_init(params, ctx)  # pyright: ignore[reportPrivateUsage]

    assert result == "OK"
    assert ctx.authorized is not None
    assert cast("MagicMock", server.endpoint_registry).update_models.call_count == 1


@pytest.mark.asyncio
async def test_on_init_with_check_key_non_200_raises():
    server = make_server()
    ctx = InfraWsData(authorized=None)
    params = InitRequest(auth=MESH_KEY, name="sub", url="http://sub", api_key="key", models=[], check_key="ck")

    http_response = MagicMock()
    http_response.response.status = 403

    with (
        patch("server.websockets.infra_websocket_server.make_http_request", new=AsyncMock(return_value=http_response)),
        pytest.raises(ApiError) as exc_info,
    ):
        await server._on_init(params, ctx)  # pyright: ignore[reportPrivateUsage]

    assert exc_info.value.code == 4


def test_on_usage_change_authorised():
    server = make_server()
    ctx = InfraWsData(authorized=make_authorized())
    params = UsageChangeRequest(id="reg-1", usage=7)

    result = server._on_usage_change(params, ctx)  # pyright: ignore[reportPrivateUsage]

    assert result == "OK"
    assert cast("MagicMock", server.endpoint_registry).update_usage.call_count == 1
    assert cast("MagicMock", server.endpoint_registry).update_usage.call_args == call(params)


def test_on_usage_change_unauthorised_raises():
    server = make_server()
    ctx = InfraWsData(authorized=None)
    params = UsageChangeRequest(id="reg-1", usage=7)

    with pytest.raises(ApiError) as exc_info:
        server._on_usage_change(params, ctx)  # pyright: ignore[reportPrivateUsage]

    assert exc_info.value.code == 3


def test_on_update_models_authorised():
    server = make_server()
    auth = make_authorized()
    old_models = list(auth.models)
    ctx = InfraWsData(authorized=auth)
    new_models = [make_model("new-model")]
    params = UpdateModelsRequest(models=new_models)

    result = server._on_update_models(params, ctx)  # pyright: ignore[reportPrivateUsage]

    assert result == "OK"
    assert cast("MagicMock", server.endpoint_registry).update_models.call_count == 1
    assert cast("MagicMock", server.endpoint_registry).update_models.call_args == call(old_models, new_models, auth.url, auth.api_key)
    assert ctx.authorized is not None
    assert ctx.authorized.models == new_models


def test_on_update_models_unauthorised_raises():
    server = make_server()
    ctx = InfraWsData(authorized=None)
    params = UpdateModelsRequest(models=[])

    with pytest.raises(ApiError) as exc_info:
        server._on_update_models(params, ctx)  # pyright: ignore[reportPrivateUsage]

    assert exc_info.value.code == 3


def test_get_mesh_info_no_connections():
    server = make_server()
    server.connections = set()  # type: ignore[assignment]

    info = server.get_mesh_info()

    assert isinstance(info, MeshInfo)
    assert info.connections == []


def test_get_mesh_info_mixed_connections():
    server = make_server()

    auth = make_authorized(name="infra-a", url="http://a")
    authorised_ctx = MagicMock()
    authorised_ctx.data = InfraWsData(authorized=auth)

    unauthorised_ctx = MagicMock()
    unauthorised_ctx.data = InfraWsData(authorized=None)

    server.connections = {authorised_ctx, unauthorised_ctx}  # type: ignore[assignment]

    info = server.get_mesh_info()

    assert len(info.connections) == 1
    assert info.connections[0].name == "infra-a"
    assert info.connections[0].url == "http://a"
