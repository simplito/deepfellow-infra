# DeepFellow Software Framework.
# Copyright © 2026 Simplito sp. z o.o.
#
# This file is part of the DeepFellow Software Framework (https://deepfellow.ai).
# This software is Licensed under the DeepFellow Free License.
#
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import cast
from unittest.mock import AsyncMock, MagicMock

import pytest

from server.utils.json_rpc_client import JsonRpcClient
from server.websockets.infra_client import InfraClient
from server.websockets.infra_websocket_server import InfraWebsocketServer, InfraWsData
from server.websockets.models import AncestorInfo, InitRequest, InitResponse, TopologyUpdateRequest
from server.websockets.parent_infra import ParentInfra


def _make_server(
    mesh_key: str = "secret",
    connect_to_mesh_url: str = "http://core.url",
    parent_enabled: bool = True,
    parent_ancestors: list[str] | None = None,
    infra_url: str = "http://self.url",
    name: str = "self",
) -> InfraWebsocketServer:
    config = MagicMock()
    config.mesh_key.get_secret_value.return_value = mesh_key
    config.connect_to_mesh_url = connect_to_mesh_url
    config.infra_url = infra_url
    config.name = name

    parent_infra = MagicMock()
    parent_infra.enabled = parent_enabled
    parent_infra.ancestors = [AncestorInfo(url=u, name="") for u in (parent_ancestors or [])]
    parent_infra.parent_urls = [connect_to_mesh_url] if parent_enabled and connect_to_mesh_url else []
    parent_infra.ws = None  # not connected → topology_update sends are skipped

    endpoint_registry = MagicMock()
    endpoint_registry.update_models.return_value = None
    endpoint_registry.list_models.return_value = []

    return InfraWebsocketServer(config, parent_infra, endpoint_registry)


def _init_params(**kwargs: object) -> InitRequest:
    defaults: dict[str, object] = {"auth": "secret", "name": "child", "url": "http://child.url", "api_key": "key", "models": []}
    defaults.update(kwargs)
    return InitRequest(**defaults)  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_on_init_returns_init_response_with_ancestors() -> None:
    # ParentInfraGroup.ancestors already includes the parent URL + its ancestors
    server = _make_server(parent_ancestors=["http://core.url"])
    context = InfraWsData(authorized=None)
    result = await server._on_init(_init_params(), context)  # type: ignore[reportPrivateUsage]
    assert isinstance(result, InitResponse)
    assert [a.url for a in result.ancestors] == ["http://self.url", "http://core.url"]


@pytest.mark.asyncio
async def test_on_init_root_node_returns_empty_ancestors() -> None:
    server = _make_server(parent_enabled=False, parent_ancestors=[])
    context = InfraWsData(authorized=None)
    result = await server._on_init(_init_params(), context)  # type: ignore[reportPrivateUsage]
    assert [a.url for a in result.ancestors] == ["http://self.url"]


@pytest.mark.asyncio
async def test_on_init_propagates_grandparent_ancestors() -> None:
    server = _make_server(connect_to_mesh_url="http://a.url", parent_ancestors=["http://a.url", "http://core.url"])
    context = InfraWsData(authorized=None)
    result = await server._on_init(_init_params(), context)  # type: ignore[reportPrivateUsage]
    assert [a.url for a in result.ancestors] == ["http://self.url", "http://a.url", "http://core.url"]


@pytest.mark.asyncio
async def test_on_init_sends_topology_update_join_to_parent() -> None:
    server = _make_server()
    context = InfraWsData(authorized=None)
    await server._on_init(_init_params(url="http://child.url", name="child"), context)  # type: ignore[reportPrivateUsage]

    cast("MagicMock", server.parent_infra.send_topology_update).assert_called_once_with("join", "http://child.url", "child", [], {})


@pytest.mark.asyncio
async def test_on_init_registers_child_in_sub_connections() -> None:
    server = _make_server()
    context = InfraWsData(authorized=None)
    await server._on_init(_init_params(url="http://child.url", name="child"), context)  # type: ignore[reportPrivateUsage]
    assert server._sub_connections["http://child.url"] == "child"  # type: ignore[reportPrivateUsage]


def test_handle_disconnect_sends_topology_update_leave() -> None:
    server = _make_server()

    context = MagicMock()
    context.data.authorized.url = "http://child.url"
    context.data.authorized.name = "child"
    context.data.authorized.models = []
    context.data.authorized.api_key = "key"

    server._sub_connections["http://child.url"] = "child"  # type: ignore[reportPrivateUsage]

    server.handle_disconnect(context)

    assert "http://child.url" not in server._sub_connections  # type: ignore[reportPrivateUsage]
    cast("MagicMock", server.parent_infra.send_topology_update).assert_called_once_with("leave", "http://child.url", "", [], {})


def test_handle_disconnect_cleans_nested_topology() -> None:
    server = _make_server()
    server.parent_infra.ws = None  # type: ignore[reportAttributeAccessIssue]  # suppress topology_update send

    context = MagicMock()
    context.data.authorized.url = "http://a.url"
    context.data.authorized.name = "A"
    context.data.authorized.models = []
    context.data.authorized.api_key = "key"

    server._sub_connections["http://a.url"] = "A"  # type: ignore[reportPrivateUsage]
    server._nested_topology["http://a.url"] = {"http://d.url": ("D", [])}  # type: ignore[reportPrivateUsage]

    server.handle_disconnect(context)

    assert "http://a.url" not in server._nested_topology  # type: ignore[reportPrivateUsage]


@pytest.mark.asyncio
async def test_on_start_stores_ancestors_from_init_response() -> None:
    config = MagicMock()
    config.connect_to_mesh_url = "http://core.url"
    config.connect_to_mesh_key.get_secret_value.return_value = "key"
    config.name = "node"
    config.infra_url = "http://node.url"
    config.infra_api_key.get_secret_value.return_value = "apikey"

    task_manager = MagicMock()
    parent = ParentInfra.__new__(ParentInfra)
    parent.config = config
    parent.task_manager = task_manager
    parent._ancestors = []  # type: ignore[misc]
    parent.process_loop = True

    endpoint_registry = MagicMock()
    endpoint_registry.list_models.return_value = []
    parent.endpoint_registry = endpoint_registry

    parent.infra_client = MagicMock(spec=InfraClient)
    ancestor = AncestorInfo(url="http://root.url", name="")
    parent.infra_client.init = AsyncMock(return_value=InitResponse(ancestors=[ancestor]))
    parent.one_time_key = MagicMock()
    parent.one_time_key.key = "otk"
    parent.get_children = dict  # type: ignore[misc]

    await parent.on_start()

    assert parent.ancestors == [ancestor]


@pytest.mark.asyncio
async def test_infra_client_init_handles_plain_ok_response() -> None:
    mock_rpc = MagicMock(spec=JsonRpcClient)
    mock_rpc.request = AsyncMock(return_value="OK")

    client = InfraClient(mock_rpc)
    params = InitRequest(auth="a", name="n", url="http://x.url", api_key="k", models=[])
    result = await client.init(params)

    assert isinstance(result, InitResponse)
    assert result.ancestors == []


@pytest.mark.asyncio
async def test_infra_client_init_parses_init_response() -> None:
    mock_rpc = MagicMock(spec=JsonRpcClient)
    mock_rpc.request = AsyncMock(return_value={"ancestors": [{"url": "http://root.url", "name": ""}]})

    client = InfraClient(mock_rpc)
    params = InitRequest(auth="a", name="n", url="http://x.url", api_key="k", models=[])
    result = await client.init(params)

    assert result.ancestors == [AncestorInfo(url="http://root.url", name="")]


def test_get_topology_returns_correct_structure() -> None:
    server = _make_server(infra_url="http://self.url", name="self", connect_to_mesh_url="http://core.url")

    # Simulate child A connected with grandchild D reported via topology_update
    server._sub_connections["http://a.url"] = "A"  # type: ignore[reportPrivateUsage]
    server._nested_topology["http://a.url"] = TopologyUpdateRequest(  # type: ignore[reportPrivateUsage]
        action="join",
        url="http://a.url",
        name="A",
        children={"http://d.url": TopologyUpdateRequest(action="join", url="http://d.url", name="D")},
    )

    conn_a = MagicMock()
    conn_a.data.authorized = MagicMock()
    conn_a.data.authorized.url = "http://a.url"
    conn_a.data.authorized.name = "A"
    conn_a.data.authorized.models = []
    server.connections = [conn_a]  # type: ignore[assignment]

    result = server.get_topology()

    assert isinstance(result, list)
    self_node = result[0]
    assert self_node.url == "http://self.url"
    assert self_node.name == "self"
    assert self_node.you_are_here is True
    assert len(self_node.children) == 1
    node_a = self_node.children[0]
    assert node_a.url == "http://a.url"
    assert node_a.name == "A"
    assert node_a.models == []
    assert len(node_a.children) == 1
    node_d = node_a.children[0]
    assert node_d.url == "http://d.url"
    assert node_d.name == "D"
    assert node_d.models == []


def test_get_topology_leaf_node_has_empty_children() -> None:
    server = _make_server()
    server._sub_connections["http://a.url"] = "A"  # type: ignore[reportPrivateUsage]

    conn_a = MagicMock()
    conn_a.data.authorized = MagicMock()
    conn_a.data.authorized.url = "http://a.url"
    conn_a.data.authorized.name = "A"
    conn_a.data.authorized.models = []
    server.connections = [conn_a]  # type: ignore[assignment]

    result = server.get_topology()

    assert result[0].children[0].children == []


def test_get_topology_shows_full_depth() -> None:
    """Topology must expose all levels, not just grandchildren."""
    server = _make_server(infra_url="http://self.url", name="self")

    conn_a = MagicMock()
    conn_a.data.authorized = MagicMock()
    conn_a.data.authorized.url = "http://a.url"
    conn_a.data.authorized.name = "A"
    conn_a.data.authorized.models = []
    server.connections = [conn_a]  # type: ignore[assignment]

    # Self → A → D → E (4 levels deep)
    server._nested_topology["http://a.url"] = TopologyUpdateRequest(  # type: ignore[reportPrivateUsage]
        action="join",
        url="http://a.url",
        name="A",
        children={
            "http://d.url": TopologyUpdateRequest(
                action="join",
                url="http://d.url",
                name="D",
                children={
                    "http://e.url": TopologyUpdateRequest(action="join", url="http://e.url", name="E"),
                },
            )
        },
    )

    result = server.get_topology()

    node_d = result[0].children[0].children[0]
    assert node_d.name == "D"
    node_e = node_d.children[0]
    assert node_e.url == "http://e.url"
    assert node_e.name == "E"
    assert node_e.children == []


def test_get_topology_wraps_self_in_ancestor_stubs() -> None:
    server = _make_server(
        infra_url="http://self.url",
        name="self",
        connect_to_mesh_url="http://parent.url",
        parent_ancestors=["http://parent.url", "http://root.url"],
    )
    server.connections = []  # type: ignore[assignment]

    result = server.get_topology()

    # Root stub is oldest ancestor
    assert result[0].url == "http://root.url"
    assert result[0].you_are_here is False
    # Next level is direct parent
    parent_stub = result[0].children[0]
    assert parent_stub.url == "http://parent.url"
    # Current infra is at the bottom of the stub chain
    self_node = parent_stub.children[0]
    assert self_node.url == "http://self.url"
    assert self_node.you_are_here is True


def test_get_topology_skips_unauthorised_connections() -> None:
    server = _make_server(infra_url="http://self.url", name="self", connect_to_mesh_url="")
    conn = MagicMock()
    conn.data.authorized = None
    server.connections = [conn]  # type: ignore[assignment]

    result = server.get_topology()

    assert result[0].children == []


def test_get_topology_no_connections_returns_empty() -> None:
    server = _make_server(infra_url="http://self.url", name="self", connect_to_mesh_url="")
    server.connections = []  # type: ignore[assignment]
    result = server.get_topology()
    assert isinstance(result, list)
    assert result[0].url == "http://self.url"
    assert result[0].children == []


def test_infra_websocket_server_registers_get_children_on_parents() -> None:
    parent_a = MagicMock()
    parent_b = MagicMock()

    parent_infra = MagicMock()
    parent_infra.parents = [parent_a, parent_b]
    parent_infra.ancestors = []

    server = InfraWebsocketServer(MagicMock(), parent_infra, MagicMock())

    assert callable(parent_a.get_children)
    assert callable(parent_b.get_children)
    # get_children returns a snapshot of _nested_topology
    server._nested_topology["http://x.url"] = TopologyUpdateRequest(action="join", url="http://x.url", name="X")  # type: ignore[reportPrivateUsage]
    assert "http://x.url" in cast("dict[str, object]", parent_a.get_children())
    assert "http://x.url" in cast("dict[str, object]", parent_b.get_children())


@pytest.mark.asyncio
async def test_on_init_stores_and_propagates_children_from_init_request() -> None:
    server = _make_server()
    context = InfraWsData(authorized=None)
    child_d = TopologyUpdateRequest(action="join", url="http://d.url", name="D")
    params = _init_params(
        url="http://a.url",
        name="A",
        children={"http://d.url": child_d},
    )

    await server._on_init(params, context)  # type: ignore[reportPrivateUsage]

    stored = server._nested_topology["http://a.url"]  # type: ignore[reportPrivateUsage]
    assert "http://d.url" in stored.children

    cast("MagicMock", server.parent_infra.send_topology_update).assert_called_once_with(
        "join", "http://a.url", "A", [], {"http://d.url": child_d}
    )


@pytest.mark.asyncio
async def test_on_start_sends_children_from_get_children() -> None:
    config = MagicMock()
    config.connect_to_mesh_url = "http://core.url"
    config.connect_to_mesh_key.get_secret_value.return_value = "key"
    config.name = "node"
    config.infra_url = "http://node.url"
    config.infra_api_key.get_secret_value.return_value = "apikey"

    parent = ParentInfra.__new__(ParentInfra)
    parent.config = config
    parent.task_manager = MagicMock()
    parent._ancestors = []  # type: ignore[misc]
    parent.process_loop = True

    endpoint_registry = MagicMock()
    endpoint_registry.list_models.return_value = []
    parent.endpoint_registry = endpoint_registry

    child_x = TopologyUpdateRequest(action="join", url="http://x.url", name="X")
    parent.get_children = lambda: {"http://x.url": child_x}  # type: ignore[misc]

    parent.infra_client = MagicMock(spec=InfraClient)
    parent.infra_client.init = AsyncMock(return_value=InitResponse(ancestors=[]))
    parent.one_time_key = MagicMock()
    parent.one_time_key.key = "otk"

    await parent.on_start()

    sent: InitRequest = parent.infra_client.init.call_args[0][0]
    assert sent.children == {"http://x.url": child_x}
