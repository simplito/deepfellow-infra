# DeepFellow Software Framework.
# Copyright © 2026 Simplito sp. z o.o.
#
# This file is part of the DeepFellow Software Framework (https://deepfellow.ai).
# This software is Licensed under the DeepFellow Free License.
#
# See the License for the specific language governing permissions and
# limitations under the License.

from unittest.mock import AsyncMock, MagicMock

import pytest

from server.websockets.models import AncestorInfo
from server.websockets.parent_infra_group import ParentInfraGroup


def _make_parent(url: str, enabled: bool = True, ancestors: list[str] | None = None, ws: object = None) -> MagicMock:
    p = MagicMock()
    p.parent_url = url
    p.enabled = enabled
    # Mirror real behaviour: InitResponse.ancestors starts with the direct parent's own URL
    p.ancestors = [AncestorInfo(url=u, name="") for u in [url, *(ancestors or [])]]
    p.ws = ws
    return p


# --- enabled ---


def test_group_enabled_when_any_parent_enabled() -> None:
    group = ParentInfraGroup([_make_parent("http://a.url"), _make_parent("http://b.url", enabled=False)])
    assert group.enabled is True


def test_group_disabled_when_no_parents() -> None:
    assert ParentInfraGroup([]).enabled is False


def test_group_disabled_when_all_parents_disabled() -> None:
    group = ParentInfraGroup([_make_parent("http://a.url", enabled=False)])
    assert group.enabled is False


# --- ancestors ---


def test_ancestors_merges_parent_urls_and_their_chains() -> None:
    p1 = _make_parent("http://a.url", ancestors=["http://root.url"])
    p2 = _make_parent("http://b.url", ancestors=["http://root.url", "http://super.url"])
    group = ParentInfraGroup([p1, p2])
    result = group.ancestors
    assert [a.url for a in result] == ["http://a.url", "http://root.url", "http://b.url", "http://super.url"]


def test_ancestors_deduplicates() -> None:
    p1 = _make_parent("http://a.url", ancestors=["http://root.url"])
    p2 = _make_parent("http://a.url", ancestors=["http://root.url"])
    group = ParentInfraGroup([p1, p2])
    assert [a.url for a in group.ancestors] == ["http://a.url", "http://root.url"]


def test_ancestors_skips_disabled_parents() -> None:
    p1 = _make_parent("http://a.url", enabled=False, ancestors=["http://root.url"])
    p2 = _make_parent("http://b.url", ancestors=[])
    group = ParentInfraGroup([p1, p2])
    assert [a.url for a in group.ancestors] == ["http://b.url"]


def test_ancestors_empty_when_no_parents() -> None:
    assert ParentInfraGroup([]).ancestors == []


# --- send_topology_update ---


def test_send_topology_update_broadcasts_to_all_connected() -> None:
    p1 = _make_parent("http://a.url", ws=MagicMock())
    p2 = _make_parent("http://b.url", ws=MagicMock())
    group = ParentInfraGroup([p1, p2])
    group.send_topology_update("join", "http://child.url", "child", [])
    p1.task_manager.add_task_safe.assert_called_once()
    p2.task_manager.add_task_safe.assert_called_once()


def test_send_topology_update_skips_disconnected_parents() -> None:
    p1 = _make_parent("http://a.url", ws=None)
    p2 = _make_parent("http://b.url", ws=MagicMock())
    group = ParentInfraGroup([p1, p2])
    group.send_topology_update("join", "http://child.url", "child", [])
    p1.task_manager.add_task_safe.assert_not_called()
    p2.task_manager.add_task_safe.assert_called_once()


# --- send_models_list / send_usage ---


def test_send_models_list_broadcasts_to_all() -> None:
    p1 = _make_parent("http://a.url")
    p2 = _make_parent("http://b.url")
    group = ParentInfraGroup([p1, p2])
    group.send_models_list()
    p1.send_models_list.assert_called_once()
    p2.send_models_list.assert_called_once()


def test_send_usage_broadcasts_to_all() -> None:
    p1 = _make_parent("http://a.url")
    p2 = _make_parent("http://b.url")
    group = ParentInfraGroup([p1, p2])
    usage = MagicMock()
    group.send_usage(usage)
    p1.send_usage.assert_called_once_with(usage)
    p2.send_usage.assert_called_once_with(usage)


# --- check_subinfra_connection ---


def test_check_subinfra_connection_returns_true_if_any_matches() -> None:
    p1 = _make_parent("http://a.url")
    p1.check_subinfra_connection.return_value = False
    p2 = _make_parent("http://b.url")
    p2.check_subinfra_connection.return_value = True
    group = ParentInfraGroup([p1, p2])
    assert group.check_subinfra_connection(MagicMock()) is True


def test_check_subinfra_connection_returns_false_if_none_match() -> None:
    p1 = _make_parent("http://a.url")
    p1.check_subinfra_connection.return_value = False
    group = ParentInfraGroup([p1])
    assert group.check_subinfra_connection(MagicMock()) is False


# --- endpoint_registry setter ---


def test_endpoint_registry_setter_propagates_to_all_parents() -> None:
    p1 = _make_parent("http://a.url")
    p2 = _make_parent("http://b.url")
    group = ParentInfraGroup([p1, p2])
    registry = MagicMock()
    group.endpoint_registry = registry
    assert p1.endpoint_registry == registry
    assert p2.endpoint_registry == registry


# --- run ---


@pytest.mark.asyncio
async def test_run_empty_group_returns_immediately() -> None:
    group = ParentInfraGroup([])
    await group.run()  # should not raise


@pytest.mark.asyncio
async def test_run_with_parents_runs_all() -> None:
    p1 = _make_parent("http://a.url")
    p2 = _make_parent("http://b.url")
    p1.run = AsyncMock()
    p2.run = AsyncMock()
    group = ParentInfraGroup([p1, p2])
    await group.run()
    p1.run.assert_awaited_once()
    p2.run.assert_awaited_once()


# --- endpoint_registry getter ---


def test_endpoint_registry_getter_returns_none_when_no_parents() -> None:
    group = ParentInfraGroup([])
    assert group.endpoint_registry is None
