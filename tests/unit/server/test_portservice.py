# DeepFellow Software Framework.
# Copyright © 2026 Simplito sp. z o.o.
#
# This file is part of the DeepFellow Software Framework (https://deepfellow.ai).
# This software is Licensed under the DeepFellow Free License.
#
# See the License for the specific language governing permissions and
# limitations under the License.

from unittest.mock import MagicMock, call, patch

import pytest

from server.portservice import PortService


@pytest.fixture
def port_service() -> PortService:
    return PortService()


def test_allocated_ports_empty_on_init(port_service: PortService) -> None:
    assert len(port_service.allocated_ports) == 0


def test_is_port_available_returns_true_when_bind_succeeds(port_service: PortService) -> None:
    mock_sock = MagicMock()

    with patch("socket.socket", return_value=mock_sock):
        result = port_service.is_port_available(25000)

    assert result is True
    assert mock_sock.bind.call_count == 1
    assert mock_sock.bind.call_args == call(("", 25000))
    assert mock_sock.close.call_count == 1


def test_is_port_available_returns_false_when_bind_raises_oserror(port_service: PortService) -> None:
    mock_sock = MagicMock()
    mock_sock.bind.side_effect = OSError("address in use")

    with patch("socket.socket", return_value=mock_sock):
        result = port_service.is_port_available(25000)

    assert result is False


def test_get_free_port_returns_first_available_port(port_service: PortService) -> None:
    with patch.object(port_service, "is_port_available", return_value=True):
        port = port_service.get_free_port(20000, 30000)

    assert port == 20000


def test_get_free_port_skips_allocated_ports(port_service: PortService) -> None:
    port_service.allocated_ports.add(20000)
    port_service.allocated_ports.add(20001)

    with patch.object(port_service, "is_port_available", return_value=True):
        port = port_service.get_free_port(20000, 30000)

    assert port == 20002


def test_get_free_port_skips_unavailable_ports(port_service: PortService) -> None:
    def available(port: int) -> bool:
        return port >= 20003

    with patch.object(port_service, "is_port_available", side_effect=available):
        port = port_service.get_free_port(20000, 30000)

    assert port == 20003


def test_get_free_port_adds_port_to_allocated_after_returning(port_service: PortService) -> None:
    with patch.object(port_service, "is_port_available", return_value=True):
        port = port_service.get_free_port(20000, 30000)

    assert port in port_service.allocated_ports


def test_get_free_port_raises_when_no_free_port_in_range(port_service: PortService) -> None:
    with patch.object(port_service, "is_port_available", return_value=False), pytest.raises(RuntimeError):
        port_service.get_free_port(20000, 20002)


def test_get_free_port_custom_range(port_service: PortService) -> None:
    with patch.object(port_service, "is_port_available", return_value=True):
        port = port_service.get_free_port(5000, 6000)

    assert port == 5000


def test_get_free_port_range_end_is_inclusive(port_service: PortService) -> None:
    def available(port: int) -> bool:
        return port == 5001

    with patch.object(port_service, "is_port_available", side_effect=available):
        port = port_service.get_free_port(5000, 5001)

    assert port == 5001
