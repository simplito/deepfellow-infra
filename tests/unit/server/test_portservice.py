# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Simplito sp. z o.o.

import errno
import socket
from collections.abc import Iterator
from unittest.mock import MagicMock, call, patch

import pytest

from server.portservice import PortService


@pytest.fixture
def port_service() -> PortService:
    return PortService()


# Real listeners, not mocks: a bind-based probe reports the 127.0.0.1 port as free on BSD-derived stacks (macOS).
@pytest.fixture(params=["127.0.0.1", "0.0.0.0"], ids=["loopback", "wildcard"])
def listener_port(request: pytest.FixtureRequest) -> Iterator[int]:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind((request.param, 0))
        listener.listen()
        yield listener.getsockname()[1]


def test_allocated_ports_empty_on_init(port_service: PortService) -> None:
    assert len(port_service.allocated_ports) == 0


@pytest.mark.parametrize(
    ("connect_ex_result", "expected"),
    [
        (errno.ECONNREFUSED, True),
        (0, False),
        (errno.EAGAIN, False),
        (errno.ETIMEDOUT, False),
        (errno.ENETUNREACH, False),
    ],
)
@patch("server.portservice.socket.socket")
def test_is_port_available_is_true_only_when_connection_is_refused(
    mock_socket: MagicMock, port_service: PortService, connect_ex_result: int, expected: bool
) -> None:
    mock_socket.return_value.__enter__.return_value.connect_ex.return_value = connect_ex_result

    result = port_service.is_port_available(25000)

    assert result is expected


@pytest.mark.parametrize("error", [OSError("network is unreachable"), TimeoutError()])
@patch("server.portservice.socket.socket")
def test_is_port_available_returns_false_when_connect_raises(mock_socket: MagicMock, port_service: PortService, error: OSError) -> None:
    mock_socket.return_value.__enter__.return_value.connect_ex.side_effect = error

    result = port_service.is_port_available(25000)

    assert result is False


@patch("server.portservice.socket.socket")
def test_is_port_available_connects_to_loopback_with_timeout(mock_socket: MagicMock, port_service: PortService) -> None:
    _mock_sock = mock_socket.return_value.__enter__.return_value
    _mock_sock.connect_ex.return_value = errno.ECONNREFUSED

    port_service.is_port_available(25000)

    assert mock_socket.call_args == call(socket.AF_INET, socket.SOCK_STREAM)
    assert _mock_sock.settimeout.call_count == 1
    assert _mock_sock.settimeout.call_args == call(0.5)
    assert _mock_sock.connect_ex.call_count == 1
    assert _mock_sock.connect_ex.call_args == call(("127.0.0.1", 25000))


@patch("server.portservice.socket.socket")
def test_is_port_available_closes_socket_after_probe(mock_socket: MagicMock, port_service: PortService) -> None:
    mock_socket.return_value.__enter__.return_value.connect_ex.return_value = errno.ECONNREFUSED

    port_service.is_port_available(25000)

    assert mock_socket.return_value.__exit__.call_count == 1


@patch("server.portservice.socket.socket")
def test_is_port_available_closes_socket_when_connect_raises(mock_socket: MagicMock, port_service: PortService) -> None:
    mock_socket.return_value.__enter__.return_value.connect_ex.side_effect = OSError("boom")

    port_service.is_port_available(25000)

    assert mock_socket.return_value.__exit__.call_count == 1


def test_is_port_available_is_false_when_a_listener_holds_the_port(port_service: PortService, listener_port: int) -> None:
    result = port_service.is_port_available(listener_port)

    assert result is False


def test_is_port_available_is_true_for_a_port_that_was_bound_and_closed(port_service: PortService) -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]

    result = port_service.is_port_available(port)

    assert result is True


def test_get_free_port_raises_when_a_listener_holds_the_only_port(port_service: PortService, listener_port: int) -> None:
    with pytest.raises(RuntimeError):
        port_service.get_free_port(listener_port, listener_port)

    assert listener_port not in port_service.allocated_ports


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


def test_release_port_removes_allocated_port(port_service: PortService) -> None:
    port_service.allocated_ports.add(20000)

    port_service.release_port(20000)

    assert 20000 not in port_service.allocated_ports


def test_release_port_is_noop_for_unallocated_port(port_service: PortService) -> None:
    port_service.release_port(20000)

    assert len(port_service.allocated_ports) == 0


def test_release_port_allows_reallocation(port_service: PortService) -> None:
    with patch.object(port_service, "is_port_available", return_value=True):
        port = port_service.get_free_port(20000, 30000)
        port_service.release_port(port)
        reallocated = port_service.get_free_port(20000, 30000)

    assert reallocated == port
