# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Simplito sp. z o.o.

"""Port Service module."""

import errno
import socket

# Containers are published on loopback only (see DockerService.generate_docker_compose_content).
_PUBLISH_HOST = "127.0.0.1"
# Loopback answers at once; the timeout only bounds a listener that never replies.
_PROBE_TIMEOUT_SECONDS = 0.5


class PortService:
    def __init__(
        self,
    ):
        self.allocated_ports = set[int]()

    def get_free_port(self, start: int = 20_000, end: int = 30_000) -> int:
        """Get next free port."""
        for port in range(start, end + 1):
            if port in self.allocated_ports:
                continue
            if self.is_port_available(port):
                self.allocated_ports.add(port)
                return port
        raise RuntimeError("No free port in range", (start, end))

    def release_port(self, port: int) -> None:
        """Return a previously allocated port to the pool."""
        self.allocated_ports.discard(port)

    def is_port_available(self, port: int) -> bool:
        """Check whether nothing is listening on port at the address containers are published on.

        It connects instead of binding: whether a bind succeeds next to a listener on another address depends on
        the OS (macOS lets a wildcard bind with SO_REUSEADDR through beside a listener on 127.0.0.1), a connection
        does not. Only a refused connection means free; a timeout or any other outcome counts as in use.
        """
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.settimeout(_PROBE_TIMEOUT_SECONDS)
                return sock.connect_ex((_PUBLISH_HOST, port)) == errno.ECONNREFUSED
        except OSError:
            return False
