# host-port-allocation Specification

## Purpose

Defines how the system chooses the host port it publishes a container on, so that a port reported as free is genuinely free to bind on every supported host operating system and installing a container never collides with a port something else is already listening on.

## Requirements

### Requirement: A selected host port has no listener on the publish address
The system SHALL NOT select a host port for a container while any process or container is listening on that port at the loopback address (`127.0.0.1`) the container is published on, on every supported host operating system.

#### Scenario: Port held by another container on the loopback address
- **WHEN** another container is already published on `127.0.0.1:20000` and the system selects a free port for a new container from a range that includes `20000`
- **THEN** `20000` is not selected and the lowest port above it that has no listener is selected instead

#### Scenario: Port held by a process listening on all addresses
- **WHEN** a process is listening on a port on the wildcard address (all interfaces) and the system selects a free port from a range that includes it
- **THEN** that port is not selected

#### Scenario: Port with no listener
- **WHEN** no process or container is listening on a port, even if connections on that port were recently closed
- **THEN** that port is eligible and the lowest such port in the range is selected

### Requirement: A port that cannot be confirmed free is treated as in use
The system SHALL treat a port as unavailable whenever it cannot establish that nothing is listening on it at the publish address, rather than assuming it is free.

#### Scenario: Probing the port gives no definite answer
- **WHEN** an attempt to reach a port on the publish address neither succeeds nor is refused (for example it does not complete in time)
- **THEN** the port is treated as in use and is not selected

### Requirement: A previously recorded port is reused only under the same rule
The system SHALL reuse a host port recorded for an existing container configuration only if that port satisfies the same availability rule as a newly selected port; otherwise it SHALL select a new port.

#### Scenario: Recorded port is now held by a different container
- **WHEN** an existing container configuration records host port `P`, the container is not running, and another container is now listening on `127.0.0.1:P`
- **THEN** the system does not reuse `P` and starts the container on a newly selected free port
