# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Simplito sp. z o.o.

"""Install/teardown of a scenario's service and model, with a session ledger as a safety net."""

from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from tests.e2e.client import E2EClient, is_transient
from tests.e2e.scenarios import Scenario


class TeardownError(Exception):
    """Uninstalling after a scenario failed; the session sweep retries."""


class Ledger:
    """What the suite may have installed and not yet removed.

    An entry is recorded *before* its install request, so a failed or timed-out install that left a container behind is
    still cleaned up. Removing something that turns out not to be installed counts as done.
    """

    def __init__(self, client: E2EClient, preexisting: set[str] | None = None) -> None:
        self.client = client
        # Services installed before the session (only with DF_E2E_ALLOW_DIRTY); scenarios never touch them.
        self.preexisting = preexisting or set()
        self.services: list[str] = []
        self.models: list[tuple[str, str]] = []

    def remove_model(self, service_id: str, model_id: str) -> None:
        try:
            self.client.uninstall_model(service_id, model_id)
        except Exception:
            if self._service_installed(service_id) and self._model_installed(service_id, model_id):
                raise
        self.models.remove((service_id, model_id))

    def remove_service(self, service_id: str) -> None:
        try:
            self.client.uninstall_service(service_id)
        except Exception:
            if self._service_installed(service_id):
                raise
        self.services.remove(service_id)
        # Uninstalling a service takes its models with it.
        self.models = [(s, m) for s, m in self.models if s != service_id]

    def sweep(self) -> list[str]:
        """Remove everything still recorded; return the errors hit along the way."""
        errors: list[str] = []
        for service_id, model_id in list(self.models):
            try:
                self.remove_model(service_id, model_id)
            except Exception as e:
                errors.append(f"model {service_id}/{model_id}: {e}")
        for service_id in list(self.services):
            try:
                self.remove_service(service_id)
            except Exception as e:
                errors.append(f"service {service_id}: {e}")
        return errors

    def _service_installed(self, service_id: str) -> bool:
        return self.client.get_service(service_id).is_installed

    def _model_installed(self, service_id: str, model_id: str) -> bool:
        return self.client.model_installed(service_id, model_id)


class GpuRequiredError(AssertionError):
    """A `gpu` scenario found no GPU on the target although DF_E2E_REQUIRE_GPU is set."""


def gpu_option(client: E2EClient, service_id: str) -> str | None:
    """The `hardware` value that puts the service on GPU on this target, or None if it offers none."""
    options = client.get_service(service_id).hardware_options()
    for preferred in ("GPU", "GPUs"):
        if preferred in options:
            return preferred
    named = sorted(o for o in options if o.startswith("GPU | "))
    return named[0] if named else None


def skip_reason(client: E2EClient, ledger: Ledger, scenario: Scenario) -> str | None:
    """Why the target cannot run this scenario, or None if it can.

    Raises:
        GpuRequiredError: the scenario needs a GPU, the target offers none, and DF_E2E_REQUIRE_GPU is set.
    """
    if missing := scenario.missing_settings(client.settings):
        return "missing " + ", ".join(missing)
    if scenario.service_id in ledger.preexisting:
        return f"{scenario.service_id} was installed before the session; the suite does not touch it"
    if "gpu" in scenario.marks and gpu_option(client, scenario.service_id) is None:
        reason = f"no GPU available for {scenario.service_id} on the target instance"
        if client.settings.require_gpu:
            raise GpuRequiredError(reason + " (DF_E2E_REQUIRE_GPU is set)")
        return reason
    return None


def _service_spec(client: E2EClient, scenario: Scenario) -> dict[str, Any]:
    spec = scenario.resolved_service_spec(client.settings)
    if "gpu" in scenario.marks and (option := gpu_option(client, scenario.service_id)):
        spec["hardware"] = option
    return spec


def _teardown(ledger: Ledger, scenarios: Sequence[Scenario]) -> list[str]:
    errors: list[str] = []
    for scenario in scenarios:
        if (scenario.service_id, scenario.model_id) in ledger.models:
            try:
                ledger.remove_model(scenario.service_id, scenario.model_id)
            except Exception as e:
                errors.append(f"uninstall model {scenario.model_id}: {e}")
    for service_id in _services(scenarios):
        if service_id in ledger.services:
            try:
                ledger.remove_service(service_id)
            except Exception as e:
                errors.append(f"uninstall service {service_id}: {e}")
    return errors


def _services(scenarios: Sequence[Scenario]) -> dict[str, Scenario]:
    """Each service once, installed with the spec of its first scenario."""
    services: dict[str, Scenario] = {}
    for scenario in scenarios:
        services.setdefault(scenario.service_id, scenario)
    return services


def _install(install: Callable[[], None]) -> None:
    """Install, once more after a connection or gateway error: a registry or Hugging Face outage is not an Infra bug."""
    try:
        install()
    except Exception as e:
        if not is_transient(e):
            raise
        install()


def save_logs(client: E2EClient, scenarios: Sequence[Scenario]) -> list[str]:
    """Save the service containers' logs before teardown removes them; return the files written."""
    log_dir = Path(client.settings.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    targets = [(service_id, None) for service_id in _services(scenarios)] + [(s.service_id, s.model_id) for s in scenarios]
    for service_id, model_id in targets:
        try:
            logs = client.docker_logs(service_id, model_id)
        except Exception:
            continue
        path = log_dir / ("-".join(filter(None, [service_id, model_id])).replace("/", "_").replace(" ", "_") + ".log")
        path.write_text(logs)
        written.append(str(path))
    return written


@contextmanager
def installed(client: E2EClient, ledger: Ledger, *scenarios: Scenario) -> Iterator[None]:
    """Install the scenarios' services, then their models, and uninstall all of it on exit - also when an install or
    the body fails.

    A failure is noted with the phase it happened in, and the service logs are saved first when DF_E2E_LOG_DIR is set.
    A teardown failure never hides the scenarios' own failure: it is attached to that exception as a note, and only
    raised as `TeardownError` when the body itself passed.
    """
    phase = ""
    try:
        for service_id, scenario in _services(scenarios).items():
            phase = f"install service {service_id}"
            ledger.services.append(service_id)
            _install(lambda service_id=service_id, scenario=scenario: client.install_service(service_id, _service_spec(client, scenario)))
        for scenario in scenarios:
            phase = f"install model {scenario.model_id}"
            ledger.models.append((scenario.service_id, scenario.model_id))
            _install(lambda scenario=scenario: client.install_model(scenario.service_id, scenario.model_id, scenario.model_spec))
        phase = "inference"
        yield
    except BaseException as e:
        e.add_note(f"phase: {phase}")
        if client.settings.log_dir:
            for path in save_logs(client, scenarios):
                e.add_note(f"service log: {path}")
        for error in _teardown(ledger, scenarios):
            e.add_note(f"teardown: {error}")
        raise
    if errors := _teardown(ledger, scenarios):
        raise TeardownError(", ".join(s.id for s in scenarios) + ": " + "; ".join(errors))
