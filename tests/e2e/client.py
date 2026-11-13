# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Simplito sp. z o.o.

"""HTTP client for the target instance.

Response models are kept local and minimal on purpose: the suite is black-box, so it must not import `server.models`
and silently follow a breaking API change.
"""

from typing import Any

import httpx
from pydantic import BaseModel

from tests.e2e.config import E2ESettings


class E2EError(Exception):
    """A request to the target instance returned a non-2xx status."""

    def __init__(self, msg: str, status_code: int | None = None) -> None:
        super().__init__(msg)
        self.status_code = status_code


def is_transient(error: Exception) -> bool:
    """A connection failure or a gateway error - worth one more try, unlike an answer that is simply wrong."""
    if isinstance(error, httpx.TransportError):
        return True
    return isinstance(error, E2EError) and error.status_code in (502, 503, 504)


class InfoOut(BaseModel):
    version: str


class ServiceField(BaseModel):
    name: str
    values: list[str | dict[str, Any]] | None = None


class ServiceSpec(BaseModel):
    fields: list[ServiceField]


class ServiceOut(BaseModel):
    id: str
    instance: str
    installed: bool | dict[str, Any]
    spec: ServiceSpec

    @property
    def is_installed(self) -> bool:
        return self.installed is not False

    @property
    def is_loading(self) -> bool:
        """Still installing, or still loading after an Infra start: `installed` is then a progress, not the options."""
        return isinstance(self.installed, dict) and "stage" in self.installed

    def hardware_options(self) -> set[str]:
        """Values the `hardware` field accepts on this instance ("CPU", "GPU", ...); empty if the service has none."""
        for field in self.spec.fields:
            if field.name == "hardware":
                return {v if isinstance(v, str) else str(v.get("value")) for v in field.values or []}
        return set()


class ListServicesOut(BaseModel):
    list: list[ServiceOut]


def _check(response: httpx.Response) -> httpx.Response:
    if not response.is_success:
        msg = f"{response.request.method} {response.request.url.path} -> {response.status_code}: {response.text[:2000]}"
        raise E2EError(msg, response.status_code)
    return response


class E2EClient:
    """Thin wrapper over httpx: one auth header per route family, errors carry the response body."""

    def __init__(self, settings: E2ESettings) -> None:
        self.settings = settings
        self._http = httpx.Client(base_url=settings.base_url.rstrip("/"), timeout=settings.inference_timeout)

    def close(self) -> None:
        self._http.close()

    def _admin_headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.settings.admin_api_key.get_secret_value()}"}

    def _api_headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.settings.api_key.get_secret_value()}"}

    def health(self) -> bool:
        try:
            return self._http.get("/health", timeout=5).is_success
        except httpx.TransportError:
            return False

    def info(self) -> InfoOut:
        return InfoOut.model_validate(_check(self._http.get("/info", headers=self._api_headers())).json())

    def list_services(self) -> list[ServiceOut]:
        response = _check(self._http.get("/admin/services", headers=self._admin_headers()))
        return ListServicesOut.model_validate(response.json()).list

    def get_service(self, service_id: str) -> ServiceOut:
        for service in self.list_services():
            if service.id == service_id and service.instance == "default":
                return service
        msg = f"service {service_id!r} is not offered by the target instance"
        raise E2EError(msg)

    def install_service(self, service_id: str, spec: dict[str, Any]) -> None:
        body = {"stream": False, "spec": spec}
        _check(
            self._http.post(
                f"/admin/services/{service_id}", json=body, headers=self._admin_headers(), timeout=self.settings.install_timeout
            )
        )

    def uninstall_service(self, service_id: str) -> None:
        body = {"purge": self.settings.purge}
        _check(self._http.request("DELETE", f"/admin/services/{service_id}", json=body, headers=self._admin_headers()))

    def install_model(self, service_id: str, model_id: str, spec: dict[str, Any] | None) -> None:
        body = {"stream": False, "spec": spec}
        _check(
            self._http.post(
                f"/admin/services/{service_id}/models/_",
                params={"model_id": model_id},
                json=body,
                headers=self._admin_headers(),
                timeout=self.settings.install_timeout,
            )
        )

    def model_installed(self, service_id: str, model_id: str) -> bool:
        return self._model_installed_field(service_id, model_id) not in (None, False)

    def model_loading(self, service_id: str, model_id: str) -> bool:
        """Still installing, or still loading after an Infra start: `installed` is then a progress, not the options."""
        installed = self._model_installed_field(service_id, model_id)
        return isinstance(installed, dict) and "stage" in installed

    def _model_installed_field(self, service_id: str, model_id: str) -> object:
        response = _check(
            self._http.get(f"/admin/services/{service_id}/models/_", params={"model_id": model_id}, headers=self._admin_headers())
        )
        return response.json().get("installed")

    def uninstall_model(self, service_id: str, model_id: str) -> None:
        body = {"purge": self.settings.purge}
        _check(
            self._http.request(
                "DELETE", f"/admin/services/{service_id}/models/_", params={"model_id": model_id}, json=body, headers=self._admin_headers()
            )
        )

    def gpu_used_vram_gb(self) -> float | None:
        stats = _check(self._http.get("/admin/settings/hardware/gpu-stats", headers=self._admin_headers())).json()
        return None if stats is None else float(stats["used_vram_gb"])

    def docker_logs(self, service_id: str, model_id: str | None = None) -> str:
        params = {"model_id": model_id} if model_id else {}
        response = self._http.get(f"/admin/services/{service_id}/docker/logs", params=params, headers=self._admin_headers())
        return str(_check(response).json()["logs"])

    def served_model(self, model_id: str) -> dict[str, Any] | None:
        """The model as `/v1/models` lists it, with its props; None when Infra does not serve it."""
        response = self._http.get(f"/v1/models/{model_id}", params={"additional_data": "true"}, headers=self._api_headers())
        if response.status_code == 404:
            return None
        return _check(response).json()

    def post_api(self, path: str, **kwargs: Any) -> httpx.Response:
        return _check(self._http.post(path, headers=self._api_headers(), **kwargs))

    def get_api(self, path: str, **kwargs: Any) -> httpx.Response:
        return _check(self._http.get(path, headers=self._api_headers(), **kwargs))

    def delete_api(self, path: str, **kwargs: Any) -> httpx.Response:
        return _check(self._http.delete(path, headers=self._api_headers(), **kwargs))
