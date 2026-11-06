# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Simplito sp. z o.o.

"""Every route that changes services must be rejected while services.json is not loaded."""

import importlib
import pkgutil

import pytest
from fastapi import APIRouter
from fastapi.routing import APIRoute

import server.api
from server.core.dependencies import require_services_config_loaded

CHANGING_METHODS = {"POST", "PUT", "PATCH", "DELETE"}

# None of these write services.json, so they stay open while it is not loaded.
UNGUARDED_ROUTES = {
    ("POST", "/admin/services/{service_id}/cancel"),
    ("POST", "/admin/services/{service_id}/models/cancel"),
    ("POST", "/admin/services/{service_id}/catalog/refresh"),
    ("POST", "/admin/services/{service_id}/models/{model_id}/oauth/start"),
}


def _service_routes() -> dict[tuple[str, str], APIRoute]:
    """Collect the routes of every router in `server.api`, so a route in a new router file is checked too."""
    routes: dict[tuple[str, str], APIRoute] = {}
    for module_info in pkgutil.iter_modules(server.api.__path__):
        module = importlib.import_module(f"server.api.{module_info.name}")
        for router in vars(module).values():
            if not isinstance(router, APIRouter):
                continue
            for route in router.routes:
                if isinstance(route, APIRoute) and route.path.startswith("/admin/services"):
                    routes.update({(method, route.path): route for method in route.methods & CHANGING_METHODS})
    return routes


SERVICE_ROUTES = _service_routes()


def _is_guarded(route: APIRoute) -> bool:
    return any(dep.dependency is require_services_config_loaded for dep in route.dependencies)


@pytest.mark.parametrize(("method", "path"), sorted(SERVICE_ROUTES))
def test_service_changing_route_requires_services_config_loaded(method: str, path: str) -> None:
    route = SERVICE_ROUTES[(method, path)]

    assert _is_guarded(route) or (method, path) in UNGUARDED_ROUTES, (
        f"{method} {path} must depend on require_services_config_loaded or be listed in UNGUARDED_ROUTES"
    )


@pytest.mark.parametrize(("method", "path"), sorted(UNGUARDED_ROUTES))
def test_unguarded_route_exists_and_is_not_guarded(method: str, path: str) -> None:
    route = SERVICE_ROUTES.get((method, path))

    assert route is not None, f"{method} {path} no longer exists, remove it from UNGUARDED_ROUTES"
    assert not _is_guarded(route), f"{method} {path} is guarded now, remove it from UNGUARDED_ROUTES"
