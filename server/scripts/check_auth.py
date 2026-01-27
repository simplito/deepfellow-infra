#!/usr/bin/env python3

# DeepFellow Software Framework.
# Copyright © 2026 Simplito sp. z o.o.
#
# This file is part of the DeepFellow Software Framework (https://deepfellow.ai).
# This software is Licensed under the DeepFellow Free License.
#
# See the License for the specific language governing permissions and
# limitations under the License.

"""Scan FastAPI app for missing authorization."""

import ast
import fnmatch
import importlib.util
import inspect
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated

import typer
from fastapi.routing import APIRoute

app = typer.Typer(help="FastAPI authorization audit tool")

AUTH_DEPENDENCIES = {"auth_admin", "auth_server", "auth_metrics"}

PUBLIC_WHITELIST: dict[str, set[str] | None] = {
    "/health": {"GET"},
    "/admin/mesh/check": {"POST"},
}


@dataclass
class AuditResult:
    """Result of authorization audit."""

    total: int = 0
    issues: list[dict[str, str | int | set[str]]] = field(default_factory=list)
    all_endpoints: list[dict[str, str | int | set[str]]] = field(default_factory=list)

    @property
    def secure(self) -> int:
        """Return number of secure endpoints."""
        return self.total - len(self.issues)

    @property
    def is_ok(self) -> bool:
        """Return True if all endpoints are secure."""
        return len(self.issues) == 0


def _extract_depends_name(depends_str: str) -> str | None:
    """Extract function name from Depends call."""
    match = re.search(r"Depends\((\w+)", depends_str)
    return match.group(1) if match else None


def _path_pattern_matches(path: str, pattern: str) -> bool:
    """Check if path matches pattern with wildcards."""
    normalized_pattern = re.sub(r"\{[^}]+\}", "*", pattern)
    normalized_path = re.sub(r"\{[^}]+\}", "*", path)

    if "*" in normalized_pattern:
        return fnmatch.fnmatch(normalized_path, normalized_pattern)
    return normalized_path == normalized_pattern


def _methods_allowed(methods: set[str] | None, allowed: set[str] | None) -> bool:
    """Check if methods are allowed."""
    if allowed is None or methods is None:
        return True
    return bool(methods & allowed)


def _path_matches_whitelist(path: str, methods: set[str] | None = None) -> bool:
    """Check if path and method combination is whitelisted."""
    for pattern, allowed_methods in PUBLIC_WHITELIST.items():
        if not _path_pattern_matches(path, pattern):
            continue
        if _methods_allowed(methods, allowed_methods):
            return True
    return False


def _extract_deps_from_defaults(node: ast.FunctionDef | ast.AsyncFunctionDef) -> set[str]:
    """Extract dependency names from function defaults."""
    deps = set()
    for default in node.args.defaults + node.args.kw_defaults:
        if default:
            default_str = ast.unparse(default)
            if "Depends(" in default_str:
                dep_name = _extract_depends_name(default_str)
                if dep_name:
                    deps.add(dep_name)
    return deps


def _extract_deps_from_annotations(node: ast.FunctionDef | ast.AsyncFunctionDef) -> set[str]:
    """Extract dependency names from function annotations."""
    deps = set()
    for arg in node.args.args + node.args.kwonlyargs:
        if arg.annotation:
            ann_str = ast.unparse(arg.annotation)
            if "Depends(" in ann_str:
                dep_name = _extract_depends_name(ann_str)
                if dep_name:
                    deps.add(dep_name)
    return deps


def _parse_file_for_deps(py_file: Path, debug: bool) -> dict[str, set[str]]:
    """Parse single file for dependencies."""
    graph: dict[str, set[str]] = {}
    try:
        content = py_file.read_text()
        tree = ast.parse(content)
    except (SyntaxError, UnicodeDecodeError):
        return graph

    for node in ast.walk(tree):
        if not isinstance(node, (ast.AsyncFunctionDef, ast.FunctionDef)):
            continue

        deps = _extract_deps_from_defaults(node)
        deps |= _extract_deps_from_annotations(node)

        if deps:
            if debug:
                typer.echo(f"DEBUG graph: {node.name} -> {deps} ({py_file.name})")
            graph[node.name] = deps

    return graph


def _build_dependency_graph(directory: Path, debug: bool = False) -> dict[str, set[str]]:
    """Build graph mapping functions to their dependencies."""
    graph: dict[str, set[str]] = {}
    for py_file in directory.rglob("*.py"):
        file_graph = _parse_file_for_deps(py_file, debug)
        graph.update(file_graph)
    return graph


def _check_auth_recursive(
    func_name: str,
    graph: dict[str, set[str]],
    visited: set[str],
    debug: bool,
    indent: int,
) -> bool:
    """Check recursively if function has auth in dependency chain."""
    prefix = "  " * indent

    if func_name in visited:
        if debug:
            typer.echo(f"{prefix}DEBUG chain: {func_name} (already visited)")
        return False

    if func_name in AUTH_DEPENDENCIES:
        if debug:
            typer.echo(f"{prefix}DEBUG chain: {func_name} ✓ IS AUTH")
        return True

    visited.add(func_name)

    if func_name not in graph:
        if debug:
            typer.echo(f"{prefix}DEBUG chain: {func_name} (not in graph)")
        return False

    if debug:
        typer.echo(f"{prefix}DEBUG chain: {func_name} -> {graph[func_name]}")

    return any(_check_auth_recursive(dep, graph, visited, debug, indent + 1) for dep in graph[func_name])


def _has_auth_in_chain(
    func_name: str,
    graph: dict[str, set[str]],
    debug: bool = False,
) -> bool:
    """Check if function has auth dependency in its chain."""
    return _check_auth_recursive(func_name, graph, set(), debug, 0)


def _extract_prefix_kwarg(call_node: ast.Call) -> str:
    """Extract prefix keyword argument from APIRouter call."""
    for kw in call_node.keywords:
        if kw.arg == "prefix" and isinstance(kw.value, ast.Constant):
            value = kw.value.value
            if isinstance(value, str):
                return value
    return ""


def _extract_router_prefixes(content: str) -> dict[str, str]:
    """Extract router variable to prefix mapping."""
    prefixes: dict[str, str] = {}
    tree = ast.parse(content)

    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        if not node.targets:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name):
            continue
        if not isinstance(node.value, ast.Call):
            continue

        call_str = ast.unparse(node.value)
        if "APIRouter" in call_str:
            prefixes[target.id] = _extract_prefix_kwarg(node.value)

    return prefixes


def _check_func_deps_deep(  # noqa: C901
    func: object,
    visited: set[str],
    debug: bool = False,
    indent: int = 0,
) -> bool:
    """Recursively check function's dependencies for auth."""
    from typing import get_args, get_origin, get_type_hints

    prefix = "    " * indent

    try:
        sig = inspect.signature(func)  # type: ignore[arg-type]
    except (ValueError, TypeError):
        return False

    # Check defaults: arg = Depends(...)
    for param in sig.parameters.values():
        if param.default is not inspect.Parameter.empty:
            default = param.default
            if hasattr(default, "dependency"):
                dep_func = default.dependency
                dep_name = getattr(dep_func, "__name__", str(dep_func))

                if debug:
                    typer.echo(f"{prefix}  -> nested dep (default): {dep_name}")

                if dep_name in AUTH_DEPENDENCIES:
                    if debug:
                        typer.echo(f"{prefix}     ✓ IS AUTH")
                    return True

                if dep_name in visited:
                    if debug:
                        typer.echo(f"{prefix}     (already visited)")
                    continue
                visited.add(dep_name)

                if _check_func_deps_deep(dep_func, visited, debug, indent + 1):
                    return True

    # Check annotations: arg: Annotated[..., Depends(...)]
    try:
        hints = get_type_hints(func, include_extras=True)  # type: ignore[arg-type]
    except Exception:
        return False

    for hint in hints.values():
        if get_origin(hint) is not Annotated:
            continue

        for arg in get_args(hint):
            if hasattr(arg, "dependency"):
                dep_func = arg.dependency
                dep_name = getattr(dep_func, "__name__", str(dep_func))

                if debug:
                    typer.echo(f"{prefix}  -> nested dep (annotated): {dep_name}")

                if dep_name in AUTH_DEPENDENCIES:
                    if debug:
                        typer.echo(f"{prefix}     ✓ IS AUTH")
                    return True

                if dep_name in visited:
                    if debug:
                        typer.echo(f"{prefix}     (already visited)")
                    continue
                visited.add(dep_name)

                if _check_func_deps_deep(dep_func, visited, debug, indent + 1):
                    return True

    return False


def _route_has_auth(  # noqa: C901
    route: APIRoute,
    debug: bool = False,
) -> bool:
    """Check if route has auth dependency (deep check)."""
    visited: set[str] = set()

    dependencies = route.dependencies or []

    for dep in dependencies:
        dep_func = dep.dependency
        dep_name = getattr(dep_func, "__name__", str(dep_func))

        if debug:
            typer.echo(f"    route dep: {dep_name}")

        if dep_name in AUTH_DEPENDENCIES:
            if debug:
                typer.echo("      ✓ IS AUTH")
            return True

        if dep_name in visited:
            continue
        visited.add(dep_name)

        if _check_func_deps_deep(dep_func, visited, debug, 1):
            return True

    if route.dependant:
        for dep in route.dependant.dependencies:
            dep_func = dep.call
            dep_name = getattr(dep_func, "__name__", str(dep_func))

            if debug:
                typer.echo(f"    dependant dep: {dep_name}")

            if dep_name in AUTH_DEPENDENCIES:
                if debug:
                    typer.echo("      ✓ IS AUTH")
                return True

            if dep_name in visited:
                continue
            visited.add(dep_name)

            if _check_func_deps_deep(dep_func, visited, debug, 1):
                return True

    return False


def _check_runtime(app_path: Path, debug: bool = False) -> AuditResult:
    """Analyze running application for missing auth."""
    spec = importlib.util.spec_from_file_location("app", app_path)
    if spec is None or spec.loader is None:
        msg = f"Cannot load module from {app_path}"
        raise typer.BadParameter(msg)

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    fastapi_app = getattr(module, "app", None)
    if not fastapi_app:
        msg = f"No 'app' found in {app_path}"
        raise typer.BadParameter(msg)

    result = AuditResult()

    for route in fastapi_app.routes:
        if not isinstance(route, APIRoute):
            continue

        if debug:
            typer.echo(f"\nDEBUG route: {route.methods} {route.path}")

        endpoint_info: dict[str, str | int | set[str]] = {
            "path": route.path,
            "methods": route.methods,
            "name": route.name,
            "endpoint": f"{route.endpoint.__module__}.{route.endpoint.__name__}",
        }

        if _path_matches_whitelist(route.path, route.methods):
            if debug:
                typer.echo("  -> WHITELISTED")
            endpoint_info["status"] = "whitelisted"
            result.all_endpoints.append(endpoint_info)
            continue

        result.total += 1
        has_auth = _route_has_auth(route, debug)

        if debug:
            typer.echo(f"  -> {'HAS AUTH' if has_auth else 'NO AUTH'}")

        endpoint_info["status"] = "secure" if has_auth else "NOT SECURE"
        result.all_endpoints.append(endpoint_info)

        if not has_auth:
            result.issues.append(endpoint_info)

    return result


def _extract_route_info(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
) -> tuple[str | None, str | None, str | None]:
    """Extract decorator, router name and method from function."""
    for dec in node.decorator_list:
        dec_str = ast.unparse(dec)
        for m in [".get(", ".post(", ".put(", ".delete(", ".patch("]:
            if m in dec_str:
                router_name = dec_str.split(".")[0]
                method = m[1:-1].upper()
                return dec_str, router_name, method
    return None, None, None


def _extract_path_from_decorator(decorator: str) -> str:
    """Extract path from route decorator."""
    try:
        path_match = decorator.split("(")[1].split(")")[0].split(",")[0]
        return path_match.strip("'\"")
    except IndexError:
        return ""


def _check_auth_in_decorator(decorator: str) -> bool:
    """Check if auth is in decorator."""
    return any(auth in decorator for auth in AUTH_DEPENDENCIES)


def _check_auth_in_annotations(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
    graph: dict[str, set[str]],
    debug: bool,
) -> bool:
    """Check if auth is in function annotations."""
    for arg in node.args.args + node.args.kwonlyargs:
        if not arg.annotation:
            continue

        ann_str = ast.unparse(arg.annotation)

        if any(auth in ann_str for auth in AUTH_DEPENDENCIES):
            if debug:
                typer.echo(f"  AUTH direct in annotation: {ann_str[:60]}")
            return True

        if "Depends(" in ann_str:
            dep_name = _extract_depends_name(ann_str)
            if debug:
                typer.echo(f"  Checking dep: {dep_name} (from {ann_str[:60]})")
            if dep_name and _has_auth_in_chain(dep_name, graph, debug):
                return True

    return False


def _check_auth_in_defaults(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
    graph: dict[str, set[str]],
    debug: bool,
) -> bool:
    """Check if auth is in function defaults."""
    for default in node.args.defaults + node.args.kw_defaults:
        if not default:
            continue

        default_str = ast.unparse(default)

        if any(auth in default_str for auth in AUTH_DEPENDENCIES):
            if debug:
                typer.echo(f"  AUTH in default: {default_str[:60]}")
            return True

        dep_name = _extract_depends_name(default_str)
        if dep_name and _has_auth_in_chain(dep_name, graph, debug):
            return True

    return False


def _process_route_node(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
    prefixes: dict[str, str],
    graph: dict[str, set[str]],
    debug: bool,
    py_file: Path,
) -> tuple[dict[str, str | int | set[str]] | None, bool]:
    """Process route node. Return (endpoint_info, has_auth). None if not a route."""
    decorator, router_name, method = _extract_route_info(node)
    if not decorator:
        return None, False

    path = _extract_path_from_decorator(decorator)
    if router_name and router_name in prefixes:
        path = prefixes[router_name] + path

    methods = {method} if method else None

    endpoint_info: dict[str, str | int | set[str]] = {
        "line": node.lineno,
        "function": node.name,
        "method": method or "?",
        "path": path or "(empty)",
        "file": str(py_file),
    }

    if _path_matches_whitelist(path, methods):
        if debug:
            typer.echo(f"DEBUG: {node.name} whitelisted ({method} {path})")
        endpoint_info["status"] = "whitelisted"
        return endpoint_info, True  # whitelisted counts as "has auth"

    if debug:
        typer.echo(f"\nDEBUG checking: {node.name} ({method} {path})")

    has_auth = (
        _check_auth_in_decorator(decorator) or _check_auth_in_annotations(node, graph, debug) or _check_auth_in_defaults(node, graph, debug)
    )

    if debug:
        typer.echo(f"  Result: {'✓ HAS AUTH' if has_auth else '✗ NO AUTH'}")

    endpoint_info["status"] = "secure" if has_auth else "NOT SECURE"
    return endpoint_info, has_auth


def _process_file(
    py_file: Path,
    graph: dict[str, set[str]],
    debug: bool,
) -> tuple[int, list[dict[str, str | int | set[str]]], list[dict[str, str | int | set[str]]]]:
    """Process single file for auth issues. Return (total, issues, all_endpoints)."""
    total = 0
    issues: list[dict[str, str | int | set[str]]] = []
    all_endpoints: list[dict[str, str | int | set[str]]] = []

    try:
        content = py_file.read_text()
    except UnicodeDecodeError:
        return 0, [], []

    if "@app." not in content and "@router." not in content:
        return 0, [], []

    try:
        tree = ast.parse(content)
    except SyntaxError:
        return 0, [], []

    prefixes = _extract_router_prefixes(content)

    for node in ast.walk(tree):
        if not isinstance(node, (ast.AsyncFunctionDef, ast.FunctionDef)):
            continue

        endpoint_info, has_auth = _process_route_node(node, prefixes, graph, debug, py_file)

        if endpoint_info is None:
            continue

        # Whitelisted endpoints are tracked but not counted in total
        if endpoint_info.get("status") == "whitelisted":
            all_endpoints.append(endpoint_info)
            continue

        total += 1
        all_endpoints.append(endpoint_info)

        if not has_auth:
            issues.append(endpoint_info)

    return total, issues, all_endpoints


def _check_static(directory: Path, debug: bool = False) -> AuditResult:
    """Analyze code statically for missing auth."""
    graph = _build_dependency_graph(directory, debug)

    if debug:
        typer.echo(f"\nDEBUG: Graph has {len(graph)} functions\n")

    result = AuditResult()

    for py_file in directory.rglob("*.py"):
        file_total, file_issues, file_endpoints = _process_file(py_file, graph, debug)
        result.total += file_total
        result.issues.extend(file_issues)
        result.all_endpoints.extend(file_endpoints)

    return result


def _print_result(result: AuditResult, verbose: bool = False) -> None:
    """Display audit result."""
    if result.is_ok:
        typer.secho(
            f"✓ {result.total} endpoints tested: All secure",
            fg=typer.colors.GREEN,
        )
        raise typer.Exit(0)

    if not verbose:
        typer.secho(
            f"✗ {result.total} endpoints tested: {result.secure} secure, {len(result.issues)} NOT SECURE",
            fg=typer.colors.RED,
        )
        raise typer.Exit(1)

    typer.secho(
        f"✗ {result.total} endpoints tested: {result.secure} secure, {len(result.issues)} NOT SECURE\n",
        fg=typer.colors.RED,
    )

    for issue in result.issues:
        if "line" in issue:
            method = issue.get("method", "?")
            typer.echo(f"  {method} {issue['path']}")
            typer.echo(f"    {issue['file']}:{issue['line']}")
            typer.echo(f"    Function: {issue['function']}")
        else:
            methods_val = issue.get("methods")
            methods = ", ".join(methods_val) if isinstance(methods_val, set) else "?"
            typer.echo(f"  {methods} {issue['path']}")
            typer.echo(f"    Endpoint: {issue['endpoint']}")
        typer.echo()

    raise typer.Exit(1)


def _print_all_endpoints(result: AuditResult) -> None:
    """Print all found endpoints."""
    typer.echo(f"Found {len(result.all_endpoints)} endpoints:\n")

    for ep in sorted(result.all_endpoints, key=lambda x: str(x.get("path", ""))):
        methods = ep.get("methods")
        methods_str = ", ".join(sorted(methods)) if isinstance(methods, set) else str(ep.get("method", "?"))

        status = ep.get("status", "?")
        path = ep.get("path", "?")

        if status == "secure":
            typer.secho(f"  ✓ {methods_str:8} {path}", fg=typer.colors.GREEN)
        elif status == "whitelisted":
            typer.secho(f"  ○ {methods_str:8} {path} (whitelisted)", fg=typer.colors.YELLOW)
        else:
            typer.secho(f"  ✗ {methods_str:8} {path}", fg=typer.colors.RED)


@app.command()
def static(
    directory: Annotated[
        Path,
        typer.Argument(
            exists=True,
            file_okay=False,
            resolve_path=True,
            help="Directory with FastAPI code",
        ),
    ],
    verbose: Annotated[bool, typer.Option("-v", "--verbose", help="Show detailed output")] = False,
    debug: Annotated[bool, typer.Option("--debug", help="Show debug output")] = False,
    list_all: Annotated[bool, typer.Option("--list-all", help="List all found endpoints")] = False,
) -> None:
    """Analyze code statically for missing authorization."""
    result = _check_static(directory, debug)
    if list_all:
        _print_all_endpoints(result)
    elif not debug:
        _print_result(result, verbose)


@app.command()
def runtime(
    app_path: Annotated[
        Path,
        typer.Argument(
            exists=True,
            dir_okay=False,
            resolve_path=True,
            help="Path to main.py with FastAPI app",
        ),
    ],
    project_root: Annotated[
        Path | None,
        typer.Option(
            "--project-root",
            "-p",
            exists=True,
            file_okay=False,
            resolve_path=True,
            help="Project root directory to add to sys.path",
        ),
    ] = None,
    verbose: Annotated[bool, typer.Option("-v", "--verbose", help="Show detailed output")] = False,
    debug: Annotated[bool, typer.Option("--debug", help="Show debug output")] = False,
    list_all: Annotated[bool, typer.Option("--list-all", help="List all found endpoints")] = False,
) -> None:
    """Analyze running application for missing authorization."""
    root = project_root or app_path.parent.parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    result = _check_runtime(app_path, debug)
    if list_all:
        _print_all_endpoints(result)
    elif not debug:
        _print_result(result, verbose)


@app.command()
def config() -> None:
    """Show current configuration."""
    typer.echo("Auth dependencies:")
    for dep in sorted(AUTH_DEPENDENCIES):
        typer.echo(f"  - {dep}")

    typer.echo("\nPublic whitelist:")
    for path, methods in sorted(PUBLIC_WHITELIST.items()):
        methods_str = ", ".join(sorted(methods)) if methods else "ALL"
        typer.echo(f"  - {path} [{methods_str}]")


@app.command()
def graph(
    directory: Annotated[
        Path,
        typer.Argument(
            exists=True,
            file_okay=False,
            resolve_path=True,
            help="Directory with FastAPI code",
        ),
    ],
    debug: Annotated[bool, typer.Option("--debug", help="Show debug output")] = False,
) -> None:
    """Show dependency graph for debugging."""
    dep_graph = _build_dependency_graph(directory, debug)

    if not dep_graph:
        typer.echo("No dependencies found")
        raise typer.Exit(0)

    typer.echo("Dependency graph:\n")
    for func, deps in sorted(dep_graph.items()):
        has_auth = _has_auth_in_chain(func, dep_graph)
        marker = "🔐" if has_auth else "  "
        typer.echo(f"{marker} {func}")
        for dep in sorted(deps):
            is_auth = dep in AUTH_DEPENDENCIES
            dep_marker = "🔑" if is_auth else "  "
            typer.echo(f"     {dep_marker} └── {dep}")


if __name__ == "__main__":
    app()
