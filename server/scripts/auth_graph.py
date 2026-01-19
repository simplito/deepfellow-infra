#!/usr/bin/env python3
"""FastAPI auth dependency chain analyzer.

Analyzes FastAPI endpoints and groups them by authorization dependency chain.
Useful for auditing which endpoints use which auth mechanisms.
"""

import ast
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import typer

app = typer.Typer()


@dataclass
class DependencyNode:
    """Represents a function and its dependencies."""

    name: str
    file: str
    line: int
    depends_on: list[str] = field(default_factory=list)


@dataclass
class Endpoint:
    """Represents an API endpoint."""

    name: str
    file: str
    line: int
    method: str
    path: str
    direct_deps: list[str] = field(default_factory=list)


class DependencyGraphBuilder(ast.NodeVisitor):
    """AST visitor that builds a dependency graph from FastAPI code."""

    def __init__(self, debug: bool = False) -> None:
        self.dependencies: dict[str, DependencyNode] = {}
        self.endpoints: list[Endpoint] = []
        self.current_file: str | None = None
        self.debug = debug
        self._in_class = False
        self._router_prefixes: dict[str, str] = {}
        self._current_prefixes: dict[str, str] = {}

    def visit_Module(self, node: ast.Module) -> None:
        """Reset prefixes for each file."""
        self._current_prefixes = {}
        self.generic_visit(node)

    def visit_Assign(self, node: ast.Assign) -> None:
        """Look for router = APIRouter(prefix='...')."""
        if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            var_name = node.targets[0].id
            prefix = self._extract_router_prefix(node.value)
            if prefix is not None:
                self._current_prefixes[var_name] = prefix
                if self.debug:
                    typer.echo(f"  [ROUTER] {var_name} = APIRouter(prefix='{prefix}')")

        self.generic_visit(node)

    def _extract_router_prefix(self, node: ast.expr) -> str | None:
        """Extract prefix from APIRouter(prefix='...')."""
        if not isinstance(node, ast.Call):
            return None

        func_name = ""
        if isinstance(node.func, ast.Name):
            func_name = node.func.id
        elif isinstance(node.func, ast.Attribute):
            func_name = node.func.attr

        if func_name != "APIRouter":
            return None

        for keyword in node.keywords:
            if keyword.arg == "prefix" and isinstance(keyword.value, ast.Constant):
                value = keyword.value.value
                if isinstance(value, str):
                    return value

        if node.args and isinstance(node.args[0], ast.Constant):
            value = node.args[0].value
            if isinstance(value, str):
                return value

        return ""

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        """Track when inside a class to skip methods."""
        old_in_class = self._in_class
        self._in_class = True
        self.generic_visit(node)
        self._in_class = old_in_class

    def visit_FunctionDef(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        """Process function definitions, skip class methods."""
        if self._in_class:
            return

        deps = self._extract_depends(node)

        self.dependencies[node.name] = DependencyNode(
            name=node.name,
            file=self.current_file or "",
            line=node.lineno,
            depends_on=deps,
        )

        endpoint_info = self._get_endpoint_info(node)
        if endpoint_info:
            method, path, router_var = endpoint_info

            prefix = self._current_prefixes.get(router_var, "")
            full_path = prefix + path

            self.endpoints.append(
                Endpoint(
                    name=node.name,
                    file=self.current_file or "",
                    line=node.lineno,
                    method=method,
                    path=full_path,
                    direct_deps=deps,
                )
            )

        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        """Process async function definitions."""
        self.visit_FunctionDef(node)

    def _extract_depends(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> list[str]:
        """Extract all Depends() from function arguments."""
        deps = []

        all_defaults = node.args.defaults + [d for d in node.args.kw_defaults if d]
        for default in all_defaults:
            dep_name = self._extract_depends_arg(default)
            if dep_name:
                deps.append(dep_name)

        for arg in node.args.args:
            if arg.annotation:
                found = self._extract_all_depends_from_annotation(arg.annotation)
                deps.extend(found)

        if self.debug and deps:
            typer.echo(f"  [PARSE] {node.name} -> {deps}")

        return deps

    def _extract_depends_arg(self, node: ast.expr) -> str | None:
        """Extract function name from Depends(func)."""
        if not isinstance(node, ast.Call):
            return None

        func_name = ""
        if isinstance(node.func, ast.Name):
            func_name = node.func.id
        elif isinstance(node.func, ast.Attribute):
            func_name = node.func.attr

        if func_name != "Depends":
            return None

        if node.args:
            arg = node.args[0]
            if isinstance(arg, ast.Name):
                return arg.id
            if isinstance(arg, ast.Attribute):
                return arg.attr
            if isinstance(arg, ast.Call) and isinstance(arg.func, ast.Name):
                return arg.func.id

        return None

    def _is_annotated(self, node: ast.expr) -> bool:
        """Check if node is Annotated type."""
        if isinstance(node, ast.Name):
            return node.id == "Annotated"
        if isinstance(node, ast.Attribute):
            return node.attr == "Annotated"
        return False

    def _extract_all_depends_from_annotation(self, annotation: ast.expr) -> list[str]:
        """Extract ALL Depends from annotation (including Annotated)."""
        deps = []

        if isinstance(annotation, ast.Subscript) and self._is_annotated(annotation.value):
            if isinstance(annotation.slice, ast.Tuple):
                for elt in annotation.slice.elts[1:]:
                    dep = self._extract_depends_arg(elt)
                    if dep:
                        deps.append(dep)
            else:
                dep = self._extract_depends_arg(annotation.slice)
                if dep:
                    deps.append(dep)

        return deps

    def _get_endpoint_info(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> tuple[str, str, str] | None:
        """Return (method, path, router_var) or None."""
        route_methods = {"get", "post", "put", "delete", "patch", "head", "options"}

        for dec in node.decorator_list:
            if isinstance(dec, ast.Call) and isinstance(dec.func, ast.Attribute):
                method = dec.func.attr
                if method in route_methods:
                    router_var = ""
                    if isinstance(dec.func.value, ast.Name):
                        router_var = dec.func.value.id

                    path = ""
                    if dec.args and isinstance(dec.args[0], ast.Constant):
                        value = dec.args[0].value
                        if isinstance(value, str):
                            path = value

                    return (method.upper(), path, router_var)

        return None


class AuthChainAnalyzer:
    """Analyzes auth dependency chains for endpoints."""

    def __init__(
        self,
        graph: DependencyGraphBuilder,
        terminals: set[str],
        debug: bool = False,
    ) -> None:
        self.deps = graph.dependencies
        self.endpoints = graph.endpoints
        self.terminals = terminals
        self.debug = debug

    def find_path_to_terminal(self, func_name: str, visited: set[str] | None = None) -> list[str] | None:
        """Find path from func_name to any terminal."""
        if visited is None:
            visited = set()

        if func_name in visited:
            return None
        visited.add(func_name)

        # Found a terminal
        if func_name in self.terminals:
            return [func_name]

        if func_name not in self.deps:
            return None

        node = self.deps[func_name]

        for dep in node.depends_on:
            if dep in self.terminals:
                return [func_name, dep]

            sub_path = self.find_path_to_terminal(dep, visited.copy())
            if sub_path:
                return [func_name, *sub_path]

        return None

    def get_auth_path(self, endpoint: Endpoint) -> list[str] | None:
        """Return path to terminal for endpoint."""
        for dep in endpoint.direct_deps:
            if dep in self.terminals:
                return [dep]

            path = self.find_path_to_terminal(dep)
            if path:
                return path

        return None

    def _get_signature(self, path: list[str] | None, tail_depth: int | None) -> tuple[str, ...]:
        """Get auth signature from path."""
        if not path:
            return ("NO_AUTH",)
        return tuple(path[-tail_depth:]) if tail_depth else tuple(path)

    def group_by_auth(self, tail_depth: int | None = None) -> dict[tuple[str, ...], list[Endpoint]]:
        """Group endpoints by path to terminal."""
        groups: dict[tuple[str, ...], list[Endpoint]] = defaultdict(list)

        for endpoint in self.endpoints:
            path = self.get_auth_path(endpoint)
            signature = self._get_signature(path, tail_depth)
            groups[signature].append(endpoint)

        return dict(groups)


class ReportFormatter:
    """Format analysis results for different outputs."""

    def __init__(self, analyzer: AuthChainAnalyzer, tail_depth: int | None = None) -> None:
        self.analyzer = analyzer
        self.groups = analyzer.group_by_auth(tail_depth)
        self.sorted_groups = sorted(self.groups.items(), key=lambda x: (x[0] == ("NO_AUTH",), x[0]))

    def _get_summary(self) -> tuple[int, int, int]:
        """Return (total, with_auth, no_auth) counts."""
        total = len(self.analyzer.endpoints)
        no_auth = len(self.groups.get(("NO_AUTH",), []))
        with_auth = total - no_auth
        return total, with_auth, no_auth

    def format_terminal(self, no_auth_only: bool = False) -> str:
        """Format as colored terminal output."""
        lines = []
        terminals_str = ", ".join(sorted(self.analyzer.terminals))

        for auth_chain, endpoints in self.sorted_groups:
            is_no_auth = auth_chain == ("NO_AUTH",)

            if no_auth_only and not is_no_auth:
                continue

            chain_str = " → ".join(auth_chain)
            lines.append(f"\n{'=' * 70}")
            lines.append(f"AUTH: {chain_str}")
            lines.append(f"{'=' * 70}")

            for ep in sorted(endpoints, key=lambda e: (e.file, e.path)):
                lines.append(f"  {ep.method:6} {ep.path:40} {ep.file}:{ep.line}")

        total, with_auth, no_auth = self._get_summary()
        lines.append(f"\n{'=' * 70}")
        lines.append(f"TOTAL: {total} endpoints ({with_auth} with auth, {no_auth} without path to [{terminals_str}])")

        return "\n".join(lines)

    def format_markdown(self, no_auth_only: bool = False) -> str:
        """Format as Markdown with checkboxes for PR audit."""
        terminals_str = ", ".join(f"`{t}`" for t in sorted(self.analyzer.terminals))

        lines = [
            "# Auth Audit Report",
            "",
            f"Terminal functions: {terminals_str}",
            "",
        ]

        total, with_auth, no_auth = self._get_summary()
        lines.extend(
            [
                "## Summary",
                "",
                f"- **Total endpoints:** {total}",
                f"- **With auth:** {with_auth}",
                f"- **Without auth:** {no_auth}",
                "",
            ]
        )

        for auth_chain, endpoints in self.sorted_groups:
            is_no_auth = auth_chain == ("NO_AUTH",)

            if no_auth_only and not is_no_auth:
                continue

            chain_str = " → ".join(f"`{c}`" for c in auth_chain)
            icon = "🔴" if is_no_auth else "🟢"

            lines.extend(
                [
                    f"## {icon} {chain_str}",
                    "",
                ]
            )

            for ep in sorted(endpoints, key=lambda e: (e.file, e.path)):
                lines.append(f"- [ ] **{ep.method}** `{ep.path}` — `{ep.file}:{ep.line}`")

            lines.append("")

        return "\n".join(lines)


def analyze_project(path: Path, terminals: set[str], debug: bool = False) -> AuthChainAnalyzer:
    """Analyze project and build dependency graph."""
    builder = DependencyGraphBuilder(debug=debug)

    for py_file in sorted(path.rglob("*.py")):
        try:
            if debug:
                typer.echo(f"[FILE] {py_file}")
            tree = ast.parse(py_file.read_text())
            builder.current_file = str(py_file.relative_to(path))
            builder.visit(tree)
        except SyntaxError as e:
            typer.echo(f"Syntax error in {py_file}: {e}", err=True)

    return AuthChainAnalyzer(builder, terminals, debug=debug)


@app.command()
def main(
    directory: Path = typer.Argument(  # noqa: B008
        Path("./server"),
        help="Directory with FastAPI code",
        exists=True,
        file_okay=False,
        dir_okay=True,
    ),
    terminal: list[str] = typer.Option(  # noqa: B008
        ["auth_admin", "auth_server"],
        "--terminal",
        "-t",
        help="Terminal function(s) (auth source). Can be specified multiple times.",
    ),
    tail_depth: int = typer.Option(
        2,
        "--depth",
        "-d",
        help="Group by last N levels of path to terminal",
    ),
    full_chain: bool = typer.Option(
        False,
        "--full",
        "-f",
        help="Show full path (ignores --depth)",
    ),
    no_auth_only: bool = typer.Option(
        False,
        "--no-auth-only",
        "-n",
        help="Show only endpoints without path to terminal",
    ),
    output_format: str = typer.Option(
        "terminal",
        "--format",
        "-o",
        help="Output format: terminal, md",
    ),
    output_file: Path | None = typer.Option(  # noqa: B008
        None,
        "--output-file",
        "-O",
        help="Write output to file",
    ),
    debug: bool = typer.Option(
        False,
        "--debug",
        help="Show debug info",
    ),
) -> None:
    """Analyze FastAPI endpoints and group them by auth dependency chain."""
    terminals = set(terminal)

    if debug:
        typer.secho("\n=== Parsing files ===", fg=typer.colors.YELLOW)

    analyzer = analyze_project(directory, terminals, debug=debug)

    if debug:
        typer.secho("\n=== Found dependencies ===", fg=typer.colors.YELLOW)
        for name, node in sorted(analyzer.deps.items()):
            if node.depends_on:
                typer.echo(f"  {name}: {node.depends_on}")

        typer.secho(f"\n=== Paths to {terminals} ===", fg=typer.colors.YELLOW)
        for name, node in sorted(analyzer.deps.items()):
            if node.depends_on:
                path = analyzer.find_path_to_terminal(name)
                if path:
                    typer.echo(f"  {name} -> {' → '.join(path)}")

    depth = None if full_chain else tail_depth
    formatter = ReportFormatter(analyzer, depth)

    output = formatter.format_markdown(no_auth_only) if output_format == "md" else formatter.format_terminal(no_auth_only)

    if output_file:
        output_file.write_text(output)
        typer.secho(f"Report written to {output_file}", fg=typer.colors.GREEN)
    else:
        typer.echo(output)


if __name__ == "__main__":
    app()
