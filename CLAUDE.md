# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

DeepFellow Infra is a self-hosted AI infrastructure stack — a FastAPI backend with an OpenAI-compatible API plus a React web UI. It manages containerized AI services (LLMs, embeddings, image generation) via Docker and exposes them through a unified endpoint registry.

## Commands

All Python commands use `uv` via `just`. Run from the project root.

| Command | Purpose |
|---------|---------|
| `just dev` | Start backend (localhost:8086, hot-reload, debug logging) |
| `just test [FLAGS]` | Run pytest with coverage |
| `just ntest [FLAGS]` | Same but parallel (`-n auto`) |
| `just check` | Full lint suite: license headers, ruff, ruff-format, pyright, auth checks |
| `just ruff` | Lint Python files |
| `just ruff-format` | Format Python files |
| `just pyright` | Type-check with pyright |
| `just ui-rebuild` | Build WebUI and copy to `static/` |
| `just test-webui` | Run WebUI unit tests (Vitest) |

WebUI (from `webui/` directory):
- `npm run dev` — dev server on port 3000
- `npm run buildx` — build and copy to `../static/`
- `npm run check` — biome lint + format check
- `npm test` — run unit tests once (same as `just test-webui`)
- `npx vitest` — run tests in watch mode (re-runs on file change)

## Code Style

**Python:**
- Line length: 140 characters
- Formatter: `ruff format`; linter: `ruff check`
- Type checker: pyright (strict); all code must be fully annotated
- All `.py` files require a DeepFellow Free License copyright header — run `just license-check` to verify
- Prefer Pydantic models over plain dicts

**TypeScript/React (`webui/`):**
- Formatter and linter: Biome; indent: 2 spaces; quotes: double
- `src/routeTree.gen.ts` and `src/components/ui/*` are auto-generated — do not edit manually
- File-based routing via TanStack Router

## Branch and Commit Conventions

- Branch naming: `<gitlab-issue-id>-<short-description>` (e.g. `333-add-claude-service`)
- Commits: conventional commits format (enforced by pre-commit hook)
- Run `just pre-commit` before pushing
- Never add `Co-Authored-By: Claude` trailers to commits
- Never add `🤖 Generated with Claude Code` or similar AI attribution to PR/MR descriptions

## Testing

**Backend (Python):**
- Pytest with strict async mode; per-function scope
- Tests load `.test.env` — create from `example.env` if missing
- Single test: `just test -k 'test_name'`
- Coverage covers `server/` and `scripts/`

**WebUI (TypeScript):**
- Framework: Vitest — installed automatically with `npm install` (it's in `devDependencies`)
- No extra setup required; test files live next to source files as `*.test.ts`
- Run once: `just test-webui` or `npm test` (from `webui/`)
- Watch mode: `npx vitest` (from `webui/`) — reruns on save
- Tests cover pure utilities and store logic (`src/utils/`, `src/state/`)

## Required Environment Variables

Must be set before running:
- `DF_NAME`, `DF_INFRA_URL`, `DF_INFRA_API_KEY`, `DF_INFRA_ADMIN_API_KEY`, `DF_MESH_KEY`

Copy `example.env` to `.env` to get started. See `server/config.py` for full list.

## Gotchas

- App requires Docker socket at runtime (`/run/user/1000/docker.sock` rootless or `/var/run/docker.sock` standard)
- WebUI dev server proxies to backend — set `VITE_DF_SERVER_URL=http://localhost:8086/`
- Auth is validated separately from linting: `just auth-static` (static) and `just auth-runtime` (runtime)
