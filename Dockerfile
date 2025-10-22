FROM hub.simplito.com/public/python-docker:3.13.7-docker28.3.3@sha256:de7aca8e8eddc7e4a46add921fc03d2132f3d4a4dae67e4b99c698bed02195e4 AS base

FROM base AS builder
COPY --from=hub.simplito.com/public/uv:0.8.12@sha256:f64ad69940b634e75d2e4d799eb5238066c5eeda49f76e782d4873c3d014ea33 /uv /uvx /bin/

WORKDIR /app
COPY . .
RUN uv venv .venv
RUN uv sync --frozen
RUN uv run ruff check server/ tests/
RUN uv run ruff format server/ tests/ --check
RUN PYRIGHT_PYTHON_NODE_VERSION=24.10.0 uv run pyright
RUN uv run pytest --showlocals --tb=auto -ra --cov server --cov-branch --cov-report=term-missing tests/
RUN rm -rf .venv
RUN uv sync --frozen --no-dev
RUN rm -rf .venv/lib/python*/site-packages/*/test
RUN rm -rf .venv/lib/python*/site-packages/*/tests
RUN rm -rf tests .ruff_cache .pytest_cache

FROM base AS runner

WORKDIR /app
COPY --from=builder /app /app

CMD ["./.venv/bin/uvicorn", "server.main:app", "--host", "0.0.0.0", "--port", "8086"]
