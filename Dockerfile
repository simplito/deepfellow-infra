FROM simplito/python-docker:3.13.7-docker28.3.3 AS base

FROM base AS builder
COPY --from=ghcr.io/astral-sh/uv:0.8.12 /uv /uvx /bin/

WORKDIR /app
COPY . .
RUN uv venv .venv \
  && uv sync --frozen \
  && uv run ruff check server/ tests/ \
  && uv run ruff format server/ tests/ --check \
  && uv run pyright \
  && uv run pytest --showlocals --tb=auto -ra --cov server --cov-branch --cov-report=term-missing tests/ \
  && rm -rf .venv \
  && uv sync --frozen --no-dev \
  && rm -rf .venv/lib/python*/site-packages/*/test \
  && rm -rf .venv/lib/python*/site-packages/*/tests \
  && rm -rf tests .ruff_cache .pytest_cache

FROM base AS runner

WORKDIR /app
COPY --from=builder /app /app

CMD ["./.venv/bin/uvicorn", "server.main:app", "--host", "0.0.0.0", "--port", "8086"]
