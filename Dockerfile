FROM hub.simplito.com/public/python-docker:3.13.11-docker29.1.2@sha256:11f127bf40f09b49f3b21ada7174242c391c81185811f8d413bd165680fa0d0b AS base

FROM base AS builder
COPY --from=hub.simplito.com/public/uv:0.8.12@sha256:f64ad69940b634e75d2e4d799eb5238066c5eeda49f76e782d4873c3d014ea33 /uv /uvx /bin/

WORKDIR /app
COPY . .
RUN uv venv .venv
RUN uv sync --frozen
RUN uv run pytest --showlocals --tb=auto -ra --cov server --cov-branch --cov-report=term-missing tests/
RUN rm -rf .venv
RUN uv sync --frozen --no-dev
RUN rm -rf .venv/lib/python*/site-packages/*/test
RUN rm -rf .venv/lib/python*/site-packages/*/tests
RUN rm -rf tests .ruff_cache .pytest_cache

FROM hub.simplito.com/public/node:22.20.0-bookworm AS webbuilder
WORKDIR /app/webui
COPY webui .
RUN npm install
RUN npm run build

FROM base AS runner

WORKDIR /app
COPY --from=builder /app /app
COPY --from=webbuilder /app/webui/dist /app/static

CMD ["./.venv/bin/uvicorn", "server.main:app", "--host", "0.0.0.0", "--port", "8086", "--log-config", "server/utils/logging_config.yaml"]
