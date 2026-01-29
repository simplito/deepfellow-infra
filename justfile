dev:
    uv run uvicorn server.main:app --host "localhost" --port 8086 --reload --log-level debug

dev-trace:
    uv run uvicorn server.main:app --reload --log-level trace

test *FLAGS:
    uv run pytest --showlocals --tb=auto -ra --cov server --cov scripts --cov-branch --cov-report=term-missing --no-cov-on-fail {{FLAGS}}

ntest *FLAGS:
    uv run pytest --showlocals --tb=auto -ra --cov server --cov-branch --cov-report=term-missing --no-cov-on-fail -n auto {{FLAGS}}

ruff *FLAGS:
    uv run ruff check server/ tests/ {{FLAGS}}

ruff-format *FLAGS:
    uv run ruff format server/ tests/ {{FLAGS}}

auth-static:
    uv run python ./server/scripts/check_auth.py static ./server/ -v

auth-runtime:
    uv run python ./server/scripts/check_auth.py runtime ./server/main.py -v

pyright:
    uv run pyright

mypy *FLAGS:
    uv run mypy server/ tests/ {{FLAGS}}

pre-commit:
    uv run pre-commit run --all-files

todo:
    git grep "# TODO" -- "*.py"
    git grep "# FIX" -- "*.py"
    git grep "# DONE" -- "*.py"

license-check *FLAGS:
    uv run scripts/check_license_header.py {{FLAGS}}

check: license-check ruff ruff-format pyright auth-static auth-runtime

ui-rebuild:
   (cd $(git rev-parse --show-toplevel)/webui && npm run buildx)

env-copy:
    uv run python ./scripts/copy_envs.py

replace-docker *FLAGS:
    docker stop infra; just dev {{FLAGS}}
