dev:
    uv run uvicorn server.main:app --host "localhost" --port 8086 --reload --log-level debug

dev-trace:
    uv run uvicorn server.main:app --reload --log-level trace

test *FLAGS:
    uv run pytest --showlocals --tb=auto -ra --cov server --cov-branch --cov-report=term-missing --no-cov-on-fail {{FLAGS}}

ruff *FLAGS:
    uv run ruff check server/ tests/ {{FLAGS}}

ruff-format *FLAGS:
    uv run ruff format server/ tests/ {{FLAGS}}

mypy *FLAGS:
    uv run mypy server/ tests/ {{FLAGS}}

pre-commit:
    uv run pre-commit run --all-files

print:
    git grep "print(" -- "*.py"

todo:
    git grep "# TODO" -- "*.py"
    git grep "# FIX" -- "*.py"
    git grep "# DONE" -- "*.py"
