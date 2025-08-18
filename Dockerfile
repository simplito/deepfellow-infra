FROM simplito/python-docker:3.13.7-docker28.3.3

WORKDIR /app
COPY . .
RUN pip install uv && uv venv .venv && uv sync

CMD ["uv", "run", "uvicorn", "server.main:app", "--host", "0.0.0.0", "--port", "8086"]
