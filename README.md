# DeepFellow Infra

DeepFellow Infra is a self-hosted infrastructure stack purpose-built to run AI on your own hardware. It provides the compute foundation for LLMs, embedding models, and MLAs — giving you full ownership of the environment without relying on third-party cloud APIs.

DeepFellow Infra is part of **DeepFellow** — a private AI framework that enables software teams to deliver compliant, auditable, and scalable AI tailored to the most demanding sectors.

Unlike SaaS-based LLM providers, DeepFellow runs entirely within **your environment**, adapts to your internal knowledge, and **scales** with your operations — **without exposing sensitive data** or compromising auditability. You control which data the system learns from, who can access it, and where it runs.

## Key Features

- **Minimal setup** — get from zero to a running inference stack in minutes.
- **OpenAI-compatible API** — drop-in replacement endpoint that works with existing tools, SDKs, and integrations out of the box.
- **Straightforward scaling** — go from a single GPU to a multi-node cluster with minimal configuration changes.
- **Multi-GPU support** — automatically manages workload distribution across multiple GPUs on a single host.
- **Cloud GPU ready** — operate on cloud GPU nodes.
- **Unified management** — control any infrastructure host through the `deepfellow infra` CLI.

## Installation

Requires Python 3.13 with [uv](https://docs.astral.sh/uv/). To install dependencies:

```bash
uv sync
```

## Server Start

Requires [just](https://github.com/casey/just). To start the development server:

```bash
just dev
```

## Docker Image

Build the image:

```bash
docker build -t infra .
```

Running the container requires access to the Docker socket. The socket path depends on your Docker setup — `/run/user/$UID/docker.sock` for rootless installations, or `/var/run/docker.sock` for standard (root) installations.

Example for a rootless setup:

```
docker run -it --rm \
  -p 8086:8086 \
  -v $PWD/storage:/app/storage \
  -v /run/user/$UID/docker.sock:/var/run/docker.sock \
  infra
```

## License

MIT
