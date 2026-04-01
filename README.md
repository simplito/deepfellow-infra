# DeepFellow Infra
Key features:

- Ultra-simple installation.
- OpenAI-compatible model API — so that the DF Infra endpoint can be used with various 3rd-party tools.
- Ultra-simple scaling path.
- Handling multi-GPU box scenarios.
- Operation on cloud GPU nodes.
- Full management of any infra host via the deepfellow infra command.

## Install
You need python 3.13 with uv, to install dependencies:
```bash
uv sync
```

## Server start
You need [just](https://github.com/casey/just). To start server type:

```bash
just dev
```

## Docker image

Build
```bash
docker build -t infra .
```

To run application in docker you need share `docker.sock`. For rootless it is `/run/user/$UID/docker.sock` for normal setup it is `/var/run/docker.sock`. To run docker image with rootless call:
```
docker run -it --rm \
  -p 8086:8086 \
  -v $PWD/storage:/app/storage \
  -v /run/user/$UID/docker.sock:/var/run/docker.sock \
  infra
```

## License
MIT
