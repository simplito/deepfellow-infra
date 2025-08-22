"""BentoML service."""

from server.docker import DockerOptions, docker

service = docker(
    DockerOptions(
        image_port=3000, name="bentoml", image="summarization:7dsnrfcvtk4b3hdl", command="serve", env_vars={}, api_endpoint="/summarize"
    )
)
