"""Docker backend."""

import asyncio
import json
import shutil
from collections.abc import Callable, Coroutine, Mapping
from pathlib import Path
from typing import Any, Literal

import yaml
from fastapi import Request

from server.applicationcontext import ApplicationContext
from server.utils.core import CommandResult2, Utils
from server.utils.exceptions import AppError

from .endpointregistry import ProxyOptions, SimpleEndpoint


class DockerNotInstalledError(Exception):
    pass


class ChatCompletionsOptions:
    def __init__(self, model: str, remove_model: bool = False, rewrite_model_to: str | None = None):
        self.model = model
        self.remove_model = remove_model
        self.rewrite_model_to = rewrite_model_to


class FunctionHandler:
    type: Literal["function"]

    def __init__(self, func: Callable[[int, dict, Request], Coroutine[Any, Any, Any]]):
        self.type = "function"
        self.func = func


class ProxyHandler:
    type: Literal["proxy"]

    def __init__(self, url: str = "/v1/audio/speech"):
        self.type = "proxy"
        self.url = url


class ImageGenerationsOptions:
    def __init__(self, model: str, handler: FunctionHandler | ProxyHandler):
        self.model = model
        self.handler = handler


class AudioSpeechOptions:
    def __init__(self, model: str, handler: FunctionHandler | ProxyHandler):
        self.model = model
        self.handler = handler


class ImageOptions:
    pass


class DockerOptions:
    def __init__(
        self,
        name: str,
        image: str,
        image_port: int,
        command: str | None = None,
        use_gpu: bool = False,
        volumes: list | None = None,
        restart: str | None = None,
        env_vars: Mapping | None = None,
        additional_bootstrap_args: list[str] | None = None,
        api_endpoint: str | None = None,
        chat_completion: ChatCompletionsOptions | None = None,
        image_generations: ImageGenerationsOptions | None = None,
        audio_generation: AudioSpeechOptions | None = None,
        ulimits: Mapping | None = None,
        shm_size: str | None = None,
        entrypoint: str | None = None,
        healthcheck: str | None = None,
    ):
        self.name = name
        self.image = image
        self.command = command
        self.image_port = image_port
        self.additional_bootstrap_args = additional_bootstrap_args or []
        self.env_vars = env_vars or {}
        self.api_endpoint = api_endpoint
        self.service_name = Utils.sanitize_service_name(name)
        self.restart = restart
        self.volumes = volumes
        self.use_gpu = use_gpu
        self.chat_completion = chat_completion
        self.image_generations = image_generations
        self.audio_generation = audio_generation
        self.uliumits = ulimits
        self.shm_size = shm_size
        self.entrypoint = entrypoint
        self.healthcheck = healthcheck

        if shutil.which("docker-compose"):
            self.docker_compose_cmd = "docker-compose"
        elif shutil.which("docker"):
            self.docker_compose_cmd = "docker compose"
        else:
            raise DockerNotInstalledError("Docker is not installed.")

    def __repr__(self):
        """Return string representation fo the object."""
        return (
            f"DockerOptions(name='{self.name}', image='{self.image}', command='{self.command}', image_port={self.image_port}, "
            f"service_name='{self.service_name}', docker_compose_cmd='{self.docker_compose_cmd}, "
            f"audio_generation='{self.audio_generation}')"
        )


def docker(options: DockerOptions) -> Callable[[ApplicationContext, list[str]], Any]:  # noqa: C901
    """Prepare docker setup."""

    async def _is_docker_compose_running(docker_compose_file_path: Path, service_name: str) -> bool:
        docker_compose_cmd = options.docker_compose_cmd
        if not docker_compose_file_path.exists():
            return False
        try:
            cmd = f"{docker_compose_cmd} -f {docker_compose_file_path} ps --services --filter status=running"
            result = await Utils.run_command(cmd)
            return service_name in result.stdout  # noqa: TRY300
        except Exception:
            return False

    async def _is_docker_compose_healthy(docker_compose_file_path: Path, service_name: str) -> bool:
        docker_compose_cmd = options.docker_compose_cmd
        if not docker_compose_file_path.exists():
            print(f"{docker_compose_file_path} not found")
            return False

        try:
            cmd = f"{docker_compose_cmd} -f {docker_compose_file_path} ps {service_name} --format json"
            result = await Utils.run_command(cmd)

            if result.exit_code != 0:
                print(f"{docker_compose_file_path} {cmd} exit code is not 0. Exit code is {result.exit_code}")
                return False

            container = json.loads(result.stdout)

            health = container.get("Health", "").lower()
            if "unhealthy" in health:
                print(f"Docker container {service_name} is unhealthy")
                return False
            if "healthy" in health:
                return True

            state = container.get("State", "").lower()
            if state == "running":
                return True

            print(f"Docker container {service_name} is in state {state}")
            return False  # noqa: TRY300
        except Exception as exc:
            print(f"Error while checking health of docker container {service_name}. Error: {exc}")
            return False

    def _register_api_endpoint(options: DockerOptions, ctx: ApplicationContext, port: int) -> None:  # noqa: C901
        if options.api_endpoint is not None:
            url = Utils.join_url(f"http://localhost:{port}", options.api_endpoint)
            ctx.endpoint_registry.register_custom_endpoint_as_proxy(
                options.api_endpoint,
                ProxyOptions(
                    url=url,
                ),
            )
        if options.chat_completion is not None:
            url = f"http://localhost:{port}/v1/chat/completions"
            ctx.endpoint_registry.register_chat_completion_as_proxy(
                options.chat_completion.model,
                ProxyOptions(
                    url=url, rewrite_model_to=options.chat_completion.rewrite_model_to, remove_model=options.chat_completion.remove_model
                ),
            )
        if options.audio_generation:
            if options.audio_generation.handler.type == "proxy":
                url = Utils.join_url(f"http://localhost:{port}", options.audio_generation.handler.url)
                ctx.endpoint_registry.register_audio_speech_as_proxy(
                    options.audio_generation.model,
                    ProxyOptions(
                        url=url,
                    ),
                )
            elif options.audio_generation.handler.type == "function":
                audio_handler = options.audio_generation.handler

                async def my_audio_handler(body: dict, req: Request) -> Any:  # noqa: ANN401
                    return await audio_handler.func(port, body, req)

                ctx.endpoint_registry.register_audio_speech(options.audio_generation.model, SimpleEndpoint(on_request=my_audio_handler))

        if options.image_generations:
            if options.image_generations.handler.type == "proxy":
                url = Utils.join_url(f"http://localhost:{port}", options.image_generations.handler.url)
                ctx.endpoint_registry.register_image_generations_as_proxy(
                    options.image_generations.model,
                    ProxyOptions(
                        url=url,
                    ),
                )
            elif options.image_generations.handler.type == "function":
                img_handler = options.image_generations.handler

                async def my_img_handler(body: dict, req: Request) -> Any:  # noqa: ANN401
                    return await img_handler.func(port, body, req)

                ctx.endpoint_registry.register_image_generations(options.image_generations.model, SimpleEndpoint(on_request=my_img_handler))

    def _generate_docker_compose_content(options: DockerOptions, port: int) -> dict:
        docker_compose_content = {
            "services": {
                options.service_name: {
                    "image": options.image,
                    "ports": [f"{port}:{options.image_port}"],
                    "environment": options.env_vars,
                }
            }
        }
        if options.healthcheck:
            docker_compose_content["services"][options.service_name]["healthcheck"] = options.healthcheck
        if options.command:
            docker_compose_content["services"][options.service_name]["command"] = options.command
        if options.volumes:
            docker_compose_content["services"][options.service_name]["volumes"] = options.volumes
        if options.restart:
            docker_compose_content["services"][options.service_name]["restart"] = options.restart
        if options.shm_size:
            docker_compose_content["services"][options.service_name]["shm_size"] = options.shm_size
        if options.entrypoint:
            docker_compose_content["services"][options.service_name]["entrypoint"] = options.entrypoint
        if options.use_gpu:
            docker_compose_content["services"][options.service_name]["deploy"] = {
                "resources": {"reservations": {"devices": [{"driver": "nvidia", "count": 1, "capabilities": ["gpu"]}]}}
            }
        return docker_compose_content

    async def _has_docker_compose_difference(docker_compose_file_path: Path, options: DockerOptions) -> tuple[bool, int | None]:
        if not docker_compose_file_path.exists():
            return True, None

        try:
            # Read current docker compose file
            current_content = docker_compose_file_path.read_text()
            current_config = yaml.safe_load(current_content)

            # # Generate desired configuration
            current_service = current_config.get("services", {}).get(options.service_name, {})
            current_ports = current_service.get("ports", [])
            current_port = int(current_ports[0].split(":")[0])

            desired_config = _generate_docker_compose_content(options, current_port)
            desired_content = yaml.dump(desired_config, default_flow_style=False, sort_keys=False)
            # Check image, command, environment
            return (current_content != desired_content, current_port)  # noqa: TRY300
        except Exception:
            return True, None

    async def _render_docker_compose(docker_compose_file_path: Path, options: DockerOptions, ctx: ApplicationContext) -> int:
        port = None
        # Check if old port is occupied and get a new one if needed
        if docker_compose_file_path.exists():
            try:
                current_content = docker_compose_file_path.read_text()
                current_config = yaml.safe_load(current_content)
                current_service = current_config.get("services", {}).get(options.service_name, {})
                current_ports = current_service.get("ports", [])

                if current_ports:
                    current_port = int(current_ports[0].split(":")[0])
                    # Check if port is still available
                    if ctx.is_port_available(current_port):
                        port = current_port
            except Exception:
                pass
        # Get new port
        if port is None:
            port = ctx.get_free_port()
        docker_compose_file_content = _generate_docker_compose_content(options, port)

        # Save rendered docker compose as file
        docker_compose_yaml = yaml.dump(docker_compose_file_content, default_flow_style=False, sort_keys=False)
        Utils.save_file(docker_compose_file_path.parent / (options.name + ".yaml"), docker_compose_yaml)

        return port

    async def _start_docker_compose(docker_compose_file_path: Path) -> CommandResult2:
        docker_compose_cmd = options.docker_compose_cmd
        cmd_parts = [*docker_compose_cmd.split(), "-f", str(docker_compose_file_path), "up", "-d"]
        command = " ".join(Utils.shell_escape(part) for part in cmd_parts)
        return await Utils.run_command_for_success(command)

    async def _stop_docker_compose(docker_compose_file_path: Path) -> None:
        docker_compose_cmd = options.docker_compose_cmd
        cmd_parts = [*docker_compose_cmd.split(), "-f", str(docker_compose_file_path), "down", "--remove-orphans"]
        command = " ".join(Utils.shell_escape(part) for part in cmd_parts)
        await Utils.run_command_for_success(command)

    async def handler(ctx: ApplicationContext, args: list[str]) -> dict[str, Any]:  # noqa: C901
        service, command = args
        bootstrap_args = [service, "run"] + (options.additional_bootstrap_args or [])

        if command == "install":
            docker_compose_dir = ctx.get_docker_compose_dir()
            docker_compose_file_path = docker_compose_dir / (options.name + ".yaml")

            port = await _render_docker_compose(docker_compose_file_path, options, ctx)
            await _start_docker_compose(docker_compose_file_path)

            _register_api_endpoint(options, ctx, port)

            await ctx.add_command_to_bootstrap(bootstrap_args)
            return {"success": True}

        if command == "run":
            port = None
            docker_compose_dir = ctx.get_docker_compose_dir()
            docker_compose_file_path = docker_compose_dir / (options.name + ".yaml")
            service_name = options.service_name

            # Check if docker compose is working
            is_running = await _is_docker_compose_running(docker_compose_file_path, service_name)

            # Check if there would be a difference in docker compose
            has_difference, port = await _has_docker_compose_difference(docker_compose_file_path, options)

            print(f"{service_name}\n{is_running=}\n{has_difference=}\n")
            start_output = ""

            # Handle different scenarios based on running state, health, and differences
            if not is_running and not has_difference:
                # Not running, no difference -> start
                start_output = await _start_docker_compose(docker_compose_file_path)
            elif not is_running and has_difference:
                # Not running, has difference -> render then start
                port = await _render_docker_compose(docker_compose_file_path, options, ctx)
                start_output = await _start_docker_compose(docker_compose_file_path)
            elif is_running and has_difference:
                # Running but has difference -> stop -> render -> start
                print(f"{service_name} config was changed. Restarting...")
                await _stop_docker_compose(docker_compose_file_path)
                port = await _render_docker_compose(docker_compose_file_path, options, ctx)
                start_output = await _start_docker_compose(docker_compose_file_path)

            # Check if container is healthy after starting
            is_healthy = await _is_docker_compose_healthy(docker_compose_file_path, service_name)
            retry_count = 0
            max_retries = 3
            while not is_healthy and retry_count < max_retries:
                print(
                    f"Container {service} wasn't healthy. Restarting and allocating a new port."
                    f"(Attempt {retry_count + 1}/{max_retries})\n Error: {start_output}"
                )
                await _stop_docker_compose(docker_compose_file_path)
                port = await _render_docker_compose(docker_compose_file_path, options, ctx)
                start_output = await _start_docker_compose(docker_compose_file_path)

                # Wait before checking health again
                await asyncio.sleep(2)

                is_healthy = await _is_docker_compose_healthy(docker_compose_file_path, service_name)
                retry_count += 1

            if not is_healthy:
                msg = f"Container {service} failed to become healthy after {max_retries} attempts"
                raise AppError(msg)

            if port is not None:
                _register_api_endpoint(options, ctx, port)
            else:
                raise AppError("Cannot register service.")

            return {"success": True}

        if command == "uninstall":
            await ctx.remove_command_from_bootstrap(bootstrap_args)

            if options.api_endpoint is not None:
                ctx.endpoint_registry.unregister_custom_endpoint(options.api_endpoint)
            if options.chat_completion is not None:
                ctx.endpoint_registry.unregister_chat_completion(options.chat_completion.model)

            docker_compose_file = Path(ctx.get_docker_compose_dir() / (options.name + ".yaml"))
            if docker_compose_file.is_file():
                docker_compose_file.unlink()

            return {"success": True}

        if command == "status":
            docker_compose_cmd = options.docker_compose_cmd
            docker_compose_file = str(Path(ctx.get_docker_compose_dir() / (options.name + ".yaml")))
            res = await Utils.run_command(f"{docker_compose_cmd} logs -f {docker_compose_file}")
            if res.exit_code == 1 and res.stderr.strip() == f"Error: file '{docker_compose_file}' not found":
                return {"success": True, "info": "not found"}
            return {"success": True, "info": res.stdout}

        return {"success": False, "error": f"Unknown command {command}"}

    return handler
