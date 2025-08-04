import importlib.util
import socket 
from pathlib import Path
from .endpointregistry import EndpointRegistry
from .config import Config
from .serviceprovider import ServiceProvider
from .utils import Utils

class PortUnavailableException(Exception):
    pass
class ApplicationContext:
    def __init__(self, endpoint_registry: EndpointRegistry, config: Config, service_provider: ServiceProvider):
        self.endpoint_registry = endpoint_registry
        self.config = config
        self.service_provider = service_provider
        self.allocated_ports = set()

    async def run(self, args):
        service_name = args[0]
        if not service_name or ".." in service_name or "/" in service_name or "\\" in service_name:
            raise Exception("Invalid module name")
        service_path = (Path(__file__).parent / f"./services/{service_name}/index.py").resolve()
        if not service_path.is_file():
            raise FileNotFoundError(f"Service '{service_name}' not found at {service_path}")
        
        spec = importlib.util.spec_from_file_location(service_name, service_path)
        plugin = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(plugin)
        if not hasattr(plugin, "service"):
            raise AttributeError(f"Service '{service_name}' does not define a 'service(context, args)' function")
        return await plugin.service(self, args)
    
    async def load(self):
        info = self.service_provider.load()
        for x in info["bootstrapCommands"]:
            await self.run(x["args"])
    
    def add_command_to_bootstrap(self, args):
        return self.service_provider.save_command(args)
    
    def remove_command_from_bootstrap(self, args):
        return self.service_provider.remove_command(args)
    
    def get_free_port(self, start=20_000, end=30_000):
        for port in range(start, end+1):
            if port in self.allocated_ports:
                continue
            if self.is_port_available(port):
                self.allocated_ports.add(port)
                return port
        raise RuntimeError(f"No free port in range {start} - {end}")
    
    def get_docker_compose_dir(self):
        dir = (Path(__file__).resolve().parent / "../../storage/config")
        if not dir.is_dir():
            dir.mkdir()

        return dir

    def get_tts_dir(self):
        dir = (Path(__file__).resolve().parent / "../../storage/tts")
        if not dir.is_dir():
            dir.mkdir()

        return dir
    
    def get_images_dir(self):
        dir = (Path(__file__).resolve().parent / "../../storage/images")
        if not dir.is_dir():
            dir.mkdir()
            for name in ["extensions", "outputs", "embeddings", "models"]:
                (dir / name).mkdir()

        return dir

    def get_model_dir(self):
        dir = (Path(__file__).resolve().parent / "../../storage/models")
        if not dir.is_dir():
            dir.mkdir()

        return dir
    
    def is_port_available(self, port: int):
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind(('', port))
            sock.close()
            return True
        except (OSError, socket.error):
            return False
    
    
        
        

