"""Usage Manager."""

import logging
from collections.abc import AsyncGenerator, Awaitable, Callable
from typing import Any, TypeVar

from fastapi.responses import StreamingResponse

from server.config import AppSettings
from server.endpointregistry import EndpointRegistry
from server.websockets.models import InfraInfo, Models, Usages
from server.websockets.parent_infra import ParentInfra

logger = logging.getLogger("uvicorn.error")

T = TypeVar("T")


class UsageManager:
    models_usage: Usages
    models_list: Models
    infra_infos: list[InfraInfo]

    def __init__(
        self,
        parent_infra: ParentInfra,
        config: AppSettings,
        endpoint_registry: EndpointRegistry,
    ):
        self.parent_infra = parent_infra
        self.parent_infra.usage_manager = self
        self.config = config
        self.models_usage = {}
        self.models_list = {}
        self.infra_infos = []
        self.endpoint_registry = endpoint_registry
        self.endpoint_registry.usage_manager = self

    def clear_urls(self, urls: list[str]) -> None:
        """Clear urls."""
        self._clear_infos(urls)
        self._clear_models(urls)
        self._clear_usage(urls)

    def get_key(self, url: str) -> str:
        """Get key for url."""
        return next((info.api_key.get_secret_value() for info in self.infra_infos if info.url == url), "")

    def update_infos(self, params: list[InfraInfo]) -> None:
        """Update infos."""
        self.infra_infos.extend(params)

    def _clear_infos(self, urls: list[str]) -> None:
        """Clear infos for specified infra urls."""
        new_infos: list[InfraInfo] = []

        # Create new list instead of removing element from old
        for info in self.infra_infos:
            if info.url not in urls:
                new_infos.append(info)

        self.infra_infos = new_infos

        debug_msg = f"Infra infos list after clear: {new_infos}"
        logger.debug(debug_msg)

    def get_models(self) -> Models:
        """Get models."""
        return self.models_list

    def load_models(self) -> None:
        """Load models."""
        all_models = self.endpoint_registry.list_models()
        models: dict[str, list[str]] = {}
        model_names: list[str] = []
        for model in all_models:
            if not models.get(model.type):
                models[model.type] = []

            models[model.type].append(model.id)
            model_names.append(model.id)

        self.models_list[self.config.url] = models
        for model in model_names:
            self.models_usage[model] = {self.config.url: 0}

    def _clear_models(self, urls: list[str]) -> None:
        """Clear models for specified infra urls."""
        urls_to_delete: list[str] = []
        for url, _ in self.models_list.items():
            if url in urls:
                urls_to_delete.append(url)

        for url in urls_to_delete:
            del self.models_list[url]

        debug_msg = f"Models list after clear: {self.models_list}"
        logger.debug(debug_msg)

    def update_models(self, params: Models, urls: list[str]) -> None:
        """Update models."""
        for url, type_and_models in params.items():
            urls.append(url)

            self.models_list[url] = type_and_models
            for _, models in type_and_models.items():
                for model in models:
                    if self.models_usage.get(model, {}).get(url):
                        continue

                    self.models_usage[model].update({url: 0})

        debug_msg = f"Models list after handle message: {self.models_list}"
        logger.debug(debug_msg)

        debug_msg = f"Models usage after handle message: {self.models_usage}"
        logger.debug(debug_msg)

    def get_usage(self, model: str) -> dict[str, int] | None:
        """Get usage for given model."""
        return self.models_usage.get(model)

    def _add_usage(self, model: str) -> None:
        """Add usage to specified model."""
        url: str = self.config.url
        new_usage: int
        new_usage = 1
        if (url_and_usage := self.models_usage.get(model)) and (usage := url_and_usage.get(url)):
            new_usage = usage + 1

        if self.models_usage.get(model):
            self.models_usage[model].update({url: new_usage})
        else:
            self.models_usage[model] = {url: new_usage}

        debug_msg = f"Add usage for local infra in model: {model} to level {new_usage}"
        logger.debug(debug_msg)

        self.parent_infra.send_usage(self.models_usage)

    def _remove_usage(self, model: str) -> None:
        """Remove usage to specified model."""
        url: str = self.config.url
        new_usage: int
        new_usage = 0

        if (url_and_usage := self.models_usage.get(model)) and (usage := url_and_usage.get(url)):
            new_usage = usage - 1

        if self.models_usage.get(model):
            self.models_usage[model].update({url: new_usage})

        debug_msg = f"Remove usage for local infra in model: {model} to level {new_usage}"
        logger.debug(debug_msg)

        self.parent_infra.send_usage(self.models_usage)

    def _clear_usage(self, urls: list[str]) -> None:
        """Clear usage for specified infra urls."""
        to_delete: list[tuple[str, str]] = []
        for model, url_and_usage in self.models_usage.items():
            for url, _ in url_and_usage.items():
                if url in urls:
                    to_delete.append((model, url))

        for model, url in to_delete:
            del self.models_usage[model][url]

            if self.models_usage.get(model, {}) == {}:
                del self.models_usage[model]

        debug_msg = f"Models usage after clear: {self.models_usage}"
        logger.debug(debug_msg)

    def update_usage(self, params: Usages) -> None:
        """Update usage."""
        debug_msg = f"Models usage before update usage: {self.models_usage}"
        logger.debug(debug_msg)

        for model, url_and_usage in params.items():
            if model not in self.models_usage:
                self.models_usage[model] = url_and_usage
            else:
                self.models_usage[model].update(url_and_usage)

        debug_msg = f"Models usage after update usage: {self.models_usage}"
        logger.debug(debug_msg)

    async def with_usage(self, model: str, func: Callable[[], Awaitable[T]]) -> T:
        """With usage."""
        self._add_usage(model)
        try:
            resp = await func()
        except Exception:
            self._remove_usage(model)
            raise

        if isinstance(resp, StreamingResponse):

            async def create_generator() -> AsyncGenerator[Any]:
                """Add usage for response."""
                try:
                    async for chunk in resp.body_iterator:
                        yield chunk
                finally:
                    self._remove_usage(model)

            return StreamingResponse(create_generator(), media_type=resp.media_type, status_code=resp.status_code, headers=resp.headers)  # pyright: ignore[reportReturnType]

        self._remove_usage(model)
        return resp
