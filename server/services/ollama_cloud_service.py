# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Simplito sp. z o.o.

"""Ollama Cloud service."""

import asyncio
import json
import time

import aiohttp
from fastapi import HTTPException

from server.models.models import CustomModelSpecification, InstallModelIn, InstallModelOut, ListModelsFilters, ListModelsOut
from server.services.base2_service import CustomModel, InstanceConfig
from server.services.remote_service import DefaultRemoteServiceOptions, RemoteConst, RemoteModel, RemoteService
from server.utils.core import PromiseWithProgress, StreamChunk

MODELS_TTL = 3600.0


class OllamaCloudService(RemoteService):
    is_cloud = True
    options_class = DefaultRemoteServiceOptions
    _models_cache_time: float

    def _after_init(self) -> None:
        super()._after_init()
        self._models_cache_time = float("-inf")

    def get_type(self) -> str:
        """Return the service type."""
        return "ollama-cloud"

    def get_description(self) -> str:
        """Return the service description."""
        return "Remote access to Ollama cloud models via ollama.com."

    def get_default_url(self) -> str:
        """Return the default API URL."""
        return "https://ollama.com"

    def get_custom_model_spec(self) -> CustomModelSpecification | None:
        """Return None — custom models are not supported, only models listed by ollama.com /api/tags are available."""
        return None

    def _add_custom_model(self, instance: str, model: CustomModel) -> None:  # noqa: ARG002
        """Reject custom models."""
        raise HTTPException(400, "This service does not support custom models.")

    async def load_instance(self, instance: str, instance_data: InstanceConfig) -> None:
        """Load instance, dropping custom models left in the config from before they were unsupported."""
        await super().load_instance(instance, instance_data.model_copy(update={"custom": None}))

    def get_models_registry(self) -> RemoteConst:
        """Return empty registry — models are fetched dynamically."""
        return RemoteConst(models={})

    async def _fetch_models_from_api(self, instance: str) -> dict[str, RemoteModel] | None:
        """Fetch available models from ollama.com /api/tags, typing those with the `decision` capability as decision models.

        Returns None on error or empty list.
        """
        info = self.get_instance_installed_info(instance)
        api_url = info.parsed_options.api_url
        api_key = info.parsed_options.api_key
        headers = {"Authorization": f"Bearer {api_key}"}

        try:
            async with aiohttp.ClientSession() as session, session.get(f"{api_url}/api/tags", headers=headers) as response:
                if response.status != 200:
                    return None
                body = json.loads(await response.text())
                models_raw = body.get("models", [])
                if not models_raw:
                    return None
                return {
                    m["name"]: RemoteModel(type="decision" if "decision" in (m.get("capabilities") or []) else "llm")
                    for m in models_raw
                    if m.get("name")
                }
        except Exception:
            return None

    def _cleanup_removed_models(self, instance: str, new_models: dict[str, RemoteModel]) -> None:
        """Unregister installed models that are no longer in the new model list."""
        info = self.get_instance_installed_info(instance)
        removed = [model for model_id, model in info.models.items() if model_id not in new_models]
        for model in removed:
            if model.type == "llm":
                self.endpoint_registry.unregister_chat_completion(model.registered_name, model.registration_id)
            elif model.type == "tts":
                self.endpoint_registry.unregister_audio_speech(model.registered_name, model.registration_id)
            elif model.type == "stt":
                self.endpoint_registry.unregister_audio_transcriptions(model.registered_name, model.registration_id)
            elif model.type == "txt2img":
                self.endpoint_registry.unregister_image_generations(model.registered_name, model.registration_id)
            elif model.type == "embedding":
                self.endpoint_registry.unregister_embeddings(model.registered_name, model.registration_id)
            elif model.type == "decision":
                self.endpoint_registry.unregister_systemone(model.registered_name, model.registration_id)
            del info.models[model.id]

    async def _refresh_models(self, instance: str, *, force: bool = False) -> None:
        """Refresh model list from API if TTL expired or force=True."""
        now = time.monotonic()
        if not force and (now - self._models_cache_time) < MODELS_TTL:
            return
        new_models = await self._fetch_models_from_api(instance)
        if new_models is None:
            return
        self._cleanup_removed_models(instance, new_models)
        self.models[instance] = new_models
        self._models_cache_time = now

    async def list_models(self, input_instance: str | list[str] | None, filters: ListModelsFilters) -> ListModelsOut:
        """List models, refreshing from API if TTL expired."""
        instances = [input_instance] if isinstance(input_instance, str) else input_instance if input_instance else list(self.instances_info)
        await asyncio.gather(*[self._refresh_models(inst) for inst in instances if self.get_instance_info(inst).installed])
        return await super().list_models(input_instance, filters)

    async def _install_model(
        self,
        instance: str,
        model_id: str,
        options: InstallModelIn,
        container_name_override: str | None = None,  # noqa: ARG002
    ) -> PromiseWithProgress[InstallModelOut, StreamChunk]:
        """Install model, doing a force refresh if model is not in current cache."""
        if not self.models.get(instance) or model_id not in self.models[instance]:
            await self._refresh_models(instance, force=True)
        return await super()._install_model(instance, model_id, options)
