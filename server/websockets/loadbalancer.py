"""Loadbalancer."""

import logging
import sys
import urllib.parse
from typing import Literal, NamedTuple

from server.config import AppSettings
from server.endpointregistry import ProxyOptions
from server.websockets.usage_manager import UsageManager

logger = logging.getLogger("uvicorn.error")


class InfraConnectData(NamedTuple):
    url: str
    key: str

    def get_proxy_options(self, url_path: str) -> ProxyOptions:
        """Get proxy options."""
        return ProxyOptions(url=urllib.parse.urljoin(self.url, url_path), headers={"Authorization": f"Bearer {self.key}"})


class LoadBalancer:
    def __init__(
        self,
        config: AppSettings,
        usage_manager: UsageManager,
    ):
        self.config = config
        self.usage_manager = usage_manager

    def get_lazy_infra(self, model: str) -> InfraConnectData | Literal["internal"] | None:
        """Get most lazy (with least requests) infra data."""
        url = self._get_infra_url(model)
        if url is None:
            return None
        if url == self.config.url:
            return "internal"
        key = self.usage_manager.get_key(url)
        return InfraConnectData(url, key)

    def _get_infra_url(self, model: str) -> str | None:
        """Get infra url."""
        choosen_url: str | None = None
        lowest_usage: int = sys.maxsize
        if model_data := self.usage_manager.get_usage(model):
            for url, usage in model_data.items():
                if usage < lowest_usage:
                    lowest_usage = usage
                    choosen_url = url

                if lowest_usage == 0:
                    break

            return choosen_url

        return None
