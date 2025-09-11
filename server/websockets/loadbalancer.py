"""Loadbalancer."""

import logging
import sys

from server.websockets.models import InfraInfo

logger = logging.getLogger("uvicorn.error")


def get_key(url: str, infra_infos: list[InfraInfo]) -> str:
    """Get key for url."""
    return next((info.api_key.get_secret_value() for info in infra_infos if info.url == url), "")


def get_infra_url(model: str, models_usage: dict[str, dict[str, int]]) -> str:
    """Get infra url."""
    choosen_url: str = ""
    lowest_usage: int = sys.maxsize
    if model_data := models_usage.get(model):
        for url, usage in model_data.items():
            if usage < lowest_usage:
                lowest_usage = usage
                choosen_url = url

            if lowest_usage == 0:
                break

        return choosen_url

    return ""
