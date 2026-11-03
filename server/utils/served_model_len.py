# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Simplito sp. z o.o.

"""Ask a running vLLM/SGLang server how many tokens it accepts for a model."""

import json

from server.utils.core import fetch_from
from server.utils.logger import logger

_MODELS_TIMEOUT_SECONDS = 10


async def fetch_served_max_model_len(base_url: str, served_model_name: str) -> int | None:
    """Return `max_model_len` that `GET /v1/models` reports for `served_model_name`, or None if it can't tell."""
    try:
        result = await fetch_from(f"{base_url}/v1/models", timeout=_MODELS_TIMEOUT_SECONDS)
        if result.status_code != 200:
            logger.warning("%s/v1/models returned status %d", base_url, result.status_code)
            return None
        cards = json.loads(result.data).get("data") or []
    except Exception:
        logger.warning("Fetching %s/v1/models failed", base_url, exc_info=True)
        return None
    if not isinstance(cards, list):
        return None
    for card in cards:
        if isinstance(card, dict) and card.get("id") == served_model_name:
            value = card.get("max_model_len")
            return value if isinstance(value, int) and value > 0 else None
    return None
