# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Simplito sp. z o.o.

"""The one error type provisioning raises, and scrubbing credentials out of text."""

import os
import re

# Env vars the Job gets credentials through (ADMIN_PASSWORD, DF_MONGO_PASSWORD, DF_INFRA_*_KEY,
# backend API keys). Matched by name so a credential added later is covered too.
_SECRET_ENV = re.compile(r"PASSWORD|KEY|TOKEN|SECRET")
_MIN_SECRET_LEN = 4


class ProvisionError(Exception):
    """A step failed. The message says which, and never contains a credential."""


def redact(text: str) -> str:
    """Replace every credential the Job holds in its environment with ***."""
    for name, value in os.environ.items():
        if _SECRET_ENV.search(name) and len(value) >= _MIN_SECRET_LEN:
            text = text.replace(value, "***")
    return text
