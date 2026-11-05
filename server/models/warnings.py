# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Simplito sp. z o.o.

"""Warning models."""

from datetime import datetime

from pydantic import BaseModel


class ServiceWarning(BaseModel):
    id: str
    created_at: datetime
    service_id: str
    instance: str | None = None
    model_id: str | None = None
    message: str


class ListWarningsOut(BaseModel):
    list: list[ServiceWarning]
    load_error: str | None = None
