"""Common types."""

from typing import Any

from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

RequestBody = dict[str, Any]
FormFields = dict[str, Any]
JsonSerializable = dict[str, Any]
StarletteResponse = JsonSerializable | JSONResponse | BaseModel | StreamingResponse
