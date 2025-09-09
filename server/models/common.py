"""Common types."""

from typing import Any

# from fastapi.responses import JSONResponse, StreamingResponse
# from pydantic import BaseModel

type FormFields = dict[str, Any]
type JsonSerializable = dict[str, Any]
type StarletteResponse = Any  # JsonSerializable | JSONResponse | BaseModel | StreamingResponse
