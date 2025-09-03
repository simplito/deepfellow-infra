"""Main app module."""

from typing import Annotated

from fastapi import Depends, HTTPException, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

oauth2_scheme = HTTPBearer()


def auth_server(request: Request, api_key: Annotated[HTTPAuthorizationCredentials, Depends(oauth2_scheme)]) -> str:
    """Authenticate an server key."""
    if api_key.credentials == request.app.state.context.config.api_key:
        return api_key.credentials

    raise HTTPException(status_code=401, detail="Unauthorized")


def auth_admin(request: Request, api_key: Annotated[HTTPAuthorizationCredentials, Depends(oauth2_scheme)]) -> str:
    """Authenticate administrator."""
    if api_key.credentials == request.app.state.context.config.admin_api_key:
        return api_key.credentials

    raise HTTPException(status_code=401, detail="Unauthorized")
