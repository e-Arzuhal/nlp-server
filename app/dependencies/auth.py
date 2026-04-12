import logging
import os

from fastapi import Security, HTTPException
from fastapi.security.api_key import APIKeyHeader

logger = logging.getLogger(__name__)

_api_key_header = APIKeyHeader(name="X-Internal-API-Key", auto_error=False)


async def verify_internal_token(token: str = Security(_api_key_header)) -> None:
    expected = os.getenv("INTERNAL_API_KEY", "")
    if not expected:
        logger.error("auth_key_not_configured")
        raise HTTPException(status_code=500, detail="INTERNAL_API_KEY not configured on server")
    if token != expected:
        logger.warning("auth_token_invalid")
        raise HTTPException(status_code=401, detail="Unauthorized")
