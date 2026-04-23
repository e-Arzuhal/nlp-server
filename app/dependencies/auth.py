import logging
import os

from fastapi import Security, HTTPException
from fastapi.security.api_key import APIKeyHeader

logger = logging.getLogger(__name__)

_api_key_header = APIKeyHeader(name="X-Internal-API-Key", auto_error=False)


async def verify_internal_token(token: str = Security(_api_key_header)) -> None:
    expected = os.getenv("INTERNAL_API_KEY", "")
    _debug = os.getenv("DEBUG", "false").lower() == "true"
    if not expected:
        if _debug:
            logger.warning("auth_key_not_configured_bypass — only allowed in debug mode")
            return
        raise HTTPException(status_code=503, detail="Server misconfigured: INTERNAL_API_KEY is required")
    if token != expected:
        logger.warning("auth_token_invalid")
        raise HTTPException(status_code=401, detail="Unauthorized")
