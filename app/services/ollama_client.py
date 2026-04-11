import logging
import os
import time

import httpx

logger = logging.getLogger(__name__)

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:7b-instruct-q4_K_M")
OLLAMA_TIMEOUT = float(os.getenv("OLLAMA_TIMEOUT", "60"))


async def generate(prompt: str, system: str, format_json: bool = True) -> str:
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "system": system,
        "stream": False,
    }
    if format_json:
        payload["format"] = "json"

    start = time.time()
    try:
        async with httpx.AsyncClient(timeout=OLLAMA_TIMEOUT) as client:
            resp = await client.post(f"{OLLAMA_BASE_URL}/api/generate", json=payload)
            resp.raise_for_status()
            logger.debug("ollama_generate", extra={
                "model": OLLAMA_MODEL,
                "format_json": format_json,
                "status": resp.status_code,
                "ms": int((time.time() - start) * 1000),
            })
            return resp.json()["response"]
    except Exception:
        logger.error("ollama_generate_failed", extra={
            "model": OLLAMA_MODEL,
            "ms": int((time.time() - start) * 1000),
        })
        raise


async def health_check() -> bool:
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(f"{OLLAMA_BASE_URL}/api/tags")
            resp.raise_for_status()
            models = resp.json().get("models", [])
            available = any(OLLAMA_MODEL in m.get("name", "") for m in models)
            logger.debug("ollama_health_check", extra={"available": available})
            return available
    except Exception:
        logger.error("ollama_health_check_failed")
        return False
