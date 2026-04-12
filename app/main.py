from dotenv import load_dotenv
load_dotenv()

import logging
import os
import time

from fastapi import FastAPI, Request

from app.core.logging import setup_logging

setup_logging(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger(__name__)
logger.info("NLP server starting", extra={"log_level": os.getenv("LOG_LEVEL", "INFO")})

from app.routers.extract import router
from app.routers.chat_intent import router as chat_intent_router

app = FastAPI(
    title="NLP Server",
    description="Turkish contract entity extraction",
    version="1.0.0"
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    # Skip health/root probes to avoid log noise from periodic polling.
    if request.url.path in ("/health", "/"):
        return await call_next(request)
    start = time.time()
    response = await call_next(request)
    logger.info("http_request", extra={
        "method": request.method,
        "path": request.url.path,
        "status": response.status_code,
        "ms": int((time.time() - start) * 1000),
    })
    return response


app.include_router(router)
app.include_router(chat_intent_router)


@app.get("/")
def root():
    return {"status": "NLP Server running"}


@app.get("/health")
async def health():
    from app.services.ner_service import ner_service
    model_ok = await ner_service.health_check()
    return {
        "status": "ok" if model_ok else "ollama_unavailable",
        "model": ner_service.model_name,
        "model_loaded": model_ok
    }
