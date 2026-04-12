import os
from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from app.routers.extract import router
from app.routers.chat_intent import router as chat_intent_router

_debug = os.getenv("DEBUG", "true").lower() == "true"
_internal_api_key = os.getenv("INTERNAL_API_KEY", "")
_allowed_origins = [
    o.strip() for o in os.getenv(
        "ALLOWED_ORIGINS",
        "http://localhost:8080"
    ).split(",")
]

app = FastAPI(
    title="NLP Server",
    description="Turkish contract entity extraction",
    version="1.0.0",
    docs_url="/docs" if _debug else None,
    redoc_url="/redoc" if _debug else None,
)

# CORS — yalnızca main-server erişmeli
app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.middleware("http")
async def api_key_middleware(request: Request, call_next):
    """Internal API key kontrolü. INTERNAL_API_KEY set edilmemişse (dev) pas geçer."""
    if request.url.path in ("/health", "/"):
        return await call_next(request)
    if _internal_api_key and request.headers.get("X-Internal-API-Key") != _internal_api_key:
        return JSONResponse(status_code=401, content={"detail": "Geçersiz veya eksik API anahtarı"})
    return await call_next(request)

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
