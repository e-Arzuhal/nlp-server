from fastapi import FastAPI
from app.routers.extract import router

app = FastAPI(
    title="NLP Server",
    description="Turkish contract entity extraction",
    version="1.0.0"
)

app.include_router(router)


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
