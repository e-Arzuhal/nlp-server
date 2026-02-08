"""
e-Arzuhal NLP Server
FastAPI uygulama giris noktasi
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import HOST, PORT, DEBUG
from app.routers import nlp
from app.models.schemas import HealthResponse


# FastAPI app
app = FastAPI(
    title="e-Arzuhal NLP Server",
    description="Dogal dil isleme servisi - sozlesme tipi siniflandirma ve entity extraction",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Production'da kisitlanmali
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routers
app.include_router(nlp.router)


@app.get("/", tags=["Root"])
async def root():
    """API root"""
    return {
        "service": "e-Arzuhal NLP Server",
        "version": "0.1.0",
        "docs": "/docs",
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Saglik kontrolu"""
    return HealthResponse(
        status="healthy",
        version="0.1.0",
        models_loaded=bool(getattr(app.state, "models_loaded", False)),
    )


# Startup event
@app.on_event("startup")
async def startup_event():
    """Uygulama baslarken modelleri yukle"""
    print("NLP Server baslatiliyor...")
    app.state.models_loaded = False

    from app.services.contract_classifier import get_contract_classifier
    from app.services.entity_extractor import get_entity_extractor

    # Classifier her durumda
    get_contract_classifier()

    # Entity extractor: spaCy varsa spacy mod; yoksa regex/lite mod
    extractor = get_entity_extractor()
    app.state.entity_mode = getattr(extractor, "mode", "unknown")

    app.state.models_loaded = True
    print(f"NLP Server hazir - http://{HOST}:{PORT} (entity_mode={app.state.entity_mode})")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host=HOST, port=PORT, reload=DEBUG)
