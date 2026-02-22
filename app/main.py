"""
e-Arzuhal NLP Server
====================
A lightweight Named Entity Recognition (NER) microservice for Turkish text.
This service extracts entities (PERSON, MONEY, LOCATION, DATE, OBJECT_OR_PROPERTY)
using SpaCy + custom Regex rules.

NO classification logic - this server is strictly for entity extraction.
"""
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware

from app.models import ExtractionRequest, ExtractionResponse, HealthResponse, EntitiesSchema
from app.services.extractor import get_extractor, TurkishEntityExtractor


# --- Application Configuration ---
VERSION = "1.0.0"
SERVICE_NAME = "e-Arzuhal NLP Server"


# --- Lifespan Management ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.
    Loads the SpaCy model on startup for efficient reuse.
    """
    print(f"[STARTUP] {SERVICE_NAME} v{VERSION} starting...")
    
    # Initialize the extractor (loads SpaCy model)
    extractor = get_extractor()
    app.state.extractor = extractor
    
    if extractor.is_spacy_loaded:
        print("[STARTUP] SpaCy model loaded successfully.")
    else:
        print("[STARTUP] Running in regex-only mode (SpaCy not available).")
    
    print(f"[STARTUP] {SERVICE_NAME} is ready!")
    
    yield  # Application runs here
    
    # Cleanup on shutdown
    print(f"[SHUTDOWN] {SERVICE_NAME} shutting down...")


# --- FastAPI Application ---
app = FastAPI(
    title=SERVICE_NAME,
    description=(
        "Named Entity Recognition (NER) service for Turkish text. "
        "Extracts PERSON, MONEY, LOCATION, DATE, and OBJECT_OR_PROPERTY entities "
        "using SpaCy with the tr_core_news_md model, augmented by custom Regex rules."
    ),
    version=VERSION,
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)


# --- CORS Middleware ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # TODO: Restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Routes ---

@app.get("/", tags=["Root"])
async def root():
    """
    Root endpoint - service information.
    """
    return {
        "service": SERVICE_NAME,
        "version": VERSION,
        "description": "Turkish NER extraction service",
        "endpoints": {
            "extract": "POST /api/extract",
            "health": "GET /health",
            "docs": "GET /docs",
        },
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """
    Health check endpoint.
    Returns service status and whether SpaCy model is loaded.
    """
    extractor: TurkishEntityExtractor = getattr(app.state, "extractor", None)
    spacy_loaded = extractor.is_spacy_loaded if extractor else False
    
    return HealthResponse(
        status="healthy",
        version=VERSION,
        spacy_model_loaded=spacy_loaded,
    )


@app.post(
    "/api/extract",
    response_model=ExtractionResponse,
    tags=["NER"],
    summary="Extract named entities from Turkish text",
    description=(
        "Processes Turkish text and extracts named entities. "
        "Returns entities grouped by type: PERSON, MONEY, LOCATION, DATE, OBJECT_OR_PROPERTY."
    ),
)
async def extract_entities(request: ExtractionRequest):
    """
    Extract named entities from Turkish text.
    
    Args:
        request: ExtractionRequest containing the raw text.
        
    Returns:
        ExtractionResponse with raw_text and extracted entities.
        
    Raises:
        HTTPException 500: If extraction fails unexpectedly.
    """
    try:
        # Get the extractor from app state
        extractor: TurkishEntityExtractor = app.state.extractor
        
        # Perform entity extraction
        extracted = extractor.extract(request.text)
        
        # Build response
        return ExtractionResponse(
            raw_text=request.text,
            entities=EntitiesSchema(
                PERSON=extracted.get("PERSON", []),
                MONEY=extracted.get("MONEY", []),
                LOCATION=extracted.get("LOCATION", []),
                DATE=extracted.get("DATE", []),
                OBJECT_OR_PROPERTY=extracted.get("OBJECT_OR_PROPERTY", []),
            ),
        )
    
    except Exception as e:
        # Log the error (in production, use proper logging)
        print(f"[ERROR] Entity extraction failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Entity extraction failed: {str(e)}",
        )


# --- Main Entry Point ---
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )

