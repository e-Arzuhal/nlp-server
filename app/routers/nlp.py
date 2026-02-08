"""
e-Arzuhal NLP Server - API Routes
"""
from fastapi import APIRouter, HTTPException

from app.models.schemas import (
    AnalyzeRequest,
    AnalyzeResponse,
    ClassifyRequest,
    ClassifyResponse,
    ExtractEntitiesRequest,
    ExtractEntitiesResponse,
    Entity,
)
from app.services.analyzer import get_analyzer
from app.services.contract_classifier import get_contract_classifier
from app.services.entity_extractor import get_entity_extractor


router = APIRouter(prefix="/api/nlp", tags=["NLP"])


@router.post("/analyze", response_model=AnalyzeResponse)
async def analyze_text(request: AnalyzeRequest):
    """
    Metni analiz et - sozlesme tipi + entity extraction + oneriler
    
    Main endpoint - frontend buraya istek atar
    """
    try:
        analyzer = get_analyzer()
        result = analyzer.analyze(request.text)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/classify", response_model=ClassifyResponse)
async def classify_contract_type(request: ClassifyRequest):
    """
    Sadece sozlesme tipi siniflandirmasi
    """
    try:
        classifier = get_contract_classifier()
        contract_type, confidence, all_scores = classifier.predict(request.text)
        
        return ClassifyResponse(
            contract_type=contract_type,
            confidence=confidence,
            all_scores=all_scores,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/entities", response_model=ExtractEntitiesResponse)
async def extract_entities(request: ExtractEntitiesRequest):
    """
    Sadece entity extraction
    """
    try:
        extractor = get_entity_extractor()
        result = extractor.extract(request.text)
        
        entities = [
            Entity(
                text=e["text"],
                label=e["label"],
                start=e["start"],
                end=e["end"],
                mapped_field=e["mapped_field"],
            )
            for e in result["entities"]
        ]
        
        return ExtractEntitiesResponse(
            entities=entities,
            extracted_fields=result["extracted_fields"],
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
