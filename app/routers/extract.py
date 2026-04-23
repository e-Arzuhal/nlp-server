import logging
import time

from fastapi import APIRouter, Depends
from models.schemas import ExtractRequest, ExtractResponse
from app.dependencies.auth import verify_internal_token
from app.services.contract_classifier import classify_contract
from app.services.ner_service import ner_service
from app.services.postprocessor import extract_all
from app.services.entity_merger import merge_entities
from app.services.classifier_eval import get_cached_metrics

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1")


@router.post("/extract", response_model=ExtractResponse, dependencies=[Depends(verify_internal_token)])
async def extract_entities(request: ExtractRequest):
    start_time = time.time()
    text = request.text

    # Step 1: Contract type
    contract_type, type_confidence = classify_contract(text)
    logger.debug("pipeline_step", extra={"step": "contract_classify", "result": contract_type})

    # Step 2: LLM NER → all 7 entity types
    llm_results = await ner_service.extract(text)
    logger.debug("pipeline_step", extra={
        "step": "ner_extract",
        "entity_count": sum(len(v) for v in llm_results.values()),
    })

    # Step 3: Regex safety net → MONEY, DATE, CARDINAL, PERCENT
    regex_results = extract_all(text)
    logger.debug("pipeline_step", extra={"step": "regex_extract"})

    # Step 4: Merge LLM + regex results
    extracted_entities = merge_entities(llm_results, regex_results)
    logger.debug("pipeline_step", extra={"step": "merge"})

    elapsed_ms = int((time.time() - start_time) * 1000)

    logger.info("extraction_complete", extra={
        "contract_type": contract_type,
        "processing_time_ms": elapsed_ms,
        "total_entities": sum(len(v) for v in extracted_entities.values()),
    })

    return ExtractResponse(
        contract_type=contract_type,
        contract_type_confidence=type_confidence,
        extracted_entities=extracted_entities,
        raw_text_length=len(text),
        processing_time_ms=elapsed_ms
    )


@router.get("/metrics")
def get_metrics():
    """Classifier precision / recall / F1 on curated Turkish legal test set."""
    return get_cached_metrics()
