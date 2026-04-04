from fastapi import APIRouter
from models.schemas import ExtractRequest, ExtractResponse
from app.services.contract_classifier import classify_contract
from app.services.ner_service import ner_service
from app.services.postprocessor import extract_all
from app.services.spacy_mapper import to_spacy_format
import time

router = APIRouter(prefix="/api/v1")


@router.post("/extract", response_model=ExtractResponse)
async def extract_entities(request: ExtractRequest):
    start_time = time.time()
    text = request.text

    # Step 1: Contract type
    contract_type, type_confidence = classify_contract(text)

    # Step 2: BERT NER → PER, ORG, LOC
    bert_results = ner_service.extract(text)

    # Step 3: Regex → MONEY, DATE, CARDINAL, PERCENT
    regex_results = extract_all(text)

    # Step 4: Map to spaCy format for GraphRAG compatibility
    extracted_entities = to_spacy_format(bert_results, regex_results)

    elapsed_ms = int((time.time() - start_time) * 1000)

    return ExtractResponse(
        contract_type=contract_type,
        contract_type_confidence=type_confidence,
        extracted_entities=extracted_entities,
        raw_text_length=len(text),
        processing_time_ms=elapsed_ms
    )
