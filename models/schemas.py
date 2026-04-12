from pydantic import BaseModel, Field
from typing import Optional, List, Dict


class ExtractRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=10000)


class ExtractResponse(BaseModel):
    # GraphRAG-compatible fields
    contract_type: Optional[str]
    extracted_entities: Dict[str, List[str]]  # spaCy format

    # Extra metadata (used by main-server only, not forwarded to GraphRAG)
    contract_type_confidence: float
    raw_text_length: int
    processing_time_ms: int


# ── Chat Intent ──

class ChatIntentRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000)


class ChatIntentResponse(BaseModel):
    intent: str
    confidence: float
    sanitized_message: str
    detected_entities: Dict[str, List[str]]
