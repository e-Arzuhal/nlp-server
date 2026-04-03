from pydantic import BaseModel
from typing import Optional, List, Dict


class ExtractRequest(BaseModel):
    text: str


class ExtractResponse(BaseModel):
    # GraphRAG-compatible fields
    contract_type: Optional[str]
    extracted_entities: Dict[str, List[str]]  # spaCy format

    # Extra metadata (used by main-server only, not forwarded to GraphRAG)
    contract_type_confidence: float
    raw_text_length: int
    processing_time_ms: int
