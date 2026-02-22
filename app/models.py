"""
e-Arzuhal NLP Server - Pydantic Models
Named Entity Recognition (NER) Request/Response schemas.
"""
from pydantic import BaseModel, Field
from typing import List


class EntitiesSchema(BaseModel):
    """
    Extracted entities grouped by type.
    All keys are always present; empty list if none found.
    """
    PERSON: List[str] = Field(default_factory=list, description="Kişi isimleri (Person names)")
    MONEY: List[str] = Field(default_factory=list, description="Para miktarları (Monetary amounts)")
    LOCATION: List[str] = Field(default_factory=list, description="Yer/konum isimleri (Locations)")
    DATE: List[str] = Field(default_factory=list, description="Tarih ifadeleri (Date expressions)")
    OBJECT_OR_PROPERTY: List[str] = Field(
        default_factory=list,
        description="Nesne veya mülk türleri (Objects or property types)"
    )


class ExtractionRequest(BaseModel):
    """
    Request schema for entity extraction.
    Contains the raw Turkish text to be processed.
    """
    text: str = Field(
        ...,
        min_length=1,
        max_length=10000,
        description="İşlenecek Türkçe metin (Turkish text to process)"
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "text": "Ahmet Yılmaz'a Antalya'daki evimi aylık 15.000 TL'ye 1 yıllığına kiralayacağım. 20.000 TL depozito alacağım."
            }
        }
    }


class ExtractionResponse(BaseModel):
    """
    Response schema for entity extraction.
    Contains the original text and extracted entities.
    """
    raw_text: str = Field(..., description="Orijinal metin (Original input text)")
    entities: EntitiesSchema = Field(
        default_factory=EntitiesSchema,
        description="Çıkarılan varlıklar (Extracted entities grouped by type)"
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "raw_text": "Ahmet Yılmaz'a Antalya'daki evimi aylık 15.000 TL'ye 1 yıllığına kiralayacağım. 20.000 TL depozito alacağım.",
                "entities": {
                    "PERSON": ["Ahmet Yılmaz"],
                    "MONEY": ["15.000 TL", "20.000 TL"],
                    "LOCATION": ["Antalya"],
                    "DATE": ["1 yıllığına"],
                    "OBJECT_OR_PROPERTY": ["ev", "depozito"]
                }
            }
        }
    }


class HealthResponse(BaseModel):
    """Health check response schema."""
    status: str = Field(..., description="Servis durumu (Service status)")
    version: str = Field(..., description="Servis versiyonu (Service version)")
    spacy_model_loaded: bool = Field(..., description="SpaCy modeli yüklü mü (Is SpaCy model loaded)")
