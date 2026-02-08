"""
e-Arzuhal NLP Server - Pydantic Schemas
Request/Response modelleri
"""
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from enum import Enum


class ContractType(str, Enum):
    BORC = "borc_sozlesmesi"
    KIRA = "kira_sozlesmesi"
    HIZMET = "hizmet_sozlesmesi"
    SATIS = "satis_sozlesmesi"
    IS = "is_sozlesmesi"
    VEKALETNAME = "vekaletname"
    TAAHHUTNAME = "taahhutname"


class Entity(BaseModel):
    """Cikarilan entity"""
    text: str = Field(..., description="Entity metni")
    label: str = Field(..., description="Entity tipi (PERSON, MONEY, DATE, vb.)")
    start: int = Field(..., description="Baslangic pozisyonu")
    end: int = Field(..., description="Bitis pozisyonu")
    mapped_field: Optional[str] = Field(None, description="Sozlesme alanina mapping (taraf, tutar, vb.)")


class AnalyzeRequest(BaseModel):
    """Metin analiz istegi"""
    text: str = Field(..., min_length=10, max_length=5000, description="Analiz edilecek metin")
    
    class Config:
        json_schema_extra = {
            "example": {
                "text": "Ahmet Yilmaz'a 50.000 TL borc verecegim, 6 ay icinde geri odeyecek."
            }
        }


class AnalyzeResponse(BaseModel):
    """Metin analiz sonucu"""
    success: bool = True
    contract_type: str = Field(..., description="Tespit edilen sozlesme tipi")
    confidence: float = Field(..., ge=0, le=1, description="Guven skoru (0-1)")
    entities: List[Entity] = Field(default_factory=list, description="Cikarilan entityler")
    extracted_fields: Dict[str, Any] = Field(default_factory=dict, description="Yapilandirilmis alanlar")
    suggestions: List[str] = Field(default_factory=list, description="Eksik alan onerileri")
    raw_text: str = Field(..., description="Orijinal metin")


class ClassifyRequest(BaseModel):
    """Sadece sozlesme tipi siniflandirma istegi"""
    text: str = Field(..., min_length=10, max_length=5000)


class ClassifyResponse(BaseModel):
    """Siniflandirma sonucu"""
    contract_type: str
    confidence: float
    all_scores: Dict[str, float] = Field(default_factory=dict, description="Tum tipler icin skorlar")


class ExtractEntitiesRequest(BaseModel):
    """Entity cikarma istegi"""
    text: str = Field(..., min_length=5, max_length=5000)


class ExtractEntitiesResponse(BaseModel):
    """Entity cikarma sonucu"""
    entities: List[Entity]
    extracted_fields: Dict[str, Any]


class HealthResponse(BaseModel):
    """Saglik kontrolu"""
    status: str = "healthy"
    version: str = "0.1.0"
    models_loaded: bool = True
