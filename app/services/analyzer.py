"""
e-Arzuhal NLP Server - Text Analyzer
Metin analiz servisi - classifier + entity extractor birlesimi
"""
from typing import Dict, Any, List

from app.services.contract_classifier import get_contract_classifier
from app.services.entity_extractor import get_entity_extractor
from app.models.schemas import AnalyzeResponse, Entity


class TextAnalyzer:
    """
    Ana metin analiz servisi
    Contract classification + Entity extraction + Suggestions
    """
    
    # Sozlesme tiplerine gore zorunlu alanlar
    REQUIRED_FIELDS = {
        "borc_sozlesmesi": ["taraflar", "tutar", "tarih"],
        "kira_sozlesmesi": ["taraflar", "tutar", "tarih", "lokasyon"],
        "hizmet_sozlesmesi": ["taraflar", "tutar"],
        "satis_sozlesmesi": ["taraflar", "tutar"],
        "is_sozlesmesi": ["taraflar", "tutar"],
        "vekaletname": ["taraflar"],
        "taahhutname": ["taraflar"],
    }
    
    # Turkce alan isimleri
    FIELD_NAMES_TR = {
        "taraflar": "Taraflar (alacakli/borclu, kiraci/ev sahibi vb.)",
        "tutar": "Tutar/Bedel",
        "tarih": "Tarih/Vade/Sure",
        "lokasyon": "Adres/Konum",
        "kurum": "Sirket/Kurum",
    }
    
    def __init__(self):
        self.classifier = get_contract_classifier()
        self.extractor = get_entity_extractor()
    
    def analyze(self, text: str) -> AnalyzeResponse:
        """
        Metni analiz et
        
        Args:
            text: Analiz edilecek metin
            
        Returns:
            AnalyzeResponse
        """
        # 1. Sozlesme tipi siniflandirma
        contract_type, confidence, all_scores = self.classifier.predict(text)
        
        # 2. Entity extraction
        extraction_result = self.extractor.extract(text)
        entities = extraction_result["entities"]
        extracted_fields = extraction_result["extracted_fields"]
        
        # 3. Eksik alan kontrolu ve oneriler
        suggestions = self._generate_suggestions(contract_type, extracted_fields)
        
        # 4. Entity'leri schema formatina cevir
        entity_objects = [
            Entity(
                text=e["text"],
                label=e["label"],
                start=e["start"],
                end=e["end"],
                mapped_field=e["mapped_field"],
            )
            for e in entities
        ]
        
        return AnalyzeResponse(
            success=True,
            contract_type=contract_type,
            confidence=confidence,
            entities=entity_objects,
            extracted_fields=extracted_fields,
            suggestions=suggestions,
            raw_text=text,
        )
    
    def _generate_suggestions(self, contract_type: str, extracted_fields: Dict[str, Any]) -> List[str]:
        """Eksik alanlara gore oneriler olustur"""
        suggestions = []
        required = self.REQUIRED_FIELDS.get(contract_type, [])
        
        for field in required:
            value = extracted_fields.get(field)
            
            # Bos mu kontrol et
            is_empty = (
                value is None or
                value == "" or
                (isinstance(value, list) and len(value) == 0)
            )
            
            if is_empty:
                field_name = self.FIELD_NAMES_TR.get(field, field)
                suggestions.append(f"{field_name} belirtilmemis. Lutfen ekleyin.")
        
        # Ozel oneriler
        if contract_type == "borc_sozlesmesi":
            if "faiz" not in extracted_fields or not extracted_fields.get("faiz"):
                suggestions.append("Faiz orani belirtmek ister misiniz? (Kullanicilarin %78'i ekliyor)")
        
        if contract_type == "kira_sozlesmesi":
            suggestions.append("Depozito tutari eklemek ister misiniz?")
        
        return suggestions


# Singleton
_analyzer = None


def get_analyzer() -> TextAnalyzer:
    """Text analyzer singleton"""
    global _analyzer
    if _analyzer is None:
        _analyzer = TextAnalyzer()
    return _analyzer
