"""
e-Arzuhal NLP Server - Turkish Entity Extractor Service
Uses SpaCy with tr_core_news_md model + custom Regex rules for robust NER.
"""
import re
from typing import Dict, List, Set
from dataclasses import dataclass, field


@dataclass
class TurkishEntityExtractor:
    """
    Named Entity Recognition for Turkish text.
    Uses SpaCy's tr_core_news_md model combined with custom Regex patterns
    for enhanced extraction of MONEY, DATE, and OBJECT_OR_PROPERTY entities.
    """
    
    nlp: object = field(default=None, repr=False)
    _spacy_loaded: bool = field(default=False, init=False)
    
    # --- REGEX PATTERNS ---
    
    # Money patterns: matches Turkish currency formats
    # Examples: 15.000 TL, 20.000,50 ₺, 1000 lira, 5.500,00 TL
    MONEY_PATTERN = re.compile(
        r'\b\d{1,3}(?:[.,]\d{3})*(?:[.,]\d{1,2})?\s*(?:TL|₺|tl|Tl|lira|Lira|LIRA|türk lirası|Türk Lirası)\b',
        re.IGNORECASE | re.UNICODE
    )
    
    # Date patterns: matches Turkish date expressions
    # Examples: 1 yıl, 6 ay, 3 hafta, 10 gün, 1 yıllığına, 6 aylık, 15 Ocak 2024
    DATE_PATTERN = re.compile(
        r'''
        (?:
            # Duration patterns: "1 yıl", "6 ay", "3 hafta", "10 gün"
            \b\d+\s*(?:yıl(?:lık|lığına)?|ay(?:lık|lığına)?|hafta(?:lık|lığına)?|gün(?:lük|lüğüne)?)\b
            |
            # Full date: "15 Ocak 2024", "1 Şubat 2023"
            \b\d{1,2}\s+(?:Ocak|Şubat|Mart|Nisan|Mayıs|Haziran|Temmuz|Ağustos|Eylül|Ekim|Kasım|Aralık)\s+\d{4}\b
            |
            # Numeric date: "15/01/2024", "15.01.2024", "2024-01-15"
            \b\d{1,2}[./]\d{1,2}[./]\d{2,4}\b
            |
            \b\d{4}[-]\d{1,2}[-]\d{1,2}\b
        )
        ''',
        re.IGNORECASE | re.VERBOSE | re.UNICODE
    )
    
    # Object/Property patterns: common Turkish property and object types
    OBJECT_PROPERTY_KEYWORDS = {
        # Real estate
        'ev', 'daire', 'apartman', 'villa', 'konut', 'residence', 'rezidans',
        'arsa', 'tarla', 'arazi', 'bina', 'işyeri', 'iş yeri', 'dükkan', 'mağaza',
        'ofis', 'büro', 'depo', 'garaj', 'otopark',
        # Vehicles
        'araç', 'araba', 'otomobil', 'motosiklet', 'motorsiklet', 'kamyon', 'minibüs',
        'otobüs', 'traktör', 'tekne', 'yat',
        # Financial
        'depozito', 'kapora', 'teminat', 'kefalet', 'ipotek', 'rehin',
        # General items
        'eşya', 'mobilya', 'beyaz eşya', 'elektronik', 'bilgisayar', 'telefon',
    }
    
    def __post_init__(self):
        """Initialize SpaCy model after dataclass initialization."""
        self._load_spacy_model()
    
    def _load_spacy_model(self) -> None:
        """Load SpaCy Turkish model. Falls back to regex-only if unavailable."""
        try:
            import spacy
            self.nlp = spacy.load("tr_core_news_trf")
            self._spacy_loaded = True
            print("[INFO] SpaCy tr_core_web_lg model loaded successfully.")
        except ImportError:
            print("[WARNING] SpaCy not installed. Using regex-only mode.")
            self._spacy_loaded = False
        except OSError:
            print("[WARNING] SpaCy model 'tr_core_web_lg' not found.")
            print("[INFO] Install it with: python -m spacy download tr_core_web_lg")
            print("[INFO] Falling back to regex-only mode.")
            self._spacy_loaded = False
    
    @property
    def is_spacy_loaded(self) -> bool:
        """Check if SpaCy model is loaded."""
        return self._spacy_loaded
    
    def extract(self, text: str) -> Dict[str, List[str]]:
        """
        Extract named entities from Turkish text.
        
        Args:
            text: Raw Turkish text to process.
            
        Returns:
            Dictionary with entity types as keys and lists of extracted entities as values.
            Keys: PERSON, MONEY, LOCATION, DATE, OBJECT_OR_PROPERTY
        """
        # Initialize result with empty lists for all entity types
        entities: Dict[str, Set[str]] = {
            "PERSON": set(),
            "MONEY": set(),
            "LOCATION": set(),
            "DATE": set(),
            "OBJECT_OR_PROPERTY": set(),
        }
        
        # Step 1: Extract using SpaCy if available
        if self._spacy_loaded and self.nlp:
            entities = self._extract_with_spacy(text, entities)
        
        # Step 2: Augment with Regex patterns (always run for robustness)
        entities = self._extract_with_regex(text, entities)
        
        # Convert sets to sorted lists for consistent output
        return {key: sorted(list(values)) for key, values in entities.items()}
    
    def _extract_with_spacy(
        self, text: str, entities: Dict[str, Set[str]]
    ) -> Dict[str, Set[str]]:
        """Extract entities using SpaCy NER."""
        doc = self.nlp(text)
        
        # SpaCy entity label mapping to our schema
        label_map = {
            "PER": "PERSON",
            "PERSON": "PERSON",
            "LOC": "LOCATION",
            "GPE": "LOCATION",  # Geo-Political Entity (countries, cities)
            "LOCATION": "LOCATION",
            "MONEY": "MONEY",
            "DATE": "DATE",
            "TIME": "DATE",
        }
        
        for ent in doc.ents:
            mapped_label = label_map.get(ent.label_)
            if mapped_label and mapped_label in entities:
                # Clean and normalize the entity text
                clean_text = self._clean_entity_text(ent.text)
                if clean_text:
                    entities[mapped_label].add(clean_text)
        
        return entities
    
    def _extract_with_regex(
        self, text: str, entities: Dict[str, Set[str]]
    ) -> Dict[str, Set[str]]:
        """Extract entities using Regex patterns."""
        
        # Extract MONEY
        for match in self.MONEY_PATTERN.finditer(text):
            clean_text = self._clean_entity_text(match.group())
            if clean_text:
                entities["MONEY"].add(clean_text)
        
        # Extract DATE
        for match in self.DATE_PATTERN.finditer(text):
            clean_text = self._clean_entity_text(match.group())
            if clean_text:
                entities["DATE"].add(clean_text)
        
        # Extract OBJECT_OR_PROPERTY using keyword matching
        text_lower = text.lower()
        for keyword in self.OBJECT_PROPERTY_KEYWORDS:
            # Use word boundary matching
            pattern = rf'\b{re.escape(keyword)}\w*\b'
            matches = re.findall(pattern, text_lower, re.UNICODE)
            for match in matches:
                # Get original case from text
                original = self._find_original_case(text, match)
                if original:
                    entities["OBJECT_OR_PROPERTY"].add(original)
        
        return entities
    
    def _clean_entity_text(self, text: str) -> str:
        """Clean and normalize entity text."""
        # Remove leading/trailing whitespace
        text = text.strip()
        # Remove Turkish suffixes that shouldn't be part of entity
        # e.g., "Antalya'daki" -> "Antalya"
        text = re.sub(r"[''`](?:daki|deki|dan|den|da|de|ya|ye|a|e)$", "", text)
        return text.strip()
    
    def _find_original_case(self, text: str, lowercase_match: str) -> str:
        """Find the original-case version of a lowercase match in text."""
        pattern = rf'\b{re.escape(lowercase_match)}\b'
        match = re.search(pattern, text, re.IGNORECASE | re.UNICODE)
        if match:
            return match.group()
        return lowercase_match


# Singleton instance for reuse
_extractor_instance: TurkishEntityExtractor | None = None


def get_extractor() -> TurkishEntityExtractor:
    """
    Get or create the singleton TurkishEntityExtractor instance.
    This ensures the SpaCy model is loaded only once.
    """
    global _extractor_instance
    if _extractor_instance is None:
        _extractor_instance = TurkishEntityExtractor()
    return _extractor_instance
