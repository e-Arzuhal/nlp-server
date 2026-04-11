"""
Chatbot Intent Classification + PII Masking Service
Qwen 2 (Ollama) ile chatbot mesajlarının niyetini tespit eder ve kişisel bilgileri maskeler.
"""
import re
from typing import Dict, List, Optional, Tuple

from app.services import ollama_client

INTENTS = [
    "CONTRACT_CLAUSE_QUESTION",
    "MISSING_CLAUSE_QUESTION",
    "LEGAL_QUESTION",
    "LAW_REFERENCE",
    "GENERAL_HELP",
]

INTENT_SYSTEM_PROMPT = """Sen bir Turkce metin niyet siniflandirma uzmanisin.
Kullanicinin chatbot'a yazdigi mesajin niyetini tespit edeceksin.

NIYET TURLERI:
- CONTRACT_CLAUSE_QUESTION: Olusturulan sozlesme maddeleri hakkinda soru. Ornek: "Bu sozlesmede hangi maddeler var?", "Sozlesme icerigi ne?"
- MISSING_CLAUSE_QUESTION: Eksik maddeler hakkinda soru. Ornek: "Eksik maddeler neler?", "Baska neler eklenmeli?", "Sozlesmede ne eksik?"
- LEGAL_QUESTION: Genel hukuki soru. Ornek: "Kiracinin haklari neler?", "Fesih nasil yapilir?", "Borc sozlesmesinde faiz siniri ne?"
- LAW_REFERENCE: Kanun maddeleri, TBK/HMK referanslari. Ornek: "TBK 299 ne der?", "Kanun maddesi nedir?", "Yasal dayanak ne?"
- GENERAL_HELP: Uygulama kullanimi, SSS. Ornek: "PDF nasil indirilir?", "Sozlesme nasil olusturulur?", "Nasil kayit olurum?"

KURALLAR:
1. Sadece niyet turunu don: CONTRACT_CLAUSE_QUESTION, MISSING_CLAUSE_QUESTION, LEGAL_QUESTION, LAW_REFERENCE veya GENERAL_HELP
2. Emin degilsen GENERAL_HELP don.
3. Sadece niyet ismini yaz, baska bir sey yazma.

Ornek:
Mesaj: "Bu sozlesmede cezai sart maddesi var mi?"
Yanit: CONTRACT_CLAUSE_QUESTION

Mesaj: "PDF nasil indirilir?"
Yanit: GENERAL_HELP"""


# ── PII Regex Patterns ──
TC_PATTERN = re.compile(r'\b\d{11}\b')
PHONE_PATTERN = re.compile(r'\b(?:0|\+90)?\s*(?:5\d{2})\s*\d{3}\s*\d{2}\s*\d{2}\b')
EMAIL_PATTERN = re.compile(r'\b[\w.+-]+@[\w-]+\.[\w.]+\b')


async def classify_intent(message: str) -> Tuple[str, float]:
    """
    Qwen 2 ile mesajın niyetini sınıflandırır.
    Returns: (intent, confidence)
    """
    try:
        raw = await ollama_client.generate(
            prompt=f"Mesaj: \"{message}\"",
            system=INTENT_SYSTEM_PROMPT,
            format_json=False,
        )
        result = raw.strip().upper().replace(" ", "_")

        # Bilinen intent'lerden birine eşle
        for intent in INTENTS:
            if intent in result:
                return intent, 0.9

        return "GENERAL_HELP", 0.5
    except Exception as e:
        print(f"Intent classification failed: {e}")
        return "GENERAL_HELP", 0.0


def sanitize_message(message: str, entities: Dict[str, List[str]]) -> str:
    """
    Mesajdaki kişisel bilgileri placeholder ile değiştirir.
    """
    sanitized = message

    # TC Kimlik
    sanitized = TC_PATTERN.sub('[TC_KİMLİK]', sanitized)

    # Telefon
    sanitized = PHONE_PATTERN.sub('[TELEFON]', sanitized)

    # E-posta
    sanitized = EMAIL_PATTERN.sub('[E_POSTA]', sanitized)

    # NER'den gelen PERSON entity'leri
    persons = entities.get("PERSON", [])
    for i, person in enumerate(persons, 1):
        if person and len(person) > 1:
            sanitized = sanitized.replace(person, f'[KİŞİ_{i}]')

    return sanitized


async def process_chat_intent(message: str) -> dict:
    """
    Ana fonksiyon: intent sınıflandırma + PII maskeleme.
    """
    # 1. Basit NER — PERSON çıkarma (hızlı, Ollama'ya gerek yok)
    entities = _extract_basic_entities(message)

    # 2. Intent sınıflandırma (Qwen 2 ile)
    intent, confidence = await classify_intent(message)

    # 3. PII maskeleme
    sanitized = sanitize_message(message, entities)

    return {
        "intent": intent,
        "confidence": confidence,
        "sanitized_message": sanitized,
        "detected_entities": entities,
    }


def _extract_basic_entities(text: str) -> Dict[str, List[str]]:
    """
    Basit regex tabanlı entity çıkarma (chatbot hızı için).
    Tam NER yerine sadece PII maskeleme için yeterli.
    """
    entities: Dict[str, List[str]] = {"PERSON": [], "MONEY": [], "TC": []}

    # TC Kimlik
    entities["TC"] = TC_PATTERN.findall(text)

    # Para tutarları
    money_pattern = re.compile(
        r'\b\d{1,3}(?:[.,]\d{3})*(?:\s*(?:TL|tl|₺|lira|dolar|euro|EUR|USD))\b',
        re.IGNORECASE
    )
    entities["MONEY"] = money_pattern.findall(text)

    # Basit isim tespiti — büyük harfle başlayan ardışık kelimeler
    # (Tam NER değil ama PII maskeleme için yeterli)
    name_pattern = re.compile(r'\b([A-ZÇĞİÖŞÜ][a-zçğıöşü]+(?:\s+[A-ZÇĞİÖŞÜ][a-zçğıöşü]+)+)\b')
    potential_names = name_pattern.findall(text)

    # Türkçe genel kelimeler filtrele
    stop_words = {
        "Türk Borçlar", "Borçlar Kanunu", "Türk Ceza", "Türk Medeni",
        "Yargıtay Kararı", "Anayasa Mahkemesi",
    }
    entities["PERSON"] = [n for n in potential_names if n not in stop_words]

    return entities
