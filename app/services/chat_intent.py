"""
Chatbot Intent Classification + PII Masking Service
Qwen 2 (Ollama) ile chatbot mesajlarının niyetini tespit eder ve kişisel bilgileri maskeler.
"""
import logging
import re
from typing import Dict, List, Tuple

from app.services import ollama_client

logger = logging.getLogger(__name__)

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


# ── PII Regex Patterns (module load zamanında bir kez derle) ──
TC_PATTERN = re.compile(r'\b\d{11}\b')
PHONE_PATTERN = re.compile(r'\b(?:0|\+90)?\s*(?:5\d{2})\s*\d{3}\s*\d{2}\s*\d{2}\b')
EMAIL_PATTERN = re.compile(r'\b[\w.+-]+@[\w-]+\.[\w.]+\b')
MONEY_PATTERN = re.compile(
    r'\b\d{1,3}(?:[.,]\d{3})*(?:\s*(?:TL|tl|₺|lira|dolar|euro|EUR|USD))\b',
    re.IGNORECASE,
)
# Büyük harfle başlayan ardışık 2+ kelime — basit isim tespiti
NAME_PATTERN = re.compile(
    r'\b([A-ZÇĞİÖŞÜ][a-zçğıöşü]+(?:\s+[A-ZÇĞİÖŞÜ][a-zçğıöşü]+)+)\b'
)

# Hukuki terimler ve kurum isimleri — kişi ismi değil
NAME_STOP_WORDS = frozenset({
    "Türk Borçlar", "Borçlar Kanunu", "Türk Ceza", "Türk Medeni",
    "Yargıtay Kararı", "Anayasa Mahkemesi",
})


async def classify_intent(message: str) -> Tuple[str, float]:
    """
    Qwen 2 ile mesajın niyetini sınıflandırır.
    UYARI: Bu fonksiyona sanitize edilmiş mesaj gönderilmelidir — ham PII LLM'e sızmamalı.
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
                logger.info("intent_classified", extra={"intent": intent, "confidence": 0.9})
                return intent, 0.9

        logger.info("intent_classified", extra={"intent": "GENERAL_HELP", "confidence": 0.5})
        return "GENERAL_HELP", 0.5
    except Exception:
        logger.error("intent_classification_failed", exc_info=True)
        return "GENERAL_HELP", 0.0


def _extract_basic_entities(text: str) -> Dict[str, List[str]]:
    """
    Basit regex tabanlı entity çıkarma (chatbot hızı için).
    Tam NER yerine sadece PII maskeleme için yeterli.
    """
    potential_names = NAME_PATTERN.findall(text)
    persons = [n for n in potential_names if n not in NAME_STOP_WORDS]
    result = {
        "TC": TC_PATTERN.findall(text),
        "MONEY": MONEY_PATTERN.findall(text),
        "PERSON": persons,
    }
    logger.debug("pii_entities_detected", extra={
        "tc_count": len(result["TC"]),
        "money_count": len(result["MONEY"]),
        "person_count": len(result["PERSON"]),
    })
    return result


def sanitize_message(message: str, entities: Dict[str, List[str]]) -> str:
    """
    Mesajdaki kişisel bilgileri placeholder ile değiştirir.
    """
    sanitized = message

    # Telefon — TC'den önce çalıştır, aksi halde 11 haneli telefon
    # numarası (örn. 05551234567) TC pattern tarafından yutuluyor.
    sanitized = PHONE_PATTERN.sub('[TELEFON]', sanitized)

    # TC Kimlik
    sanitized = TC_PATTERN.sub('[TC_KİMLİK]', sanitized)

    # E-posta
    sanitized = EMAIL_PATTERN.sub('[E_POSTA]', sanitized)

    # NER'den gelen PERSON entity'leri — kelime sınırı ile değiştir,
    # "Ali" gibi kısa adın "Alişan" içinde substring eşleşmesini engelle.
    persons = entities.get("PERSON", [])
    seen: set = set()
    idx = 0
    # Uzun isimleri önce değiştir ki "Ahmet Yılmaz" öncesi "Ahmet" yutulmasın
    for person in sorted(persons, key=len, reverse=True):
        if not person or len(person) <= 1 or person in seen:
            continue
        seen.add(person)
        idx += 1
        pattern = re.compile(r'\b' + re.escape(person) + r'\b')
        sanitized = pattern.sub(f'[KİŞİ_{idx}]', sanitized)

    return sanitized


def _mask_entities(entities: Dict[str, List[str]]) -> Dict[str, List[str]]:
    """
    `detected_entities` alanındaki ham PII'yi placeholder ile değiştirir.
    Yanıt dışarı çıktığında ham kişisel veri ifşa olmamalı — schema
    (Dict[str, List[str]]) korunur, sadece değerler maskelenir.
    """
    placeholder_map = {
        "TC": "[TC_KİMLİK]",
        "MONEY": "[TUTAR]",
        "PERSON": "[KİŞİ]",
        "PHONE": "[TELEFON]",
        "EMAIL": "[E_POSTA]",
    }
    masked: Dict[str, List[str]] = {}
    for key, values in entities.items():
        if not values:
            masked[key] = []
            continue
        tag = placeholder_map.get(key, f"[{key}]")
        masked[key] = [f"{tag}_{i}" for i, _ in enumerate(values, 1)]
    return masked


async def process_chat_intent(message: str) -> dict:
    """
    Ana fonksiyon: önce PII maskeleme, sonra sanitize edilmiş mesaj ile intent sınıflandırma.
    Ham PII asla Ollama'ya gönderilmez ve yanıtta ifşa edilmez.
    """
    logger.debug("chat_intent_start", extra={"message_length": len(message)})

    # 1. Basit NER — PII tespit
    entities = _extract_basic_entities(message)

    # 2. PII maskeleme — LLM'e gönderilmeden önce
    sanitized = sanitize_message(message, entities)

    # 3. Intent sınıflandırma — sanitize edilmiş mesaj ile (ham PII LLM'e sızmasın)
    intent, confidence = await classify_intent(sanitized)

    return {
        "intent": intent,
        "confidence": confidence,
        "sanitized_message": sanitized,
        "detected_entities": _mask_entities(entities),
    }
