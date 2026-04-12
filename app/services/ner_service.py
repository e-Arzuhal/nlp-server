import json
import logging
import re
from typing import Dict, List

from app.services import ollama_client

logger = logging.getLogger(__name__)

ENTITY_KEYS = ["PERSON", "ORG", "LOC", "DATE", "MONEY", "CARDINAL", "PERCENT"]

SYSTEM_PROMPT = """Sen bir Turkce sozlesme metni analiz uzmanisin.
Verilen metinden varliklari cikarip JSON olarak doneceksin.

VARLIK TURLERI:
- PERSON: Kisi isimleri. Metinde gectigi sekilde yaz.
- ORG: Kurum, sirket, devlet kurumu isimleri.
- LOC: Sehir, ilce, mahalle, adres gibi yer isimleri. Her yeri AYRI AYRI yaz, birlestirme.
- DATE: Tarihler ve tekrar eden tarih ifadeleri. "her ayin 10u", "aylik", "yillik" gibi ifadeler DATE DEGILDIR.
- MONEY: Para tutarlari, birim ile birlikte. Metinde nasil yazildiysa oyle yaz (ornek: "15000 tl", "25.000 TL").
- CARDINAL: SADECE sure ifadeleri: "X yil", "X ay", "X gun", "X hafta" gibi. Baska sayilari ekleme.
- PERCENT: Yuzde ifadeleri: "%25", "yuzde 10" gibi.

KURALLAR:
1. SADECE metinde gecen varliklari cikar. Uydurma, tahmin etme.
2. Varliklari metinde AYNEN gectigi sekilde yaz. Yazim duzeltme yapma. "ahmet yilmaza" yaziyorsa "ahmet yilmaz" olarak cikar ama harfleri degistirme (i'yi ı yapma, u'yu ü yapma).
3. Bulunamayan turler icin bos liste [] kullan.
4. Metin resmi olmayabilir, yazim hatali veya bozuk olabilir. Anlamaya calis.
5. Ayni varlik birden fazla geciyorsa sadece bir kez yaz.

ORNEK 1:
Metin: "ben ali veli. mehmet oza evimi 15000 tl karsılıgı kiraya vericem. 12 aylık sozlesme olucak."
Yanit: {"PERSON": ["ali veli", "mehmet oz"], "ORG": [], "LOC": [], "DATE": [], "MONEY": ["15000 tl"], "CARDINAL": ["12 aylık"], "PERCENT": []}

ORNEK 2:
Metin: "Kiracı Ayse Kaya ile kiraya veren Fatma Demir arasinda 01.06.2025 tarihinde ankara cankaya icin kira sozlesmesi yapildi. aylik 18.500 TL, depozito 37.000 TL. sure 2 yil. artis %20."
Yanit: {"PERSON": ["Ayse Kaya", "Fatma Demir"], "ORG": [], "LOC": ["ankara", "cankaya"], "DATE": ["01.06.2025"], "MONEY": ["18.500 TL", "37.000 TL"], "CARDINAL": ["2 yil"], "PERCENT": ["%20"]}

Yanit SADECE JSON olsun, baska bir sey yazma."""


CARDINAL_PATTERN = re.compile(
    r'^\d+\s*(?:yıl(?:lık)?|yil(?:lik)?|ay(?:lık|lik)?|hafta(?:lık|lik)?|gün(?:lük)?|gun(?:luk)?)$',
    re.IGNORECASE,
)


def _empty_result() -> Dict[str, List[str]]:
    return {key: [] for key in ENTITY_KEYS}


def _parse_response(raw: str) -> Dict[str, List[str]]:
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        logger.debug("ner_response_fallback_json_parse")
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if not match:
            return _empty_result()
        try:
            data = json.loads(match.group())
        except json.JSONDecodeError:
            return _empty_result()

    result = _empty_result()
    for key in ENTITY_KEYS:
        val = data.get(key, [])
        if isinstance(val, list):
            items = [str(v).strip() for v in val if v]
            if key == "CARDINAL":
                items = [v for v in items if CARDINAL_PATTERN.match(v)]
            result[key] = items
    return result


class NERService:
    def __init__(self):
        self.model_name = ollama_client.OLLAMA_MODEL

    async def extract(self, text: str) -> Dict[str, List[str]]:
        logger.debug("ner_extract_start", extra={"text_length": len(text)})
        try:
            raw = await ollama_client.generate(
                prompt=text,
                system=SYSTEM_PROMPT,
                format_json=True,
            )
            result = _parse_response(raw)
            logger.info("ner_extract_complete", extra={
                "entity_types": list(result.keys()),
                "total_entities": sum(len(v) for v in result.values()),
            })
            return result
        except Exception:
            logger.error("ner_extraction_failed", exc_info=True)
            return _empty_result()

    async def health_check(self) -> bool:
        return await ollama_client.health_check()


# Singleton
ner_service = NERService()
