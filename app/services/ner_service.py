import json
import re
from typing import Dict, List

from app.services import ollama_client

ENTITY_KEYS = ["PERSON", "ORG", "LOC", "DATE", "MONEY", "CARDINAL", "PERCENT"]

SYSTEM_PROMPT = """Sen bir Turkce sozlesme metni analiz uzmanisin.
Verilen metinden asagidaki varlik turlerini cikar ve JSON olarak don:

- PERSON: Kisi isimleri (ornek: "Ahmet Yilmaz", "Fatma Demir")
- ORG: Kurum/sirket isimleri (ornek: "ABC Teknoloji A.S.", "SGK")
- LOC: Yer/konum isimleri (ornek: "Istanbul", "Kadikoy")
- DATE: Tarihler (ornek: "01.03.2025", "1 Mart 2025")
- MONEY: Para tutarlari birim ile birlikte (ornek: "25.000 TL", "1.500 USD")
- CARDINAL: Sure/miktar ifadeleri (ornek: "2 ay", "1 yillik", "3 gun")
- PERCENT: Yuzde ifadeleri (ornek: "%25", "yuzde 10")

KURALLAR:
1. Sadece metinde gecen varliklari cikar, uydurma.
2. Her anahtar icin bir liste don. Bulunamayan turler icin bos liste [] kullan.
3. Yanit SADECE JSON olsun, baska bir sey yazma.
4. Metin bozuk, yazim hatali veya OCR'den gelmis olabilir. En iyi tahminini yap.
5. Varliklari metinde gectigi sekilde yaz, duzeltme yapma.

Yanit formati:
{"PERSON": [...], "ORG": [...], "LOC": [...], "DATE": [...], "MONEY": [...], "CARDINAL": [...], "PERCENT": [...]}"""


def _empty_result() -> Dict[str, List[str]]:
    return {key: [] for key in ENTITY_KEYS}


def _parse_response(raw: str) -> Dict[str, List[str]]:
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
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
            result[key] = [str(v) for v in val if v]
    return result


class NERService:
    def __init__(self):
        self.model_name = ollama_client.OLLAMA_MODEL

    async def extract(self, text: str) -> Dict[str, List[str]]:
        try:
            raw = await ollama_client.generate(
                prompt=text,
                system=SYSTEM_PROMPT,
                format_json=True,
            )
            return _parse_response(raw)
        except Exception as e:
            print(f"Ollama NER extraction failed: {e}")
            return _empty_result()

    async def health_check(self) -> bool:
        return await ollama_client.health_check()


# Singleton
ner_service = NERService()
