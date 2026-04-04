from typing import Tuple, Optional

CONTRACT_KEYWORDS = {
    "is_sozlesmesi": [
        "işçi", "işveren", "kıdem tazminatı", "ihbar süresi",
        "ücret", "hizmet akdi", "iş akdi", "brüt maaş", "net maaş",
        "deneme süresi", "fazla mesai", "yıllık izin", "çalışma saatleri",
        "sgk", "sigorta primi"
    ],
    "kira_sozlesmesi": [
        "kiracı", "kiraya veren", "kira bedeli", "kira süresi",
        "tahliye", "depozito", "aidat", "konut", "işyeri kirası",
        "kira artışı", "tfe", "tüfe"
    ],
    "satis_sozlesmesi": [
        "satıcı", "alıcı", "satış bedeli", "tapu", "devir",
        "mülkiyet", "satın alma", "ödeme planı", "teslim tarihi"
    ],
    "hizmet_sozlesmesi": [
        "hizmet bedeli", "hizmet sağlayıcı", "danışmanlık",
        "proje", "serbest meslek", "freelance", "fatura", "kdv"
    ],
    "vekaletname": [
        "vekil", "vekalet", "temsil", "müvekkil", "noter",
        "yetki", "vekaleten"
    ],
    "taahhutname": [
        "taahhüt", "taahhüt eder", "beyan eder",
        "yükümlülük altına", "taahhütname"
    ],
    "kefalet_sozlesmesi": [
        "kefil", "kefalet", "müteselsil",
        "kefil olduğumu", "güvence"
    ]
}


def classify_contract(text: str) -> Tuple[Optional[str], float]:
    """
    Returns (contract_type, confidence_score).
    Returns (None, 0.0) if type cannot be determined.
    """
    text_lower = text.lower()
    scores = {}

    for contract_type, keywords in CONTRACT_KEYWORDS.items():
        matches = sum(1 for kw in keywords if kw in text_lower)
        scores[contract_type] = matches / len(keywords)

    best_type = max(scores, key=scores.get)
    best_score = scores[best_type]

    if best_score < 0.05:
        return None, 0.0

    confidence = min(1.0, best_score * 3.5)
    return best_type, round(confidence, 2)
