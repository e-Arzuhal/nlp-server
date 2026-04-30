import logging
from typing import Tuple, Optional

logger = logging.getLogger(__name__)

CONTRACT_KEYWORDS = {
    "is_sozlesmesi": [
        "işçi", "işveren", "kıdem tazminatı", "ihbar süresi",
        "ücret", "hizmet akdi", "iş akdi", "brüt maaş", "net maaş",
        "deneme süresi", "fazla mesai", "yıllık izin", "çalışma saatleri",
        "sgk", "sigorta primi", "iş sözleşmesi", "is sozlesmesi",
        "işe alım", "işten çıkar", "işten çıkış", "personel",
    ],
    "kira_sozlesmesi": [
        "kiracı", "kiraya veren", "kira bedeli", "kira süresi",
        "tahliye", "depozito", "aidat", "konut", "işyeri kirası",
        "kira artışı", "tfe", "tüfe", "kira sözleşmesi", "kira sozlesmesi",
        "kiraya ver", "kiraya vereceğim", "kiraya verecegim",
        "evimi kiraya", "dairemi kiraya", "ofisimi kiraya",
    ],
    "satis_sozlesmesi": [
        "satıcı", "alıcı", "satış bedeli", "tapu", "devir",
        "mülkiyet", "satın alma", "satin alma", "ödeme planı",
        "teslim tarihi", "satış sözleşmesi", "satis sozlesmesi",
        "satacağım", "satacagim", "satıyorum", "satiyorum",
        "satın aldım", "satin aldim",
    ],
    "hizmet_sozlesmesi": [
        "hizmet bedeli", "hizmet sağlayıcı", "danışmanlık",
        "proje", "serbest meslek", "freelance", "fatura", "kdv",
        "hizmet sözleşmesi", "hizmet sozlesmesi", "danışman",
        "danisman", "müşteri",
    ],
    "vekaletname": [
        "vekil", "vekalet", "temsil", "müvekkil", "noter",
        "yetki", "vekaleten", "vekâlet", "vekâletname",
    ],
    "taahhutname": [
        "taahhüt", "taahhüt eder", "beyan eder",
        "yükümlülük altına", "taahhütname", "taahhutname",
        "beyan ederim", "söz veriyorum", "soz veriyorum",
    ],
    "kefalet_sozlesmesi": [
        "kefil", "kefalet", "müteselsil",
        "kefil olduğumu", "kefil oldugumu", "güvence", "kefalet sözleşmesi",
        "kefalet sozlesmesi", "kefil olarak",
    ],
    "borc_sozlesmesi": [
        # Borç verme / alma — senet niteliğinde sözleşmeler
        "borç", "borc", "borç verecek", "borc verecek",
        "borç vereceğim", "borc verecegim",
        "borç alacak", "borc alacak", "borç alacağım", "borc alacagim",
        "ödünç", "odunc", "ödünç verecek", "odunc verecek",
        "ödünç alacak", "odunc alacak",
        "borç sözleşmesi", "borc sozlesmesi",
        "borçlu", "borclu", "alacaklı", "alacakli",
        "geri ödeme", "geri odeme", "vade", "vadeli",
        "senet", "bono", "tahvil",
        "borçlanma", "borclanma",
    ],
    "gizlilik_sozlesmesi": [
        "gizlilik", "gizli bilgi", "ifşa etmemek", "ifsa etmemek",
        "nda", "non-disclosure", "ticari sır", "ticari sir",
        "gizlilik sözleşmesi", "gizlilik sozlesmesi",
        "gizli tutmayı", "gizli tutmayi",
    ],
}


def classify_contract(text: str) -> Tuple[Optional[str], float]:
    """
    Returns (contract_type, confidence_score).
    Returns (None, 0.0) if type cannot be determined.
    """
    text_lower = text.lower()
    scores = {}

    for contract_type, keywords in CONTRACT_KEYWORDS.items():
        # Anahtar kelime kümeleri farklı boyuttadır — eşleşme sayısını
        # küme boyutuyla normalize etmek küçük kümeleri haksız üstün kılıyordu.
        # Mutlak eşleşme sayısına ek olarak küçük bir normalizasyon faktörü
        # uygulanır; bu sayede tek kelimelik isabet de yakalanır.
        matches = sum(1 for kw in keywords if kw in text_lower)
        scores[contract_type] = matches

    best_type = max(scores, key=scores.get)
    best_count = scores[best_type]

    if best_count == 0:
        logger.debug("contract_classification_no_match")
        return None, 0.0

    # Mutlak eşleşme sayısına göre güven hesapla — 1 isabet ~0.55,
    # 2 isabet ~0.85, 3+ isabet ~0.95+ civarı.
    confidence = min(0.99, 0.4 + 0.18 * best_count)
    logger.info("contract_classified", extra={
        "contract_type": best_type,
        "matches": best_count,
        "confidence": round(confidence, 3),
    })
    return best_type, round(confidence, 2)
