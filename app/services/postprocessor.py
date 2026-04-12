import logging
import re
from typing import List, Dict

logger = logging.getLogger(__name__)

# --- DATE ---
DATE_PATTERNS = [
    r'\b(\d{1,2})[./](\d{1,2})[./](\d{4})\b',
    r'\b(\d{4})[./](\d{1,2})[./](\d{1,2})\b',
    r'\b(\d{1,2})\s+(Ocak|Şubat|Mart|Nisan|Mayıs|Haziran|'
    r'Temmuz|Ağustos|Eylül|Ekim|Kasım|Aralık)\s+(\d{4})\b',
]

# --- MONEY ---
MONEY_PATTERNS = [
    r'(?<!\d)\d{1,3}(?:[.,]\d{3})*(?:[.,]\d{2})?\s*(?:TL|₺|lira)\b',
    r'(?<!\d)\d{1,3}(?:[.,]\d{3})*(?:[.,]\d{2})?\s*(?:USD|\$|dolar)\b',
    r'(?<!\d)\d{1,3}(?:[.,]\d{3})*(?:[.,]\d{2})?\s*(?:EUR|€|euro)\b',
    r'(?<!\d)\d{4,}\s*(?:TL|₺|lira)\b',
    r'(?<!\d)\d{4,}\s*(?:USD|\$|dolar)\b',
    r'(?<!\d)\d{4,}\s*(?:EUR|€|euro)\b',
]

# --- DURATION → CARDINAL ---
DURATION_PATTERNS = [
    r'(?<!\d)\d+\s*yıl(?:lık)?\b',
    r'(?<!\d)\d+\s*ay(?:lık)?\b',
    r'(?<!\d)\d+\s*hafta(?:lık)?\b',
    r'(?<!\d)\d+\s*gün(?:lük)?\b',
]

# --- PERCENT ---
PERCENT_PATTERNS = [
    r'%\s*\d+(?:[.,]\d+)?\b',
    r'(?<!\d)\d+(?:[.,]\d+)?\s*(?:yüzde|%)',
]


def _find_all(patterns: List[str], text: str) -> List[str]:
    found = []
    for pattern in patterns:
        for m in re.finditer(pattern, text, re.IGNORECASE):
            found.append(m.group().strip())
    return list(dict.fromkeys(found))  # deduplicate, preserve order


def extract_all(text: str) -> Dict[str, List[str]]:
    dates = _find_all(DATE_PATTERNS, text)
    money = _find_all(MONEY_PATTERNS, text)
    cardinal = _find_all(DURATION_PATTERNS, text)
    percent = _find_all(PERCENT_PATTERNS, text)
    logger.debug("postprocessor_complete", extra={
        "DATE": len(dates),
        "MONEY": len(money),
        "CARDINAL": len(cardinal),
        "PERCENT": len(percent),
    })
    return {
        "DATE":     dates,
        "MONEY":    money,
        "CARDINAL": cardinal,
        "PERCENT":  percent,
    }
