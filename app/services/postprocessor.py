import re
from typing import List, Dict

# --- DATE ---
DATE_PATTERNS = [
    r'\b(\d{1,2})[./](\d{1,2})[./](\d{4})\b',
    r'\b(\d{4})[./](\d{1,2})[./](\d{1,2})\b',
    r'\b(\d{1,2})\s+(Ocak|Şubat|Mart|Nisan|Mayıs|Haziran|'
    r'Temmuz|Ağustos|Eylül|Ekim|Kasım|Aralık)\s+(\d{4})\b',
]

# --- MONEY ---
MONEY_PATTERNS = [
    r'\d{1,3}(?:[.,]\d{3})*(?:[.,]\d{2})?\s*(?:TL|₺|lira)',
    r'\d{1,3}(?:[.,]\d{3})*(?:[.,]\d{2})?\s*(?:USD|\$|dolar)',
    r'\d{1,3}(?:[.,]\d{3})*(?:[.,]\d{2})?\s*(?:EUR|€|euro)',
]

# --- DURATION → CARDINAL ---
DURATION_PATTERNS = [
    r'\d+\s*yıl(?:lık)?',
    r'\d+\s*ay(?:lık)?',
    r'\d+\s*hafta(?:lık)?',
    r'\d+\s*gün(?:lük)?',
]

# --- PERCENT ---
PERCENT_PATTERNS = [
    r'%\s*\d+(?:[.,]\d+)?',
    r'\d+(?:[.,]\d+)?\s*(?:yüzde|%)',
]


def _find_all(patterns: List[str], text: str) -> List[str]:
    found = []
    for pattern in patterns:
        for m in re.finditer(pattern, text, re.IGNORECASE):
            found.append(m.group().strip())
    return list(dict.fromkeys(found))  # deduplicate, preserve order


def extract_all(text: str) -> Dict[str, List[str]]:
    return {
        "DATE":     _find_all(DATE_PATTERNS, text),
        "MONEY":    _find_all(MONEY_PATTERNS, text),
        "CARDINAL": _find_all(DURATION_PATTERNS, text),
        "PERCENT":  _find_all(PERCENT_PATTERNS, text),
    }
