import logging
import re
from typing import Dict, List

logger = logging.getLogger(__name__)

ENTITY_KEYS = ["PERSON", "ORG", "LOC", "DATE", "MONEY", "CARDINAL", "PERCENT"]


_TR_MAP = str.maketrans('çğıöşüâîûÇĞİÖŞÜÂÎÛ', 'cgiosuaiuCGIOSUAIU')


def _normalize_for_compare(val: str) -> str:
    return re.sub(r'\s+', '', val.lower().translate(_TR_MAP))


def _is_duplicate_of_existing(val: str, existing: List[str]) -> bool:
    val_norm = _normalize_for_compare(val)
    for e in existing:
        e_norm = _normalize_for_compare(e)
        if val_norm in e_norm or e_norm in val_norm:
            return True
    return False


def merge_entities(
    llm_results: Dict[str, List[str]],
    regex_results: Dict[str, List[str]],
) -> Dict[str, List[str]]:
    merged = {}
    for key in ENTITY_KEYS:
        llm_vals = llm_results.get(key, [])
        regex_vals = regex_results.get(key, [])

        # Start with deduplicated LLM values (primary source)
        combined = []
        for val in llm_vals:
            normalized = val.strip()
            if normalized and not _is_duplicate_of_existing(normalized, combined):
                combined.append(normalized)

        # Add regex values only if they don't overlap with existing values
        for val in regex_vals:
            normalized = val.strip()
            if not normalized:
                continue
            if _is_duplicate_of_existing(normalized, combined):
                continue
            combined.append(normalized)

        regex_added = len(combined) - len([v for v in llm_vals if v.strip()])
        logger.debug("entity_merge_complete", extra={
            "entity_type": key,
            "llm_count": len(llm_vals),
            "regex_added": max(0, regex_added),
            "total": len(combined),
        })
        merged[key] = combined
    return merged
