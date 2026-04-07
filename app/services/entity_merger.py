from typing import Dict, List

ENTITY_KEYS = ["PERSON", "ORG", "LOC", "DATE", "MONEY", "CARDINAL", "PERCENT"]


def merge_entities(
    llm_results: Dict[str, List[str]],
    regex_results: Dict[str, List[str]],
) -> Dict[str, List[str]]:
    merged = {}
    for key in ENTITY_KEYS:
        llm_vals = llm_results.get(key, [])
        regex_vals = regex_results.get(key, [])
        seen = set()
        combined = []
        for val in llm_vals + regex_vals:
            normalized = val.strip()
            if normalized and normalized.lower() not in seen:
                seen.add(normalized.lower())
                combined.append(normalized)
        merged[key] = combined
    return merged
