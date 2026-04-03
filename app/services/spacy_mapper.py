from typing import Dict, List


def to_spacy_format(
    bert_results: Dict[str, List[Dict]],
    regex_results: Dict[str, List[str]]
) -> Dict[str, List[str]]:
    """
    Combines BERT NER output and regex output into spaCy-compatible format.

    BERT groups:   PER → PERSON  |  ORG → ORG  |  LOC → LOC
    Regex groups:  DATE, MONEY, CARDINAL, PERCENT (already in spaCy label format)
    """
    return {
        "PERSON":   [e["text"] for e in bert_results.get("PER", [])],
        "ORG":      [e["text"] for e in bert_results.get("ORG", [])],
        "LOC":      [e["text"] for e in bert_results.get("LOC", [])],
        "MONEY":    regex_results.get("MONEY", []),
        "DATE":     regex_results.get("DATE", []),
        "CARDINAL": regex_results.get("CARDINAL", []),
        "PERCENT":  regex_results.get("PERCENT", []),
    }
