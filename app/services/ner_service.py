from transformers import pipeline
from typing import Dict, List
import torch


class NERService:
    def __init__(self):
        self.model_name = "savasy/bert-base-turkish-ner-cased"
        self.nlp = None
        self._load_model()

    def _load_model(self):
        print(f"Loading NER model: {self.model_name}")
        self.nlp = pipeline(
            "ner",
            model=self.model_name,
            tokenizer=self.model_name,
            aggregation_strategy="simple",
            device=0 if torch.cuda.is_available() else -1
        )
        print("NER model loaded.")

    def extract(self, text: str) -> Dict[str, List[Dict]]:
        """
        Returns raw NER results grouped by entity type.
        Groups from this model: PER, ORG, LOC
        """
        results = self.nlp(text)
        extracted = {"PER": [], "ORG": [], "LOC": []}

        for entity in results:
            group = entity.get("entity_group", "")
            if group in extracted:
                extracted[group].append({
                    "text": entity["word"],
                    "score": round(entity["score"], 3),
                    "start": entity["start"],
                    "end": entity["end"]
                })

        return extracted


# Singleton — model loads once at startup
ner_service = NERService()
