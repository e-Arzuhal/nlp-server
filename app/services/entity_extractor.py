"""
e-Arzuhal NLP Server - Entity Extractor
spaCy kullanarak NER (Named Entity Recognition)
spaCy yoksa / model yuklenemezse regex/lite fallback kullanir.
"""
from typing import Dict, Any, Optional, TYPE_CHECKING
import re

from app.config import SPACY_MODEL, ENTITY_LABELS, ALLOW_SPACY_DOWNLOAD, USE_SPACY, REQUIRE_SPACY

if TYPE_CHECKING:  # pragma: no cover
    import spacy  # only for type hints


class EntityExtractor:
    """
    spaCy tabanli entity cikarici (opsiyonel) + regex/lite fallback
    """

    def __init__(self):
        self.nlp: Optional["spacy.Language"] = None
        self.mode: str = "regex"  # "spacy" | "regex"
        self._load_model()

    def _load_model(self):
        """spaCy modelini yukle; basarisizsa regex/lite moda dus"""
        if not USE_SPACY:
            self.nlp = None
            self.mode = "regex"
            return

        try:
            import spacy  # type: ignore
        except Exception as e:
            if REQUIRE_SPACY:
                raise RuntimeError("spaCy import edilemedi (REQUIRE_SPACY=true).") from e
            self.nlp = None
            self.mode = "regex"
            return

        try:
            nlp = spacy.load(SPACY_MODEL)
        except OSError as e:
            if ALLOW_SPACY_DOWNLOAD:
                try:
                    import spacy.cli  # type: ignore
                    spacy.cli.download(SPACY_MODEL)  # type: ignore[attr-defined]
                    nlp = spacy.load(SPACY_MODEL)
                except Exception as e2:
                    if REQUIRE_SPACY:
                        raise RuntimeError(f"spaCy modeli indirilemedi/yuklenemedi: {SPACY_MODEL}") from e2
                    self.nlp = None
                    self.mode = "regex"
                    return
            else:
                if REQUIRE_SPACY:
                    raise RuntimeError(
                        f"spaCy modeli bulunamadi: {SPACY_MODEL}. "
                        f"python -m spacy download {SPACY_MODEL} (veya ALLOW_SPACY_DOWNLOAD=true)."
                    ) from e
                self.nlp = None
                self.mode = "regex"
                return

        keep = {"ner", "tok2vec"}
        disable = [p for p in nlp.pipe_names if p not in keep]
        if disable:
            nlp.disable_pipes(*disable)

        self.nlp = nlp
        self.mode = "spacy"
        print(f"spaCy model yuklendi: {SPACY_MODEL} (enabled pipes: {self.nlp.pipe_names})")

    def extract(self, text: str) -> Dict[str, Any]:
        if self.mode == "spacy" and self.nlp:
            return self._extract_with_spacy(text)
        return self._extract_with_regex(text)

    def _extract_with_spacy(self, text: str) -> Dict[str, Any]:
        doc = self.nlp(text)  # type: ignore[misc]

        entities = []
        extracted_fields = {
            "taraflar": [],
            "tutar": None,
            "tarih": None,
            "lokasyon": None,
            "kurum": None,
        }

        for ent in doc.ents:
            label = ent.label_
            if label == "PER":
                label = "PERSON"
            elif label == "LOC":
                label = "GPE"

            mapped_field = ENTITY_LABELS.get(ent.label_) or ENTITY_LABELS.get(label)

            entities.append(
                {
                    "text": ent.text,
                    "label": label,
                    "start": ent.start_char,
                    "end": ent.end_char,
                    "mapped_field": mapped_field,
                }
            )

            if label == "PERSON":
                extracted_fields["taraflar"].append(ent.text)
            elif label == "ORG":
                extracted_fields["kurum"] = ent.text
            elif label == "MONEY":
                extracted_fields["tutar"] = self._normalize_money(ent.text)
            elif label == "DATE":
                extracted_fields["tarih"] = ent.text
            elif label == "GPE":
                extracted_fields["lokasyon"] = ent.text

        # spaCy kacirirsa regex fallback ile tamamla
        if not extracted_fields["tutar"]:
            extracted_fields["tutar"] = self._extract_turkish_money(text)
        if not extracted_fields["tarih"]:
            extracted_fields["tarih"] = self._extract_turkish_date(text)
        if not extracted_fields["taraflar"]:
            extracted_fields["taraflar"] = self._extract_simple_persons(text)
        if not extracted_fields["lokasyon"]:
            extracted_fields["lokasyon"] = self._extract_simple_location(text)
        if not extracted_fields["kurum"]:
            extracted_fields["kurum"] = self._extract_simple_org(text)

        return {"entities": entities, "extracted_fields": extracted_fields}

    def _extract_with_regex(self, text: str) -> Dict[str, Any]:
        entities: list[dict[str, Any]] = []
        extracted_fields = {
            "taraflar": [],
            "tutar": None,
            "tarih": None,
            "lokasyon": None,
            "kurum": None,
        }

        def add_entity(label: str, m: re.Match, override_text: Optional[str] = None):
            t = (override_text if override_text is not None else m.group(0)).strip()
            if not t:
                return
            mapped_field = ENTITY_LABELS.get(label)
            entities.append(
                {
                    "text": t,
                    "label": label,
                    "start": m.start(),
                    "end": m.end(),
                    "mapped_field": mapped_field,
                }
            )

        # MONEY
        money_patterns = [
            r"(\d{1,3}(?:[.,]\d{3})*(?:[.,]\d{2})?)\s*(TL|tl|Tl|lira|Lira)",
            r"(\d+(?:\.\d+)?)\s*(bin|milyon|milyar)?\s*(TL|tl|lira)",
        ]
        for pat in money_patterns:
            for m in re.finditer(pat, text):
                add_entity("MONEY", m)
                if extracted_fields["tutar"] is None:
                    extracted_fields["tutar"] = self._normalize_money(m.group(0))

        # DATE / sure
        for m in re.finditer(r"(\d{1,2}[./]\d{1,2}[./]\d{2,4})", text):
            add_entity("DATE", m)
            extracted_fields["tarih"] = extracted_fields["tarih"] or m.group(0)
        for m in re.finditer(r"(\d+)\s*(ay|gun|hafta|yil)", text, flags=re.IGNORECASE):
            add_entity("DATE", m)
            extracted_fields["tarih"] = extracted_fields["tarih"] or m.group(0)

        # PERSON (lite)
        person_pat = r"\b([A-ZÇĞİÖŞÜ][a-zçğıöşü]+(?:\s+[A-ZÇĞİÖŞÜ][a-zçğıöşü]+)?(?:\s+(?:Bey|Hanim))?)\b"
        for m in re.finditer(person_pat, text):
            cand = m.group(1).strip()
            if cand in {"TL", "USD", "EUR"}:
                continue
            add_entity("PERSON", m, override_text=cand)
            if cand not in extracted_fields["taraflar"]:
                extracted_fields["taraflar"].append(cand)

        # GPE (lite)
        loc_m = re.search(r"\b([A-ZÇĞİÖŞÜ][a-zçğıöşü]+)(?:'deki|'daki|'de|'da)\b", text)
        if loc_m:
            add_entity("GPE", loc_m, override_text=loc_m.group(1))
            extracted_fields["lokasyon"] = extracted_fields["lokasyon"] or loc_m.group(1)

        # ORG (lite)
        org_m = re.search(r"\b([A-Z0-9][A-Za-z0-9ÇĞİÖŞÜçğıöşü\s]+)\s+(?:Sirketi|A\.S\.|Ltd\.|LTD)\b", text)
        if org_m:
            add_entity("ORG", org_m)
            extracted_fields["kurum"] = extracted_fields["kurum"] or org_m.group(0).strip()

        # Son bir tamamlayici (eski helper'lar)
        extracted_fields["tutar"] = extracted_fields["tutar"] or self._extract_turkish_money(text)
        extracted_fields["tarih"] = extracted_fields["tarih"] or self._extract_turkish_date(text)
        if not extracted_fields["taraflar"]:
            extracted_fields["taraflar"] = self._extract_simple_persons(text)

        return {"entities": entities, "extracted_fields": extracted_fields}

    def _normalize_money(self, text: str) -> str:
        """Para degerini normalize et"""
        cleaned = re.sub(r"[^\d.,]", " ", text).strip()

        if "tl" in text.lower() or "lira" in text.lower():
            return f"{cleaned} TL"
        if "dolar" in text.lower() or "usd" in text.lower():
            return f"{cleaned} USD"
        if "euro" in text.lower() or "eur" in text.lower():
            return f"{cleaned} EUR"
        return text

    def _extract_turkish_money(self, text: str) -> Optional[str]:
        """Turkce metin icinden para degerini cikar"""
        patterns = [
            r"(\d{1,3}(?:[.,]\d{3})*(?:[.,]\d{2})?)\s*(?:TL|tl|Tl|lira|Lira)",
            r"(\d+(?:\.\d+)?)\s*(?:bin|milyon|milyar)?\s*(?:TL|tl|lira)",
        ]
        for pattern in patterns:
            m = re.search(pattern, text)
            if m:
                return f"{m.group(1)} TL"
        return None

    def _extract_turkish_date(self, text: str) -> Optional[str]:
        """Turkce metin icinden tarih cikar"""
        patterns = [
            r"(\d{1,2}[./]\d{1,2}[./]\d{2,4})",
            r"(\d{1,2})\s*(Ocak|Subat|Mart|Nisan|Mayis|Haziran|Temmuz|Agustos|Eylul|Ekim|Kasim|Aralik)\s*(\d{2,4})",
            r"(\d+)\s*(ay|gun|hafta|yil)",
        ]
        for pattern in patterns:
            m = re.search(pattern, text, re.IGNORECASE)
            if m:
                return m.group(0)
        return None

    def _extract_simple_persons(self, text: str) -> list[str]:
        pat = r"\b([A-ZÇĞİÖŞÜ][a-zçğıöşü]+(?:\s+[A-ZÇĞİÖŞÜ][a-zçğıöşü]+)?(?:\s+(?:Bey|Hanim))?)\b"
        candidates = re.findall(pat, text)
        stop = {"TL", "USD", "EUR"}
        out = []
        for c in candidates:
            c = c.strip()
            if c in stop:
                continue
            if c not in out:
                out.append(c)
        return out

    def _extract_simple_location(self, text: str) -> Optional[str]:
        m = re.search(r"\b([A-ZÇĞİÖŞÜ][a-zçğıöşü]+)(?:'deki|'daki|'de|'da)\b", text)
        return m.group(1) if m else None

    def _extract_simple_org(self, text: str) -> Optional[str]:
        m = re.search(
            r"\b([A-Z0-9][A-Za-z0-9ÇĞİÖŞÜçğıöşü\s]+)\s+(?:Sirketi|A\.S\.|Ltd\.|LTD)\b", text
        )
        return m.group(0).strip() if m else None


# Singleton instance
_extractor: Optional[EntityExtractor] = None


def get_entity_extractor() -> EntityExtractor:
    """Entity extractor singleton'i dondur"""
    global _extractor
    if _extractor is None:
        _extractor = EntityExtractor()
    return _extractor
