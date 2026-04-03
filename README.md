# NLP Server

Turkish contract entity extraction microservice. Receives raw contract text, returns extracted entities in spaCy format + contract type — ready for GraphRAG server consumption.

## What It Does

- Classifies the contract type (iş, kira, satış, hizmet, etc.) using keyword matching
- Extracts named entities (persons, orgs, locations) via BERT (`savasy/bert-base-turkish-ner-cased`)
- Extracts structured values (money, dates, durations, percentages) via regex
- Returns everything in a spaCy-compatible format that GraphRAG expects

## Extracted Entity Types

| Key | Source | Examples |
|-----|--------|---------|
| `PERSON` | BERT | Ahmet Yılmaz, Fatma Demir |
| `ORG` | BERT | ABC Teknoloji A.Ş. |
| `LOC` | BERT | İstanbul, Kadıköy |
| `MONEY` | Regex | 25.000 TL, 500 USD, 1.200 EUR |
| `DATE` | Regex | 01.03.2025, 15 Ocak 2024 |
| `CARDINAL` | Regex | 2 ay, 1 yıllık, 30 gün |
| `PERCENT` | Regex | %25, 10 yüzde |

## Installation

```bash
# Create and activate environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

## Running

```bash
uvicorn app.main:app --reload --port 8001
```

## Docker

```bash
docker build -t nlp-server .
docker run -p 8001:8001 nlp-server
```

## API

### POST /api/v1/extract

Ana analiz endpoint'i.

**İstek:**
```json
{
  "text": "Bu iş sözleşmesi Ahmet Yılmaz ile ABC Teknoloji A.Ş. arasında 01.03.2025 tarihinde imzalanmıştır. Aylık brüt ücret 25.000 TL olarak kararlaştırılmıştır. Deneme süresi 2 ay olarak belirlenmiştir."
}
```

**Yanıt:**
```json
{
  "contract_type": "is_sozlesmesi",
  "contract_type_confidence": 0.47,
  "extracted_entities": {
    "PERSON":   ["Ahmet Yılmaz"],
    "ORG":      ["ABC Teknoloji A.Ş."],
    "LOC":      [],
    "MONEY":    ["25.000 TL"],
    "DATE":     ["01.03.2025"],
    "CARDINAL": ["2 ay"],
    "PERCENT":  []
  },
  "raw_text_length": 194,
  "processing_time_ms": 11
}
```

> `contract_type_confidence`, `raw_text_length`, `processing_time_ms` are consumed by the main server for logging/routing. They are **not** forwarded to GraphRAG.

### GET /health

```json
{
  "status": "ok",
  "model": "savasy/bert-base-turkish-ner-cased",
  "model_loaded": true
}
```

## Project Structure

```
nlp-server/
├── app/
│   ├── main.py
│   ├── routers/
│   │   └── extract.py
│   └── services/
│       ├── contract_classifier.py   # keyword-based contract type detection
│       ├── ner_service.py           # BERT NER singleton (PER, ORG, LOC)
│       ├── postprocessor.py         # regex (MONEY, DATE, CARDINAL, PERCENT)
│       └── spacy_mapper.py          # combines BERT + regex → spaCy format
├── models/
│   └── schemas.py
├── requirements.txt
└── Dockerfile
```

## Architecture

```
POST /api/v1/extract
        │
        ├── 1. Contract Type Classifier   (keyword-based)
        │
        ├── 2. BERT NER                   (PER, ORG, LOC)
        │
        ├── 3. Regex Post-Processor       (MONEY, DATE, CARDINAL, PERCENT)
        │
        └── 4. spaCy Format Mapper        (GraphRAG-compatible output)
```

## Contract Types

| Value | Description |
|-------|-------------|
| `is_sozlesmesi` | İş Sözleşmesi |
| `kira_sozlesmesi` | Kira Sözleşmesi |
| `satis_sozlesmesi` | Satış Sözleşmesi |
| `hizmet_sozlesmesi` | Hizmet Sözleşmesi |
| `vekaletname` | Vekaletname |
| `taahhutname` | Taahhütname |
| `kefalet_sozlesmesi` | Kefalet Sözleşmesi |

If `contract_type_confidence < 0.6`, the main server should ask the user to confirm before proceeding.
