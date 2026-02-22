# e-Arzuhal NLP Server

A lightweight Named Entity Recognition (NER) microservice for Turkish text extraction.

## Purpose

This service has **ONE JOB**: Extract named entities from Turkish text. It does NOT perform classification.

## Features

- **Named Entity Recognition**: SpaCy `tr_core_news_md` + custom Regex patterns
- **Turkish Support**: Optimized for Turkish currency (TL, ₺, lira), dates, and property types
- **Lightweight**: No heavy deep learning models - CPU-friendly
- **Fast**: Single endpoint, minimal processing overhead

## Extracted Entity Types

| Type | Description | Examples |
|------|-------------|----------|
| `PERSON` | Person names | Ahmet Yılmaz, Fatma Demir |
| `MONEY` | Monetary amounts | 15.000 TL, 20.000 ₺, 5000 lira |
| `LOCATION` | Locations/places | Antalya, İstanbul, Ankara |
| `DATE` | Date expressions | 1 yıllığına, 6 ay, 15 Ocak 2024 |
| `OBJECT_OR_PROPERTY` | Objects/property types | ev, daire, araç, depozito |

## Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt

# Download SpaCy Turkish model
python -m spacy download tr_core_news_md

# Run the server
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

## Docker

```bash
# Build
docker build -t nlp-server .

# Run
docker run -p 8000:8000 nlp-server
```

## API Endpoints

### POST /api/extract

Main extraction endpoint - extracts named entities from Turkish text.

**Request:**
```json
{
  "text": "Ahmet Yılmaz'a Antalya'daki evimi aylık 15.000 TL'ye 1 yıllığına kiralayacağım. 20.000 TL depozito alacağım."
}
```

**Response:**
```json
{
  "raw_text": "Ahmet Yılmaz'a Antalya'daki evimi aylık 15.000 TL'ye 1 yıllığına kiralayacağım. 20.000 TL depozito alacağım.",
  "entities": {
    "PERSON": ["Ahmet Yılmaz"],
    "MONEY": ["15.000 TL", "20.000 TL"],
    "LOCATION": ["Antalya"],
    "DATE": ["1 yıllığına"],
    "OBJECT_OR_PROPERTY": ["ev", "depozito"]
  }
}
```

### GET /health

Health check endpoint.

```json
{
  "status": "healthy",
  "version": "1.0.0",
  "spacy_model_loaded": true
}
```

### GET /docs

Interactive API documentation (Swagger UI).

## Project Structure

```
nlp-server/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI application
│   ├── models.py            # Pydantic schemas
│   └── services/
│       ├── __init__.py
│       └── extractor.py     # TurkishEntityExtractor
├── tests/
│   └── test_nlp.py
├── Dockerfile
├── requirements.txt
└── README.md
```

## Architecture

This service is part of a **Star Topology** where a central "Main Server" orchestrates:

```
                    ┌─────────────┐
                    │ Main Server │
                    │ (Orchestrator)│
                    └──────┬──────┘
                           │
         ┌─────────────────┼─────────────────┐
         │                 │                 │
         ▼                 ▼                 ▼
   ┌───────────┐    ┌───────────┐    ┌───────────┐
   │ NLP Server│    │ Neo4j     │    │ LLM       │
   │ (This)    │    │ Server    │    │ Server    │
   └───────────┘    └───────────┘    └───────────┘
         │
         ▼
   Extract entities
   from Turkish text
```

## Notes

- If SpaCy model is not available, the service falls back to regex-only mode
- All entity type keys are always present in the response (empty list if none found)
- The service is stateless and can be scaled horizontally
