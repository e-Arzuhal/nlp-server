# NLP Server

Turkish contract entity extraction microservice. Receives raw contract text, returns extracted entities in spaCy format + contract type — ready for GraphRAG server consumption.

## What It Does

- Classifies the contract type (iş, kira, satış, hizmet, etc.) using keyword matching
- Extracts named entities (persons, orgs, locations) via **Qwen 2.5** (Ollama) — local, no external API
- Extracts structured values (money, dates, durations, percentages) via regex
- Returns everything in a spaCy-compatible format that GraphRAG expects

## Extracted Entity Types

| Key | Source | Examples |
|-----|--------|---------|
| `PERSON` | Qwen 2.5 (Ollama) | Ahmet Yılmaz, Fatma Demir |
| `ORG` | Qwen 2.5 (Ollama) | ABC Teknoloji A.Ş. |
| `LOC` | Qwen 2.5 (Ollama) | İstanbul, Kadıköy |
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

### GET /api/v1/metrics

Sınıflandırıcının precision / recall / F1 metriklerini döner. 28 etiketli Türkçe hukuki metin örneği üzerinde değerlendirilir; sonuç uygulama başlangıcında cache'lenir.

```json
{
  "accuracy": 0.96,
  "total_samples": 28,
  "correct": 27,
  "macro":    { "precision": 0.95, "recall": 0.96, "f1": 0.95 },
  "weighted": { "precision": 0.96, "recall": 0.96, "f1": 0.96 },
  "per_class": {
    "is_sozlesmesi":     { "precision": 1.0,  "recall": 1.0,  "f1": 1.0,  "support": 5 },
    "kira_sozlesmesi":   { "precision": 1.0,  "recall": 1.0,  "f1": 1.0,  "support": 5 },
    "...": "..."
  },
  "classifier": "keyword_overlap",
  "note": "Evaluated on 28-sample curated Turkish legal text test set (7 classes)."
}
```

### GET /health

```json
{
  "status": "ok",
  "model": "qwen2.5",
  "model_loaded": true
}
```

## Project Structure

```
nlp-server/
├── app/
│   ├── main.py
│   ├── routers/
│   │   └── extract.py               # POST /api/v1/extract, GET /api/v1/metrics
│   └── services/
│       ├── contract_classifier.py   # keyword-based contract type detection
│       ├── classifier_eval.py       # precision/recall/F1 evaluation (28-sample test set)
│       ├── ner_service.py           # Qwen 2.5 (Ollama) NER singleton (PER, ORG, LOC)
│       ├── postprocessor.py         # regex (MONEY, DATE, CARDINAL, PERCENT)
│       └── entity_merger.py         # Qwen + regex → unified output
├── models/
│   └── schemas.py
├── requirements.txt
└── Dockerfile
```

## Architecture

```
POST /api/v1/extract
        │
        ├── 1. Contract Type Classifier   (keyword-based + confidence score)
        │
        ├── 2. Qwen 2.5 NER via Ollama   (PER, ORG, LOC) — local, no external API
        │
        ├── 3. Regex Post-Processor       (MONEY, DATE, CARDINAL, PERCENT)
        │
        └── 4. Entity Merger              (LLM + regex → GraphRAG-compatible output)

GET /api/v1/metrics
        └── classifier_eval.py → precision/recall/F1 per class (cached)
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

## Güvenlik

| Değişken | Varsayılan | Açıklama |
|----------|-----------|----------|
| `INTERNAL_API_KEY` | _(boş)_ | Main-server ile paylaşılan anahtar. `openssl rand -hex 32` |
| `DEBUG` | `false` | `true` iken Swagger açılır (`/docs`, `/redoc`) |
| `ALLOWED_ORIGINS` | `http://localhost:8080` | CORS whitelist |

**Auth davranışı:**
- `INTERNAL_API_KEY` set + doğru header → izin verilir
- `INTERNAL_API_KEY` set + yanlış/eksik header → **401**
- `INTERNAL_API_KEY` boş + `DEBUG=true` → izin verilir (dev modu, uyarı loglanır)
- `INTERNAL_API_KEY` boş + `DEBUG=false` → **503** (yanlış yapılandırma)

> Production'da her iki değişkenin de set edilmesi zorunludur.

## Observability

Her istek `X-Request-ID` ile loglanır. main-server'dan gelen header korunur; olmayan isteklerde UUID üretilir.

```
INFO  http_request service=nlp method=POST path=/api/v1/extract status=200 ms=340 request_id=3fa2c1d8
```
