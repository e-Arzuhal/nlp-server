# e-Arzuhal NLP Server

Doğal dil işleme servisi — sözleşme tipi sınıflandırma ve varlık (entity) çıkarımı.

---

## Özellikler

- **Sözleşme Türü Sınıflandırma** — TF-IDF + Naive Bayes ile 7 Türkçe sözleşme türü
- **Named Entity Recognition** — spaCy (opsiyonel) veya regex/lite mode
- **Para Miktarı Çıkarımı** — TL/lira suffix'siz bağlamsal eşleşme (kira bedeli, ücret, tutar vb.)
- **Süre / Tarih Ayrımı** — `sure` (12 ay, 6 hafta) ile `tarih` (01.01.2026) ayrı alanlarda
- **DURATION Entity** — Ay/gün/hafta/yıl ifadeleri ayrı entity tipi olarak işaretlenir

---

## Desteklenen Sözleşme Tipleri

| Tip | Açıklama |
|-----|----------|
| `borc_sozlesmesi` | Borç / Kredi sözleşmesi |
| `kira_sozlesmesi` | Kira sözleşmesi |
| `hizmet_sozlesmesi` | Hizmet / Danışmanlık sözleşmesi |
| `satis_sozlesmesi` | Satış sözleşmesi |
| `is_sozlesmesi` | İş / İstihdam sözleşmesi |
| `vekaletname` | Vekaletname |
| `taahhutname` | Taahhütname |

---

## Kurulum

```bash
cd nlp-server
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/macOS

pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8001 --reload
```

---

## spaCy (Opsiyonel)

Servis spaCy varsa spaCy NER, yoksa otomatik **regex/lite** moduna geçer.

```bash
# Önerilen (Python 3.11/3.12):
python -m spacy download tr_core_news_sm
```

| Değişken | Varsayılan | Açıklama |
|----------|-----------|----------|
| `USE_SPACY` | `true` | `false` → zorla regex/lite mod |
| `REQUIRE_SPACY` | `false` | `true` → spaCy yoksa servis başlamaz |
| `ALLOW_SPACY_DOWNLOAD` | `false` | `true` → runtime'da model indirir |

---

## API Endpoints

### POST /api/nlp/analyze

Ana analiz endpoint'i.

**İstek:**
```json
{
  "text": "Kadıköy dairemi Ahmet Yılmaz'a 15000 kira bedeli ile 12 ay kiraya vereceğim."
}
```

**Yanıt:**
```json
{
  "success": true,
  "contract_type": "kira_sozlesmesi",
  "confidence": 0.92,
  "entities": [
    {"text": "Ahmet Yılmaz", "label": "PERSON", "mapped_field": "taraf"},
    {"text": "15000 kira bedeli", "label": "MONEY", "mapped_field": "tutar"},
    {"text": "12 ay", "label": "DURATION", "mapped_field": "sure"}
  ],
  "extracted_fields": {
    "taraflar": ["Ahmet Yılmaz"],
    "tutar": "15000 TL",
    "tarih": null,
    "sure": "12 ay",
    "lokasyon": "Kadıköy",
    "kurum": null
  },
  "suggestions": [
    "Depozito miktarı belirtmek ister misiniz? (Kullanıcıların %84'ü ekliyor)"
  ]
}
```

> **Not:** `tarih` yalnızca takvim tarihlerini içerir (ör. `01/06/2026`).
> Süre ifadeleri (`12 ay`, `6 hafta`) `sure` alanına yazılır.

### POST /api/nlp/classify

Sadece sözleşme türü sınıflandırma.

### POST /api/nlp/entities

Sadece entity extraction.

### GET /health

Sağlık kontrolü.

---

## Entity Tipleri

| Label | Açıklama | extracted_fields |
|-------|----------|------------------|
| `PERSON` | Kişi adı | `taraflar` |
| `MONEY` | Para (TL suffix veya bağlamsal) | `tutar` |
| `DATE` | Takvim tarihi (gg/aa/yyyy, ay isimleri) | `tarih` |
| `DURATION` | Süre (ay, gün, hafta, yıl) | `sure` |
| `ORG` | Kurum adı | `kurum` |
| `GPE` | Yer adı | `lokasyon` |

---

## Proje Yapısı

```
nlp-server/
├── app/
│   ├── main.py
│   ├── config.py
│   ├── models/schemas.py
│   ├── routers/nlp.py
│   └── services/
│       ├── analyzer.py
│       ├── contract_classifier.py
│       └── entity_extractor.py    # Para + süre/tarih ayrımı; DURATION entity
├── data/
│   ├── train/
│   └── models/
├── tests/
├── requirements.txt
└── .env.example
```

---

## Ortam Değişkenleri

| Değişken | Varsayılan | Açıklama |
|----------|-----------|----------|
| `HOST` | `0.0.0.0` | Dinleme adresi |
| `PORT` | `8001` | Port |
| `DEBUG` | `true` | Swagger UI + detaylı log |
| `ALLOWED_ORIGINS` | `http://localhost:8080` | CORS whitelist |
| `INTERNAL_API_KEY` | _(boş)_ | Prod'da set edilmeli |

---

## Test Örnekleri

### curl

```bash
curl -X POST "http://localhost:8001/api/nlp/analyze" \
  -H "Content-Type: application/json" \
  -d '{"text":"Kadikoy dairemi 15000 kira bedeli ile 12 ay kiraya verecegim"}'
```

### PowerShell

```powershell
Invoke-RestMethod -Method Post -Uri "http://localhost:8001/api/nlp/analyze" `
  -ContentType "application/json" `
  -Body (@{ text = "Ali Veli'ye 100.000 TL borc verecegim, 6 ay icinde odeyecek" } | ConvertTo-Json) `
| ConvertTo-Json -Depth 6
```

---

## Ekip

- **Deniz Eren ARICI**
- **Burak DERE** — AI & Data Engineer
