# e-Arzuhal NLP Server

Dogal dil isleme servisi - sozlesme tipi siniflandirma ve entity extraction.

## Ozellikler

- **Contract Type Classification**: Metin tabanli sozlesme tipi siniflandirma (TF-IDF + Naive Bayes)
- **Named Entity Recognition**: spaCy ile PERSON, MONEY, DATE, ORG, GPE extraction
- **Turkce Destek**: Turkce para birimi ve tarih pattern'leri
- **Oneriler**: Eksik alan tespiti ve kullanici onerileri

## Desteklenen Sozlesme Tipleri

| Tip | Aciklama |
|-----|----------|
| `borc_sozlesmesi` | Borc/Kredi sozlesmesi |
| `kira_sozlesmesi` | Kira sozlesmesi |
| `hizmet_sozlesmesi` | Hizmet/Danismanlik sozlesmesi |
| `satis_sozlesmesi` | Satis sozlesmesi |
| `is_sozlesmesi` | Is/Istihdam sozlesmesi |
| `vekaletname` | Vekaletname |
| `taahhutname` | Taahhutname |

## Kurulum

```bash
python -m pip install --upgrade pip setuptools wheel

python -m venv venv
venv\Scripts\activate

pip install -r requirements.txt

uvicorn app.main:app --host 0.0.0.0 --port 8001 --reload
```

## spaCy (Opsiyonel)
- Servis **spaCy varsa** spaCy NER kullanir, yoksa otomatik **lite/regex** entity extraction moduna duser.
- spaCy NER kalitesi icin (onerilen): **Python 3.11/3.12** + model indir:
```bash
python -m spacy download en_core_web_sm
```
- Python 3.13’te: `requirements.txt` spaCy’yi otomatik kurmaz; servis regex/lite modda calisir.
  - Isterseniz `USE_SPACY=false` ile zorla lite modda tutabilirsiniz.
  - `REQUIRE_SPACY=true` yaparsaniz spaCy/model yoksa servis acilmaz.

## Notlar (Performans)
- Default olarak `tr_core_news_sm` kullanin (CPU, hizli).
- Server, entity extraction icin spaCy pipeline'inda NER disi bilesenleri disable eder (daha hizli inference).
- Runtime'da otomatik model indirme kapali (deterministik + hizli startup). Gerekirse:
  - `ALLOW_SPACY_DOWNLOAD=true`

## API Endpoints

### POST /api/nlp/analyze
Ana analiz endpoint'i - sozlesme tipi + entity'ler + oneriler

**Request:**
```json
{
  "text": "Ahmet Yilmaz'a 50.000 TL borc verecegim, 6 ay icinde geri odeyecek."
}
```

**Response:**
```json
{
  "success": true,
  "contract_type": "borc_sozlesmesi",
  "confidence": 0.89,
  "entities": [
    {"text": "Ahmet Yilmaz", "label": "PERSON", "mapped_field": "taraf"},
    {"text": "50.000 TL", "label": "MONEY", "mapped_field": "tutar"}
  ],
  "extracted_fields": {
    "taraflar": ["Ahmet Yilmaz"],
    "tutar": "50.000 TL",
    "tarih": "6 ay"
  },
  "suggestions": [
    "Faiz orani belirtmek ister misiniz? (Kullanicilarin %78'i ekliyor)"
  ]
}
```

### POST /api/nlp/classify
Sadece sozlesme tipi siniflandirma

### POST /api/nlp/entities
Sadece entity extraction

### GET /health
Saglik kontrolu

## Proje Yapisi

```
nlp-server/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI app
│   ├── config.py            # Konfigürasyon
│   ├── models/
│   │   ├── __init__.py
│   │   └── schemas.py       # Pydantic modelleri
│   ├── routers/
│   │   ├── __init__.py
│   │   └── nlp.py           # API routes
│   └── services/
│       ├── __init__.py
│       ├── analyzer.py      # Ana analiz servisi
│       ├── contract_classifier.py  # Siniflandirici
│       └── entity_extractor.py     # NER
├── data/
│   ├── train/               # Egitim verisi
│   ├── test/                # Test verisi
│   └── models/              # Kaydedilmis modeller
├── tests/
├── requirements.txt
├── .env.example
└── README.md
```

## Gelistirme

### Model Egitimi

Classifier ilk calistirmada default Turkce egitim verisi ile egitilir.
Daha fazla veri eklemek icin `data/train/` klasorune JSON dosyalari eklenebilir.

### Test

```bash
pytest tests/
```

## Ekip
- **Deniz Eren Arıcı** 
- **Burak DERE** - AI & Data Engineer

## Terminalden deneme (curl)

### Bash / zsh / Git Bash (Windows)
```bash
curl -X POST "http://localhost:8001/api/nlp/analyze" \
  -H "Content-Type: application/json" \
  -d '{"text":"Ahmet Yilmaz'\''a 50.000 TL borc verecegim, 6 ay icinde odeyecek"}'
```

### Windows PowerShell
> Not: PowerShell'de `curl` bazen `Invoke-WebRequest` alias'idir. Gercek curl icin `curl.exe` kullanin.

```powershell
curl.exe -X POST "http://localhost:8001/api/nlp/analyze" `
  -H "Content-Type: application/json" `
  -d "{`"text`":`"Ahmet Yilmaz'a 50.000 TL borc verecegim, 6 ay icinde odeyecek`"}"
```

Alternatif (PowerShell native):
```powershell
Invoke-RestMethod -Method Post -Uri "http://localhost:8001/api/nlp/analyze" `
  -ContentType "application/json" `
  -Body (@{ text = "Ahmet Yilmaz'a 50.000 TL borc verecegim, 6 ay icinde odeyecek" } | ConvertTo-Json)
```

### PowerShell (Invoke-RestMethod) - daha fazla ornek

> Ipucu: Tek seferlik baz URL tanimlayin:
```powershell
$baseUrl = "http://localhost:8001"
```

#### Health check
```powershell
Invoke-RestMethod -Method Get -Uri "$baseUrl/health" | Format-List
```

#### Classify (skor tablosu ile)
```powershell
$r = Invoke-RestMethod -Method Post -Uri "$baseUrl/api/nlp/classify" `
  -ContentType "application/json" `
  -Body (@{ text = "Dairemi kiraya verecegim aylik 15000 TL" } | ConvertTo-Json)

$r | Select-Object contract_type, confidence | Format-List
$r.all_scores.GetEnumerator() | Sort-Object Value -Descending | Format-Table -AutoSize
```

#### Entities (sadece alanlar)
```powershell
Invoke-RestMethod -Method Post -Uri "$baseUrl/api/nlp/entities" `
  -ContentType "application/json" `
  -Body (@{ text = "Kadikoy'deki dukkanimi aylik 20.000 TL'ye kiraya verecegim" } | ConvertTo-Json) `
| Select-Object -ExpandProperty extracted_fields | ConvertTo-Json -Depth 5
```

#### Analyze (tum ciktiyi JSON gor)
```powershell
Invoke-RestMethod -Method Post -Uri "$baseUrl/api/nlp/analyze" `
  -ContentType "application/json" `
  -Body (@{ text = "Ali Veli'ye 100.000 TL borc verecegim, 6 ay icinde odeyecek" } | ConvertTo-Json) `
| ConvertTo-Json -Depth 6
```
