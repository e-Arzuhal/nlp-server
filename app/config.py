"""
e-Arzuhal NLP Server Configuration
"""
import os
from dotenv import load_dotenv

load_dotenv()

# Server
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", 8001))
DEBUG = os.getenv("DEBUG", "true").lower() == "true"

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(DATA_DIR, "models")
TRAIN_DIR = os.path.join(DATA_DIR, "train")
TEST_DIR = os.path.join(DATA_DIR, "test")

# Model files
CONTRACT_TYPE_MODEL_PATH = os.path.join(MODELS_DIR, "contract_type_model.pkl")
LABEL_ENCODER_PATH = os.path.join(MODELS_DIR, "label_encoder.pkl")

# spaCy model (default Turkce)
SPACY_MODEL = os.getenv("SPACY_MODEL", "tr_core_news_sm")

# spaCy kullanimi opsiyonel (yoksa regex/lite moda dusulur)
USE_SPACY = os.getenv("USE_SPACY", "true").lower() == "true"
REQUIRE_SPACY = os.getenv("REQUIRE_SPACY", "false").lower() == "true"

# Runtime'da otomatik model indirme (prod/perf icin kapali onerilir)
ALLOW_SPACY_DOWNLOAD = os.getenv("ALLOW_SPACY_DOWNLOAD", "false").lower() == "true"

# Contract Types (Turkce)
CONTRACT_TYPES = [
    "borc_sozlesmesi",      # Loan Agreement
    "kira_sozlesmesi",      # Rental Contract  
    "hizmet_sozlesmesi",    # Service Agreement
    "satis_sozlesmesi",     # Sales Contract
    "is_sozlesmesi",        # Employment Contract
    "vekaletname",          # Power of Attorney
    "taahhutname",          # Letter of Commitment
]

# Entity mappings
# Not: Türkçe modeller PER/LOC gibi etiketler döndürebilir; normalize etmek için ekledik.
ENTITY_LABELS = {
    # spaCy / OntoNotes benzeri
    "PERSON": "taraf",
    "ORG": "kurum",
    "MONEY": "tutar",
    "DATE": "tarih",
    "GPE": "lokasyon",
    # tr_core_news_* etiketleri
    "PER": "taraf",
    "LOC": "lokasyon",
}
