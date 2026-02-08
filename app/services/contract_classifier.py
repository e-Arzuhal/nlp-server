"""
e-Arzuhal NLP Server - Contract Type Classifier
Metin tabanli sozlesme tipi siniflandirma
"""
import os
import json
import joblib
import numpy as np
from typing import Dict, Tuple, Optional, List
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

from app.config import CONTRACT_TYPE_MODEL_PATH, LABEL_ENCODER_PATH, TRAIN_DIR


class ContractClassifier:
    """
    Sozlesme tipi siniflandirici
    TF-IDF + Naive Bayes kullanir
    """
    
    def __init__(self):
        self.model: Optional[Pipeline] = None
        self.label_encoder: Optional[LabelEncoder] = None
        self._load_or_create_model()
    
    def _load_or_create_model(self):
        """Modeli yukle veya yeni olustur"""
        if os.path.exists(CONTRACT_TYPE_MODEL_PATH) and os.path.exists(LABEL_ENCODER_PATH):
            try:
                self.model = joblib.load(CONTRACT_TYPE_MODEL_PATH)
                self.label_encoder = joblib.load(LABEL_ENCODER_PATH)
                print("Model yuklendi:", CONTRACT_TYPE_MODEL_PATH)
                return
            except Exception as e:
                print(f"Model yukleme hatasi: {e}")
        
        # Yeni model olustur ve default data ile egit
        print("Yeni model olusturuluyor...")
        self._create_and_train_default()
    
    def _create_and_train_default(self):
        """Default egitim verisi ile model olustur (once data/train/contracts.json dene)"""
        texts, labels = self._load_training_from_json(os.path.join(TRAIN_DIR, "contracts.json"))

        if not texts:
            # Turkce egitim verisi
            training_data = [
                # Borc Sozlesmesi
                ("borca para verecegim odeyecek", "borc_sozlesmesi"),
                ("kredi verecegim geri odeme", "borc_sozlesmesi"),
                ("borc alacak taksit odeme", "borc_sozlesmesi"),
                ("para odunc verecegim iade", "borc_sozlesmesi"),
                ("borc sozlesmesi duzenlemek istiyorum", "borc_sozlesmesi"),
                ("arkadasa para verecegim geri alacagim", "borc_sozlesmesi"),
                ("faiz ile borc vermek istiyorum", "borc_sozlesmesi"),
                ("taksitle geri odeme borc", "borc_sozlesmesi"),
                
                # Kira Sozlesmesi
                ("ev kiralamak istiyorum aylik kira", "kira_sozlesmesi"),
                ("daire kiralayacagim depozito", "kira_sozlesmesi"),
                ("kira sozlesmesi ev daire", "kira_sozlesmesi"),
                ("kiralama aylik odeme ev", "kira_sozlesmesi"),
                ("kiraci ev sahibi kontrat", "kira_sozlesmesi"),
                ("dukkan kiralama isyeri", "kira_sozlesmesi"),
                ("kira bedeli aylik yillik", "kira_sozlesmesi"),
                ("mulk kiralama sozlesme", "kira_sozlesmesi"),
                
                # Hizmet Sozlesmesi
                ("hizmet verecegim is yapacagim", "hizmet_sozlesmesi"),
                ("danismanlik hizmeti sunacagim", "hizmet_sozlesmesi"),
                ("freelance is proje teslim", "hizmet_sozlesmesi"),
                ("hizmet bedeli is tamamlama", "hizmet_sozlesmesi"),
                ("yazilim gelistirme hizmeti", "hizmet_sozlesmesi"),
                ("tasarim hizmeti grafik web", "hizmet_sozlesmesi"),
                ("danismanlik sozlesmesi", "hizmet_sozlesmesi"),
                ("proje bazli calisma hizmet", "hizmet_sozlesmesi"),
                
                # Satis Sozlesmesi
                ("satis yapacagim urun mal", "satis_sozlesmesi"),
                ("arac satisi araba devir", "satis_sozlesmesi"),
                ("mal satimi teslim bedel", "satis_sozlesmesi"),
                ("satis sozlesmesi alim satim", "satis_sozlesmesi"),
                ("urun satisi fatura teslim", "satis_sozlesmesi"),
                ("gayrimenkul satisi tapu devir", "satis_sozlesmesi"),
                ("ikinci el satis devir", "satis_sozlesmesi"),
                
                # Is Sozlesmesi
                ("ise alacagim calisan maas", "is_sozlesmesi"),
                ("istihdam sozlesmesi personel", "is_sozlesmesi"),
                ("calisan ise baslama ucret", "is_sozlesmesi"),
                ("is sozlesmesi maas izin", "is_sozlesmesi"),
                ("personel alimi is iliskisi", "is_sozlesmesi"),
                ("tam zamanli calisma sozlesme", "is_sozlesmesi"),
                ("part time is anlasma", "is_sozlesmesi"),
                
                # Vekaletname
                ("vekalet vermek istiyorum temsil", "vekaletname"),
                ("vekaletname yetki devir", "vekaletname"),
                ("adima islem yapma yetkisi", "vekaletname"),
                ("temsil yetkisi vekil", "vekaletname"),
                ("noter vekaletname islem", "vekaletname"),
                
                # Taahhutname
                ("taahhut ediyorum sorumluluk", "taahhutname"),
                ("taahhutname yukumluluk", "taahhutname"),
                ("sorumluluk beyan taahhut", "taahhutname"),
                ("garanti taahhut belge", "taahhutname"),
            ]
            texts = [t[0] for t in training_data]
            labels = [t[1] for t in training_data]

        self.label_encoder = LabelEncoder()
        encoded_labels = self.label_encoder.fit_transform(labels)

        self.model = Pipeline(
            [
                ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=1000, lowercase=True)),
                ("clf", MultinomialNB(alpha=0.1)),
            ]
        )

        self.model.fit(texts, encoded_labels)

        os.makedirs(os.path.dirname(CONTRACT_TYPE_MODEL_PATH), exist_ok=True)
        joblib.dump(self.model, CONTRACT_TYPE_MODEL_PATH)
        joblib.dump(self.label_encoder, LABEL_ENCODER_PATH)
        print("Model egitildi ve kaydedildi")

    def _load_training_from_json(self, path: str) -> Tuple[List[str], List[str]]:
        try:
            if not os.path.exists(path):
                return [], []
            raw = ""
            with open(path, "r", encoding="utf-8") as f:
                raw = f.read()

            # JSON dosyasina yanlislikla // yorum satiri eklenirse tolerant ol
            raw = "\n".join(line for line in raw.splitlines() if not line.lstrip().startswith("//"))
            data = json.loads(raw)

            texts = [row.get("text") for row in data if row.get("text") and row.get("type")]
            labels = [row.get("type") for row in data if row.get("text") and row.get("type")]
            if texts:
                print(f"Egitim verisi yuklendi: {path} (n={len(texts)})")
            return texts, labels
        except Exception as e:
            print(f"Egitim verisi okunamadi ({path}): {e}")
            return [], []

    def predict(self, text: str) -> Tuple[str, float, Dict[str, float]]:
        """
        Metin icin sozlesme tipini tahmin et
        
        Args:
            text: Siniflandirilacak metin
            
        Returns:
            (contract_type, confidence, all_scores)
        """
        if not self.model or not self.label_encoder:
            raise RuntimeError("Model yuklenmemis")
        
        # Tahmin
        proba = self.model.predict_proba([text])[0]
        predicted_idx = np.argmax(proba)
        confidence = float(proba[predicted_idx])
        
        # Tip
        contract_type = self.label_encoder.inverse_transform([predicted_idx])[0]
        
        # Tum skorlar
        all_scores = {}
        for idx, score in enumerate(proba):
            label = self.label_encoder.inverse_transform([idx])[0]
            all_scores[label] = float(score)
        
        return contract_type, confidence, all_scores
    
    def train(self, texts: List[str], labels: List[str]):
        """
        Yeni veri ile modeli yeniden egit
        
        Args:
            texts: Egitim metinleri
            labels: Etiketler
        """
        encoded_labels = self.label_encoder.fit_transform(labels)
        self.model.fit(texts, encoded_labels)
        
        # Kaydet
        joblib.dump(self.model, CONTRACT_TYPE_MODEL_PATH)
        joblib.dump(self.label_encoder, LABEL_ENCODER_PATH)


# Singleton instance
_classifier: Optional[ContractClassifier] = None


def get_contract_classifier() -> ContractClassifier:
    """Contract classifier singleton'i dondur"""
    global _classifier
    if _classifier is None:
        _classifier = ContractClassifier()
    return _classifier
