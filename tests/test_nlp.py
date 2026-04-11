import os
import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.services.contract_classifier import classify_contract
from app.services.postprocessor import extract_all
from app.services.entity_merger import merge_entities
from app.services.chat_intent import sanitize_message, _extract_basic_entities, _mask_entities

client = TestClient(app)

ENTITY_KEYS = {"PERSON", "ORG", "LOC", "MONEY", "DATE", "CARDINAL", "PERCENT"}


# --- Health ---

class TestHealth:
    def test_health_ok(self):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "model_loaded" in data
        assert "model" in data


# --- Root ---

class TestRoot:
    def test_root(self):
        response = client.get("/")
        assert response.status_code == 200
        assert "status" in response.json()


# --- POST /api/v1/extract ---

class TestExtractEndpoint:
    def test_response_structure(self):
        response = client.post("/api/v1/extract", json={"text": "Test metni burada."})
        assert response.status_code == 200
        data = response.json()
        assert "contract_type" in data
        assert "contract_type_confidence" in data
        assert "extracted_entities" in data
        assert "raw_text_length" in data
        assert "processing_time_ms" in data

    def test_all_entity_keys_always_present(self):
        response = client.post("/api/v1/extract", json={"text": "Bu bir test metnidir."})
        assert response.status_code == 200
        entities = response.json()["extracted_entities"]
        assert set(entities.keys()) == ENTITY_KEYS
        for key in ENTITY_KEYS:
            assert isinstance(entities[key], list)

    def test_raw_text_length(self):
        text = "Kısa bir metin."
        response = client.post("/api/v1/extract", json={"text": text})
        assert response.status_code == 200
        assert response.json()["raw_text_length"] == len(text)

    def test_processing_time_present(self):
        response = client.post("/api/v1/extract", json={"text": "Herhangi bir metin."})
        assert response.status_code == 200
        assert response.json()["processing_time_ms"] >= 0

    def test_empty_text_fails(self):
        response = client.post("/api/v1/extract", json={"text": ""})
        assert response.status_code == 422

    def test_missing_text_field_fails(self):
        response = client.post("/api/v1/extract", json={})
        assert response.status_code == 422


# --- Integration: realistic Turkish contract text (requires Ollama running) ---

class TestIntegration:
    def test_is_sozlesmesi(self):
        text = (
            "Bu iş sözleşmesi Ahmet Yılmaz ile ABC Teknoloji A.Ş. arasında "
            "01.03.2025 tarihinde imzalanmıştır. Aylık brüt ücret 25.000 TL "
            "olarak kararlaştırılmıştır. Deneme süresi 2 ay olarak belirlenmiştir. "
            "Çalışma yeri İstanbul, Kadıköy ofisidir."
        )
        response = client.post("/api/v1/extract", json={"text": text})
        assert response.status_code == 200
        data = response.json()
        entities = data["extracted_entities"]

        assert data["contract_type"] == "is_sozlesmesi"
        assert data["contract_type_confidence"] > 0
        assert "Ahmet Yılmaz" in entities["PERSON"]
        assert any("ABC" in org for org in entities["ORG"])
        assert "25.000 TL" in entities["MONEY"]
        assert "01.03.2025" in entities["DATE"]
        assert "2 ay" in entities["CARDINAL"]
        assert any("İstanbul" in loc for loc in entities["LOC"])

    def test_kira_sozlesmesi(self):
        text = (
            "Kiracı Fatma Demir ile kiraya veren Ali Kaya arasında "
            "kira sözleşmesi yapılmıştır. Aylık kira bedeli 15.000 TL, "
            "depozito 30.000 TL. Kira süresi 1 yıllık. Artış oranı %25."
        )
        response = client.post("/api/v1/extract", json={"text": text})
        assert response.status_code == 200
        data = response.json()
        entities = data["extracted_entities"]

        assert data["contract_type"] == "kira_sozlesmesi"
        assert len(entities["MONEY"]) >= 1
        assert len(entities["CARDINAL"]) >= 1
        assert len(entities["PERCENT"]) >= 1


# --- Unit: contract_classifier ---

class TestContractClassifier:
    def test_is_sozlesmesi_detected(self):
        text = "işçi işveren arasında brüt maaş ücret deneme süresi sgk"
        ctype, confidence = classify_contract(text)
        assert ctype == "is_sozlesmesi"
        assert confidence > 0

    def test_kira_sozlesmesi_detected(self):
        text = "kiracı kiraya veren kira bedeli depozito tahliye"
        ctype, confidence = classify_contract(text)
        assert ctype == "kira_sozlesmesi"

    def test_unknown_text_returns_none(self):
        ctype, confidence = classify_contract("Bugün hava çok güzeldi.")
        assert ctype is None
        assert confidence == 0.0

    def test_confidence_between_0_and_1(self):
        _, conf = classify_contract("işçi işveren ücret maaş sgk kıdem tazminatı")
        assert 0.0 <= conf <= 1.0


# --- Unit: postprocessor ---

class TestPostprocessor:
    def test_date_extraction(self):
        result = extract_all("Sözleşme 01.03.2025 tarihinde imzalandı.")
        assert "01.03.2025" in result["DATE"]

    def test_money_tl(self):
        result = extract_all("Ücret 25.000 TL olarak belirlendi.")
        assert "25.000 TL" in result["MONEY"]

    def test_cardinal_duration(self):
        result = extract_all("Deneme süresi 2 ay olarak belirlenmiştir.")
        assert any("2 ay" in c for c in result["CARDINAL"])

    def test_percent(self):
        result = extract_all("Artış oranı %25 olarak uygulanır.")
        assert any("25" in p for p in result["PERCENT"])

    def test_empty_lists_when_nothing_found(self):
        result = extract_all("Herhangi bir metin.")
        assert result["DATE"] == []
        assert result["MONEY"] == []
        assert result["CARDINAL"] == []
        assert result["PERCENT"] == []


# --- Unit: entity_merger ---

class TestEntityMerger:
    def test_all_keys_present(self):
        result = merge_entities(
            {k: [] for k in ENTITY_KEYS},
            {"MONEY": [], "DATE": [], "CARDINAL": [], "PERCENT": []}
        )
        assert set(result.keys()) == ENTITY_KEYS

    def test_llm_and_regex_merged(self):
        llm = {
            "PERSON": ["Ahmet Yılmaz"],
            "ORG": ["ABC A.Ş."],
            "LOC": ["İstanbul"],
            "MONEY": ["25.000 TL"],
            "DATE": ["01.03.2025"],
            "CARDINAL": ["2 ay"],
            "PERCENT": [],
        }
        regex = {"MONEY": ["25.000 TL"], "DATE": ["01.03.2025"], "CARDINAL": ["2 ay"], "PERCENT": ["%10"]}
        result = merge_entities(llm, regex)
        assert result["PERSON"] == ["Ahmet Yılmaz"]
        assert result["ORG"] == ["ABC A.Ş."]
        assert result["LOC"] == ["İstanbul"]
        assert result["MONEY"] == ["25.000 TL"]  # deduplicated
        assert result["DATE"] == ["01.03.2025"]   # deduplicated
        assert result["CARDINAL"] == ["2 ay"]
        assert result["PERCENT"] == ["%10"]        # from regex

    def test_deduplication(self):
        llm = {"PERSON": ["Ahmet", "Ahmet"], "ORG": [], "LOC": [], "DATE": [], "MONEY": [], "CARDINAL": [], "PERCENT": []}
        regex = {"MONEY": [], "DATE": [], "CARDINAL": [], "PERCENT": []}
        result = merge_entities(llm, regex)
        assert result["PERSON"] == ["Ahmet"]


AUTH_HEADER = {"X-Internal-API-Key": os.getenv("INTERNAL_API_KEY", "")}


# --- POST /api/v1/chat-intent ---

class TestChatIntentEndpoint:
    def test_response_structure(self):
        response = client.post(
            "/api/v1/chat-intent",
            json={"message": "Bu sözleşmede cezai şart var mı?"},
            headers=AUTH_HEADER,
        )
        assert response.status_code == 200
        data = response.json()
        assert "intent" in data
        assert "confidence" in data
        assert "sanitized_message" in data
        assert "detected_entities" in data

    def test_confidence_between_0_and_1(self):
        response = client.post(
            "/api/v1/chat-intent",
            json={"message": "Eksik maddeler neler?"},
            headers=AUTH_HEADER,
        )
        assert response.status_code == 200
        assert 0.0 <= response.json()["confidence"] <= 1.0

    def test_empty_message_fails(self):
        response = client.post(
            "/api/v1/chat-intent",
            json={"message": ""},
            headers=AUTH_HEADER,
        )
        assert response.status_code == 422

    def test_missing_message_field_fails(self):
        response = client.post(
            "/api/v1/chat-intent",
            json={},
            headers=AUTH_HEADER,
        )
        assert response.status_code == 422

    def test_no_auth_header_returns_401(self):
        response = client.post(
            "/api/v1/chat-intent",
            json={"message": "Test mesajı"},
        )
        assert response.status_code == 401

    def test_wrong_auth_header_returns_401(self):
        response = client.post(
            "/api/v1/chat-intent",
            json={"message": "Test mesajı"},
            headers={"X-Internal-API-Key": "wrong-key"},
        )
        assert response.status_code == 401


# --- Unit: PII sanitization ---

class TestPIISanitization:
    def test_tc_masked(self):
        entities = _extract_basic_entities("TC kimlik 12345678901 numaralı kişi")
        assert "12345678901" in entities["TC"]

    def test_phone_masked(self):
        msg = "Beni 05551234567 numaradan arayın"
        sanitized = sanitize_message(msg, _extract_basic_entities(msg))
        assert "05551234567" not in sanitized
        assert "[TELEFON]" in sanitized

    def test_email_masked(self):
        msg = "Mail adresim ahmet@example.com olarak kayıtlı"
        sanitized = sanitize_message(msg, _extract_basic_entities(msg))
        assert "ahmet@example.com" not in sanitized
        assert "[E_POSTA]" in sanitized

    def test_person_name_masked(self):
        msg = "Kiracı Ahmet Yılmaz ile görüştük"
        entities = _extract_basic_entities(msg)
        sanitized = sanitize_message(msg, entities)
        assert "Ahmet Yılmaz" not in sanitized
        assert "[KİŞİ_1]" in sanitized

    def test_legal_terms_not_masked_as_person(self):
        entities = _extract_basic_entities("Türk Borçlar Kanunu madde 299")
        assert "Türk Borçlar" not in entities["PERSON"]

    def test_mask_entities_replaces_values(self):
        entities = {"TC": ["12345678901"], "MONEY": ["25.000 TL"], "PERSON": ["Ahmet Yılmaz"]}
        masked = _mask_entities(entities)
        assert masked["TC"] == ["[TC_KİMLİK]_1"]
        assert masked["MONEY"] == ["[TUTAR]_1"]
        assert masked["PERSON"] == ["[KİŞİ]_1"]

    def test_mask_entities_empty_lists_preserved(self):
        entities = {"TC": [], "MONEY": [], "PERSON": []}
        masked = _mask_entities(entities)
        for key in entities:
            assert masked[key] == []


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
