"""
e-Arzuhal NLP Server Tests
"""
import pytest
from fastapi.testclient import TestClient
from app.main import app


client = TestClient(app)


class TestHealthCheck:
    """Health endpoint testleri"""
    
    def test_health_check(self):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["models_loaded"] == True


class TestAnalyzeEndpoint:
    """Analyze endpoint testleri"""
    
    def test_analyze_borc_sozlesmesi(self):
        response = client.post("/api/nlp/analyze", json={
            "text": "Ahmet'e 50.000 TL borc verecegim, 6 ay icinde odeyecek"
        })
        assert response.status_code == 200
        data = response.json()
        assert data["success"] == True
        assert data["contract_type"] == "borc_sozlesmesi"
        assert data["confidence"] > 0.5
    
    def test_analyze_kira_sozlesmesi(self):
        response = client.post("/api/nlp/analyze", json={
            "text": "Evimi kiraya vermek istiyorum aylik 15.000 TL"
        })
        assert response.status_code == 200
        data = response.json()
        assert data["contract_type"] == "kira_sozlesmesi"
    
    def test_analyze_extracts_entities(self):
        response = client.post("/api/nlp/analyze", json={
            "text": "Ali Veli'ye 100.000 TL borc verecegim"
        })
        assert response.status_code == 200
        data = response.json()
        # Tutar cikarilmis olmali
        assert data["extracted_fields"]["tutar"] is not None
    
    def test_analyze_empty_text_fails(self):
        response = client.post("/api/nlp/analyze", json={
            "text": ""
        })
        assert response.status_code == 422  # Validation error


class TestClassifyEndpoint:
    """Classify endpoint testleri"""
    
    def test_classify_returns_scores(self):
        response = client.post("/api/nlp/classify", json={
            "text": "Dairemi kiraya verecegim"
        })
        assert response.status_code == 200
        data = response.json()
        assert "contract_type" in data
        assert "confidence" in data
        assert "all_scores" in data
        assert len(data["all_scores"]) > 0


class TestEntitiesEndpoint:
    """Entities endpoint testleri"""
    
    def test_extract_money(self):
        response = client.post("/api/nlp/entities", json={
            "text": "50.000 TL odeme yapilacak"
        })
        assert response.status_code == 200
        data = response.json()
        assert data["extracted_fields"]["tutar"] is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
