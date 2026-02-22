"""
e-Arzuhal NLP Server Tests
Tests for the Turkish Named Entity Recognition service.
"""
import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.services.extractor import TurkishEntityExtractor, get_extractor


client = TestClient(app)


class TestHealthCheck:
    """Health endpoint tests"""
    
    def test_health_check(self):
        """Test health endpoint returns healthy status"""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "version" in data
        assert "spacy_model_loaded" in data


class TestRootEndpoint:
    """Root endpoint tests"""
    
    def test_root_returns_service_info(self):
        """Test root endpoint returns service information"""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "service" in data
        assert "version" in data
        assert "endpoints" in data


class TestExtractEndpoint:
    """POST /api/extract endpoint tests"""
    
    def test_extract_returns_correct_structure(self):
        """Test that response has correct structure"""
        response = client.post("/api/extract", json={
            "text": "Test metni"
        })
        assert response.status_code == 200
        data = response.json()
        assert "raw_text" in data
        assert "entities" in data
        assert "PERSON" in data["entities"]
        assert "MONEY" in data["entities"]
        assert "LOCATION" in data["entities"]
        assert "DATE" in data["entities"]
        assert "OBJECT_OR_PROPERTY" in data["entities"]
    
    def test_extract_money_tl(self):
        """Test extraction of Turkish Lira amounts"""
        response = client.post("/api/extract", json={
            "text": "Kiracıdan aylık 15.000 TL alacağım, depozito olarak 20.000 TL"
        })
        assert response.status_code == 200
        data = response.json()
        money = data["entities"]["MONEY"]
        assert len(money) >= 2
        assert any("15.000 TL" in m for m in money)
        assert any("20.000 TL" in m for m in money)
    
    def test_extract_money_lira(self):
        """Test extraction of 'lira' format"""
        response = client.post("/api/extract", json={
            "text": "5000 lira ödeme yapılacak"
        })
        assert response.status_code == 200
        data = response.json()
        money = data["entities"]["MONEY"]
        assert len(money) >= 1
    
    def test_extract_date_duration(self):
        """Test extraction of duration expressions"""
        response = client.post("/api/extract", json={
            "text": "Sözleşme 1 yıllığına geçerli olacak, 6 ay sonra yenilenecek"
        })
        assert response.status_code == 200
        data = response.json()
        dates = data["entities"]["DATE"]
        assert len(dates) >= 1
    
    def test_extract_objects(self):
        """Test extraction of property/object types"""
        response = client.post("/api/extract", json={
            "text": "Antalya'daki evimi kiraya vereceğim, depozito alacağım"
        })
        assert response.status_code == 200
        data = response.json()
        objects = data["entities"]["OBJECT_OR_PROPERTY"]
        assert len(objects) >= 1
    
    def test_extract_returns_raw_text(self):
        """Test that raw_text is returned unchanged"""
        original_text = "Ahmet Yılmaz'a ev kiralayacağım"
        response = client.post("/api/extract", json={
            "text": original_text
        })
        assert response.status_code == 200
        data = response.json()
        assert data["raw_text"] == original_text
    
    def test_extract_empty_text_fails(self):
        """Test that empty text returns validation error"""
        response = client.post("/api/extract", json={
            "text": ""
        })
        assert response.status_code == 422  # Validation error
    
    def test_extract_all_entity_types_present(self):
        """Test that all entity type keys are present even when empty"""
        response = client.post("/api/extract", json={
            "text": "Bu bir test metnidir"
        })
        assert response.status_code == 200
        data = response.json()
        entities = data["entities"]
        # All keys must be present
        assert isinstance(entities["PERSON"], list)
        assert isinstance(entities["MONEY"], list)
        assert isinstance(entities["LOCATION"], list)
        assert isinstance(entities["DATE"], list)
        assert isinstance(entities["OBJECT_OR_PROPERTY"], list)


class TestExtractorUnit:
    """Unit tests for TurkishEntityExtractor"""
    
    def test_extractor_singleton(self):
        """Test that get_extractor returns singleton"""
        extractor1 = get_extractor()
        extractor2 = get_extractor()
        assert extractor1 is extractor2
    
    def test_extractor_money_patterns(self):
        """Test money regex patterns directly"""
        extractor = TurkishEntityExtractor()
        
        test_cases = [
            "15.000 TL",
            "20.000,50 ₺",
            "1000 lira",
            "5.500,00 TL",
        ]
        
        for test_text in test_cases:
            result = extractor.extract(test_text)
            assert len(result["MONEY"]) >= 1, f"Failed to extract money from: {test_text}"
    
    def test_extractor_date_patterns(self):
        """Test date regex patterns directly"""
        extractor = TurkishEntityExtractor()
        
        test_cases = [
            "1 yıl",
            "6 ay",
            "3 hafta",
            "10 gün",
            "1 yıllığına",
        ]
        
        for test_text in test_cases:
            result = extractor.extract(test_text)
            assert len(result["DATE"]) >= 1, f"Failed to extract date from: {test_text}"


class TestIntegration:
    """Integration tests with realistic Turkish text"""
    
    def test_full_rental_contract_text(self):
        """Test extraction from realistic rental contract text"""
        text = (
            "Ahmet Yılmaz'a Antalya'daki evimi aylık 15.000 TL'ye "
            "1 yıllığına kiralayacağım. 20.000 TL depozito alacağım."
        )
        response = client.post("/api/extract", json={"text": text})
        assert response.status_code == 200
        data = response.json()
        
        # Should extract money
        assert len(data["entities"]["MONEY"]) >= 2
        
        # Should extract date/duration
        assert len(data["entities"]["DATE"]) >= 1
        
        # Should extract objects
        assert len(data["entities"]["OBJECT_OR_PROPERTY"]) >= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

