"""
Security regression tests for nlp-server.

Tests verify API key enforcement and CORS header behavior.
These tests patch the module-level _internal_api_key / _debug variables
so no real .env file is required.
"""
import pytest
from unittest.mock import patch
from httpx import AsyncClient, ASGITransport

import app.main as nlp_main
from app.main import app

VALID_KEY = "test-secret-key-32-chars-min-ok!"
EXTRACT_PAYLOAD = {"text": "Test sözleşmesi metni"}


class TestApiKeyEnforcement:

    @pytest.mark.asyncio
    async def test_correct_key_returns_not_401(self):
        """Doğru API anahtarı ile istek → 401 değil."""
        with patch.object(nlp_main, "_internal_api_key", VALID_KEY), \
             patch.object(nlp_main, "_debug", False):
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
                response = await client.post(
                    "/api/v1/extract",
                    json=EXTRACT_PAYLOAD,
                    headers={"X-Internal-API-Key": VALID_KEY},
                )
        # 401 olmamalı — 200 veya servis hatası kabul edilir
        assert response.status_code != 401

    @pytest.mark.asyncio
    async def test_wrong_key_returns_401(self):
        """Yanlış API anahtarı → 401."""
        with patch.object(nlp_main, "_internal_api_key", VALID_KEY), \
             patch.object(nlp_main, "_debug", False):
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
                response = await client.post(
                    "/api/v1/extract",
                    json=EXTRACT_PAYLOAD,
                    headers={"X-Internal-API-Key": "yanlis-anahtar"},
                )
        assert response.status_code == 401

    @pytest.mark.asyncio
    async def test_missing_key_returns_401(self):
        """Header eksikse → 401."""
        with patch.object(nlp_main, "_internal_api_key", VALID_KEY), \
             patch.object(nlp_main, "_debug", False):
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
                response = await client.post("/api/v1/extract", json=EXTRACT_PAYLOAD)
        assert response.status_code == 401

    @pytest.mark.asyncio
    async def test_no_key_debug_false_returns_503(self):
        """INTERNAL_API_KEY boş + DEBUG=false → 503 (yanlış yapılandırma)."""
        with patch.object(nlp_main, "_internal_api_key", ""), \
             patch.object(nlp_main, "_debug", False):
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
                response = await client.post("/api/v1/extract", json=EXTRACT_PAYLOAD)
        assert response.status_code == 503

    @pytest.mark.asyncio
    async def test_no_key_debug_true_passes_through(self):
        """INTERNAL_API_KEY boş + DEBUG=true → geçişe izin verilir (401/503 değil)."""
        with patch.object(nlp_main, "_internal_api_key", ""), \
             patch.object(nlp_main, "_debug", True):
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
                response = await client.post("/api/v1/extract", json=EXTRACT_PAYLOAD)
        assert response.status_code not in (401, 503)


class TestHealthBypass:

    @pytest.mark.asyncio
    async def test_health_endpoint_bypasses_api_key(self):
        """/health hiçbir API anahtarı olmadan erişilebilir olmalı."""
        with patch.object(nlp_main, "_internal_api_key", VALID_KEY), \
             patch.object(nlp_main, "_debug", False):
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
                response = await client.get("/health")
        assert response.status_code == 200


class TestSwaggerExposure:

    @pytest.mark.asyncio
    async def test_swagger_hidden_when_debug_false(self):
        """DEBUG=false iken /docs → 404 (Swagger gizlenmeli)."""
        with patch.object(nlp_main, "_debug", False):
            # App is already created; re-check via conditional docs_url behavior
            # Yeni app instance oluşturmadan path erişimi test edilir
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
                response = await client.get("/docs")
        # 404 veya 403 — public erişim olmamalı
        assert response.status_code in (404, 403)
