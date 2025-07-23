# tests/test_main.py

import pytest
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock
from app.main import app, get_field_generator
from app.rfq_agent import RFQFieldGenerator

# Initialize FastAPI TestClient
client = TestClient(app)

# --------------------------
# Dependency Override Setup
# --------------------------

def mock_rfq_generator_factory(success=True):
    """Factory to return a mocked RFQFieldGenerator"""
    mock_gen = RFQFieldGenerator()
    if success:
        mock_gen.generate_async = AsyncMock(return_value={
            "title": "Mock RFQ",
            "confidence_score": 0.9,
            "success": True,
            "requires_review": False,
            "message": "Mock success"
        })
    else:
        mock_gen.generate_async = AsyncMock(side_effect=Exception("Mock LLM failure"))
    return mock_gen

# --------------------------
# Test: Successful text parse
# --------------------------

@pytest.mark.asyncio
def test_parse_text_success():
    app.dependency_overrides[get_field_generator] = lambda: mock_rfq_generator_factory(success=True)

    response = client.post("/parse-text/", json={"text": "Mock input"})
    
    print(f"[TEST] ✅ test_parse_text_success: Status {response.status_code}, Response: {response.json()}")

    assert response.status_code == 200
    assert response.json()["success"] is True
    assert "confidence_score" in response.json()["data"]

# --------------------------
# Test: Invalid JSON payload
# --------------------------

@pytest.mark.asyncio
def test_parse_text_invalid_json():
    app.dependency_overrides[get_field_generator] = lambda: mock_rfq_generator_factory(success=True)

    response = client.post("/parse-text/", data="not-json")  # invalid content type
    
    print(f"[TEST] ✅ test_parse_text_invalid_json: Status {response.status_code}, Response: {response.json()}")

    assert response.status_code == 400
    assert response.json()["success"] is False
    assert "Invalid JSON" in response.json()["error"]

# --------------------------
# Test: Empty text field
# --------------------------

@pytest.mark.asyncio
def test_parse_text_empty_text():
    app.dependency_overrides[get_field_generator] = lambda: mock_rfq_generator_factory(success=True)

    response = client.post("/parse-text/", json={"text": ""})
    
    print(f"[TEST] ✅ test_parse_text_empty_text: Status {response.status_code}, Response: {response.json()}")

    assert response.status_code == 400
    assert response.json()["success"] is False
    assert "Text field is required" in response.json()["error"]

# --------------------------
# Test: Generator throws exception
# --------------------------

@pytest.mark.asyncio
def test_parse_text_llm_failure():
    app.dependency_overrides[get_field_generator] = lambda: mock_rfq_generator_factory(success=False)

    response = client.post("/parse-text/", json={"text": "Trigger failure"})
    
    print(f"[TEST] ✅ test_parse_text_llm_failure: Status {response.status_code}, Response: {response.json()}")

    assert response.status_code == 500
    assert response.json()["success"] is False
    assert "RFQ processing failed" in response.json()["error"]

# --------------------------
# Cleanup after all tests
# --------------------------

def teardown_module(module):
    """Clear dependency overrides after test module finishes"""
    app.dependency_overrides.clear()
    print("[TEST] ✅ Dependency overrides cleared.")
