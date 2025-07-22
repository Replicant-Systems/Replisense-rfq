import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_parse_text_success(monkeypatch):
    def mock_generate(raw_text, source_file):
        return {"success": True, "message": "Parsed"}
    monkeypatch.setattr("main.field_generator.generate", mock_generate)

    response = client.post("/parse-text/", json={"text": "Hello RFQ!"})
    assert response.status_code == 200
    assert response.json()["success"] is True

def test_parse_text_failure(monkeypatch):
    def mock_generate(raw_text, source_file):
        raise Exception("Mock LLM fail")
    monkeypatch.setattr("main.field_generator.generate", mock_generate)

    response = client.post("/parse-text/", json={"text": "fail"})
    assert response.status_code == 200
    assert response.json()["success"] is False
