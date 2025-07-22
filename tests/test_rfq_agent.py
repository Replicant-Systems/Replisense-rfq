import pytest
from app.rfq_agent import RFQFieldGenerator

def test_generate_valid_json(monkeypatch):
    agent = RFQFieldGenerator()
    
    def mock_generate_reply(msgs):
        return {"title": "Test RFQ", "confidence_score": 0.9}
    
    monkeypatch.setattr(agent.agent, "generate_reply", mock_generate_reply)

    result = agent.generate("RFQ sample text")
    assert result["success"] is True
    assert "title" in result

def test_generate_invalid_json(monkeypatch):
    agent = RFQFieldGenerator()

    # Simulate string return with bad JSON
    monkeypatch.setattr(agent.agent, "generate_reply", lambda x: '{"invalid": ')
    result = agent.generate("bad json")
    assert result["success"] is False

def test_generate_non_dict(monkeypatch):
    agent = RFQFieldGenerator()
    monkeypatch.setattr(agent.agent, "generate_reply", lambda x: 42)
    result = agent.generate("wrong type")
    assert result["success"] is False
