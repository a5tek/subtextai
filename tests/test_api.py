"""
Unit and Integration Tests for FastAPI Application (Milestone 10)
"""

import pytest
from fastapi.testclient import TestClient

from app.backend.main import app


@pytest.fixture(scope="module")
def client():
    return TestClient(app)


def test_health_check_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "roberta" in data["model_backbone"].lower()
    assert data["num_classes"] == 4
    assert "NON-CLINICAL" in data["disclaimer"]


def test_demonstration_ui_endpoint(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "Subtext AI" in response.text
    assert "Non-Clinical Research System" in response.text


def test_predict_validation_error(client):
    # Empty string should fail validation
    response = client.post("/api/v1/predict", json={"text": ""})
    assert response.status_code == 422


def test_predict_endpoint_success(client):
    payload = {
        "text": "Midterm is killing me right now, totally exhausted!",
        "calibrate": True,
        "explain": False,
    }
    response = client.post("/api/v1/predict", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert "predicted_class" in data
    assert data["predicted_class"] in [0, 1, 2, 3]
    assert data["severity_label"] in ["Control", "Low Stress", "Moderate Distress", "Severe Crisis"]
    assert "probabilities" in data
    assert len(data["probabilities"]) == 4
    assert pytest.approx(sum(data["probabilities"].values()), abs=1e-2) == 1.0
    assert "uncertainty_score" in data
    assert "uncertainty_rating" in data
    assert "disclaimer" in data
    assert "NON-CLINICAL" in data["disclaimer"]


def test_predict_pii_sanitization_trigger(client):
    payload = {
        "text": "Reach out to user@example.com or visit https://hotline.org",
        "calibrate": True,
        "explain": False,
    }
    response = client.post("/api/v1/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["pii_redacted"] is True
    assert "[EMAIL]" in data["sanitized_text"]
    assert "[URL]" in data["sanitized_text"]


def test_predict_crisis_safety_intervention(client):
    payload = {
        "text": "I have decided to end my life tonight, goodbye world.",
        "calibrate": True,
        "model_type": "classical",
    }
    response = client.post("/api/v1/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["severe_crisis_flag"] is True
    assert data["safety_intervention"] is not None
    assert data["safety_intervention"]["intervention_active"] is True
    assert len(data["safety_intervention"]["hotlines"]) >= 3
    hotline_names = [h["name"] for h in data["safety_intervention"]["hotlines"]]
    assert any("988" in name or "Suicide" in name for name in hotline_names)
    assert any("Crisis Text Line" in name for name in hotline_names)


def test_predict_dark_humor_masking(client):
    payload = {
        "text": "Lol wishing my car would swerve off the bridge on the way to work, classic me haha",
        "calibrate": True,
        "model_type": "classical",
    }
    response = client.post("/api/v1/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["linguistic_nuance"] is not None
    assert data["linguistic_nuance"]["masking_detected"] is True
    assert data["linguistic_nuance"]["masking_type"] == "DARK_HUMOR_MASK"
    # Even if unigram tokens dragged prediction towards control, safety intervention MUST trigger
    assert data["severe_crisis_flag"] is True
    assert data["safety_intervention"] is not None


def test_predict_ambiguous_intent_gaming_mitigation(client):
    payload = {
        "text": "I am so done and cannot take this anymore, lost 5 ranked matches in a row, rage quitting!",
        "calibrate": True,
        "model_type": "classical",
    }
    response = client.post("/api/v1/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["linguistic_nuance"] is not None
    assert data["linguistic_nuance"]["ambiguity_detected"] is True
    assert data["linguistic_nuance"]["context_domain"] == "situational_inconvenience"
    # Situational frustration should not trigger severe crisis hotline alarm
    assert data["severe_crisis_flag"] is False
    assert data["safety_intervention"] is None
    assert data["predicted_class"] != 3

