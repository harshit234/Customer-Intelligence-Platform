"""
test_payload.py â€” Unit tests for serving schemas and payload validation.
"""
import pytest
from pydantic import ValidationError
from src.serving.schemas import (
    PredictRequest, PredictResponse,
    BatchScoreRequest, BatchScoreResponse, BatchScoreRecordResponse,
    AskComplaintsRequest, AskComplaintsResponse,
    CustomerIntelRequest, CustomerIntelResponse,
    HealthResponse, MetricsResponse
)
@pytest.fixture
def valid_predict_payload():
    return {
        "age": 32,
        "job": "management",
        "marital": "single",
        "education": "tertiary",
        "default": "no",
        "balance": 2343,
        "housing": "yes",
        "loan": "no",
        "contact": "cellular",
        "day": 5,
        "month": "may",
        "campaign": 1,
        "pdays": -1,
        "previous": 0,
        "poutcome": "unknown"
    }
def test_predict_request_validation(valid_predict_payload):
    req = PredictRequest(**valid_predict_payload)
    assert req.age == 32
    assert req.job == "management"
    assert req.marital == "single"
    assert req.balance == 2343
    invalid_payload = valid_predict_payload.copy()
    del invalid_payload["age"]
    with pytest.raises(ValidationError):
        PredictRequest(**invalid_payload)
    invalid_type_payload = valid_predict_payload.copy()
    invalid_type_payload["balance"] = "thousands"
    with pytest.raises(ValidationError):
        PredictRequest(**invalid_type_payload)
def test_predict_response_validation():
    res_data = {
        "prediction": 1,
        "probability": 0.85,
        "threshold_decision": True,
        "model_version": "v1.0.0"
    }
    res = PredictResponse(**res_data)
    assert res.prediction == 1
    assert res.probability == 0.85
    assert res.threshold_decision is True
    assert res.model_version == "v1.0.0"
    invalid_res = res_data.copy()
    del invalid_res["prediction"]
    with pytest.raises(ValidationError):
        PredictResponse(**invalid_res)
def test_batch_score_validation(valid_predict_payload):
    req_data = {
        "records": [valid_predict_payload, valid_predict_payload]
    }
    req = BatchScoreRequest(**req_data)
    assert len(req.records) == 2
    assert req.records[0].age == 32
    with pytest.raises(ValidationError):
        BatchScoreRequest(records="not-a-list")
    res_data = {
        "scores": [
            {
                "record_index": 0,
                "prediction": 0,
                "probability": 0.25,
                "conversion_band": "LOW"
            },
            {
                "record_index": 1,
                "prediction": 1,
                "probability": 0.75,
                "conversion_band": "HIGH"
            }
        ],
        "summary": {"HIGH": 1, "MEDIUM": 0, "LOW": 1},
        "model_version": "v1.0.0"
    }
    res = BatchScoreResponse(**res_data)
    assert len(res.scores) == 2
    assert res.scores[0].conversion_band == "LOW"
    assert res.scores[1].conversion_band == "HIGH"
    assert res.summary == {"HIGH": 1, "MEDIUM": 0, "LOW": 1}
    assert res.model_version == "v1.0.0"
def test_ask_complaints_request_validation():
    payload = {
        "query": "Is there a billing problem?",
        "filters": {"product": "Credit card", "company": "Capital One"}
    }
    req = AskComplaintsRequest(**payload)
    assert req.query == "Is there a billing problem?"
    assert req.filters == {"product": "Credit card", "company": "Capital One"}
    payload_no_filters = {"query": "What is Capital One response time?"}
    req = AskComplaintsRequest(**payload_no_filters)
    assert req.query == "What is Capital One response time?"
    assert req.filters is None
def test_ask_complaints_response_validation():
    res_data = {
        "answer": "Yes, there are major billing disputes.",
        "evidence_ids": ["123", "456"],
        "evidence_sufficiency": "HIGH",
        "prompt_version": "v1.0"
    }
    res = AskComplaintsResponse(**res_data)
    assert res.answer == "Yes, there are major billing disputes."
    assert res.evidence_ids == ["123", "456"]
    assert res.evidence_sufficiency == "HIGH"
    assert res.prompt_version == "v1.0"
def test_customer_intel_validation(valid_predict_payload):
    req_data = {
        "customer_features": valid_predict_payload,
        "complaint_filters": {"company": "Equifax"}
    }
    req = CustomerIntelRequest(**req_data)
    assert req.customer_features.age == 32
    assert req.complaint_filters == {"company": "Equifax"}
    res_data = {
        "conversion_band": "HIGH",
        "probability": 0.82,
        "top_complaint_themes": [
            {
                "complaint_id": "999",
                "score": 0.95,
                "product": "Credit card",
                "issue": "Billing",
                "snippet": "Theme snippet"
            }
        ],
        "model_version": "v1.0",
        "index_version": "idx-v1"
    }
    res = CustomerIntelResponse(**res_data)
    assert res.conversion_band == "HIGH"
    assert res.probability == 0.82
    assert len(res.top_complaint_themes) == 1
    assert res.top_complaint_themes[0]["complaint_id"] == "999"
def test_health_and_metrics_validation():
    health_data = {
        "status": "healthy",
        "model_version": "ml-v1",
        "index_version": "faiss-v1",
        "uptime_seconds": 120.5
    }
    health = HealthResponse(**health_data)
    assert health.status == "healthy"
    assert health.uptime_seconds == 120.5
    metrics_data = {
        "uptime_seconds": 3600.0,
        "request_counts": {"health": 10, "predict": 25},
        "error_counts": {"health": 0, "predict": 1},
        "latencies_ms": {"health": 1.2, "predict": 15.4},
        "predictions_count": {"0": 15, "1": 10},
        "rag_retrievals_count": 5,
        "rag_avg_evidence_count": 3.2
    }
    metrics = MetricsResponse(**metrics_data)
    assert metrics.uptime_seconds == 3600.0
    assert metrics.request_counts["predict"] == 25
    assert metrics.error_counts["predict"] == 1
    assert metrics.latencies_ms["predict"] == 15.4
    assert metrics.predictions_count == {"0": 15, "1": 10}
