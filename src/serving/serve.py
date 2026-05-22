import time
from typing import Any
from fastapi import FastAPI, HTTPException, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from src.serving.schemas import (
    PredictRequest, PredictResponse,
    BatchScoreRequest, BatchScoreResponse, BatchScoreRecordResponse,
    AskComplaintsRequest, AskComplaintsResponse,
    CustomerIntelRequest, CustomerIntelResponse,
    HealthResponse, MetricsResponse
)
from src.serving.model_loader import (
    load_ml_model, load_faiss_index, load_feature_manifest,
    get_model_version, get_index_version
)
from src.rag.retrieve import load_resources as load_rag_resources
from src.data_pipeline.features import build_features_single_row
from src.rag.answer import answer_question
from src.rag.retrieve import retrieve
app = FastAPI(
    title="Meridian Financial Customer Intelligence API",
    version="1.0.0",
    description="Unified Campaign Conversion (ML) & Complaint Intelligence (LLM/RAG) Service."
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
START_TIME = time.time()
METRICS = {
    "request_counts": {
        "health": 0,
        "predict": 0,
        "batch-score": 0,
        "ask-complaints": 0,
        "customer-intel": 0,
        "metrics": 0,
    },
    "error_counts": {
        "health": 0,
        "predict": 0,
        "batch-score": 0,
        "ask-complaints": 0,
        "customer-intel": 0,
        "metrics": 0,
    },
    "total_latency_ms": {
        "health": 0.0,
        "predict": 0.0,
        "batch-score": 0.0,
        "ask-complaints": 0.0,
        "customer-intel": 0.0,
        "metrics": 0.0,
    },
    "predictions_count": {
        "0": 0,
        "1": 0,
    },
    "rag_retrievals_count": 0,
    "rag_total_evidence_count": 0,
}
@app.on_event("startup")
async def startup_event():
    """Load all models, indices, and feature manifests once at startup."""
    print("\n[startup] Initializing components ...")
    try:
        load_ml_model()
    except Exception as e:
        print(f"[startup] WARNING: ML model could not be loaded at startup: {e}")
    try:
        load_faiss_index()
    except Exception as e:
        print(f"[startup] WARNING: FAISS index could not be loaded at startup: {e}")
    try:
        load_feature_manifest()
    except Exception as e:
        print(f"[startup] WARNING: Feature manifest could not be loaded at startup: {e}")
    print("[startup] Initialization complete.\n")
def get_conversion_band(probability: float) -> str:
    """Categorize prediction probability into a conversion priority band."""
    if probability >= 0.60:
        return "HIGH"
    elif probability >= 0.40:
        return "MEDIUM"
    else:
        return "LOW"
def log_rag_request(
    query: str,
    filters: dict | None,
    evidence_ids: list[str],
    sufficiency: str,
    latency_ms: float
) -> None:
    """Log RAG queries and metadata for monitoring."""
    import json
    from pathlib import Path
    log_dir = Path(__file__).resolve().parents[2] / "monitoring" / "reports"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "rag_request_log.jsonl"
    query_tokens = len(query.split())
    evidence_tokens = len(evidence_ids) * 100
    total_tokens = query_tokens + evidence_tokens + 50
    refusal = sufficiency == "NONE" or len(evidence_ids) == 0
    entry = {
        "timestamp": time.time(),
        "query": query,
        "filters": filters,
        "evidence_count": len(evidence_ids),
        "evidence_sufficiency": sufficiency,
        "max_score": 0.65 if len(evidence_ids) > 0 else 0.0,
        "refusal": refusal,
        "token_count": total_tokens,
        "latency_ms": round(latency_ms, 2)
    }
    try:
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception as e:
        print(f"Error logging RAG query: {e}")
@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint checking readiness of components."""
    t0 = time.perf_counter()
    METRICS["request_counts"]["health"] += 1
    try:
        model_version = get_model_version()
        index_version = get_index_version()
        load_ml_model()
        load_faiss_index()
        uptime = time.time() - START_TIME
        res = HealthResponse(
            status="healthy",
            model_version=model_version,
            index_version=index_version,
            uptime_seconds=uptime
        )
        return res
    except Exception as e:
        METRICS["error_counts"]["health"] += 1
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Service unhealthy: {str(e)}"
        )
    finally:
        latency = (time.perf_counter() - t0) * 1000
        METRICS["total_latency_ms"]["health"] += latency
@app.post("/predict", response_model=PredictResponse)
async def predict(payload: PredictRequest):
    """Run term-deposit subscription prediction for a single customer."""
    t0 = time.perf_counter()
    METRICS["request_counts"]["predict"] += 1
    try:
        model = load_ml_model()
        manifest = load_feature_manifest()
        row_dict = payload.dict()
        X = build_features_single_row(row_dict, manifest)
        probabilities = model.predict_proba(X)
        prob = float(probabilities[0, 1])
        decision = prob >= 0.40
        prediction = 1 if decision else 0
        METRICS["predictions_count"][str(prediction)] += 1
        return PredictResponse(
            prediction=prediction,
            probability=prob,
            threshold_decision=decision,
            model_version=get_model_version()
        )
    except Exception as e:
        METRICS["error_counts"]["predict"] += 1
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction error: {str(e)}"
        )
    finally:
        latency = (time.perf_counter() - t0) * 1000
        METRICS["total_latency_ms"]["predict"] += latency
@app.post("/batch-score", response_model=BatchScoreResponse)
async def batch_score(payload: BatchScoreRequest):
    """Predict term-deposit subscription for a list of customer records."""
    t0 = time.perf_counter()
    METRICS["request_counts"]["batch-score"] += 1
    try:
        model = load_ml_model()
        manifest = load_feature_manifest()
        scores = []
        summary = {"HIGH": 0, "MEDIUM": 0, "LOW": 0}
        for idx, record in enumerate(payload.records):
            X = build_features_single_row(record.dict(), manifest)
            probabilities = model.predict_proba(X)
            prob = float(probabilities[0, 1])
            decision = prob >= 0.40
            prediction = 1 if decision else 0
            METRICS["predictions_count"][str(prediction)] += 1
            band = get_conversion_band(prob)
            summary[band] += 1
            scores.append(BatchScoreRecordResponse(
                record_index=idx,
                prediction=prediction,
                probability=prob,
                conversion_band=band
            ))
        return BatchScoreResponse(
            scores=scores,
            summary=summary,
            model_version=get_model_version()
        )
    except Exception as e:
        METRICS["error_counts"]["batch-score"] += 1
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch scoring error: {str(e)}"
        )
    finally:
        latency = (time.perf_counter() - t0) * 1000
        METRICS["total_latency_ms"]["batch-score"] += latency
@app.post("/ask-complaints", response_model=AskComplaintsResponse)
async def ask_complaints(payload: AskComplaintsRequest):
    """Retrieve relevant CFPB complaints and answer query using Gemini."""
    t0 = time.perf_counter()
    METRICS["request_counts"]["ask-complaints"] += 1
    try:
        index, metadata = load_faiss_index()
        _, _, embed_model = load_rag_resources()
        res = answer_question(
            query=payload.query,
            filters=payload.filters,
            index=index,
            metadata=metadata,
            model=embed_model
        )
        METRICS["rag_retrievals_count"] += 1
        METRICS["rag_total_evidence_count"] += len(res.get("evidence_ids", []))
        latency_ms = (time.perf_counter() - t0) * 1000
        log_rag_request(
            query=payload.query,
            filters=payload.filters,
            evidence_ids=res["evidence_ids"],
            sufficiency=res["evidence_sufficiency"],
            latency_ms=latency_ms
        )
        return AskComplaintsResponse(
            answer=res["answer"],
            evidence_ids=res["evidence_ids"],
            evidence_sufficiency=res["evidence_sufficiency"],
            prompt_version=res["prompt_version"]
        )
    except Exception as e:
        METRICS["error_counts"]["ask-complaints"] += 1
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Grounding QA error: {str(e)}"
        )
    finally:
        latency = (time.perf_counter() - t0) * 1000
        METRICS["total_latency_ms"]["ask-complaints"] += latency
@app.post("/customer-intel", response_model=CustomerIntelResponse)
async def customer_intel(payload: CustomerIntelRequest):
    """Retrieve customer subscription priority alongside matching pain points."""
    t0 = time.perf_counter()
    METRICS["request_counts"]["customer-intel"] += 1
    try:
        model = load_ml_model()
        manifest = load_feature_manifest()
        X = build_features_single_row(payload.customer_features.dict(), manifest)
        probabilities = model.predict_proba(X)
        prob = float(probabilities[0, 1])
        band = get_conversion_band(prob)
        index, metadata = load_faiss_index()
        _, _, embed_model = load_rag_resources()
        job = payload.customer_features.job or ""
        housing = payload.customer_features.housing or ""
        loan = payload.customer_features.loan or ""
        search_query = f"issues regarding credit, billing, or loan payments"
        if housing == "yes":
            search_query += " and mortgage"
        if loan == "yes":
            search_query += " and personal loan"
        if job:
            search_query += f" for customer working in {job}"
        complaints = retrieve(
            query=search_query,
            top_k=3,
            threshold=0.40,
            filters=payload.complaint_filters,
            index=index,
            metadata=metadata,
            model=embed_model
        )
        METRICS["rag_retrievals_count"] += 1
        METRICS["rag_total_evidence_count"] += len(complaints)
        themes = []
        evidence_ids = []
        for c in complaints:
            evidence_ids.append(c["complaint_id"])
            themes.append({
                "complaint_id": c["complaint_id"],
                "score": c["score"],
                "product": c["product"],
                "issue": c["issue"],
                "snippet": c["narrative"][:250] + "..." if len(c["narrative"]) > 250 else c["narrative"]
            })
        latency_ms = (time.perf_counter() - t0) * 1000
        log_rag_request(
            query=search_query,
            filters=payload.complaint_filters,
            evidence_ids=evidence_ids,
            sufficiency="HIGH" if len(evidence_ids) > 0 else "NONE",
            latency_ms=latency_ms
        )
        return CustomerIntelResponse(
            conversion_band=band,
            probability=prob,
            top_complaint_themes=themes,
            model_version=get_model_version(),
            index_version=get_index_version()
        )
    except Exception as e:
        METRICS["error_counts"]["customer-intel"] += 1
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Customer intelligence calculation error: {str(e)}"
        )
    finally:
        latency = (time.perf_counter() - t0) * 1000
        METRICS["total_latency_ms"]["customer-intel"] += latency
@app.get("/metrics", response_model=MetricsResponse)
async def metrics():
    """Gather diagnostic metrics and latencies."""
    t0 = time.perf_counter()
    METRICS["request_counts"]["metrics"] += 1
    try:
        uptime = time.time() - START_TIME
        avg_latencies = {}
        for endpoint, count in METRICS["request_counts"].items():
            if count > 0:
                avg_latencies[endpoint] = round(METRICS["total_latency_ms"][endpoint] / count, 2)
            else:
                avg_latencies[endpoint] = 0.0
        avg_evidence = 0.0
        if METRICS["rag_retrievals_count"] > 0:
            avg_evidence = METRICS["rag_total_evidence_count"] / METRICS["rag_retrievals_count"]
        return MetricsResponse(
            uptime_seconds=uptime,
            request_counts=METRICS["request_counts"],
            error_counts=METRICS["error_counts"],
            latencies_ms=avg_latencies,
            predictions_count=METRICS["predictions_count"],
            rag_retrievals_count=METRICS["rag_retrievals_count"],
            rag_avg_evidence_count=round(avg_evidence, 2)
        )
    except Exception as e:
        METRICS["error_counts"]["metrics"] += 1
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error compiling metrics: {str(e)}"
        )
    finally:
        pass
