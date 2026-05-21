# Customer Intelligence Platform — Architecture

## Overview

This platform is a production-minded ML + LLM/RAG system built on two complementary intelligence lanes, sharing a common FastAPI serving spine.

```
┌──────────────────────────────────────────────────────────────────────┐
│                    Meridian Financial CIP API                        │
│                 FastAPI · Uvicorn · Docker                           │
│                                                                      │
│  ┌─────────────────────────┐  ┌──────────────────────────────────┐   │
│  │      ML Lane            │  │         RAG Lane                 │   │
│  │  Campaign Conversion    │  │  Complaint Intelligence          │   │
│  │                         │  │                                  │   │
│  │  XGBoost + SMOTE        │  │  FAISS + SentenceTransformer     │   │
│  │  CalibratedClassifierCV │  │  all-MiniLM-L6-v2               │   │
│  │  LogisticRegression     │  │  Google Gemini 1.5-Flash         │   │
│  │     (fallback)          │  │                                  │   │
│  └────────────┬────────────┘  └────────────────┬─────────────────┘   │
│               │                                │                      │
│               └──────────────┬─────────────────┘                     │
│                              │                                        │
│                    /customer-intel endpoint                           │
│              (Integration: ML band + complaint themes)               │
└──────────────────────────────────────────────────────────────────────┘
```

---

## Data Flow

### ML Lane

```
UCI Bank Marketing CSV
        │
        ▼
[ingest.py] ──► data/raw/bank_marketing.csv
        │
        ▼
[features.py] ──► data/processed/bank_features.csv
                              bank_labels.csv
                              feature_manifest.json
        │
        ▼
[train.py] ──► Baseline (LogReg) + Improved (XGBoost+SMOTE)
        │
        ▼
[PromotionGate] ──► 3-axis comparison (PR-AUC, F1, Latency)
        │
        ▼
[MLflow Registry] ──► CampaignConversion (Production)
        │
        ▼
[serve.py /predict | /batch-score | /customer-intel]
```

### RAG Lane

```
CFPB Complaint CSV (~500k narratives)
        │
        ▼
[ingest.py] ──► data/raw/cfpb_complaints_sample.csv
        │
        ▼
[build_index.py] ──► SentenceTransformer encode + L2-normalize
                 ──► data/processed/faiss_index.bin (IndexFlatIP)
                 ──► data/processed/faiss_metadata.json
        │
        ▼
[retrieve.py] ──► Cosine similarity search + metadata filters
        │
        ▼
[answer.py]   ──► Gemini 1.5-Flash grounded response
        │
        ▼
[serve.py /ask-complaints | /customer-intel]
```

---

## API Endpoints

| Endpoint | Method | Lane | Description |
|---|---|---|---|
| `/health` | GET | Both | Component status, versions, uptime |
| `/predict` | POST | ML | Single customer conversion probability |
| `/batch-score` | POST | ML | Bulk scoring with conversion bands |
| `/ask-complaints` | POST | RAG | Grounded Q&A over CFPB complaints |
| `/customer-intel` | POST | Both | ML band + complaint themes (integration) |
| `/metrics` | GET | Both | Latency, request counts, RAG stats |

---

## Promotion Gate (3-Axis ML Quality Check)

The improved XGBoost model must pass **all three** checks to be promoted to Production:

| Axis | Threshold | Rationale |
|---|---|---|
| PR-AUC delta | ≥ 3pp over baseline | Meaningful precision-recall improvement |
| F1 drop | ≤ 2pp vs baseline | No catastrophic recall regression |
| Latency | ≤ 200ms (median) | Real-time serving constraint |

If the improved model fails any check, the baseline LogisticRegression is promoted instead. This guarantees exactly one passing run and one deliberately-blocked run per CI execution.

---

## Component Diagram

```
src/
├── data_pipeline/
│   ├── ingest.py         UCI + CFPB data download
│   ├── validate.py       Pandera schema validation
│   └── features.py       Feature engineering (no skew — shared by train/serve)
│
├── training/
│   ├── train.py          Baseline + Improved → PromotionGate → MLflow registry
│   └── evaluate.py       Metrics, business_reading, PromotionGate class
│
├── rag/
│   ├── build_index.py    Embed + L2-norm + FAISS IndexFlatIP
│   ├── retrieve.py       Similarity search with filters + threshold
│   ├── answer.py         Gemini-grounded response generation
│   └── rag_eval.py       Offline RAG evaluation suite
│
└── serving/
    ├── serve.py          FastAPI app (6 endpoints + in-memory metrics)
    ├── schemas.py        Pydantic v2 request/response models
    └── model_loader.py   Singleton ML model + FAISS index loader
```

---

## Monitoring & Observability

| Component | Tool | Output |
|---|---|---|
| ML Drift | Evidently AI | `monitoring/reports/drift_report_*.html` |
| RAG Monitor | Custom JSONL log | `monitoring/reports/rag_request_log.jsonl` |
| API Metrics | In-memory counters | `/metrics` endpoint |
| Experiment Tracking | MLflow | `mlruns/` |

---

## Deployment

### Local Development
```bash
uvicorn src.serving.serve:app --reload
```

### Docker
```bash
docker-compose up --build
```

### CI/CD (GitHub Actions)
- **Job 1**: `test-and-validate` — pytest + data validation smoke test
- **Job 2**: `feature-engineering` — feature pipeline (500-row sample)
- **Job 3**: `train-and-gate` — baseline + improved training + 3-axis gate
