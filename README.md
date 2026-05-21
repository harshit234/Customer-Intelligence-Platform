# Customer Intelligence Platform

A production-minded ML + LLM/RAG system for campaign conversion prediction and complaint intelligence.

## Architecture

- **ML Lane**: Predicts campaign conversion (term-deposit subscription) using UCI Bank Marketing data
- **RAG Lane**: Answers complaint intelligence questions over CFPB complaint narratives with cited evidence
- **Shared Spine**: CI/CD gate, model registry, monitoring, drift detection, integration endpoint

## Quick Start

```bash
# 1. Clone and setup
git clone <your-repo-url>
cd customer-intelligence-platform
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt

# 2. Set environment variables
cp .env.example .env
# Edit .env and add your GEMINI_API_KEY

# 3. Download data
python src/data_pipeline/ingest.py

# 4. Validate data
python src/data_pipeline/validate.py

# 5. Train ML model
python src/training/train.py

# 6. Build RAG index
python src/rag/build_index.py

# 7. Start API server
uvicorn src.serving.serve:app --reload
```

## API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Status, model version, index version |
| `/predict` | POST | Campaign conversion prediction |
| `/batch-score` | POST | Batch scoring CSV/JSON |
| `/ask-complaints` | POST | RAG complaint Q&A with cited evidence |
| `/customer-intel` | POST | ML band + complaint themes (integration) |
| `/metrics` | GET | Latency, request counts, RAG stats |

## Project Structure

```
customer-intelligence-platform/
├── data/
│   ├── raw/            # Original downloaded data (gitignored except samples)
│   ├── processed/      # Cleaned, feature-engineered data
│   └── samples/        # Small samples committed to Git
├── src/
│   ├── data_pipeline/  # ingest.py, validate.py, features.py
│   ├── training/       # train.py, evaluate.py
│   ├── serving/        # serve.py (FastAPI), schemas.py, model_loader.py
│   └── rag/            # build_index.py, retrieve.py, answer.py, rag_eval.py
├── tests/              # Unit tests for features, schema, payload, retrieval
├── monitoring/         # ml_drift.py, rag_monitor.py
├── docs/               # Architecture diagram, decision log, RAG eval report
├── .github/workflows/  # CI pipeline YAML
├── requirements.txt
├── .env.example
├── docker-compose.yml
└── reflection.md
```

## Stack

- **ML**: scikit-learn, XGBoost, MLflow
- **RAG**: LangChain, FAISS, Google Gemini API
- **Serving**: FastAPI, Uvicorn, Docker
- **Monitoring**: Evidently AI
- **CI/CD**: GitHub Actions

## Data Sources

- [UCI Bank Marketing Dataset](https://archive.ics.uci.edu/dataset/222/bank+marketing)
- [CFPB Consumer Complaint Database](https://www.consumerfinance.gov/data-research/consumer-complaints/)

> Only small samples (≤500 rows) are committed to Git. Full data is downloaded via `ingest.py`.
