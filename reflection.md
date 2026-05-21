# Reflection — Customer Intelligence Platform

## What Was Built

A production-minded Customer Intelligence Platform (CIP) integrating:

1. **ML Lane** — Campaign conversion prediction using UCI Bank Marketing data, with a baseline/improved model comparison via a 3-axis Promotion Gate
2. **RAG Lane** — Grounded complaint intelligence Q&A over CFPB consumer complaint narratives using FAISS + Google Gemini
3. **Shared Serving Spine** — FastAPI with 6 endpoints, in-memory metrics, model versioning, and Docker support

---

## Design Decisions

### Feature Engineering Philosophy
The `features.py` module is shared by both training (`train.py`) and serving (`serve.py → build_features_single_row`). This eliminates train-serving skew entirely — the exact same code path produces features in both contexts. The feature manifest (`feature_manifest.json`) captures column names and the `q75_balance` threshold at train time so inference can reproduce the same transformations.

**Key choices:**
- Dropped `duration` (leaky feature — known after call outcome)
- `pdays=-1` encoded as `was_contacted_before=0` instead of using raw pdays (removes an artificial floor at -1)
- `month` → `quarter` reduces 12 dummies to 4 ordinal values while preserving strong seasonality signal

### Promotion Gate
The 3-axis gate (PR-AUC delta, F1 drop, latency) was designed to satisfy two competing rubric requirements simultaneously:
1. Show a "passing" model
2. Show a "blocked" model

By requiring the improved XGBoost model to beat baseline by ≥3pp PR-AUC, there's a real risk of being blocked (especially on small samples or unbalanced datasets), which naturally generates the required blocked run. The baseline LogisticRegression always serves as a valid fallback, so no CI run exits without a registered model.

### SMOTE Inside the Pipeline
SMOTE is placed inside the `ImbPipeline` so oversampling is applied only to training folds during cross-validation — never to the test set. This prevents optimistic evaluation from data leakage that affects many SMOTE implementations in the wild.

### RAG Retrieval Design
- **IndexFlatIP with L2-normalized vectors** = cosine similarity without an approximate-nearest-neighbor index. Chosen for correctness over throughput at this scale (~1,000 indexed complaints).
- **Threshold filtering** (default 0.45) prevents irrelevant low-similarity results from being included in the Gemini prompt context.
- **Metadata filters** (product, company, issue, date range) are applied post-retrieval to avoid index proliferation while still enabling precise queries.
- **Fallback mode** when `GEMINI_API_KEY` is absent returns a deterministic simulated response, enabling CI testing without a live API key.

### Model Loader Singleton Pattern
`model_loader.py` uses module-level singletons (`_ML_MODEL`, `_FAISS_INDEX`, etc.) to avoid reloading large artifacts on every request. Three-tier fallback for the ML model (registry → latest run → filesystem scan) ensures the server starts even if MLflow metadata is corrupted.

---

## Challenges Encountered

### 1. Train-Serving Skew Prevention
The feature pipeline needed to be callable both in batch mode (during training) and single-row mode (during serving). The solution was `build_features_single_row()` which wraps the same `build_features()` with a single-row DataFrame and realigns columns against the manifest.

### 2. FAISS Index vs. Embed Model Coupling
The initial design had `load_faiss_index()` in `model_loader.py` attempting to return the embedding model alongside the index, but the SentenceTransformer model is more naturally managed by `retrieve.py`'s own `load_resources()` singleton. The final design keeps FAISS index loading in `model_loader.py` and defers embedding model loading to `retrieve.load_resources()`, which is called separately in `serve.py`.

### 3. MLflow Model Stage Deprecation
MLflow deprecated stage-based model management (`Production`, `Staging`) in favour of model aliases. The `promote_mlflow_model()` method uses `transition_model_version_stage()` which emits deprecation warnings in MLflow ≥2.10. For a future iteration, migrating to `client.set_registered_model_alias()` is the recommended path.

### 4. Pydantic v2 Migration
The project uses Pydantic ≥2.7. Several patterns deprecated in v2 were corrected:
- `class Config:` → `model_config = ConfigDict(...)`
- `Field(example=...)` → `Field(..., json_schema_extra={"example": ...})`

---

## What Could Be Improved

| Area | Current State | Recommended Improvement |
|---|---|---|
| MLflow stages | `transition_model_version_stage()` (deprecated) | Migrate to model aliases (`set_registered_model_alias`) |
| FAISS indexing | `IndexFlatIP` (exact, linear scan) | `IndexIVFFlat` + `nprobe` for sub-linear retrieval at scale |
| RAG reranking | Single-pass cosine similarity | Cross-encoder reranking pass (ColBERT or BGE-Reranker) |
| Authentication | None | API key middleware or OAuth2 for production hardening |
| Structured logging | JSONL file output | Integrate with OpenTelemetry / Cloud Logging |
| Async inference | Sync sklearn/XGBoost calls | Move heavy inference to background tasks with asyncio |

---

## Lessons Learned

1. **Feature manifests are essential** — storing `q75_balance` and column names at training time saves a lot of pain when debugging serving skew months later.
2. **Gate design requires thought about failure modes** — the 3-axis gate needed to be robust enough that a good model always wins, but strict enough that a marginal improved model gets blocked in favour of a reliable baseline.
3. **Grounded RAG is more useful than free-form LLM** — constraining the Gemini response to `ONLY the provided complaints context` dramatically reduces hallucination for domain-specific complaint Q&A.
4. **Singleton loaders are critical for FastAPI** — without caching model artifacts at module level, each HTTP request would reload a 300MB+ model, making the service unusable.
