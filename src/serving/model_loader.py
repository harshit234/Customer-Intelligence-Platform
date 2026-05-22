
import os
import json
import pickle
from pathlib import Path
import mlflow
import mlflow.sklearn
import faiss

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = ROOT / "data" / "processed"
INDEX_PATH = PROCESSED_DIR / "faiss_index.bin"
METADATA_PATH = PROCESSED_DIR / "faiss_metadata.json"
MANIFEST_PATH = PROCESSED_DIR / "feature_manifest.json"

# Set tracking URI to local file store.
# MLflow 2.10+ requires 'file:///' scheme for local paths; absolute Windows paths
# without the scheme are rejected. We normalise here so both env-var and default
# cases produce a valid URI.
_raw_uri = os.getenv("MLFLOW_TRACKING_URI", str(ROOT / "mlruns"))
if not _raw_uri.startswith(("file://", "sqlite://", "http://", "https://", "databricks")):
    # Convert a bare local path to a file:// URI
    _p = Path(_raw_uri).resolve()
    MLFLOW_TRACKING_URI = _p.as_uri()          # e.g. file:///C:/Users/.../mlruns
else:
    MLFLOW_TRACKING_URI = _raw_uri

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# Singletons
_ML_MODEL = None
_FAISS_INDEX = None
_FAISS_METADATA = None
_FEATURE_MANIFEST = None

_MODEL_VERSION = "unknown"
_INDEX_VERSION = "unknown"


def load_ml_model():
    """
    Load the CampaignConversion ML model.
    Tries the Production stage first, falling back to the latest experiment run,
    and finally scanning mlruns directly on disk.
    """
    global _ML_MODEL, _MODEL_VERSION
    if _ML_MODEL is not None:
        return _ML_MODEL

    model_name = "CampaignConversion"
    print(f"Loading ML model '{model_name}' (Tracking URI: {MLFLOW_TRACKING_URI}) ...")

    # Fallback Path 1: MLflow registry (Production stage)
    try:
        model_uri = f"models:/{model_name}/Production"
        _ML_MODEL = mlflow.sklearn.load_model(model_uri)
        _MODEL_VERSION = "mlflow-registry-production"
        print(f"[OK] Successfully loaded model from registry: {model_uri}")
        return _ML_MODEL
    except Exception as e:
        print(f"MLflow Registry Production load failed: {e}. Trying fallback...")

    # Fallback Path 2: Find latest run in MLflow tracking runs
    try:
        client = mlflow.tracking.MlflowClient()
        # Find experiments
        exps = client.search_experiments()
        # Check both production and CI experiment names
        exp_names = ["CampaignConversion", "CI-CampaignConversion"]
        matching_exps = [e for e in exps if e.name in exp_names]
        
        for exp in matching_exps:
            runs = client.search_runs(
                experiment_ids=[exp.experiment_id],
                order_by=["attribute.start_time DESC"],
                max_results=5
            )
            for run in runs:
                # check if model artifact exists
                model_uri = f"runs:/{run.info.run_id}/model"
                try:
                    _ML_MODEL = mlflow.sklearn.load_model(model_uri)
                    _MODEL_VERSION = f"run-{run.info.run_id[:8]}"
                    print(f"[OK] Successfully loaded model from run ID: {run.info.run_id}")
                    return _ML_MODEL
                except Exception:
                    continue
    except Exception as e:
        print(f"MLflow tracking search failed: {e}. Trying local filesystem fallback...")

    # Fallback Path 3: Scan local mlruns directory recursively for model.pkl.
    # Handles both old layout (**/model/model.pkl) and new MLflow >=2.9 layout
    # (**/models/m-*/artifacts/model.pkl).
    try:
        mlruns_dir = ROOT / "mlruns"
        if mlruns_dir.exists():
            # Gather candidates from both layouts
            candidates = list(mlruns_dir.glob("**/model/model.pkl"))
            candidates += list(mlruns_dir.glob("**/models/m-*/artifacts/model.pkl"))
            # Also handle the flat layout used by new MLflow file store
            candidates += list(mlruns_dir.glob("*/models/m-*/artifacts/model.pkl"))

            pkl_files = sorted(
                candidates,
                key=lambda p: p.stat().st_mtime,
                reverse=True
            )
            if pkl_files:
                target_path = pkl_files[0]
                with open(target_path, "rb") as f:
                    # In MLflow, model.pkl is a scikit-learn / imbalanced-learn Pipeline
                    _ML_MODEL = pickle.load(f)
                _MODEL_VERSION = f"local-file-{target_path.parts[-3][:8]}"
                print(f"[OK] Successfully loaded model from local path: {target_path}")
                return _ML_MODEL
    except Exception as e:
        print(f"Local filesystem fallback failed: {e}")

    raise RuntimeError("Failed to load ML model from any location. Run train.py first.")


def load_faiss_index():
    """Load FAISS vector index and metadata from processed directory."""
    global _FAISS_INDEX, _FAISS_METADATA, _INDEX_VERSION
    if _FAISS_INDEX is not None and _FAISS_METADATA is not None:
        return _FAISS_INDEX, _FAISS_METADATA

    if not INDEX_PATH.exists() or not METADATA_PATH.exists():
        raise FileNotFoundError(
            f"FAISS index or metadata files not found. Run build_index.py first."
        )

    print(f"Loading FAISS index from {INDEX_PATH} ...")
    _FAISS_INDEX = faiss.read_index(str(INDEX_PATH))
    
    # Simple version string based on index size and modification time
    mtime = INDEX_PATH.stat().st_mtime
    _INDEX_VERSION = f"faiss-{_FAISS_INDEX.ntotal}-vectors-{int(mtime)}"

    print(f"Loading FAISS metadata from {METADATA_PATH} ...")
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        _FAISS_METADATA = json.load(f)

    return _FAISS_INDEX, _FAISS_METADATA


def load_feature_manifest():
    """Load the feature engineering manifest containing columns and statistics."""
    global _FEATURE_MANIFEST
    if _FEATURE_MANIFEST is not None:
        return _FEATURE_MANIFEST

    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(
            f"Feature manifest not found at {MANIFEST_PATH}. Run features.py first."
        )

    with open(MANIFEST_PATH, "r") as f:
        _FEATURE_MANIFEST = json.load(f)
    print(f"[OK] Loaded feature manifest containing {len(_FEATURE_MANIFEST.get('columns', []))} columns.")
    return _FEATURE_MANIFEST


def get_model_version() -> str:
    """Return version identifier of the loaded ML model."""
    return _MODEL_VERSION


def get_index_version() -> str:
    """Return version identifier of the loaded FAISS index."""
    return _INDEX_VERSION
