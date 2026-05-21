import pandas as pd
import numpy as np
import pytest
from src.data_pipeline.features import build_features, build_features_single_row
def test_build_features():
    data = {
        "age": [20, 40],
        "job": ["blue-collar", "management"],
        "marital": ["married", "single"],
        "education": ["secondary", "tertiary"],
        "default": ["no", "yes"],
        "balance": [100, 5000],
        "housing": ["yes", "no"],
        "loan": ["no", "no"],
        "contact": ["cellular", "unknown"],
        "day": [15, 20],
        "month": ["may", "jul"],
        "campaign": [2, 5],
        "pdays": [-1, 10],
        "previous": [0, 3],
        "poutcome": ["unknown", "success"],
        "y": ["no", "yes"]
    }
    df = pd.DataFrame(data)
    X, y = build_features(df, q75_balance=1000.0)
    assert "was_contacted_before" in X.columns
    assert "contact_intensity" in X.columns
    assert "high_balance" in X.columns
    assert "age_band" in X.columns
    assert "quarter" in X.columns
    assert "education" in X.columns
    assert y is not None
    assert (y == [0, 1]).all()
    assert X["age_band"].iloc[0] == 0.0
    assert X["age_band"].iloc[1] == 2.0
    assert X["high_balance"].iloc[0] == 0.0
    assert X["high_balance"].iloc[1] == 1.0
def test_build_features_single_row():
    manifest = {
        "columns": [
            "age", "default", "balance", "housing", "loan", "day", "campaign", "previous",
            "was_contacted_before", "contact_intensity", "high_balance", "age_band", "quarter",
            "education", "job_blue-collar", "job_management", "marital_married", "marital_single",
            "contact_cellular", "contact_unknown", "poutcome_success", "poutcome_unknown"
        ],
        "q75_balance": 1500.0
    }
    payload = {
        "age": 32,
        "job": "blue-collar",
        "marital": "married",
        "education": "secondary",
        "default": "no",
        "balance": 2000,
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
    X = build_features_single_row(payload, manifest)
    assert list(X.columns) == manifest["columns"]
    assert X.shape[0] == 1
    assert X["high_balance"].iloc[0] == 1.0
    assert X["was_contacted_before"].iloc[0] == 0.0
    assert X["contact_intensity"].iloc[0] == 1.0
    assert X["quarter"].iloc[0] == 2.0
