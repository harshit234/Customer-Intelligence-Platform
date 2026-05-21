"""
test_retrieval.py â€” Unit tests for retrieve.py (FAISS search and metadata filters).
"""
import pytest
import numpy as np
import faiss
from src.rag.retrieve import retrieve
def test_retrieve_filtering():
    dimension = 384
    index = faiss.IndexFlatIP(dimension)
    v1 = np.random.randn(dimension).astype("float32")
    v2 = np.random.randn(dimension).astype("float32")
    v3 = np.random.randn(dimension).astype("float32")
    faiss.normalize_L2(v1.reshape(1, -1))
    faiss.normalize_L2(v2.reshape(1, -1))
    faiss.normalize_L2(v3.reshape(1, -1))
    index.add(np.vstack([v1, v2, v3]))
    metadata = [
        {
            "complaint_id": "1",
            "product": "Credit card",
            "company": "Equifax",
            "date": "2023-05-15",
            "issue": "Billing",
            "narrative": "Dispute on billing card"
        },
        {
            "complaint_id": "2",
            "product": "Mortgage",
            "company": "Wells Fargo",
            "date": "2023-06-20",
            "issue": "Foreclosure",
            "narrative": "Mortgage issues"
        },
        {
            "complaint_id": "3",
            "product": "Credit reporting",
            "company": "Equifax",
            "date": "2023-07-10",
            "issue": "Report error",
            "narrative": "Incorrect report info"
        }
    ]
    class MockModel:
        def encode(self, texts, convert_to_numpy=True):
            return v1.reshape(1, -1)
    model = MockModel()
    res = retrieve(
        query="test query",
        top_k=2,
        threshold=-1.0,
        filters=None,
        index=index,
        metadata=metadata,
        model=model
    )
    assert len(res) == 2
    assert res[0]["complaint_id"] == "1"
    res_product = retrieve(
        query="test",
        top_k=2,
        threshold=-1.0,
        filters={"product": "Mortgage"},
        index=index,
        metadata=metadata,
        model=model
    )
    assert len(res_product) == 1
    assert res_product[0]["complaint_id"] == "2"
    assert res_product[0]["product"] == "Mortgage"
    res_company = retrieve(
        query="test",
        top_k=2,
        threshold=-1.0,
        filters={"company": "Equifax"},
        index=index,
        metadata=metadata,
        model=model
    )
    assert len(res_company) == 2
    assert all(c["company"] == "Equifax" for c in res_company)
    res_date = retrieve(
        query="test",
        top_k=3,
        threshold=-1.0,
        filters={"date_start": "2023-06-01", "date_end": "2023-07-31"},
        index=index,
        metadata=metadata,
        model=model
    )
    assert len(res_date) == 2
    assert set(c["complaint_id"] for c in res_date) == {"2", "3"}
