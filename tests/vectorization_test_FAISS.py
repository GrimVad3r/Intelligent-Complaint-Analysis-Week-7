import pytest
import numpy as np
import os
from src.text_vectorization import FAISSVectorStore

def test_faiss_integrity_check():
    """Verify that adding mismatched data raises an exception."""
    store = FAISSVectorStore(embedding_dim=4)
    vecs = np.random.rand(2, 4)
    meta = [{"id": 1}] # Missing one metadata dict
    
    with pytest.raises(ValueError):
        store.add_embeddings(vecs, meta)

def test_faiss_search_self_consistency():
    """Verify that searching for a vector returns itself with distance ~0."""
    store = FAISSVectorStore(embedding_dim=2)
    vec = np.array([[1.0, 0.0]], dtype='float32')
    store.add_embeddings(vec, [{"id": "exact_match"}])
    
    result = store.search(vec, k=1)
    assert result[0]['id'] == "exact_match"
    assert result[0]['distance'] < 1e-6

def test_faiss_save_load(tmp_path):
    """Verify persistence using a temporary directory."""
    idx_p = str(tmp_path / "test.index")
    meta_p = str(tmp_path / "test.pkl")
    
    store = FAISSVectorStore(embedding_dim=2)
    store.add_embeddings(np.array([[1.0, 1.0]]), [{"data": "val"}])
    store.save(idx_p, meta_p)
    
    new_store = FAISSVectorStore.load(idx_p, meta_p, embedding_dim=2)
    assert new_store.index.ntotal == 1
    assert new_store.metadata[0]["data"] == "val"