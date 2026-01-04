import pytest
import numpy as np
from src.text_embedding import EmbeddingGenerator

def test_embedding_dimensions():
    """Ensure the output vector size matches the model specification."""
    generator = EmbeddingGenerator()
    test_text = ["This is a test."]
    embeddings = generator.generate_embeddings(test_text)
    
    # Check shape: (number of texts, embedding dimension)
    assert embeddings.shape == (1, generator.embedding_dim)
    assert isinstance(embeddings, np.ndarray)

def test_empty_input_handling():
    """Ensure the system doesn't crash on empty input."""
    generator = EmbeddingGenerator()
    result = generator.generate_embeddings([])
    assert len(result) == 0

def test_gpu_cpu_fallback():
    """Verify device selection logic."""
    generator = EmbeddingGenerator()
    assert generator.device in ['cuda', 'cpu']