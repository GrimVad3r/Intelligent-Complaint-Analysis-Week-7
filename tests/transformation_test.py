import pytest
import pandas as pd

def test_chunking_overlap_logic():
    """Verify that chunks are actually created and respect overlap."""
    chunker = ComplaintChunker(chunk_size=50, chunk_overlap=10)
    text = "This is a very long string that should definitely be split into multiple chunks for processing."
    chunks = chunker.chunk_complaint("123", text, {})
    
    assert len(chunks) > 1
    assert "text" in chunks[0]
    assert chunks[0]['total_chunks'] == len(chunks)

def test_chunk_dataset_missing_columns():
    """Ensure the code raises a descriptive error if data is malformed."""
    chunker = ComplaintChunker()
    bad_df = pd.DataFrame({'wrong_column': [1, 2, 3]})
    
    with pytest.raises(ValueError, match="missing required columns"):
        chunker.chunk_dataset(bad_df)

def test_empty_narrative_handling():
    """Ensure None or Empty strings don't break the pipeline."""
    chunker = ComplaintChunker()
    df = pd.DataFrame({
        'Complaint ID': ['1'],
        'cleaned_narrative': [None]
    })
    result = chunker.chunk_dataset(df)
    assert len(result) == 0