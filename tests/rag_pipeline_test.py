import pytest
from unittest.mock import MagicMock

def test_pipeline_no_results():
    """Verify the pipeline handles empty retrieval gracefully."""
    mock_retriever = MagicMock()
    mock_retriever.retrieve.return_value = []
    mock_retriever.format_context.return_value = "No context."
    
    mock_gen = MagicMock()
    mock_gen.generate.return_value = "I don't know."
    
    pipeline = RAGPipeline(mock_retriever, mock_gen)
    result = pipeline.answer_question("Where is my money?")
    
    assert result['num_sources'] == 0
    assert "I don't know" in result['answer']