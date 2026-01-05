import pytest
from unittest.mock import MagicMock
from src.retriver import ComplaintRetriever

def test_retriever_filtering():
    # Mock dependencies
    mock_vs = MagicMock()
    mock_model = MagicMock()
    
    # Mock search results: one matches filter, one doesn't
    mock_vs.search.return_value = [
        {'product_category': 'Credit Card', 'text': 'Match', 'issue': 'A'},
        {'product_category': 'Mortgage', 'text': 'No Match', 'issue': 'B'}
    ]
    mock_model.encode.return_value = [[0.1, 0.2]]
    
    retriever = ComplaintRetriever(mock_vs, mock_model)
    results = retriever.retrieve("query", k=5, product_filter="Credit Card")
    
    assert len(results) == 1
    assert results[0]['product_category'] == "Credit Card"

