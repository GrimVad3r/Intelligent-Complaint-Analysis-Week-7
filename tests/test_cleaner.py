import pytest
from src.data_cleaning import clean_text
import numpy as np

# Test cases for automated validation
def test_clean_text_basic():
    input_val = "Dear Sir/Madam, I have a problem xxxx."
    expected = "i have a problem ."
    assert clean_text(input_val) == expected

def test_clean_text_nulls():
    assert clean_text(None) == ""
    assert clean_text(np.nan) == ""

def test_clean_text_whitespace():
    input_val = "   too    many    spaces   "
    assert clean_text(input_val) == "too many spaces"

def test_clean_text_boilerplate_removal():
    input_val = "To whom it may concern, I am writing to file a complaint regarding my bill."
    # The boilerplate "I am writing to file a complaint" and "To whom..." should be gone
    result = clean_text(input_val)
    assert "regarding my bill" in result
    assert "to whom it may concern" not in result