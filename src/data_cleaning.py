# Text cleaning function
import pandas as pd
import numpy as np
import re

def clean_text(text):
    """Clean and normalize complaint narratives"""
    if pd.isna(text):
        return ""
    
    # Convert to string and lowercase
    text = str(text).lower()
    
    # Remove common boilerplate phrases
    boilerplate_patterns = [
        r"i am writing to file a complaint",
        r"dear sir/madam",
        r"to whom it may concern",
        r"xxxx",  # Often used to redact sensitive info
    ]
    
    for pattern in boilerplate_patterns:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text