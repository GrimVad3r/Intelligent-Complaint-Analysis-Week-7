import pandas as pd
import numpy as np
import re
import logging

# 1. Resilience: Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def clean_text(text):
    """
    Clean and normalize complaint narratives with error handling and logging.
    """
    # Handle Nulls/NAs gracefully
    if pd.isna(text) or text is None:
        return ""

    try:
        # Convert to string and lowercase
        text = str(text).lower()

        # Define boilerplate patterns
        # Moved outside the loop for slight performance gain
        boilerplate_patterns = [
            r"i am writing to file a complaint",
            r"dear sir/madam",
            r"to whom it may concern",
            r"xxxx", 
        ]

        # 2. Resilience: Robust Regex handling
        for pattern in boilerplate_patterns:
            try:
                text = re.sub(pattern, "", text, flags=re.IGNORECASE)
            except re.error as e:
                logger.error(f"Regex error with pattern '{pattern}': {e}")
                continue

        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text

    except Exception as e:
        # 3. Resilience: Catch-all for unexpected types or memory issues
        logger.critical(f"Unexpected error cleaning text: {e}")
        return ""