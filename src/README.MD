# Data Cleaning Module

This script provides a specialized `ComplaintCleaner` class designed to preprocess and standardize consumer complaint datasets. It handles missing values, removes noise from text narratives, and prepares the data for downstream tasks like embedding and semantic search.

---

## 🛠️ Key Features

* **Null Data Handling**: Automatically identifies and removes rows with missing `Complaint ID` or `Consumer complaint narrative` to ensure data quality.
* **Narrative Cleaning**: Implements a `clean_text` method that standardizes complaint narratives by:
* Converting all text to lowercase.
* Removing special characters and punctuation (keeping only alphanumeric characters and basic whitespace).
* Stripping leading and trailing whitespace.


* **Product Categorization**: Maps varied and complex product descriptions into five simplified categories (e.g., "Credit card or prepaid card" becomes "Credit Card") for easier analysis.
* **Pipeline Ready**: Includes a `process_complaints` method that executes the entire cleaning and transformation workflow on a pandas DataFrame.

---

## 📊 Mapping Strategy

The script standardizes diverse product entries into a consistent taxonomy:

| Original Product Category | Simplified Category |
| --- | --- |
| Credit card / Prepaid card | Credit Card |
| Checking / Savings account | Savings Account |
| Money transfer / Virtual currency | Money Transfers |
| Mortgage / Loans | Personal Loan |
| Other financial services | Others |

---

## 🚀 Quick Start

### Installation

Ensure you have `pandas` installed:

```bash
pip install pandas

```

### Usage

```python
import pandas as pd
from data_cleaning import ComplaintCleaner

# 1. Load your raw dataset
df = pd.read_csv('raw_complaints.csv')

# 2. Initialize and run the cleaner
cleaner = ComplaintCleaner()
cleaned_df = cleaner.process_complaints(df)

# 3. Save the processed data
cleaned_df.to_csv('filtered_complaints.csv', index=False)

```

---

## 📂 File Summary

| File | Responsibility |
| --- | --- |
| `data_cleaning.py` | Contains the `ComplaintCleaner` class for text normalization and column mapping. |
