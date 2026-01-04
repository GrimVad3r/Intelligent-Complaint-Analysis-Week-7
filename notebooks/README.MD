# Exploratory Data Analysis (EDA) of Consumer Complaints

This Jupyter notebook contains the Exploratory Data Analysis phase of the Intelligent Complaint Analysis project. It focuses on understanding the distribution, characteristics, and quality of consumer complaint narratives before they are processed for semantic indexing.

---

## 📊 Analysis Objectives

The primary goal of this notebook is to perform a deep dive into the cleaned dataset to justify subsequent architectural decisions, such as chunking size and sampling strategies.

### 1. Data Loading & Initial Inspection

* **Source**: Loads the filtered and cleaned complaint dataset from `filtered_complaints.csv`.
* **Scope**: Inspects core features including `product_category`, `cleaned_narrative`, `Company`, and `State`.

### 2. Category Distribution

* **Proportional Analysis**: Evaluates the volume of complaints across the four primary categories: **Credit Card**, **Savings Account**, **Money Transfers**, and **Personal Loan**.
* **Sampling Foundation**: This distribution serves as the basis for the stratified sampling logic implemented in later stages of the pipeline.

### 3. Narrative Length Analysis

* **Tokenization & Counting**: Calculates the length of complaint narratives to determine the distribution of text sizes.
* **Strategic Justification**: Analyzing text length provides the empirical evidence needed to set the `chunk_size` (500) and `chunk_overlap` (50) in the `ComplaintChunker`.
* **Outlier Detection**: Identifies extremely short or excessively long complaints that may impact embedding quality.

### 4. Categorical Insights

* **Company Trends**: Identifies which companies receive the highest volume of complaints within the sampled data.
* **Geographic Trends**: Maps complaints by state to identify regional reporting patterns.

---

## 🛠️ Tools Used

| Library | Purpose |
| --- | --- |
| **Pandas** | Data manipulation and category frequency analysis. |
| **NumPy** | Numerical operations for length calculations. |
| **Matplotlib/Seaborn** | Visualization of distributions and correlations. |
| **Scikit-learn** | Initial testing of stratified split logic. |

---

## 📈 Key Findings for the Pipeline

* **Imbalance Handling**: Confirmed the need for stratified sampling to ensure smaller categories like "Personal Loan" are adequately represented.
* **Chunking Logic**: Determined that a recursive splitter is necessary because complaint narratives vary significantly in length and structure.
# Intelligent Complaint Analysis - Pipeline Documentation

This repository contains a modular pipeline for converting raw consumer complaint narratives into a searchable, semantically indexed vector database. The workflow encompasses data sampling, text chunking, embedding generation, and multi-backend vector storage.

---

## 📋 Pipeline Overview

The analysis follows a structured multi-stage process to transform unstructured text into high-dimensional vectors suitable for similarity search.

### 1. Stratified Sampling

To manage computational resources while maintaining data integrity, the pipeline begins with a stratified sampling strategy:

* **Target Size**: 12,000 complaints.
* **Proportional Representation**: Ensures the sample mirrors the original distribution across product categories (e.g., Credit Card, Savings Account, Money Transfers, and Personal Loans).
* **Integrity**: Uses `sklearn` or manual calculations to maintain exact categorical ratios.

### 2. Text Chunking

Long narratives are ineffective when embedded as single large vectors. The pipeline implements a recursive splitting strategy:

* **Mechanism**: Uses `RecursiveCharacterTextSplitter` from LangChain.
* **Configuration**: Sets a `chunk_size` of 500 characters with a 50-character `overlap` to preserve context between segments.
* **Result**: Average of approximately 3.02 chunks per complaint, resulting in over 36,000 unique document segments.

### 3. Vector Embedding

Segments are converted into numerical representations using state-of-the-art NLP models:

* **Model**: `sentence-transformers/all-MiniLM-L6-v2`.
* **Dimensions**: Generates vectors with a dimensionality of 384.
* **Hardware**: Automatically detects and uses GPU (CUDA) for acceleration when available.

### 4. Vector Store Indexing

The project supports two powerful backends for storing and querying embeddings:

* **FAISS**: A lightweight, high-performance library for dense vector similarity search, saving the index as a `.bin` file and metadata as a `.pkl`.
* **ChromaDB**: A persistent, developer-friendly vector database that supports metadata filtering and batch ingestion.

---

## 🛠️ Components & File Structure

| File | Description | Key Classes/Functions |
| --- | --- | --- |
| `text_analysis.ipynb` | The main execution notebook coordinating the full pipeline. | Sampling, chunking, and storage calls. |
| `text_transformations.py` | Core logic for document segmenting. | `ComplaintChunker` |
| `text_embedding.py` | Handles model loading and vector generation. | `EmbeddingGenerator` |
| `text_vectorization.py` | Manages FAISS and ChromaDB storage backends. | `FAISSVectorStore`, `ChromaVectorStore` |

---

## 🚀 Usage

### Prerequisites

Ensure all dependencies are installed:

```bash
pip install pandas numpy seaborn scikit-learn langchain-text-splitters sentence-transformers faiss-cpu chromadb

```

### Running the Pipeline

1. **Sample Data**: Load your cleaned complaints and execute the stratified sampling block to generate `sample_complaints.csv`.
2. **Generate Chunks**: Use the `ComplaintChunker` to split narratives into segments stored in `complaint_chunks.csv`.
3. **Embed & Index**:
* Run the `EmbeddingGenerator` to create the vector representations.
* Initialize either `FAISSVectorStore` or `ChromaVectorStore` to persist the data.



---
