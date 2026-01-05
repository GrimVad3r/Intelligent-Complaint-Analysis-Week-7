# CrediTrust Complaint Analysis RAG Pipeline

This repository contains a production-ready **Retrieval-Augmented Generation (RAG)** pipeline designed for **CrediTrust Financial**. The system processes customer complaint narratives, stores them in a vector database, and utilizes a Large Language Model (LLM) to provide evidence-based insights into financial product issues.

---

## 🏗️ System Architecture

The pipeline is divided into modular components for data cleaning, transformation, vector storage, and intelligent generation:

### 1. Data Processing & Cleaning

* **`data_cleaning.py`**: Normalizes raw complaint text by removing boilerplate language (e.g., "Dear Sir/Madam"), handling null values, and standardizing whitespace.
* **`text_transformations.py`**: Uses `RecursiveCharacterTextSplitter` to break long narratives into semantically meaningful chunks while preserving metadata like Product Category and Issue type.

### 2. Semantic Embedding & Retrieval

* **`text_embedding.py`**: Generates numerical vectors using `Sentence-Transformers` (default: `all-MiniLM-L6-v2`). Includes GPU acceleration and SSL resilience for restricted environments.
* **`text_vectorization.py`**: Offers dual-vector store support:
* **FAISS**: Optimized for fast, in-memory exact search.
* **ChromaDB**: A persistent database supporting complex metadata filtering.


* **`retriver.py`**: Orchestrates the retrieval logic, allowing the system to fetch the top-k most relevant complaint excerpts, optionally filtered by product line.

### 3. RAG Pipeline

* **`pipeline_RAG.py`**: The core execution engine. It combines the retriever with a `Mistral-7B` (or similar) LLM to generate natural language answers based strictly on the retrieved context.

---

## 🚀 Key Features

* **Resilient Infrastructure**: Built-in logging, SSL verification toggles for local development, and robust error handling for model initialization and search operations.
* **Evidence-Based Answers**: The `PromptTemplate` enforces strict adherence to provided context, requiring the LLM to cite specific examples from complaints.
* **Metadata Awareness**: Maintains the link between text chunks and their original metadata (Complaint ID, Company, Product), enabling filtered searches.
* **Simplified Taxonomy**: Standardizes diverse financial products into five core categories: Credit Cards, Personal Loans, Savings Accounts, Money Transfers, and Others.

---

## 🛠️ Installation

```bash
pip install pandas numpy torch transformers sentence-transformers faiss-cpu chromadb langchain-text-splitters requests

```

---

## 💻 Usage Example

```python
from retriver import ComplaintRetriever
from pipeline_RAG import ComplaintGenerator, RAGPipeline
from text_vectorization import FAISSVectorStore
from sentence_transformers import SentenceTransformer

# 1. Initialize Components
embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
vector_store = FAISSVectorStore.load("index.faiss", "meta.pkl", embedding_dim=384)
retriever = ComplaintRetriever(vector_store, embedding_model)
generator = ComplaintGenerator(model_name="mistralai/Mistral-7B-Instruct-v0.2")

# 2. Setup Pipeline
rag_pipeline = RAGPipeline(retriever, generator)

# 3. Query the System
result = rag_pipeline.answer_question(
    "What are the common themes in credit card unauthorized charge complaints?",
    product_filter="Credit Card",
    k=5
)

print(f"Answer: {result['answer']}")

```

---

## 📂 File Summary

| File | Responsibility |
| --- | --- |
| **`pipeline_RAG.py`** | Manages LLM prompting and the end-to-end RAG workflow. |
| **`retriver.py`** | Handles query embedding and context retrieval logic. |
| **`data_cleaning.py`** | Text normalization and boilerplate removal. |
| **`text_transformations.py`** | Recursive text chunking and metadata mapping. |
| **`text_vectorization.py`** | FAISS and ChromaDB implementation for vector storage. |
| **`text_embedding.py`** | Vector generation using Sentence-Transformers. |

**Would you like me to create a shell script to automate the initialization of this pipeline?**