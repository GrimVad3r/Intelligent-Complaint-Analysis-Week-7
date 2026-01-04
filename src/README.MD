# Consumer Complaint Semantic Search Pipeline

This project provides a comprehensive modular pipeline for processing, embedding, and storing consumer complaint narratives for semantic search. It leverages **LangChain** for intelligent text splitting, **Sentence-Transformers** for high-quality vector generation, and provides dual support for **FAISS** and **ChromaDB** vector stores.

---

## 🛠️ Module Overview

### 1. Text Transformation (`text_transformations.py`)

This module handles the initial preparation of raw complaint data, ensuring long narratives are broken down into semantically meaningful segments.

* **Recursive Splitting**: Utilizes `RecursiveCharacterTextSplitter` to split text at natural boundaries like double newlines or periods to maintain context.
* **Metadata Management**: Automatically extracts and maps dataset columns (Product, Issue, Company, State) to each individual chunk for downstream filtering.
* **Safety Features**: Includes built-in handling for null or non-string inputs to prevent pipeline crashes during batch processing.

### 2. Text Embedding (`text_embedding.py`)

Converts processed text chunks into numerical vectors (embeddings) that represent the semantic meaning of the complaint.

* **Model Selection**: Defaults to the `all-MiniLM-L6-v2` model for an optimal balance between performance and inference speed.
* **Hardware Optimization**: Automatically detects and utilizes **NVIDIA GPU (CUDA)** for faster embedding generation, falling back to CPU if necessary.
* **SSL Configuration**: Includes pre-configured environment overrides to bypass SSL verification issues in restricted network environments.

### 3. Vector Storage (`text_vectorization.py`)

Provides two distinct options for storing and querying your complaint embeddings.

#### **FAISS (Facebook AI Similarity Search)**

* **Lightweight**: Best for fast, in-memory similarity searches using Euclidean distance (`IndexFlatL2`).
* **Persistence**: Simple `save` and `load` methods to export the index and metadata to disk via pickle.

#### **ChromaDB**

* **Persistent Storage**: Uses a `PersistentClient` to maintain a database on disk between sessions.
* **Metadata Filtering**: Supports complex queries with "where" filters (e.g., searching only within specific product categories).
* **Batch Ingestion**: Features a batching mechanism to efficiently add large volumes of documents (1,000 per batch).

---

## 🚀 Quick Start

### Installation

```bash
pip install pandas numpy torch sentence-transformers faiss-cpu chromadb langchain-text-splitters

```

### Usage Example

```python
import pandas as pd
from text_transformations import ComplaintChunker
from text_embedding import EmbeddingGenerator
from text_vectorization import ChromaVectorStore

# 1. Process and Chunk Data
df = pd.read_csv("complaints.csv")
chunker = ComplaintChunker(chunk_size=500, chunk_overlap=50)
df_chunks = chunker.chunk_dataset(df)

# 2. Generate Embeddings
generator = EmbeddingGenerator()
embeddings = generator.generate_embeddings(df_chunks['text'].tolist())
df_chunks['embedding'] = embeddings.tolist()

# 3. Store in ChromaDB
vector_store = ChromaVectorStore(persist_directory="./complaint_db")
vector_store.add_documents(df_chunks)

# 4. Perform a Semantic Search
query_vector = generator.generate_embeddings(["Unauthorized credit card charges"])
results = vector_store.search(query_vector[0], k=3)

```

---

## 📂 File Structure

| File | Responsibility | Key Class |
| --- | --- | --- |
| `text_transformations.py` | Text cleaning and recursive chunking | `ComplaintChunker` |
| `text_embedding.py` | Vector generation with GPU support | `EmbeddingGenerator` |
| `text_vectorization.py` | Storage, indexing, and similarity search | `FAISSVectorStore`, `ChromaVectorStore` |

