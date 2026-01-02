
# RAG-Powered Complaint Analysis Chatbot

## Overview

This project implements a Retrieval-Augmented Generation (RAG) system for analyzing customer complaints at CrediTrust Financial. The system processes 464K+ complaints across Credit Cards, Personal Loans, Savings Accounts, and Money Transfers.

## Features

- **Semantic Search**: Find relevant complaints using vector similarity
- **Product Filtering**: Focus analysis on specific product categories
- **Interactive UI**: Chat interface built with Gradio/Streamlit
- **Scalable Architecture**: Handles large-scale complaint datasets
- **Source Attribution**: Always shows which complaints informed the answer

## Project Structure

```
rag-complaint-chatbot/
├── data/
│   ├── raw/                    # Original CFPB data
│   └── processed/              # Cleaned and filtered data
├── vector_store/               # FAISS index and metadata
├── notebooks/                  # EDA and experimentation
├── src/
│   ├── __init__.py
│   └── rag_pipeline.py        # Core RAG components
├── tests/
│   ├── __init__.py
│   └── test_rag_pipeline.py   # Unit tests
├── app.py                      # Gradio interface
├── requirements.txt
└── README.md
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/GrimVad3r/Intelligent-Complaint-Analysis-Week-7.git
cd rag-complaint-chatbot
```

2. Create virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Download data:
   - Full dataset: [CFPB Complaints](link)
   - Pre-built embeddings: [complaint_embeddings.parquet](link)

## Quick Start

### Task 1: EDA and Preprocessing

```bash
jupyter notebook notebooks/01_eda_preprocessing.ipynb
```

### Task 2: Build Vector Store (Sample)

```python
from src.rag_pipeline import (
    ComplaintPreprocessor,
    ComplaintChunker,
    EmbeddingGenerator,
    FAISSVectorStore
)

# Load and preprocess
preprocessor = ComplaintPreprocessor()
df = pd.read_csv('data/raw/complaints.csv')
df_clean = preprocessor.process_dataframe(df)

# Create sample
df_sample = preprocessor.create_stratified_sample(df_clean, 12000)

# Chunk
chunker = ComplaintChunker(chunk_size=500, chunk_overlap=50)
chunks = chunker.chunk_dataframe(df_sample)

# Embed
generator = EmbeddingGenerator()
embeddings = generator.embed_texts(chunks['text'].tolist())

# Store
store = FAISSVectorStore(384)
store.add(embeddings, chunks.to_dict('records'))
store.save('vector_store/faiss_index.bin', 'vector_store/metadata.pkl')
```

### Task 3: Run RAG Pipeline

```python
from src.rag_pipeline import RAGPipeline, ComplaintRetriever

# Load vector store
store = FAISSVectorStore.load(
    'vector_store/faiss_full_index.bin',
    'vector_store/full_metadata.pkl',
    384
)

# Initialize retriever
retriever = ComplaintRetriever(store, generator)

# Create pipeline
rag = RAGPipeline(retriever)

# Ask questions
result = rag.answer("What are the main credit card complaints?", k=5)
print(result['answer'])
```

### Task 4: Launch UI

**Gradio:**
```bash
python app.py
```

## Testing

Run unit tests:
```bash
python -m pytest tests/ -v
```

Run specific test:
```bash
python -m pytest tests/test_rag_pipeline.py::TestComplaintPreprocessor -v
```

## Evaluation

Evaluate the RAG system:

```python
from src.rag_pipeline import RAGEvaluator

questions = [
    {'question': 'What are common credit card complaints?'},
    {'question': 'Why are loan applications denied?'},
    # ... more questions
]

evaluator = RAGEvaluator(rag)
results = evaluator.evaluate_questions(questions)
results.to_csv('evaluation_results.csv')
```

## Configuration

Key parameters to tune:

- **Chunk size**: 500 characters (balance between context and relevance)
- **Chunk overlap**: 50 characters (maintains continuity)
- **Top-k retrieval**: 5 sources (balance between context and noise)
- **Embedding model**: all-MiniLM-L6-v2 (384 dims, fast and effective)

## Performance

- **Embedding generation**: ~2-3 hours for full dataset (464K complaints)
- **Vector store size**: ~530MB for FAISS index + metadata
- **Query latency**: <200ms for retrieval + generation
- **Memory usage**: ~2GB for full index in memory

## Future Improvements

1. **Better Generator**: Replace rule-based with actual LLM (GPT-4, Claude, Llama)
2. **Advanced Retrieval**: Implement hybrid search (keyword + semantic)
3. **Fine-tuning**: Train domain-specific embedding model
4. **Caching**: Cache common queries for faster response
5. **Analytics**: Add dashboard for complaint trends over time


## License

This project is part of the KAIM program and is for educational purposes.
