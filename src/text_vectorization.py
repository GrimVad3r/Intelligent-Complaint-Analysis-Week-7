import faiss
import pickle
import chromadb
from chromadb.config import Settings
import pandas as pd
import numpy as np
from typing import List, Dict
import sys
import os
import logging

# Configure logging for production monitoring
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FAISSVectorStore:
    """FAISS-based vector store with synchronized persistence and error handling."""
    
    def __init__(self, embedding_dim: int):
        self.embedding_dim = embedding_dim
        # Use IndexFlatL2 for exact search; consider IndexIVFFlat for larger production datasets
        self.index = faiss.IndexFlatL2(embedding_dim)
        self.metadata = []

    def add_embeddings(self, embeddings: np.ndarray, metadata: List[Dict]):
        """Resilient addition with input validation to prevent de-syncing."""
        if len(embeddings) != len(metadata):
            logger.error(f"Length mismatch: {len(embeddings)} embeddings vs {len(metadata)} metadata items.")
            raise ValueError("Embeddings and metadata must have the same length.")
        
        try:
            # Ensure data is float32 for FAISS compatibility
            self.index.add(embeddings.astype('float32'))
            self.metadata.extend(metadata)
            logger.info(f"Added {len(embeddings)} vectors. Total store size: {len(self.metadata)}")
        except Exception as e:
            logger.critical(f"Critical failure during FAISS insertion: {e}")
            raise

    def search(self, query_embedding: np.ndarray, k: int = 5) -> List[Dict]:
        """Search with boundary checks for empty indices."""
        if self.index.ntotal == 0:
            logger.warning("Search attempted on an empty FAISS index.")
            return []

        try:
            distances, indices = self.index.search(
                query_embedding.reshape(1, -1).astype('float32'), 
                k
            )
            
            results = []
            for idx, distance in zip(indices[0], distances[0]):
                # FAISS returns -1 if no neighbor is found
                if idx != -1 and idx < len(self.metadata):
                    result = {
                        'distance': float(distance),
                        'score': float(1 / (1 + distance)),
                        **self.metadata[idx]
                    }
                    results.append(result)
            return results
        except Exception as e:
            logger.error(f"Search operation failed: {e}")
            return []

    def save(self, index_path: str, metadata_path: str):
        """Saves index and metadata with directory safety."""
        try:
            # Create directories if they don't exist
            for path in [index_path, metadata_path]:
                dir_name = os.path.dirname(path)
                if dir_name:
                    os.makedirs(dir_name, exist_ok=True)

            faiss.write_index(self.index, index_path)
            with open(metadata_path, 'wb') as f:
                pickle.dump(self.metadata, f)
            logger.info("Successfully persisted FAISS index and metadata.")
        except (IOError, OSError) as e:
            logger.error(f"Persistence failed: {e}")
            raise

    @classmethod
    def load(cls, index_path: str, metadata_path: str, embedding_dim: int):
        """Load index and metadata with existence verification."""
        if not os.path.exists(index_path) or not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Required files not found at {index_path} or {metadata_path}")
            
        try:
            store = cls(embedding_dim)
            store.index = faiss.read_index(index_path)
            with open(metadata_path, 'rb') as f:
                store.metadata = pickle.load(f)
            
            # Final validation check
            if store.index.ntotal != len(store.metadata):
                logger.warning("Loaded index and metadata are out of sync!")
            
            return store
        except Exception as e:
            logger.error(f"Failed to load FAISS store: {e}")
            raise

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ChromaVectorStore:
    """Production-ready ChromaDB-based vector store with batch safety and logging."""
    
    def __init__(self, collection_name: str = "complaints", persist_directory: str = "../vector_store/CHROMADB"):
        try:
            if not os.path.exists(persist_directory):
                os.makedirs(persist_directory, exist_ok=True)
            
            self.client = chromadb.PersistentClient(path=persist_directory)
            self.collection = self.client.get_or_create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            logger.info(f"Successfully initialized ChromaDB collection: {collection_name}")
        except Exception as e:
            logger.critical(f"Failed to initialize ChromaDB: {e}")
            raise

    def add_documents(self, df_chunks: pd.DataFrame, batch_size: int = 1000):
        """Add documents to ChromaDB with validation and error recovery."""
        # Resilience: Validate required columns before processing
        required_cols = ['complaint_id', 'chunk_index', 'text', 'embedding']
        if not all(col in df_chunks.columns for col in required_cols):
            missing = [c for c in required_cols if c not in df_chunks.columns]
            raise ValueError(f"DataFrame is missing required columns: {missing}")

        total_rows = len(df_chunks)
        logger.info(f"Starting ingestion of {total_rows} documents.")

        for i in range(0, total_rows, batch_size):
            batch_df = df_chunks.iloc[i : i + batch_size]
            
            try:
                # Prepare batch data using list comprehensions (more efficient than iterrows)
                ids = [f"{row['complaint_id']}_{row['chunk_index']}" for _, row in batch_df.iterrows()]
                documents = batch_df['text'].tolist()
                embeddings = [list(emb) if hasattr(emb, "__iter__") else emb for emb in batch_df['embedding'].tolist()]
                
                metadatas = [
                    {
                        'complaint_id': str(row.get('complaint_id', '')),
                        'product_category': str(row.get('product_category', 'N/A')),
                        'issue': str(row.get('issue', 'N/A')),
                        'chunk_index': int(row.get('chunk_index', 0))
                    } for _, row in batch_df.iterrows()
                ]

                self.collection.add(
                    ids=ids,
                    documents=documents,
                    embeddings=embeddings,
                    metadatas=metadatas
                )
                logger.info(f"Successfully added batch {i//batch_size + 1}")
                
            except Exception as e:
                # Resilience: Log the error but keep the process running if possible
                logger.error(f"Failed to add batch starting at index {i}: {e}")
                continue

    def search(self, query_embedding: List[float], k: int = 5, product_filter: str = None) -> List[Dict]:
        """Search with error handling and result formatting."""
        try:
            where_filter = {"product_category": product_filter} if product_filter else None
            
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=k,
                where=where_filter
            )
            
            # Formatting with safety check for empty results
            if not results or not results['ids']:
                return []

            return [
                {
                    'id': results['ids'][0][idx],
                    'text': results['documents'][0][idx],
                    'distance': results['distances'][0][idx],
                    'metadata': results['metadatas'][0][idx]
                }
                for idx in range(len(results['ids'][0]))
            ]
        except Exception as e:
            logger.error(f"Search operation failed: {e}")
            return []
