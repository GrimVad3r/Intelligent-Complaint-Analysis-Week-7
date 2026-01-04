import faiss
import pickle
import chromadb
from chromadb.config import Settings
import pandas as pd
import numpy as np
from typing import List, Dict
import sys
import os

class FAISSVectorStore:
    """FAISS-based vector store for semantic search"""
    
    def __init__(self, embedding_dim: int):
        self.embedding_dim = embedding_dim
        self.index = faiss.IndexFlatL2(embedding_dim)
        self.metadata = []
    
    def add_embeddings(self, embeddings: np.ndarray, metadata: List[Dict]):
        """Add embeddings and metadata to the store"""
        self.index.add(embeddings.astype('float32'))
        self.metadata.extend(metadata)
    
    def search(self, query_embedding: np.ndarray, k: int = 5) -> List[Dict]:
        """Search for k most similar documents"""
        distances, indices = self.index.search(
            query_embedding.reshape(1, -1).astype('float32'), 
            k
        )
        
        results = []
        for idx, distance in zip(indices[0], distances[0]):
            if idx < len(self.metadata):
                result = {
                    'distance': float(distance),
                    'score': float(1 / (1 + distance)),  # Convert distance to similarity
                    **self.metadata[idx]
                }
                results.append(result)
        
        return results
    
    def save(self, index_path: str, metadata_path: str):
        """Save index and metadata"""
        faiss.write_index(self.index, index_path)
        with open(metadata_path, 'wb') as f:
            pickle.dump(self.metadata, f)
    
    @classmethod
    def load(cls, index_path: str, metadata_path: str, embedding_dim: int):
        """Load index and metadata"""
        store = cls(embedding_dim)
        store.index = faiss.read_index(index_path)
        with open(metadata_path, 'rb') as f:
            store.metadata = pickle.load(f)
        return store


class ChromaVectorStore:
    """ChromaDB-based vector store"""
    
    def __init__(self, collection_name: str = "complaints", persist_directory: str = "../vector_store/CHROMADB"):

        # 1. Ensure the directory exists physically
        if not os.path.exists(persist_directory):
            os.makedirs(persist_directory, exist_ok=True)
            
        # 2. Use PersistentClient instead of the generic Client
        # This is the "switch" that tells Chroma to write to your folder
        self.client = chromadb.PersistentClient(path=persist_directory)
        
        # 3. Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )    
    def add_documents(self, df_chunks: pd.DataFrame):
        """Add documents to ChromaDB"""
        # Prepare data
        ids = [f"{row['complaint_id']}_{row['chunk_index']}" for _, row in df_chunks.iterrows()]
        documents = df_chunks['text'].tolist()
        embeddings = [emb for emb in df_chunks['embedding'].tolist()]
        
        # Prepare metadata
        metadatas = []
        for _, row in df_chunks.iterrows():
            metadata = {
                'complaint_id': str(row['complaint_id']),
                'product_category': row['product_category'],
                'product': row['product'],
                'issue': row['issue'],
                'chunk_index': int(row['chunk_index']),
                'total_chunks': int(row['total_chunks'])
            }
            metadatas.append(metadata)
        
        # Add to collection in batches
        batch_size = 1000
        for i in range(0, len(ids), batch_size):
            batch_ids = ids[i:i+batch_size]
            batch_docs = documents[i:i+batch_size]
            batch_embeddings = embeddings[i:i+batch_size]
            batch_metadatas = metadatas[i:i+batch_size]
            
            self.collection.add(
                ids=batch_ids,
                documents=batch_docs,
                embeddings=batch_embeddings,
                metadatas=batch_metadatas
            )
            
            print(f"Added batch {i//batch_size + 1}/{(len(ids)-1)//batch_size + 1}")
    
    def search(self, query_embedding: List[float], k: int = 5, 
               product_filter: str = None) -> List[Dict]:
        """Search with optional filtering"""
        where_filter = None
        if product_filter:
            where_filter = {"product_category": product_filter}
        
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
            where=where_filter
        )
        
        # Format results
        formatted_results = []
        for idx in range(len(results['ids'][0])):
            formatted_results.append({
                'id': results['ids'][0][idx],
                'text': results['documents'][0][idx],
                'distance': results['distances'][0][idx],
                'metadata': results['metadatas'][0][idx]
            })
        
        return formatted_results
