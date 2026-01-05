import os
import sys
import logging
import ssl  # FIXED: Was missing, required for disable_ssl_verification
import requests
import pandas as pd
from typing import List, Dict
from sentence_transformers import SentenceTransformer
from urllib3.exceptions import InsecureRequestWarning

# 1. Resilience: Configuration
DEBUG_MODE = os.getenv("APP_DEBUG", "false").lower() == "true"
MODEL_NAME = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def disable_ssl_verification():
    """Only use this for local development behind restrictive firewalls."""
    logger.warning("SSL Verification is DISABLED. This is a security risk for production.")
    os.environ['HF_HUB_DISABLE_SSL_VERIFY'] = '1'
    os.environ['PYTHONHTTPSVERIFY'] = '0'
    requests.packages.urllib3.disable_warnings(category=InsecureRequestWarning)
    # Fixed: ssl was not imported previously
    ssl._create_default_https_context = ssl._create_unverified_context

if DEBUG_MODE:
    disable_ssl_verification()

# 2. Resilience: Robust Path Handling
# Using __file__ is more reliable than os.getcwd() for production scripts
current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(current_dir, '..', 'src')
abs_src_path = os.path.abspath(src_path)

if abs_src_path not in sys.path and os.path.isdir(abs_src_path):
    sys.path.append(abs_src_path)
    logger.info(f"Added {abs_src_path} to sys.path")

# Now we can import from src
try:
    from text_vectorization import FAISSVectorStore
except ImportError as e:
    logger.error(f"Failed to import FAISSVectorStore: {e}")
    raise

class ComplaintRetriever:
    """Retrieves relevant complaint chunks for a given query"""
    
    def __init__(self, vector_store: FAISSVectorStore, embedding_model: SentenceTransformer):
        self.vector_store = vector_store
        self.embedding_model = embedding_model
    
    def retrieve(self, query: str, k: int = 5, product_filter: str = None) -> List[Dict]:
        """Retrieve top-k relevant chunks with error handling."""
        if not query.strip():
            logger.warning("Empty query received.")
            return []

        try:
            # Generate query embedding
            query_embedding = self.embedding_model.encode([query])[0]
            
            # 3. Resilience: Logic Improvement
            # If a filter is used, we fetch more results to ensure we have k left after filtering
            fetch_k = k * 10 if product_filter else k
            results = self.vector_store.search(query_embedding, k=fetch_k)
            
            if product_filter:
                results = [r for r in results if r.get('product_category') == product_filter]
            
            return results[:k]
        except Exception as e:
            logger.error(f"Retrieval failed: {e}")
            return []
    
    def format_context(self, retrieved_chunks: List[Dict]) -> str:
        """Format chunks into a clean context string."""
        if not retrieved_chunks:
            return "No relevant context found."

        context_parts = []
        for idx, chunk in enumerate(retrieved_chunks, 1):
            # Using .get() prevents KeyErrors if metadata is inconsistent
            p_cat = chunk.get('product_category', 'Unknown')
            issue = chunk.get('issue', 'Unknown')
            text = chunk.get('document', '[No Text]')
            
            context_parts.append(f"[Source {idx}] Product: {p_cat} | Issue: {issue}\n{text}\n")
        
        return "\n".join(context_parts)