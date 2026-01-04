import os

# Set these BEFORE importing any NLP libraries
os.environ['CURL_CA_BUNDLE'] = ''
os.environ['REQUESTS_CA_BUNDLE'] = ''
os.environ['SSL_CERT_FILE'] = ''
os.environ['HF_HUB_DISABLE_SSL_VERIFY'] = '1'
os.environ['PYTHONHTTPSVERIFY'] = '0'

import ssl
import requests

# Disable warnings for unverified HTTPS requests
from urllib3.exceptions import InsecureRequestWarning
requests.packages.urllib3.disable_warnings(category=InsecureRequestWarning)

# Global SSL context override
ssl._create_default_https_context = ssl._create_unverified_context


from sentence_transformers import SentenceTransformer
import torch
from typing import List, Dict
import pandas as pd
import numpy as np


class EmbeddingGenerator:
    """Generates embeddings for text chunks"""
    
    def __init__(self, model_name: str = 'sentence-transformers/all-MiniLM-L6-v2'):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        
        # Use GPU if available
        if torch.cuda.is_available():
            self.model = self.model.to('cuda')
            print("Using GPU for embeddings")
        else:
            print("Using CPU for embeddings")
    
    def generate_embeddings(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """Generate embeddings for a list of texts"""
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        return embeddings
