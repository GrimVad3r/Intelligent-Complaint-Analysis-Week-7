import os
import ssl
import logging
import torch
import numpy as np
from typing import List, Optional
from sentence_transformers import SentenceTransformer
from urllib3.exceptions import InsecureRequestWarning
import requests

# 1. Resilience: Configuration via Environment Variables
DEBUG_MODE = os.getenv("APP_DEBUG", "false").lower() == "true"
MODEL_NAME = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def disable_ssl_verification():
    """Only use this for local development behind restrictive firewalls."""
    logger.warning("SSL Verification is DISABLED. This is a security risk for production.")
    os.environ['CURL_CA_BUNDLE'] = ''
    os.environ['REQUESTS_CA_BUNDLE'] = ''
    os.environ['SSL_CERT_FILE'] = ''
    os.environ['HF_HUB_DISABLE_SSL_VERIFY'] = '1'
    os.environ['PYTHONHTTPSVERIFY'] = '0'
    requests.packages.urllib3.disable_warnings(category=InsecureRequestWarning)
    ssl._create_default_https_context = ssl._create_unverified_context

if DEBUG_MODE:
    disable_ssl_verification()

class EmbeddingGenerator:
    """Generates embeddings for text chunks with robust error handling."""
    
    def __init__(self, model_name: str = MODEL_NAME):
        self.model_name = model_name
        try:
            logger.info(f"Loading model: {model_name}")
            self.model = SentenceTransformer(model_name)
            
            # Use GPU if available
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self.model = self.model.to(self.device)
            logger.info(f"Model loaded successfully on {self.device}")
            
            self.embedding_dim = self.model.get_sentence_embedding_dimension()
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            raise RuntimeError(f"Initialization failed: {e}")

    def generate_embeddings(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """Generate embeddings with input validation and error handling."""
        if not texts:
            logger.warning("Empty list of texts provided for embedding.")
            return np.array([])

        try:
            embeddings = self.model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=False,
                convert_to_numpy=True
            )
            return embeddings
        except Exception as e:
            logger.error(f"Error during embedding generation: {e}")
            # In production, you might want to return an empty array or re-raise
            raise