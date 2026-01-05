import sys
import os
import ssl
import logging
import torch
import requests
from typing import Dict, List
from urllib3.exceptions import InsecureRequestWarning
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

# 1. SETUP LOGGING FIRST (Resilience: Prevent NameError in disable_ssl_verification)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 2. Resilience: Configuration via Environment Variables
DEBUG_MODE = os.getenv("APP_DEBUG", "false").lower() == "true"

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

# 3. Path Management
current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(current_dir, '..', 'src')
abs_src_path = os.path.abspath(src_path)

if abs_src_path not in sys.path and os.path.isdir(abs_src_path):
    sys.path.append(abs_src_path)
    logger.info(f"Added {abs_src_path} to sys.path")

# 4. Imports 
try:
    from text_vectorization import FAISSVectorStore
    from retriver import ComplaintRetriever 
except ImportError as e:
    logger.error(f"Failed to import dependencies: {e}")
    raise

class PromptTemplate:
    """Manages prompt templates for the RAG system"""
    
    SYSTEM_PROMPT = """You are a financial analyst assistant for CrediTrust Financial, specializing in analyzing customer complaints. Your role is to provide clear, evidence-based insights about customer issues across our product lines: Credit Cards, Personal Loans, Savings Accounts, and Money Transfers."""
    
    RAG_TEMPLATE = """You are a financial analyst assistant for CrediTrust. Your task is to answer questions about customer complaints using ONLY the provided complaint excerpts.

Guidelines:
- Base your answer strictly on the context provided
- If the context doesn't contain sufficient information, clearly state this
- Cite specific examples from the complaints when possible
- Identify patterns and common themes
- Be concise but thorough

Context (Retrieved Complaint Excerpts):
{context}

Question: {question}

Answer:"""
    
    @staticmethod
    def format_prompt(question: str, context: str) -> str:
        """Format the complete prompt"""
        return PromptTemplate.RAG_TEMPLATE.format(
            context=context,
            question=question
        )

class ComplaintGenerator:
    """Generates answers using an LLM"""
    def __init__(self, model_name: str = "mistralai/Mistral-7B-Instruct-v0.2"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")
        
        try:
            # Added trust_remote_code and pad_token logic for Mistral/Phi models
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto"
            )
            
            self.pipe = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                max_new_tokens=512,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        except Exception as e:
            logger.error(f"Model initialization failed: {e}")
            raise

    def generate(self, prompt: str) -> str:
        try:
            result = self.pipe(prompt)
            generated_text = result[0]['generated_text']
            return generated_text.split("Answer:")[-1].strip() if "Answer:" in generated_text else generated_text[len(prompt):].strip()
        except Exception as e:
            logger.error(f"Generation error: {e}")
            return "I'm sorry, I encountered an error generating the response."

class RAGPipeline:
    
    """Complete RAG pipeline combining retrieval and generation"""
    def __init__(self, retriever: ComplaintRetriever, generator):
        self.retriever = retriever
        self.generator = generator
    
    def answer_question(self, question: str, k: int = 5, product_filter: str = None) -> Dict:
        # Resilience: Handle empty retrieval
        retrieved_chunks = self.retriever.retrieve(question, k=k, product_filter=product_filter)
        context = self.retriever.format_context(retrieved_chunks) if retrieved_chunks else "No relevant complaints found."
        
        prompt = PromptTemplate.format_prompt(question, context)
        answer = self.generator.generate(prompt)
        
        return {
            'question': question,
            'answer': answer,
            'sources': retrieved_chunks,
            'num_sources': len(retrieved_chunks)
        }