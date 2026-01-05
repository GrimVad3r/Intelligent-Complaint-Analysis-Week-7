import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
import pickle
import faiss
from typing import List, Dict
import time
from urllib3.exceptions import InsecureRequestWarning
import sys
import os
import ssl
import logging
import requests
from huggingface_hub import configure_http_backend

# 1. SETUP LOGGING
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def disable_ssl_verification():
    """Forces all libraries to ignore SSL verification."""
    logger.warning("SSL Verification DISABLED. Use only for local dev.")
    
    # LEVEL 1: Environment Variables
    os.environ['CURL_CA_BUNDLE'] = ''
    os.environ['REQUESTS_CA_BUNDLE'] = ''
    os.environ['SSL_CERT_FILE'] = ''
    os.environ['HF_HUB_DISABLE_SSL_VERIFY'] = '1'
    os.environ['PYTHONHTTPSVERIFY'] = '0'
    
    # LEVEL 2: Standard Python Libraries
    requests.packages.urllib3.disable_warnings(category=InsecureRequestWarning)
    ssl._create_default_https_context = ssl._create_unverified_context
    
    # LEVEL 3: Hugging Face Hub Specific Backend fix
    def backend_factory() -> requests.Session:
        session = requests.Session()
        session.verify = False  # The magic line
        return session
    configure_http_backend(backend_factory=backend_factory)

# Call this immediately before anything else
disable_ssl_verification()

# ============================================================================
# Page Configuration
# ============================================================================

st.set_page_config(
    page_title="CrediTrust Complaint Analysis",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# RAG Chatbot Class
# ============================================================================

@st.cache_resource
def load_chatbot():
    """Load and cache the RAG chatbot"""
    
    class RAGChatbot:
        def __init__(self, vector_store_path: str, metadata_path: str):
            self.embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
            self.index = faiss.read_index(vector_store_path)
            
            with open(metadata_path, 'rb') as f:
                self.metadata = pickle.load(f)
        
        def retrieve(self, query: str, k: int = 5, product_filter: str = None) -> List[Dict]:
            query_embedding = self.embedding_model.encode([query])[0]

            # Increase fetch_k when filtering to ensure we have enough candidates

            # If filtering, we fetch more candidates to ensure we find 'k' matches
            is_filtering = product_filter and product_filter != "All Products"
            fetch_k = k * 20 if is_filtering else k
            
            distances, indices = self.index.search(
                query_embedding.reshape(1, -1).astype('float32'), 
                fetch_k
            )
            
            results = []
            for idx, distance in zip(indices[0], distances[0]):
                if idx < len(self.metadata):
                    # The structure from your Parquet file puts everything in 'metadata'
                    source_data = self.metadata[idx]
                    result = {
                        'distance': float(distance),
                        'score': float(1 / (1 + distance)),
                        **self.metadata[idx]
                    }
                    # THE FIX: Access the nested metadata for filtering
                    inner_metadata = result.get('metadata', {})
                    actual_category = inner_metadata.get('product_category')
                    
                    # THE CRITICAL FIX: Only apply the filter if it's NOT "All Products"
                    if is_filtering:
                        inner_meta = result.get('metadata', {})
                        if inner_meta.get('product_category') == product_filter:
                            results.append(result)
                    else:
                        # If "All Products" or None, just add the result
                        results.append(result)
            
            return results[:k]
        
        def generate_answer(self, question: str, sources: List[Dict]) -> str:
            issue_counts = {}
            
            for source in sources:
                metadata = source.get('metadata', {})
                issue = metadata.get('issue') or 'General Inquiry'
                issue_counts[issue] = issue_counts.get(issue, 0) + 1
            
            top_issues = sorted(issue_counts.items(), key=lambda x: x[1], reverse=True)[:3]
            
            answer_parts = [
                f"Based on {len(sources)} relevant complaints:\n"
            ]
            
            answer_parts.append("**Key Issues:**")
            for issue, count in top_issues:
                answer_parts.append(f"- {issue} ({count} mentions)")
            
            return "\n".join(answer_parts)
    
    return RAGChatbot(
        vector_store_path='./vector_store/FAISS/faiss_full_index.bin',
        metadata_path='./vector_store/FAISS/full_metadata.pkl'
    )

# ============================================================================
# Initialize Session State
# ============================================================================

if 'messages' not in st.session_state:
    st.session_state.messages = []

if 'chatbot' not in st.session_state:
    with st.spinner("Loading AI models..."):
        st.session_state.chatbot = load_chatbot()

# ============================================================================
# Sidebar
# ============================================================================

with st.sidebar:
    st.title("⚙️ Settings")
    
    num_sources = st.slider(
        "Number of Sources",
        min_value=3,
        max_value=10,
        value=5,
        help="How many complaint sources to retrieve"
    )
    
    product_filter = st.selectbox(
        "Filter by Product",
        ["All Products", "Credit Card", "Personal Loan", 
         "Savings Account", "Money Transfers"],
        help="Focus on specific product categories"
    )
    
    st.divider()
    
    st.markdown("""
    ### 📊 About
    
    This chatbot analyzes customer complaints using:
    - **464K+** real complaints
    - **Semantic search** for relevance
    - **AI-powered** insights
    
    ### 💡 Tips
    - Ask specific questions
    - Use product filters
    - Check sources below answers
    """)
    
    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

# ============================================================================
# Main Interface
# ============================================================================

st.title("💬 CrediTrust Complaint Analysis")
st.markdown("Ask questions about customer complaints and get AI-powered insights.")

# Example questions
with st.expander("📝 Example Questions"):
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Credit card fee complaints"):
            st.session_state.example_q = "What are the main complaints about credit card fees?"
        if st.button("Personal loan denials"):
            st.session_state.example_q = "Why are customers' personal loan applications being denied?"
    
    with col2:
        if st.button("Money transfer issues"):
            st.session_state.example_q = "What problems do customers face with money transfers?"
        if st.button("Savings account access"):
            st.session_state.example_q = "What are the complaints about accessing savings accounts?"

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        if "sources" in message:
            with st.expander("📚 View Sources"):
                for idx, source in enumerate(message["sources"], 1):
                    metadata = source.get('metadata', {})
                    st.markdown(f"""
                    **Source {idx}** (Relevance: {source['score']*100:.1f}%)
                    - **Product:** {metadata.get('product_category', 'N/A')}
                    - **Issue:** {metadata.get('issue', 'N/A')}
                    - **Text:** {metadata.get('text', source.get('document', 'No text'))[:200]}...
                    """)
                    st.divider()

# Chat input
if prompt := st.chat_input("Ask about customer complaints...") or st.session_state.get('example_q'):
    
    # Use example question if set
    if 'example_q' in st.session_state:
        prompt = st.session_state.example_q
        del st.session_state.example_q
    
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Analyzing complaints..."):
            # Retrieve sources
            sources = st.session_state.chatbot.retrieve(
                prompt, 
                k=num_sources, 
                product_filter=product_filter
            )
            
            if not sources:
                response = "No relevant complaints found. Try adjusting your filters."
                st.markdown(response)
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": response
                })
            else:
                # Generate answer
                answer = st.session_state.chatbot.generate_answer(prompt, sources)
                
                # Display with streaming effect
                message_placeholder = st.empty()
                full_response = ""
                
                for chunk in answer.split():
                    full_response += chunk + " "
                    time.sleep(0.05)
                    message_placeholder.markdown(full_response + "▌")
                
                message_placeholder.markdown(full_response)
                
                # Display sources
                with st.expander("📚 View Sources", expanded=True):
                    for idx, source in enumerate(sources, 1):
                        metadata = source.get('metadata', {})
                        st.markdown(f"""
                        **Source {idx}** (Relevance: {source['score']*100:.1f}%)
                        - **Product:** {metadata.get('product_category', 'N/A')}
                        - **Issue:** {metadata.get('issue', 'N/A')}
                        - **Text:** {metadata.get('text', source.get('document', 'No text'))[:200]}...
                        """)
                        st.divider()
                
                # Save to history
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": full_response,
                    "sources": sources
                })

# ============================================================================
# Footer
# ============================================================================

st.divider()
st.markdown("""
<div style='text-align: center; color: #666;'>
    <small>CrediTrust Complaint Analysis System | Powered by RAG & AI</small>
</div>
""", unsafe_allow_html=True)