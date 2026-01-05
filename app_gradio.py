import gradio as gr
import pandas as pd
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
import pickle
import faiss
from typing import List, Dict, Tuple
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
# RAG Chatbot Class
# ============================================================================

class RAGChatbot:
    def __init__(self, vector_store_path: str, metadata_path: str):
        # ... (init remains the same)
        self.embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        self.index = faiss.read_index(vector_store_path)
        with open(metadata_path, 'rb') as f:
            self.metadata = pickle.load(f)

    def retrieve(self, query: str, k: int = 5, product_filter: str = None) -> List[Dict]:
        """Retrieve relevant complaint chunks with logic fix for 'All Products'"""
        query_embedding = self.embedding_model.encode([query])[0]
        
        # FIX 1: If "All Products" is selected, we are NOT filtering
        is_filtering = product_filter and product_filter != "All Products"
        
        # FIX 2: Fetch more candidates (k*20) if filtering to ensure we find 'k' matches
        fetch_k = k * 20 if is_filtering else k
        
        distances, indices = self.index.search(
            query_embedding.reshape(1, -1).astype('float32'), 
            fetch_k
        )
        
        results = []
        for idx, distance in zip(indices[0], distances[0]):
            if idx < len(self.metadata):
                # Ensure we handle nested 'metadata' if it exists in your .pkl
                source_data = self.metadata[idx]
                result = {
                    'distance': float(distance),
                    'score': float(1 / (1 + distance)),
                    **source_data
                }
                
                # FIX 3: Critical filtering logic
                if is_filtering:
                    # Check if product matches (handling both flat and nested metadata)
                    actual_category = result.get('product_category') or result.get('metadata', {}).get('product_category')
                    if actual_category == product_filter:
                        results.append(result)
                else:
                    # If "All Products", just add every result found
                    results.append(result)
        
        return results[:k]

    def generate_simple_answer(self, question: str, sources: List[Dict]) -> str:
        """Generate answer with robust key handling"""
        issue_counts = {}
        for source in sources:
            # Handle nested metadata if present
            meta = source.get('metadata', source) 
            issue = meta.get('issue', 'Unknown')
            issue_counts[issue] = issue_counts.get(issue, 0) + 1
        
        top_issues = sorted(issue_counts.items(), key=lambda x: x[1], reverse=True)[:3]
        
        answer_parts = [f"Based on {len(sources)} relevant complaints, here are the key findings:\n"]
        answer_parts.append("\n**Main Issues:**")
        for issue, count in top_issues:
            answer_parts.append(f"- {issue} ({count} complaints)")
        
        answer_parts.append("\n**Sample Complaints:**")
        for idx, source in enumerate(sources[:2], 1):
            # FIX: Use .get() for 'document' or 'text' to prevent KeyErrors
            meta = source.get('metadata', source)
            text = source.get('document') or source.get('text') or "No content available"
            text_preview = text[:150] + "..."
            answer_parts.append(f"\n{idx}. *{meta.get('product_category', 'N/A')}*: {text_preview}")
        
        return "\n".join(answer_parts)

    def format_sources(self, sources: List[Dict]) -> str:
        """Format sources using Gradio CSS variables for theme compatibility"""
        # Using var(--block-background-fill) instead of #f5f5f5
        source_html = "<div style='margin-top: 20px; padding: 15px; background: var(--block-background-fill); border-radius: 8px; border: 1px solid var(--border-color-primary);'>"
        source_html += "<h4 style='color: var(--body-text-color); margin-bottom: 10px;'>📚 Source Documents</h4>"
        
        for idx, source in enumerate(sources, 1):
            score_pct = source['score'] * 100
            meta = source.get('metadata', source)
            text = source.get('document') or source.get('text') or "[No content]"
            
            # Use var(--background-fill-primary) and var(--body-text-color) for visibility
            source_html += f"""
            <div style='margin-bottom: 15px; padding: 12px; background: var(--background-fill-primary); border-left: 4px solid var(--button-primary-background-fill); border-radius: 4px; box-shadow: var(--block-shadow);'>
                <p style='margin: 0; color: var(--body-text-color); font-weight: bold;'>Source {idx} <span style='font-weight: normal; color: var(--body-text-color-subdued);'>(Relevance: {score_pct:.1f}%)</span></p>
                <p style='margin: 5px 0; font-size: 0.9em; color: var(--primary-500);'>
                    <strong>{meta.get('product_category', 'N/A')}</strong> | {meta.get('issue', 'N/A')}
                </p>
                <p style='color: var(--body-text-color); line-height: 1.4;'>{text[:300]}...</p>
            </div>
            """
        
        source_html += "</div>"
        return source_html
    # ============================================================================
    # Interface Modernization
    # ============================================================================

    # Use a specific professional theme and add custom CSS for fine-tuning
    

    def answer_question(self, question: str, num_sources: int, 
                       product_filter: str) -> Tuple[str, str]:
        """Main method to answer questions"""
        
        if not question.strip():
            return "Please enter a question.", ""
        
        # Retrieve relevant sources
        sources = self.retrieve(question, k=num_sources, product_filter=product_filter)
        
        if not sources:
            return "No relevant complaints found. Try adjusting your filters.", ""
        
        # Generate answer
        answer = self.generate_simple_answer(question, sources)
        
        # Format sources
        sources_html = self.format_sources(sources)
        
        return answer, sources_html

# ============================================================================
# Initialize Chatbot
# ============================================================================

chatbot = RAGChatbot(
    vector_store_path='./vector_store/FAISS/faiss_full_index.bin',
    metadata_path='./vector_store/FAISS/full_metadata.pkl'
)

# ============================================================================
# Gradio Interface
# ============================================================================

def chat_interface(question, num_sources, product_filter):
    """Gradio chat interface function"""
    answer, sources = chatbot.answer_question(question, num_sources, product_filter)
    return answer, sources

# Example questions
examples = [
    ["What are the most common credit card complaints?", 5, "Credit Card"],
    ["Why are customers having issues with personal loans?", 5, "Personal Loan"],
    ["What problems do people face with money transfers?", 5, "Money Transfers"],
    ["Tell me about savings account complaints", 5, "Savings Account"],
    ["What are the main customer complaints?", 10, "All Products"]
]

# Create Gradio interface

custom_css = """
.main-title { font-size: 28px !important; font-weight: 700; color: var(--primary-500); text-align: center; }
.description { text-align: center; color: var(--body-text-color-subdued); margin-bottom: 20px; }

/* FORCE VISIBILITY ON MARKDOWN OUTPUT */
.prose, .prose p, .prose li, .prose h1, .prose h2, .prose h3 {
    color: var(--body-text-color) !important;
    line-height: 1.6;
}

/* Ensure bold text stands out */
.prose strong {
    color: var(--primary-600) !important;
    font-weight: 700;
}
"""

# Custom CSS to force visibility and style the headers
custom_css = """
.main-header { text-align: center; margin-bottom: 1.5rem; }
.main-header h1 { color: var(--primary-500) !important; font-weight: 800; }

/* Fix for invisible/faded markdown text */
.prose, .prose p, .prose li, .prose h1, .prose h2, .prose h3 {
    color: var(--body-text-color) !important;
    line-height: 1.6;
}

/* Make sure strong/bold text is highly visible */
.prose strong {
    color: var(--primary-600) !important;
    font-weight: 700;
}
"""

with gr.Blocks(theme=gr.themes.Soft(primary_hue="blue", neutral_hue="slate"), css=custom_css) as demo:
    
    # Header Section
    with gr.Column(elem_classes="main-header"):
        gr.Markdown("# 💬 CrediTrust Complaint Analysis Chatbot")
        gr.Markdown("Search and analyze customer complaints across financial products using AI retrieval.")
    
    with gr.Row():
        # Input Column
        with gr.Column(scale=2):
            question_input = gr.Textbox(
                label="Your Question",
                placeholder="e.g., What are customers saying about credit card fees?",
                lines=2
            )
            
            with gr.Row():
                num_sources = gr.Slider(
                    minimum=3, maximum=10, value=5, step=1, 
                    label="Analysis Depth (Sources)"
                )
                
                product_filter = gr.Dropdown(
                    choices=["All Products", "Credit Card", "Personal Loan", 
                            "Savings Account", "Money Transfers"],
                    value="All Products",
                    label="Filter by Product"
                )
            
            with gr.Row():
                submit_btn = gr.Button("🔍 Analyze Complaints", variant="primary")
                clear_btn = gr.Button("Clear")
        
        # Sidebar/Tips Column
        with gr.Column(scale=1):
            with gr.Group():
                gr.Markdown("### 💡 Quick Tips")
                gr.Markdown("- Use **Filters** to narrow down product specific issues.")
                gr.Markdown("- Increase **Depth** for a more comprehensive summary.")
                gr.Info("The 'Analysis' tab summarizes findings, while 'Raw Sources' shows original texts.")

    # Output Section with Tabs for better organization
    with gr.Tabs():
        with gr.TabItem("📊 Analysis Results"):
            # Using elem_classes="prose" to trigger the visibility CSS fix
            answer_output = gr.Markdown(
                value="*Results will appear here after analysis...*", 
                elem_classes="prose"
            )
            
        with gr.TabItem("📚 Supporting Sources"):
            sources_output = gr.HTML()
    
    # Examples Section
    gr.Examples(
        examples=examples,
        inputs=[question_input, num_sources, product_filter]
    )
    
    # Footer
    gr.Markdown("""
    ---
    **System Note:** This demo uses semantic vector search to find relevant context and rule-based logic for summary generation.
    """)

    # Event handlers
    submit_btn.click(
        fn=chat_interface,
        inputs=[question_input, num_sources, product_filter],
        outputs=[answer_output, sources_output]
    )
    
    clear_btn.click(
        fn=lambda: ("", 5, "All Products", "*Results cleared*", ""),
        outputs=[question_input, num_sources, product_filter, answer_output, sources_output]
    )

# ============================================================================
# Launch App
# ============================================================================

if __name__ == "__main__":
    demo.launch(
        theme=gr.themes.Soft(),                  
        share=False,
        server_name="0.0.0.0",
        server_port=7860
    )