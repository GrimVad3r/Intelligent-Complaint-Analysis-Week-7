import pandas as pd
from typing import List, Dict
from langchain_text_splitters import RecursiveCharacterTextSplitter

class ComplaintChunker:
    """Handles text chunking for complaint narratives"""
    
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
    
    def chunk_complaint(self, complaint_id: str, text: str, metadata: Dict) -> List[Dict]:
        """
        Chunk a single complaint and return chunks with metadata
        """
        # Handle cases where text might be None/NaN
        if not isinstance(text, str):
            return []

        chunks = self.splitter.split_text(text)
        
        chunk_docs = []
        for idx, chunk in enumerate(chunks):
            chunk_doc = {
                'complaint_id': complaint_id,
                'chunk_index': idx,
                'total_chunks': len(chunks),
                'text': chunk,
                **metadata  # Include all metadata
            }
            chunk_docs.append(chunk_doc)
        
        return chunk_docs
    
    def chunk_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """Chunk entire dataset"""
        all_chunks = []
        
        for _, row in df.iterrows():
            # Using .get() is safer to avoid KeyErrors if columns are missing
            metadata = {
                'product_category': row.get('product_category', ''),
                'product': row.get('Product', ''),
                'issue': row.get('Issue', ''),
                'sub_issue': row.get('Sub-issue', ''),
                'company': row.get('Company', ''),
                'state': row.get('State', ''),
                'date_received': str(row.get('Date received', ''))
            }
            
            chunks = self.chunk_complaint(
                complaint_id=str(row.get('Complaint ID', 'unknown')),
                text=row.get('cleaned_narrative', ''),
                metadata=metadata
            )
            all_chunks.extend(chunks)
        
        return pd.DataFrame(all_chunks)