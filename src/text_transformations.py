import pandas as pd
import logging
from typing import List, Dict, Optional
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ComplaintChunker:
    """Handles text chunking for complaint narratives with production-grade validation."""
    
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )

    def chunk_complaint(self, complaint_id: str, text: Optional[str], metadata: Dict) -> List[Dict]:
        """Chunk a single complaint with safety checks."""
        if not isinstance(text, str) or not text.strip():
            logger.debug(f"Skipping empty or non-string text for ID: {complaint_id}")
            return []

        try:
            chunks = self.splitter.split_text(text)
            return [
                {
                    'complaint_id': complaint_id,
                    'chunk_index': idx,
                    'total_chunks': len(chunks),
                    'text': chunk,
                    **metadata
                }
                for idx, chunk in enumerate(chunks)
            ]
        except Exception as e:
            logger.error(f"Failed to chunk complaint {complaint_id}: {e}")
            return []

    def chunk_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """Efficiently chunk entire dataset with schema validation."""
        # 1. Resilience: Validate Schema
        required_cols = ['Complaint ID', 'cleaned_narrative']
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            error_msg = f"DataFrame missing required columns: {missing}"
            logger.error(error_msg)
            raise ValueError(error_msg)

        all_chunks = []
        
        # 2. Efficiency: Using a loop with dict conversion is often faster than iterrows
        data_dicts = df.to_dict('records')
        
        for row in data_dicts:
            metadata = {
                'product_category': row.get('product_category', ''),
                'product': row.get('Product', ''),
                'issue': row.get('Issue', ''),
                'company': row.get('Company', ''),
            }
            
            chunks = self.chunk_complaint(
                complaint_id=str(row.get('Complaint ID', 'unknown')),
                text=row.get('cleaned_narrative'),
                metadata=metadata
            )
            all_chunks.extend(chunks)
        
        logger.info(f"Chunking complete. Created {len(all_chunks)} chunks from {len(df)} rows.")
        return pd.DataFrame(all_chunks)