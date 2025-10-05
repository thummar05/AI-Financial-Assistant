# services/vector_search_service.py
import faiss
import numpy as np
import json
import os
from typing import List, Dict, Any, Optional

# --- Configuration ---
FAISS_INDEX_PATH = "embeddings/vector_store.faiss"
DATA_INDEX_PATH = "embeddings/index_metadata.json"

class VectorSearchService:
    def __init__(self):
        self.index: Optional[faiss.IndexFlatL2] = None
        self.transactions: List[Dict[str, Any]] = []
        # Don't auto-load during initialization - let it be called explicitly
        # self._load_index()

    def _load_index(self):
        """Loads the FAISS index and transaction metadata."""
        # Check if files exist before trying to load
        if not os.path.exists(FAISS_INDEX_PATH) or not os.path.exists(DATA_INDEX_PATH):
            print(f"VectorSearchService: Index files not found. Run setup first.")
            self.index = None
            self.transactions = []
            return
        
        try:
            self.index = faiss.read_index(FAISS_INDEX_PATH)
            with open(DATA_INDEX_PATH, 'r') as f:
                self.transactions = json.load(f)
            print("VectorSearchService: FAISS index and metadata loaded successfully.")
        except (FileNotFoundError, OSError, json.JSONDecodeError) as e:
            print(f"VectorSearchService: Could not load index files. Error: {e}")
            self.index = None
            self.transactions = []

    def search_transactions(self, query_embedding: np.ndarray, k: int = 10) -> List[Dict[str, Any]]:
        """Performs semantic search on the FAISS index."""
        if not self.index:
            print("Search failed: FAISS index is not initialized.")
            return []
        
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        # Ensure k is not larger than the number of items in the index
        k = min(k, self.index.ntotal)
        
        # D is distances, I is indices
        D, I = self.index.search(query_embedding, k)
        
        results = []
        for rank, index in enumerate(I[0]):
            if index < len(self.transactions):
                result_txn = self.transactions[index].copy()
                result_txn['distance'] = float(D[0][rank]) # Add distance/score for context
                results.append(result_txn)
        
        return results

def save_faiss_index(embeddings_matrix: np.ndarray, transactions: List[Dict[str, Any]]):
    """Creates a FAISS index and saves both the index and the transaction metadata."""
    if not embeddings_matrix.shape[0] or not transactions:
        print("Cannot save FAISS index: No data or embeddings provided.")
        return

    # Ensure the embeddings directory exists
    os.makedirs(os.path.dirname(FAISS_INDEX_PATH), exist_ok=True)

    # 1. Create FAISS Index
    d = embeddings_matrix.shape[1] # Dimension of the embeddings
    index = faiss.IndexFlatL2(d)
    index.add(embeddings_matrix)
    
    # 2. Save the Index
    faiss.write_index(index, FAISS_INDEX_PATH)
    print(f"  -> FAISS index saved to {FAISS_INDEX_PATH}. Total vectors: {index.ntotal}")
    
    # 3. Save the Transaction Metadata (must correspond one-to-one with vectors)
    with open(DATA_INDEX_PATH, 'w') as f:
        json.dump(transactions, f, indent=4)
    print(f"  -> Metadata saved to {DATA_INDEX_PATH}.")

# Initialize the service for use in the API
vector_search_service = VectorSearchService()