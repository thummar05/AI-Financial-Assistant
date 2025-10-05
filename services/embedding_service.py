# services/embedding_service.py
import json
import numpy as np
import requests
import time
from typing import List, Dict, Any, Optional
from services.data_generator import DATA_FILE_PATH
import os
from dotenv import load_dotenv

# Load environment variables from .env file if it exists
load_dotenv()
# --- Configuration ---
API_KEY = os.getenv("GOOGLE_API_KEY")  # Set this environment variable with your Gemini API key
EMBEDDING_MODEL_NAME = "text-embedding-004"
EMBEDDING_API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{EMBEDDING_MODEL_NAME}:embedContent?key={API_KEY}" if API_KEY else None
MAX_RETRIES = 5

def format_transaction_for_embedding(txn: Dict[str, Any]) -> str:
    """Creates a comprehensive text representation of a transaction."""
    amount_str = f"₹{txn['amount']}"
    
    if txn['type'] == 'Credit':
        action = f"Credit of {amount_str}"
        preposition = "from"
    else:
        action = f"Debit of {amount_str}"
        preposition = "for"

    return (
        f"{action} on {txn['date']} {preposition} '{txn['description']}' "
        f"under the '{txn['category']}' category. "
        f"The resulting balance was ₹{txn['balance']}."
    )

def call_embedding_api(text: str) -> Optional[List[float]]:
    """Calls the Gemini API to get an embedding with exponential backoff."""
    # CRITICAL: Check if API_KEY is missing before making the call
    if not API_KEY or not EMBEDDING_API_URL:
        print("  -> ERROR: GOOGLE_API_KEY environment variable is not set!")
        print("  -> Please set it using: set GOOGLE_API_KEY=your_api_key_here")
        return None
        
    payload = {
        "model": EMBEDDING_MODEL_NAME,
        "content": {
            "parts": [{"text": text}]
        }
    }
        
    for attempt in range(MAX_RETRIES):
        try:
            response = requests.post(EMBEDDING_API_URL, json=payload, timeout=10)
            
            # Detailed Error Logging
            if not response.ok:
                print(f"  -> API Error (Attempt {attempt+1}): HTTP {response.status_code}")
                try:
                    error_details = response.json()
                    print(f"     -> Details: {error_details.get('error', {}).get('message', 'No message available')}")
                except json.JSONDecodeError:
                    print(f"     -> Details: Non-JSON response received. Raw: {response.text[:100]}...")
                
                if response.status_code in [400, 401, 403]:
                    print("     -> Fatal Error (Bad Key/Quota/Input). Aborting retries.")
                    return None
                    
                if response.status_code in [429, 500, 503] and attempt < MAX_RETRIES - 1:
                    wait_time = 2 ** attempt
                    print(f"  -> Retrying in {wait_time}s due to temporary server/rate limit error...")
                    time.sleep(wait_time)
                else:
                    return None

            else:
                # Success
                result = response.json()
                embedding = result['embedding']['values']
                return embedding
            
        except requests.exceptions.RequestException as e:
            print(f"  -> Network error during embedding: {e}")
            return None
        except Exception as e:
            print(f"  -> Unexpected error during embedding: {e}")
            return None
            
    return None

def get_transaction_embeddings(transactions: List[Dict[str, Any]]) -> List[np.ndarray]:
    """Generates embeddings for a list of transactions."""
    embeddings = []
    
    for i, txn in enumerate(transactions):
        txn_text = format_transaction_for_embedding(txn)
        embedding_vector = call_embedding_api(txn_text)
        
        if embedding_vector:
            embeddings.append(np.array(embedding_vector, dtype='float32'))
        else:
            print(f"  -> Skipping transaction ID {txn['id']} due to failed embedding. Cannot proceed with FAISS.")
            return []
        
        if (i + 1) % 50 == 0:
            print(f"  -> Processed {i+1}/{len(transactions)} transactions.")
            
    return embeddings

def create_and_save_vector_store():
    """Main function to load data, generate embeddings, and save the FAISS index."""
    # Import here to avoid circular import and premature loading
    from services.vector_search_service import save_faiss_index
    
    try:
        with open(DATA_FILE_PATH, 'r') as f:
            transactions = json.load(f)
    except FileNotFoundError:
        print(f"Error: Transaction data not found at {DATA_FILE_PATH}. Run data_generator first.")
        return

    print(f"  -> Loaded {len(transactions)} transactions. Starting embedding generation...")
    
    # Check if API_KEY is set
    if not API_KEY:
        print("  -> ERROR: GOOGLE_API_KEY environment variable is not set!")
        print("  -> Please set it using: set GOOGLE_API_KEY=your_api_key_here")
        print("  -> Or create a .env file with: GOOGLE_API_KEY=your_api_key_here")
        return
    
    embeddings_list = get_transaction_embeddings(transactions)
    
    if embeddings_list:
        embeddings_matrix = np.vstack(embeddings_list)
        print(f"  -> Embeddings matrix shape: {embeddings_matrix.shape}")
        
        # Save the FAISS index
        save_faiss_index(embeddings_matrix, transactions)
    else:
        print("  -> ERROR: No embeddings were generated (likely due to API key or network issues). Cannot create vector store.")