# api/app.py
import json
import numpy as np
import requests
import time
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, Query, HTTPException
from pydantic import BaseModel, Field
from fastapi.middleware.cors import CORSMiddleware

# Local imports
from services.embedding_service import call_embedding_api
from services.vector_search_service import vector_search_service
import os
# --- Configuration ---
API_KEY = os.getenv("GOOGLE_API_KEY")  # ADD YOUR GEMINI API KEY HERE (same as embedding_service.py)
SUMMARIZATION_MODEL = "gemini-2.5-flash"  # Free tier model
SUMMARIZATION_API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{SUMMARIZATION_MODEL}:generateContent?key={API_KEY}"
MAX_RETRIES = 5
DEFAULT_K = 50  # Fetch more results initially, then filter by userId

# --- Data Models for API ---
class SearchResult(BaseModel):
    id: str
    userId: str
    date: str
    description: str
    amount: float
    type: str
    category: str
    balance: float
    distance: float = Field(..., description="Cosine distance score (lower is more similar)")

class SearchResponse(BaseModel):
    query: str
    userId: Optional[str] = Field(None, description="User ID filter applied")
    summary: Optional[str] = Field(None, description="Optional LLM-generated summary of the results.")
    matches: List[SearchResult]

# --- FastAPI Setup ---
app = FastAPI(
    title="AI Financial Data Assistant API",
    description="Semantic Search API for financial transactions using Gemini Embeddings and FAISS.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# --- LLM Summarization Service ---
def summarize_transactions(query: str, transactions: List[Dict[str, Any]], user_id: Optional[str] = None) -> str:
    """Uses LLM to summarize the found transactions."""
    
    if not transactions:
        return "No relevant transactions were found to summarize."

    # Get current date context
    today = datetime.now()
    current_month = today.strftime("%B %Y")
    last_month = (today.replace(day=1) - timedelta(days=1)).strftime("%B %Y")
    
    # Format the transactions into a concise string for the LLM context
    txn_context = "\n".join([
        f"- {t['date']} | {t['type']} ₹{t['amount']} | {t['category']} | {t['description']}"
        for t in transactions
    ])
    
    # System Instruction: Expert Financial Analyst
    system_prompt = (
        "You are a helpful and concise financial data assistant. "
        "When answering queries about expenses or transactions:\n"
        "1. If asked about a specific time period (last month, previous month, etc.), ONLY include transactions from that period\n"
        "2. List transactions in chronological order (newest first)\n"
        "3. Format each transaction as: - Description – ₹Amount – Date\n"
        "4. After the list, provide a summary starting with '💬 Summary:'\n\n"
        "Always use the Indian Rupee symbol (₹) for amounts. Do not mention distances or search scores. "
        "Focus on transactions that match both the category (e.g., food, rent) AND the time period mentioned. "
        "Keep descriptions concise (use the merchant/vendor name). "
        "Calculate totals when showing multiple transactions."
    )
    
    user_context = f"Today's date is {today.strftime('%Y-%m-%d')}. The current month is {current_month}. Last month was {last_month}."
    if user_id:
        user_context += f" You are analyzing transactions for {user_id}."
    
    user_query = (
        f"{user_context}\n\n"
        f"The user asked: '{query}'. "
        f"Here are the most relevant transactions (Date | Type Amount | Category | Description):\n"
        f"{txn_context}\n\n"
        f"Based *only* on the transactions provided, please answer the user's query. "
        f"If asked about expenses, focus on Debit transactions. "
        f"If asked about top expenses, list them in descending order by amount with this format:\n"
        f"- Description – ₹Amount – Date\n\n"
        f"Then provide a summary line starting with '💬 Summary:' that includes total amount and key insights."
    )
    
    payload = {
        "contents": [{ "parts": [{ "text": user_query }] }],
        "systemInstruction": { "parts": [{ "text": system_prompt }] },
    }

    for attempt in range(MAX_RETRIES):
        try:
            response = requests.post(SUMMARIZATION_API_URL, json=payload, timeout=20)
            response.raise_for_status() 
            result = response.json()
            
            # Extract text from the response
            text = result.get('candidates', [{}])[0].get('content', {}).get('parts', [{}])[0].get('text')
            if text:
                return text
            
        except requests.exceptions.HTTPError as e:
            if response.status_code in [429, 500, 503] and attempt < MAX_RETRIES - 1:
                wait_time = 2 ** attempt
                time.sleep(wait_time)
            else:
                return f"LLM Summarization failed (HTTP Error {response.status_code})."
        except requests.exceptions.RequestException:
            return "LLM Summarization failed due to a network error."
        except Exception:
            return "LLM Summarization failed due to an unexpected error in the response format."
            
    return "LLM Summarization failed after multiple retries."


# --- API Endpoint ---
@app.get("/query", response_model=SearchResponse)
def semantic_search_query(
    q: str = Query(..., description="Natural language query for financial data."),
    userId: Optional[str] = Query(None, description="Filter results by user ID (e.g., user_1, user_2, user_3)"),
    k: int = Query(10, description="Number of top results to return after filtering."),
):
    """
    Accepts a natural language query, performs semantic search against the FAISS index,
    filters by userId if provided, and returns the most relevant transactions with an LLM summary.
    """
    
    # 1. Embed the query
    query_embedding_list = call_embedding_api(q)
    
    if not query_embedding_list:
        raise HTTPException(status_code=500, detail="Failed to generate embedding for the query.")
        
    query_embedding = np.array(query_embedding_list, dtype='float32')

    # 2. Find similar transactions (fetch more initially to account for filtering)
    found_transactions = vector_search_service.search_transactions(query_embedding, k=DEFAULT_K)
    
    # 3. Filter by userId if provided
    if userId:
        found_transactions = [txn for txn in found_transactions if txn['userId'] == userId]
        
        if not found_transactions:
            return SearchResponse(
                query=q, 
                userId=userId,
                summary=f"No relevant transactions were found for {userId}.", 
                matches=[]
            )
    
    # 4. Sort transactions by date (newest first)
    found_transactions.sort(key=lambda x: x['date'], reverse=True)
    
    # 5. Limit to top k results after filtering and sorting
    found_transactions = found_transactions[:k]
    
    if not found_transactions:
        return SearchResponse(query=q, userId=userId, summary="No relevant transactions were found in the database.", matches=[])

    # 5. Summarize results using LLM (with sorted data)
    summary_text = summarize_transactions(q, found_transactions, userId)
    
    # 6. Return formatted response
    return SearchResponse(
        query=q,
        userId=userId,
        summary=summary_text,
        matches=[SearchResult(**txn) for txn in found_transactions]
    )

# Ensure the vector search service is initialized when the app starts
vector_search_service._load_index()