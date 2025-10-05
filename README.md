# 💰 AI-Powered Financial Data Assistant

## 💡 Project Overview
This project implements an **AI-powered financial data assistant** that allows users to query synthetic bank transaction data using **natural language**.  
It leverages **vector embeddings** and **FAISS** for efficient semantic search (Retrieval-Augmented Generation or RAG), making it capable of finding transactions relevant to a query’s *meaning*, not just keywords.  
The API is built using **FastAPI** for easy deployment and testing.

---

## 🧠 Key Technologies
- **Vector Database:** FAISS (for high-performance similarity search)  
- **Embedding / LLM:** Google Gemini (for generating embeddings and optional summarization)  
- **API Framework:** FastAPI  
- **Data Generation:** Faker  

---

## 🛠️ Setup and Installation

### 1. Clone the Repository & Directory Structure
Ensure your project structure matches the layout below:

```
financial-data-assistant/
│
├── data/
├── embeddings/
├── app.py
├── services/
│   ├── data_generator.py
│   ├── embedding_service.py
│   └── vector_search_service.py
├── index.html
├── run_setup.py
└── README.md
```

---

### 2. Install Dependencies
Install all necessary Python packages:

```bash
pip install -r requirements.txt
```

---

### 3. API Key Configuration
The application uses the **Gemini API** for generating embeddings and summarization.  
The API key is sourced from an environment variable named `GOOGLE_API_KEY`.

**Create .env file**
```bash
    GOOGLE_API_KEY=YOUR_GOOGLE_API_KEY
```

---

## 🚀 Running the Application

The application requires a **two-step process**:  
1. Setup (data + vector index creation)  
2. API start  

### Step 1: Run Setup (Generate Data and Embeddings)
This step generates synthetic data (`data/transactions.json`) and the FAISS vector store (`embeddings/vector_store.faiss`).  
It calls the Gemini API to generate embeddings and may take a few minutes.

```bash
python run_setup.py
```

**Expected Output:**
```
--- Running AI Financial Data Assistant Setup ---

[1/3] Generating synthetic financial data (transactions.json)...
      -> Data generation complete.

[2/3] Generating embeddings and creating FAISS index (vector_store.faiss)...
      -> Loaded [X] transactions. Starting embedding generation...
      -> Processed [Y]/[X] transactions.
      -> Vector store creation complete.

[3/3] Setup Complete.
```

---

### Step 2: Start the API Server
Once setup is successful, start the FastAPI application using **Uvicorn**:

```bash
uvicorn api.app:app --reload
```

The API will be available at:  
👉 [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

---

## 📝 API Endpoints and Usage

### 1. **Search Endpoint** (`/query`)
Performs a semantic search against the FAISS vector store and returns relevant transactions.

| Method | Path   | Description |
|---------|--------|-------------|
| `GET`   | `/query` | Searches transactions based on a natural language prompt. |

**Query Parameters:**

| Parameter | Type | Required | Default | Description |
|------------|------|-----------|----------|-------------|
| `q` | string | ✅ Yes | N/A | The natural language query (e.g., *"Where did I spend the most money on food in October?"*) |
| `limit` | integer | ❌ No | 10 | Maximum number of transactions to return |
| `summarize` | boolean | ❌ No | False | If True, sends transactions to Gemini API for a summary |

