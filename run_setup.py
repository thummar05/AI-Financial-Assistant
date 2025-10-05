# run_setup.py
import subprocess
import os
import sys

# Create necessary directories
os.makedirs("data", exist_ok=True)
os.makedirs("embeddings", exist_ok=True)

print("--- Running AI Financial Data Assistant Setup ---")

# Step 1: Generate Data
print("\n[1/3] Generating synthetic financial data (transactions.json)...")
try:
    from services.data_generator import generate_data
    generate_data()
    print("      -> Data generation complete.")
except ImportError as e:
    print(f"      -> ERROR: Could not import data_generator.py. {e}")
    sys.exit(1)
except Exception as e:
    print(f"      -> Data generation failed: {e}")
    sys.exit(1)

# Step 2: Generate Embeddings and FAISS Index
print("\n[2/3] Generating embeddings and creating FAISS index (vector_store.faiss)...")
try:
    # Import only the function, not the service instance
    from services.embedding_service import create_and_save_vector_store
    create_and_save_vector_store()
    print("      -> Vector store creation complete.")
except ImportError as e:
    print(f"      -> ERROR: Could not import embedding_service.py. {e}")
    sys.exit(1)
except Exception as e:
    print(f"      -> Vector store creation failed: {e}")
    sys.exit(1)

# Step 3: Instructions for running the API
print("\n[3/3] Setup Complete.")
print("\n--- Next Steps ---")
print("To start the API, run the following command in your terminal:")
print("uvicorn app:app --reload --host 0.0.0.0 --port 8000")
print("The API will be available at http://127.0.0.1:8000")
print("Test the endpoint: http://127.0.0.1:8000/query?q=biggest%20expenses%20last%20month")

if __name__ == "__main__":
    pass