# services/data_generator.py
import json
import random
from datetime import datetime, timedelta
from faker import Faker
from pydantic import BaseModel, Field
from typing import Literal

# --- Configuration ---
NUM_USERS = 3
TXN_PER_USER = 150  # 450 total transactions
DATA_FILE_PATH = "data/transactions.json"
CATEGORIES = ["Food", "Shopping", "Rent", "Salary", "Utilities", "Entertainment", "Travel", "Others"]
DEBIT_CATEGORIES = [c for c in CATEGORIES if c != "Salary"]

# --- Pydantic Data Model ---
class Transaction(BaseModel):
    id: str
    userId: str = Field(..., description="User identifier")
    date: str = Field(..., description="Transaction date in YYYY-MM-DD format")
    description: str
    amount: float
    type: Literal["Credit", "Debit"]
    category: str
    balance: float = Field(..., description="Account balance after the transaction")

def generate_data():
    """Generates synthetic financial transaction data for multiple users."""
    fake = Faker('en_IN') # Using Indian locale for relevant names/places
    all_transactions = []
    
    start_date = datetime.now() - timedelta(days=90)
    
    for i in range(1, NUM_USERS + 1):
        user_id = f"user_{i}"
        current_balance = random.uniform(50000, 100000)
        
        for j in range(TXN_PER_USER):
            txn_id = f"txn_{user_id}_{j+1}"
            txn_date = start_date + timedelta(days=random.randint(0, 90), hours=random.randint(0, 23))
            
            # 1 in 15 chance of salary (Credit)
            if j % 15 == 0:
                txn_type = "Credit"
                category = "Salary"
                amount = random.uniform(40000, 100000)
                description = f"Monthly salary deposit from {fake.company()}"
            else:
                txn_type = "Debit"
                category = random.choice(DEBIT_CATEGORIES)
                
                if category == "Rent":
                    amount = random.uniform(15000, 35000)
                    description = f"Rent payment for {txn_date.strftime('%B')} to landlord."
                elif category == "Utilities":
                    amount = random.uniform(1000, 5000)
                    description = f"{random.choice(['Electricity', 'Water', 'Internet'])} bill payment."
                elif category == "Food":
                    amount = random.uniform(100, 1500)
                    description = f"UPI payment to {random.choice(['Swiggy', 'Zomato', 'Local Cafe'])}."
                elif category == "Shopping":
                    amount = random.uniform(500, 10000)
                    description = f"Online purchase from {random.choice(['Amazon', 'Flipkart', 'Myntra'])}."
                else: # Entertainment, Travel, Others
                    amount = random.uniform(200, 8000)
                    description = fake.sentence(nb_words=6).replace('.', '')
            
            # Update balance
            if txn_type == "Credit":
                current_balance += amount
            else:
                current_balance -= amount
                
            transaction = Transaction(
                id=txn_id,
                userId=user_id,
                date=txn_date.strftime("%Y-%m-%d"),
                description=description,
                amount=round(amount, 2),
                type=txn_type,
                category=category,
                balance=round(current_balance, 2)
            )
            all_transactions.append(transaction.model_dump())

    # Save data to JSON file
    with open(DATA_FILE_PATH, 'w') as f:
        json.dump(all_transactions, f, indent=4)

if __name__ == '__main__':
    generate_data()
