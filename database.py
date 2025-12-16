import sqlite3
import pandas as pd
from datetime import datetime
import bcrypt
from typing import List, Dict, Any, Optional

class DatabaseManager:
    def __init__(self, db_name="financial_tracker.db"):
        self.db_name = db_name
        self.init_database()
    
    def get_connection(self):
        return sqlite3.connect(self.db_name)
    
    def init_database(self):
        with self.get_connection() as conn:
            # Users table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT UNIQUE NOT NULL,
                    password_hash TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Transactions table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS transactions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    purpose TEXT NOT NULL,
                    amount REAL NOT NULL,
                    transaction_date DATE NOT NULL,
                    transaction_type TEXT,
                    category TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users (id)
                )
            """)
            
            conn.commit()
    
    def create_user(self, username: str, password: str) -> bool:
        with self.get_connection() as conn:
            # Check if user exists
            cursor = conn.execute("SELECT id FROM users WHERE username = ?", (username,))
            if cursor.fetchone():
                return False
            
            # Hash password and create user
            password_hash = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
            conn.execute(
                "INSERT INTO users (username, password_hash) VALUES (?, ?)",
                (username, password_hash.decode('utf-8'))
            )
            conn.commit()
            return True
    
    def authenticate_user(self, username: str, password: str) -> Optional[int]:
        with self.get_connection() as conn:
            cursor = conn.execute(
                "SELECT id, password_hash FROM users WHERE username = ?",
                (username,)
            )
            result = cursor.fetchone()
            
            if result and bcrypt.checkpw(password.encode('utf-8'), result[1].encode('utf-8')):
                return result[0]  # user_id
            return None
    
    def add_transaction(self, user_id: int, purpose: str, amount: float, 
                       transaction_date: str, transaction_type: str = None, 
                       category: str = None) -> int:
        with self.get_connection() as conn:
            cursor = conn.execute("""
                INSERT INTO transactions 
                (user_id, purpose, amount, transaction_date, transaction_type, category)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (user_id, purpose, amount, transaction_date, transaction_type, category))
            conn.commit()
            return cursor.lastrowid
    
    def get_user_transactions(self, user_id: int, limit: int = 100) -> List[Dict]:
        with self.get_connection() as conn:
            cursor = conn.execute("""
                SELECT id, purpose, amount, transaction_date, transaction_type, category
                FROM transactions 
                WHERE user_id = ? 
                ORDER BY transaction_date DESC
                LIMIT ?
            """, (user_id, limit))
            
            columns = [desc[0] for desc in cursor.description]
            transactions = []
            for row in cursor.fetchall():
                transactions.append(dict(zip(columns, row)))
            return transactions
    
    def get_transactions_dataframe(self, user_id: int) -> pd.DataFrame:
        transactions = self.get_user_transactions(user_id)
        if transactions:
            df = pd.DataFrame(transactions)
            df['transaction_date'] = pd.to_datetime(df['transaction_date'])
            return df
        return pd.DataFrame()
    
    def get_summary_stats(self, user_id: int) -> Dict[str, Any]:
        with self.get_connection() as conn:
            cursor = conn.execute("""
                SELECT 
                    COUNT(*) as total_transactions,
                    SUM(amount) as total_amount,
                    AVG(amount) as avg_amount,
                    MIN(amount) as min_amount,
                    MAX(amount) as max_amount
                FROM transactions 
                WHERE user_id = ?
            """, (user_id,))
            
            result = cursor.fetchone()
            return {
                'total_transactions': result[0] or 0,
                'total_amount': result[1] or 0,
                'avg_amount': result[2] or 0,
                'min_amount': result[3] or 0,
                'max_amount': result[4] or 0
            }