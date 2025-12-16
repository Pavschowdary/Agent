import os
from typing import Dict, Any, List
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langgraph.graph import StateGraph, END
from pydantic import BaseModel, Field
from datetime import datetime
import json

# Initialize LLM (using Gemini by default, fallback to Ollama)
class LLMController:
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if self.api_key:
            self.llm = ChatGoogleGenerativeAI(
                model="gemini-2.0-flash-exp",
                google_api_key=self.api_key,
                temperature=0.1
            )
        else:
            from langchain_ollama import ChatOllama
            self.llm = ChatOllama(model="llama3.2:3b", temperature=0.1)
        
        self.embeddings = OllamaEmbeddings(model="nomic-embed-text")
        self.vector_store = None
    
    def set_vector_store(self, user_id: int, transactions: List[Dict]):
        """Create vector store from transactions for semantic search"""
        documents = []
        for transaction in transactions:
            content = f"""
            Purpose: {transaction['purpose']}
            Amount: ${transaction['amount']}
            Date: {transaction['transaction_date']}
            Type: {transaction.get('transaction_type', 'N/A')}
            Category: {transaction.get('category', 'N/A')}
            """
            documents.append(Document(
                page_content=content,
                metadata={"id": transaction['id'], "user_id": user_id}
            ))
        
        if documents:
            self.vector_store = Chroma.from_documents(
                documents=documents,
                embedding=self.embeddings,
                collection_name=f"transactions_{user_id}"
            )

# Agent 1: Transaction Classifier
class TransactionClassifier:
    def __init__(self, llm_controller: LLMController):
        self.llm = llm_controller.llm
    
    def classify_transaction(self, purpose: str, amount: float) -> Dict[str, str]:
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a financial transaction classifier. 
            Analyze the transaction purpose and amount to determine:
            1. Transaction Type: Income, Expense, or Transfer
            2. Category: Food, Transportation, Shopping, Entertainment, 
               Bills, Healthcare, Education, Salary, Investment, Other
            
            Respond in JSON format with keys: 'transaction_type' and 'category'"""),
            ("human", "Purpose: {purpose}\nAmount: {amount}")
        ])
        
        chain = prompt | self.llm
        response = chain.invoke({"purpose": purpose, "amount": amount})
        
        try:
            # Extract JSON from response
            content = response.content
            if '```json' in content:
                json_str = content.split('```json')[1].split('```')[0].strip()
            elif '```' in content:
                json_str = content.split('```')[1].split('```')[0].strip()
            else:
                json_str = content.strip()
            
            result = json.loads(json_str)
            return {
                "transaction_type": result.get("transaction_type", "Expense"),
                "category": result.get("category", "Other")
            }
        except:
            return {"transaction_type": "Expense", "category": "Other"}

# Agent 2: Chat Agent
class TransactionChatAgent:
    def __init__(self, llm_controller: LLMController, db_manager, user_id: int):
        self.llm = llm_controller.llm
        self.db_manager = db_manager
        self.user_id = user_id
        self.llm_controller = llm_controller
        
        # Load transactions into vector store for semantic search
        transactions = self.db_manager.get_user_transactions(user_id)
        self.llm_controller.set_vector_store(user_id, transactions)
    
    def chat(self, query: str) -> str:
        # Retrieve relevant transactions using vector store
        context = ""
        if self.llm_controller.vector_store:
            docs = self.llm_controller.vector_store.similarity_search(query, k=5)
            context = "\n".join([doc.page_content for doc in docs])
        
        # Get summary statistics
        stats = self.db_manager.get_summary_stats(self.user_id)
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a helpful financial assistant analyzing the user's transaction data.
            Use the provided transaction history and statistics to answer questions accurately.
            Be specific and reference actual amounts and dates when possible.
            
            Available Statistics:
            - Total Transactions: {total_transactions}
            - Total Amount: ${total_amount:.2f}
            - Average Transaction: ${avg_amount:.2f}
            
            Transaction History:
            {context}
            
            Answer concisely and helpfully."""),
            ("human", "{query}")
        ])
        
        chain = prompt | self.llm
        response = chain.invoke({
            "context": context,
            "query": query,
            "total_transactions": stats['total_transactions'],
            "total_amount": stats['total_amount'],
            "avg_amount": stats['avg_amount']
        })
        
        return response.content

# Agent 3: Dashboard Agent
class DashboardAgent:
    def __init__(self, llm_controller: LLMController):
        self.llm = llm_controller.llm
    
    def create_dashboard(self, df: pd.DataFrame) -> Dict[str, Any]:
        if df.empty:
            return {"plot": None, "insights": "No transactions to display"}
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Expenses by Category',
                'Monthly Spending Trend',
                'Transaction Type Distribution',
                'Top 5 Transactions'
            ),
            specs=[
                [{"type": "pie"}, {"type": "scatter"}],
                [{"type": "bar"}, {"type": "bar"}]
            ]
        )
        
        # 1. Pie chart: Expenses by Category
        if 'category' in df.columns:
            category_sum = df.groupby('category')['amount'].sum().reset_index()
            fig.add_trace(
                go.Pie(
                    labels=category_sum['category'],
                    values=category_sum['amount'],
                    hole=0.3,
                    name="By Category"
                ),
                row=1, col=1
            )
        
        # 2. Line chart: Monthly Spending Trend
        df['month'] = df['transaction_date'].dt.to_period('M').astype(str)
        monthly_sum = df.groupby('month')['amount'].sum().reset_index()
        fig.add_trace(
            go.Scatter(
                x=monthly_sum['month'],
                y=monthly_sum['amount'],
                mode='lines+markers',
                name='Monthly Trend'
            ),
            row=1, col=2
        )
        
        # 3. Bar chart: Transaction Type Distribution
        if 'transaction_type' in df.columns:
            type_counts = df['transaction_type'].value_counts()
            fig.add_trace(
                go.Bar(
                    x=type_counts.index,
                    y=type_counts.values,
                    name='Type Distribution'
                ),
                row=2, col=1
            )
        
        # 4. Bar chart: Top 5 Transactions
        top_transactions = df.nlargest(5, 'amount')
        fig.add_trace(
            go.Bar(
                x=top_transactions['purpose'].str[:20] + "...",
                y=top_transactions['amount'],
                name='Top Transactions'
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            height=800,
            showlegend=True,
            title_text="Financial Dashboard",
            template="plotly_white"
        )
        
        # Generate insights using LLM
        insights = self._generate_insights(df)
        
        return {
            "plot": fig,
            "insights": insights,
            "dataframe": df
        }
    
    def _generate_insights(self, df: pd.DataFrame) -> str:
        if df.empty:
            return "No transactions available for analysis."
        
        summary = df.describe()
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a financial analyst. Provide 3-5 key insights based on this transaction data summary.
            Focus on spending patterns, unusual transactions, and recommendations.
            Be concise and actionable."""),
            ("human", f"""
            Transaction Summary Statistics:
            {summary.to_string()}
            
            Total Transactions: {len(df)}
            Date Range: {df['transaction_date'].min()} to {df['transaction_date'].max()}
            
            Provide insights:
            """)
        ])
        
        chain = prompt | self.llm
        response = chain.invoke({})
        return response.content

# Main Agent Orchestrator using LangGraph
class AgentOrchestrator:
    def __init__(self, api_key: str = None):
        self.llm_controller = LLMController(api_key)
        self.classifier = TransactionClassifier(self.llm_controller)
        self.dashboard_agent = DashboardAgent(self.llm_controller)
    
    def get_chat_agent(self, db_manager, user_id: int):
        return TransactionChatAgent(self.llm_controller, db_manager, user_id)