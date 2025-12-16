import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
from datetime import datetime
import plotly.graph_objects as go

from database import DatabaseManager
from auth import init_session_state, login_form, signup_form, logout
from agents import AgentOrchestrator

# Page configuration
st.set_page_config(
    page_title="Financial Tracker",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Initialize session state
init_session_state()

# Initialize agents if logged in
if st.session_state.logged_in and st.session_state.agent_orchestrator is None:
    st.session_state.agent_orchestrator = AgentOrchestrator()

# Main header
st.title("💰 Financial Tracker AI")

# Menu configuration based on login state
if not st.session_state.logged_in:
    # Not logged in - Show Home, Login, SignUp
    selected = option_menu(
        menu_title=None,
        options=["Home", "Login", "SignUp"],
        icons=["house", "box-arrow-in-right", "person-plus"],
        menu_icon="cast",
        default_index=0,
        orientation="horizontal",
    )
    
    if selected == "Home":
        st.header("Welcome to Financial Tracker AI")
        
        st.subheader("🚀 Features:")
        st.markdown("""
        - **Smart Transaction Classification**: AI-powered categorization of your expenses
        - **Natural Language Chat**: Ask questions about your finances in plain English
        - **Intelligent Dashboards**: Automated insights and visualizations
        - **Secure Storage**: Your data stays private and secure
        """)
        
        st.subheader("📊 How It Works:")
        st.markdown("""
        1. **Sign up** for a free account
        2. **Log in** to access your dashboard
        3. **Add transactions** with purpose, amount, and date
        4. **Chat with AI** to get insights about your spending
        5. **View dashboards** with automated visualizations
        """)
        
        st.subheader("🔒 Security:")
        st.markdown("""
        - All passwords are encrypted using bcrypt
        - Your transaction data is stored securely
        - No sharing of personal financial information
        """)
        
        st.subheader("🛠️ Tech Stack:")
        st.markdown("""
        - **LangChain & LangGraph** for AI agent orchestration
        - **Gemini AI / Ollama** for natural language understanding
        - **SQLite** for secure data storage
        - **Plotly** for interactive visualizations
        - **Streamlit** for beautiful UI
        """)
        
        st.info("Ready to take control of your finances? Sign up now! 🎯")
        
    elif selected == "Login":
        st.header("🔐 Login to Your Account")
        login_form()
        
    elif selected == "SignUp":
        st.header("📝 Create New Account")
        signup_form()

else:
    # Logged in - Show Transaction Entry, Chat, Dashboard, Logout
    selected = option_menu(
        menu_title=None,
        options=["Transaction Entry", "Chat", "Dashboard", "Logout"],
        icons=["plus-circle", "chat", "bar-chart", "box-arrow-right"],
        menu_icon="cast",
        default_index=0,
        orientation="horizontal",
    )
    
    # Welcome message
    st.sidebar.success(f"Welcome, {st.session_state.username}!")
    
    if selected == "Transaction Entry":
        st.header("💳 Add New Transaction")
        
        col1, col2 = st.columns(2)
        
        with col1:
            purpose = st.text_area("Transaction Purpose", 
                                 placeholder="e.g., Grocery shopping at Walmart, Salary deposit, Amazon purchase...")
            amount = st.number_input("Amount (Rs)", min_value=0.01, step=0.01, format="%.2f")
        
        with col2:
            transaction_date = st.date_input("Transaction Date", value=datetime.now())
            auto_classify = st.checkbox("Auto-classify using AI", value=True)
        
        if st.button("Add Transaction", type="primary"):
            if purpose and amount > 0:
                # Classify transaction if enabled
                transaction_type = None
                category = None
                
                if auto_classify and st.session_state.agent_orchestrator:
                    with st.spinner("🤖 AI is classifying your transaction..."):
                        classification = st.session_state.agent_orchestrator.classifier.classify_transaction(
                            purpose, amount
                        )
                        transaction_type = classification["transaction_type"]
                        category = classification["category"]
                        st.info(f"AI Classification: {transaction_type} - {category}")
                
                # Add to database
                transaction_id = st.session_state.db_manager.add_transaction(
                    user_id=st.session_state.user_id,
                    purpose=purpose,
                    amount=amount,
                    transaction_date=transaction_date.strftime("%Y-%m-%d"),
                    transaction_type=transaction_type,
                    category=category
                )
                
                st.success(f"Transaction added successfully! (ID: {transaction_id})")
                
            else:
                st.error("Please fill in all required fields!")
    
    elif selected == "Chat":
        st.header("💬 Chat with Your Financial Assistant")
        
        # Initialize chat agent
        chat_agent = st.session_state.agent_orchestrator.get_chat_agent(
            st.session_state.db_manager, st.session_state.user_id
        )
        
        # Chat interface
        if "messages" not in st.session_state:
            st.session_state.messages = [
                {"role": "assistant", "content": "Hello! I'm your financial assistant. Ask me anything about your transactions!"}
            ]
        
        # Display chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])
        
        # Chat input
        if prompt := st.chat_input("Ask about your transactions..."):
            # Add user message
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.write(prompt)
            
            # Get AI response
            with st.chat_message("assistant"):
                with st.spinner("🤖 Thinking..."):
                    response = chat_agent.chat(prompt)
                    st.write(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
        
        # Example questions
        with st.expander("💡 Example Questions"):
            st.write("""
            - "What was my biggest expense this month?"
            - "How much did I spend on groceries?"
            - "Show me all transactions related to Amazon"
            - "What's my spending pattern like?"
            - "Give me insights on my recent expenses"
            """)
    
    elif selected == "Dashboard":
        st.header("📊 Financial Dashboard")
        
        # Get transaction data
        df = st.session_state.db_manager.get_transactions_dataframe(st.session_state.user_id)
        
        if df.empty:
            st.info("No transactions yet. Add some transactions to see your dashboard!")
        else:
            # Calculate financial metrics
            total_transactions = len(df)
            
            # Check if transaction_type column exists, if not create default values
            if 'transaction_type' not in df.columns:
                # Try to infer from amount or use default
                df['transaction_type'] = df['amount'].apply(lambda x: 'Income' if x >= 0 else 'Expense')
            
            # Calculate debit amount (Expenses/Withdrawals)
            debit_amount = df[df['transaction_type'].str.contains('Expense|expense|Debit|debit', na=False)]['amount'].abs().sum()
            
            # Calculate credit amount (Income/Deposits)
            credit_amount = df[df['transaction_type'].str.contains('Income|income|Credit|credit|Salary|salary', na=False)]['amount'].abs().sum()
            
            # Calculate total amount (absolute sum of all transactions)
            total_amount = df['amount'].abs().sum()
            
            # Calculate net total (credit - debit)
            net_total = credit_amount - debit_amount
            
            # Display financial metrics
            st.subheader("Financial Summary")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    label="Total Amount",
                    value=f"${total_amount:,.2f}",
                    help="Sum of absolute values of all transactions"
                )
            
            with col2:
                st.metric(
                    label="Credit Amount",
                    value=f"${credit_amount:,.2f}",
                    delta=f"{(credit_amount/total_amount*100):.1f}%" if total_amount > 0 else "0%",
                    delta_color="normal",
                    help="Total income/deposit amount"
                )
            
            with col3:
                st.metric(
                    label="Debit Amount",
                    value=f"${debit_amount:,.2f}",
                    delta=f"{(debit_amount/total_amount*100):.1f}%" if total_amount > 0 else "0%",
                    delta_color="inverse",
                    help="Total expense/withdrawal amount"
                )
            
            with col4:
                # Determine color for net total
                delta_color = "normal" if net_total >= 0 else "inverse"
                st.metric(
                    label="Net Total",
                    value=f"${net_total:,.2f}",
                    delta=f"{(net_total/credit_amount*100):.1f}%" if credit_amount > 0 else "0%",
                    delta_color=delta_color,
                    help="Net amount (Credit - Debit)"
                )
            
            # Additional metrics row
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Transactions", total_transactions)
            
            with col2:
                avg_transaction = total_amount / total_transactions if total_transactions > 0 else 0
                st.metric("Average Transaction", f"${avg_transaction:,.2f}")
            
            with col3:
                min_amount = df['amount'].min()
                st.metric("Min Transaction", f"${min_amount:,.2f}")
            
            with col4:
                max_amount = df['amount'].max()
                st.metric("Max Transaction", f"${max_amount:,.2f}")
            
            # Display transaction breakdown by type
            st.subheader("Transaction Breakdown")
            
            if 'transaction_type' in df.columns:
                type_summary = df.groupby('transaction_type').agg({
                    'amount': ['count', 'sum', 'mean']
                }).round(2)
                
                # Format the summary dataframe
                type_summary.columns = ['Count', 'Total Amount', 'Average Amount']
                type_summary = type_summary.reset_index()
                
                # Display type breakdown
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**By Transaction Type:**")
                    st.dataframe(type_summary, width='stretch')
                
                with col2:
                    # Create pie chart for transaction types
                    if not type_summary.empty:
                        fig_pie = go.Figure(data=[go.Pie(
                            labels=type_summary['transaction_type'],
                            values=type_summary['Total Amount'].abs(),
                            hole=0.3,
                            textinfo='label+percent',
                            insidetextorientation='radial'
                        )])
                        fig_pie.update_layout(
                            title="Transaction Type Distribution",
                            height=400
                        )
                        st.plotly_chart(fig_pie, width='stretch')
            
            # Display raw data
            st.subheader("Transaction Data")
            st.dataframe(df, width='stretch')
            
            # Download option
            csv = df.to_csv(index=False)
            st.download_button(
                label="📥 Download CSV",
                data=csv,
                file_name=f"transactions_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv",
            )
            
            # Generate AI dashboard
            with st.spinner("🤖 Creating intelligent dashboard..."):
                dashboard = st.session_state.agent_orchestrator.dashboard_agent.create_dashboard(df)
                
                # Display AI-generated plot
                if dashboard["plot"]:
                    st.subheader("AI-Generated Visualizations")
                    st.plotly_chart(dashboard["plot"], width='stretch')
                
                # Display AI insights
                st.subheader("AI Insights")
                st.write(dashboard["insights"])
                
                # Quick stats
                st.subheader("Category Analysis")
                
                if 'category' in df.columns:
                    category_summary = df.groupby('category').agg({
                        'amount': ['count', 'sum']
                    }).round(2)
                    category_summary.columns = ['Count', 'Total Amount']
                    category_summary = category_summary.sort_values('Total Amount', ascending=False).reset_index()
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**Top Categories:**")
                        st.dataframe(category_summary.head(), width='stretch')
                    
                    with col2:
                        if not category_summary.empty:
                            top_categories = category_summary.head(5)
                            fig_bar = go.Figure(data=[go.Bar(
                                x=top_categories['category'],
                                y=top_categories['Total Amount'].abs(),
                                text=top_categories['Total Amount'].abs().apply(lambda x: f'${x:,.0f}'),
                                textposition='auto',
                            )])
                            fig_bar.update_layout(
                                title="Top 5 Categories by Amount",
                                xaxis_title="Category",
                                yaxis_title="Amount ($)",
                                height=400
                            )
                            st.plotly_chart(fig_bar, width='stretch')
    
    elif selected == "Logout":
        st.header("👋 Logout")
        st.warning("Are you sure you want to logout?")
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("Yes, Logout"):
                logout()

# Footer
st.divider()
st.caption("Financial Tracker AI • Built with Streamlit, LangChain, and ❤️")