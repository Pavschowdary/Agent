import streamlit as st
from database import DatabaseManager

def init_session_state():
    """Initialize session state variables"""
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False
    if 'user_id' not in st.session_state:
        st.session_state.user_id = None
    if 'username' not in st.session_state:
        st.session_state.username = None
    if 'db_manager' not in st.session_state:
        st.session_state.db_manager = DatabaseManager()
    if 'agent_orchestrator' not in st.session_state:
        st.session_state.agent_orchestrator = None

def login_form():
    """Display login form"""
    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submit = st.form_submit_button("Login")
        
        if submit:
            user_id = st.session_state.db_manager.authenticate_user(username, password)
            if user_id:
                st.session_state.logged_in = True
                st.session_state.user_id = user_id
                st.session_state.username = username
                st.success("Login successful!")
                st.rerun()
            else:
                st.error("Invalid username or password")

def signup_form():
    """Display signup form"""
    with st.form("signup_form"):
        username = st.text_input("Choose Username")
        password = st.text_input("Choose Password", type="password")
        confirm_password = st.text_input("Confirm Password", type="password")
        submit = st.form_submit_button("Sign Up")
        
        if submit:
            if password != confirm_password:
                st.error("Passwords do not match!")
            elif len(password) < 6:
                st.error("Password must be at least 6 characters long")
            elif len(username) < 3:
                st.error("Username must be at least 3 characters long")
            else:
                success = st.session_state.db_manager.create_user(username, password)
                if success:
                    st.success("Account created successfully! Please login.")
                else:
                    st.error("Username already exists!")

def logout():
    """Logout user"""
    st.session_state.logged_in = False
    st.session_state.user_id = None
    st.session_state.username = None
    st.session_state.agent_orchestrator = None
    st.success("Logged out successfully!")
    st.rerun()