import streamlit as st

# PAGE CONFIG
st.set_page_config(
    page_title="AI Mutual Fund Advisor",
    layout="wide",
    initial_sidebar_state="expanded"
)

# REDIRECT TO DASHBOARD
st.switch_page("pages/1_Dashboard.py")