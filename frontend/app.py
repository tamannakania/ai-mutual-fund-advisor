import streamlit as st

st.set_page_config(
    page_title="AI Mutual Fund Advisor",
    layout="wide",
    initial_sidebar_state="expanded"
)

<<<<<<< HEAD
# Only switch pages when running inside a Streamlit session.
# Importing/running this module in plain Python (or in some IDE test runners)
# can raise NoSessionContext.
try:
    from streamlit.runtime.scriptrunner import get_script_run_ctx
    ctx = get_script_run_ctx()
except Exception:
    ctx = None

if ctx is not None:
    st.switch_page("pages/1_Dashboard.py")
else:
    st.warning("Run this app using: streamlit run frontend/app.py")
=======
st.switch_page("pages/1_Dashboard.py")
>>>>>>> eb50b25 (Updated ML models and frontend dashboard)
