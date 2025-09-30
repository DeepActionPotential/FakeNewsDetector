import streamlit as st
from ui import render_ui
from model import LSTMClassifier


st.set_page_config(
    page_title="Fake News Detector",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Dark-mode styling
st.markdown(
    """
    <style>
      .reportview-container, .main {
        background-color: #343541;
        color: #D1D5DB;
      }
      textarea, .stButton>button {
        background-color: #444654;
        color: #D1D5DB;
      }
      .stTextArea>div>div>textarea {
        background-color: #444654 !important;
      }
    </style>
    """,
    unsafe_allow_html=True
)

render_ui()
