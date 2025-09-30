import streamlit as st
from utils import predict
from model import LSTMClassifier

def render_ui():
    st.markdown(
        "<h1 style='text-align:center; color:#D1D5DB;'>ChatGPT-Style Fake News Detector</h1>",
        unsafe_allow_html=True
    )
    article = st.text_area(
        "Enter news article text (>= 400 words):",
        height=300,
        key="article_input"
    )
    if st.button("Detect", key="detect_btn"):
        wc = len(article.split())
        if wc < 400:
            st.warning(f"Please enter at least 400 words (you entered {wc}).")
        else:
            label = predict(article)
            if label == 1:
                st.success("✅ Real News (1)")
            else:
                st.error("🚨 Fake News (0)")