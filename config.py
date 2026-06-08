import os
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# Universal Loader: Checks local laptop environment first, then checks native Streamlit Cloud Secrets
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY') or os.getenv('gemini_api_key')
SERPER_API_KEY = os.getenv('SERPER_API_KEY') or os.getenv('serper_api_key')

# If running on Streamlit Cloud and os.getenv fails, pull directly from st.secrets
if not GEMINI_API_KEY and "gemini_api_key" in st.secrets:
    GEMINI_API_KEY = st.secrets["gemini_api_key"]
if not SERPER_API_KEY and "serper_api_key" in st.secrets:
    SERPER_API_KEY = st.secrets["serper_api_key"]
