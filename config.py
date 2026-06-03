# config.py
import os
from dotenv import load_dotenv

load_dotenv()

# This looks for keys locally in your hidden .env OR in Streamlit's Advanced Settings online
GEMINI_API_KEY = os.getenv('AQ.Ab8RN6J_5--UGkzcKcHBRGjU1FrthheZJ2cGefVvgGXJHy73nQ')
SERPER_API_KEY = os.getenv('8b9556f0b0dda79176f81835f3c9ce6d069fd7d4')