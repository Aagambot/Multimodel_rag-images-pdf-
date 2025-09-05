"""
Configuration and Setup
======================
This file handles all the configuration, imports, and setup for the application.
"""

import streamlit as st
import os
import tempfile
import uuid
import base64
import io
import time
from pathlib import Path

# Core libraries
import fitz  # PyMuPDF - for PDF processing
from PIL import Image  # for image handling
import arxiv  # for searching research papers
import re  # for text processing

# LangChain imports (updated to fix deprecation warnings)
from langchain.storage import InMemoryStore
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain.docstore.document import Document
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Try to import Google Generative AI with compatibility check
try:
    from langchain_google_genai import ChatGoogleGenerativeAI
    GOOGLE_AI_AVAILABLE = True
    IMPORT_ERROR = None
except ImportError as e:
    GOOGLE_AI_AVAILABLE = False
    IMPORT_ERROR = str(e)
except Exception as e:
    GOOGLE_AI_AVAILABLE = False
    IMPORT_ERROR = f"Unexpected error: {str(e)}"

# Streamlit page configuration
def setup_page():
    """Configure the Streamlit page settings"""
    st.set_page_config(
        page_title="Multimodal Research Paper Q&A Agent",
        page_icon="📚",
        layout="wide",
        initial_sidebar_state="expanded"
    )

# Custom CSS for better UI
def load_custom_css():
    """Load custom CSS styling for the application"""
    st.markdown("""
    <style>
        .main-header {
            font-size: 2.5rem;
            font-weight: bold;
            color: #1f77b4;
            text-align: center;
            margin-bottom: 2rem;
        }
        .feature-box {
            background-color: #f0f2f6;
            padding: 1rem;
            border-radius: 10px;
            margin: 1rem 0;
        }
        .success-box {
            background-color: #d4edda;
            border: 1px solid #c3e6cb;
            color: #155724;
            padding: 1rem;
            border-radius: 5px;
            margin: 1rem 0;
        }
        .error-box {
            background-color: #f8d7da;
            border: 1px solid #f5c6cb;
            color: #721c24;
            padding: 1rem;
            border-radius: 5px;
            margin: 1rem 0;
        }
    </style>
    """, unsafe_allow_html=True)

def check_dependencies():
    """Check if all required dependencies are available"""
    if not GOOGLE_AI_AVAILABLE:
        st.error("❌ Google Generative AI is not properly installed!")
        
        if IMPORT_ERROR:
            st.error(f"**Error Details:** {IMPORT_ERROR}")
        
        st.markdown("""
        ### 🔧 Quick Fix:
        
        **Option 1: Run the fix script (Recommended)**
        ```bash
        python fix_dependencies.py
        ```
        
        **Option 2: Manual installation**
        ```bash
        pip uninstall -y google-generativeai langchain-google-genai
        pip install google-generativeai==0.7.2
        pip install langchain-google-genai==1.0.10
        pip install langchain-community==0.2.16
        ```
        
        **Then restart the application:**
        ```bash
        streamlit run main_app.py
        ```
        """)
        
        st.info("💡 **Tip:** The fix script automatically handles all version conflicts!")
        st.stop()
        return False
    return True