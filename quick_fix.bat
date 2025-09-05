@echo off
echo 🔧 Quick Fix for Multimodal RAG Dependencies
echo ============================================

echo.
echo 1. Uninstalling conflicting packages...
pip uninstall -y torch torchvision transformers sentence-transformers google-generativeai langchain-google-genai

echo.
echo 2. Installing compatible versions...
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cpu
pip install transformers==4.36.0
pip install sentence-transformers==2.7.0
pip install google-generativeai==0.7.2
pip install langchain-google-genai==1.0.10
pip install langchain-community==0.2.16

echo.
echo 3. Testing imports...
python -c "import torch, transformers, sentence_transformers; from langchain_google_genai import ChatGoogleGenerativeAI; from langchain_community.embeddings import SentenceTransformerEmbeddings; print('✅ All imports successful!')"

echo.
echo 🎉 Fix complete! You can now run: streamlit run main_app.py
pause