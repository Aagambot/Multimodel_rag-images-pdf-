@echo off
echo 🚀 Starting Multimodal RAG App (Conda)
echo ======================================

echo Activating conda environment...
call conda activate multimodal-rag

if errorlevel 1 (
    echo ❌ Environment not found! Please run setup_conda.bat first
    pause
    exit /b 1
)

echo Starting Streamlit app...
streamlit run main_app.py

pause