@echo off
echo 🚀 Conda Setup for Multimodal RAG
echo ==================================

echo.
echo This will create a clean conda environment with all dependencies.
echo Make sure you're running this in Anaconda Prompt!
echo.

pause

echo 1. Creating conda environment...
conda env create -f environment.yml

echo.
echo 2. Environment created successfully!
echo.
echo To use the app:
echo   1. Open Anaconda Prompt
echo   2. conda activate multimodal-rag  
echo   3. streamlit run main_app.py
echo.
echo Or just run: run_conda_app.bat
echo.
pause