@echo off
echo 🔧 Fixing PyTorch DLL Issue in Conda Environment
echo ================================================

echo Activating conda environment...
call conda activate multimodal-rag

echo.
echo Removing problematic PyTorch installation...
conda remove pytorch torchvision cpuonly -y

echo.
echo Installing PyTorch via pip (more reliable on Windows)...
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

echo.
echo Reinstalling sentence-transformers...
pip install --force-reinstall sentence-transformers

echo.
echo Testing installation...
python -c "import torch; import sentence_transformers; print('✅ PyTorch and sentence-transformers working!')"

echo.
echo 🎉 Fix complete! Try running the app again.
pause