# 🚀 Easy Setup Guide - No More Dependency Hell!

Choose the method that works best for you:

## 🥇 **Option 1: Conda Environment (Recommended)**

### **Super Simple - Just 2 Commands:**
```bash
# 1. Create environment with all dependencies
conda env create -f environment.yml

# 2. Run the app
conda activate multimodal-rag
streamlit run main_app.py
```

### **Even Easier - Use Batch Files (Windows):**
```bash
# 1. Setup everything
setup_easy.bat

# 2. Run app anytime
run_app.bat
```

---

## 🥈 **Option 2: Docker (Zero Setup)**

### **One Command Setup:**
```bash
# Build the container (one time)
docker build -t multimodal-rag .

# Run the app (anytime)
docker run -p 8501:8501 multimodal-rag
```

### **Access the app:**
Open: http://localhost:8501

---

## 🥉 **Option 3: Simple Pip (If others fail)**

```bash
# Create virtual environment
python -m venv rag_env
rag_env\Scripts\activate  # Windows
# source rag_env/bin/activate  # Mac/Linux

# Install simple requirements
pip install -r requirements_simple.txt

# Run app
streamlit run main_app.py
```

---

## 🎯 **Why These Work Better**

### **Conda Environment:**
- ✅ **Automatic dependency resolution**
- ✅ **No version conflicts**
- ✅ **Works with both Jupyter and Streamlit**
- ✅ **Isolated from system Python**

### **Docker:**
- ✅ **Completely isolated**
- ✅ **Same environment everywhere**
- ✅ **No installation issues**
- ✅ **Works on any system**

### **Simple Pip:**
- ✅ **Let pip figure out versions**
- ✅ **No manual version pinning**
- ✅ **Virtual environment isolation**

---

## 🆘 **If You Still Have Issues**

### **Check Your Setup:**
```bash
# Check if conda is installed
conda --version

# Check if Docker is installed
docker --version

# Check Python version
python --version
```

### **Quick Fixes:**
1. **Install Anaconda/Miniconda** if you don't have conda
2. **Install Docker Desktop** if you want the Docker option
3. **Use Python 3.11** (most compatible version)

---

## 🎉 **Success!**

Once setup is complete, you'll have:
- ✅ **Working Streamlit app**
- ✅ **Working Jupyter notebook**
- ✅ **No dependency conflicts**
- ✅ **Easy to run anytime**

**Just run:** `streamlit run main_app.py` and enjoy! 🚀