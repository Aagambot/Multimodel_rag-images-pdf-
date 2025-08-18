📚 Multimodal Research Paper Q&A Agent
This project is a sophisticated AI agent that can find, process, and answer questions about scientific research papers from ArXiv. It leverages a multimodal Retrieval-Augmented Generation (RAG) pipeline, allowing it to understand and reason about both the text and the images within a PDF document.

The agent is implemented as a Jupyter Notebook (demo.ipynb) for experimentation and a Streamlit web application (streamlit_app.py) for an interactive user experience.

✨ Features
Dynamic Paper Sourcing: Finds relevant research papers on ArXiv based on a user-provided topic.

Intelligent Filtering: Automatically searches for papers within a reasonable page length (e.g., 15-25 pages) to ensure efficient processing.

Multimodal Processing: Extracts both text and images from PDF documents using PyMuPDF.

AI-Powered Image Summarization: Uses the gemini-2.5-flash model to generate rich, contextual summaries for all images in the paper.

Advanced RAG Pipeline: Employs a Multi-Vector Retriever to store text chunks and image summaries, enabling semantic search over the paper's full content.

Comprehensive Q&A: Utilizes the powerful gemini-2.5-flash model to synthesize answers based on a combination of retrieved text and images.

Interactive UI: A Streamlit application provides a user-friendly interface for entering topics and asking questions.

🛠️ Tech Stack
Core Logic: Python 3.x

LLM & RAG Framework: LangChain

Generative Models: Google Gemini 2.5 Flash

Web Interface: Streamlit

PDF Processing: PyMuPDF

Vector Store: ChromaDB (in-memory)

Embeddings: all-MiniLM-L6-v2 (SentenceTransformers)

Paper Sourcing: ArXiv Python Library

🚀 Setup and Installation
Clone the Repository:

git clone <your-repository-url>
cd <your-repository-name>

Create a Python Environment: It's highly recommended to use a virtual environment.

python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`

Install Dependencies: Install all required packages using the provided requirements.txt file (or install them manually).

pip install -r requirements.txt

If a requirements.txt is not available, install manually:

pip install langchain langchain-google-genai langchain_community pymupdf pillow chromadb transformers torch sentence-transformers arxiv streamlit

Set Up Your API Key: You need a Google API key to use the Gemini models.

Get your key from Google AI Studio.

The application will prompt you to enter the key when you run it for the first time.

▶️ How to Run
You can run this project in two ways:

1. Jupyter Notebook (demo.ipynb)
This is ideal for testing, debugging, and understanding the step-by-step logic of the RAG pipeline.

Start Jupyter Lab or Jupyter Notebook:

jupyter lab

Open the demo.ipynb file.

Run the cells in order from top to bottom. You can change the topic and question variables in the final cell to experiment with different papers and queries.

2. Streamlit Web Application (streamlit_app.py)
This provides a polished, interactive experience for end-users.

Ensure you are in the project's root directory in your terminal.

Run the following command:

streamlit run streamlit_app.py

Your web browser will automatically open with the application interface.

📁 File Structure
.
├── demo.ipynb          # Jupyter Notebook for development and 
└── README.md           # This file.
