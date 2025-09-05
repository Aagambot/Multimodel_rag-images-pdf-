"""
UI Components Module
===================
This file contains all the user interface components:
- Sidebar configuration
- File upload interface
- Chat interface
- Display components
"""

import streamlit as st
import tempfile
import os
from arxiv_search import find_and_download_paper, display_paper_info, get_search_suggestions
from pdf_processor import get_pdf_info


def setup_sidebar():
    """
    Set up the sidebar with configuration options.
    
    Returns:
        tuple: (api_key, source_option)
    """
    with st.sidebar:
        st.header("🔧 Configuration")
        
        # API Key input
        api_key = st.text_input(
            "Google API Key",
            type="password",
            help="Enter your Google API key for Gemini models. Get it from: https://makersuite.google.com/app/apikey"
        )
        
        if api_key:
            os.environ["GOOGLE_API_KEY"] = api_key
            st.success("✅ API Key set successfully!")
        else:
            st.warning("⚠️ Please enter your Google API Key to continue")
            st.info("💡 Get your free API key from Google AI Studio")
            return None, None
        
        st.divider()
        
        # Paper source selection
        st.header("📄 Paper Source")
        source_option = st.radio(
            "Choose how to get your research paper:",
            ["Upload PDF", "Search ArXiv"],
            help="Upload your own PDF or search ArXiv for papers"
        )
        
        st.divider()
        
        # Help section
        st.header("❓ Help")
        with st.expander("How to use this app"):
            st.markdown("""
            1. **Enter API Key**: Get free key from Google AI Studio
            2. **Choose Source**: Upload PDF or search ArXiv
            3. **Process Paper**: Wait for AI to analyze content
            4. **Ask Questions**: Chat with your research paper!
            """)
        
        # Clear session button
        if st.button("🔄 Clear Session", help="Reset all data and start fresh"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()
    
    return api_key, source_option


def handle_pdf_upload():
    """
    Handle PDF file upload interface.
    
    Returns:
        tuple: (file_path, file_name) if successful, (None, None) otherwise
    """
    st.subheader("📤 Upload Research Paper")
    
    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type="pdf",
        help="Upload a research paper in PDF format (max 200MB)"
    )
    
    if uploaded_file is not None:
        # Save uploaded file temporarily
        temp_dir = tempfile.mkdtemp()
        temp_path = os.path.join(temp_dir, uploaded_file.name)
        
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Display file info
        file_size = len(uploaded_file.getbuffer()) / (1024 * 1024)  # MB
        st.success(f"✅ Uploaded: {uploaded_file.name}")
        st.info(f"📊 File size: {file_size:.2f} MB")
        
        # Get and display PDF info
        pdf_info = get_pdf_info(temp_path)
        with st.expander("📋 PDF Information", expanded=True):
            st.write(f"**📄 Pages:** {pdf_info['pages']}")
            st.write(f"**📖 Title:** {pdf_info['title']}")
            st.write(f"**👤 Author:** {pdf_info['author']}")
        
        if st.button("🔄 Process PDF", type="primary"):
            return temp_path, uploaded_file.name
    
    return None, None


def handle_arxiv_search():
    """
    Handle ArXiv search interface.
    
    Returns:
        tuple: (file_path, paper_title) if successful, (None, None) otherwise
    """
    st.subheader("🔍 Search ArXiv")
    
    # Search suggestions
    suggestions = get_search_suggestions()
    selected_suggestion = st.selectbox(
        "💡 Quick suggestions (or type your own below):",
        [""] + suggestions,
        help="Select a suggested topic or enter your own search terms"
    )
    
    # Search input
    search_query = st.text_input(
        "🔎 Research topic",
        value=selected_suggestion,
        placeholder="e.g., multimodal large language models",
        help="Enter keywords to search for papers on ArXiv"
    )
    
    # Page range settings
    col_min, col_max = st.columns(2)
    with col_min:
        min_pages = st.number_input("📄 Min pages", value=15, min_value=1, max_value=100)
    with col_max:
        max_pages = st.number_input("📄 Max pages", value=25, min_value=1, max_value=100)
    
    if st.button("🔍 Search & Download", type="primary", disabled=not search_query):
        if search_query:
            paper_info = find_and_download_paper(search_query, min_pages, max_pages)
            
            if paper_info:
                display_paper_info(paper_info)
                return paper_info['filepath'], paper_info['title']
            else:
                st.error("❌ No suitable papers found. Try different keywords or page ranges.")
    
    return None, None


def setup_chat_interface():
    """
    Set up the chat interface for Q&A.
    
    Returns:
        str: User question if entered, None otherwise
    """
    st.subheader("💬 Ask Questions About Your Paper")
    
    if not st.session_state.paper_processed:
        st.info("👆 Please upload a PDF or search ArXiv to start asking questions!")
        return None
    
    # Display current paper info
    if st.session_state.current_paper:
        st.info(f"📖 Current paper: {st.session_state.current_paper}")
    
    # Display chat history
    if st.session_state.chat_history:
        st.subheader("📜 Chat History")
        for i, (question, answer) in enumerate(st.session_state.chat_history):
            with st.expander(f"Q{i+1}: {question[:50]}...", expanded=False):
                st.write(f"**❓ Question:** {question}")
                st.write(f"**🤖 Answer:** {answer}")
    
    # New question input
    question = st.text_area(
        "Your question:",
        placeholder="e.g., What are the main contributions of this paper?",
        help="Ask any question about the paper content, methodology, results, etc.",
        key="question_input"
    )
    
    # Action buttons
    col_ask, col_clear, col_sample = st.columns([2, 1, 1])
    
    with col_ask:
        ask_button = st.button("🤔 Ask Question", type="primary", disabled=not question)
    
    with col_clear:
        if st.button("🗑️ Clear Chat"):
            st.session_state.chat_history = []
            st.rerun()
    
    with col_sample:
        show_samples = st.button("💡 Sample Questions")
    
    # Sample questions
    if show_samples or not st.session_state.chat_history:
        display_sample_questions()
    
    return question if ask_button else None


def display_sample_questions():
    """Display sample questions that users can ask."""
    st.subheader("💡 Sample Questions")
    
    sample_questions = [
        "What are the main contributions of this paper?",
        "What methodology does the paper use?",
        "What are the key findings and results?",
        "What are the limitations mentioned in the paper?",
        "How does this work compare to previous research?",
        "What datasets were used in the experiments?",
        "What are the future research directions suggested?",
        "Can you explain the figures and charts in the paper?"
    ]
    
    for i, sq in enumerate(sample_questions):
        if st.button(f"📝 {sq}", key=f"sample_{i}"):
            st.session_state.question_input = sq
            st.rerun()


def display_features():
    """Display the features of the application."""
    st.subheader("✨ Features")
    
    features = [
        "🔍 **ArXiv Integration**: Automatically search and download research papers",
        "📄 **PDF Upload**: Upload your own research papers",
        "🖼️ **Multimodal Processing**: Analyze both text and images in papers",
        "🤖 **AI-Powered Q&A**: Ask questions in natural language",
        "💾 **Chat History**: Keep track of your questions and answers",
        "⚡ **Real-time Processing**: Get instant answers to your queries",
        "🎯 **Smart Search**: Filter papers by page count and relevance",
        "🔒 **Secure**: Your API key and data stay private"
    ]
    
    for feature in features:
        st.markdown(f"- {feature}")


def show_processing_status(step, total_steps, message):
    """
    Show processing status with progress.
    
    Args:
        step (int): Current step number
        total_steps (int): Total number of steps
        message (str): Status message
    """
    progress = step / total_steps
    st.progress(progress)
    st.info(f"Step {step}/{total_steps}: {message}")


def display_error_help():
    """Display help information for common errors."""
    with st.expander("🆘 Troubleshooting", expanded=False):
        st.markdown("""
        **Common Issues:**
        
        1. **API Key Error**: Make sure you have a valid Google API key
        2. **PDF Processing Error**: Ensure your PDF is not corrupted or password-protected
        3. **Memory Error**: Try with smaller PDF files (< 50MB)
        4. **Network Error**: Check your internet connection for ArXiv downloads
        
        **Need Help?**
        - Check the console for detailed error messages
        - Try refreshing the page and starting over
        - Make sure all dependencies are installed correctly
        """)


def initialize_session_state():
    """Initialize all session state variables."""
    if 'retriever' not in st.session_state:
        st.session_state.retriever = None
    if 'rag_chain' not in st.session_state:
        st.session_state.rag_chain = None
    if 'paper_processed' not in st.session_state:
        st.session_state.paper_processed = False
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'current_paper' not in st.session_state:
        st.session_state.current_paper = None