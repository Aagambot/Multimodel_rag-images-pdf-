"""
Main Application
===============
This is the main Streamlit application file.
It brings together all the modules to create the complete app.

Run this file with: streamlit run main_app.py
"""

import streamlit as st

# Import our custom modules
from config import setup_page, load_custom_css, check_dependencies
from ui_components import (
    setup_sidebar, 
    handle_pdf_upload, 
    handle_arxiv_search, 
    setup_chat_interface,
    display_features,
    display_error_help,
    initialize_session_state
)
from rag_system_simple import build_multimodal_retriever_simple, create_rag_chain_simple


def process_paper(pdf_path, paper_title, api_key):
    """
    Process the paper and build the RAG system.
    
    Args:
        pdf_path (str): Path to the PDF file
        paper_title (str): Title of the paper
        api_key (str): Google API key
    """
    try:
        # Build the simple multimodal retriever (no PyTorch dependencies)
        with st.spinner("🔧 Building AI system for your paper..."):
            retriever = build_multimodal_retriever_simple(pdf_path, api_key)
            rag_chain = create_rag_chain_simple(retriever, api_key)
        
        # Update session state
        st.session_state.retriever = retriever
        st.session_state.rag_chain = rag_chain
        st.session_state.paper_processed = True
        st.session_state.current_paper = paper_title
        st.session_state.chat_history = []  # Reset chat history for new paper
        
        st.success("🎉 Paper processed successfully! You can now ask questions.")
        st.balloons()  # Celebration animation
        
    except Exception as e:
        st.error(f"❌ Error processing paper: {str(e)}")
        st.session_state.paper_processed = False
        display_error_help()


def main():
    """Main application function."""
    
    # Setup page configuration
    setup_page()
    load_custom_css()
    
    # Check dependencies
    if not check_dependencies():
        return
    
    # Initialize session state
    initialize_session_state()
    
    # Header
    st.markdown('<h1 class="main-header">📚 Multimodal Research Paper Q&A Agent</h1>', unsafe_allow_html=True)
    
    # Setup sidebar
    api_key, source_option = setup_sidebar()
    if not api_key:
        return
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    # Left column - Paper Input
    with col1:
        st.header("📥 Paper Input")
        
        if source_option == "Upload PDF":
            pdf_path, paper_name = handle_pdf_upload()
            if pdf_path and paper_name:
                process_paper(pdf_path, paper_name, api_key)
        
        else:  # ArXiv search
            pdf_path, paper_title = handle_arxiv_search()
            if pdf_path and paper_title:
                process_paper(pdf_path, paper_title, api_key)
    
    # Right column - Q&A Interface
    with col2:
        st.header("💬 Q&A Interface")
        
        if st.session_state.paper_processed and st.session_state.rag_chain:
            # Handle chat interface
            question = setup_chat_interface()
            
            if question:
                with st.spinner("🤔 Thinking about your question..."):
                    try:
                        answer = st.session_state.rag_chain.invoke(question)
                        st.session_state.chat_history.append((question, answer))
                        
                        # Display the latest answer
                        st.success("✅ Answer generated!")
                        with st.expander("💡 Latest Answer", expanded=True):
                            st.write(f"**❓ Question:** {question}")
                            st.write(f"**🤖 Answer:** {answer}")
                        
                    except Exception as e:
                        st.error(f"❌ Error generating answer: {str(e)}")
                        display_error_help()
        
        else:
            # Show features when no paper is loaded
            display_features()
    
    # Footer
    st.divider()
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 20px;'>
        <p>🚀 Built with Streamlit, LangChain, and Google Gemini AI</p>
        <p>📚 Multimodal Research Paper Q&A Agent</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()