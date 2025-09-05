"""
PDF Processing Module
====================
This file handles all PDF-related operations:
- Extracting text from PDFs
- Extracting images from PDFs
- Processing PDF content for the RAG system
"""

import streamlit as st
import fitz  # PyMuPDF
from PIL import Image
import io


@st.cache_data
def extract_pdf_elements(pdf_path):
    """
    Extract text and images from a PDF file.
    
    Args:
        pdf_path (str): Path to the PDF file
        
    Returns:
        tuple: (text_chunks, images) - lists of extracted text and images
    """
    try:
        st.info("📄 Opening PDF file...")
        doc = fitz.open(pdf_path)
        text_chunks, images = [], []
        
        # Create progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        st.info(f"📖 Processing {len(doc)} pages...")
        
        for i, page in enumerate(doc):
            # Update progress
            progress_bar.progress((i + 1) / len(doc))
            status_text.text(f"Processing page {i + 1}/{len(doc)}")
            
            # Extract text from page
            text = page.get_text()
            if text.strip():  # Only add non-empty text
                text_chunks.append(text)
            
            # Extract images from page
            for img_index, img in enumerate(page.get_images(full=True)):
                try:
                    xref = img[0]
                    base_image = doc.extract_image(xref)
                    image = Image.open(io.BytesIO(base_image["image"]))
                    images.append(image)
                except Exception as e:
                    st.warning(f"⚠️ Could not extract image {img_index} from page {i + 1}: {str(e)}")
        
        # Clean up progress indicators
        progress_bar.empty()
        status_text.empty()
        
        st.success(f"✅ Successfully extracted {len(text_chunks)} text chunks and {len(images)} images")
        return text_chunks, images
        
    except Exception as e:
        st.error(f"❌ Error processing PDF: {str(e)}")
        raise e


def get_pdf_info(pdf_path):
    """
    Get basic information about a PDF file.
    
    Args:
        pdf_path (str): Path to the PDF file
        
    Returns:
        dict: PDF information (pages, title, etc.)
    """
    try:
        doc = fitz.open(pdf_path)
        info = {
            'pages': len(doc),
            'title': doc.metadata.get('title', 'Unknown'),
            'author': doc.metadata.get('author', 'Unknown'),
            'subject': doc.metadata.get('subject', 'Unknown')
        }
        doc.close()
        return info
    except Exception as e:
        st.error(f"❌ Error getting PDF info: {str(e)}")
        return {'pages': 0, 'title': 'Error', 'author': 'Error', 'subject': 'Error'}