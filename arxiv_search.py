"""
ArXiv Search Module
==================
This file handles searching and downloading papers from ArXiv:
- Search for papers by topic
- Filter papers by page count
- Download papers automatically
"""

import streamlit as st
import arxiv
import re
import os
import tempfile


def find_and_download_paper(query, min_pages=15, max_pages=25):
    """
    Search ArXiv for papers and download the first suitable result.
    
    Args:
        query (str): Search query/topic
        min_pages (int): Minimum number of pages
        max_pages (int): Maximum number of pages
        
    Returns:
        dict: Paper information if found, None otherwise
    """
    try:
        st.info(f"🔍 Searching ArXiv for: '{query}'")
        st.info(f"📄 Looking for papers with {min_pages}-{max_pages} pages")
        
        # Create ArXiv client
        client = arxiv.Client(page_size=20, delay_seconds=3, num_retries=3)
        search = arxiv.Search(
            query=query,
            max_results=20,
            sort_by=arxiv.SortCriterion.Relevance,
        )

        # Progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Search through results
        for i, result in enumerate(client.results(search)):
            progress_bar.progress((i + 1) / 20)
            status_text.text(f"Checking paper {i + 1}/20: {result.title[:50]}...")
            
            # Check if paper has page information
            if result.comment and 'pages' in result.comment.lower():
                match = re.search(r'(\d+)\s*pages', result.comment, re.IGNORECASE)
                if match:
                    pages = int(match.group(1))
                    
                    # Check if paper meets page criteria
                    if min_pages <= pages <= max_pages:
                        try:
                            # Create temporary directory for download
                            temp_dir = tempfile.mkdtemp()
                            filename = f"{result.entry_id.split('/')[-1]}.pdf"
                            filepath = os.path.join(temp_dir, filename)
                            
                            status_text.text(f"📥 Downloading: {result.title}")
                            result.download_pdf(dirpath=temp_dir, filename=filename)
                            
                            # Clean up progress indicators
                            progress_bar.empty()
                            status_text.empty()
                            
                            # Return paper information
                            paper_info = {
                                "title": result.title,
                                "filepath": filepath,
                                "authors": [author.name for author in result.authors],
                                "summary": result.summary,
                                "pages": pages,
                                "arxiv_id": result.entry_id.split('/')[-1],
                                "published": result.published.strftime("%Y-%m-%d")
                            }
                            
                            st.success(f"✅ Downloaded: {result.title}")
                            return paper_info
                            
                        except Exception as e:
                            st.warning(f"⚠️ Download failed for {result.title}: {str(e)}")
                            continue

        # Clean up if no suitable paper found
        progress_bar.empty()
        status_text.empty()
        
        st.warning(f"❌ No papers found with {min_pages}-{max_pages} pages in top 20 results")
        return None

    except Exception as e:
        st.error(f"❌ ArXiv search failed: {str(e)}")
        return None


def display_paper_info(paper_info):
    """
    Display paper information in a nice format.
    
    Args:
        paper_info (dict): Paper information dictionary
    """
    if not paper_info:
        return
        
    with st.expander("📋 Paper Details", expanded=True):
        st.markdown(f"**📖 Title:** {paper_info['title']}")
        st.markdown(f"**👥 Authors:** {', '.join(paper_info['authors'])}")
        st.markdown(f"**📄 Pages:** {paper_info['pages']}")
        st.markdown(f"**🆔 ArXiv ID:** {paper_info['arxiv_id']}")
        st.markdown(f"**📅 Published:** {paper_info['published']}")
        
        # Show abstract (truncated)
        abstract = paper_info['summary']
        if len(abstract) > 500:
            st.markdown(f"**📝 Abstract:** {abstract[:500]}...")
            with st.expander("Show full abstract"):
                st.write(abstract)
        else:
            st.markdown(f"**📝 Abstract:** {abstract}")


def get_search_suggestions():
    """
    Get suggested search topics for users.
    
    Returns:
        list: List of suggested search topics
    """
    return [
        "multimodal large language models",
        "computer vision transformers",
        "natural language processing BERT",
        "machine learning neural networks",
        "deep learning convolutional networks",
        "artificial intelligence robotics",
        "reinforcement learning algorithms",
        "generative adversarial networks",
        "attention mechanisms transformers",
        "federated learning privacy"
    ]