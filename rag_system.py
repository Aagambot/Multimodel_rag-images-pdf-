"""
RAG System Module
================
This file contains the core Retrieval-Augmented Generation system:
- Building the multimodal retriever
- Creating the RAG chain for Q&A
- Managing vector stores and document stores
"""

import streamlit as st
import uuid
from langchain.storage import InMemoryStore
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain.docstore.document import Document
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from PIL import Image

from pdf_processor import extract_pdf_elements
from image_processor import generate_image_summaries, convert_image_to_base64


def build_multimodal_retriever(pdf_path, api_key):
    """
    Build the core RAG retriever from a PDF file.
    This is the EXACT code from the notebook with error handling.
    
    Args:
        pdf_path (str): Path to the PDF file
        api_key (str): Google API key
        
    Returns:
        MultiVectorRetriever: The built retriever system
    """
    try:
        st.info("🔧 Starting multimodal retriever build process...")
        
        # EXACT code from your notebook
        id_key = "doc_id"
        
        # The vectorstore to use to index the child chunks
        try:
            vectorstore = Chroma(
                collection_name="multimodal_rag",
                embedding_function=SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
            )
            st.success("✅ Vector store created successfully")
        except Exception as e:
            st.error(f"❌ Vector store creation failed: {str(e)}")
            raise e
        
        # The storage layer for the parent documents
        try:
            docstore = InMemoryStore()
            st.success("✅ Document store created successfully")
        except Exception as e:
            st.error(f"❌ Document store creation failed: {str(e)}")
            raise e

        try:
            retriever = MultiVectorRetriever(
                vectorstore=vectorstore,
                docstore=docstore,
                id_key=id_key,
                search_kwargs={'k': 5}
            )
            st.success("✅ Multi-vector retriever created successfully")
        except Exception as e:
            st.error(f"❌ Multi-vector retriever creation failed: {str(e)}")
            raise e

        # Extract elements
        try:
            raw_text_chunks, raw_images = extract_pdf_elements(pdf_path)
            st.success(f"✅ Extracted {len(raw_text_chunks)} text chunks and {len(raw_images)} images")
        except Exception as e:
            st.error(f"❌ PDF element extraction failed: {str(e)}")
            raise e
        
        # Add text to retriever
        try:
            doc_ids_text = [str(uuid.uuid4()) for _ in raw_text_chunks]
            text_docs = [Document(page_content=chunk, metadata={id_key: doc_ids_text[i]}) 
                        for i, chunk in enumerate(raw_text_chunks)]
            retriever.vectorstore.add_documents(text_docs)
            retriever.docstore.mset(list(zip(doc_ids_text, raw_text_chunks)))
            st.success(f"✅ Added {len(raw_text_chunks)} text documents to retriever")
        except Exception as e:
            st.error(f"❌ Text document addition failed: {str(e)}")
            raise e

        # Add images and their summaries to retriever
        if raw_images:
            try:
                image_summaries = generate_image_summaries(raw_images, api_key)
                doc_ids_img = [str(uuid.uuid4()) for _ in raw_images]
                summary_docs = [Document(page_content=summary, metadata={id_key: doc_ids_img[i]}) 
                               for i, summary in enumerate(image_summaries)]
                retriever.vectorstore.add_documents(summary_docs)
                retriever.docstore.mset(list(zip(doc_ids_img, raw_images)))
                st.success(f"✅ Added {len(raw_images)} image summaries to retriever")
            except Exception as e:
                st.error(f"❌ Image processing failed: {str(e)}")
                raise e
        else:
            st.info("ℹ️ No images found in PDF")
        
        st.success("🎉 Multimodal retriever built successfully!")
        return retriever
        
    except Exception as e:
        st.error(f"❌ CRITICAL ERROR in build_multimodal_retriever:")
        st.error(f"Error Type: {type(e).__name__}")
        st.error(f"Error Message: {str(e)}")
        st.error(f"PDF Path: {pdf_path}")
        import traceback
        st.error(f"Full Traceback: {traceback.format_exc()}")
        raise e


def create_rag_chain(retriever, api_key):
    """
    Create the final RAG chain for querying.
    This is the EXACT code from the notebook with error handling.
    
    Args:
        retriever: The multimodal retriever
        api_key (str): Google API key
        
    Returns:
        Chain: The RAG chain for Q&A
    """
    try:
        st.info("🔗 Creating RAG chain...")
        
        def format_context_for_gemini(docs):
            """
            Format retrieved documents (text and images) for the Gemini model.
            Images are converted to Base64 data URIs.
            """
            try:
                context_parts = []
                for doc in docs:
                    # The MultiVectorRetriever returns the raw doc from the docstore.
                    # It can be a string (from text chunks) or a PIL Image object.
                    if isinstance(doc, str):
                        # Correctly append the text part
                        context_parts.append({"type": "text", "text": doc})
                    elif isinstance(doc, Image.Image):
                        # Convert PIL Image to Base64
                        img_base64 = convert_image_to_base64(doc)
                        context_parts.append({
                            "type": "image_url",
                            "image_url": f"data:image/jpeg;base64,{img_base64}",
                        })
                return context_parts
            except Exception as e:
                st.error(f"❌ Error formatting context: {str(e)}")
                raise e

        def create_prompt(context_parts, question):
            """Create a multimodal prompt from the context parts and question."""
            try:
                prompt_str = f"""You are an expert research assistant. Synthesize a comprehensive answer to the user's question using all the provided context below. The context contains both text excerpts and images from a research paper. You must use information from both the text and the images to form your answer.

        Question: {question}

        Context:
        """
                final_prompt_content = [{"type": "text", "text": prompt_str}]
                final_prompt_content.extend(context_parts)
                
                return [HumanMessage(content=final_prompt_content)]
            except Exception as e:
                st.error(f"❌ Error creating prompt: {str(e)}")
                raise e

        # Initialize the final, powerful model
        try:
            model = ChatGoogleGenerativeAI(
                model="gemini-2.5-flash",
                temperature=0.2,
                google_api_key=api_key
            )
            st.success("✅ Gemini model initialized successfully")
        except Exception as e:
            st.error(f"❌ Model initialization failed: {str(e)}")
            raise e
        
        # Define the final chain
        try:
            chain = (
                {"context": retriever | RunnableLambda(format_context_for_gemini), "question": RunnablePassthrough()}
                | RunnableLambda(lambda x: create_prompt(x["context"], x["question"]))
                | model
                | StrOutputParser()
            )
            st.success("✅ RAG chain created successfully")
            return chain
        except Exception as e:
            st.error(f"❌ Chain creation failed: {str(e)}")
            raise e
            
    except Exception as e:
        st.error(f"❌ CRITICAL ERROR in create_rag_chain:")
        st.error(f"Error Type: {type(e).__name__}")
        st.error(f"Error Message: {str(e)}")
        import traceback
        st.error(f"Full Traceback: {traceback.format_exc()}")
        raise e