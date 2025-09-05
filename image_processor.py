"""
Image Processing Module
======================
This file handles all image-related operations:
- Converting images to base64 for AI processing
- Generating AI summaries of images
- Processing images for the multimodal RAG system
"""

import streamlit as st
import base64
import io
import time
from PIL import Image
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage


def generate_image_summaries(images, api_key):
    """
    Generate text summaries for a list of images using Google Gemini AI.
    
    Args:
        images (list): List of PIL Image objects
        api_key (str): Google API key
        
    Returns:
        list: List of text summaries for each image
    """
    try:
        st.info(f"🖼️ Processing {len(images)} images with AI...")
        
        # Initialize the AI model
        model = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash", 
            temperature=0,
            google_api_key=api_key
        )
        
        summaries = []
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, img in enumerate(images):
            # Update progress
            progress_bar.progress((i + 1) / len(images))
            status_text.text(f"Analyzing image {i + 1}/{len(images)} with AI...")
            
            try:
                # Convert image to base64
                img_base64 = convert_image_to_base64(img)
                
                # Create the AI prompt
                image_content = {
                    "type": "image_url",
                    "image_url": f"data:image/jpeg;base64,{img_base64}",
                }

                prompt = [HumanMessage(content=[
                    {
                        "type": "text", 
                        "text": "Describe this image from a research paper. What is it showing? Be detailed and focus on scientific content, charts, diagrams, or figures."
                    },
                    image_content,
                ])]
                
                # Get AI response
                response = model.invoke(prompt)
                summaries.append(response.content)
                
                # Rate limiting to avoid API limits
                time.sleep(1)
                
            except Exception as e:
                st.warning(f"⚠️ Could not analyze image {i + 1}: {str(e)}")
                summaries.append(f"Image {i + 1}: Could not generate AI summary - {str(e)}")
        
        # Clean up progress indicators
        progress_bar.empty()
        status_text.empty()
        
        st.success(f"✅ Generated AI summaries for {len(images)} images")
        return summaries
        
    except Exception as e:
        st.error(f"❌ Error generating image summaries: {str(e)}")
        raise e


def convert_image_to_base64(img):
    """
    Convert a PIL Image to base64 string for AI processing.
    
    Args:
        img (PIL.Image): PIL Image object
        
    Returns:
        str: Base64 encoded image string
    """
    try:
        buffered = io.BytesIO()
        
        # Convert RGBA to RGB if needed (for JPEG compatibility)
        if img.mode == 'RGBA':
            img = img.convert('RGB')
            
        # Save image as JPEG to buffer
        img.save(buffered, format="JPEG")
        
        # Encode to base64
        img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
        
        return img_base64
        
    except Exception as e:
        st.error(f"❌ Error converting image to base64: {str(e)}")
        raise e


def display_image_with_summary(img, summary, index):
    """
    Display an image alongside its AI-generated summary.
    
    Args:
        img (PIL.Image): PIL Image object
        summary (str): AI-generated summary text
        index (int): Image index number
    """
    with st.expander(f"🖼️ Image {index + 1} Analysis", expanded=False):
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.image(img, caption=f"Image {index + 1}", use_column_width=True)
            
        with col2:
            st.markdown("**AI Analysis:**")
            st.write(summary)


def validate_images(images):
    """
    Validate a list of images and filter out any problematic ones.
    
    Args:
        images (list): List of PIL Image objects
        
    Returns:
        list: List of valid PIL Image objects
    """
    valid_images = []
    
    for i, img in enumerate(images):
        try:
            # Basic validation
            if img is None:
                st.warning(f"⚠️ Image {i + 1} is None, skipping...")
                continue
                
            # Check image size (skip very small images)
            if img.size[0] < 50 or img.size[1] < 50:
                st.warning(f"⚠️ Image {i + 1} is too small ({img.size}), skipping...")
                continue
                
            # Check if image can be converted
            test_conversion = convert_image_to_base64(img)
            if len(test_conversion) > 0:
                valid_images.append(img)
            else:
                st.warning(f"⚠️ Image {i + 1} conversion failed, skipping...")
                
        except Exception as e:
            st.warning(f"⚠️ Image {i + 1} validation failed: {str(e)}")
            continue
    
    st.info(f"✅ Validated {len(valid_images)} out of {len(images)} images")
    return valid_images