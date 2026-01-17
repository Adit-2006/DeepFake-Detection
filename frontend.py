# frontend.py

import streamlit as st
import tempfile
import os
from pathlib import Path

from src.image_infer import infer_image
from src.video_infer import infer_video

# Page config
st.set_page_config(
    page_title="Deepfake Detection",
    page_icon="🕵️",
    layout="centered"
)

# Title and description
st.title("🕵️ Deepfake Detection")
st.markdown("Upload an image or video to analyze for synthetic content with timestamp localization.")

# Supported formats
IMAGE_EXTS = ["jpg", "jpeg", "png"]
VIDEO_EXTS = ["mp4", "avi", "mov", "mkv"]
ALL_EXTS = IMAGE_EXTS + VIDEO_EXTS

# Sidebar info
with st.sidebar:
    st.header("ℹ️ About")
    st.markdown("""
    This tool analyzes media files to detect potential deepfake content.
    
    **Supported formats:**
    - Images: JPG, PNG
    - Videos: MP4, AVI, MOV, MKV
    """)
    
    st.divider()
    st.caption("Note: Video processing may take longer, especially on CPU.")

# File uploader with better formatting
uploaded_file = st.file_uploader(
    "**Choose a file to analyze**",
    type=ALL_EXTS,
    help=f"Upload an image ({', '.join(IMAGE_EXTS)}) or video ({', '.join(VIDEO_EXTS)})"
)

# Session state to persist results
if 'result' not in st.session_state:
    st.session_state.result = None

# Clear results when new file is uploaded
if uploaded_file and 'prev_file' in st.session_state:
    if st.session_state.prev_file != uploaded_file.name:
        st.session_state.result = None
st.session_state.prev_file = uploaded_file.name if uploaded_file else None

def process_file(file_path: str, file_ext: str):
    """Process the uploaded file based on its type."""
    if file_ext.lower() in IMAGE_EXTS:
        # Display image
        col1, col2 = st.columns([1, 2])
        with col1:
            st.image(file_path, caption="Uploaded Image", use_column_width=True)
        
        # Analyze
        with st.spinner("🔍 Analyzing image..."):
            result = infer_image(file_path)
            st.session_state.result = result
        
    elif file_ext.lower() in VIDEO_EXTS:
        # Display video with better formatting
        st.markdown("**Uploaded Video Preview:**")
        st.video(file_path)
        
        # Add progress for video processing
        progress_bar = st.progress(0, text="Starting analysis...")
        with st.spinner("🎥 Processing video frames... This may take a while."):
            # Simulate progress updates (replace with actual progress if available)
            progress_bar.progress(20, text="Loading video...")
            result = infer_video(file_path)
            progress_bar.progress(100, text="Analysis complete!")
            st.session_state.result = result
        progress_bar.empty()

def display_results():
    """Display analysis results in an organized way."""
    if st.session_state.result:
        st.divider()
        st.subheader("📊 Analysis Results")
        
        # Display JSON in expander for cleaner UI
        with st.expander("View Raw JSON Results", expanded=False):
            st.json(st.session_state.result)
        
        # Add download button for results
        import json
        result_json = json.dumps(st.session_state.result, indent=2)
        st.download_button(
            label="📥 Download Results as JSON",
            data=result_json,
            file_name="deepfake_analysis_results.json",
            mime="application/json"
        )

# Main processing logic
if uploaded_file is not None:
    # Get file extension
    file_ext = Path(uploaded_file.name).suffix[1:].lower()
    
    # Validate extension
    if file_ext not in ALL_EXTS:
        st.error(f"Unsupported file type: .{file_ext}. Please upload a supported format.")
        st.stop()
    
    # Save to temp file
    try:
        suffix = f".{file_ext}"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded_file.read())
            file_path = tmp.name
        
        # Process file
        process_file(file_path, file_ext)
        
        # Display results
        display_results()
        
    except Exception as e:
        st.error(f"An error occurred during processing: {str(e)}")
        st.info("Please try again with a different file.")
    
    finally:
        # Clean up temp file
        if 'file_path' in locals() and os.path.exists(file_path):
            os.remove(file_path)

elif st.session_state.result:
    # Show previous results if no new file is uploaded
    st.info("No new file uploaded. Displaying previous results.")
    display_results()
else:
    # Initial state or no file
    st.info("👆 Please upload a file to begin analysis.")
    
    # Example placeholder
    with st.expander("See expected output format"):
        st.json({
            "is_fake": True,
            "confidence": 0.87,
            "timestamps": [{"start": 10.5, "end": 15.2, "confidence": 0.92}],
            "version": "1.0"
        })