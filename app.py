"""
Text-Based GenAI Application with RAG
======================================
A Retrieval-Augmented Generation (RAG) application that:
- Retrieves content from user-provided URLs
- Generates embeddings for semantic search
- Uses Gemini 2.5 Flash for grounded response generation

Author: [Your Name]
Date: 09/01/2026
Course: Day 10 - Final Project
"""

import streamlit as st
import os
from typing import List, Tuple, Optional
from utils.rag_pipeline import RAGPipeline
from utils.content_loader import ContentLoader
from utils.ui_components import render_header, render_settings, render_inputs

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="Gemini 2.5 Flash RAG App",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# INITIALIZATION
# ============================================================================

@st.cache_resource
def initialize_rag_pipeline():
    """
    Initialize RAG pipeline with Gemini API key.
    
    Uses Streamlit secrets for deployment or environment variables for local dev.
    Cached to avoid re-initialization on every rerun.
    
    Returns:
        RAGPipeline: Initialized RAG pipeline instance
        
    Raises:
        SystemExit: If API key is not found
    """
    try:
        # Try to get API key from secrets
        api_key = st.secrets.get("GEMINI_API_KEY")
    except Exception as e:
        # If secrets don't exist or can't be accessed, try environment variable
        api_key = os.getenv("GEMINI_API_KEY")
    
    if not api_key:
        st.error("""
        ⚠️ **Gemini API Key Not Found**
        
        Please set your API key:
        - **Local Development**: Create `.streamlit/secrets.toml` with `GEMINI_API_KEY`
        - **Streamlit Cloud**: Add secret in app settings
        
        **Current Status:**
        - Secrets accessible: Check logs for details
        - Environment variable: Not set
        """)
        st.stop()
    
    try:
        return RAGPipeline(api_key=api_key)
    except Exception as e:
        st.error(f"Failed to initialize RAG pipeline: {str(e)}")
        st.exception(e)
        st.stop()

# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    """
    Main application function that orchestrates the RAG pipeline.
    
    Flow:
    1. Initialize RAG pipeline
    2. Render UI components
    3. Process user inputs (URL + question)
    4. Execute RAG pipeline
    5. Display results
    """
    # Render header
    render_header()
    
    # Initialize RAG pipeline
    try:
        rag_pipeline = initialize_rag_pipeline()
    except Exception as e:
        st.error(f"Failed to initialize RAG pipeline: {str(e)}")
        st.exception(e)
        st.stop()
    
    # Sidebar settings
    settings = render_settings()
    
    # Main input area
    url, question = render_inputs()
    
    # Process button
    if st.button("🚀 Generate Answer", type="primary", use_container_width=True):
        process_query(rag_pipeline, url, question, settings)

def process_query(
    rag_pipeline: RAGPipeline,
    url: str,
    question: str,
    settings: dict
):
    """
    Process user query through the RAG pipeline.
    
    Args:
        rag_pipeline: Initialized RAG pipeline instance
        url: Source URL for content retrieval
        question: User's question
        settings: Dictionary containing UI settings (top_k, threshold, debug)
    
    Steps:
    1. Validate inputs
    2. Load content from URL
    3. Generate embeddings
    4. Retrieve relevant chunks
    5. Generate grounded response
    6. Display results
    """
    # Validation
    if not validate_inputs(url, question):
        return
    
    # Progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # Step 1: Load content
        status_text.text("🔄 Loading content from URL...")
        progress_bar.progress(10)
        
        content_loader = ContentLoader()
        text_content = content_loader.load_from_url(url)
        
        if not text_content:
            st.error("❌ Failed to load content. Please check the URL and try again.")
            return
        
        if len(text_content) < 100:
            st.warning("⚠️ Content retrieved is very short. Results may be limited.")
        
        # Step 2: Process through RAG pipeline
        status_text.text("🔄 Processing content through RAG pipeline...")
        progress_bar.progress(30)
        
        # Generate embeddings and retrieve context
        relevant_chunks = rag_pipeline.retrieve_context(
            query=question,
            document=text_content,
            top_k=settings['top_k'],
            similarity_threshold=settings['threshold']
        )
        
        progress_bar.progress(70)
        
        # Debug information
        if settings['debug']:
            display_debug_info(rag_pipeline, relevant_chunks)
        
        # Step 3: Generate response
        status_text.text("🤖 Generating answer with Gemini 2.5 Flash...")
        progress_bar.progress(85)
        
        answer = rag_pipeline.generate_response(
            question=question,
            context_chunks=relevant_chunks
        )
        
        progress_bar.progress(100)
        status_text.empty()
        
        # Display results
        display_results(answer, relevant_chunks)
        
    except Exception as e:
        st.error(f"❌ Error processing query: {str(e)}")
        if settings['debug']:
            st.exception(e)
    finally:
        progress_bar.empty()

def validate_inputs(url: str, question: str) -> bool:
    """
    Validate user inputs before processing.
    
    Args:
        url: URL string to validate
        question: Question string to validate
        
    Returns:
        bool: True if inputs are valid, False otherwise
    """
    if not url or not url.strip():
        st.error("❌ Please enter a valid URL")
        return False
    
    if not question or not question.strip():
        st.error("❌ Please enter a question")
        return False
    
    # URL format validation
    from urllib.parse import urlparse
    try:
        result = urlparse(url)
        if not all([result.scheme, result.netloc]):
            st.error("❌ Invalid URL format. Please include http:// or https://")
            return False
    except Exception:
        st.error("❌ Invalid URL format")
        return False
    
    return True

def display_debug_info(rag_pipeline: RAGPipeline, relevant_chunks: List[Tuple[str, float]]):
    """
    Display debug information for transparency and testing.
    
    Args:
        rag_pipeline: RAG pipeline instance
        relevant_chunks: List of (chunk, similarity_score) tuples
    """
    st.subheader("🔍 Debug Information")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Chunks", rag_pipeline.get_chunk_count())
    
    with col2:
        st.metric("Relevant Chunks", len(relevant_chunks))
    
    with col3:
        avg_similarity = sum(score for _, score in relevant_chunks) / len(relevant_chunks) if relevant_chunks else 0
        st.metric("Avg Similarity", f"{avg_similarity:.3f}")
    
    if relevant_chunks:
        st.write("**Top Chunks with Similarity Scores:**")
        for i, (chunk, score) in enumerate(relevant_chunks, 1):
            with st.expander(f"Chunk {i} (Similarity: {score:.3f})"):
                st.text(chunk[:500] + "..." if len(chunk) > 500 else chunk)

def display_results(answer: str, relevant_chunks: List[Tuple[str, float]]):
    """
    Display the generated answer and metadata.
    
    Args:
        answer: Generated answer string
        relevant_chunks: List of chunks used for context
    """
    st.markdown("---")
    st.subheader("📝 Generated Answer")
    
    # Answer display
    st.info(answer)
    
    # Metadata
    if relevant_chunks:
        st.caption(
            f"✅ Answer generated based on {len(relevant_chunks)} relevant chunk(s) "
            f"from the source URL"
        )
    else:
        st.warning(
            "⚠️ No relevant context found. The answer may not be grounded in the source."
        )

# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        # Display error to user
        st.error("❌ **Application Error**")
        st.error(f"An error occurred: {str(e)}")
        st.exception(e)
        
        # Helpful debugging info
        st.markdown("---")
        st.markdown("### 🔧 Troubleshooting")
        st.markdown("""
        1. **Check API Key**: Ensure `GEMINI_API_KEY` is set in Streamlit Cloud secrets
        2. **Check Files**: Verify all files in `utils/` folder exist
        3. **Check Logs**: View detailed logs in Streamlit Cloud dashboard
        """)
        
        # Re-raise to show in logs
        raise
