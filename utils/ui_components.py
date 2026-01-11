"""
UI Components Module
===================
Reusable Streamlit UI components for consistent design.
"""

import streamlit as st


def render_header():
    """Render application header with title and description."""
    st.title("🤖 Gemini 2.5 Flash RAG App")
    st.markdown("### Text-Based GenAI Application with Retrieval-Augmented Generation")
    st.markdown("""
    This application uses **Retrieval-Augmented Generation (RAG)** to provide 
    accurate, context-aware answers grounded in web content.
    """)
    st.markdown("---")


def render_settings() -> dict:
    """
    Render sidebar settings panel.
    
    Returns:
        Dictionary containing settings values
    """
    st.sidebar.header("⚙️ Settings")
    
    top_k = st.sidebar.slider(
        "Top K Chunks",
        min_value=1,
        max_value=10,
        value=3,
        help="Number of most relevant chunks to retrieve"
    )
    
    similarity_threshold = st.sidebar.slider(
        "Similarity Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.3,
        step=0.1,
        help="Minimum similarity score to include a chunk"
    )
    
    debug_mode = st.sidebar.checkbox(
        "Debug Mode",
        value=False,
        help="Show detailed processing information"
    )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📚 About")
    st.sidebar.info("""
    **Day 10 Final Project**
    
    RAG-based GenAI application using:
    - Gemini 2.5 Flash
    - Semantic search
    - Grounded responses
    """)
    
    return {
        'top_k': top_k,
        'threshold': similarity_threshold,
        'debug': debug_mode
    }


def render_inputs() -> tuple:
    """
    Render main input area for URL and question.
    
    Returns:
        Tuple of (url, question) strings
    """
    col1, col2 = st.columns([2, 1])
    
    with col1:
        url = st.text_input(
            "🌐 Enter Source URL",
            placeholder="https://example.com/article",
            help="Enter a valid URL containing text content"
        )
    
    with col2:
        st.write("")  # Spacing
    
    question = st.text_area(
        "❓ Enter Your Question",
        placeholder="What is this article about?",
        help="Ask a question about the content from the URL",
        height=100
    )
    
    return url, question