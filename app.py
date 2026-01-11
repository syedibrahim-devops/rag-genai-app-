"""
URL-Based RAG GenAI App
Powered by google.genai · Gemini 2.5 Flash
"""

import streamlit as st
import os
from google import genai
import requests
from bs4 import BeautifulSoup
import math
from typing import List, Tuple

# Page configuration
st.set_page_config(
    page_title="URL-Based RAG GenAI App",
    page_icon="🤖",
    layout="wide"
)

# Title
st.title("URL-Based RAG GenAI App")
st.caption("Powered by google.genai · Gemini 2.5 Flash")

# Initialize Gemini client
@st.cache_resource
def init_client():
    """Initialize Gemini client with API key"""
    api_key = st.secrets.get("GEMINI_API_KEY", os.getenv("GEMINI_API_KEY"))
    if not api_key:
        st.error("⚠️ Gemini API key not found. Please set GEMINI_API_KEY in Streamlit secrets.")
        st.stop()
    return genai.Client(api_key=api_key)

# Load content from URL
def load_url_content(url: str) -> str:
    """Load and extract text from URL"""
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Remove scripts and styles
        for element in soup(["script", "style", "nav", "footer", "header"]):
            element.decompose()
        
        text = soup.get_text()
        # Clean whitespace
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        return ' '.join(chunk for chunk in chunks if chunk)
    except Exception as e:
        st.error(f"Error loading URL: {str(e)}")
        return None

# Chunk text
def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    """Split text into overlapping chunks"""
    if not text or len(text) < chunk_size:
        return [text] if text else []
    
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end].strip())
        start = end - overlap
    return chunks

# Get embeddings
def get_embeddings(client, texts: List[str]) -> List[List[float]]:
    """Generate embeddings"""
    embeddings = []
    for text in texts:
        try:
            response = client.models.embed_content(
                model="models/text-embedding-004",
                content=text
            )
            embeddings.append(response.embedding)
        except Exception as e:
            st.warning(f"Embedding error: {str(e)}")
            continue
    return embeddings

# Cosine similarity
def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """Calculate cosine similarity"""
    if len(vec1) != len(vec2):
        return 0.0
    dot = sum(a * b for a, b in zip(vec1, vec2))
    mag1 = math.sqrt(sum(a * a for a in vec1))
    mag2 = math.sqrt(sum(a * a for a in vec2))
    return dot / (mag1 * mag2) if mag1 and mag2 else 0.0

# Retrieve relevant chunks
def retrieve_chunks(client, query: str, document: str, top_k: int = 3) -> List[Tuple[str, float]]:
    """Retrieve most relevant chunks"""
    chunks = chunk_text(document)
    if not chunks:
        return []
    
    chunk_embeddings = get_embeddings(client, chunks)
    query_embeddings = get_embeddings(client, [query])
    
    if not chunk_embeddings or not query_embeddings:
        return []
    
    query_emb = query_embeddings[0]
    similarities = []
    
    for i, chunk_emb in enumerate(chunk_embeddings):
        sim = cosine_similarity(query_emb, chunk_emb)
        if sim >= 0.3:  # Threshold
            similarities.append((chunks[i], sim))
    
    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:top_k]

# Generate response
def generate_answer(client, question: str, context_chunks: List[Tuple[str, float]]) -> str:
    """Generate answer using Gemini"""
    if not context_chunks:
        return "I couldn't find relevant information in the provided URL to answer your question."
    
    context = "\n\n".join([chunk[0] for chunk in context_chunks])
    
    prompt = f"""You are a helpful assistant that answers questions based ONLY on the provided context from a web page.

Context from the web page:
{context}

Question: {question}

Instructions:
1. Answer the question using ONLY the information provided in the context above.
2. If the context does not contain enough information, explicitly state: "Based on the provided content, I cannot find sufficient information to answer this question."
3. Do not use any external knowledge or make assumptions beyond what is in the context.
4. Be concise and accurate.

Answer:"""

    try:
        response = client.models.generate_content(
            model="gemini-2.0-flash-exp",
            contents=prompt
        )
        return response.text
    except Exception as e:
        return f"Error generating response: {str(e)}"

# Main app
def main():
    # Initialize client
    client = init_client()
    
    # Input fields
    url = st.text_input("Enter source URL", placeholder="https://example.com/article")
    question = st.text_area("Enter your question", placeholder="What is this article about?")
    show_debug = st.checkbox("Show debug output", value=False)
    
    # Generate button
    if st.button("Generate Answer", type="primary", use_container_width=True):
        if not url or not question:
            st.error("Please enter both URL and question")
            return
        
        # Load content
        with st.spinner("Loading content from URL..."):
            text_content = load_url_content(url)
        
        if not text_content:
            st.error("Failed to load content. Please check the URL.")
            return
        
        # Retrieve context
        with st.spinner("Processing content..."):
            relevant_chunks = retrieve_chunks(client, question, text_content)
        
        # Debug output
        if show_debug:
            st.subheader("Debug Information")
            st.write(f"Total chunks: {len(chunk_text(text_content))}")
            st.write(f"Relevant chunks: {len(relevant_chunks)}")
            if relevant_chunks:
                for i, (chunk, score) in enumerate(relevant_chunks, 1):
                    with st.expander(f"Chunk {i} (Similarity: {score:.3f})"):
                        st.text(chunk[:500] + "..." if len(chunk) > 500 else chunk)
        
        # Generate answer
        with st.spinner("Generating answer..."):
            answer = generate_answer(client, question, relevant_chunks)
        
        # Display answer
        st.markdown("---")
        st.subheader("Generated Answer")
        st.info(answer)
        
        if relevant_chunks:
            st.caption(f"✅ Answer based on {len(relevant_chunks)} relevant chunk(s)")

if __name__ == "__main__":
    main()
