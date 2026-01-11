"""
RAG Pipeline Module
===================
Handles the core RAG functionality:
- Text chunking
- Embedding generation
- Similarity matching
- Response generation
"""

import math
from typing import List, Tuple, Optional
from google import genai


class RAGPipeline:
    """
    Retrieval-Augmented Generation Pipeline.
    
    This class encapsulates the entire RAG workflow from document processing
    to response generation using Gemini 2.5 Flash.
    """
    
    def __init__(self, api_key: str):
        """
        Initialize RAG pipeline with Gemini API key.
        
        Args:
            api_key: Google Gemini API key
        """
        self.client = genai.Client(api_key=api_key)
        self.embedding_model = "models/text-embedding-004"
        self.llm_model = "gemini-2.0-flash-exp"
        self._chunks = []  # Store chunks for debug purposes
    
    def chunk_text(
        self,
        text: str,
        chunk_size: int = 1000,
        overlap: int = 200
    ) -> List[str]:
        """
        Split text into overlapping chunks for better retrieval.
        
        Overlapping chunks help preserve context at boundaries and improve
        retrieval accuracy for queries that span multiple chunks.
        
        Args:
            text: Input text to chunk
            chunk_size: Maximum characters per chunk
            overlap: Number of characters to overlap between chunks
            
        Returns:
            List of text chunks
        """
        if not text or len(text) < chunk_size:
            return [text] if text else []
        
        chunks = []
        start = 0
        text_length = len(text)
        
        while start < text_length:
            end = min(start + chunk_size, text_length)
            chunk = text[start:end]
            chunks.append(chunk.strip())
            start = end - overlap  # Overlap for context preservation
        
        self._chunks = chunks
        return chunks
    
    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for text chunks using Google GenAI.
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            List of embedding vectors
        """
        embeddings = []
        
        for text in texts:
            try:
                response = self.client.models.embed_content(
                    model=self.embedding_model,
                    content=text
                )
                embeddings.append(response.embedding)
            except Exception as e:
                # Log error but continue with other chunks
                print(f"Warning: Failed to embed chunk: {str(e)}")
                continue
        
        return embeddings
    
    def cosine_similarity(
        self,
        vec1: List[float],
        vec2: List[float]
    ) -> float:
        """
        Calculate cosine similarity between two vectors.
        
        Cosine similarity measures the cosine of the angle between two vectors,
        providing a value between -1 and 1. Higher values indicate greater similarity.
        
        Args:
            vec1: First embedding vector
            vec2: Second embedding vector
            
        Returns:
            Cosine similarity score (0.0 to 1.0)
        """
        if len(vec1) != len(vec2):
            return 0.0
        
        # Dot product
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        
        # Magnitudes
        magnitude1 = math.sqrt(sum(a * a for a in vec1))
        magnitude2 = math.sqrt(sum(a * a for a in vec2))
        
        # Avoid division by zero
        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0
        
        return dot_product / (magnitude1 * magnitude2)
    
    def retrieve_context(
        self,
        query: str,
        document: str,
        top_k: int = 3,
        similarity_threshold: float = 0.3,
        chunk_size: int = 1000,
        overlap: int = 200
    ) -> List[Tuple[str, float]]:
        """
        Retrieve most relevant chunks from document based on query.
        
        This is the core retrieval step that:
        1. Chunks the document
        2. Generates embeddings for chunks and query
        3. Calculates similarity scores
        4. Returns top-k most relevant chunks
        
        Args:
            query: User's question
            document: Source document text
            top_k: Number of top chunks to retrieve
            similarity_threshold: Minimum similarity score to include chunk
            chunk_size: Size of text chunks
            overlap: Overlap between chunks
            
        Returns:
            List of (chunk_text, similarity_score) tuples, sorted by relevance
        """
        # Step 1: Chunk document
        chunks = self.chunk_text(document, chunk_size, overlap)
        
        if not chunks:
            return []
        
        # Step 2: Generate embeddings
        chunk_embeddings = self.get_embeddings(chunks)
        query_embeddings = self.get_embeddings([query])
        
        if not chunk_embeddings or not query_embeddings:
            return []
        
        query_embedding = query_embeddings[0]
        
        # Step 3: Calculate similarities
        similarities = []
        for i, chunk_emb in enumerate(chunk_embeddings):
            similarity = self.cosine_similarity(query_embedding, chunk_emb)
            if similarity >= similarity_threshold:
                similarities.append((chunks[i], similarity))
        
        # Step 4: Sort and return top-k
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]
    
    def generate_response(
        self,
        question: str,
        context_chunks: List[Tuple[str, float]]
    ) -> str:
        """
        Generate response using Gemini 2.5 Flash with retrieved context.
        
        The prompt is designed to enforce strict grounding:
        - Answer only from provided context
        - Explicitly state when information is missing
        - Avoid hallucinations
        
        Args:
            question: User's question
            context_chunks: List of (chunk, similarity) tuples
            
        Returns:
            Generated answer string
        """
        if not context_chunks:
            return (
                "I couldn't find relevant information in the provided URL "
                "to answer your question. Please ensure the URL contains "
                "relevant content or try rephrasing your question."
            )
        
        # Combine context chunks
        context = "\n\n".join([chunk[0] for chunk in context_chunks])
        
        # Construct prompt with strict grounding instructions
        prompt = f"""You are a helpful assistant that answers questions based ONLY on the provided context from a web page.

Context from the web page:
{context}

Question: {question}

Instructions:
1. Answer the question using ONLY the information provided in the context above.
2. If the context does not contain enough information to answer the question, explicitly state: "Based on the provided content, I cannot find sufficient information to answer this question."
3. Do not use any external knowledge or make assumptions beyond what is in the context.
4. Be concise and accurate.
5. If you reference specific information, indicate that it comes from the provided source.

Answer:"""

        try:
            response = self.client.models.generate_content(
                model=self.llm_model,
                contents=prompt
            )
            return response.text
        except Exception as e:
            return f"Error generating response: {str(e)}"
    
    def get_chunk_count(self) -> int:
        """Get the number of chunks from last processing."""
        return len(self._chunks)
