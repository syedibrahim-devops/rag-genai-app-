"""
Content Loader Module
=====================
Handles loading and processing content from URLs.
"""

import requests
from bs4 import BeautifulSoup
from typing import Optional


class ContentLoader:
    """
    Handles loading and extracting text content from web URLs.
    """
    
    def __init__(self, timeout: int = 10):
        """
        Initialize content loader.
        
        Args:
            timeout: Request timeout in seconds
        """
        self.timeout = timeout
        self.headers = {
            'User-Agent': (
                'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                'AppleWebKit/537.36 (KHTML, like Gecko) '
                'Chrome/91.0.4472.124 Safari/537.36'
            )
        }
    
    def load_from_url(self, url: str) -> Optional[str]:
        """
        Load and extract text content from a URL.
        
        Args:
            url: URL to load content from
            
        Returns:
            Extracted text content or None if loading fails
        """
        try:
            # Fetch content
            response = requests.get(
                url,
                headers=self.headers,
                timeout=self.timeout
            )
            response.raise_for_status()
            
            # Parse HTML
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove script and style elements
            for element in soup(["script", "style", "nav", "footer", "header"]):
                element.decompose()
            
            # Extract text
            text = soup.get_text()
            
            # Clean up whitespace
            text = self._clean_text(text)
            
            return text
            
        except requests.exceptions.RequestException as e:
            print(f"Request error: {str(e)}")
            return None
        except Exception as e:
            print(f"Error processing URL: {str(e)}")
            return None
    
    def _clean_text(self, text: str) -> str:
        """
        Clean and normalize text content.
        
        Args:
            text: Raw text to clean
            
        Returns:
            Cleaned text
        """
        # Split into lines and strip
        lines = (line.strip() for line in text.splitlines())
        
        # Split multi-space chunks
        chunks = (
            phrase.strip()
            for line in lines
            for phrase in line.split("  ")
        )
        
        # Join and remove empty chunks
        text = ' '.join(chunk for chunk in chunks if chunk)
        
        return text