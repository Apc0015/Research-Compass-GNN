#!/usr/bin/env python3
"""
Web Content Fetcher for Research Compass.

Fetches and processes content from web URLs including:
- Academic papers (arXiv, PubMed, etc.)
- Research blogs
- Documentation
- Articles
"""

import logging
import re
from typing import Optional, Dict, List
from pathlib import Path
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


class WebContentFetcher:
    """
    Fetch and process web content for the knowledge graph.
    
    Supports:
    - HTML extraction
    - PDF download from URLs
    - Academic paper metadata extraction
    - Content cleaning and chunking
    """
    
    def __init__(self, user_agent: str = "Research Compass Bot/1.0"):
        """
        Initialize web content fetcher.
        
        Args:
            user_agent: User agent string for requests
        """
        self.user_agent = user_agent
        self.session = None
        
        # Try to import optional dependencies
        try:
            import requests
            self.requests = requests
            self.session = requests.Session()
            self.session.headers.update({'User-Agent': user_agent})
        except ImportError:
            logger.warning("requests not installed. Install with: pip install requests")
            self.requests = None
        
        try:
            from bs4 import BeautifulSoup
            self.BeautifulSoup = BeautifulSoup
        except ImportError:
            logger.warning("BeautifulSoup not installed. Install with: pip install beautifulsoup4")
            self.BeautifulSoup = None
    
    def is_available(self) -> bool:
        """Check if required dependencies are available."""
        return self.requests is not None and self.BeautifulSoup is not None
    
    def fetch_url(self, url: str, timeout: int = 30) -> Optional[str]:
        """
        Fetch content from URL.
        
        Args:
            url: URL to fetch
            timeout: Request timeout in seconds
            
        Returns:
            Response text or None if failed
        """
        if not self.is_available():
            logger.error("Web fetching dependencies not available")
            return None
        
        try:
            response = self.session.get(url, timeout=timeout)
            response.raise_for_status()
            return response.text
        except Exception as e:
            logger.error(f"Error fetching URL {url}: {e}")
            return None
    
    def extract_text_from_html(self, html: str, url: str = "") -> Dict[str, str]:
        """
        Extract clean text from HTML.
        
        Args:
            html: HTML content
            url: Source URL (for metadata)
            
        Returns:
            Dictionary with title and text
        """
        if not self.BeautifulSoup:
            return {"title": "", "text": html}
        
        try:
            soup = self.BeautifulSoup(html, 'html.parser')
            
            # Remove script, style, nav, footer
            for tag in soup(['script', 'style', 'nav', 'footer', 'header', 'aside']):
                tag.decompose()
            
            # Extract title
            title = ""
            if soup.title:
                title = soup.title.string.strip()
            elif soup.find('h1'):
                title = soup.find('h1').get_text().strip()
            
            # Extract main content
            # Try common content containers first
            main_content = None
            for selector in ['article', 'main', '.content', '#content', '.post', '.paper']:
                main_content = soup.find(selector)
                if main_content:
                    break
            
            if not main_content:
                main_content = soup.body or soup
            
            # Get text
            text = main_content.get_text(separator='\n', strip=True)
            
            # Clean up text
            text = re.sub(r'\n\s*\n+', '\n\n', text)  # Remove excess blank lines
            text = re.sub(r' +', ' ', text)  # Remove excess spaces
            
            return {
                "title": title,
                "text": text,
                "url": url
            }
        
        except Exception as e:
            logger.error(f"Error extracting text from HTML: {e}")
            return {"title": "", "text": "", "url": url}
    
    def detect_content_type(self, url: str) -> str:
        """
        Detect content type from URL.
        
        Args:
            url: URL to check
            
        Returns:
            Content type: 'pdf', 'html', 'arxiv', 'pubmed', etc.
        """
        url_lower = url.lower()
        
        if url_lower.endswith('.pdf'):
            return 'pdf'
        elif 'arxiv.org' in url_lower:
            return 'arxiv'
        elif 'pubmed' in url_lower or 'ncbi.nlm.nih.gov' in url_lower:
            return 'pubmed'
        elif 'scholar.google' in url_lower:
            return 'google_scholar'
        elif 'doi.org' in url_lower:
            return 'doi'
        else:
            return 'html'
    
    def fetch_arxiv_paper(self, url: str) -> Optional[Dict[str, str]]:
        """
        Fetch arXiv paper metadata and content.
        
        Args:
            url: arXiv URL
            
        Returns:
            Dictionary with paper info
        """
        try:
            # Extract arXiv ID
            arxiv_id_match = re.search(r'arxiv\.org/(?:abs|pdf)/(\d+\.\d+)', url)
            if not arxiv_id_match:
                logger.warning(f"Could not extract arXiv ID from {url}")
                return None
            
            arxiv_id = arxiv_id_match.group(1)
            
            # Fetch abstract page
            abs_url = f"https://arxiv.org/abs/{arxiv_id}"
            html = self.fetch_url(abs_url)
            if not html:
                return None
            
            soup = self.BeautifulSoup(html, 'html.parser')
            
            # Extract metadata
            title = soup.find('h1', class_='title')
            title = title.get_text().replace('Title:', '').strip() if title else ""
            
            authors_block = soup.find('div', class_='authors')
            authors = []
            if authors_block:
                authors = [a.get_text().strip() for a in authors_block.find_all('a')]
            
            abstract_block = soup.find('blockquote', class_='abstract')
            abstract = ""
            if abstract_block:
                abstract = abstract_block.get_text().replace('Abstract:', '').strip()
            
            return {
                "title": title,
                "authors": authors,
                "abstract": abstract,
                "text": abstract,  # Use abstract as text (PDF download would require additional handling)
                "url": url,
                "arxiv_id": arxiv_id,
                "source": "arxiv"
            }
        
        except Exception as e:
            logger.error(f"Error fetching arXiv paper {url}: {e}")
            return None
    
    def fetch_web_content(self, url: str) -> Optional[Dict[str, str]]:
        """
        Fetch and extract content from web URL.
        
        Args:
            url: URL to fetch
            
        Returns:
            Dictionary with extracted content
        """
        if not self.is_available():
            logger.error("Web fetching not available - install: pip install requests beautifulsoup4")
            return None
        
        content_type = self.detect_content_type(url)
        logger.info(f"Fetching {content_type} content from: {url}")
        
        # Handle arXiv specially
        if content_type == 'arxiv':
            return self.fetch_arxiv_paper(url)
        
        # Handle PDF
        if content_type == 'pdf':
            # For now, return URL - actual PDF download would be handled by DocumentProcessor
            return {
                "title": Path(urlparse(url).path).stem,
                "text": "",
                "url": url,
                "content_type": "pdf",
                "note": "PDF download required"
            }
        
        # Fetch HTML
        html = self.fetch_url(url)
        if not html:
            return None
        
        # Extract text
        result = self.extract_text_from_html(html, url)
        result['source'] = 'web'
        result['content_type'] = content_type
        
        return result
    
    def fetch_multiple_urls(self, urls: List[str]) -> List[Dict[str, str]]:
        """
        Fetch content from multiple URLs.
        
        Args:
            urls: List of URLs to fetch
            
        Returns:
            List of extracted content dictionaries
        """
        results = []
        for url in urls:
            content = self.fetch_web_content(url)
            if content:
                results.append(content)
        return results


if __name__ == '__main__':
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    fetcher = WebContentFetcher()
    
    if not fetcher.is_available():
        print("Install dependencies: pip install requests beautifulsoup4")
    else:
        # Test with arXiv paper
        content = fetcher.fetch_web_content("https://arxiv.org/abs/1706.03762")
        if content:
            print(f"Title: {content.get('title')}")
            print(f"Authors: {content.get('authors', [])}")
            print(f"Abstract: {content.get('abstract', '')[:200]}...")
