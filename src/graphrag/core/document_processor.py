"""
Document Processing Module
Handles document loading, parsing, and chunking for GraphRAG.
"""

from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass
import PyPDF2
from docx import Document as DocxDocument
from typing import Union
import logging

logger = logging.getLogger(__name__)

# Optional academic metadata extraction integration
try:
    from .metadata_extractor import AcademicMetadataExtractor, ExtractionResult
except Exception:
    AcademicMetadataExtractor = None
    ExtractionResult = None

# Optional web fetcher integration
try:
    from .web_fetcher import WebContentFetcher
except Exception:
    WebContentFetcher = None


@dataclass
class DocumentChunk:
    """Represents a chunk of text from a document."""
    text: str
    doc_name: str
    chunk_id: int
    metadata: Dict = None


class DocumentProcessor:
    """Processes various document formats and creates text chunks."""

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        """
        Initialize the document processor.

        Args:
            chunk_size: Target size for each text chunk
            chunk_overlap: Number of characters to overlap between chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.supported_extensions = ['.pdf', '.txt', '.md', '.docx', '.doc', '.csv']

    def load_pdf(self, file_path: Path) -> str:
        """Extract text from PDF file."""
        text = ""
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
        except Exception as e:
            raise ValueError(f"Error reading PDF {file_path}: {str(e)}")
        return text

    def load_docx(self, file_path: Path) -> str:
        """Extract text from DOCX file."""
        try:
            doc = DocxDocument(file_path)
            return "\n".join([paragraph.text for paragraph in doc.paragraphs])
        except Exception as e:
            raise ValueError(f"Error reading DOCX {file_path}: {str(e)}")

    def load_text(self, file_path: Path) -> str:
        """Load text from TXT or MD file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read()
        except Exception as e:
            raise ValueError(f"Error reading text file {file_path}: {str(e)}")

    def load_document(self, file_path: Path) -> str:
        """
        Load document based on file extension.

        Args:
            file_path: Path to the document file

        Returns:
            Extracted text content
        """
        file_path = Path(file_path)
        extension = file_path.suffix.lower()

        if extension not in self.supported_extensions:
            raise ValueError(f"Unsupported file type: {extension}")

        if extension == '.pdf':
            return self.load_pdf(file_path)
        elif extension in ['.docx', '.doc']:
            return self.load_docx(file_path)
        elif extension in ['.txt', '.md']:
            return self.load_text(file_path)
        else:
            raise ValueError(f"No handler for extension: {extension}")

    def chunk_text(self, text: str, doc_name: str) -> List[DocumentChunk]:
        """
        Split text into overlapping chunks.

        Args:
            text: Text content to chunk
            doc_name: Name of the source document

        Returns:
            List of DocumentChunk objects
        """
        chunks = []
        start = 0
        chunk_id = 0

        # If text is empty, return a single empty chunk (tests expect this behavior)
        if not text:
            chunks.append(DocumentChunk(
                text="",
                doc_name=doc_name,
                chunk_id=0,
                metadata={'start': 0, 'end': 0}
            ))
            return chunks

        while start < len(text):
            end = start + self.chunk_size
            chunk_text = text[start:end]

            # Create chunk object
            chunks.append(DocumentChunk(
                text=chunk_text,
                doc_name=doc_name,
                chunk_id=chunk_id,
                metadata={'start': start, 'end': min(end, len(text))}
            ))

            chunk_id += 1
            start = end - self.chunk_overlap

        return chunks

    def process_document(self, file_path: Path) -> List[DocumentChunk]:
        """
        Load and chunk a document.

        Args:
            file_path: Path to the document

        Returns:
            List of document chunks
        """
        text = self.load_document(file_path)
        doc_name = file_path.name
        return self.chunk_text(text, doc_name)

    def process_academic_paper(
        self,
        file_path: Union[Path, str],
        academic_graph_manager=None,
        extract_metadata: bool = True
    ) -> Dict:
        """Process an academic PDF end-to-end (light-weight integration).

        Args:
            file_path: Path to PDF
            academic_graph_manager: Optional AcademicGraphManager instance. If
                provided, created Paper/Author nodes will be stored in Neo4j or
                the in-memory fallback.
            extract_metadata: Whether to attempt metadata extraction

        Returns:
            Dictionary containing extracted metadata and created node ids
        """
        file_path = Path(file_path)
        text = self.load_pdf(file_path)
        extractor = None
        metadata_result = None

        if extract_metadata and AcademicMetadataExtractor is not None:
            try:
                extractor = AcademicMetadataExtractor()
                metadata_result = extractor.extract_from_text(text)
            except Exception:
                metadata_result = None

        # Build a minimal PaperNode dict compatible with academic_schema
        paper_info = {
            'title': None,
            'abstract': None,
            'authors': [],
            'year': None,
            'venue': None,
            'keywords': [],
            'doi': None,
        }

        if metadata_result:
            paper_info.update({k: v for k, v in metadata_result.data.items() if k in paper_info})

        created = {}

        # If an academic_graph_manager is provided, create nodes
        if academic_graph_manager is not None:
            from .academic_schema import PaperNode, AuthorNode
            import uuid

            paper_id = uuid.uuid4().hex
            paper_node = PaperNode(
                id=paper_id,
                title=paper_info.get('title') or file_path.stem,
                abstract=paper_info.get('abstract'),
                year=paper_info.get('year'),
                authors=paper_info.get('authors') or [],
                citations=[],
                venue=paper_info.get('venue'),
                keywords=paper_info.get('keywords') or [],
            )

            academic_graph_manager.add_paper(paper_node)
            created['paper_id'] = paper_id

            # Create author nodes and authorship links
            for idx, author_name in enumerate(paper_info.get('authors') or []):
                author_id = uuid.uuid5(uuid.NAMESPACE_DNS, author_name).hex
                author_node = AuthorNode(id=author_id, name=author_name)
                # add_author will MERGE on id
                academic_graph_manager.add_author(author_node)
                academic_graph_manager.create_authorship_link(paper_id, author_id, position=idx)
            created['authors'] = paper_info.get('authors') or []

        return {
            'metadata': metadata_result.data if metadata_result else paper_info,
            'confidence': metadata_result.confidence if metadata_result else {},
            'created': created,
        }
    
    def process_multiple_files(
        self,
        file_paths: List[Union[Path, str]],
        academic_graph_manager=None,
        extract_metadata: bool = True
    ) -> List[Dict]:
        """
        Process multiple files at once.
        
        Args:
            file_paths: List of file paths to process
            academic_graph_manager: Optional graph manager
            extract_metadata: Whether to extract metadata
            
        Returns:
            List of processing results for each file
        """
        results = []
        for file_path in file_paths:
            try:
                result = self.process_academic_paper(
                    file_path,
                    academic_graph_manager=academic_graph_manager,
                    extract_metadata=extract_metadata
                )
                result['file_path'] = str(file_path)
                result['status'] = 'success'
                results.append(result)
                logger.info(f"Successfully processed: {file_path}")
            except Exception as e:
                logger.error(f"Error processing {file_path}: {e}")
                results.append({
                    'file_path': str(file_path),
                    'status': 'error',
                    'error': str(e)
                })
        
        return results
    
    def process_web_url(
        self,
        url: str,
        academic_graph_manager=None
    ) -> Dict:
        """
        Process content from a web URL.
        
        Args:
            url: Web URL to fetch and process
            academic_graph_manager: Optional graph manager
            
        Returns:
            Dictionary with processing results
        """
        if WebContentFetcher is None:
            return {
                'url': url,
                'status': 'error',
                'error': 'Web fetching not available. Install: pip install requests beautifulsoup4'
            }
        
        try:
            fetcher = WebContentFetcher()
            if not fetcher.is_available():
                return {
                    'url': url,
                    'status': 'error',
                    'error': 'Missing dependencies: requests, beautifulsoup4'
                }
            
            # Fetch content
            content = fetcher.fetch_web_content(url)
            if not content:
                return {
                    'url': url,
                    'status': 'error',
                    'error': 'Failed to fetch content from URL'
                }
            
            # Process the content
            created = {}
            if academic_graph_manager is not None and content.get('text'):
                from .academic_schema import PaperNode
                import uuid
                
                paper_id = uuid.uuid4().hex
                paper_node = PaperNode(
                    id=paper_id,
                    title=content.get('title', 'Untitled'),
                    abstract=content.get('abstract', content.get('text', '')[:500]),
                    year=None,
                    authors=content.get('authors', []),
                    citations=[],
                    venue=content.get('source', 'web'),
                    keywords=[],
                )
                
                academic_graph_manager.add_paper(paper_node)
                created['paper_id'] = paper_id
                
                # Add authors if available
                for idx, author_name in enumerate(content.get('authors', [])):
                    from .academic_schema import AuthorNode
                    author_id = uuid.uuid5(uuid.NAMESPACE_DNS, author_name).hex
                    author_node = AuthorNode(id=author_id, name=author_name)
                    academic_graph_manager.add_author(author_node)
                    academic_graph_manager.create_authorship_link(paper_id, author_id, position=idx)
            
            return {
                'url': url,
                'status': 'success',
                'metadata': content,
                'created': created
            }
        
        except Exception as e:
            logger.error(f"Error processing URL {url}: {e}")
            return {
                'url': url,
                'status': 'error',
                'error': str(e)
            }
    
    def process_multiple_urls(
        self,
        urls: List[str],
        academic_graph_manager=None
    ) -> List[Dict]:
        """
        Process multiple web URLs.
        
        Args:
            urls: List of URLs to process
            academic_graph_manager: Optional graph manager
            
        Returns:
            List of processing results
        """
        results = []
        for url in urls:
            result = self.process_web_url(url, academic_graph_manager)
            results.append(result)
        
        return results
