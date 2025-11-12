"""
Multi-format document processor for Research Compass GNN

Supports:
- Multiple document formats: PDF, TXT, DOCX, HTML, XML
- Web URL downloads: arXiv, DOI, direct links
- Archive extraction: TAR, TAR.GZ, ZIP
- Metadata extraction from various sources
"""

import io
import os
import re
import tarfile
import zipfile
import tempfile
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union
from urllib.parse import urlparse, unquote
import warnings

# Document processing
import PyPDF2
try:
    from docx import Document as DocxDocument
    DOCX_AVAILABLE = True
except ImportError:
    DOCX_AVAILABLE = False
    warnings.warn("python-docx not available. DOCX support disabled.")

# Web fetching
import requests
from bs4 import BeautifulSoup

# Text processing
import html


# ============================================================================
# CONFIGURATION
# ============================================================================

SUPPORTED_FORMATS = {
    'pdf': ['.pdf'],
    'text': ['.txt', '.md', '.text'],
    'docx': ['.docx', '.doc'],
    'html': ['.html', '.htm'],
    'xml': ['.xml'],
    'archive': ['.tar', '.tar.gz', '.tgz', '.zip']
}

ARXIV_PATTERNS = [
    r'arxiv\.org/abs/(\d+\.\d+)',
    r'arxiv\.org/pdf/(\d+\.\d+)',
    r'(\d{4}\.\d{4,5})'  # Direct arXiv ID
]

DOI_PATTERNS = [
    r'doi\.org/(10\.\d+/[^\s]+)',
    r'dx\.doi\.org/(10\.\d+/[^\s]+)',
    r'(10\.\d+/[^\s]+)'  # Direct DOI
]

ALLOWED_DOMAINS = [
    'arxiv.org',
    'doi.org',
    'dx.doi.org',
    'aclanthology.org',
    'openreview.net',
    'proceedings.mlr.press',
    'papers.nips.cc',
    'proceedings.neurips.cc'
]

MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB
MAX_ARCHIVE_FILES = 200
DOWNLOAD_TIMEOUT = 30


# ============================================================================
# TEXT EXTRACTION FUNCTIONS
# ============================================================================

def extract_text_from_pdf(pdf_data: Union[bytes, io.BytesIO]) -> str:
    """
    Extract text from PDF file
    
    Args:
        pdf_data: PDF as bytes or BytesIO object
        
    Returns:
        Extracted text content
    """
    try:
        if isinstance(pdf_data, bytes):
            pdf_data = io.BytesIO(pdf_data)
        
        pdf_reader = PyPDF2.PdfReader(pdf_data)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        return text.strip()
    except Exception as e:
        return f"Error extracting PDF: {str(e)}"


def extract_text_from_txt(txt_data: Union[bytes, str]) -> str:
    """
    Extract text from plain text file
    
    Args:
        txt_data: Text as bytes or string
        
    Returns:
        Extracted text content
    """
    try:
        if isinstance(txt_data, bytes):
            # Try multiple encodings
            for encoding in ['utf-8', 'latin-1', 'cp1252']:
                try:
                    return txt_data.decode(encoding).strip()
                except UnicodeDecodeError:
                    continue
            return txt_data.decode('utf-8', errors='ignore').strip()
        return str(txt_data).strip()
    except Exception as e:
        return f"Error extracting text: {str(e)}"


def extract_text_from_docx(docx_data: Union[bytes, io.BytesIO]) -> str:
    """
    Extract text from DOCX file
    
    Args:
        docx_data: DOCX as bytes or BytesIO object
        
    Returns:
        Extracted text content
    """
    if not DOCX_AVAILABLE:
        return "Error: python-docx not installed. Install with: pip install python-docx"
    
    try:
        if isinstance(docx_data, bytes):
            docx_data = io.BytesIO(docx_data)
        
        doc = DocxDocument(docx_data)
        text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
        return text.strip()
    except Exception as e:
        return f"Error extracting DOCX: {str(e)}"


def extract_text_from_html(html_data: Union[bytes, str]) -> str:
    """
    Extract text from HTML file
    
    Args:
        html_data: HTML as bytes or string
        
    Returns:
        Extracted text content
    """
    try:
        if isinstance(html_data, bytes):
            html_data = html_data.decode('utf-8', errors='ignore')
        
        soup = BeautifulSoup(html_data, 'lxml')
        
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
        
        # Get text
        text = soup.get_text()
        
        # Clean up whitespace
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = '\n'.join(chunk for chunk in chunks if chunk)
        
        return text.strip()
    except Exception as e:
        return f"Error extracting HTML: {str(e)}"


def extract_text_from_xml(xml_data: Union[bytes, str]) -> str:
    """
    Extract text from XML file
    
    Args:
        xml_data: XML as bytes or string
        
    Returns:
        Extracted text content
    """
    try:
        if isinstance(xml_data, bytes):
            xml_data = xml_data.decode('utf-8', errors='ignore')
        
        soup = BeautifulSoup(xml_data, 'xml')
        text = soup.get_text()
        
        # Clean up whitespace
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    except Exception as e:
        return f"Error extracting XML: {str(e)}"


def extract_text_from_file(file_data: Union[bytes, io.BytesIO], filename: str) -> str:
    """
    Auto-detect format and extract text
    
    Args:
        file_data: File data as bytes or BytesIO
        filename: Original filename
        
    Returns:
        Extracted text content
    """
    ext = Path(filename).suffix.lower()
    
    if ext in SUPPORTED_FORMATS['pdf']:
        return extract_text_from_pdf(file_data)
    elif ext in SUPPORTED_FORMATS['text']:
        return extract_text_from_txt(file_data)
    elif ext in SUPPORTED_FORMATS['docx']:
        return extract_text_from_docx(file_data)
    elif ext in SUPPORTED_FORMATS['html']:
        return extract_text_from_html(file_data)
    elif ext in SUPPORTED_FORMATS['xml']:
        return extract_text_from_xml(file_data)
    else:
        return f"Unsupported format: {ext}"


# ============================================================================
# URL DOWNLOAD FUNCTIONS
# ============================================================================

def is_arxiv_url(url: str) -> Optional[str]:
    """
    Check if URL is an arXiv link and extract paper ID
    
    Args:
        url: URL to check
        
    Returns:
        arXiv ID if valid, None otherwise
    """
    for pattern in ARXIV_PATTERNS:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None


def is_doi_url(url: str) -> Optional[str]:
    """
    Check if URL is a DOI link and extract DOI
    
    Args:
        url: URL to check
        
    Returns:
        DOI if valid, None otherwise
    """
    for pattern in DOI_PATTERNS:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None


def is_allowed_domain(url: str) -> bool:
    """
    Check if URL domain is in allowed list
    
    Args:
        url: URL to check
        
    Returns:
        True if domain is allowed
    """
    parsed = urlparse(url)
    domain = parsed.netloc.lower()
    
    # Remove 'www.' prefix
    domain = domain.replace('www.', '')
    
    return any(allowed in domain for allowed in ALLOWED_DOMAINS)


def download_arxiv_paper(arxiv_id: str) -> Tuple[Optional[bytes], Dict]:
    """
    Download paper from arXiv
    
    Args:
        arxiv_id: arXiv paper ID (e.g., '1706.03762')
        
    Returns:
        Tuple of (PDF bytes, metadata dict)
    """
    pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
    metadata = {'source': 'arxiv', 'arxiv_id': arxiv_id}
    
    try:
        # Download PDF
        response = requests.get(pdf_url, timeout=DOWNLOAD_TIMEOUT)
        response.raise_for_status()
        
        # Fetch metadata from abstract page
        try:
            abs_url = f"https://arxiv.org/abs/{arxiv_id}"
            meta_response = requests.get(abs_url, timeout=DOWNLOAD_TIMEOUT)
            meta_response.raise_for_status()
            
            soup = BeautifulSoup(meta_response.text, 'html.parser')
            
            # Extract title
            title_elem = soup.find('h1', class_='title')
            if title_elem:
                metadata['title'] = title_elem.get_text().replace('Title:', '').strip()
            
            # Extract authors
            authors_elem = soup.find('div', class_='authors')
            if authors_elem:
                metadata['authors'] = authors_elem.get_text().replace('Authors:', '').strip()
            
            # Extract abstract
            abstract_elem = soup.find('blockquote', class_='abstract')
            if abstract_elem:
                metadata['abstract'] = abstract_elem.get_text().replace('Abstract:', '').strip()
                
        except Exception as e:
            print(f"Warning: Could not fetch arXiv metadata: {e}")
        
        return response.content, metadata
        
    except Exception as e:
        print(f"Error downloading arXiv paper {arxiv_id}: {e}")
        return None, metadata


def download_from_url(url: str) -> Tuple[Optional[bytes], Dict, str]:
    """
    Download file from URL
    
    Args:
        url: URL to download from
        
    Returns:
        Tuple of (file bytes, metadata dict, filename)
    """
    metadata = {'source': 'url', 'url': url}
    
    # Check if arXiv
    arxiv_id = is_arxiv_url(url)
    if arxiv_id:
        content, meta = download_arxiv_paper(arxiv_id)
        metadata.update(meta)
        filename = f"arxiv_{arxiv_id}.pdf"
        return content, metadata, filename
    
    # Check if allowed domain
    if not is_allowed_domain(url):
        print(f"Warning: Domain not in allowed list: {url}")
        # Continue anyway but log warning
    
    try:
        # Download file
        headers = {
            'User-Agent': 'Mozilla/5.0 (Research Compass GNN Bot)'
        }
        response = requests.get(url, headers=headers, timeout=DOWNLOAD_TIMEOUT)
        response.raise_for_status()
        
        # Extract filename from URL or Content-Disposition
        filename = "downloaded_paper.pdf"
        
        # Try Content-Disposition header
        content_disp = response.headers.get('content-disposition', '')
        if content_disp:
            filename_match = re.findall('filename="?([^"]+)"?', content_disp)
            if filename_match:
                filename = filename_match[0]
        else:
            # Extract from URL
            path = urlparse(url).path
            if path:
                filename = unquote(Path(path).name)
        
        # Detect content type
        content_type = response.headers.get('content-type', '').lower()
        if 'pdf' in content_type and not filename.endswith('.pdf'):
            filename += '.pdf'
        
        metadata['content_type'] = content_type
        
        return response.content, metadata, filename
        
    except Exception as e:
        print(f"Error downloading from URL {url}: {e}")
        return None, metadata, "error.txt"


def download_from_urls(urls: List[str]) -> List[Dict]:
    """
    Download papers from multiple URLs
    
    Args:
        urls: List of URLs to download
        
    Returns:
        List of dicts with 'content', 'metadata', 'filename'
    """
    results = []
    
    for url in urls:
        url = url.strip()
        if not url:
            continue
        
        print(f"📥 Downloading from: {url}")
        content, metadata, filename = download_from_url(url)
        
        if content:
            results.append({
                'content': content,
                'metadata': metadata,
                'filename': filename,
                'size': len(content)
            })
            print(f"   ✅ Downloaded: {filename} ({len(content)} bytes)")
        else:
            print(f"   ❌ Failed to download from: {url}")
    
    return results


# ============================================================================
# ARCHIVE EXTRACTION FUNCTIONS
# ============================================================================

def extract_tar(tar_data: Union[bytes, io.BytesIO], max_files: int = MAX_ARCHIVE_FILES) -> List[Dict]:
    """
    Extract files from TAR archive
    
    Args:
        tar_data: TAR data as bytes or BytesIO
        max_files: Maximum number of files to extract
        
    Returns:
        List of dicts with extracted file info
    """
    try:
        if isinstance(tar_data, bytes):
            tar_data = io.BytesIO(tar_data)
        
        extracted = []
        
        with tarfile.open(fileobj=tar_data, mode='r:*') as tar:
            members = tar.getmembers()
            
            if len(members) > max_files:
                print(f"⚠️  Warning: Archive contains {len(members)} files, limiting to {max_files}")
                members = members[:max_files]
            
            for member in members:
                if member.isfile():
                    # Check file extension
                    ext = Path(member.name).suffix.lower()
                    supported = any(ext in formats for formats in SUPPORTED_FORMATS.values())
                    
                    if supported and ext not in SUPPORTED_FORMATS['archive']:
                        # Extract file
                        file_obj = tar.extractfile(member)
                        if file_obj:
                            content = file_obj.read()
                            
                            if len(content) <= MAX_FILE_SIZE:
                                extracted.append({
                                    'content': content,
                                    'filename': Path(member.name).name,
                                    'path': member.name,
                                    'size': len(content)
                                })
        
        return extracted
        
    except Exception as e:
        print(f"Error extracting TAR archive: {e}")
        return []


def extract_zip(zip_data: Union[bytes, io.BytesIO], max_files: int = MAX_ARCHIVE_FILES) -> List[Dict]:
    """
    Extract files from ZIP archive
    
    Args:
        zip_data: ZIP data as bytes or BytesIO
        max_files: Maximum number of files to extract
        
    Returns:
        List of dicts with extracted file info
    """
    try:
        if isinstance(zip_data, bytes):
            zip_data = io.BytesIO(zip_data)
        
        extracted = []
        
        with zipfile.ZipFile(zip_data, 'r') as zf:
            namelist = zf.namelist()
            
            if len(namelist) > max_files:
                print(f"⚠️  Warning: Archive contains {len(namelist)} files, limiting to {max_files}")
                namelist = namelist[:max_files]
            
            for name in namelist:
                info = zf.getinfo(name)
                
                if not info.is_dir():
                    # Check file extension
                    ext = Path(name).suffix.lower()
                    supported = any(ext in formats for formats in SUPPORTED_FORMATS.values())
                    
                    if supported and ext not in SUPPORTED_FORMATS['archive']:
                        # Extract file
                        content = zf.read(name)
                        
                        if len(content) <= MAX_FILE_SIZE:
                            extracted.append({
                                'content': content,
                                'filename': Path(name).name,
                                'path': name,
                                'size': len(content)
                            })
        
        return extracted
        
    except Exception as e:
        print(f"Error extracting ZIP archive: {e}")
        return []


def extract_archive(archive_data: Union[bytes, io.BytesIO], filename: str, 
                   max_files: int = MAX_ARCHIVE_FILES) -> List[Dict]:
    """
    Auto-detect and extract archive
    
    Args:
        archive_data: Archive data as bytes or BytesIO
        filename: Archive filename
        max_files: Maximum number of files to extract
        
    Returns:
        List of dicts with extracted file info
    """
    ext = Path(filename).suffix.lower()
    
    if filename.endswith('.tar.gz') or filename.endswith('.tgz'):
        return extract_tar(archive_data, max_files)
    elif ext == '.tar':
        return extract_tar(archive_data, max_files)
    elif ext == '.zip':
        return extract_zip(archive_data, max_files)
    else:
        print(f"Unsupported archive format: {ext}")
        return []


# ============================================================================
# MAIN PROCESSING FUNCTION
# ============================================================================

def process_multi_format_input(
    files: Optional[List] = None,
    urls: Optional[List[str]] = None,
    extract_citations: bool = True,
    max_archive_files: int = MAX_ARCHIVE_FILES
) -> Tuple[List[Dict], str]:
    """
    Process multiple input sources: uploaded files, URLs
    
    Args:
        files: List of uploaded files (Gradio format)
        urls: List of URLs to download
        extract_citations: Whether to extract citations
        max_archive_files: Max files to extract from archives
        
    Returns:
        Tuple of (papers_data list, status message)
    """
    papers_data = []
    status = "🔄 Processing multi-format inputs...\n\n"
    
    # Process uploaded files
    if files:
        status += f"📁 Processing {len(files)} uploaded file(s)...\n"
        
        for idx, file in enumerate(files):
            # Handle different file upload formats
            if hasattr(file, 'name'):
                filename = file.name
                file_data = file.read() if hasattr(file, 'read') else file
            elif isinstance(file, dict):
                filename = file.get('name', f'file_{idx}')
                if 'data' in file:
                    file_data = file['data']
                elif 'path' in file:
                    with open(file['path'], 'rb') as f:
                        file_data = f.read()
                else:
                    file_data = file
            elif isinstance(file, (bytes, bytearray)):
                filename = f'file_{idx}'
                file_data = file
            else:
                filename = f'file_{idx}'
                file_data = file
            
            ext = Path(filename).suffix.lower()
            
            # Check if archive
            if ext in SUPPORTED_FORMATS['archive'] or filename.endswith('.tar.gz') or filename.endswith('.tgz'):
                status += f"📦 Extracting archive: {filename}\n"
                extracted = extract_archive(file_data, filename, max_archive_files)
                status += f"   Found {len(extracted)} files in archive\n"
                
                # Process each extracted file
                for ext_file in extracted:
                    text = extract_text_from_file(ext_file['content'], ext_file['filename'])
                    
                    paper_info = {
                        'name': ext_file['filename'],
                        'text': text[:5000],  # First 5000 chars
                        'citations': [],
                        'metadata': {
                            'source': 'archive',
                            'archive_name': filename,
                            'path': ext_file['path'],
                            'size': ext_file['size']
                        }
                    }
                    
                    # Extract citations if enabled
                    if extract_citations and not text.startswith('Error'):
                        from scripts.launcher import extract_citations_simple
                        citations = extract_citations_simple(text)
                        paper_info['citations'] = citations
                    
                    papers_data.append(paper_info)
                
            else:
                # Regular file
                status += f"📄 Processing: {filename}\n"
                
                text = extract_text_from_file(file_data, filename)
                
                paper_info = {
                    'name': filename,
                    'text': text[:5000],  # First 5000 chars
                    'citations': [],
                    'metadata': {
                        'source': 'upload',
                        'format': ext,
                        'size': len(file_data) if isinstance(file_data, bytes) else 0
                    }
                }
                
                # Extract citations if enabled
                if extract_citations and not text.startswith('Error'):
                    from scripts.launcher import extract_citations_simple
                    citations = extract_citations_simple(text)
                    paper_info['citations'] = citations
                    status += f"   Found {len(citations)} citations\n"
                
                papers_data.append(paper_info)
    
    # Process URLs
    if urls:
        # Parse URLs (handle comma-separated or newline-separated)
        url_list = []
        for url_input in urls:
            if isinstance(url_input, str):
                # Split by newlines or commas
                for url in re.split(r'[,\n]', url_input):
                    url = url.strip()
                    if url:
                        url_list.append(url)
        
        if url_list:
            status += f"\n🌐 Downloading from {len(url_list)} URL(s)...\n"
            downloaded = download_from_urls(url_list)
            
            # Process downloaded files
            for dl_file in downloaded:
                text = extract_text_from_file(dl_file['content'], dl_file['filename'])
                
                paper_info = {
                    'name': dl_file['filename'],
                    'text': text[:5000],
                    'citations': [],
                    'metadata': dl_file['metadata']
                }
                
                # Extract citations if enabled
                if extract_citations and not text.startswith('Error'):
                    from scripts.launcher import extract_citations_simple
                    citations = extract_citations_simple(text)
                    paper_info['citations'] = citations
                
                papers_data.append(paper_info)
    
    status += f"\n✅ Processed {len(papers_data)} papers total!\n"
    
    return papers_data, status


def get_supported_extensions() -> List[str]:
    """Get list of all supported file extensions"""
    extensions = []
    for format_type, exts in SUPPORTED_FORMATS.items():
        extensions.extend(exts)
    return extensions
