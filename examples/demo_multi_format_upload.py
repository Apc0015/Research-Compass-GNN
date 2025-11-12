#!/usr/bin/env python3
"""
Demo: Multi-Format Upload and URL Download

This script demonstrates the new multi-format upload capabilities:
- Upload PDF, DOCX, TXT, HTML, XML files
- Download papers from arXiv URLs
- Extract and process TAR/ZIP archives
- Build citation networks from mixed sources
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import directly from file to avoid PyTorch Geometric dependency
import importlib.util
spec = importlib.util.spec_from_file_location(
    'multi_format_processor',
    Path(__file__).parent.parent / 'data' / 'multi_format_processor.py'
)
mfp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mfp)

# Use functions from the module
extract_text_from_txt = mfp.extract_text_from_txt
extract_text_from_html = mfp.extract_text_from_html
download_from_url = mfp.download_from_url
is_arxiv_url = mfp.is_arxiv_url
get_supported_extensions = mfp.get_supported_extensions


def demo_text_extraction():
    """Demo: Extract text from different formats"""
    print("=" * 60)
    print("DEMO 1: Text Extraction from Different Formats")
    print("=" * 60)
    
    # Plain text
    txt_content = b"This is a plain text research paper abstract."
    text = extract_text_from_txt(txt_content)
    print(f"✅ TXT: {text[:50]}...")
    
    # HTML
    html_content = """
    <html>
        <body>
            <h1>Graph Neural Networks: A Review</h1>
            <p>Graph Neural Networks have become increasingly popular...</p>
        </body>
    </html>
    """
    text = extract_text_from_html(html_content)
    print(f"✅ HTML: {text[:50]}...")
    
    print()


def demo_url_detection():
    """Demo: Detect and parse different URL types"""
    print("=" * 60)
    print("DEMO 2: URL Detection and Parsing")
    print("=" * 60)
    
    test_urls = [
        "https://arxiv.org/abs/1706.03762",
        "https://arxiv.org/pdf/2010.11929.pdf",
        "1706.03762",
        "https://doi.org/10.1145/3292500.3330989",
        "10.1145/3292500.3330989"
    ]
    
    for url in test_urls:
        arxiv_id = is_arxiv_url(url)
        if arxiv_id:
            print(f"✅ arXiv detected: {url} → ID: {arxiv_id}")
        else:
            print(f"ℹ️  Other URL: {url}")
    
    print()


def demo_supported_formats():
    """Demo: Show all supported file formats"""
    print("=" * 60)
    print("DEMO 3: Supported File Formats")
    print("=" * 60)
    
    exts = get_supported_extensions()
    print(f"Total formats supported: {len(exts)}")
    print(f"Extensions: {', '.join(exts)}")
    
    print()


def demo_arxiv_download():
    """Demo: Download paper from arXiv (optional - requires internet)"""
    print("=" * 60)
    print("DEMO 4: Download from arXiv (Optional)")
    print("=" * 60)
    
    print("To download a paper from arXiv:")
    print("  from data.multi_format_processor import download_from_url")
    print("  content, metadata, filename = download_from_url('https://arxiv.org/abs/1706.03762')")
    print("  print(f'Downloaded: {filename}')")
    print("  print(f'Metadata: {metadata}')")
    
    # Uncomment below to actually download (requires internet)
    # try:
    #     content, metadata, filename = download_from_url('https://arxiv.org/abs/1706.03762')
    #     print(f"✅ Downloaded: {filename} ({len(content)} bytes)")
    #     print(f"✅ Metadata: {metadata}")
    # except Exception as e:
    #     print(f"⚠️  Download skipped or failed: {e}")
    
    print()


def demo_usage_example():
    """Demo: Show complete usage example"""
    print("=" * 60)
    print("DEMO 5: Complete Usage Example")
    print("=" * 60)
    
    example_code = """
# Example: Upload mixed formats and URLs to Gradio UI

# 1. Launch the UI
python scripts/launcher.py

# 2. In the "Real Data Training" tab:
#    - Upload files: paper1.pdf, notes.txt, archive.tar.gz
#    - Add URLs (one per line):
#        https://arxiv.org/abs/1706.03762
#        https://arxiv.org/abs/2010.11929
#        10.1145/3292500.3330989
#    
# 3. Click "Process Papers & Build Graph"
#
# 4. The system will:
#    - Extract text from all formats
#    - Download papers from URLs
#    - Extract files from archives
#    - Build a citation network
#    - Train GNN models

# Example: Programmatic usage
from data.multi_format_processor import process_multi_format_input

files = ['paper1.pdf', 'paper2.docx', 'archive.zip']
urls = ['https://arxiv.org/abs/1706.03762']

papers_data, status = process_multi_format_input(
    files=files,
    urls=urls,
    extract_citations=True
)

print(f"Processed {len(papers_data)} papers")
for paper in papers_data:
    print(f"  - {paper['name']}: {len(paper['citations'])} citations")
"""
    
    print(example_code)
    print()


def main():
    """Run all demos"""
    print("\n" + "=" * 60)
    print("Multi-Format Upload & URL Download Demo")
    print("Research Compass GNN")
    print("=" * 60 + "\n")
    
    demo_text_extraction()
    demo_url_detection()
    demo_supported_formats()
    demo_arxiv_download()
    demo_usage_example()
    
    print("=" * 60)
    print("✅ All demos completed!")
    print("=" * 60)
    print("\nNext steps:")
    print("  1. Run: python scripts/launcher.py")
    print("  2. Go to 'Real Data Training' tab")
    print("  3. Upload files or add URLs")
    print("  4. Train GNN models on your data!")
    print()


if __name__ == "__main__":
    main()
