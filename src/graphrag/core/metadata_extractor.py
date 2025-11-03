"""Academic metadata extraction utilities.

Heuristics + spaCy + LLM fallback (if configured externally) are used to
extract title, authors, abstract, year, venue, references, keywords and DOI
from academic PDF text. This implementation focuses on robust fallbacks and
returns confidence scores for each field.
"""
from __future__ import annotations

from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import re
import logging
from pathlib import Path
from datetime import datetime

try:
    import pdfplumber
except Exception:
    pdfplumber = None

try:
    import PyPDF2
except Exception:
    PyPDF2 = None

try:
    import spacy
    _SPACY_AVAILABLE = True
except Exception:
    spacy = None
    _SPACY_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class ExtractionResult:
    data: Dict
    confidence: Dict[str, float]


class AcademicMetadataExtractor:
    """Extract metadata from academic PDFs or raw text strings.

    The extractor provides several helper methods that can be used independently
    in tests (title/author/abstract extraction from text). The main entry
    point is `extract_from_pdf` which returns a dictionary and per-field
    confidence scores.
    """

    TITLE_REGEX = re.compile(r"^(?P<title>[A-Z][^\n]{10,200})\n", re.M)
    DOI_REGEX = re.compile(r"10\.\d{4,9}/[-._;()/:A-Z0-9]+", re.I)
    YEAR_REGEX = re.compile(r"\b(19|20)\d{2}\b")

    def __init__(self, use_spacy: bool = True):
        self.use_spacy = use_spacy and _SPACY_AVAILABLE
        if self.use_spacy:
            try:
                # prefer a medium or large model if available
                self.nlp = spacy.load("en_core_web_sm")
            except Exception:
                try:
                    self.nlp = spacy.load("en_core_web_md")
                except Exception:
                    self.nlp = None
                    self.use_spacy = False

    def _load_pdf_text(self, pdf_path: Path) -> str:
        pdf_path = Path(pdf_path)
        if pdfplumber:
            try:
                text = []
                with pdfplumber.open(str(pdf_path)) as pdf:
                    for page in pdf.pages:
                        p = page.extract_text()
                        if p:
                            text.append(p)
                return "\n".join(text)
            except Exception as e:
                logger.warning("pdfplumber failed, falling back to PyPDF2: %s", e)

        if PyPDF2:
            try:
                with open(pdf_path, "rb") as f:
                    reader = PyPDF2.PdfReader(f)
                    pages = []
                    for p in reader.pages:
                        t = p.extract_text()
                        if t:
                            pages.append(t)
                    return "\n".join(pages)
            except Exception as e:
                logger.exception("PyPDF2 failed to read PDF: %s", e)

        raise ValueError("No PDF parsing library available or failed to read the file")

    def extract_from_pdf(self, pdf_path: str) -> ExtractionResult:
        """Extract metadata from a PDF file path.

        Returns an ExtractionResult containing the structured data and
        confidence scores for each field.
        """
        text = self._load_pdf_text(Path(pdf_path))
        return self.extract_from_text(text)

    def extract_from_text(self, text: str) -> ExtractionResult:
        """Extract metadata from raw text (useful for tests)."""
        result: Dict[str, Optional[object]] = {}
        confidence: Dict[str, float] = {}

        # Title heuristic: first non-empty line with TitleCase or long length
        title = self.extract_title(text)
        result["title"] = title
        confidence["title"] = 0.8 if title else 0.0

        authors = self.extract_authors(text)
        result["authors"] = authors
        confidence["authors"] = 0.75 if authors else 0.0

        abstract = self.extract_abstract(text)
        result["abstract"] = abstract
        confidence["abstract"] = 0.85 if abstract else 0.0

        year = self.extract_year(text)
        result["year"] = year
        confidence["year"] = 0.6 if year else 0.0

        venue = self.extract_venue(text)
        result["venue"] = venue
        confidence["venue"] = 0.5 if venue else 0.0

        references = self.extract_references(text)
        result["references"] = references
        confidence["references"] = 0.6 if references else 0.0

        keywords = self.extract_keywords(text)
        result["keywords"] = keywords
        confidence["keywords"] = 0.4 if keywords else 0.0

        doi = self.extract_doi(text)
        result["doi"] = doi
        confidence["doi"] = 0.9 if doi else 0.0

        result["extracted_at"] = datetime.utcnow().isoformat()

        return ExtractionResult(data=result, confidence=confidence)

    def extract_title(self, text: str) -> Optional[str]:
        # Try common patterns: lines at top, bold/large heuristics aren't available in plain text
        lines = [l.strip() for l in text.splitlines() if l.strip()]
        if not lines:
            return None

        # Heuristic 1: first line longer than 10 characters and not 'Abstract'
        for i, line in enumerate(lines[:10]):
            if len(line) > 10 and len(line.split()) > 2 and not line.lower().startswith('abstract'):
                # Exclude lines that are clearly author lists (contain commas and ' and ')
                if ',' in line and len(line.split()) < 6:
                    continue
                return line

        # fallback regex
        m = self.TITLE_REGEX.search(text)
        if m:
            return m.group('title').strip()

        return None

    def extract_authors(self, text: str) -> List[str]:
        # Look at the lines immediately after title
        lines = [l.strip() for l in text.splitlines() if l.strip()]
        authors = []
        if len(lines) >= 3:
            candidate = lines[1]
            # common patterns: 'A. Author, B. Author and C. Author' or 'Author1 Author2'
            # split by ' and ' or commas
            if ' and ' in candidate.lower() or ',' in candidate:
                parts = re.split(r',| and ', candidate)
                for p in parts:
                    name = p.strip()
                    # simple filter: contains a space and a capitalized word
                    if ' ' in name and any(ch.isalpha() for ch in name):
                        authors.append(name)

        # spaCy NER fallback: PERSON entities in first page
        if not authors and self.use_spacy and self.nlp:
            doc = self.nlp('\n'.join(lines[:60]))
            persons = [ent.text for ent in doc.ents if ent.label_ == 'PERSON']
            # dedupe and return
            seen = set()
            for p in persons:
                if p not in seen and len(p.split()) <= 4:
                    authors.append(p)
                    seen.add(p)

        return authors

    def extract_abstract(self, text: str) -> Optional[str]:
        # Search for the 'Abstract' section
        m = re.search(r"(?i)abstract[:\n\s]+(.{50,2000}?)(?:\n\s*(?:1\.|introduction|introd))", text, re.S)
        if m:
            return m.group(1).strip()

        # fallback: look for the word 'Abstract' and grab following paragraph
        parts = re.split(r"\n\s*abstract\s*\n", text, flags=re.I)
        if len(parts) > 1:
            candidate = parts[1].split('\n\n')[0]
            if len(candidate) > 50:
                return candidate.strip()

        return None

    def extract_year(self, text: str) -> Optional[int]:
        # Look for years in header/footer or early in the document
        m = self.YEAR_REGEX.search(text)
        if m:
            try:
                return int(m.group(0))
            except Exception:
                return None
        return None

    def extract_venue(self, text: str) -> Optional[str]:
        # Heuristic: Search for typical venue patterns: 'Proceedings of', 'In:' or journal names
        m = re.search(r"(?i)(Proceedings of|In:|Journal of|Proceedings|IEEE Transactions on|ACM)\s+([A-Za-z0-9\-\s,:]+)", text)
        if m:
            return (m.group(1) + ' ' + (m.group(2) or '')).strip()
        return None

    def extract_references(self, text: str) -> List[str]:
        # Find reference section and split into lines
        refs = []
        m = re.search(r"(?is)\n\s*references\s*\n(.*)$", text)
        if m:
            ref_text = m.group(1)
            # split on numbered items or line breaks between references
            candidates = re.split(r"\n\s*(?:\[\d+\]|\d+\.|\n)\s*", ref_text)
            for c in candidates:
                s = c.strip()
                if len(s) > 20:
                    refs.append(s)

        return refs

    def extract_keywords(self, text: str) -> List[str]:
        # Look for 'Keywords:' line
        m = re.search(r"(?i)keywords?:\s*(.+)", text)
        if m:
            kws = [k.strip().lower() for k in re.split(r'[;,]', m.group(1)) if k.strip()]
            return kws
        return []

    def extract_doi(self, text: str) -> Optional[str]:
        m = self.DOI_REGEX.search(text)
        if m:
            return m.group(0)
        return None


__all__ = [
    'AcademicMetadataExtractor',
    'ExtractionResult',
]
