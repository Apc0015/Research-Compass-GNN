"""Chunking strategies for academic papers.

Provides multiple chunkers: section-based, semantic, hybrid and key-chunk extraction.
"""
from __future__ import annotations

from typing import List
import re
import textwrap


class AcademicPaperChunker:
    """Chunk academic papers by sections and semantic boundaries."""

    SECTION_HEADERS = re.compile(r"^(abstract|introduction|methods|methodology|results|discussion|conclusion|references)\b", flags=re.I | re.M)

    def chunk_by_section(self, paper_text: str) -> List[str]:
        """Split text into sections by common section headers.

        Returns list of section texts (strings).
        """
        parts = []
        # naive split by headers -- keep header with section
        splits = self.SECTION_HEADERS.split(paper_text)
        if len(splits) <= 1:
            return [paper_text]

        # splits structure: [pre, header1, body1, header2, body2, ...]
        i = 0
        if splits[0].strip():
            parts.append(splits[0].strip())
        i = 1
        while i + 1 < len(splits):
            header = splits[i].strip()
            body = splits[i + 1].strip()
            parts.append(f"{header}\n{body}")
            i += 2

        return parts

    def chunk_semantic(self, paper_text: str, chunk_size: int = 512) -> List[str]:
        """Split text semantically into approximate chunk_size-word pieces.

        Uses paragraph boundaries where possible.
        """
        paras = [p.strip() for p in paper_text.split('\n\n') if p.strip()]
        chunks = []
        cur = []
        cur_len = 0
        for p in paras:
            words = p.split()
            if cur_len + len(words) <= chunk_size:
                cur.append(p)
                cur_len += len(words)
            else:
                if cur:
                    chunks.append('\n\n'.join(cur))
                # if paragraph itself is huge, break it
                if len(words) > chunk_size:
                    for i in range(0, len(words), chunk_size):
                        chunks.append(' '.join(words[i:i+chunk_size]))
                    cur = []
                    cur_len = 0
                else:
                    cur = [p]
                    cur_len = len(words)

        if cur:
            chunks.append('\n\n'.join(cur))

        return chunks

    def chunk_hybrid(self, paper_text: str) -> List[str]:
        """Combine section-level chunks and semantic chunks, prioritizing sections."""
        sections = self.chunk_by_section(paper_text)
        chunks = []
        for s in sections:
            # for short sections, keep as-is; otherwise semantic split
            if len(s.split()) < 400:
                chunks.append(s)
            else:
                chunks.extend(self.chunk_semantic(s, chunk_size=400))
        return chunks

    def extract_key_chunks(self, paper_text: str, num_chunks: int = 5) -> List[str]:
        """Return the top N most information-dense chunks (heuristic).

        Currently uses paragraph length as a proxy for information density.
        """
        paras = [p.strip() for p in paper_text.split('\n\n') if p.strip()]
        paras.sort(key=lambda p: len(p.split()), reverse=True)
        return paras[:num_chunks]
