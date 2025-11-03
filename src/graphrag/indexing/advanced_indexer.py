"""Advanced indexing utilities integrating LlamaIndex (optional) and FAISS.

This module provides an AdvancedDocumentIndexer that can build multiple
representations for academic papers: dense vector store (FAISS), optional
LlamaIndex indices, and links to Neo4j graph nodes.
"""
from __future__ import annotations

from typing import List, Dict, Optional
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

try:
    # LlamaIndex import is optional
    from llama_index import Document, ServiceContext, VectorStoreIndex
    LLM_AVAILABLE = True
except Exception:
    LLM_AVAILABLE = False

from src.graphrag.core.vector_search import VectorSearch
from src.graphrag.indexing.chunking_strategies import AcademicPaperChunker


class AdvancedDocumentIndexer:
    """Build advanced indices for academic papers.

    Attributes:
        vector_search: VectorSearch instance (FAISS)
        chunker: AcademicPaperChunker instance
    """

    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2", use_llama_index: bool = True):
        self.vector_search = VectorSearch(model_name=embedding_model)
        self.chunker = AcademicPaperChunker()
        self.use_llama_index = use_llama_index and LLM_AVAILABLE
        self.llama_index = None

    def index_academic_paper(self, paper: Dict, full_text: str, persist_dir: Optional[Path] = None) -> Dict:
        """Index a single academic paper.

        Args:
            paper: dict with paper metadata (id, title, abstract, etc.)
            full_text: the full text of the paper
            persist_dir: optional directory to save indices

        Returns:
            metadata about indexed chunks
        """
        # Chunk by sections & semantic chunks
        sections = self.chunker.chunk_by_section(full_text)
        semantic = self.chunker.chunk_semantic(full_text)
        hybrid = self.chunker.chunk_hybrid(full_text)

        # Build a combined set of chunks with metadata
        chunks = []
        meta = []
        for i, s in enumerate(hybrid):
            chunks.append(s)
            meta.append({
                'paper_id': paper.get('id'),
                'section': None,
                'chunk_index': i,
            })

        # Build FAISS index (or add to existing)
        if self.vector_search.index is None:
            self.vector_search.build_index(chunks, metadata=meta)
        else:
            self.vector_search.add_documents(chunks, metadata=meta)

        # Optionally create a LlamaIndex representation
        if self.use_llama_index:
            try:
                documents = [Document(text=txt, extra_info=md) for txt, md in zip(chunks, meta)]
                sc = ServiceContext.from_defaults()
                self.llama_index = VectorStoreIndex.from_documents(documents, service_context=sc)
            except Exception:
                logger.exception("Failed to build llama-index; continuing with FAISS-only")

        # Persist if requested
        if persist_dir:
            try:
                self.vector_search.save_index(Path(persist_dir) / f"index_{paper.get('id')}")
            except Exception:
                logger.exception("Failed to persist index for paper %s", paper.get('id'))

        return {'paper_id': paper.get('id'), 'num_chunks': len(chunks)}

    def index_with_llama_index(self, documents: List[Dict]) -> None:
        """Build a LlamaIndex for a list of document dicts (requires llama-index).

        documents: list of dicts with keys 'text' and optional 'meta'
        """
        if not LLM_AVAILABLE:
            raise RuntimeError("llama-index not available in the environment")

        docs = []
        for d in documents:
            text = d.get('text')
            meta = d.get('meta', {})
            docs.append(Document(text, extra_info=meta))

        sc = ServiceContext.from_defaults()
        self.llama_index = VectorStoreIndex.from_documents(docs, service_context=sc)

    def create_graph_aware_index(self):
        """Combine vector indices with graph-aware signals.

        This is a placeholder to link document chunks with Neo4j entities. In a
        full implementation, we would enrich chunk metadata with entity ids and
        create cross-index mappings.
        """
        # For now, this method is a no-op and serves as a hook for richer implementations
        logger.info("create_graph_aware_index: no-op hook executed")
