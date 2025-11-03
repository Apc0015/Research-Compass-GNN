"""Query engine wrapper for LlamaIndex-backed retrieval and reasoning.

Provides a safe wrapper that falls back to vector search when llama-index
is not available.
"""
from __future__ import annotations

from typing import Any, Dict
import logging

logger = logging.getLogger(__name__)

try:
    from llama_index import ServiceContext, VectorStoreIndex
    LLAMA_AVAILABLE = True
except Exception:
    LLAMA_AVAILABLE = False

from src.graphrag.core.vector_search import VectorSearch


class LlamaIndexQueryEngine:
    """High-level query engine that uses LlamaIndex when available.

    If llama-index is not installed, this class provides a thin wrapper around
    VectorSearch to at least run semantic queries.
    """

    def __init__(self, vector_search: VectorSearch = None):
        self.vs = vector_search or VectorSearch()
        self.index = None

    def create_query_engine(self):
        if not LLAMA_AVAILABLE:
            logger.warning("llama-index not available; using vector search fallback")
            return None
        # Build a service context and attach the vector index if available
        try:
            sc = ServiceContext.from_defaults()
            # For now we do not persist an LlamaIndex in this wrapper
            return sc
        except Exception:
            logger.exception("Failed to create LlamaIndex ServiceContext")
            return None

    def query_with_reasoning(self, query: str) -> Dict[str, Any]:
        if LLAMA_AVAILABLE and self.index:
            try:
                resp = self.index.query(query)
                return {'answer': str(resp), 'sources': []}
            except Exception:
                logger.exception("LlamaIndex query failed; falling back to vector search")

        # Fallback: basic vector search
        try:
            hits = self.vs.search(query, top_k=5)
            return {'answer': None, 'sources': hits}
        except Exception:
            logger.exception("Vector search failed in query_with_reasoning")
            return {'answer': None, 'sources': []}

    def query_with_sub_questions(self, query: str) -> Dict[str, Any]:
        # Very lightweight: call query_with_reasoning and return placeholder
        return self.query_with_reasoning(query)

    def query_graph_and_vector(self, query: str) -> Dict[str, Any]:
        # Run vector search and allow callers to enrich with graph queries
        hits = self.vs.search(query, top_k=10)
        return {'vector_hits': hits}
