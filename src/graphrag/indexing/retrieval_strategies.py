"""Retrieval strategies combining dense, sparse and graph signals."""
from __future__ import annotations

from typing import List, Tuple, Dict, Optional
import logging

from src.graphrag.core.vector_search import VectorSearch

logger = logging.getLogger(__name__)


class HybridRetriever:
    """Combine dense (embeddings) and sparse (BM25) retrieval.

    This implementation focuses on dense retrieval using the existing
    VectorSearch class and uses simple term matching as a sparse fallback.
    """

    def __init__(self, vector_search: Optional[VectorSearch] = None):
        self.vs = vector_search or VectorSearch()

    def retrieve_hybrid(self, query: str, top_k: int = 10, alpha: float = 0.5) -> List[Tuple[str, float, Dict]]:
        # Dense results
        dense = []
        try:
            dense = self.vs.search(query, top_k=top_k)
        except (ValueError, AttributeError, RuntimeError, ConnectionError) as e:
            logger.error("Dense retrieval failed: %s. Returning empty list", str(e), exc_info=True)

        # Sparse: naive substring matches
        sparse_hits = []
        try:
            for i, txt in enumerate(self.vs.chunks):
                if query.lower() in txt.lower():
                    # score heuristic: proportional to occurrences
                    score = txt.lower().count(query.lower())
                    sparse_hits.append((txt, float(score), self.vs.documents[i] if i < len(self.vs.documents) else {}))
            sparse_hits.sort(key=lambda x: x[1], reverse=True)
        except (AttributeError, TypeError, IndexError) as e:
            logger.warning("Sparse retrieval failed: %s. Returning empty list", str(e))
            sparse_hits = []

        # Combine and rerank by weighted sum
        combined = {}
        for i, (txt, score, meta) in enumerate(dense):
            combined[txt] = combined.get(txt, 0.0) + alpha * (1.0 / (1.0 + score))
        for txt, score, meta in sparse_hits:
            combined[txt] = combined.get(txt, 0.0) + (1 - alpha) * score

        # Return top_k by combined score
        results = sorted(combined.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return [(txt, sc, {}) for txt, sc in results]


class GraphAwareRetriever:
    def __init__(self, vector_search: Optional[VectorSearch] = None, graph=None):
        self.vs = vector_search or VectorSearch()
        self.graph = graph

    def retrieve_with_graph_context(self, query: str, entity_types: List[str] = None, relationship_types: List[str] = None, max_depth: int = 2) -> Dict:
        """Start with vector search and expand via graph relationships.

        Returns: {'chunks': [...], 'graph_context': {...}}
        """
        chunks = []
        try:
            chunks = self.vs.search(query, top_k=10)
        except (ValueError, AttributeError, RuntimeError, ConnectionError) as e:
            logger.error("Vector search failed in GraphAwareRetriever: %s", str(e), exc_info=True)

        graph_context = []
        if self.graph:
            # try to expand using entity IDs present in chunk metadata
            for _, _, meta in chunks:
                pid = meta.get('paper_id')
                if pid:
                    try:
                        ctx = self.graph.export_subgraph(pid, max_depth)
                        graph_context.append({'paper_id': pid, 'subgraph': ctx})
                    except (AttributeError, KeyError, ValueError, RuntimeError) as e:
                        logger.debug("No graph context for paper %s: %s", pid, str(e))

        return {'chunks': chunks, 'graph_context': graph_context}


class CitationAwareRetriever:
    def __init__(self, vector_search: Optional[VectorSearch] = None, graph=None):
        self.vs = vector_search or VectorSearch()
        self.graph = graph

    def retrieve_with_citations(self, query: str, include_citing: bool = True, include_cited: bool = True, max_citation_depth: int = 2) -> List[Dict]:
        results = []
        chunks = []
        try:
            chunks = self.vs.search(query, top_k=10)
        except (ValueError, AttributeError, RuntimeError, ConnectionError) as e:
            logger.error("Vector search failed in CitationAwareRetriever: %s", str(e), exc_info=True)

        if not self.graph:
            return [{'chunk': c, 'citations': []} for c in chunks]

        for txt, score, meta in chunks:
            pid = meta.get('paper_id')
            citations = []
            if pid:
                # use graph to fetch cited/citing papers
                try:
                    if include_cited:
                        cited = self.graph.get_citation_network(pid, depth=max_citation_depth)
                        citations.append({'cited': cited})
                    if include_citing:
                        citing = self.graph.get_citation_network(pid, depth=max_citation_depth)
                        citations.append({'citing': citing})
                except (AttributeError, KeyError, ValueError, RuntimeError) as e:
                    logger.error("Failed to fetch citation network for paper %s: %s", pid, str(e), exc_info=True)

            results.append({'chunk': txt, 'score': score, 'citations': citations})

        return results


class TemporalRetriever:
    def __init__(self, vector_search: Optional[VectorSearch] = None):
        self.vs = vector_search or VectorSearch()

    def retrieve_temporal(self, query: str, time_range: Optional[tuple] = None, recency_boost: float = 0.2):
        # Basic vector search + recency boost (metadata must contain 'year')
        hits = []
        try:
            hits = self.vs.search(query, top_k=50)
        except (ValueError, AttributeError, RuntimeError, ConnectionError) as e:
            logger.error("Vector search failed in TemporalRetriever: %s", str(e), exc_info=True)

        ranked = []
        for txt, score, meta in hits:
            year = meta.get('year') if isinstance(meta, dict) else None
            boost = 0.0
            if year and time_range:
                try:
                    start, end = time_range
                    if start <= int(year) <= end:
                        boost = recency_boost
                except (ValueError, TypeError) as e:
                    logger.debug("Failed to parse year %s with time_range %s: %s", year, time_range, str(e))
            ranked.append((txt, score - boost))

        ranked.sort(key=lambda x: x[1])
        return ranked[:20]


class MultiHopRetriever:
    def __init__(self, vector_search: Optional[VectorSearch] = None, graph=None):
        self.vs = vector_search or VectorSearch()
        self.graph = graph

    def retrieve_multi_hop(self, query: str, num_hops: int = 2):
        # Very simple multi-hop: perform vector search for initial seeds, then expand via citation graph
        seeds = []
        try:
            seeds = self.vs.search(query, top_k=10)
        except (ValueError, AttributeError, RuntimeError, ConnectionError) as e:
            logger.error("Vector search failed in MultiHopRetriever: %s", str(e), exc_info=True)

        expanded = []
        if self.graph:
            for txt, score, meta in seeds:
                pid = meta.get('paper_id')
                if pid:
                    try:
                        sub = self.graph.get_citation_network(pid, depth=num_hops)
                        expanded.append({'seed': pid, 'subgraph': sub})
                    except (AttributeError, KeyError, ValueError, RuntimeError) as e:
                        logger.debug("Failed to get citation network for paper %s: %s", pid, str(e))

        return {'seeds': seeds, 'expanded': expanded}


class RetrievalStrategies:
    """Container class for all retrieval strategies."""
    
    def __init__(self, vector_search=None, graph=None):
        self.hybrid = HybridRetriever(vector_search)
        self.graph_aware = GraphAwareRetriever(vector_search, graph)
        self.citation_aware = CitationAwareRetriever(vector_search, graph)
        self.temporal = TemporalRetriever(vector_search)
        self.multi_hop = MultiHopRetriever(vector_search, graph)
