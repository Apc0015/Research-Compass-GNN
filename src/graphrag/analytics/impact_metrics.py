"""Impact metrics calculator for papers, authors and venues.

Provides basic implementations of h-index, PageRank, betweenness, and other
metrics derived from the citation graph.
"""
from __future__ import annotations

from typing import Dict, Optional
import logging

import networkx as nx

logger = logging.getLogger(__name__)


class ImpactMetricsCalculator:
    """Compute various impact and centrality metrics."""

    def __init__(self, graph_manager):
        self.graph = graph_manager

    def calculate_h_index(self, author_id: str) -> int:
        """Calculate h-index for an author based on citation counts of their papers."""
        try:
            if getattr(self.graph, '_use_neo4j', False):
                with self.graph.driver.session() as session:
                    q = """
                    MATCH (a:Author {id:$id})<-[:AUTHORED_BY]-(p:Entity)
                    OPTIONAL MATCH (p)<-[r:CITES]-()
                    RETURN p.id as id, count(r) as citations
                    """
                    res = session.run(q, id=author_id)
                    counts = [int(r.get('citations', 0)) for r in res]
            else:
                counts = []
                for nid, props in getattr(self.graph, '_node_props', {}).items():
                    if props.get('node_type') == 'PAPER' and author_id in (props.get('authors') or []):
                        # count inbound CITES edges
                        c = 0
                        for u, v, data in getattr(self.graph, '_graph').in_edges(nid, data=True):
                            if data.get('type') == 'CITES':
                                c += 1
                        counts.append(c)

            counts.sort(reverse=True)
            h = 0
            for i, c in enumerate(counts, 1):
                if c >= i:
                    h = i
                else:
                    break
            return h
        except (AttributeError, KeyError, ValueError, RuntimeError) as e:
            logger.error("Failed to calculate h-index for author %s: %s", author_id, str(e), exc_info=True)
            return 0

    def calculate_paper_pagerank(self, paper_id: str) -> float:
        """Compute PageRank for a paper in the citation network."""
        try:
            if getattr(self.graph, '_use_neo4j', False):
                # Fetch the citation graph into networkx then compute PageRank
                with self.graph.driver.session() as session:
                    res = session.run("MATCH (a:Entity)-[r:CITES]->(b:Entity) RETURN a.id as s, b.id as t")
                    G = nx.DiGraph()
                    for r in res:
                        G.add_edge(r['s'], r['t'])
            else:
                G = getattr(self.graph, '_graph')

            pr = nx.pagerank(G)
            return float(pr.get(paper_id, 0.0))
        except (AttributeError, KeyError, ValueError, RuntimeError, nx.NetworkXError) as e:
            logger.error("Failed to compute PageRank for paper %s: %s", paper_id, str(e), exc_info=True)
            return 0.0

    def calculate_author_betweenness(self, author_id: str) -> float:
        """Betweenness centrality for an author in co-author network."""
        try:
            # Build coauthor graph
            coG = nx.Graph()
            for nid, props in getattr(self.graph, '_node_props', {}).items():
                if props.get('node_type') == 'PAPER' or props.get('label') == 'Paper':
                    authors = props.get('authors') or []
                    for i in range(len(authors)):
                        for j in range(i + 1, len(authors)):
                            coG.add_edge(authors[i], authors[j])

            bet = nx.betweenness_centrality(coG)
            return float(bet.get(author_id, 0.0))
        except (AttributeError, KeyError, ValueError, RuntimeError, nx.NetworkXError) as e:
            logger.error("Failed to compute author betweenness for %s: %s", author_id, str(e), exc_info=True)
            return 0.0

    def calculate_venue_impact_factor(self, venue_id: str) -> float:
        """Simple impact factor: average citations per paper for a venue."""
        try:
            if getattr(self.graph, '_use_neo4j', False):
                with self.graph.driver.session() as session:
                    q = """
                    MATCH (v:Venue {id:$id})<-[:PUBLISHED_IN]-(p:Entity)
                    OPTIONAL MATCH (p)<-[r:CITES]-()
                    RETURN count(p) as papers, count(r) as citations
                    """
                    r = session.run(q, id=venue_id).single()
                    if r and r.get('papers'):
                        return float(r.get('citations', 0)) / float(r.get('papers'))
                    return 0.0
            else:
                # Fallback: compute from node_props
                papers = [p for p in getattr(self.graph, '_node_props', {}).values() if p.get('venue') == venue_id]
                if not papers:
                    return 0.0
                total_cites = 0
                for p in papers:
                    pid = p.get('id')
                    for u, v, data in getattr(self.graph, '_graph').in_edges(pid, data=True):
                        if data.get('type') == 'CITES':
                            total_cites += 1
                return float(total_cites) / float(len(papers))
        except (AttributeError, KeyError, ValueError, ZeroDivisionError, RuntimeError) as e:
            logger.error("Failed to compute venue impact factor for %s: %s", venue_id, str(e), exc_info=True)
            return 0.0


    def calculate_topic_momentum(self, topic_id: str) -> float:
        """Naive momentum: increase in publications year-over-year for a topic."""
        # This requires topic mapping; return placeholder for now
        return 0.0

    def get_comprehensive_metrics(self, paper_id: str) -> Dict:
        """Return a suite of metrics for a paper."""
        return {
            'citation_count': 0,
            'pagerank': self.calculate_paper_pagerank(paper_id),
            'temporal_metrics': {},
            'field_normalized': 0.0
        }
