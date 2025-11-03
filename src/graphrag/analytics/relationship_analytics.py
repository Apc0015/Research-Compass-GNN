"""Analytics around relationships: citation patterns, collaboration, method adoption.
"""
from __future__ import annotations

from typing import Dict, List
import logging

logger = logging.getLogger(__name__)


class RelationshipAnalytics:
    """Compute simple relationship analytics using GraphManager data."""

    def __init__(self, graph):
        self.graph = graph

    def analyze_citation_patterns(self, paper_id: str) -> Dict:
        try:
            # reuse AcademicGraphManager.get_citation_network if available
            nodes = []
            edges = []
            if hasattr(self.graph, 'get_citation_network'):
                res = self.graph.get_citation_network(paper_id, depth=2)
                nodes = res.get('nodes', [])
                edges = res.get('edges', [])
            else:
                # fallback: basic stats from graph
                stats = self.graph.get_graph_stats()
                nodes = []
                edges = []

            return {
                'paper_id': paper_id,
                'node_count': len(nodes),
                'edge_count': len(edges),
                'citation_velocity': []  # placeholder
            }
        except Exception:
            logger.exception("Failed to analyze citation patterns")
            return {}

    def analyze_collaboration_patterns(self, author_id: str) -> Dict:
        try:
            # simple wrapper around existing coauthor logic
            if hasattr(self.graph, 'get_coauthor_network'):
                coauthors = self.graph.get_coauthor_network(author_id)
                return {'author_id': author_id, 'num_coauthors': len(coauthors), 'coauthors': coauthors}
            else:
                return {'author_id': author_id, 'num_coauthors': 0, 'coauthors': []}
        except Exception:
            logger.exception("Failed to analyze collaboration patterns")
            return {}

    def analyze_method_adoption(self, method_id: str) -> Dict:
        try:
            # count papers that use a method
            count = 0
            papers = []
            if getattr(self.graph, '_use_neo4j', False):
                with self.graph.driver.session() as session:
                    res = session.run("MATCH (p:Paper)-[r:USES_METHOD]->(m:Method {id:$id}) RETURN p.id as id, r.usage_type as usage", id=method_id)
                    for r in res:
                        count += 1
                        papers.append({'id': r['id'], 'usage': r.get('usage')})
            else:
                for pid, props in self.graph._node_props.items():
                    for _, tgt, data in self.graph._graph.out_edges(pid, data=True):
                        if data.get('type') == 'USES_METHOD' and tgt == method_id:
                            count += 1
                            papers.append({'id': pid, 'usage': data.get('usage_type')})

            return {'method_id': method_id, 'count': count, 'papers': papers}
        except Exception:
            logger.exception("Failed to analyze method adoption")
            return {}

    def analyze_topic_evolution(self, topic_id: str) -> Dict:
        try:
            # Placeholder: would compute topic changes over time
            return {'topic_id': topic_id, 'evolution': []}
        except Exception:
            logger.exception("Failed to analyze topic evolution")
            return {}
