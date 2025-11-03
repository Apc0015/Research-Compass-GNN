"""Collaboration network analysis utilities.

Analyze co-authorship, suggest collaborations, detect groups and rising researchers.
"""
from __future__ import annotations

from typing import Dict, List, Optional
import logging

import networkx as nx

logger = logging.getLogger(__name__)


class CollaborationAnalyzer:
    """Analyze author collaboration networks using GraphManager or NetworkX."""

    def __init__(self, graph_manager):
        self.graph = graph_manager

    def analyze_coauthor_network(self, author_id: str) -> Dict:
        """Return direct collaborators, collaboration strength and centrality metrics."""
        result = {
            'author_id': author_id,
            'direct_collaborators': [],
            'collaboration_strength': {},
            'centrality': {}
        }

        try:
            # Try Neo4j with co-authorship pattern
            if getattr(self.graph, '_use_neo4j', False):
                with self.graph.driver.session() as session:
                    q = """
                    MATCH (a:Author {id:$id})<-[:AUTHORED_BY]-(p:Entity)-[:AUTHORED_BY]->(co:Author)
                    RETURN co.id as id, co.name as name, count(p) as papers_together
                    ORDER BY papers_together DESC
                    """
                    res = session.run(q, id=author_id)
                    for r in res:
                        cid = r['id']
                        name = r.get('name')
                        strength = int(r.get('papers_together', 0))
                        result['direct_collaborators'].append({'id': cid, 'name': name, 'papers_together': strength})
                        result['collaboration_strength'][cid] = strength
                    # Centrality: degree in coauthor subgraph
                    return result

        except Exception:
            logger.exception("Neo4j coauthor query failed; falling back to in-memory")

        # In-memory fallback
        try:
            G = getattr(self.graph, '_graph')
            # Build coauthor graph: authors connected if they coauthored a paper
            coG = nx.Graph()
            for nid, props in getattr(self.graph, '_node_props', {}).items():
                if props.get('label') == 'Paper' or props.get('node_type') == 'PAPER':
                    authors = props.get('authors') or []
                    for i in range(len(authors)):
                        for j in range(i + 1, len(authors)):
                            coG.add_edge(authors[i], authors[j])

            if author_id in coG:
                neighbors = list(coG[author_id])
                for n in neighbors:
                    strength = coG.number_of_edges()  # naive placeholder
                    result['direct_collaborators'].append({'id': n, 'papers_together': coG.number_of_edges()})

            # Centrality measures
            deg = nx.degree_centrality(coG)
            bet = nx.betweenness_centrality(coG)

            result['centrality'] = {
                'degree': deg.get(author_id, 0.0),
                'betweenness': bet.get(author_id, 0.0)
            }

        except Exception:
            logger.exception("Failed to compute coauthor network in fallback")

        return result

    def suggest_collaborations(self, author_id: str, top_k: int = 10) -> List[Dict]:
        """Suggest potential collaborators based on second-degree connections and topic overlap."""
        suggestions = []
        try:
            # Use embedding similarity from GraphEmbedder if available (via Neo4j)
            # Fallback: second-degree neighbors in coauthor graph
            if getattr(self.graph, '_use_neo4j', False):
                # simple cypher to find second-degree collaborators
                with self.graph.driver.session() as session:
                    q = """
                    MATCH (a:Author {id:$id})<-[:AUTHORED_BY]-(p:Entity)-[:AUTHORED_BY]->(co:Author)
                    MATCH (co)<-[:AUTHORED_BY]-(p2:Entity)-[:AUTHORED_BY]->(cand:Author)
                    WHERE NOT (a)-[:AUTHORED_BY]-()<-[:AUTHORED_BY]-(cand)
                    RETURN cand.id as id, cand.name as name, count(DISTINCT p2) as score
                    ORDER BY score DESC
                    LIMIT $k
                    """
                    res = session.run(q, id=author_id, k=top_k)
                    for r in res:
                        suggestions.append({'id': r['id'], 'name': r.get('name'), 'score': int(r.get('score', 0))})
                    return suggestions
        except Exception:
            logger.exception("Neo4j suggestion query failed; falling back to in-memory")

        # In-memory fallback
        try:
            coG = nx.Graph()
            for nid, props in getattr(self.graph, '_node_props', {}).items():
                if props.get('label') == 'Paper' or props.get('node_type') == 'PAPER':
                    authors = props.get('authors') or []
                    for i in range(len(authors)):
                        for j in range(i + 1, len(authors)):
                            coG.add_edge(authors[i], authors[j])

            # Second-degree neighbors
            if author_id in coG:
                first = set(coG.neighbors(author_id))
                second = set()
                for f in first:
                    second.update(set(coG.neighbors(f)))
                second -= first
                second.discard(author_id)
                for s in list(second)[:top_k]:
                    suggestions.append({'id': s, 'score': 1.0})

        except Exception:
            logger.exception("Failed to suggest collaborators in-memory")

        return suggestions

    def detect_research_groups(self) -> List[Dict]:
        """Detect tightly connected author clusters (simple community detection)."""
        clusters = []
        try:
            coG = nx.Graph()
            for nid, props in getattr(self.graph, '_node_props', {}).items():
                if props.get('label') == 'Paper' or props.get('node_type') == 'PAPER':
                    authors = props.get('authors') or []
                    for i in range(len(authors)):
                        for j in range(i + 1, len(authors)):
                            coG.add_edge(authors[i], authors[j])

            comps = list(nx.connected_components(coG))
            for i, c in enumerate(comps):
                clusters.append({'id': i, 'members': list(c), 'size': len(c)})
        except Exception:
            logger.exception("Failed to detect research groups")

        return clusters

    def find_rising_researchers(self, topic: str, years: int = 3) -> List[Dict]:
        """Identify authors with rapid publication/citation growth on a topic (naive heuristic)."""
        rising = []
        try:
            if getattr(self.graph, '_use_neo4j', False):
                with self.graph.driver.session() as session:
                    q = """
                    MATCH (a:Author)-[:AUTHORED_BY]<-(p:Entity)
                    WHERE toLower(p.name) CONTAINS toLower($topic) OR toLower(p.title) CONTAINS toLower($topic)
                    WITH a, p
                    ORDER BY p.year DESC
                    RETURN a.id as id, a.name as name, count(p) as recent_pubs
                    LIMIT 50
                    """
                    res = session.run(q, topic=topic)
                    for r in res:
                        rising.append({'id': r['id'], 'name': r.get('name'), 'recent_pubs': int(r.get('recent_pubs', 0))})
        except Exception:
            logger.exception("Neo4j rising researchers query failed; falling back to in-memory")

        return rising
