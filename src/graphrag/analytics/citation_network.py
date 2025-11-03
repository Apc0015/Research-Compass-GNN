"""Citation network analysis utilities.

Provides analysis functions to compute influence, citation flow, clusters,
and simple forecasting using historical citations.
"""
from __future__ import annotations

from typing import Dict, List, Tuple, Optional
from datetime import datetime
import logging

import networkx as nx
try:
    import community as community_louvain
except Exception:
    community_louvain = None

logger = logging.getLogger(__name__)


class CitationNetworkAnalyzer:
    """Analyze citation networks using Neo4j when available, otherwise NetworkX.

    Methods implement conservative, fast algorithms and return JSON-serializable
    results suitable for UI display.
    """

    def __init__(self, graph_manager):
        """Initialize with an instance of `GraphManager` (or similar)."""
        self.graph = graph_manager

    def analyze_paper_influence(self, paper_id: str) -> Dict:
        """Compute influence metrics for a paper.

        Returns a dict with direct citations, indirect citations count, an h-index
        style estimate and citations over time (per year bins).
        """
        # Build local citation subgraph up to depth 2
        sub = self.get_citation_network(paper_id, depth=2)

        nodes = sub.get('nodes', [])
        edges = sub.get('edges', [])

        G = nx.DiGraph()
        G.add_nodes_from(nodes)
        G.add_edges_from(edges)

        direct = list(G.successors(paper_id)) if paper_id in G else []
        indirect = set()
        for d in direct:
            indirect.update(G.successors(d))

        # h-index style: count citations per citing author/paper
        citation_counts = [len(list(G.predecessors(n))) for n in G.nodes()]
        citation_counts.sort(reverse=True)
        h = 0
        for i, c in enumerate(citation_counts, 1):
            if c >= i:
                h = i
            else:
                break

        # Temporal citation counts - try to fetch years from node props via Neo4j if available
        citations_per_year = {}
        try:
            if getattr(self.graph, '_use_neo4j', False):
                with self.graph.driver.session() as session:
                    result = session.run(
                        "MATCH (c:Entity)-[:CITES]->(p:Entity {id:$id}) RETURN c.year as year, count(c) as cnt GROUP BY c.year",
                        id=paper_id
                    )
                    for r in result:
                        y = r.get('year')
                        if y:
                            citations_per_year[str(y)] = int(r.get('cnt', 0))
        except Exception:
            logger.exception("Failed to fetch temporal citations via Neo4j, falling back to text parsing")

        return {
            'paper_id': paper_id,
            'direct_citations': len(direct),
            'indirect_citations': len(indirect),
            'h_index_like': h,
            'citations_per_year': citations_per_year,
        }

    def get_citation_network(self, paper_id: str, depth: int = 2) -> Dict:
        """Return nodes and edges around a paper up to `depth` hops following CITES edges.

        Edge format: list of (source, target) tuples. Nodes: list of node ids.
        """
        nodes = set()
        edges = []

        # Prefer Neo4j path expansion for accurate results
        if getattr(self.graph, '_use_neo4j', False):
            try:
                with self.graph.driver.session() as session:
                    query = (
                        "MATCH (p:Entity {id:$id})-[r:CITES*1..$depth]-(m:Entity) "
                        "UNWIND relationships((p)-[r]-(m)) as rel "
                        "RETURN DISTINCT startNode(rel).id as source, endNode(rel).id as target"
                    )
                    result = session.run(query, id=paper_id, depth=depth)
                    for rec in result:
                        s = rec.get('source')
                        t = rec.get('target')
                        if s and t:
                            nodes.add(s)
                            nodes.add(t)
                            edges.append((s, t))
                    return {'nodes': list(nodes), 'edges': edges}
            except Exception:
                logger.exception("Neo4j citation query failed; falling back to in-memory")

        # In-memory fallback using GraphManager._graph (networkx)
        try:
            G = getattr(self.graph, '_graph')
            # BFS following edges marked with type 'CITES'
            from collections import deque
            q = deque()
            q.append((paper_id, 0))
            seen = {paper_id}
            while q:
                curr, d = q.popleft()
                nodes.add(curr)
                if d >= depth:
                    continue
                for _, tgt, data in G.out_edges(curr, data=True):
                    if data.get('type') == 'CITES' or data.get('label') == 'CITES':
                        edges.append((curr, tgt))
                        if tgt not in seen:
                            seen.add(tgt)
                            q.append((tgt, d + 1))
            return {'nodes': list(nodes), 'edges': edges}
        except Exception:
            logger.exception("Failed to build citation network in fallback")
            return {'nodes': [], 'edges': []}

    def get_citation_flow(self, paper1_id: str, paper2_id: str) -> Dict:
        """Return paths and basic metrics between two papers."""
        # Use Neo4j shortest path if available
        paths = []
        shortest = None
        try:
            if getattr(self.graph, '_use_neo4j', False):
                with self.graph.driver.session() as session:
                    query = (
                        "MATCH path=shortestPath((p1:Entity {id:$id1})-[*..6]-(p2:Entity {id:$id2})) "
                        "RETURN [n IN nodes(path) | n.id] as nodes"
                    )
                    res = session.run(query, id1=paper1_id, id2=paper2_id)
                    rec = res.single()
                    if rec:
                        shortest = rec.get('nodes')
                        paths.append(shortest)
                        return {'paths': paths, 'shortest': shortest}
        except Exception:
            logger.exception("Neo4j shortestPath failed; falling back to NetworkX")

        # In-memory fallback
        try:
            G = getattr(self.graph, '_graph').to_undirected()
            if paper1_id in G and paper2_id in G:
                all_paths = list(nx.all_shortest_paths(G, source=paper1_id, target=paper2_id))
                paths = all_paths
                shortest = all_paths[0] if all_paths else None
        except Exception:
            logger.exception("Failed to compute paths in-memory")

        return {'paths': paths, 'shortest': shortest}

    def detect_citation_clusters(self) -> List[Dict]:
        """Detect communities in the citation network using Louvain algorithm.

        Returns a list of clusters with member ids.
        """
        # Build a directed graph of citations
        if getattr(self.graph, '_use_neo4j', False):
            try:
                with self.graph.driver.session() as session:
                    result = session.run("MATCH (a:Entity)-[r:CITES]->(b:Entity) RETURN a.id as s, b.id as t")
                    G = nx.Graph()
                    for rec in result:
                        G.add_edge(rec['s'], rec['t'])
            except Exception:
                logger.exception("Neo4j fetch failed; falling back to in-memory")
                G = getattr(self.graph, '_graph').to_undirected()
        else:
            try:
                G = getattr(self.graph, '_graph').to_undirected()
            except Exception:
                logger.exception("No graph available for clustering")
                return []

        if community_louvain is None:
            logger.warning("python-louvain not installed; returning connected components instead")
            comps = list(nx.connected_components(G))
            return [{'id': i, 'members': list(c)} for i, c in enumerate(comps)]

        partition = community_louvain.best_partition(G)
        clusters = {}
        for node, cid in partition.items():
            clusters.setdefault(cid, []).append(node)

        return [{'id': cid, 'members': members} for cid, members in clusters.items()]

    def find_seminal_papers(self, topic: str, min_citations: int = 50) -> List[Dict]:
        """Return top papers for a topic by citation count. Topic matching is naive (name contains topic)."""
        papers = []
        try:
            if getattr(self.graph, '_use_neo4j', False):
                with self.graph.driver.session() as session:
                    query = (
                        "MATCH (p:Entity) WHERE toLower(p.name) CONTAINS toLower($topic) "
                        "OPTIONAL MATCH (p)<-[r:CITES]-() "
                        "WITH p, count(r) as c RETURN p.id as id, p.name as name, c ORDER BY c DESC LIMIT 100"
                    )
                    result = session.run(query, topic=topic)
                    for rec in result:
                        if rec['c'] >= min_citations:
                            papers.append({'id': rec['id'], 'title': rec['name'], 'citations': int(rec['c'])})
        except Exception:
            logger.exception("Failed to query seminal papers")

        return papers

    def get_citation_timeline(self, paper_id: str) -> List[Dict]:
        """Return citations per year timeline (simple implementation)."""
        timeline = []
        try:
            if getattr(self.graph, '_use_neo4j', False):
                with self.graph.driver.session() as session:
                    query = (
                        "MATCH (c:Entity)-[:CITES]->(p:Entity {id:$id}) "
                        "RETURN c.year as year, count(c) as cnt ORDER BY year"
                    )
                    res = session.run(query, id=paper_id)
                    for r in res:
                        y = r.get('year')
                        if y:
                            timeline.append({'year': int(y), 'count': int(r.get('cnt', 0))})
        except Exception:
            logger.exception("Failed to fetch citation timeline")

        return timeline

    def predict_future_citations(self, paper_id: str, months_ahead: int = 12) -> float:
        """Very simple forecast using linear regression on yearly counts (naive)."""
        timeline = self.get_citation_timeline(paper_id)
        if not timeline:
            return 0.0

        # Convert to arrays
        years = [t['year'] for t in timeline]
        counts = [t['count'] for t in timeline]

        # Fit simple linear model y = a*x + b
        try:
            import numpy as np
            X = np.array(years)
            Y = np.array(counts)
            if len(X) < 2:
                return float(counts[-1])  # return last year's count
            A = np.vstack([X, np.ones(len(X))]).T
            m, c = np.linalg.lstsq(A, Y, rcond=None)[0]
            # predict months ahead -> convert to years fraction
            future_year = years[-1] + months_ahead / 12.0
            pred = m * future_year + c
            return float(max(pred, 0.0))
        except Exception:
            logger.exception("Failed to forecast citations")
            return float(counts[-1])
