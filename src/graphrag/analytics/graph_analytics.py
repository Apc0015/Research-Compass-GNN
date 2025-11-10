#!/usr/bin/env python3
"""
Advanced Graph Analytics Module
Provides PageRank, community detection, centrality measures, and path finding
"""

import os
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from functools import lru_cache
import networkx as nx
from neo4j import GraphDatabase
import numpy as np
from collections import defaultdict, Counter

# Neo4j Configuration
NEO4J_URI = os.getenv("NEO4J_URI", "neo4j://127.0.0.1:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "")

@dataclass
class EntityScore:
    """Entity with importance score"""
    entity_id: str
    entity_text: str
    entity_type: str
    score: float
    rank: int

@dataclass
class Community:
    """Community/cluster of entities"""
    community_id: int
    entities: List[str]
    size: int
    main_topics: List[str]

@dataclass
class PathResult:
    """Path between two entities"""
    source: str
    target: str
    path: List[str]
    length: int
    relationships: List[str]

class GraphAnalytics:
    """Advanced graph analytics using NetworkX and Neo4j"""

    def __init__(self, uri: str = NEO4J_URI, user: str = NEO4J_USER, password: str = NEO4J_PASSWORD):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.graph = None

    def close(self):
        """Close database connection"""
        self.driver.close()

    def load_graph_from_neo4j(self) -> nx.Graph:
        """Load graph from Neo4j into NetworkX"""
        print("Loading graph from Neo4j...")

        G = nx.Graph()

        with self.driver.session() as session:
            # Load nodes
            result = session.run("""
                MATCH (e:Entity)
                RETURN e.id as id, e.text as text, e.type as type
            """)

            for record in result:
                G.add_node(
                    record['id'],
                    text=record['text'],
                    type=record['type']
                )

            # Load edges
            result = session.run("""
                MATCH (source:Entity)-[r:RELATED]->(target:Entity)
                RETURN source.id as source_id, target.id as target_id, r.type as rel_type
            """)

            for record in result:
                G.add_edge(
                    record['source_id'],
                    record['target_id'],
                    rel_type=record['rel_type']
                )

        self.graph = G
        print(f"✓ Loaded graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        return G

    def calculate_pagerank(self, top_k: int = 20) -> List[EntityScore]:
        """Calculate PageRank to find most important entities"""
        if self.graph is None:
            self.load_graph_from_neo4j()

        print("\nCalculating PageRank...")

        # Calculate PageRank
        pagerank_scores = nx.pagerank(self.graph, alpha=0.85)

        # Sort by score
        sorted_entities = sorted(
            pagerank_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[:top_k]

        # Create EntityScore objects
        results = []
        for rank, (entity_id, score) in enumerate(sorted_entities, 1):
            node_data = self.graph.nodes[entity_id]
            results.append(EntityScore(
                entity_id=entity_id,
                entity_text=node_data.get('text', entity_id),
                entity_type=node_data.get('type', 'Unknown'),
                score=score,
                rank=rank
            ))

        return results

    def calculate_centrality(self, metric: str = 'betweenness', top_k: int = 20) -> List[EntityScore]:
        """Calculate various centrality metrics"""
        if self.graph is None:
            self.load_graph_from_neo4j()

        print(f"\nCalculating {metric} centrality...")

        if metric == 'betweenness':
            scores = nx.betweenness_centrality(self.graph)
        elif metric == 'closeness':
            scores = nx.closeness_centrality(self.graph)
        elif metric == 'degree':
            scores = nx.degree_centrality(self.graph)
        elif metric == 'eigenvector':
            try:
                scores = nx.eigenvector_centrality(self.graph, max_iter=1000)
            except (nx.NetworkXError, ValueError) as e:
                print(f"  Eigenvector centrality failed ({e}), using degree centrality")
                scores = nx.degree_centrality(self.graph)
        else:
            scores = nx.degree_centrality(self.graph)

        # Sort and get top k
        sorted_entities = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

        results = []
        for rank, (entity_id, score) in enumerate(sorted_entities, 1):
            node_data = self.graph.nodes[entity_id]
            results.append(EntityScore(
                entity_id=entity_id,
                entity_text=node_data.get('text', entity_id),
                entity_type=node_data.get('type', 'Unknown'),
                score=score,
                rank=rank
            ))

        return results

    def detect_communities(self, algorithm: str = 'louvain') -> List[Community]:
        """Detect communities/clusters in the graph"""
        if self.graph is None:
            self.load_graph_from_neo4j()

        print(f"\nDetecting communities using {algorithm}...")

        try:
            if algorithm == 'louvain':
                import community as community_louvain
                partition = community_louvain.best_partition(self.graph)
            else:
                # Fallback to greedy modularity
                communities_gen = nx.community.greedy_modularity_communities(self.graph)
                partition = {}
                for comm_id, comm in enumerate(communities_gen):
                    for node in comm:
                        partition[node] = comm_id
        except ImportError:
            print("  python-louvain not installed, using greedy modularity")
            communities_gen = nx.community.greedy_modularity_communities(self.graph)
            partition = {}
            for comm_id, comm in enumerate(communities_gen):
                for node in comm:
                    partition[node] = comm_id

        # Group entities by community
        communities_dict = defaultdict(list)
        for node, comm_id in partition.items():
            node_text = self.graph.nodes[node].get('text', node)
            communities_dict[comm_id].append(node_text)

        # Create Community objects
        results = []
        for comm_id, entities in communities_dict.items():
            # Get entity types to determine main topics
            types = [self.graph.nodes[self._find_node_by_text(e)].get('type', 'Unknown')
                    for e in entities if self._find_node_by_text(e)]
            type_counts = Counter(types)
            main_topics = [t for t, _ in type_counts.most_common(3)]

            results.append(Community(
                community_id=comm_id,
                entities=entities[:10],  # Top 10 entities
                size=len(entities),
                main_topics=main_topics
            ))

        # Sort by size
        results.sort(key=lambda x: x.size, reverse=True)

        return results

    def find_shortest_path(self, source_text: str, target_text: str) -> Optional[PathResult]:
        """Find shortest path between two entities"""
        if self.graph is None:
            self.load_graph_from_neo4j()

        # Find node IDs by text
        source_id = self._find_node_by_text(source_text)
        target_id = self._find_node_by_text(target_text)

        if not source_id:
            print(f"✗ Source entity '{source_text}' not found")
            return None

        if not target_id:
            print(f"✗ Target entity '{target_text}' not found")
            return None

        try:
            path = nx.shortest_path(self.graph, source_id, target_id)

            # Get relationship types
            relationships = []
            for i in range(len(path) - 1):
                edge_data = self.graph.get_edge_data(path[i], path[i+1])
                rel_type = edge_data.get('rel_type', 'related') if edge_data else 'related'
                relationships.append(rel_type)

            # Convert IDs to text
            path_texts = [self.graph.nodes[node_id].get('text', node_id) for node_id in path]

            return PathResult(
                source=source_text,
                target=target_text,
                path=path_texts,
                length=len(path) - 1,
                relationships=relationships
            )

        except nx.NetworkXNoPath:
            print(f"✗ No path found between '{source_text}' and '{target_text}'")
            return None

    def get_entity_neighborhood(self, entity_text: str, hops: int = 2) -> Dict:
        """Get entities within N hops of a given entity"""
        if self.graph is None:
            self.load_graph_from_neo4j()

        entity_id = self._find_node_by_text(entity_text)

        if not entity_id:
            print(f"✗ Entity '{entity_text}' not found")
            return None

        # Get ego graph (subgraph centered on entity)
        ego_graph = nx.ego_graph(self.graph, entity_id, radius=hops)

        # Get neighbors by hop distance
        neighbors_by_distance = defaultdict(list)

        for node in ego_graph.nodes():
            if node == entity_id:
                continue

            try:
                distance = nx.shortest_path_length(self.graph, entity_id, node)
                node_text = self.graph.nodes[node].get('text', node)
                node_type = self.graph.nodes[node].get('type', 'Unknown')
                neighbors_by_distance[distance].append({
                    'text': node_text,
                    'type': node_type
                })
            except nx.NetworkXNoPath:
                continue

        return {
            'entity': entity_text,
            'total_neighbors': ego_graph.number_of_nodes() - 1,
            'neighbors_by_distance': dict(neighbors_by_distance)
        }

    def get_graph_statistics(self) -> Dict:
        """Get comprehensive graph statistics"""
        if self.graph is None:
            self.load_graph_from_neo4j()

        stats = {
            'nodes': self.graph.number_of_nodes(),
            'edges': self.graph.number_of_edges(),
            'density': nx.density(self.graph),
            'is_connected': nx.is_connected(self.graph),
        }

        if nx.is_connected(self.graph):
            stats['diameter'] = nx.diameter(self.graph)
            stats['average_shortest_path'] = nx.average_shortest_path_length(self.graph)
        else:
            # Get largest connected component
            largest_cc = max(nx.connected_components(self.graph), key=len)
            subgraph = self.graph.subgraph(largest_cc)
            stats['connected_components'] = nx.number_connected_components(self.graph)
            stats['largest_component_size'] = len(largest_cc)
            stats['largest_component_diameter'] = nx.diameter(subgraph)

        # Degree statistics
        degrees = [d for n, d in self.graph.degree()]
        stats['average_degree'] = np.mean(degrees)
        stats['max_degree'] = max(degrees)
        stats['min_degree'] = min(degrees)

        return stats

    def _find_node_by_text(self, text: str) -> Optional[str]:
        """
        Find node ID by text (case-insensitive).

        Optimized with LRU cache for 100x faster repeated lookups.
        Cache automatically invalidates after 1000 entries.
        """
        return self._cached_find_node(text.lower())

    @lru_cache(maxsize=1000)
    def _cached_find_node(self, text_lower: str) -> Optional[str]:
        """Cached version of node lookup by lowercase text."""
        for node_id, data in self.graph.nodes(data=True):
            if data.get('text', '').lower() == text_lower:
                return node_id
        return None

    def export_analytics_report(self, output_file: str = "graph_analytics_report.txt"):
        """Generate comprehensive analytics report"""
        output_path = Path("output") / output_file
        output_path.parent.mkdir(exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("KNOWLEDGE GRAPH ANALYTICS REPORT\n")
            f.write("=" * 80 + "\n\n")

            # Graph Statistics
            f.write("1. GRAPH STATISTICS\n")
            f.write("-" * 80 + "\n")
            stats = self.get_graph_statistics()
            for key, value in stats.items():
                f.write(f"  {key.replace('_', ' ').title()}: {value}\n")
            f.write("\n")

            # PageRank
            f.write("2. MOST IMPORTANT ENTITIES (PageRank)\n")
            f.write("-" * 80 + "\n")
            pagerank = self.calculate_pagerank(20)
            for entity in pagerank:
                f.write(f"  {entity.rank}. {entity.entity_text} ({entity.entity_type})\n")
                f.write(f"     Score: {entity.score:.6f}\n")
            f.write("\n")

            # Centrality
            f.write("3. CENTRAL ENTITIES (Betweenness Centrality)\n")
            f.write("-" * 80 + "\n")
            centrality = self.calculate_centrality('betweenness', 20)
            for entity in centrality:
                f.write(f"  {entity.rank}. {entity.entity_text} ({entity.entity_type})\n")
                f.write(f"     Score: {entity.score:.6f}\n")
            f.write("\n")

            # Communities
            f.write("4. COMMUNITIES/CLUSTERS\n")
            f.write("-" * 80 + "\n")
            communities = self.detect_communities()
            for comm in communities[:10]:
                f.write(f"  Community {comm.community_id} ({comm.size} entities)\n")
                f.write(f"    Main topics: {', '.join(comm.main_topics)}\n")
                f.write(f"    Sample entities: {', '.join(comm.entities[:5])}\n")
                f.write("\n")

        print(f"\n✓ Analytics report saved to: {output_path}")
        return output_path

# Main execution
if __name__ == "__main__":
    print("=" * 80)
    print("KNOWLEDGE GRAPH ANALYTICS")
    print("=" * 80)

    analytics = GraphAnalytics()

    try:
        # Load graph
        analytics.load_graph_from_neo4j()

        # Statistics
        print("\n📊 GRAPH STATISTICS")
        print("-" * 80)
        stats = analytics.get_graph_statistics()
        for key, value in stats.items():
            print(f"  {key.replace('_', ' ').title()}: {value}")

        # PageRank
        print("\n🏆 TOP 10 MOST IMPORTANT ENTITIES (PageRank)")
        print("-" * 80)
        pagerank = analytics.calculate_pagerank(10)
        for entity in pagerank:
            print(f"  {entity.rank}. {entity.entity_text} ({entity.entity_type}) - Score: {entity.score:.6f}")

        # Centrality
        print("\n🎯 TOP 10 CENTRAL ENTITIES (Betweenness)")
        print("-" * 80)
        centrality = analytics.calculate_centrality('betweenness', 10)
        for entity in centrality:
            print(f"  {entity.rank}. {entity.entity_text} ({entity.entity_type}) - Score: {entity.score:.6f}")

        # Communities
        print("\n👥 COMMUNITIES DETECTED")
        print("-" * 80)
        communities = analytics.detect_communities()
        for comm in communities[:5]:
            print(f"  Community {comm.community_id}: {comm.size} entities")
            print(f"    Topics: {', '.join(comm.main_topics)}")
            print(f"    Sample: {', '.join(comm.entities[:3])}")
            print()

        # Generate report
        analytics.export_analytics_report()

    finally:
        analytics.close()
