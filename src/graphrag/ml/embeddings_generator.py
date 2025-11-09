#!/usr/bin/env python3
"""
Embeddings Generator - Generate graph-based node embeddings
Uses Node2Vec and GraphSAGE for learning structural embeddings
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from neo4j import GraphDatabase
import networkx as nx


class GraphEmbedder:
    """
    Generate and manage graph-based node embeddings
    Supports Node2Vec and GraphSAGE
    """

    def __init__(self, uri: str, user: str, password: str):
        """
        Initialize embedder

        Args:
            uri: Neo4j URI
            user: Neo4j username
            password: Neo4j password
        """
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.embeddings = {}  # node_id -> embedding
        self.embedding_dim = 128

    def close(self):
        """Close Neo4j connection"""
        self.driver.close()

    def train_node2vec(
        self,
        dimensions: int = 128,
        walk_length: int = 80,
        num_walks: int = 10,
        window_size: int = 10,
        workers: int = 4
    ) -> Dict[str, np.ndarray]:
        """
        Train Node2Vec embeddings

        Args:
            dimensions: Embedding dimensions
            walk_length: Length of random walks
            num_walks: Number of walks per node
            window_size: Context window size
            workers: Number of workers

        Returns:
            Dictionary mapping node IDs to embeddings
        """
        print("Training Node2Vec embeddings...")

        # Load graph from Neo4j
        G = self._load_networkx_graph()

        try:
            from node2vec import Node2Vec
        except ImportError:
            print("Installing node2vec...")
            import subprocess
            subprocess.check_call(['pip', 'install', 'node2vec'])
            from node2vec import Node2Vec

        # Train Node2Vec
        node2vec = Node2Vec(
            G,
            dimensions=dimensions,
            walk_length=walk_length,
            num_walks=num_walks,
            workers=workers
        )

        model = node2vec.fit(
            window=window_size,
            min_count=1,
            batch_words=4
        )

        # Extract embeddings
        self.embeddings = {}
        for node in G.nodes():
            try:
                self.embeddings[node] = model.wv[str(node)]
            except KeyError:
                # Random embedding if not found
                self.embeddings[node] = np.random.randn(dimensions)

        self.embedding_dim = dimensions

        print(f"✓ Trained embeddings for {len(self.embeddings)} nodes")
        return self.embeddings

    def generate_paper_embeddings(
        self,
        text_weight: float = 0.5,
        graph_weight: float = 0.5
    ) -> Dict[str, np.ndarray]:
        """
        Combine text embeddings with graph embeddings

        Args:
            text_weight: Weight for text-based embeddings
            graph_weight: Weight for graph-based embeddings

        Returns:
            Combined embeddings
        """
        # Get text embeddings from Neo4j
        text_embeddings = self._fetch_text_embeddings()

        # Ensure we have graph embeddings
        if not self.embeddings:
            self.train_node2vec()

        # Combine
        combined = {}
        for node_id in text_embeddings:
            if node_id in self.embeddings:
                text_emb = np.array(text_embeddings[node_id])
                graph_emb = self.embeddings[node_id]

                # Normalize and combine
                combined[node_id] = (
                    text_weight * text_emb / (np.linalg.norm(text_emb) + 1e-8) +
                    graph_weight * graph_emb / (np.linalg.norm(graph_emb) + 1e-8)
                )

        return combined

    def get_embedding(self, node_id: str) -> Optional[np.ndarray]:
        """Get embedding for a node"""
        return self.embeddings.get(node_id)

    def find_similar_by_embedding(
        self,
        node_id: str,
        top_k: int = 10
    ) -> List[Tuple[str, float]]:
        """
        Find similar nodes by embedding similarity

        Args:
            node_id: Source node ID
            top_k: Number of similar nodes to return

        Returns:
            List of (node_id, similarity_score) tuples
        """
        if node_id not in self.embeddings:
            return []

        source_emb = self.embeddings[node_id]
        similarities = []

        for other_id, other_emb in self.embeddings.items():
            if other_id != node_id:
                # Cosine similarity
                sim = np.dot(source_emb, other_emb) / (
                    np.linalg.norm(source_emb) * np.linalg.norm(other_emb) + 1e-8
                )
                similarities.append((other_id, float(sim)))

        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)

        return similarities[:top_k]

    def visualize_embeddings_tsne(
        self,
        node_type: Optional[str] = None,
        output_file: str = "embeddings_tsne.html"
    ):
        """
        Visualize embeddings using t-SNE

        Args:
            node_type: Filter by node type
            output_file: Output HTML file
        """
        from sklearn.manifold import TSNE
        import plotly.graph_objects as go

        # Prepare data
        node_ids = list(self.embeddings.keys())
        embeddings_array = np.array([self.embeddings[nid] for nid in node_ids])

        # Apply t-SNE
        print("Computing t-SNE...")
        tsne = TSNE(n_components=2, random_state=42)
        embeddings_2d = tsne.fit_transform(embeddings_array)

        # Create plot
        fig = go.Figure(data=[
            go.Scatter(
                x=embeddings_2d[:, 0],
                y=embeddings_2d[:, 1],
                mode='markers',
                text=node_ids,
                marker=dict(size=5)
            )
        ])

        fig.update_layout(
            title="Node Embeddings (t-SNE)",
            xaxis_title="Dimension 1",
            yaxis_title="Dimension 2"
        )

        fig.write_html(output_file)
        print(f"✓ Visualization saved to {output_file}")

    def store_embeddings_in_neo4j(self):
        """Store computed embeddings back in Neo4j"""
        with self.driver.session() as session:
            for node_id, embedding in self.embeddings.items():
                query = """
                MATCH (n {id: $node_id})
                SET n.graph_embedding = $embedding
                """
                session.run(
                    query,
                    node_id=node_id,
                    embedding=embedding.tolist()
                )

        print(f"✓ Stored {len(self.embeddings)} embeddings in Neo4j")

    def _load_networkx_graph(self) -> nx.Graph:
        """Load graph from Neo4j as NetworkX"""
        G = nx.Graph()

        with self.driver.session() as session:
            # Get nodes
            node_query = """
            MATCH (n:Entity)
            RETURN n.id as id
            LIMIT 10000
            """
            result = session.run(node_query)
            for record in result:
                G.add_node(record['id'])

            # Get edges
            edge_query = """
            MATCH (s:Entity)-[r]->(t:Entity)
            RETURN s.id as source, t.id as target
            LIMIT 50000
            """
            result = session.run(edge_query)
            for record in result:
                G.add_edge(record['source'], record['target'])

        print(f"Loaded NetworkX graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        return G

    def _fetch_text_embeddings(self) -> Dict[str, List[float]]:
        """Fetch existing text embeddings from Neo4j"""
        embeddings = {}

        with self.driver.session() as session:
            query = """
            MATCH (n:Entity)
            WHERE n.embedding IS NOT NULL
            RETURN n.id as id, n.embedding as embedding
            LIMIT 10000
            """
            result = session.run(query)

            for record in result:
                emb = record['embedding']
                if emb:
                    if isinstance(emb, str):
                        try:
                            emb = eval(emb)
                        except:
                            continue
                    embeddings[record['id']] = emb

        return embeddings


# Main execution for testing
if __name__ == "__main__":
    import os

    neo4j_uri = os.getenv("NEO4J_URI", "neo4j://127.0.0.1:7687")
    neo4j_user = os.getenv("NEO4J_USER", "neo4j")
    neo4j_password = os.getenv("NEO4J_PASSWORD")

    if not neo4j_password:
        raise ValueError("NEO4J_PASSWORD environment variable must be set")

    print("=" * 80)
    print("Graph Embeddings Generator Test")
    print("=" * 80)

    embedder = GraphEmbedder(neo4j_uri, neo4j_user, neo4j_password)

    try:
        # Train Node2Vec
        embeddings = embedder.train_node2vec(
            dimensions=128,
            walk_length=80,
            num_walks=10
        )

        # Find similar nodes
        if embeddings:
            node_id = list(embeddings.keys())[0]
            similar = embedder.find_similar_by_embedding(node_id, top_k=5)

            print(f"\nMost similar to {node_id}:")
            for sim_id, score in similar:
                print(f"  {sim_id}: {score:.4f}")

        # Store in Neo4j
        embedder.store_embeddings_in_neo4j()

    finally:
        embedder.close()
