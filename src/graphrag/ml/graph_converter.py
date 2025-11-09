#!/usr/bin/env python3
"""
Graph Converter - Convert Neo4j graphs to PyTorch Geometric format
Handles graph export for GNN training and inference
"""

import torch
import numpy as np
from typing import List, Dict, Optional, Tuple
from neo4j import GraphDatabase
import pickle
from pathlib import Path
import logging

from .gnn_utils import safe_parse_embedding, detect_embedding_dimension

logger = logging.getLogger(__name__)


class Neo4jToTorchGeometric:
    """
    Convert Neo4j graph to PyTorch Geometric Data format
    Supports various node and edge types for academic graphs
    """

    def __init__(self, uri: str, user: str, password: str, embedding_dim: Optional[int] = None):
        """
        Initialize converter with Neo4j connection

        Args:
            uri: Neo4j connection URI
            user: Neo4j username
            password: Neo4j password
            embedding_dim: Optional embedding dimension (auto-detected if None)
        """
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.cache_dir = Path("cache/pyg_graphs")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Auto-detect or use provided embedding dimension
        if embedding_dim is None:
            self.embedding_dim = self._detect_embedding_dim()
        else:
            self.embedding_dim = embedding_dim

        logger.info(f"Graph converter initialized with embedding dimension: {self.embedding_dim}")

    def close(self):
        """Close Neo4j connection"""
        self.driver.close()

    def _detect_embedding_dim(self) -> int:
        """
        Auto-detect embedding dimension from first node in database.

        Returns:
            Detected embedding dimension (default: 384)
        """
        try:
            with self.driver.session() as session:
                result = session.run("""
                    MATCH (n)
                    WHERE n.embedding IS NOT NULL
                    RETURN n.embedding LIMIT 1
                """)
                record = result.single()
                if record and record['embedding']:
                    embedding = safe_parse_embedding(record['embedding'], default_dim=384)
                    dim = len(embedding)
                    logger.info(f"Auto-detected embedding dimension: {dim}")
                    return dim
        except Exception as e:
            logger.warning(f"Could not detect embedding dimension: {e}")

        # Default fallback
        logger.info("Using default embedding dimension: 384")
        return 384

    def export_graph_to_pyg(
        self,
        node_types: Optional[List[str]] = None,
        edge_types: Optional[List[str]] = None,
        use_cache: bool = True,
        max_nodes: int = 10000,
        use_batching: bool = False,
        batch_size: int = 1000
    ):
        """
        Export Neo4j graph to PyTorch Geometric Data object

        Args:
            node_types: List of node labels to include (None = all)
            edge_types: List of relationship types to include (None = all)
            use_cache: Whether to use cached version if available
            max_nodes: Maximum number of nodes to load
            use_batching: Use batched loading for large graphs (memory efficient)
            batch_size: Batch size for batched loading

        Returns:
            torch_geometric.data.Data object
        """
        cache_key = f"{'_'.join(node_types or ['all'])}_{'_'.join(edge_types or ['all'])}"
        cache_file = self.cache_dir / f"{cache_key}.pkl"

        if use_cache and cache_file.exists():
            print(f"Loading cached graph from {cache_file}")
            with open(cache_file, 'rb') as f:
                return pickle.load(f)

        print("Exporting graph from Neo4j...")

        if use_batching:
            print(f"Using batched loading (batch_size={batch_size}) for memory efficiency")
            # Load nodes in batches
            all_nodes = []
            for batch in self._fetch_nodes_batched(node_types, batch_size, max_nodes):
                all_nodes.extend(batch)
                print(f"  Loaded {len(all_nodes)} nodes so far...")
            nodes_data = all_nodes
        else:
            # Get nodes with embeddings (all at once)
            nodes_data = self._fetch_nodes(node_types, max_nodes)

        # Get edges
        edges_data = self._fetch_edges(node_types, edge_types)

        # Convert to PyG format
        data = self._convert_to_pyg_format(nodes_data, edges_data)

        # Cache the result
        with open(cache_file, 'wb') as f:
            pickle.dump(data, f)

        print(f"Graph exported: {data.num_nodes} nodes, {data.num_edges} edges")
        return data

    def export_papers_graph(self) -> 'Data':
        """
        Export graph containing only Paper nodes and citations

        Returns:
            PyG Data object with paper citation network
        """
        return self.export_graph_to_pyg(
            node_types=['Entity'],  # Using generic Entity for now
            edge_types=['RELATED', 'CITES']
        )

    def export_citation_network(self, min_citations: int = 1) -> 'Data':
        """
        Export citation network with metadata

        Args:
            min_citations: Minimum citations for a paper to be included

        Returns:
            PyG Data object
        """
        with self.driver.session() as session:
            # Query for papers with minimum citations
            query = """
            MATCH (p:Entity)
            WHERE p.type = 'PAPER' OR p.type = 'PUBLICATION'
            OPTIONAL MATCH (p)<-[r:CITES]-()
            WITH p, count(r) as citation_count
            WHERE citation_count >= $min_citations
            RETURN p.id as id, p.name as name, p.embedding as embedding,
                   citation_count
            ORDER BY citation_count DESC
            """

            result = session.run(query, min_citations=min_citations)
            nodes_data = [dict(record) for record in result]

        # Get citation edges between these papers
        paper_ids = [n['id'] for n in nodes_data]

        with self.driver.session() as session:
            edge_query = """
            MATCH (p1:Entity)-[r:CITES]->(p2:Entity)
            WHERE p1.id IN $paper_ids AND p2.id IN $paper_ids
            RETURN p1.id as source, p2.id as target
            """

            result = session.run(edge_query, paper_ids=paper_ids)
            edges_data = [dict(record) for record in result]

        return self._convert_to_pyg_format(nodes_data, edges_data)

    def export_coauthor_network(self) -> 'Data':
        """
        Export co-authorship network

        Returns:
            PyG Data object with author collaboration network
        """
        return self.export_graph_to_pyg(
            node_types=['AUTHOR', 'Entity'],
            edge_types=['COAUTHOR', 'COLLABORATED_WITH', 'RELATED']
        )

    def _fetch_nodes(self, node_types: Optional[List[str]] = None, max_nodes: int = 10000) -> List[Dict]:
        """
        Fetch nodes from Neo4j with embeddings

        Args:
            node_types: Node labels to fetch
            max_nodes: Maximum number of nodes to fetch

        Returns:
            List of node dictionaries with embeddings
        """
        with self.driver.session() as session:
            if node_types:
                # Fetch specific node types
                label_filter = " OR ".join([f"'{label}' IN labels(n)" for label in node_types])
                query = f"""
                MATCH (n)
                WHERE {label_filter}
                RETURN n.id as id, n.name as name, n.embedding as embedding,
                       labels(n) as labels, properties(n) as properties
                LIMIT {max_nodes}
                """
            else:
                # Fetch all nodes
                query = f"""
                MATCH (n:Entity)
                RETURN n.id as id, n.name as name, n.embedding as embedding,
                       labels(n) as labels, properties(n) as properties
                LIMIT {max_nodes}
                """

            result = session.run(query)
            nodes = []

            for record in result:
                node_dict = dict(record)

                # Handle embedding - SECURITY FIX: Use safe_parse_embedding instead of eval()
                embedding = node_dict.get('embedding')
                embedding = safe_parse_embedding(embedding, default_dim=self.embedding_dim)

                node_dict['embedding'] = embedding
                nodes.append(node_dict)

            return nodes

    def _fetch_nodes_batched(
        self,
        node_types: Optional[List[str]] = None,
        batch_size: int = 1000,
        max_nodes: Optional[int] = None
    ):
        """
        Fetch nodes from Neo4j in batches (generator for memory efficiency)

        Args:
            node_types: Node labels to fetch
            batch_size: Number of nodes per batch
            max_nodes: Maximum total nodes (None = unlimited)

        Yields:
            Batches of node dictionaries
        """
        skip = 0
        total_fetched = 0

        while True:
            # Determine how many to fetch in this batch
            limit = batch_size
            if max_nodes and (total_fetched + batch_size) > max_nodes:
                limit = max_nodes - total_fetched
                if limit <= 0:
                    break

            with self.driver.session() as session:
                if node_types:
                    label_filter = " OR ".join([f"'{label}' IN labels(n)" for label in node_types])
                    query = f"""
                    MATCH (n)
                    WHERE {label_filter}
                    RETURN n.id as id, n.name as name, n.embedding as embedding,
                           labels(n) as labels, properties(n) as properties
                    SKIP {skip}
                    LIMIT {limit}
                    """
                else:
                    query = f"""
                    MATCH (n:Entity)
                    RETURN n.id as id, n.name as name, n.embedding as embedding,
                           labels(n) as labels, properties(n) as properties
                    SKIP {skip}
                    LIMIT {limit}
                    """

                result = session.run(query)
                batch = []

                for record in result:
                    node_dict = dict(record)
                    embedding = node_dict.get('embedding')
                    embedding = safe_parse_embedding(embedding, default_dim=self.embedding_dim)
                    node_dict['embedding'] = embedding
                    batch.append(node_dict)

                if not batch:
                    # No more nodes
                    break

                yield batch

                total_fetched += len(batch)
                skip += batch_size

                if max_nodes and total_fetched >= max_nodes:
                    break

    def _fetch_edges(
        self,
        node_types: Optional[List[str]] = None,
        edge_types: Optional[List[str]] = None
    ) -> List[Dict]:
        """
        Fetch edges from Neo4j

        Args:
            node_types: Filter edges by source/target node types
            edge_types: Relationship types to fetch

        Returns:
            List of edge dictionaries
        """
        with self.driver.session() as session:
            # Build query
            if edge_types:
                rel_filter = "|".join(edge_types)
                query = f"""
                MATCH (s)-[r:{rel_filter}]->(t)
                RETURN s.id as source, t.id as target, type(r) as type
                LIMIT 50000
                """
            else:
                query = """
                MATCH (s:Entity)-[r]->(t:Entity)
                RETURN s.id as source, t.id as target, type(r) as type
                LIMIT 50000
                """

            result = session.run(query)
            edges = [dict(record) for record in result]

            return edges

    def _convert_to_pyg_format(
        self,
        nodes_data: List[Dict],
        edges_data: List[Dict]
    ) -> 'Data':
        """
        Convert node and edge data to PyTorch Geometric format

        Args:
            nodes_data: List of node dictionaries
            edges_data: List of edge dictionaries

        Returns:
            torch_geometric.data.Data object
        """
        try:
            from torch_geometric.data import Data
        except ImportError:
            print("Warning: torch_geometric not installed. Installing...")
            import subprocess
            subprocess.check_call(['pip', 'install', 'torch-geometric'])
            from torch_geometric.data import Data

        # Create node ID mapping
        node_id_map = {node['id']: idx for idx, node in enumerate(nodes_data)}

        # Extract node features (embeddings)
        node_features = []
        for node in nodes_data:
            embedding = node.get('embedding', [])
            # Use safe_parse_embedding for consistent handling
            embedding = safe_parse_embedding(embedding, default_dim=self.embedding_dim)
            node_features.append(embedding)

        # Convert to tensor
        x = torch.tensor(node_features, dtype=torch.float)

        # Create edge index
        edge_index = []
        for edge in edges_data:
            source = edge['source']
            target = edge['target']

            # Check if both nodes exist in our node set
            if source in node_id_map and target in node_id_map:
                source_idx = node_id_map[source]
                target_idx = node_id_map[target]
                edge_index.append([source_idx, target_idx])

        if not edge_index:
            # Create at least one self-loop to avoid empty graph
            edge_index = [[0, 0]]

        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

        # Create Data object
        data = Data(x=x, edge_index=edge_index)

        # Store original node IDs for reference
        data.node_ids = [node['id'] for node in nodes_data]
        data.node_names = [node.get('name', '') for node in nodes_data]

        return data

    def create_train_val_test_split(
        self,
        data: 'Data',
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        min_samples_per_split: int = 10,
        seed: Optional[int] = None
    ) -> 'Data':
        """
        Create train/validation/test split masks with validation

        Args:
            data: PyG Data object
            train_ratio: Proportion for training (0.0-1.0)
            val_ratio: Proportion for validation (0.0-1.0)
            test_ratio: Proportion for testing (0.0-1.0)
            min_samples_per_split: Minimum samples required per split
            seed: Random seed for reproducibility

        Returns:
            Data object with train_mask, val_mask, test_mask

        Raises:
            ValueError: If ratios don't sum to 1.0 or splits are too small
        """
        # Validate ratios
        if not (0 < train_ratio <= 1.0 and 0 <= val_ratio <= 1.0 and 0 <= test_ratio <= 1.0):
            raise ValueError(f"Ratios must be between 0 and 1. Got: train={train_ratio}, val={val_ratio}, test={test_ratio}")

        if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
            raise ValueError(f"Ratios must sum to 1.0. Got: {train_ratio + val_ratio + test_ratio}")

        # Validate minimum samples
        num_nodes = data.num_nodes

        if num_nodes < 3 * min_samples_per_split:
            raise ValueError(
                f"Graph too small for splits. Need at least {3 * min_samples_per_split} nodes "
                f"(3 splits × {min_samples_per_split} min samples), but have {num_nodes}."
            )

        # Calculate split sizes
        train_size = int(num_nodes * train_ratio)
        val_size = int(num_nodes * val_ratio)
        test_size = num_nodes - train_size - val_size  # Ensure all nodes are used

        # Validate each split meets minimum
        if train_size < min_samples_per_split:
            raise ValueError(f"Training split too small: {train_size} < {min_samples_per_split}. Increase train_ratio or reduce minimum.")

        if val_ratio > 0 and val_size < min_samples_per_split:
            raise ValueError(f"Validation split too small: {val_size} < {min_samples_per_split}. Increase val_ratio or reduce minimum.")

        if test_ratio > 0 and test_size < min_samples_per_split:
            raise ValueError(f"Test split too small: {test_size} < {min_samples_per_split}. Increase test_ratio or reduce minimum.")

        # Create random permutation (with optional seed)
        if seed is not None:
            torch.manual_seed(seed)

        indices = torch.randperm(num_nodes)

        # Create masks
        train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        val_mask = torch.zeros(num_nodes, dtype=torch.bool)
        test_mask = torch.zeros(num_nodes, dtype=torch.bool)

        train_mask[indices[:train_size]] = True
        val_mask[indices[train_size:train_size + val_size]] = True
        test_mask[indices[train_size + val_size:]] = True

        data.train_mask = train_mask
        data.val_mask = val_mask
        data.test_mask = test_mask

        # Log split info
        logger.info(f"Created splits: train={train_size}, val={val_size}, test={test_size}")

        return data

    def add_node_labels(
        self,
        data: 'Data',
        label_property: str = 'type'
    ) -> Tuple['Data', Dict[str, int]]:
        """
        Add node labels from Neo4j properties

        Args:
            data: PyG Data object
            label_property: Property name containing labels

        Returns:
            Tuple of (Data with labels, label_to_idx mapping)
        """
        with self.driver.session() as session:
            query = f"""
            MATCH (n:Entity)
            WHERE n.id IN $node_ids
            RETURN n.id as id, n.{label_property} as label
            """

            result = session.run(query, node_ids=data.node_ids)
            labels_dict = {record['id']: record['label'] for record in result}

        # Create label to index mapping
        unique_labels = list(set(labels_dict.values()))
        label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}

        # Create label tensor
        labels = []
        for node_id in data.node_ids:
            label = labels_dict.get(node_id, unique_labels[0])
            labels.append(label_to_idx[label])

        data.y = torch.tensor(labels, dtype=torch.long)

        return data, label_to_idx


# Main execution for testing
if __name__ == "__main__":
    import os

    # Get configuration
    neo4j_uri = os.getenv("NEO4J_URI", "neo4j://127.0.0.1:7687")
    neo4j_user = os.getenv("NEO4J_USER", "neo4j")
    neo4j_password = os.getenv("NEO4J_PASSWORD")

    if not neo4j_password:
        raise ValueError("NEO4J_PASSWORD environment variable must be set")

    print("=" * 80)
    print("Graph Converter Test")
    print("=" * 80)

    converter = Neo4jToTorchGeometric(neo4j_uri, neo4j_user, neo4j_password)

    try:
        # Export full graph
        print("\n1. Exporting full graph...")
        data = converter.export_papers_graph()
        print(f"   Nodes: {data.num_nodes}")
        print(f"   Edges: {data.num_edges}")
        print(f"   Features: {data.x.shape}")

        # Create splits
        print("\n2. Creating train/val/test splits...")
        data = converter.create_train_val_test_split(data)
        print(f"   Train: {data.train_mask.sum().item()}")
        print(f"   Val: {data.val_mask.sum().item()}")
        print(f"   Test: {data.test_mask.sum().item()}")

        # Add labels
        print("\n3. Adding node labels...")
        data, label_map = converter.add_node_labels(data)
        print(f"   Labels: {data.y.shape}")
        print(f"   Unique labels: {len(label_map)}")

        print("\n✓ Graph conversion successful!")

    finally:
        converter.close()
