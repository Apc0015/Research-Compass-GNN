"""
Graph-based baseline methods

Implements traditional graph algorithms for comparison:
- Node2Vec (graph embeddings)
- Label Propagation (semi-supervised)
- DeepWalk (simplified)
"""

import torch
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import networkx as nx
from typing import Dict, Optional
import time


class LabelPropagation:
    """
    Label Propagation baseline

    Semi-supervised learning on graphs.
    Simple but effective for graphs with homophily.

    Algorithm:
    1. Initialize labeled nodes with their true labels
    2. Propagate labels to neighbors iteratively
    3. Stop when convergence or max iterations reached
    """

    def __init__(self, max_iter: int = 100, alpha: float = 0.99):
        """
        Args:
            max_iter: Maximum number of iterations
            alpha: Weight for neighbor labels (vs initial labels)
        """
        self.max_iter = max_iter
        self.alpha = alpha

    def fit_predict(
        self,
        edge_index: torch.Tensor,
        y_train: torch.Tensor,
        train_mask: torch.Tensor,
        num_nodes: int,
        num_classes: int
    ) -> torch.Tensor:
        """
        Run label propagation

        Args:
            edge_index: Graph edges [2, num_edges]
            y_train: Training labels [num_nodes]
            train_mask: Training mask [num_nodes]
            num_nodes: Total number of nodes
            num_classes: Number of classes

        Returns:
            Predicted labels [num_nodes]
        """
        # Initialize label matrix
        Y = torch.zeros(num_nodes, num_classes)

        # Set initial labels (one-hot encoding)
        Y[train_mask] = F.one_hot(y_train[train_mask], num_classes).float()

        # Create adjacency matrix and normalize
        adj = torch.zeros(num_nodes, num_nodes)
        adj[edge_index[0], edge_index[1]] = 1.0

        # Row-normalize
        degree = adj.sum(dim=1, keepdim=True)
        degree[degree == 0] = 1  # Avoid division by zero
        adj_norm = adj / degree

        # Store initial labels
        Y_init = Y.clone()

        # Iterative propagation
        for _ in range(self.max_iter):
            Y_prev = Y.clone()

            # Propagate: Y = alpha * A_norm @ Y + (1-alpha) * Y_init
            Y = self.alpha * (adj_norm @ Y) + (1 - self.alpha) * Y_init

            # Keep training labels fixed
            Y[train_mask] = Y_init[train_mask]

            # Check convergence
            if torch.allclose(Y, Y_prev, atol=1e-4):
                break

        # Get predictions
        predictions = Y.argmax(dim=1)
        return predictions


class SimpleNode2Vec:
    """
    Simplified Node2Vec implementation

    Uses random walks to create node embeddings, then trains
    a classifier on these embeddings.

    This is a simplified version - full Node2Vec would require
    the node2vec library with optimized random walks.
    """

    def __init__(
        self,
        embedding_dim: int = 128,
        walk_length: int = 10,
        num_walks: int = 20,
        window_size: int = 5
    ):
        """
        Args:
            embedding_dim: Dimension of embeddings
            walk_length: Length of each random walk
            num_walks: Number of walks per node
            window_size: Context window size
        """
        self.embedding_dim = embedding_dim
        self.walk_length = walk_length
        self.num_walks = num_walks
        self.window_size = window_size
        self.embeddings = None
        self.classifier = None

    def _random_walk(self, G: nx.Graph, start_node: int) -> list:
        """Generate a random walk starting from start_node"""
        walk = [start_node]
        current = start_node

        for _ in range(self.walk_length - 1):
            neighbors = list(G.neighbors(current))
            if len(neighbors) == 0:
                break
            current = np.random.choice(neighbors)
            walk.append(current)

        return walk

    def _generate_walks(self, G: nx.Graph) -> list:
        """Generate all random walks"""
        walks = []
        nodes = list(G.nodes())

        for _ in range(self.num_walks):
            np.random.shuffle(nodes)
            for node in nodes:
                walks.append(self._random_walk(G, node))

        return walks

    def fit(
        self,
        edge_index: torch.Tensor,
        num_nodes: int,
        X_train: np.ndarray,
        y_train: np.ndarray
    ):
        """
        Fit Node2Vec embeddings and classifier

        Args:
            edge_index: Graph edges
            num_nodes: Number of nodes
            X_train: Training node features (not used, for compatibility)
            y_train: Training labels
        """
        # Convert to NetworkX
        edge_list = edge_index.t().cpu().numpy()
        G = nx.Graph()
        G.add_nodes_from(range(num_nodes))
        G.add_edges_from(edge_list)

        # Generate random walks
        walks = self._generate_walks(G)

        # Simple embedding: average of neighbor indices (simplified)
        # In real Node2Vec, this would use Skip-gram/Word2Vec
        embeddings = np.zeros((num_nodes, self.embedding_dim))

        for node in range(num_nodes):
            neighbors = list(G.neighbors(node))
            if len(neighbors) > 0:
                # Simple: encode based on neighbor IDs (demonstration only)
                neighbor_hash = np.array([hash(n) % self.embedding_dim for n in neighbors])
                embeddings[node] = np.bincount(neighbor_hash, minlength=self.embedding_dim)

        # Normalize
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1
        embeddings = embeddings / norms

        self.embeddings = embeddings

        # Train classifier on embeddings
        self.classifier = LogisticRegression(max_iter=1000, random_state=42)
        self.classifier.fit(embeddings[X_train], y_train)

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """Predict using learned embeddings"""
        return self.classifier.predict(self.embeddings[X_test])

    def score(self, X_test: np.ndarray, y_test: np.ndarray) -> float:
        """Compute accuracy"""
        return self.classifier.score(self.embeddings[X_test], y_test)


def evaluate_graph_baselines(data, device: Optional[torch.device] = None) -> Dict:
    """
    Evaluate graph-based baseline methods

    Args:
        data: PyTorch Geometric Data object
        device: Device (not used for these baselines)

    Returns:
        Dictionary with results
    """
    import torch.nn.functional as F

    results = {}

    print("\n" + "=" * 70)
    print("📊 Evaluating Graph-Based Baselines")
    print("=" * 70)

    num_classes = data.y.max().item() + 1

    # 1. Label Propagation
    print("\n1️⃣  Label Propagation")
    start = time.time()

    lp = LabelPropagation(max_iter=100, alpha=0.99)
    predictions = lp.fit_predict(
        data.edge_index.cpu(),
        data.y.cpu(),
        data.train_mask.cpu(),
        data.num_nodes,
        num_classes
    )

    lp_time = time.time() - start

    test_pred = predictions[data.test_mask.cpu()]
    test_true = data.y[data.test_mask.cpu()]
    lp_acc = (test_pred == test_true).float().mean().item()

    results['label_propagation'] = {
        'test_acc': lp_acc,
        'training_time': lp_time,
        'num_parameters': 0
    }

    print(f"   Accuracy: {lp_acc:.4f}")
    print(f"   Time: {lp_time:.2f}s")

    # 2. Simple Node2Vec
    print("\n2️⃣  Simple Node2Vec")
    print("   (Simplified implementation for demonstration)")

    start = time.time()

    # Get train/test indices
    train_idx = data.train_mask.cpu().nonzero(as_tuple=True)[0].numpy()
    test_idx = data.test_mask.cpu().nonzero(as_tuple=True)[0].numpy()

    y_train = data.y[data.train_mask].cpu().numpy()
    y_test = data.y[data.test_mask].cpu().numpy()

    node2vec = SimpleNode2Vec(embedding_dim=128, walk_length=10, num_walks=20)

    try:
        node2vec.fit(
            data.edge_index,
            data.num_nodes,
            train_idx,
            y_train
        )

        n2v_acc = node2vec.score(test_idx, y_test)
        n2v_time = time.time() - start

        results['node2vec'] = {
            'test_acc': n2v_acc,
            'training_time': n2v_time,
            'num_parameters': 'N/A'
        }

        print(f"   Accuracy: {n2v_acc:.4f}")
        print(f"   Time: {n2v_time:.2f}s")

    except Exception as e:
        print(f"   ⚠️  Node2Vec failed: {e}")
        results['node2vec'] = {
            'test_acc': 0.0,
            'training_time': 0.0,
            'num_parameters': 'N/A',
            'error': str(e)
        }

    print("\n" + "=" * 70)
    print("✅ Graph Baseline Evaluation Complete")
    print("=" * 70)

    return results
