"""
Dataset utilities for creating and loading citation networks

Supports:
- Synthetic citation networks
- Real datasets (Cora, CiteSeer, PubMed)
- Dataset statistics and visualization
"""

import torch
import numpy as np
from torch_geometric.data import Data
from torch_geometric.datasets import Planetoid
from typing import Tuple, Dict, Optional
import os


def create_synthetic_citation_network(
    num_papers: int = 200,
    num_topics: int = 5,
    feature_dim: int = 384,
    avg_citations: int = 8,
    temporal: bool = True,
    topic_clustering: float = 0.8,
    seed: Optional[int] = None
) -> Data:
    """
    Create a realistic synthetic citation network

    Features:
    - Power-law citation distribution (some papers highly cited)
    - Topic clustering (papers in same topic cite each other more often)
    - Temporal ordering (papers only cite older papers)
    - Realistic feature embeddings

    Args:
        num_papers: Number of papers/nodes
        num_topics: Number of research topics/classes
        feature_dim: Dimensionality of node features (384 for Sentence-BERT)
        avg_citations: Average number of citations per paper
        temporal: Whether to enforce temporal constraints
        topic_clustering: Probability of citing same-topic papers (0-1)
        seed: Random seed for reproducibility

    Returns:
        PyTorch Geometric Data object

    Example:
        >>> data = create_synthetic_citation_network(num_papers=500, num_topics=5)
        >>> print(f"Graph: {data.num_nodes} nodes, {data.num_edges} edges")
    """
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    print(f"🔬 Creating synthetic citation network...")
    print(f"   Papers: {num_papers}, Topics: {num_topics}, Avg citations: {avg_citations}")

    # Node features (random embeddings simulating Sentence-BERT)
    x = torch.randn(num_papers, feature_dim)

    # Normalize features
    x = torch.nn.functional.normalize(x, p=2, dim=1)

    # Topic labels (ground truth for node classification)
    y = torch.randint(0, num_topics, (num_papers,))

    # Generate edges with realistic patterns
    edges = []

    if temporal:
        # Papers can only cite older papers (directed acyclic graph)
        for target in range(1, num_papers):
            # Number of citations (exponential distribution for power-law effect)
            num_citations = max(1, int(np.random.exponential(avg_citations)))
            num_citations = min(num_citations, target)  # Can't cite more than exist

            # Topic of target paper
            target_topic = y[target].item()

            for _ in range(num_citations):
                if np.random.rand() < topic_clustering:
                    # Prefer papers from same topic
                    same_topic_papers = [i for i in range(target) if y[i].item() == target_topic]
                    if same_topic_papers:
                        source = np.random.choice(same_topic_papers)
                    else:
                        source = np.random.randint(0, target)
                else:
                    # Different topic (interdisciplinary citation)
                    source = np.random.randint(0, target)

                edges.append([source, target])
    else:
        # Random edges (for comparison/ablation studies)
        num_edges = num_papers * avg_citations
        for _ in range(num_edges):
            source = np.random.randint(0, num_papers)
            target = np.random.randint(0, num_papers)
            if source != target:
                edges.append([source, target])

    edge_index = torch.tensor(edges, dtype=torch.long).t()

    # Create train/val/test masks (60/20/20 split)
    num_train = int(0.6 * num_papers)
    num_val = int(0.2 * num_papers)

    perm = torch.randperm(num_papers)
    train_mask = torch.zeros(num_papers, dtype=torch.bool)
    val_mask = torch.zeros(num_papers, dtype=torch.bool)
    test_mask = torch.zeros(num_papers, dtype=torch.bool)

    train_mask[perm[:num_train]] = True
    val_mask[perm[num_train:num_train + num_val]] = True
    test_mask[perm[num_train + num_val:]] = True

    data = Data(
        x=x,
        edge_index=edge_index,
        y=y,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask
    )

    print(f"✅ Created graph: {num_papers} nodes, {edge_index.shape[1]} edges")
    print(f"   Train: {train_mask.sum().item()}, Val: {val_mask.sum().item()}, Test: {test_mask.sum().item()}")

    return data


def load_citation_dataset(
    name: str = 'Cora',
    root: str = 'data/raw',
    transform: Optional[callable] = None
) -> Tuple[Data, Dict]:
    """
    Load real citation network datasets

    Available datasets:
    - Cora: 2,708 papers, 7 classes, 5,429 citations
    - CiteSeer: 3,327 papers, 6 classes, 4,732 citations
    - PubMed: 19,717 papers, 3 classes, 44,338 citations

    Args:
        name: Dataset name ('Cora', 'CiteSeer', or 'PubMed')
        root: Root directory to store dataset
        transform: Optional transform to apply

    Returns:
        Tuple of (data, info_dict)

    Example:
        >>> data, info = load_citation_dataset('Cora')
        >>> print(f"Loaded {info['name']}: {info['num_nodes']} nodes, {info['num_classes']} classes")
    """
    print(f"📚 Loading {name} dataset...")

    # Load dataset using PyTorch Geometric
    dataset = Planetoid(root=root, name=name, transform=transform)
    data = dataset[0]

    # Dataset info
    info = {
        'name': name,
        'num_nodes': data.num_nodes,
        'num_edges': data.num_edges,
        'num_features': data.num_features,
        'num_classes': dataset.num_classes,
        'num_train': data.train_mask.sum().item(),
        'num_val': data.val_mask.sum().item(),
        'num_test': data.test_mask.sum().item(),
        'is_directed': True,  # Citation networks are directed
        'has_node_features': True,
        'has_edge_features': False
    }

    print(f"✅ Loaded {name}")
    print(f"   Nodes: {info['num_nodes']}, Edges: {info['num_edges']}, Classes: {info['num_classes']}")
    print(f"   Features: {info['num_features']}")
    print(f"   Train: {info['num_train']}, Val: {info['num_val']}, Test: {info['num_test']}")

    return data, info


def get_dataset_statistics(data: Data) -> Dict:
    """
    Compute comprehensive dataset statistics

    Args:
        data: PyTorch Geometric Data object

    Returns:
        Dictionary with statistics

    Example:
        >>> stats = get_dataset_statistics(data)
        >>> print(f"Average degree: {stats['avg_degree']:.2f}")
    """
    stats = {}

    # Basic stats
    stats['num_nodes'] = data.num_nodes
    stats['num_edges'] = data.num_edges
    stats['num_features'] = data.num_features if hasattr(data, 'num_features') else data.x.shape[1]

    # Degree statistics
    edge_index = data.edge_index
    degrees = torch.bincount(edge_index[0])  # Out-degrees for directed graph

    stats['avg_degree'] = degrees.float().mean().item()
    stats['max_degree'] = degrees.max().item()
    stats['min_degree'] = degrees.min().item()
    stats['std_degree'] = degrees.float().std().item()

    # Class distribution
    if hasattr(data, 'y'):
        num_classes = data.y.max().item() + 1
        class_counts = torch.bincount(data.y, minlength=num_classes)
        stats['num_classes'] = num_classes
        stats['class_distribution'] = class_counts.tolist()

        # Class imbalance ratio
        max_count = class_counts.max().item()
        min_count = class_counts.min().item()
        stats['class_imbalance_ratio'] = max_count / min_count if min_count > 0 else float('inf')

    # Graph density
    max_edges = data.num_nodes * (data.num_nodes - 1)  # For directed graph
    stats['density'] = data.num_edges / max_edges

    # Split sizes
    if hasattr(data, 'train_mask'):
        stats['train_size'] = data.train_mask.sum().item()
        stats['val_size'] = data.val_mask.sum().item()
        stats['test_size'] = data.test_mask.sum().item()

        stats['train_ratio'] = stats['train_size'] / data.num_nodes
        stats['val_ratio'] = stats['val_size'] / data.num_nodes
        stats['test_ratio'] = stats['test_size'] / data.num_nodes

    return stats


def print_dataset_info(data: Data, name: str = "Dataset"):
    """
    Print formatted dataset information

    Args:
        data: PyTorch Geometric Data object
        name: Dataset name

    Example:
        >>> print_dataset_info(data, "Cora")
    """
    stats = get_dataset_statistics(data)

    print("\n" + "=" * 60)
    print(f"📊 {name} Statistics")
    print("=" * 60)

    print(f"\n🔢 Graph Structure:")
    print(f"   Nodes: {stats['num_nodes']:,}")
    print(f"   Edges: {stats['num_edges']:,}")
    print(f"   Features: {stats['num_features']:,}")
    print(f"   Density: {stats['density']:.6f}")

    print(f"\n📈 Degree Statistics:")
    print(f"   Average: {stats['avg_degree']:.2f}")
    print(f"   Min: {stats['min_degree']}")
    print(f"   Max: {stats['max_degree']}")
    print(f"   Std: {stats['std_degree']:.2f}")

    if 'num_classes' in stats:
        print(f"\n🏷️  Classes:")
        print(f"   Number of classes: {stats['num_classes']}")
        print(f"   Class distribution: {stats['class_distribution']}")
        print(f"   Imbalance ratio: {stats['class_imbalance_ratio']:.2f}")

    if 'train_size' in stats:
        print(f"\n✂️  Data Split:")
        print(f"   Train: {stats['train_size']:,} ({stats['train_ratio']:.1%})")
        print(f"   Val: {stats['val_size']:,} ({stats['val_ratio']:.1%})")
        print(f"   Test: {stats['test_size']:,} ({stats['test_ratio']:.1%})")

    print("=" * 60 + "\n")


def create_link_prediction_split(
    data: Data,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: Optional[int] = None
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Create train/val/test splits for link prediction

    Args:
        data: Graph data
        val_ratio: Validation edge ratio
        test_ratio: Test edge ratio
        seed: Random seed

    Returns:
        Tuple of (train_edge_index, val_edge_index, test_edge_index, neg_edge_index)

    Example:
        >>> train_edges, val_edges, test_edges, neg_edges = create_link_prediction_split(data)
    """
    if seed is not None:
        torch.manual_seed(seed)

    edge_index = data.edge_index
    num_edges = edge_index.shape[1]

    # Random permutation
    perm = torch.randperm(num_edges)

    # Split indices
    num_val = int(val_ratio * num_edges)
    num_test = int(test_ratio * num_edges)

    val_idx = perm[:num_val]
    test_idx = perm[num_val:num_val + num_test]
    train_idx = perm[num_val + num_test:]

    train_edge_index = edge_index[:, train_idx]
    val_edge_index = edge_index[:, val_idx]
    test_edge_index = edge_index[:, test_idx]

    # Generate negative edges (non-existent edges)
    num_nodes = data.num_nodes
    num_neg = num_val + num_test

    neg_edges = []
    edge_set = set(map(tuple, edge_index.t().tolist()))

    while len(neg_edges) < num_neg:
        src = torch.randint(0, num_nodes, (1,)).item()
        dst = torch.randint(0, num_nodes, (1,)).item()

        if src != dst and (src, dst) not in edge_set:
            neg_edges.append([src, dst])

    neg_edge_index = torch.tensor(neg_edges, dtype=torch.long).t()

    return train_edge_index, val_edge_index, test_edge_index, neg_edge_index
