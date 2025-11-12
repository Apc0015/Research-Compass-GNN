"""
Heterogeneous Attention Network (HAN)
For multi-relational graph learning with hierarchical attention
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, GATConv, Linear
from typing import Dict, List, Optional


class HANModel(nn.Module):
    """
    Heterogeneous Attention Network (HAN)

    Implements hierarchical attention:
    1. Node-level attention: Within each metapath (using GAT)
    2. Semantic-level attention: Across different metapaths

    Architecture:
    - Layer 1: Heterogeneous convolutions for each edge type
    - Layer 2: Semantic attention to combine metapath embeddings
    - Layer 3: Final classification/prediction

    Args:
        metadata: Tuple of (node_types, edge_types) from HeteroData
        hidden_dim: Hidden layer dimension (default: 128)
        out_dim: Output dimension (default: 64 for embeddings)
        num_heads: Number of attention heads (default: 8)
        dropout: Dropout probability (default: 0.3)
        task: 'classification' or 'embedding' (default: 'classification')
        num_classes: Number of classes if task='classification' (default: 7)

    Example:
        >>> from data import convert_to_heterogeneous
        >>> hetero_data = convert_to_heterogeneous(data)
        >>> model = HANModel(
        ...     metadata=hetero_data.metadata(),
        ...     hidden_dim=128,
        ...     num_heads=8,
        ...     task='classification',
        ...     num_classes=7
        ... )
        >>> out = model(hetero_data.x_dict, hetero_data.edge_index_dict)
    """

    def __init__(
        self,
        metadata: tuple,  # (node_types, edge_types)
        hidden_dim: int = 128,
        out_dim: int = 64,
        num_heads: int = 8,
        dropout: float = 0.3,
        task: str = 'classification',
        num_classes: int = 7
    ):
        super().__init__()

        self.metadata = metadata
        self.node_types, self.edge_types = metadata
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.task = task
        self.num_classes = num_classes

        # Input projection for each node type (lazy initialization)
        self.input_projs = nn.ModuleDict()
        for node_type in self.node_types:
            self.input_projs[node_type] = None

        # Heterogeneous convolutions (node-level attention)
        self.conv1 = HeteroConv({
            edge_type: GATConv(
                (-1, -1),  # Lazy initialization
                hidden_dim,
                heads=num_heads,
                dropout=dropout,
                add_self_loops=False
            )
            for edge_type in self.edge_types
        }, aggr='sum')

        self.conv2 = HeteroConv({
            edge_type: GATConv(
                hidden_dim * num_heads,
                out_dim,
                heads=1,
                dropout=dropout,
                add_self_loops=False
            )
            for edge_type in self.edge_types
        }, aggr='sum')

        # Semantic-level attention
        # Learns importance of different metapaths for each node type
        self.semantic_attention = nn.ModuleDict()
        for node_type in self.node_types:
            num_metapaths = self._count_metapaths_for_node(node_type)
            if num_metapaths > 1:
                self.semantic_attention[node_type] = SemanticAttention(
                    in_dim=out_dim,
                    hidden_dim=128
                )

        # Task-specific heads
        if task == 'classification':
            self.classifier = nn.Linear(out_dim, num_classes)
        elif task == 'embedding':
            # No additional layer, return embeddings directly
            pass
        else:
            raise ValueError(f"Unknown task: {task}. Choose 'classification' or 'embedding'")

    def _count_metapaths_for_node(self, node_type: str) -> int:
        """Count number of metapaths starting from or ending at node_type"""
        count = 0
        for src, _, dst in self.edge_types:
            if src == node_type or dst == node_type:
                count += 1
        return max(count, 1)

    def _initialize_input_projections(self, x_dict: Dict[str, torch.Tensor]):
        """Initialize input projections based on actual input dimensions"""
        for node_type, x in x_dict.items():
            if self.input_projs[node_type] is None:
                in_dim = x.shape[1]
                self.input_projs[node_type] = Linear(in_dim, self.hidden_dim)
                # Move to same device as input
                self.input_projs[node_type] = self.input_projs[node_type].to(x.device)

    def forward(
        self,
        x_dict: Dict[str, torch.Tensor],
        edge_index_dict: Dict[tuple, torch.Tensor],
        return_attention: bool = False
    ):
        """
        Forward pass

        Args:
            x_dict: Dictionary of node features {node_type: features}
            edge_index_dict: Dictionary of edge indices {edge_type: edge_index}
            return_attention: Whether to return attention weights

        Returns:
            If task='classification': Dictionary {node_type: class_logits}
            If task='embedding': Dictionary {node_type: embeddings}
            If return_attention=True: Also returns attention weights
        """
        # Initialize input projections if needed
        self._initialize_input_projections(x_dict)

        # Project input features to hidden dimension
        x_dict_proj = {}
        for node_type, x in x_dict.items():
            x_dict_proj[node_type] = F.relu(self.input_projs[node_type](x))
            x_dict_proj[node_type] = F.dropout(
                x_dict_proj[node_type],
                p=self.dropout,
                training=self.training
            )

        # First heterogeneous conv layer
        x_dict_conv1 = self.conv1(x_dict_proj, edge_index_dict)
        x_dict_conv1 = {
            key: F.elu(x) for key, x in x_dict_conv1.items()
        }
        x_dict_conv1 = {
            key: F.dropout(x, p=self.dropout, training=self.training)
            for key, x in x_dict_conv1.items()
        }

        # Second heterogeneous conv layer
        x_dict_conv2 = self.conv2(x_dict_conv1, edge_index_dict)

        # Semantic-level attention (if applicable)
        x_dict_final = {}
        attention_weights = {}

        for node_type in x_dict_conv2.keys():
            if node_type in self.semantic_attention:
                # Apply semantic attention
                x_final, attn = self.semantic_attention[node_type](
                    x_dict_conv2[node_type]
                )
                x_dict_final[node_type] = x_final
                attention_weights[node_type] = attn
            else:
                # No semantic attention needed
                x_dict_final[node_type] = x_dict_conv2[node_type]
                attention_weights[node_type] = None

        # Task-specific output
        if self.task == 'classification':
            out_dict = {}
            for node_type, x in x_dict_final.items():
                if node_type == 'paper':  # Only classify papers
                    out_dict[node_type] = self.classifier(x)

            if return_attention:
                return out_dict, attention_weights
            return out_dict

        elif self.task == 'embedding':
            if return_attention:
                return x_dict_final, attention_weights
            return x_dict_final

    def count_parameters(self) -> int:
        """Count total trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class SemanticAttention(nn.Module):
    """
    Semantic-level attention to weight different metapaths

    Uses a simple attention mechanism to learn importance of different
    metapaths for each node type.

    Args:
        in_dim: Input embedding dimension
        hidden_dim: Hidden dimension for attention mechanism
    """

    def __init__(self, in_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim

        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1, bias=False)
        )

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: Node embeddings [num_nodes, in_dim]

        Returns:
            Attended embeddings [num_nodes, in_dim]
            Attention weights [num_nodes, 1]
        """
        # Compute attention scores
        # Shape: [num_nodes, 1]
        attn_scores = self.attention(x)

        # Normalize to get attention weights
        # Shape: [num_nodes, 1]
        attn_weights = torch.sigmoid(attn_scores)

        # Apply attention weights
        # Shape: [num_nodes, in_dim]
        out = attn_weights * x

        return out, attn_weights


def create_han_model(hetero_data, hidden_dim=128, num_heads=8, **kwargs):
    """
    Convenience function to create HAN model from heterogeneous data

    Args:
        hetero_data: HeteroData object
        hidden_dim: Hidden dimension
        num_heads: Number of attention heads
        **kwargs: Additional model arguments (task, num_classes, dropout, etc.)

    Returns:
        HANModel instance

    Example:
        >>> hetero_data = convert_to_heterogeneous(data)
        >>> model = create_han_model(
        ...     hetero_data,
        ...     hidden_dim=128,
        ...     task='classification',
        ...     num_classes=7
        ... )
    """
    model = HANModel(
        metadata=hetero_data.metadata(),
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        **kwargs
    )
    return model
