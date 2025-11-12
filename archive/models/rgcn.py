"""
Relational Graph Convolutional Network (R-GCN)
For graphs with multiple edge types (relational graphs)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import RGCNConv
from typing import Dict, List, Optional


class RGCNModel(nn.Module):
    """
    Relational GCN for citation networks with typed edges

    Handles different citation types:
    - extends: Building upon previous work
    - methodology: Using methods from cited paper
    - background: General background references
    - comparison: Comparing approaches

    Architecture:
    - Multiple R-GCN layers with relation-specific transformations
    - Batch normalization for stability
    - Dropout for regularization
    - Final classification or embedding layer

    Args:
        input_dim: Input feature dimension (default: 384)
        hidden_dim: Hidden layer dimension (default: 128)
        output_dim: Output dimension (default: 5 for classification)
        num_relations: Number of edge types (default: 4)
        num_layers: Number of R-GCN layers (default: 2)
        dropout: Dropout probability (default: 0.5)
        num_bases: Number of bases for basis-decomposition (default: None, use all)
        task: 'classification' or 'embedding' (default: 'classification')

    Example:
        >>> from data import classify_citation_types
        >>> edge_types, typed_edges = classify_citation_types(data)
        >>> model = RGCNModel(
        ...     input_dim=384,
        ...     num_relations=4,
        ...     output_dim=5,
        ...     task='classification'
        ... )
        >>> out = model(data.x, data.edge_index, edge_types)
    """

    def __init__(
        self,
        input_dim: int = 384,
        hidden_dim: int = 128,
        output_dim: int = 5,
        num_relations: int = 4,
        num_layers: int = 2,
        dropout: float = 0.5,
        num_bases: Optional[int] = None,
        task: str = 'classification'
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_relations = num_relations
        self.num_layers = num_layers
        self.dropout = dropout
        self.num_bases = num_bases
        self.task = task

        # R-GCN layers
        self.convs = nn.ModuleList()

        # First layer
        self.convs.append(
            RGCNConv(
                input_dim,
                hidden_dim,
                num_relations=num_relations,
                num_bases=num_bases
            )
        )

        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(
                RGCNConv(
                    hidden_dim,
                    hidden_dim,
                    num_relations=num_relations,
                    num_bases=num_bases
                )
            )

        # Output layer
        if task == 'classification':
            self.convs.append(
                RGCNConv(
                    hidden_dim,
                    output_dim,
                    num_relations=num_relations,
                    num_bases=num_bases
                )
            )
        elif task == 'embedding':
            self.convs.append(
                RGCNConv(
                    hidden_dim,
                    output_dim,
                    num_relations=num_relations,
                    num_bases=num_bases
                )
            )
        else:
            raise ValueError(f"Unknown task: {task}. Choose 'classification' or 'embedding'")

        # Batch normalization
        self.batch_norms = nn.ModuleList([
            nn.BatchNorm1d(hidden_dim)
            for _ in range(num_layers - 1)
        ])

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass

        Args:
            x: Node features [num_nodes, input_dim]
            edge_index: Edge indices [2, num_edges]
            edge_type: Edge types [num_edges] (0 to num_relations-1)

        Returns:
            Node predictions or embeddings [num_nodes, output_dim]
        """
        # Input layer + hidden layers
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index, edge_type)
            x = self.batch_norms[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        # Output layer (no activation for classification logits)
        x = self.convs[-1](x, edge_index, edge_type)

        return x

    def encode(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor
    ) -> torch.Tensor:
        """
        Get node embeddings (same as forward for embedding task)

        Args:
            x: Node features [num_nodes, input_dim]
            edge_index: Edge indices [2, num_edges]
            edge_type: Edge types [num_edges]

        Returns:
            Node embeddings [num_nodes, output_dim]
        """
        return self.forward(x, edge_index, edge_type)

    def count_parameters(self) -> int:
        """Count trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_relation_weights(self, layer_idx: int = 0) -> Optional[torch.Tensor]:
        """
        Get learned weights for each relation type

        Args:
            layer_idx: Which R-GCN layer to inspect (default: 0)

        Returns:
            Relation weights [num_relations, ...] or None if not accessible
        """
        if layer_idx >= len(self.convs):
            raise ValueError(f"Layer {layer_idx} does not exist (total layers: {len(self.convs)})")

        conv = self.convs[layer_idx]

        # Access relation-specific weights
        if hasattr(conv, 'weight'):
            return conv.weight
        elif hasattr(conv, 'comp'):
            # Basis-decomposition case
            return conv.comp
        else:
            return None


def create_rgcn_model(data, num_relations=4, **kwargs):
    """
    Convenience function to create R-GCN model

    Args:
        data: PyTorch Geometric Data object
        num_relations: Number of edge types
        **kwargs: Additional model arguments (hidden_dim, num_layers, dropout, etc.)

    Returns:
        RGCNModel instance

    Example:
        >>> model = create_rgcn_model(
        ...     data,
        ...     num_relations=4,
        ...     hidden_dim=128,
        ...     num_bases=30,
        ...     task='classification'
        ... )
    """
    input_dim = data.x.shape[1]
    output_dim = len(torch.unique(data.y)) if hasattr(data, 'y') else 128

    model = RGCNModel(
        input_dim=input_dim,
        output_dim=output_dim,
        num_relations=num_relations,
        **kwargs
    )

    return model
