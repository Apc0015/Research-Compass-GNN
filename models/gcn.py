"""
GCN (Graph Convolutional Network) Model
For node classification tasks on citation networks
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from typing import Optional


class GCNModel(nn.Module):
    """
    Graph Convolutional Network for Node Classification

    Architecture:
    - Multiple GCN layers with ReLU activation and dropout
    - Final layer without activation for logits
    - Suitable for citation network node classification

    Args:
        input_dim (int): Input feature dimension (default: 384 for Sentence-BERT)
        hidden_dim (int): Hidden layer dimension (default: 128)
        output_dim (int): Number of output classes (default: 5 topics)
        num_layers (int): Number of GCN layers (default: 3)
        dropout (float): Dropout probability (default: 0.5)

    Example:
        >>> model = GCNModel(input_dim=384, hidden_dim=128, output_dim=5)
        >>> out = model(x, edge_index)
        >>> pred = out.argmax(dim=1)
    """

    def __init__(
        self,
        input_dim: int = 384,
        hidden_dim: int = 128,
        output_dim: int = 5,
        num_layers: int = 3,
        dropout: float = 0.5
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.dropout = dropout

        # Build convolutional layers
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(input_dim, hidden_dim))

        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))

        self.convs.append(GCNConv(hidden_dim, output_dim))

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass

        Args:
            x: Node feature matrix [num_nodes, input_dim]
            edge_index: Edge indices [2, num_edges]

        Returns:
            Node predictions [num_nodes, output_dim]
        """
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        # Final layer without activation
        x = self.convs[-1](x, edge_index)
        return x

    def get_embeddings(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        layer: int = -2
    ) -> torch.Tensor:
        """
        Get intermediate layer embeddings for visualization/analysis

        Args:
            x: Node feature matrix
            edge_index: Edge indices
            layer: Which layer to extract embeddings from (-2 = second to last)

        Returns:
            Node embeddings from specified layer
        """
        for i, conv in enumerate(self.convs[:layer]):
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        return x

    def count_parameters(self) -> int:
        """Count trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"input_dim={self.input_dim}, "
            f"hidden_dim={self.hidden_dim}, "
            f"output_dim={self.output_dim}, "
            f"num_layers={self.num_layers}, "
            f"dropout={self.dropout}, "
            f"params={self.count_parameters():,})"
        )
