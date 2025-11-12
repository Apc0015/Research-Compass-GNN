"""
Graph Transformer Model
For generating high-quality node embeddings
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import TransformerConv
from typing import Optional


class GraphTransformerModel(nn.Module):
    """
    Graph Transformer for Node Embedding Generation

    Uses transformer-based message passing for capturing
    long-range dependencies in citation networks.

    Architecture:
    - Multi-head transformer convolutions
    - Residual connections (optional)
    - Layer normalization (optional)
    - Final projection to original dimension

    Args:
        input_dim (int): Input feature dimension (default: 384)
        hidden_dim (int): Hidden dimension per head (default: 128)
        num_layers (int): Number of transformer layers (default: 2)
        num_heads (int): Number of attention heads (default: 4)
        dropout (float): Dropout probability (default: 0.1)
        use_residual (bool): Use residual connections (default: False)
        use_layer_norm (bool): Use layer normalization (default: False)

    Example:
        >>> model = GraphTransformerModel(input_dim=384, hidden_dim=128)
        >>> embeddings = model(x, edge_index)
    """

    def __init__(
        self,
        input_dim: int = 384,
        hidden_dim: int = 128,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
        use_residual: bool = False,
        use_layer_norm: bool = False
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.use_residual = use_residual
        self.use_layer_norm = use_layer_norm

        # Transformer convolutional layers
        self.convs = nn.ModuleList()
        for i in range(num_layers):
            in_dim = input_dim if i == 0 else hidden_dim * num_heads
            self.convs.append(
                TransformerConv(
                    in_dim,
                    hidden_dim,
                    heads=num_heads,
                    dropout=dropout,
                    concat=True
                )
            )

        # Layer normalization (optional)
        if use_layer_norm:
            self.layer_norms = nn.ModuleList([
                nn.LayerNorm(hidden_dim * num_heads)
                for _ in range(num_layers)
            ])
        else:
            self.layer_norms = None

        # Output projection to original dimension
        self.output_proj = nn.Linear(hidden_dim * num_heads, input_dim)

        # Residual projection if dimensions don't match
        if use_residual and input_dim != hidden_dim * num_heads:
            self.residual_proj = nn.Linear(input_dim, hidden_dim * num_heads)
        else:
            self.residual_proj = None

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass

        Args:
            x: Node features [num_nodes, input_dim]
            edge_index: Edge indices [2, num_edges]

        Returns:
            Node embeddings [num_nodes, input_dim]
        """
        x_orig = x

        for i, conv in enumerate(self.convs):
            x_prev = x

            # Transformer convolution
            x = conv(x, edge_index)

            # Layer normalization
            if self.layer_norms is not None:
                x = self.layer_norms[i](x)

            # Activation
            x = F.relu(x)

            # Dropout
            x = F.dropout(x, p=self.dropout, training=self.training)

            # Residual connection
            if self.use_residual:
                if i == 0 and self.residual_proj is not None:
                    x_prev = self.residual_proj(x_prev)
                x = x + x_prev

        # Project back to original dimension
        x = self.output_proj(x)

        return x

    def get_layer_embeddings(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        layer: int = -1
    ) -> torch.Tensor:
        """
        Get embeddings from a specific layer

        Args:
            x: Node features
            edge_index: Edge indices
            layer: Layer index (-1 for last layer)

        Returns:
            Embeddings from specified layer
        """
        if layer < 0:
            layer = len(self.convs) + layer

        for i, conv in enumerate(self.convs[:layer + 1]):
            x = conv(x, edge_index)
            if self.layer_norms is not None:
                x = self.layer_norms[i](x)
            x = F.relu(x)

        return x

    def count_parameters(self) -> int:
        """Count trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"input_dim={self.input_dim}, "
            f"hidden_dim={self.hidden_dim}, "
            f"num_layers={self.num_layers}, "
            f"num_heads={self.num_heads}, "
            f"dropout={self.dropout}, "
            f"use_residual={self.use_residual}, "
            f"use_layer_norm={self.use_layer_norm}, "
            f"params={self.count_parameters():,})"
        )
