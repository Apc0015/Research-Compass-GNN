"""
GraphSAGE Model
For efficient inductive learning on large graphs
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from typing import Optional


class GraphSAGEModel(nn.Module):
    """
    GraphSAGE for Inductive Node Embedding and Classification

    Key advantages over transductive models:
    - Can handle unseen nodes without retraining
    - More efficient for large graphs
    - Faster inference (mini-batch friendly)

    Architecture:
    - SAGEConv layers with MEAN aggregation
    - Dropout for regularization
    - Optional final classification layer

    Args:
        input_dim (int): Input feature dimension (default: 384)
        hidden_dim (int): Hidden layer dimension (default: 256)
        output_dim (int): Output dimension (default: 128 for embeddings)
        num_layers (int): Number of SAGE layers (default: 2)
        dropout (float): Dropout probability (default: 0.5)
        aggregator (str): Aggregation method ('mean', 'max', 'lstm') (default: 'mean')
        normalize (bool): Normalize embeddings (default: True)
        task (str): Task type ('embedding' or 'classification') (default: 'embedding')
        num_classes (int): Number of classes for classification (default: 5)

    Example:
        >>> # For embeddings
        >>> model = GraphSAGEModel(task='embedding', output_dim=128)
        >>> embeddings = model(x, edge_index)
        >>>
        >>> # For classification
        >>> model = GraphSAGEModel(task='classification', num_classes=5)
        >>> predictions = model(x, edge_index)
    """

    def __init__(
        self,
        input_dim: int = 384,
        hidden_dim: int = 256,
        output_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.5,
        aggregator: str = 'mean',
        normalize: bool = True,
        task: str = 'embedding',
        num_classes: int = 5
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.aggregator = aggregator
        self.normalize = normalize
        self.task = task
        self.num_classes = num_classes

        # Build SAGE layers
        self.convs = nn.ModuleList()

        # First layer
        self.convs.append(
            SAGEConv(
                input_dim,
                hidden_dim,
                normalize=normalize,
                aggr=aggregator
            )
        )

        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(
                SAGEConv(
                    hidden_dim,
                    hidden_dim,
                    normalize=normalize,
                    aggr=aggregator
                )
            )

        # Output layer
        if task == 'classification':
            # Final layer outputs class logits
            self.convs.append(
                SAGEConv(
                    hidden_dim,
                    num_classes,
                    normalize=normalize,
                    aggr=aggregator
                )
            )
        else:
            # Final layer outputs embeddings
            self.convs.append(
                SAGEConv(
                    hidden_dim,
                    output_dim,
                    normalize=normalize,
                    aggr=aggregator
                )
            )

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
            Node embeddings or predictions [num_nodes, output_dim or num_classes]
        """
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        # Final layer
        x = self.convs[-1](x, edge_index)

        # No activation on final layer for classification (logits)
        # or embeddings
        return x

    def encode(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor
    ) -> torch.Tensor:
        """
        Get node embeddings (alias for forward when task='embedding')

        Args:
            x: Node features
            edge_index: Edge indices

        Returns:
            Node embeddings
        """
        return self.forward(x, edge_index)

    def inductive_inference(
        self,
        x_new: torch.Tensor,
        edge_index_new: torch.Tensor
    ) -> torch.Tensor:
        """
        Perform inference on new, unseen nodes

        This is the key advantage of GraphSAGE - it can handle
        nodes not seen during training.

        Args:
            x_new: Features of new nodes [num_new_nodes, input_dim]
            edge_index_new: Edges for new nodes [2, num_new_edges]

        Returns:
            Predictions for new nodes
        """
        self.eval()
        with torch.no_grad():
            return self.forward(x_new, edge_index_new)

    def get_layer_embeddings(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        layer: int = -2
    ) -> torch.Tensor:
        """
        Get embeddings from intermediate layer

        Args:
            x: Node features
            edge_index: Edge indices
            layer: Layer index (-2 = second to last)

        Returns:
            Intermediate embeddings
        """
        if layer < 0:
            layer = len(self.convs) + layer

        for i, conv in enumerate(self.convs[:layer + 1]):
            x = conv(x, edge_index)
            if i < layer:
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
            f"aggregator='{self.aggregator}', "
            f"task='{self.task}', "
            f"params={self.count_parameters():,})"
        )
